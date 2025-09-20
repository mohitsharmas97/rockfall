import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import logging
import traceback
import os
import json
import time
from twilio.rest import Client
from dotenv import load_dotenv

# --- SHAP INTEGRATION: Import shap with error handling ---
try:
    import shap
    SHAP_AVAILABLE = True
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("SHAP library imported successfully.")
except ImportError:
    SHAP_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("SHAP library not available. Explanations will be disabled.")

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)

load_dotenv()

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
twilio_client = None
if all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("Twilio client configured successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Twilio client: {e}")
else:
    logger.warning("Twilio credentials not fully configured. SMS alerts will be disabled.")
ALERT_CONFIG_FILE = 'alerts.json'

# --- 1. Model Architecture Definition ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.context_vector = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attn(lstm_output))
        attn_weights = self.context_vector(attn_weights).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_vector = self.attention(lstm_out)
        out = self.fc(context_vector)
        return out

# --- 2. Model and Scaler Loading ---
INPUT_FEATURES = 14
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 1
DROPOUT = 0.5
OPTIMAL_THRESHOLD = 0.44

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
scaler_params = None



try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of app.py
    model_path = os.path.join(BASE_DIR, "rockfall_bilstm_attention.pth")
    scaler_path = os.path.join(BASE_DIR, "scaler_params.npz")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.error(f"CRITICAL: Model or scaler file not found.")
    else:
        model = BiLSTMAttention(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        
        scaler_data = np.load(scaler_path)
        scaler_params = {'mean': scaler_data['mean'], 'scale': scaler_data['scale']}
        logger.info(f"Scaler parameters loaded successfully from {scaler_path}")
except Exception as e:
    logger.error(f"Fatal error during initialization: {e}\n{traceback.format_exc()}")

# --- SHAP INTEGRATION: Initialize the explainer ---
explainer = None
if SHAP_AVAILABLE and model is not None:
    try:
        background_data_path = 'X_lstm_ready.npy' 
        
        if not os.path.exists(background_data_path):
            logger.warning(f"'{background_data_path}' not found. SHAP explanations disabled.")
        else:
            X_train_sample = np.load(background_data_path)
            
            # Create background data for SHAP
            num_samples = X_train_sample.shape[0]
            num_timesteps = X_train_sample.shape[1]   # 10
            num_features = X_train_sample.shape[2]    # 14
            
            # Use random sampling instead of k-means for LSTM sequences
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(num_samples, size=min(50, num_samples), replace=False)
            background_summary = X_train_sample[sample_indices]
            
            logger.info(f"Created a SHAP background summary with shape: {background_summary.shape}")

            # Define prediction function for KernelExplainer
            def f(numpy_array):
                """SHAP prediction wrapper function"""
                # Convert to tensor, ensuring proper shape
                if len(numpy_array.shape) == 2:
                    # If SHAP flattened the data, reshape it back
                    batch_size = numpy_array.shape[0]
                    expected_flat_size = num_timesteps * num_features
                    if numpy_array.shape[1] == expected_flat_size:
                        reshaped_array = numpy_array.reshape(batch_size, num_timesteps, num_features)
                    else:
                        raise ValueError(f"Unexpected input shape: {numpy_array.shape}")
                elif len(numpy_array.shape) == 3:
                    # Already in the correct 3D shape
                    reshaped_array = numpy_array
                else:
                    raise ValueError(f"Unexpected input dimension: {numpy_array.shape}")
                
                # Convert to tensor for the model
                tensor = torch.tensor(reshaped_array, dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    # Get the raw model output (logits)
                    logits = model(tensor)
                
                # Return probabilities instead of raw logits for better interpretation
                probabilities = torch.sigmoid(logits)
                return probabilities.cpu().numpy()

            # Try DeepExplainer first (works better with neural networks)
            try:
                explainer = shap.DeepExplainer(model, torch.tensor(background_summary, dtype=torch.float32).to(device))
                logger.info("SHAP DeepExplainer created successfully.")
            except Exception as deep_ex:
                logger.warning(f"DeepExplainer failed ({deep_ex}), falling back to KernelExplainer...")
                
                # Flatten the background data for KernelExplainer
                background_flattened = background_summary.reshape(background_summary.shape[0], -1)
                explainer = shap.KernelExplainer(f, background_flattened)
                logger.info("SHAP KernelExplainer created successfully with flattened data.")

    except Exception as e:
        logger.error(f"Failed to initialize SHAP explainer: {e}\n{traceback.format_exc()}")
        explainer = None

# --- SHAP explanation generation function ---
def generate_shap_explanation(input_tensor, explainer):
    """Generate SHAP explanation handling both DeepExplainer and KernelExplainer"""
    if explainer is None:
        return {
            'explanation_available': False,
            'message': 'SHAP explainer not available.'
        }
    
    try:
        logger.info("Generating SHAP explanation...")
        
        # Check explainer type and generate appropriate explanations
        if hasattr(explainer, 'explainer') and 'Deep' in str(type(explainer.explainer)):
            # DeepExplainer
            shap_values = explainer.shap_values(input_tensor)
        else:
            # KernelExplainer - needs flattened input
            input_flattened = input_tensor.cpu().numpy().reshape(input_tensor.shape[0], -1)
            shap_values = explainer.shap_values(input_flattened)
        
        # Define feature names
        feature_names = [
            'Precipitation', 'Rain', 'Snowfall', 'Precipitation Hours', 
            'Evapotranspiration', 'Max Temp', 'Min Temp', 
            'Soil Moisture (0-7cm)', 'Soil Moisture (7-28cm)', 'Snow Depth', 
            'Relative Humidity', 'Elevation', 'Slope', 'Aspect'
        ]
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
        else:
            shap_vals = shap_values
        
        # Process SHAP values based on their shape
        if len(shap_vals.shape) == 3:  # (batch, time, features)
            mean_shap_values = np.mean(shap_vals[0], axis=0)  # Average over time steps
        elif len(shap_vals.shape) == 2 and shap_vals.shape[1] == len(feature_names) * 10:  # Flattened
            # Reshape and average
            reshaped = shap_vals[0].reshape(10, len(feature_names))
            mean_shap_values = np.mean(reshaped, axis=0)
        elif len(shap_vals.shape) == 2 and shap_vals.shape[1] == len(feature_names):  # Already averaged
            mean_shap_values = shap_vals[0]
        else:
            mean_shap_values = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
        
        # Create feature impact dictionary
        if len(mean_shap_values) == len(feature_names):
            feature_impacts = {name: float(val) for name, val in zip(feature_names, mean_shap_values)}
            sorted_impacts = sorted(feature_impacts.items(), key=lambda item: abs(item[1]), reverse=True)
            
            # Get top contributing factors
            top_positive = [(name, val) for name, val in sorted_impacts if val > 0][:3]
            top_negative = [(name, val) for name, val in sorted_impacts if val < 0][:3]
            
            return {
                'message': 'Feature impact on risk prediction. Positive values increase risk, negative values decrease risk.',
                'all_impacts': {name: round(val, 4) for name, val in sorted_impacts},
                'top_risk_increasing': [{'feature': name, 'impact': round(val, 4)} for name, val in top_positive],
                'top_risk_decreasing': [{'feature': name, 'impact': round(val, 4)} for name, val in top_negative],
                'explanation_available': True
            }
        else:
            logger.error(f"SHAP values length ({len(mean_shap_values)}) doesn't match feature names ({len(feature_names)})")
            return {'explanation_available': False, 'error': 'Feature count mismatch'}
            
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}")
        return {
            'explanation_available': False, 
            'error': f'Could not generate explanation: {str(e)}'
        }

# --- 3. Helper Functions ---
def safe_sum(data_list):
    if not data_list:
        return 0.0
    return sum(value for value in data_list if value is not None)

def safe_get_value(data, index, default=0.0):
    try:
        value = data[index]
        return value if value is not None else default
    except (IndexError, TypeError):
        return default

def fetch_weather_data(lat: float, lon: float, days: int = 10) -> Optional[Dict]:
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    daily_params = "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,et0_fao_evapotranspiration"
    hourly_params = "relative_humidity_2m,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,snow_depth"
    params = {'latitude': lat, 'longitude': lon, 'start_date': start_date, 'end_date': end_date, 'daily': daily_params, 'hourly': hourly_params, 'timezone': 'GMT'}

    for attempt in range(5):
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 429:
                retry_after = (attempt + 1) * 2
                logger.warning(f"Rate limit hit. Retrying in {retry_after}s...")
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed on attempt {attempt + 1}: {e}")
            if attempt >= 4: 
                return None
            time.sleep(2)
    return None



def fetch_elevation_data(lat: float, lon: float) -> Tuple[float, float, float]:
    """
    Fetch elevation and calculate slope and aspect from elevation grid.
    Returns: (elevation, slope_degrees, aspect_degrees)
    """
    try:
        # Create a small grid around the point to calculate slope and aspect
        offset = 0.001  # approximately 100m at mid-latitudes
        
        # Define 3x3 grid points around the center
        points = []
        elevations = []
        
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                lat_offset = lat + i * offset
                lon_offset = lon + j * offset
                points.append((lat_offset, lon_offset))
        
        # Batch request for all 9 points
        locations_str = "|".join([f"{lat_p},{lon_p}" for lat_p, lon_p in points])
        url = f"https://api.opentopodata.org/v1/mapzen?locations={locations_str}"
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Extract elevations
        for result in data['results']:
            elevation = result.get('elevation')
            if elevation is not None:
                elevations.append(elevation)
            else:
                # If any elevation is missing, use fallback
                logger.warning(f"Missing elevation data for {lat},{lon}. Using fallback values.")
                return 1000.0, 15.0, 180.0
        
        if len(elevations) != 9:
            logger.warning(f"Incomplete elevation grid for {lat},{lon}. Using fallback values.")
            return 1000.0, 15.0, 180.0
        
        # Arrange elevations in 3x3 grid
        # Grid layout:
        # z1 z2 z3
        # z4 z5 z6  
        # z7 z8 z9
        z = np.array(elevations).reshape(3, 3)
        
        # Get center elevation
        center_elevation = z[1, 1]
        
        # Calculate slope and aspect using Horn's method
        # This is the standard method used in GIS applications
        
        # Partial derivatives using Sobel operators
        dz_dx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * offset * 111320)
        dz_dy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * offset * 111320)
        
        # Convert offset to meters (approximately)
        # 1 degree latitude ≈ 111,320 meters
        
        # Calculate slope in radians then convert to degrees
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_degrees = np.degrees(slope_rad)
        
        # Calculate aspect in radians then convert to degrees
        if dz_dx == 0 and dz_dy == 0:
            # Flat area - aspect is undefined, use 0
            aspect_degrees = 0.0
        else:
            aspect_rad = np.arctan2(dz_dy, -dz_dx)
            aspect_degrees = np.degrees(aspect_rad)
            
            # Convert from mathematical angle to compass bearing
            # Mathematical: 0° = East, 90° = North
            # Compass: 0° = North, 90° = East
            aspect_degrees = 90 - aspect_degrees
            
            # Ensure aspect is between 0 and 360 degrees
            if aspect_degrees < 0:
                aspect_degrees += 360
            elif aspect_degrees >= 360:
                aspect_degrees -= 360
        
        logger.info(f"Successfully calculated topography for {lat},{lon}: "
                   f"elevation={center_elevation:.1f}m, slope={slope_degrees:.1f}°, aspect={aspect_degrees:.1f}°")
        
        return float(center_elevation), float(slope_degrees), float(aspect_degrees)
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Network error fetching elevation data for {lat},{lon}: {e}. Using fallback values.")
        return 1000.0, 15.0, 180.0
    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Error processing elevation data for {lat},{lon}: {e}. Using fallback values.")
        return 1000.0, 15.0, 180.0
    except Exception as e:
        logger.error(f"Unexpected error fetching topography data for {lat},{lon}: {e}. Using fallback values.")
        return 1000.0, 15.0, 180.0
    

def scale_input_features(input_data: np.ndarray, scaler_params: Dict) -> np.ndarray:
    """
    Scale input features using pre-saved scaler parameters.
    
    Args:
        input_data: Input array of shape (batch_size, timesteps, features)
        scaler_params: Dictionary containing 'mean' and 'scale' arrays
    
    Returns:
        Scaled input array of the same shape
    """
    nsamples, ntimesteps, nfeatures = input_data.shape
    
    # Verify feature count matches scaler
    if nfeatures != scaler_params['mean'].shape[0]:
        raise ValueError(f"Feature count mismatch: input has {nfeatures} features, "
                        f"but scaler expects {scaler_params['mean'].shape[0]} features")
    
    # Reshape to 2D for scaling: (samples * timesteps, features)
    data_reshaped = input_data.reshape((nsamples * ntimesteps, nfeatures))
    
    # Get scale values and handle division by zero
    scale = scaler_params['scale'].copy()  # Avoid modifying original
    scale[scale == 0] = 1.0  # Prevent division by zero
    
    # Apply standardization: (x - mean) / scale
    scaled_data = (data_reshaped - scaler_params['mean']) / scale
    
    # Reshape back to original 3D shape
    return scaled_data.reshape((nsamples, ntimesteps, nfeatures))
    
def prepare_lstm_sequence(weather_data: Dict, static_features: Dict) -> np.ndarray:
    try:
        daily = weather_data['daily']
        hourly = weather_data['hourly']
        if len(daily['time']) < 10 or len(hourly['time']) < 240:
            raise ValueError("Incomplete weather data received.")
            
        daily_feature_list = []
        for i in range(10):
            start_hour, end_hour = i * 24, (i + 1) * 24
            
            def get_daily_avg(data):
                if start_hour >= len(data) or end_hour > len(data):
                    return 0.0
                valid_readings = [v for v in data[start_hour:end_hour] if v is not None]
                return sum(valid_readings) / len(valid_readings) if valid_readings else 0.0

            day_record = [
                safe_get_value(daily['precipitation_sum'], i, 0.0),
                safe_get_value(daily['rain_sum'], i, 0.0),
                safe_get_value(daily['snowfall_sum'], i, 0.0),
                safe_get_value(daily['precipitation_hours'], i, 0.0),
                safe_get_value(daily['et0_fao_evapotranspiration'], i, 0.0),
                safe_get_value(daily['temperature_2m_max'], i, 15.0),
                safe_get_value(daily['temperature_2m_min'], i, 5.0),
                get_daily_avg(hourly.get('soil_moisture_0_to_7cm', [])),
                get_daily_avg(hourly.get('soil_moisture_7_to_28cm', [])),
                get_daily_avg(hourly.get('snow_depth', [])),
                get_daily_avg(hourly.get('relative_humidity_2m', [])),
                static_features['elevation'], static_features['slope'], static_features['aspect']
            ]
            daily_feature_list.append(day_record)
            
        sequence_array = np.array([daily_feature_list], dtype=np.float32)
        if sequence_array.shape != (1, 10, INPUT_FEATURES):
            raise ValueError(f"Final sequence shape is incorrect: {sequence_array.shape}")
        return sequence_array
    except (KeyError, ValueError) as e:
        logger.error(f"Error preparing LSTM sequence: {e}\n{traceback.format_exc()}")
        raise ValueError(f"Feature preparation failed: {e}")

# --- 4. Alerting Logic ---
def load_alert_config():
    if os.path.exists(ALERT_CONFIG_FILE):
        try:
            with open(ALERT_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.error("Failed to load alert configuration file.")
    return {}

def send_alert_notifications(prediction_data, lat, lon):
    if not twilio_client:
        return
    config = load_alert_config()
    sms_to = config.get('sms')
    if sms_to:
        message_body = (
            f"Rockfall Alert: {prediction_data['risk_level']} "
            f"({prediction_data['probability_percent']}) detected at "
            f"Lat: {lat:.4f}, Lon: {lon:.4f}. Please review immediately."
        )
        try:
            message = twilio_client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE_NUMBER,
                to=sms_to
            )
            logger.info(f"SMS alert sent successfully to {sms_to}, SID: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send SMS alert to {sms_to}: {e}")

# --- 5. Core Prediction Logic ---
def run_single_prediction(lat: float, lon: float) -> Dict:
    """Encapsulates the logic for a single prediction, returning a result dictionary."""
    if not model or not scaler_params:
        raise RuntimeError('Model or scaler not loaded properly')

    elevation, slope, aspect = fetch_elevation_data(lat, lon)
    static_features = {'elevation': elevation, 'slope': slope, 'aspect': aspect}
    
    weather_data = fetch_weather_data(lat, lon, 10)
    if not weather_data:
        raise RuntimeError('Failed to fetch weather data.')
    
    lstm_input = prepare_lstm_sequence(weather_data, static_features)
    lstm_scaled = scale_input_features(lstm_input, scaler_params)
    
    input_tensor = torch.tensor(lstm_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    # --- SHAP INTEGRATION: Generate explanation for the prediction ---
    shap_explanation = generate_shap_explanation(input_tensor, explainer)

    risk_levels = {0.7: "Very High Risk", 0.5: "High Risk", 0.3: "Moderate Risk"}
    risk_level = next((v for k, v in risk_levels.items() if probability >= k), "Low Risk")
    
    daily_precip = weather_data['daily'].get('precipitation_sum', [])
    daily_rain = weather_data['daily'].get('rain_sum', [])
    
    return {
        'probability': round(probability, 4),
        'probability_percent': f"{probability:.1%}",
        'risk_level': risk_level,
        'threshold': OPTIMAL_THRESHOLD,
        'explanation': shap_explanation,
        'location': {
            'elevation': round(elevation, 1),
            'slope': round(slope, 1),
            'aspect': round(aspect, 1)
        },
        'weather_summary': {
            'precipitation_sum_last_10_days': round(safe_sum(daily_precip), 2),
            'precipitation_sum_last_7_days': round(safe_sum(daily_precip[-7:]), 2),
            'precipitation_sum_last_3_days': round(safe_sum(daily_precip[-3:]), 2),
            'rain_sum_total': round(safe_sum(daily_rain), 2)
        }
    }

# --- 6. API Endpoints ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'model_loaded': model is not None,
        'scaler_loaded': scaler_params is not None,
        'shap_available': explainer is not None,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'lat' not in data or 'lon' not in data:
            return jsonify({'error': 'Missing latitude or longitude in request'}), 400
            
        lat, lon = float(data['lat']), float(data['lon'])
        logger.info(f"Processing prediction for lat={lat}, lon={lon}")
        
        result = run_single_prediction(lat, lon)

        if result['risk_level'] in ["High Risk", "Very High Risk"]:
            send_alert_notifications(result, lat, lon)

        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        locations = data.get('locations', [])
        if not locations:
            return jsonify({'error': 'No locations provided'}), 400
        
        results = []
        for i, loc in enumerate(locations):
            try:
                if 'lat' not in loc or 'lon' not in loc:
                    results.append({'error': f'Missing lat/lon in location {i+1}'})
                    continue
                    
                lat, lon = float(loc['lat']), float(loc['lon'])
                prediction_result = run_single_prediction(lat, lon)
                prediction_result['lat'] = lat
                prediction_result['lon'] = lon
                results.append(prediction_result)
            except Exception as e:
                logger.error(f"Error in batch for location {i+1} {loc}: {e}")
                results.append({'lat': loc.get('lat'), 'lon': loc.get('lon'), 'error': str(e)})
                
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_alerts', methods=['POST'])
def save_alerts():
    try:
        data = request.get_json()
        sms = data.get('sms')
        email = data.get('email')
        
        if sms and not (sms.startswith('+') and len(sms) > 10):
                return jsonify({'error': 'Invalid phone number format. Must include country code (e.g., +12223334444).'}), 400

        config = {'sms': sms, 'email': email}
        with open(ALERT_CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Alert settings saved: {config}")
        return jsonify({'message': 'Settings saved successfully!'})
    except Exception as e:
        logger.error(f"Failed to save alert settings: {e}")
        return jsonify({'error': 'Failed to save settings'}), 500

# --- 7. Run the App ---
if __name__ == '__main__':
    logger.info("Starting Landslide Prediction Server...")
    logger.info(f"SHAP explanations {'enabled' if explainer else 'disabled'}")
    app.run(host='0.0.0.0', port=5000, debug=False)
