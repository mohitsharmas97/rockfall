# import torch
# import torch.nn as nn
# import numpy as np
# import requests
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from datetime import datetime, timedelta
# from typing import Dict, Tuple, Optional, List
# import logging
# import traceback

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # --- 1. Initialize Flask App ---
# app = Flask(__name__)
# CORS(app)

# # --- 2. Re-define the Model Architecture ---
# class LandslideBiLSTM(nn.Module):
#     def __init__(self, input_features, hidden_size1, hidden_size2, dense_size):
#         super(LandslideBiLSTM, self).__init__()
#         self.lstm1 = nn.LSTM(input_features, hidden_size1, batch_first=True, bidirectional=True)
#         self.bn1 = nn.BatchNorm1d(hidden_size1 * 2)
#         self.dropout1 = nn.Dropout(0.3)
#         self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
#         self.bn2 = nn.BatchNorm1d(hidden_size2 * 2)
#         self.dropout2 = nn.Dropout(0.3)
#         self.dense1 = nn.Linear(hidden_size2 * 2, dense_size)
#         self.relu = nn.ReLU()
#         self.output_layer = nn.Linear(dense_size, 1)

#     def forward(self, x):
#         lstm_out1, _ = self.lstm1(x)
#         out = lstm_out1.permute(0, 2, 1)
#         out = self.bn1(out)
#         out = out.permute(0, 2, 1)
#         out = self.dropout1(out)
#         lstm_out2, _ = self.lstm2(out)
#         out = lstm_out2.permute(0, 2, 1)
#         out = self.bn2(out)
#         out = out.permute(0, 2, 1)
#         out = self.dropout2(out[:, -1, :])
#         out = self.dense1(out)
#         out = self.relu(out)
#         out = self.output_layer(out)
#         return out

# # --- 3. Model Configuration ---
# INPUT_FEATURES = 14
# HIDDEN_SIZE1 = 64
# HIDDEN_SIZE2 = 32
# DENSE_SIZE = 16
# OPTIMAL_THRESHOLD = 0.44

# # Load model and scaler
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = None
# scaler_params = None

# try:
#     model = LandslideBiLSTM(INPUT_FEATURES, HIDDEN_SIZE1, HIDDEN_SIZE2, DENSE_SIZE).to(device)
#     model.load_state_dict(torch.load('landslide_model.pth', map_location=device))
#     model.eval()
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Fatal error loading model: {e}")

# try:
#     scaler_data = np.load('scaler_params.npz')
#     scaler_params = {'mean': scaler_data['mean'], 'scale': scaler_data['scale']}
#     logger.info(f"Scaler parameters loaded successfully. Shape: {scaler_params['mean'].shape}")
# except Exception as e:
#     logger.error(f"Fatal error loading scaler parameters: {e}")

# # --- 4. Helper Functions ---
# def safe_get(data: any, default: float = 0.0) -> float:
#     """Safely get a value, returning default if None or invalid."""
#     if data is None:
#         return default
#     try:
#         return float(data)
#     except (TypeError, ValueError):
#         return default

# def safe_list_get(lst: List, index: int, default: float = 0.0) -> float:
#     """Safely get a value from a list by index."""
#     if lst is None or not isinstance(lst, list):
#         return default
#     if index < 0 or index >= len(lst):
#         return default
#     return safe_get(lst[index], default)

# def safe_mean(lst: List, default: float = 0.0) -> float:
#     """Calculate mean of a list, handling None values."""
#     if not lst or not isinstance(lst, list):
#         return default
#     valid_values = [v for v in lst if v is not None]
#     if not valid_values:
#         return default
#     try:
#         return sum(valid_values) / len(valid_values)
#     except (TypeError, ZeroDivisionError):
#         return default

# def fetch_elevation_data(lat: float, lon: float) -> Tuple[float, float, float, float]:
#     """Fetch elevation, slope, and aspect from Open Topo Data API."""
#     try:
#         grid_size = 0.01
#         positions = [
#             (lat, lon), 
#             (lat + grid_size, lon), 
#             (lat - grid_size, lon), 
#             (lat, lon + grid_size), 
#             (lat, lon - grid_size)
#         ]
#         locations_str = "|".join([f"{p_lat},{p_lon}" for p_lat, p_lon in positions])
#         url = f"https://api.opentopodata.org/v1/mapzen?locations={locations_str}"
        
#         response = requests.get(url, timeout=15)
#         response.raise_for_status()
#         data = response.json()

#         if 'results' not in data or len(data['results']) < 1:
#             raise ValueError("Invalid API response from Open Topo Data")
            
#         center_point = data['results'][0]
#         elevation = safe_get(center_point.get('elevation'), 1000.0)
#         slope = safe_get(center_point.get('slope'), 15.0)
#         aspect = safe_get(center_point.get('aspect'), 180.0)
        
#         # Calculate curvature (for API response only, not used in model)
#         curvature = 0.0
#         if len(data['results']) == 5:
#             results = data['results']
#             elevations = [safe_get(res.get('elevation'), elevation) for res in results]
#             center_elev, north_elev, south_elev, east_elev, west_elev = elevations
            
#             cell_size = grid_size * 111000  # Convert degrees to meters (approximate)
#             if cell_size > 0:
#                 d_xx = (east_elev + west_elev - 2 * center_elev) / (cell_size**2)
#                 d_yy = (north_elev + south_elev - 2 * center_elev) / (cell_size**2)
#                 curvature = (d_xx + d_yy) * -1

#         return elevation, slope, aspect, curvature
        
#     except Exception as e:
#         logger.warning(f"Error fetching topography data: {e}. Using fallback values.")
#         return 1000.0, 15.0, 180.0, 0.0

# def fetch_weather_data(lat: float, lon: float, days: int = 10) -> Optional[Dict]:
#     """Fetch historical weather data from Open-Meteo Archive API."""
#     try:
#         end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
#         start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
#         # Use the full API URL as provided
#         url = (
#             f"https://archive-api.open-meteo.com/v1/archive?"
#             f"latitude={lat}&longitude={lon}"
#             f"&start_date={start_date}&end_date={end_date}"
#             f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,"
#             f"snowfall_sum,precipitation_hours,et0_fao_evapotranspiration"
#             f"&hourly=relative_humidity_2m,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,snow_depth"
#             f"&timezone=GMT"
#         )
        
#         logger.info(f"Fetching weather data from: {url}")
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()
#         data = response.json()
        
#         if 'daily' not in data or 'hourly' not in data:
#             logger.error(f"Invalid weather data response: Missing 'daily' or 'hourly' keys")
#             return None
            
#         # Validate and clean the data
#         daily = data.get('daily', {})
#         hourly = data.get('hourly', {})
        
#         # Ensure all required fields exist with default values if missing
#         required_daily = [
#             'precipitation_sum', 'rain_sum', 'snowfall_sum', 
#             'precipitation_hours', 'et0_fao_evapotranspiration',
#             'temperature_2m_max', 'temperature_2m_min'
#         ]
        
#         required_hourly = [
#             'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm',
#             'snow_depth', 'relative_humidity_2m'
#         ]
        
#         # Fill missing daily data with defaults
#         for field in required_daily:
#             if field not in daily or daily[field] is None:
#                 daily[field] = [0.0] * days
#             else:
#                 # Replace None values with 0.0
#                 daily[field] = [safe_get(v, 0.0) for v in daily[field]]
        
#         # Fill missing hourly data with defaults
#         expected_hourly_length = days * 24
#         for field in required_hourly:
#             if field not in hourly or hourly[field] is None:
#                 hourly[field] = [0.0] * expected_hourly_length
#             else:
#                 # Replace None values with 0.0
#                 hourly[field] = [safe_get(v, 0.0) for v in hourly[field]]
        
#         data['daily'] = daily
#         data['hourly'] = hourly
        
#         logger.info(f"Weather data fetched successfully for {lat},{lon}")
#         return data
        
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Failed to fetch weather data for {lat},{lon}: {e}")
#         return None
#     except Exception as e:
#         logger.error(f"Unexpected error processing weather data: {e}")
#         return None

# def calculate_derived_features(weather_data: Dict) -> Dict:
#     """Calculate derived features from weather data robustly."""
#     daily_data = weather_data.get('daily', {})
#     features = {}
    
#     # Get precipitation data with safe defaults
#     precipitation_sum = daily_data.get('precipitation_sum', [])
#     if not isinstance(precipitation_sum, list):
#         precipitation_sum = []
    
#     # Clean the data (remove None values)
#     precipitation_sum = [safe_get(p, 0.0) for p in precipitation_sum]
    
#     # Calculate sums safely
#     features['precipitation_sum_last_10_days'] = sum(precipitation_sum) if precipitation_sum else 0.0
#     features['precipitation_sum_last_7_days'] = sum(precipitation_sum[-7:]) if len(precipitation_sum) >= 7 else sum(precipitation_sum)
#     features['precipitation_sum_last_3_days'] = sum(precipitation_sum[-3:]) if len(precipitation_sum) >= 3 else sum(precipitation_sum)
    
#     # Add more derived features if needed
#     rain_sum = daily_data.get('rain_sum', [])
#     if isinstance(rain_sum, list):
#         rain_sum = [safe_get(r, 0.0) for r in rain_sum]
#         features['rain_sum_total'] = sum(rain_sum)
#     else:
#         features['rain_sum_total'] = 0.0
    
#     return features

# def prepare_lstm_sequence(weather_data: Dict, static_features: Dict) -> np.ndarray:
#     """Prepare 10-day sequence with 14 features for the LSTM model."""
#     try:
#         daily = weather_data.get('daily', {})
#         hourly = weather_data.get('hourly', {})
        
#         # Determine number of days available
#         time_data = daily.get('time', [])
#         num_days = min(10, len(time_data)) if time_data else 10
        
#         sequence = []
        
#         for day in range(num_days):
#             # Calculate hour indices for this day
#             start_hour = day * 24
#             end_hour = (day + 1) * 24
            
#             # Extract features with safe defaults
#             day_features = [
#                 safe_list_get(daily.get('precipitation_sum', []), day, 0.0),
#                 safe_list_get(daily.get('rain_sum', []), day, 0.0),
#                 safe_list_get(daily.get('snowfall_sum', []), day, 0.0),
#                 safe_list_get(daily.get('precipitation_hours', []), day, 0.0),
#                 safe_list_get(daily.get('et0_fao_evapotranspiration', []), day, 1.0),
#                 safe_list_get(daily.get('temperature_2m_max', []), day, 20.0),
#                 safe_list_get(daily.get('temperature_2m_min', []), day, 10.0),
#                 safe_mean(hourly.get('soil_moisture_0_to_7cm', [])[start_hour:end_hour], 0.15),
#                 safe_mean(hourly.get('soil_moisture_7_to_28cm', [])[start_hour:end_hour], 0.20),
#                 safe_mean(hourly.get('snow_depth', [])[start_hour:end_hour], 0.0),
#                 safe_mean(hourly.get('relative_humidity_2m', [])[start_hour:end_hour], 60.0),
#                 static_features.get('elevation', 1000.0),
#                 static_features.get('slope', 15.0),
#                 static_features.get('aspect', 180.0)
#             ]
            
#             # Ensure all values are floats and not None
#             day_features = [float(f) if f is not None else 0.0 for f in day_features]
#             sequence.append(day_features)
        
#         # Pad sequence to 10 days if necessary
#         while len(sequence) < 10:
#             if sequence:
#                 # Repeat last day's data
#                 last_day = sequence[-1].copy()
#             else:
#                 # Use default values
#                 last_day = [0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 10.0, 0.15, 0.20, 0.0, 60.0,
#                            static_features.get('elevation', 1000.0),
#                            static_features.get('slope', 15.0),
#                            static_features.get('aspect', 180.0)]
#             sequence.append(last_day)
        
#         # Convert to numpy array
#         sequence_array = np.array([sequence], dtype=np.float32)
        
#         # Validate shape
#         if sequence_array.shape != (1, 10, INPUT_FEATURES):
#             logger.error(f"Invalid sequence shape: {sequence_array.shape}, expected (1, 10, {INPUT_FEATURES})")
#             return np.zeros((1, 10, INPUT_FEATURES), dtype=np.float32)
            
#         return sequence_array
        
#     except Exception as e:
#         logger.error(f"Error preparing LSTM sequence: {e}\n{traceback.format_exc()}")
#         # Return default values
#         default_sequence = np.zeros((1, 10, INPUT_FEATURES), dtype=np.float32)
#         # Set some reasonable defaults
#         default_sequence[:, :, 5] = 20.0  # temperature max
#         default_sequence[:, :, 6] = 10.0  # temperature min
#         default_sequence[:, :, 10] = 60.0  # humidity
#         default_sequence[:, :, 11] = static_features.get('elevation', 1000.0)
#         default_sequence[:, :, 12] = static_features.get('slope', 15.0)
#         default_sequence[:, :, 13] = static_features.get('aspect', 180.0)
#         return default_sequence

# def scale_input_features(input_data: np.ndarray, scaler_params: Dict) -> np.ndarray:
#     """Scales the input data using loaded scaler parameters."""
#     nsamples, ntimesteps, nfeatures = input_data.shape
    
#     if nfeatures != scaler_params['mean'].shape[0]:
#         raise ValueError(f"Shape mismatch: Input has {nfeatures} features, scaler expects {scaler_params['mean'].shape[0]}")
    
#     data_reshaped = input_data.reshape((nsamples * ntimesteps, nfeatures))
    
#     # Avoid division by zero
#     scale = scaler_params['scale'].copy()
#     scale[scale == 0] = 1.0
    
#     scaled_data = (data_reshaped - scaler_params['mean']) / scale
#     return scaled_data.reshape((nsamples, ntimesteps, nfeatures))

# # --- 5. API Endpoints ---
# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint to verify service status."""
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': model is not None,
#         'scaler_loaded': scaler_params is not None,
#         'device': str(device)
#     })

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Main prediction endpoint for a single location."""
#     if not model or not scaler_params:
#         return jsonify({'error': 'Model or scaler not loaded properly'}), 500
    
#     try:
#         data = request.get_json()
#         if not data or 'lat' not in data or 'lon' not in data:
#             return jsonify({'error': 'Latitude and longitude are required'}), 400
        
#         lat = float(data['lat'])
#         lon = float(data['lon'])
        
#         # Validate coordinates
#         if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
#             return jsonify({'error': 'Invalid coordinates'}), 400
        
#         logger.info(f"Processing prediction for lat={lat}, lon={lon}")
        
#         # Fetch elevation data
#         elevation, slope, aspect, curvature = fetch_elevation_data(lat, lon)
#         static_features = {
#             'elevation': elevation,
#             'slope': slope,
#             'aspect': aspect
#         }
        
#         # Fetch weather data
#         weather_data = fetch_weather_data(lat, lon, 10)
#         if not weather_data:
#             return jsonify({'error': 'Failed to fetch weather data. Please try again later.'}), 500
        
#         # Prepare LSTM input
#         lstm_input = prepare_lstm_sequence(weather_data, static_features)
        
#         # Scale features
#         lstm_scaled = scale_input_features(lstm_input, scaler_params)
        
#         # Make prediction
#         with torch.no_grad():
#             model_input = torch.tensor(lstm_scaled, dtype=torch.float32).to(device)
#             output = model(model_input)
#             probability = torch.sigmoid(output).item()
        
#         # Determine risk level
#         if probability >= 0.7:
#             risk_level = "Very High Risk"
#         elif probability >= 0.5:
#             risk_level = "High Risk"
#         elif probability >= 0.3:
#             risk_level = "Moderate Risk"
#         else:
#             risk_level = "Low Risk"
        
#         # Calculate derived features
#         derived_features = calculate_derived_features(weather_data)
        
#         # Format precipitation value
#         precip_value = derived_features.get('precipitation_sum_last_10_days', 0)
#         precip_formatted = f"{precip_value:.1f}mm" if isinstance(precip_value, (int, float)) else "N/A"
        
#         response_data = {
#             'probability': round(probability, 4),
#             'probability_percent': f"{probability:.1%}",
#             'prediction': 1 if probability >= OPTIMAL_THRESHOLD else 0,
#             'risk_level': risk_level,
#             'threshold': OPTIMAL_THRESHOLD,
#             'location': {
#                 'latitude': lat,
#                 'longitude': lon,
#                 'elevation': f"{elevation:.1f}m",
#                 'slope': f"{slope:.1f}°",
#                 'aspect': f"{aspect:.1f}°",
#                 'curvature': f"{curvature:.4f}"
#             },
#             'weather_summary': {
#                 'precipitation_sum_last_10_days': precip_formatted,
#                 'precipitation_sum_last_7_days': f"{derived_features.get('precipitation_sum_last_7_days', 0):.1f}mm",
#                 'precipitation_sum_last_3_days': f"{derived_features.get('precipitation_sum_last_3_days', 0):.1f}mm",
#                 'rain_sum_total': f"{derived_features.get('rain_sum_total', 0):.1f}mm"
#             }
#         }
        
#         logger.info(f"Prediction successful for {lat},{lon}: probability={probability:.4f}")
#         return jsonify(response_data)
        
#     except Exception as e:
#         logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
#         return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

# @app.route('/batch_predict', methods=['POST'])
# def batch_predict():
#     """Batch prediction endpoint for multiple locations."""
#     if not model or not scaler_params:
#         return jsonify({'error': 'Model or scaler not loaded'}), 500
    
#     try:
#         data = request.get_json()
#         locations = data.get('locations', []) if data else []
        
#         if not isinstance(locations, list) or not locations:
#             return jsonify({'error': 'A non-empty list of locations is required'}), 400
        
#         if len(locations) > 50:
#             return jsonify({'error': 'Maximum 50 locations allowed per batch'}), 400
        
#         results = []
        
#         for i, loc in enumerate(locations):
#             try:
#                 lat = float(loc.get('lat', 0))
#                 lon = float(loc.get('lon', 0))
                
#                 if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
#                     results.append({
#                         'index': i,
#                         'lat': lat,
#                         'lon': lon,
#                         'error': 'Invalid coordinates'
#                     })
#                     continue
                
#                 # Fetch data
#                 elevation, slope, aspect, _ = fetch_elevation_data(lat, lon)
#                 static_features = {
#                     'elevation': elevation,
#                     'slope': slope,
#                     'aspect': aspect
#                 }
                
#                 weather_data = fetch_weather_data(lat, lon, 10)
#                 if not weather_data:
#                     results.append({
#                         'index': i,
#                         'lat': lat,
#                         'lon': lon,
#                         'error': 'Weather data unavailable'
#                     })
#                     continue
                
#                 # Prepare and predict
#                 lstm_input = prepare_lstm_sequence(weather_data, static_features)
#                 lstm_scaled = scale_input_features(lstm_input, scaler_params)
                
#                 with torch.no_grad():
#                     model_input = torch.tensor(lstm_scaled, dtype=torch.float32).to(device)
#                     output = model(model_input)
#                     probability = torch.sigmoid(output).item()
                
#                 results.append({
#                     'index': i,
#                     'lat': lat,
#                     'lon': lon,
#                     'probability': round(probability, 4),
#                     'prediction': 1 if probability >= OPTIMAL_THRESHOLD else 0,
#                     'risk_level': (
#                         "Very High Risk" if probability >= 0.7 else
#                         "High Risk" if probability >= 0.5 else
#                         "Moderate Risk" if probability >= 0.3 else
#                         "Low Risk"
#                     )
#                 })
                
#             except Exception as e:
#                 logger.error(f"Error processing location {i}: {e}")
#                 results.append({
#                     'index': i,
#                     'lat': loc.get('lat'),
#                     'lon': loc.get('lon'),
#                     'error': f'Processing error: {str(e)}'
#                 })
        
#         return jsonify({'results': results})
        
#     except Exception as e:
#         logger.error(f"Batch prediction error: {e}\n{traceback.format_exc()}")
#         return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

# # --- 6. Run the App ---
# if __name__ == '__main__':
#     logger.info("Starting Landslide Prediction Server...")
#     logger.info(f"Using device: {device}")
#     logger.info(f"Model loaded: {model is not None}")
#     logger.info(f"Scaler loaded: {scaler_params is not None}")
#     app.run(host='0.0.0.0', port=5000, debug=False)



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
import time # <-- Imported to handle delays for API retries
from twilio.rest import Client
# from dotenv import load_dotenv
# load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Initialize Flask App ---
app = Flask(__name__)
CORS(app)

TWILIO_ACCOUNT_SID = "Ax"
TWILIO_AUTH_TOKEN = "x"
TWILIO_PHONE_NUMBER = "x"

if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    print("WARNING: Twilio credentials are not fully configured. SMS alerts will be disabled.")
    twilio_client = None
else:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

ALERT_CONFIG_FILE = 'alerts.json'

# --- 2. Re-define the Model Architecture ---

class Attention(nn.Module):
    def _init_(self, hidden_size):
        super(Attention, self)._init_()
        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.context_vector = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attn(lstm_output))
        attn_weights = self.context_vector(attn_weights).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context

class BiLSTMAttention(nn.Module):
    def _init_(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(BiLSTMAttention, self)._init_()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_vector = self.attention(lstm_out)
        out = self.fc(context_vector)
        return out

# --- 3. Model Configuration ---
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
    model = BiLSTMAttention(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    model.load_state_dict(torch.load('rockfall_bilstm_attention.pth', map_location=device))
    model.eval()
    logger.info("BiLSTM with Attention model loaded successfully")
except Exception as e:
    logger.error(f"Fatal error loading model: {e}")

try:
    scaler_data = np.load('scaler_params.npz')
    scaler_params = {'mean': scaler_data['mean'], 'scale': scaler_data['scale']}
    logger.info(f"Scaler parameters loaded successfully. Shape: {scaler_params['mean'].shape}")
except Exception as e:
    logger.error(f"Fatal error loading scaler parameters: {e}")

# --- 4. Helper Functions ---
def safe_get(data: any, default: float = 0.0) -> float:
    if data is None: return default
    try: return float(data)
    except (TypeError, ValueError): return default

def safe_list_get(lst: List, index: int, default: float = 0.0) -> float:
    if lst is None or not isinstance(lst, list) or not (0 <= index < len(lst)):
        return default
    return safe_get(lst[index], default)

def safe_mean(lst: List, default: float = 0.0) -> float:
    if not lst or not isinstance(lst, list): return default
    valid_values = [v for v in lst if v is not None]
    if not valid_values: return default
    try: return sum(valid_values) / len(valid_values)
    except (TypeError, ZeroDivisionError): return default

def fetch_elevation_data(lat: float, lon: float) -> Tuple[float, float, float, float]:
    try:
        grid_size = 0.01
        positions = [(lat, lon), (lat + grid_size, lon), (lat - grid_size, lon), (lat, lon + grid_size), (lat, lon - grid_size)]
        locations_str = "|".join([f"{p_lat},{p_lon}" for p_lat, p_lon in positions])
        url = f"https://api.opentopodata.org/v1/mapzen?locations={locations_str}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if 'results' not in data or not data['results']: raise ValueError("Invalid API response from Open Topo Data")
        center_point = data['results'][0]
        elevation = safe_get(center_point.get('elevation'), 1000.0)
        slope = safe_get(center_point.get('slope'), 15.0)
        aspect = safe_get(center_point.get('aspect'), 180.0)
        curvature = 0.0
        if len(data['results']) == 5:
            elevations = [safe_get(res.get('elevation'), elevation) for res in data['results']]
            center_elev, north_elev, south_elev, east_elev, west_elev = elevations
            cell_size = grid_size * 111000
            if cell_size > 0:
                d_xx = (east_elev + west_elev - 2 * center_elev) / (cell_size**2)
                d_yy = (north_elev + south_elev - 2 * center_elev) / (cell_size**2)
                curvature = (d_xx + d_yy) * -1
        return elevation, slope, aspect, curvature
    except Exception as e:
        logger.warning(f"Error fetching topography data: {e}. Using fallback values.")
        return 1000.0, 15.0, 180.0, 0.0

# --- UPDATED WEATHER FETCHING FUNCTION ---
def fetch_weather_data(lat: float, lon: float, days: int = 10) -> Optional[Dict]:
    """
    Fetch historical weather data from Open-Meteo Archive API with retry logic.
    This is the same robust method used in your training data script.
    """
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    daily_params = "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,et0_fao_evapotranspiration"
    hourly_params = "relative_humidity_2m,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,snow_depth"

    params = {
        'latitude': lat, 'longitude': lon, 'start_date': start_date, 'end_date': end_date,
        'daily': daily_params, 'hourly': hourly_params, 'timezone': 'GMT'
    }

    max_retries = 5
    backoff_factor = 2
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 429:
                retry_after = attempt * backoff_factor + backoff_factor
                logger.warning(f"Rate limit hit for ({lat},{lon}). Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()
            
            if 'daily' not in data or 'hourly' not in data:
                logger.error(f"Invalid weather data for ({lat},{lon}): Missing 'daily' or 'hourly' keys")
                return None
            
            # --- Data Validation and Cleaning (moved from old function) ---
            for field in daily_params.split(','):
                if field not in data['daily'] or data['daily'][field] is None:
                    data['daily'][field] = [0.0] * days
                else:
                    data['daily'][field] = [safe_get(v, 0.0) for v in data['daily'][field]]

            for field in hourly_params.split(','):
                if field not in data['hourly'] or data['hourly'][field] is None:
                    data['hourly'][field] = [0.0] * (days * 24)
                else:
                    data['hourly'][field] = [safe_get(v, 0.0) for v in data['hourly'][field]]

            logger.info(f"Weather data fetched successfully for ({lat},{lon})")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for ({lat},{lon}) on attempt {attempt + 1}: {e}")
            if attempt >= max_retries - 1:
                return None
            time.sleep(backoff_factor)

    return None # Failed all retries

def send_alert_notification(risk_level, probability_percent, location_info):
    if risk_level not in ["High Risk", "Very High Risk"] or not twilio_client: return
    try:
        if not os.path.exists(ALERT_CONFIG_FILE): return
        with open(ALERT_CONFIG_FILE, 'r') as f: config = json.load(f)
        target_phone_number = config.get('sms')
        if not target_phone_number: return
        lat, lon = location_info.get('lat', 'N/A'), location_info.get('lon', 'N/A')
        message_body = (
            f"GeoGuard AI Alert: {risk_level}!\n"
            f"A rockfall probability of {probability_percent} was detected near coordinates ({lat}, {lon}).\n"
            f"Please review the dashboard and initiate safety protocols immediately."
        )
        message = twilio_client.messages.create(body=message_body, from_=TWILIO_PHONE_NUMBER, to=target_phone_number)
        print(f"Successfully sent SMS alert, SID: {message.sid}")
    except Exception as e:
        print(f"FAILED to send SMS alert: {e}")

def calculate_derived_features(weather_data: Dict) -> Dict:
    daily_data = weather_data.get('daily', {})
    features = {}
    precipitation_sum = [safe_get(p, 0.0) for p in daily_data.get('precipitation_sum', [])]
    features['precipitation_sum_last_10_days'] = sum(precipitation_sum)
    features['precipitation_sum_last_7_days'] = sum(precipitation_sum[-7:])
    features['precipitation_sum_last_3_days'] = sum(precipitation_sum[-3:])
    rain_sum = [safe_get(r, 0.0) for r in daily_data.get('rain_sum', [])]
    features['rain_sum_total'] = sum(rain_sum)
    return features

def prepare_lstm_sequence(weather_data: Dict, static_features: Dict) -> np.ndarray:
    try:
        daily, hourly = weather_data['daily'], weather_data['hourly']
        num_days = min(10, len(daily.get('time', []))) if daily.get('time') else 10
        sequence = []
        for day in range(num_days):
            start_hour, end_hour = day * 24, (day + 1) * 24
            day_features = [
                safe_list_get(daily['precipitation_sum'], day), safe_list_get(daily['rain_sum'], day),
                safe_list_get(daily['snowfall_sum'], day), safe_list_get(daily['precipitation_hours'], day),
                safe_list_get(daily['et0_fao_evapotranspiration'], day, 1.0),
                safe_list_get(daily['temperature_2m_max'], day, 20.0),
                safe_list_get(daily['temperature_2m_min'], day, 10.0),
                safe_mean(hourly['soil_moisture_0_to_7cm'][start_hour:end_hour], 0.15),
                safe_mean(hourly['soil_moisture_7_to_28cm'][start_hour:end_hour], 0.20),
                safe_mean(hourly['snow_depth'][start_hour:end_hour], 0.0),
                safe_mean(hourly['relative_humidity_2m'][start_hour:end_hour], 60.0),
                static_features['elevation'], static_features['slope'], static_features['aspect']
            ]
            sequence.append([float(f) for f in day_features])
        while len(sequence) < 10:
            last_day = sequence[-1] if sequence else [0.0]*11 + [static_features['elevation'], static_features['slope'], static_features['aspect']]
            sequence.append(last_day)
        sequence_array = np.array([sequence], dtype=np.float32)
        if sequence_array.shape != (1, 10, INPUT_FEATURES):
            raise ValueError(f"Invalid sequence shape: {sequence_array.shape}")
        return sequence_array
    except Exception as e:
        logger.error(f"Error preparing LSTM sequence: {e}\n{traceback.format_exc()}")
        default_sequence = np.zeros((1, 10, INPUT_FEATURES), dtype=np.float32)
        default_sequence[:, :, 5] = 20.0; default_sequence[:, :, 6] = 10.0; default_sequence[:, :, 10] = 60.0
        default_sequence[:, :, 11:] = [static_features['elevation'], static_features['slope'], static_features['aspect']]
        return default_sequence

def scale_input_features(input_data: np.ndarray, scaler_params: Dict) -> np.ndarray:
    nsamples, ntimesteps, nfeatures = input_data.shape
    if nfeatures != scaler_params['mean'].shape[0]:
        raise ValueError(f"Shape mismatch: Input has {nfeatures} features, scaler expects {scaler_params['mean'].shape[0]}")
    data_reshaped = input_data.reshape((nsamples * ntimesteps, nfeatures))
    scale = scaler_params['scale'].copy()
    scale[scale == 0] = 1.0
    scaled_data = (data_reshaped - scaler_params['mean']) / scale
    return scaled_data.reshape((nsamples, ntimesteps, nfeatures))

# --- 5. API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None, 'scaler_loaded': scaler_params is not None})

@app.route('/save_alerts', methods=['POST'])
def save_alerts():
    try:
        data = request.json
        if not data.get('sms') and not data.get('email'):
            return jsonify({'error': 'No contact information provided'}), 400
        with open(ALERT_CONFIG_FILE, 'w') as f: json.dump(data, f)
        return jsonify({'message': 'Alert settings saved successfully!'})
    except Exception as e:
        print(f"Error saving alert config: {e}")
        return jsonify({'error': 'Failed to save settings'}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler_params:
        return jsonify({'error': 'Model or scaler not loaded properly'}), 500
    try:
        data = request.get_json()
        lat, lon = float(data['lat']), float(data['lon'])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400
        
        logger.info(f"Processing prediction for lat={lat}, lon={lon}")
        
        elevation, slope, aspect, curvature = fetch_elevation_data(lat, lon)
        static_features = {'elevation': elevation, 'slope': slope, 'aspect': aspect}
        
        weather_data = fetch_weather_data(lat, lon, 10)
        if not weather_data:
            return jsonify({'error': 'Failed to fetch weather data. Please try again later.'}), 500
        
        lstm_input = prepare_lstm_sequence(weather_data, static_features)
        lstm_scaled = scale_input_features(lstm_input, scaler_params)
        
        with torch.no_grad():
            output = model(torch.tensor(lstm_scaled, dtype=torch.float32).to(device))
            probability = torch.sigmoid(output).item()
        
        risk_levels = {0.7: "Very High Risk", 0.5: "High Risk", 0.3: "Moderate Risk"}
        risk_level = next((v for k, v in risk_levels.items() if probability >= k), "Low Risk")
        
        derived_features = calculate_derived_features(weather_data)
        
        response_data = {
            'probability': round(probability, 4), 'probability_percent': f"{probability:.1%}",
            'prediction': 1 if probability >= OPTIMAL_THRESHOLD else 0, 'risk_level': risk_level,
            'location': {'latitude': lat, 'longitude': lon, 'elevation': f"{elevation:.1f}m", 'slope': f"{slope:.1f}°", 'aspect': f"{aspect:.1f}°", 'curvature': f"{curvature:.4f}"},
            'weather_summary': {
                'precipitation_sum_last_10_days': f"{derived_features.get('precipitation_sum_last_10_days', 0):.1f}mm",
                'precipitation_sum_last_7_days': f"{derived_features.get('precipitation_sum_last_7_days', 0):.1f}mm",
                'precipitation_sum_last_3_days': f"{derived_features.get('precipitation_sum_last_3_days', 0):.1f}mm",
            }
        }
        
        send_alert_notification(risk_level, response_data['probability_percent'], {'lat': lat, 'lon': lon})
        
        logger.info(f"Prediction successful for {lat},{lon}: probability={probability:.4f}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if not model or not scaler_params:
        return jsonify({'error': 'Model or scaler not loaded'}), 500
    try:
        locations = request.get_json().get('locations', [])
        if not locations or len(locations) > 50:
            return jsonify({'error': 'A list of 1 to 50 locations is required'}), 400
        
        results = []
        for i, loc in enumerate(locations):
            try:
                lat, lon = float(loc.get('lat', 0)), float(loc.get('lon', 0))
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    results.append({'index': i, 'lat': lat, 'lon': lon, 'error': 'Invalid coordinates'})
                    continue
                
                elevation, slope, aspect, _ = fetch_elevation_data(lat, lon)
                static_features = {'elevation': elevation, 'slope': slope, 'aspect': aspect}
                
                weather_data = fetch_weather_data(lat, lon, 10)
                if not weather_data:
                    results.append({'index': i, 'lat': lat, 'lon': lon, 'error': 'Weather data unavailable'})
                    continue
                
                lstm_input = prepare_lstm_sequence(weather_data, static_features)
                lstm_scaled = scale_input_features(lstm_input, scaler_params)
                
                with torch.no_grad():
                    output = model(torch.tensor(lstm_scaled, dtype=torch.float32).to(device))
                    probability = torch.sigmoid(output).item()
                
                risk_levels = {0.7: "Very High Risk", 0.5: "High Risk", 0.3: "Moderate Risk"}
                risk_level = next((v for k, v in risk_levels.items() if probability >= k), "Low Risk")
                
                results.append({'index': i, 'lat': lat, 'lon': lon, 'probability': round(probability, 4), 'risk_level': risk_level})
            except Exception as e:
                logger.error(f"Error processing location {i}: {e}")
                results.append({'index': i, 'lat': loc.get('lat'), 'lon': loc.get('lon'), 'error': f'Processing error: {str(e)}'})
        
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Batch prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

# --- 6. Run the App ---
if __name__ == '__main__':
    logger.info("Starting Landslide Prediction Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)