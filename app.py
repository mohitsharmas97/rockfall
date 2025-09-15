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
# from dotenv import load_dotenv
# load_dotenv()

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE_NUMBER = ""
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]) else None
if not twilio_client:
    logger.warning("Twilio credentials not fully configured. SMS alerts disabled.")
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
    model = BiLSTMAttention(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    model.load_state_dict(torch.load('.pth', map_location=device))
    model.eval()
    logger.info("BiLSTM with Attention model loaded successfully")
except Exception as e:
    logger.error(f"Fatal error loading model: {e}")

try:
    scaler_data = np.load('scaler_params.npz')
    scaler_params = {'mean': scaler_data['mean'], 'scale': scaler_data['scale']}
    logger.info(f"Scaler parameters loaded successfully")
except Exception as e:
    logger.error(f"Fatal error loading scaler parameters: {e}")

# --- 3. Helper Functions ---

# --- REFACTORED FEATURE PREPARATION FUNCTION ---
def prepare_lstm_sequence(weather_data: Dict, static_features: Dict) -> np.ndarray:
    """
    Prepare 10-day sequence for the LSTM model.
    THIS FUNCTION IS NOW A DIRECT MIRROR OF THE TRAINING SCRIPT'S LOGIC.
    """
    try:
        daily = weather_data.get('daily', {})
        hourly = weather_data.get('hourly', {})

        # Basic data completeness check, similar to the training script
        if len(daily.get('time', [])) < 10 or len(hourly.get('time', [])) < 240:
            raise ValueError("Incomplete weather data received from API.")

        daily_feature_list = []
        for i in range(10): # For each of the 10 days
            start_hour_index = i * 24
            end_hour_index = start_hour_index + 24

            # Helper to calculate daily average from 24 hourly readings
            def get_daily_avg(hourly_data_list):
                hourly_slice = hourly_data_list[start_hour_index:end_hour_index]
                valid_readings = [val for val in hourly_slice if val is not None]
                return sum(valid_readings) / len(valid_readings) if valid_readings else 0.0

            # --- Feature extraction in the EXACT order as the training script ---
            # 1. daily_precip
            daily_precip = daily['precipitation_sum'][i]
            # 2. daily_rain
            daily_rain = daily['rain_sum'][i]
            # 3. daily_snow
            daily_snow = daily['snowfall_sum'][i]
            # 4. precip_hours
            precip_hours = daily['precipitation_hours'][i]
            # 5. et0_evapotranspiration
            et0_evapotranspiration = daily['et0_fao_evapotranspiration'][i]
            # 6. max_temp
            max_temp = daily['temperature_2m_max'][i]
            # 7. min_temp
            min_temp = daily['temperature_2m_min'][i]
            # 8. avg_soil_moisture_0_7cm
            avg_soil_moisture_0_7 = get_daily_avg(hourly['soil_moisture_0_to_7cm'])
            # 9. avg_soil_moisture_7_28cm
            avg_soil_moisture_7_28 = get_daily_avg(hourly['soil_moisture_7_to_28cm'])
            # 10. avg_snow_depth
            avg_snow_depth = get_daily_avg(hourly['snow_depth'])
            # 11. avg_relative_humidity
            avg_humidity = get_daily_avg(hourly['relative_humidity_2m'])
            
            # The 3 static features are added at the end of each day's record
            # 12. elevation
            elevation = static_features['elevation']
            # 13. slope
            slope = static_features['slope']
            # 14. aspect
            aspect = static_features['aspect']

            # Append the 14 features for the current day to our sequence
            day_record = [
                daily_precip, daily_rain, daily_snow, precip_hours,
                et0_evapotranspiration, max_temp, min_temp,
                avg_soil_moisture_0_7, avg_soil_moisture_7_28,
                avg_snow_depth, avg_humidity,
                elevation, slope, aspect
            ]
            daily_feature_list.append(day_record)
        
        # Convert the list of lists to a numpy array and add a batch dimension
        sequence_array = np.array([daily_feature_list], dtype=np.float32)

        if sequence_array.shape != (1, 10, INPUT_FEATURES):
            raise ValueError(f"Final sequence shape is incorrect: {sequence_array.shape}")

        return sequence_array

    except Exception as e:
        logger.error(f"Error preparing LSTM sequence: {e}\n{traceback.format_exc()}")
        # Fallback to a zero-array if preparation fails
        return np.zeros((1, 10, INPUT_FEATURES), dtype=np.float32)

# Other helper functions (no changes needed)
def fetch_weather_data(lat: float, lon: float, days: int = 10) -> Optional[Dict]:
    """Fetches weather data with retry logic."""
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
            if attempt >= 4: return None
            time.sleep(2)
    return None

def fetch_elevation_data(lat: float, lon: float) -> Tuple[float, float, float, float]:
    """Fetches elevation and related data."""
    try:
        locations_str = f"{lat},{lon}"
        url = f"https://api.opentopodata.org/v1/mapzen?locations={locations_str}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        center_point = data['results'][0]
        return (
            center_point.get('elevation', 1000.0),
            center_point.get('slope', 15.0),
            center_point.get('aspect', 180.0),
            0.0 # Curvature placeholder
        )
    except Exception as e:
        logger.warning(f"Error fetching topography data: {e}. Using fallback values.")
        return 1000.0, 15.0, 180.0, 0.0

def scale_input_features(input_data: np.ndarray, scaler_params: Dict) -> np.ndarray:
    """Scales data using loaded scaler params."""
    nsamples, ntimesteps, nfeatures = input_data.shape
    if nfeatures != scaler_params['mean'].shape[0]:
        raise ValueError("Feature count mismatch between input data and scaler.")
    data_reshaped = input_data.reshape((nsamples * ntimesteps, nfeatures))
    scale = scaler_params['scale']
    scale[scale == 0] = 1.0
    scaled_data = (data_reshaped - scaler_params['mean']) / scale
    return scaled_data.reshape((nsamples, ntimesteps, nfeatures))

# --- 4. API Endpoints ---
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
        
        logger.info(f"Processing prediction for lat={lat}, lon={lon}")
        
        elevation, slope, aspect, _ = fetch_elevation_data(lat, lon)
        static_features = {'elevation': elevation, 'slope': slope, 'aspect': aspect}
        
        weather_data = fetch_weather_data(lat, lon, 10)
        if not weather_data:
            return jsonify({'error': 'Failed to fetch weather data.'}), 500
        
        # Use the corrected feature preparation function
        lstm_input = prepare_lstm_sequence(weather_data, static_features)
        lstm_scaled = scale_input_features(lstm_input, scaler_params)
        
        with torch.no_grad():
            output = model(torch.tensor(lstm_scaled, dtype=torch.float32).to(device))
            probability = torch.sigmoid(output).item()
        
        risk_levels = {0.7: "Very High Risk", 0.5: "High Risk", 0.3: "Moderate Risk"}
        risk_level = next((v for k, v in risk_levels.items() if probability >= k), "Low Risk")
        
        # The rest of the response logic remains the same
        response_data = {
            'probability': round(probability, 4),
            'probability_percent': f"{probability:.1%}",
            'risk_level': risk_level,
            # ... other fields
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'An internal error occurred.'}), 500

# --- 5. Run the App ---
if __name__ == '__main__':
    logger.info("Starting Landslide Prediction Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)