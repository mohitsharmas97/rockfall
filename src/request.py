import csv
import requests
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os

def process_daily_data_for_lstm(weather_data, event_id, latitude, longitude):
    """
    Takes the raw 10-day weather data from the API and formats it into a list
    of daily records, suitable for an LSTM model. Each record in the list is one day.

    Args:
        weather_data (dict): The JSON response from the Open-Meteo API.
        event_id (int): A unique identifier for this event.
        latitude (float): The latitude of the event.
        longitude (float): The longitude of the event.

    Returns:
        list: A list of dictionaries, where each dictionary represents one day's data.
              Returns None if the data is incomplete.
    """
    daily = weather_data.get('daily', {})
    hourly = weather_data.get('hourly', {})

    # --- Data Completeness Checks ---
    daily_keys = ['time', 'precipitation_sum', 'rain_sum', 'temperature_2m_max', 'temperature_2m_min', 
                  'snowfall_sum', 'et0_fao_evapotranspiration', 'precipitation_hours']
    hourly_keys = ['soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'snow_depth', 'relative_humidity_2m']

    if not all(k in daily for k in daily_keys) or len(daily['time']) < 10:
        return None
    if not all(k in hourly for k in hourly_keys) or len(hourly['time']) < 240: # 10 days * 24 hours
        return None

    daily_records = []
    
    # --- Iterate through each of the 10 days of data ---
    for i in range(10):
        # Define the 24-hour slice for the current day
        start_hour_index = i * 24
        end_hour_index = start_hour_index + 24
        
        # --- Helper function to safely average hourly data ---
        def get_daily_avg(hourly_data_list):
            hourly_slice = hourly_data_list[start_hour_index:end_hour_index]
            valid_readings = [val for val in hourly_slice if val is not None]
            return sum(valid_readings) / len(valid_readings) if valid_readings else 0.0

        # --- Calculate daily averages for hourly features ---
        avg_soil_moisture_0_7 = get_daily_avg(hourly['soil_moisture_0_to_7cm'])
        avg_soil_moisture_7_28 = get_daily_avg(hourly['soil_moisture_7_to_28cm'])
        avg_snow_depth = get_daily_avg(hourly['snow_depth'])
        avg_humidity = get_daily_avg(hourly['relative_humidity_2m'])

        # --- Assemble the record for the day ---
        record = {
            'event_id': event_id,
            'day_index': i,
            'date': daily['time'][i],
            'latitude': latitude,
            'longitude': longitude,
            'daily_precip': daily['precipitation_sum'][i],
            'daily_rain': daily['rain_sum'][i],
            'daily_snow': daily['snowfall_sum'][i],
            'precip_hours': daily['precipitation_hours'][i],
            'et0_evapotranspiration': daily['et0_fao_evapotranspiration'][i],
            'max_temp': daily['temperature_2m_max'][i],
            'min_temp': daily['temperature_2m_min'][i],
            'avg_soil_moisture_0_7cm': avg_soil_moisture_0_7,
            'avg_soil_moisture_7_28cm': avg_soil_moisture_7_28,
            'avg_snow_depth': avg_snow_depth,
            'avg_relative_humidity': avg_humidity
        }
        daily_records.append(record)
        
    return daily_records


def parse_flexible_date(date_str, formats):
    """
    Tries to parse a date string using a list of possible formats.
    """
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None

def fetch_and_process_daily_sequences(input_csv_filepath, output_csv_filepath):
    """
    Reads events, fetches 10 days of pre-event daily weather data,
    and saves the resulting sequences to a new CSV file.
    """
    header_written = False
    
    if os.path.exists(output_csv_filepath):
        os.remove(output_csv_filepath)

    try:
        with open(input_csv_filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)
            
            for i, row in enumerate(tqdm(rows, desc="Processing Events for LSTM")):
                event_id = i + 1
                latitude = row.get('latitude')
                longitude = row.get('longitude')
                event_date_str = row.get('event_date')

                if not all([latitude, longitude, event_date_str]):
                    continue

                # --- THIS IS THE MODIFIED LINE ---
                possible_formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M:%S %p', '%Y-%m-%d', '%m-%d-%Y %H:%M']
                # ------------------------------------
                
                event_date_obj = parse_flexible_date(event_date_str, possible_formats)

                if event_date_obj is None:
                    print(f"Skipping row {i+1}: Could not parse date '{event_date_str}'")
                    continue
                
                end_date_obj = event_date_obj - timedelta(days=1)
                start_date_obj = end_date_obj - timedelta(days=9)
                start_date_api_format = start_date_obj.strftime('%Y-%m-%d')
                end_date_api_format = end_date_obj.strftime('%Y-%m-%d')

                base_url = "https://archive-api.open-meteo.com/v1/archive"
                
                # --- Updated API parameters to fetch new features ---
                daily_params = "precipitation_sum,rain_sum,snowfall_sum,temperature_2m_max,temperature_2m_min,et0_fao_evapotranspiration,precipitation_hours"
                hourly_params = "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,snow_depth,relative_humidity_2m"
                
                params = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'start_date': start_date_api_format,
                    'end_date': end_date_api_format,
                    'daily': daily_params,
                    'hourly': hourly_params,
                    'timezone': 'GMT'
                }

                try:
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()
                    weather_data = response.json()

                    daily_data_list = process_daily_data_for_lstm(weather_data, event_id, latitude, longitude)
                    
                    if daily_data_list:
                        with open(output_csv_filepath, mode='a', newline='', encoding='utf-8') as outfile:
                            fieldnames = daily_data_list[0].keys()
                            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                            if not header_written:
                                writer.writeheader()
                                header_written = True
                            writer.writerows(daily_data_list)
                            
                except requests.exceptions.RequestException as e:
                    print(f"-> API request failed for event ID {event_id}: {e}")
                
                time.sleep(0.005)

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_filepath}' was not found.")
        return

    print(f"\nAll sequential data successfully processed and saved to '{output_csv_filepath}'")


if __name__ == "__main__":
    input_csv = 'combined_landslide_dataset_new.csv' 
    output_csv_lstm = 'data/lstm_daily_sequences_extended.csv'
    
    os.makedirs('data', exist_ok=True)
    
    fetch_and_process_daily_sequences(input_csv, output_csv_lstm)