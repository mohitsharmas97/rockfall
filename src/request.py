import csv
import requests
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os
import concurrent.futures

# This function remains unchanged
def process_daily_data_for_lstm(weather_data, event_id, latitude, longitude):
    """
    Takes the raw 10-day weather data from the API and formats it into a list
    of daily records, suitable for an LSTM model. Each record in the list is one day.
    """
    daily = weather_data.get('daily', {})
    hourly = weather_data.get('hourly', {})

    # Data Completeness Checks
    daily_keys = ['time', 'precipitation_sum', 'rain_sum', 'temperature_2m_max', 'temperature_2m_min', 
                  'snowfall_sum', 'et0_fao_evapotranspiration', 'precipitation_hours']
    hourly_keys = ['soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'snow_depth', 'relative_humidity_2m']

    if not all(k in daily for k in daily_keys) or len(daily.get('time', [])) < 10:
        return None
    if not all(k in hourly for k in hourly_keys) or len(hourly.get('time', [])) < 240:
        return None

    daily_records = []
    
    for i in range(10):
        start_hour_index = i * 24
        end_hour_index = start_hour_index + 24
        
        def get_daily_avg(hourly_data_list):
            hourly_slice = hourly_data_list[start_hour_index:end_hour_index]
            valid_readings = [val for val in hourly_slice if val is not None]
            return sum(valid_readings) / len(valid_readings) if valid_readings else 0.0

        avg_soil_moisture_0_7 = get_daily_avg(hourly['soil_moisture_0_to_7cm'])
        avg_soil_moisture_7_28 = get_daily_avg(hourly['soil_moisture_7_to_28cm'])
        avg_snow_depth = get_daily_avg(hourly['snow_depth'])
        avg_humidity = get_daily_avg(hourly['relative_humidity_2m'])

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

# This function remains unchanged
def parse_flexible_date(date_str, formats):
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None

# NEW FUNCTION to handle a single row's fetching and processing
def fetch_single_event(row_data):
    """
    Fetches and processes data for a single event. Designed to be run in a separate thread.
    """
    i, row = row_data  # Unpack the tuple
    event_id = i + 1
    latitude = row.get('latitude')
    longitude = row.get('longitude')
    event_date_str = row.get('event_date')

    if not all([latitude, longitude, event_date_str]):
        return None

    possible_formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M:%S %p', '%Y-%m-%d', '%m-%d-%Y %H:%M']
    event_date_obj = parse_flexible_date(event_date_str, possible_formats)

    if event_date_obj is None:
        print(f"Skipping row {event_id}: Could not parse date '{event_date_str}'")
        return None
    
    end_date_obj = event_date_obj - timedelta(days=1)
    start_date_obj = end_date_obj - timedelta(days=9)
    start_date_api_format = start_date_obj.strftime('%Y-%m-%d')
    end_date_api_format = end_date_obj.strftime('%Y-%m-%d')

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    daily_params = "precipitation_sum,rain_sum,snowfall_sum,temperature_2m_max,temperature_2m_min,et0_fao_evapotranspiration,precipitation_hours"
    hourly_params = "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,snow_depth,relative_humidity_2m"
    
    params = {
        'latitude': latitude, 'longitude': longitude, 'start_date': start_date_api_format,
        'end_date': end_date_api_format, 'daily': daily_params, 'hourly': hourly_params, 'timezone': 'GMT'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        return process_daily_data_for_lstm(weather_data, event_id, latitude, longitude)
    except requests.exceptions.RequestException as e:
        print(f"-> API request failed for event ID {event_id}: {e}")
        return None

# MODIFIED main function to use the ThreadPoolExecutor
def fetch_and_process_concurrently(input_csv_filepath, output_csv_filepath, max_workers=10):
    """
    Reads events, fetches weather data concurrently using a thread pool,
    and saves the results to a new CSV file.
    """
    if os.path.exists(output_csv_filepath):
        os.remove(output_csv_filepath)

    try:
        with open(input_csv_filepath, mode='r', encoding='utf-8') as infile:
            rows = list(csv.DictReader(infile))
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_filepath}' was not found.")
        return

    header_written = False
    
    # Use a ThreadPoolExecutor to manage concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each row. Pass both the index and the row data.
        future_to_row = {executor.submit(fetch_single_event, row_data): row_data for row_data in enumerate(rows)}
        
        # Use tqdm to create a progress bar as futures complete
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(rows), desc="Fetching weather data"):
            try:
                daily_data_list = future.result() # Get the result from the completed future
                if daily_data_list:
                    with open(output_csv_filepath, mode='a', newline='', encoding='utf-8') as outfile:
                        fieldnames = daily_data_list[0].keys()
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        if not header_written:
                            # Simple check to see if the file is empty to write the header
                            outfile.seek(0, 2)
                            if outfile.tell() == 0:
                                writer.writeheader()
                                header_written = True
                        writer.writerows(daily_data_list)
            except Exception as e:
                print(f"An error occurred while processing a result: {e}")

    print(f"\nAll sequential data successfully processed and saved to '{output_csv_filepath}'")

if __name__ == "__main__":
    input_csv = 'combined_landslide_dataset_new.csv' 
    output_csv_lstm = 'data/lstm_daily_sequences_extended.csv'
    
    os.makedirs('data', exist_ok=True)
    
    # Call the new concurrent function
    # You can adjust max_workers based on your network and the API's limits
    fetch_and_process_concurrently(input_csv, output_csv_lstm, max_workers=5)