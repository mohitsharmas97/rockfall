import csv
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import concurrent.futures
import time # <--- Import the time module

# This function remains unchanged
def process_daily_data_for_lstm(weather_data, event_id, latitude, longitude):
    """
    Takes the raw 10-day weather data from the API and formats it into a list
    of daily records, suitable for an LSTM model. Each record in the list is one day.
    """
    daily = weather_data.get('daily', {})
    hourly = weather_data.get('hourly', {})
    
    # Use the precise coordinates returned by the API
    api_latitude = weather_data.get('latitude', latitude)
    api_longitude = weather_data.get('longitude', longitude)

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
            'event_id': event_id, 'day_index': i, 'date': daily['time'][i],
            # **FIX**: Use the API-provided coordinates for consistency
            'latitude': api_latitude, 'longitude': api_longitude,
            'daily_precip': daily['precipitation_sum'][i], 'daily_rain': daily['rain_sum'][i],
            'daily_snow': daily['snowfall_sum'][i], 'precip_hours': daily['precipitation_hours'][i],
            'et0_evapotranspiration': daily['et0_fao_evapotranspiration'][i],
            'max_temp': daily['temperature_2m_max'][i], 'min_temp': daily['temperature_2m_min'][i],
            'avg_soil_moisture_0_7cm': avg_soil_moisture_0_7, 'avg_soil_moisture_7_28cm': avg_soil_moisture_7_28,
            'avg_snow_depth': avg_snow_depth, 'avg_relative_humidity': avg_humidity
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

# **FIXED** function to handle API errors (like 429) and use API coordinates
def fetch_single_event(row_data):
    """
    Fetches and processes data for a single event, with retries for API rate limiting.
    """
    i, row = row_data
    event_id = i + 1
    latitude = row.get('latitude')
    longitude = row.get('longitude')
    event_date_str = row.get('event_date')

    if not all([latitude, longitude, event_date_str]):
        return None

    possible_formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M:%S %p', '%Y-%m-%d', '%m-%d-%Y %H:%M']
    event_date_obj = parse_flexible_date(event_date_str, possible_formats)

    if event_date_obj is None:
        # This print is optional if it's too noisy
        # print(f"Skipping row {event_id}: Could not parse date '{event_date_str}'")
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

    # **FIX**: Add retry logic for handling 429 errors
    max_retries = 5
    backoff_factor = 2  # Start with a 2-second delay
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params)
            
            # If we get a 429, wait and try again
            if response.status_code == 429:
                retry_after = attempt * backoff_factor + backoff_factor
                print(f"-> Rate limit hit for event {event_id}. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
                continue # Go to the next attempt

            response.raise_for_status() # Raise an exception for other bad status codes (4xx or 5xx)
            weather_data = response.json()
            
            # **FIX**: Pass the entire weather_data object to the processor
            # This ensures the API-corrected coordinates are used
            return process_daily_data_for_lstm(weather_data, event_id, latitude, longitude)

        except requests.exceptions.RequestException as e:
            print(f"-> API request failed for event {event_id} on attempt {attempt + 1}: {e}")
            if attempt >= max_retries - 1:
                return None # Failed all retries
            time.sleep(backoff_factor) # Wait before the next retry

    return None # Should not be reached if loop completes, but good practice

# This main processing function is now correct and doesn't need changes
def fetch_and_process_concurrently(input_csv_filepath, output_csv_filepath, max_workers=10):
    """
    Reads events, checks for already processed coordinates in the output file,
    fetches weather data for new events concurrently, and appends the results.
    """
    processed_coords = set()
    COORD_PRECISION = 4

    try:
        with open(output_csv_filepath, mode='r', encoding='utf-8') as outfile:
            reader = csv.DictReader(outfile)
            for row in reader:
                try:
                    lat_str = f"{float(row['latitude']):.{COORD_PRECISION}f}"
                    lon_str = f"{float(row['longitude']):.{COORD_PRECISION}f}"
                    processed_coords.add((lat_str, lon_str))
                except (ValueError, KeyError):
                    continue
        if processed_coords:
            print(f"Found {len(processed_coords)} existing unique coordinates in '{output_csv_filepath}'. These will be skipped.")
    except FileNotFoundError:
        print(f"Output file '{output_csv_filepath}' not found. Will create a new one.")
        
    try:
        with open(input_csv_filepath, mode='r', encoding='utf-8') as infile:
            all_rows = list(csv.DictReader(infile))
    except FileNotFoundError:
        print(f"Error: The input file '{input_csv_filepath}' was not found.")
        return

    rows_to_process = []
    for i, row in enumerate(all_rows):
        try:
            lat_str = f"{float(row.get('latitude')):.{COORD_PRECISION}f}"
            lon_str = f"{float(row.get('longitude')):.{COORD_PRECISION}f}"
            if (lat_str, lon_str) not in processed_coords:
                rows_to_process.append((i, row))
        except (ValueError, TypeError):
            continue

    if not rows_to_process:
        print("No new events to process. All coordinates are already in the output file.")
        return
        
    print(f"Total events in input: {len(all_rows)}. New events to fetch: {len(rows_to_process)}.")

    header_written = os.path.exists(output_csv_filepath) and os.path.getsize(output_csv_filepath) > 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(fetch_single_event, row_data): row_data for row_data in rows_to_process}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(rows_to_process), desc="Fetching new data"):
            try:
                daily_data_list = future.result()
                if daily_data_list:
                    with open(output_csv_filepath, mode='a', newline='', encoding='utf-8') as outfile:
                        writer = csv.DictWriter(outfile, fieldnames=daily_data_list[0].keys())
                        if not header_written:
                            writer.writeheader()
                            header_written = True
                        writer.writerows(daily_data_list)
            except Exception as e:
                print(f"An error occurred while writing a result: {e}")

    print(f"\nProcessing complete. Data is saved in '{output_csv_filepath}'")

if __name__ == "__main__":
    input_csv = 'combined_landslide_dataset_new.csv' 
    output_csv_lstm = 'data/lstm_daily_sequences_extended.csv'
    
    os.makedirs('data', exist_ok=True)
    
    fetch_and_process_concurrently(input_csv, output_csv_lstm, max_workers=5)