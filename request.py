import csv
import requests
from datetime import datetime, timedelta
import time
from tqdm import tqdm # Import tqdm

def calculate_slope(y_values):
    """Calculates the slope of a line for a list of y-values over x = [0, 1, 2...]."""
    n = len(y_values)
    if n < 2:
        return 0.0

    x = list(range(n))
    sum_x = sum(x)
    sum_y = sum(y_values)
    sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y_values))
    sum_x2 = sum(x_i**2 for x_i in x)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x**2

    return numerator / denominator if denominator != 0 else 0.0

def engineer_features(weather_data, latitude, longitude, event_date_str):
    """
    Takes 10 days of weather data and engineers features for a predictive model.
    Returns a single dictionary representing one event.
    """
    daily = weather_data.get('daily', {})
    hourly = weather_data.get('hourly', {})

    if not daily or not daily.get('time') or len(daily['time']) < 10:
        # print(f"--> Warning: Incomplete daily data for event on {event_date_str}. Skipping feature engineering.") # Removed print
        return None

    # --- Extract daily and hourly lists ---
    daily_precip = daily.get('precipitation_sum', [])
    daily_rain = daily.get('rain_sum', [])
    daily_temp_max = daily.get('temperature_2m_max', [])
    daily_temp_min = daily.get('temperature_2m_min', [])
    hourly_soil_moisture_0_7 = hourly.get('soil_moisture_0_to_7cm', [])

    # --- 1. Aggregations (Sum, Mean, Max) ---
    features = {
        'latitude': latitude,
        'longitude': longitude,
        'event_date': event_date_str,
        'precipitation_sum_last_10_days': sum(daily_precip[-10:]),
        'precipitation_sum_last_7_days': sum(daily_precip[-7:]),
        'precipitation_sum_last_3_days': sum(daily_precip[-3:]),
        'temperature_2m_max_in_last_10_days': max(daily_temp_max[-10:]),
        'temperature_2m_min_in_last_10_days': min(daily_temp_min[-10:]),
    }

    # Soil moisture requires averaging hourly data for the last 5 days (120 hours)
    last_120_hours_sm = [sm for sm in hourly_soil_moisture_0_7[-120:] if sm is not None]
    features['soil_moisture_0_to_7cm_mean_last_5_days'] = sum(last_120_hours_sm) / len(last_120_hours_sm) if last_120_hours_sm else 0.0

    # --- 2. Trend Features ---
    features['rainfall_trend_last_5_days'] = calculate_slope(daily_rain[-5:])

    # --- 3. Threshold-Based Features ---
    features['days_with_rain_gt_20mm_last_10_days'] = sum(1 for p in daily_precip[-10:] if p > 20)

    # Calculate max consecutive rainy days (precipitation > 0.1 mm)
    max_consecutive = 0
    current_streak = 0
    for p in daily_precip[-10:]:
        if p > 0.1:
            current_streak += 1
        else:
            max_consecutive = max(max_consecutive, current_streak)
            current_streak = 0
    features['consecutive_rainy_days'] = max(max_consecutive, current_streak)

    return features

def process_events_and_fetch_weather(input_csv_filepath, output_csv_filepath):
    """
    Reads events, fetches pre-event weather data, engineers features,
    and saves the results to a new CSV file.
    """
    featured_data = []
    try:
        with open(input_csv_filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            rows = list(reader) # Read all rows into a list to get the total count

            for row in tqdm(rows, desc="Processing Events"): # Wrap the loop with tqdm
                latitude = row.get('latitude')
                longitude = row.get('longitude')
                event_date_str = row.get('event_date')

                if not all([latitude, longitude, event_date_str]):
                    # print(f"Skipping row due to missing data: {row}") # Removed print
                    continue

                # print(f"Processing event at Lat: {latitude}, Lon: {longitude} on {event_date_str}") # Removed print

                # 1. Adjust date range to be the 10 days *before* the event
                try:
                    event_date_obj = datetime.strptime(event_date_str, '%m/%d/%Y %I:%M:%S %p')
                    end_date_obj = event_date_obj - timedelta(days=1)
                    start_date_obj = end_date_obj - timedelta(days=9) # 9 days before end_date = 10 total days
                    start_date_api_format = start_date_obj.strftime('%Y-%m-%d')
                    end_date_api_format = end_date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    # print(f"Skipping row due to invalid date format: {event_date_str}") # Removed print
                    continue

                # 2. Construct API URL with all necessary params
                base_url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'start_date': start_date_api_format,
                    'end_date': end_date_api_format,
                    'daily': 'precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min',
                    'hourly': 'soil_moisture_0_to_7cm',
                    'timezone': 'GMT'
                }

                try:
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()
                    weather_data = response.json()

                    # 3. Perform feature engineering
                    event_features = engineer_features(weather_data, latitude, longitude, event_date_str)
                    if event_features:
                        featured_data.append(event_features)
                        # print(f"-> Successfully engineered features for event on {event_date_str}") # Removed print

                except requests.exceptions.RequestException as e:
                    print(f"-> API request failed for event on {event_date_str}: {e}")

                time.sleep(1)

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_filepath}' was not found.")
        return

    # 4. Write the new featured data to the output CSV
    if not featured_data:
        print("No data was processed. Output file will not be created.")
        return

    try:
        fieldnames = list(featured_data[0].keys())
        with open(output_csv_filepath, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(featured_data)

        print(f"\nAll data successfully processed and saved to '{output_csv_filepath}'")

    except IOError as e:
        print(f"Error writing to file '{output_csv_filepath}': {e}")


if __name__ == "__main__":
    input_csv = 'landslides_hotspot_with_elevation(1).csv'
    output_csv = 'featured_weather_data.csv'
    process_events_and_fetch_weather(input_csv, output_csv)