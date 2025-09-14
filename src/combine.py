import pandas as pd

# Load all the datasets
try:
    enhanced_landslide_df = pd.read_csv('data/enhanced_landslide_dataset.csv')
    enhanced_landslide_negative_df = pd.read_csv('data\enhanced_landslide_negative_dataset.csv')
    featured_weather_data_df = pd.read_csv('data/featured_weather_data.csv')
    featured_weather_negative_data_df = pd.read_csv('data/featured_weather_negative_data.csv')
    landslides_hotspot_df = pd.read_csv('data/landslides_hotspot_with_elevation(1).csv')
    # Note: negative_samples_generated.csv is not explicitly used as its columns are a subset of 
    # enhanced_landslide_negative.csv
    # negative_samples_df = pd.read_csv('negative_samples_generated.csv')

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure all CSV files are in the same directory as the script.")
    exit()

# --- Process Positive Datasets ---

# Merge the two landslide datasets to include elevation data with the main landslide data
# A left merge is used to keep all records from the enhanced_landslide_df
positive_df = pd.merge(enhanced_landslide_df, landslides_hotspot_df[['event_id', 'elevation']], on='event_id', how='left')

# Merge the result with the weather data for landslide events.
# An inner merge is used to ensure we only have records with complete data across these sets.
positive_df = pd.merge(positive_df, featured_weather_data_df, on=['latitude', 'longitude', 'event_date'], how='inner')

# Add the 'landslide' column and set its value to 1 to indicate a landslide event
positive_df['landslide'] = 1

# --- Process Negative Datasets ---

# Merge the negative (non-landslide) dataset with its corresponding weather data.
negative_df = pd.merge(enhanced_landslide_negative_df, featured_weather_negative_data_df, on=['latitude', 'longitude', 'event_date'], how='inner')

# Add the 'landslide' column and set its value to 0
negative_df['landslide'] = 0

# --- Combine and Finalize ---

# Concatenate the positive and negative dataframes into a single dataframe
final_df = pd.concat([positive_df, negative_df], ignore_index=True)

# Define the list of columns that should be in the final dataset
desired_columns = [
    'latitude',
    'longitude',
    'event_date',
    'elevation',
    'slope',
    'aspect',
    'curvature',
    'precipitation_sum_last_10_days',
    'precipitation_sum_last_7_days',
    'precipitation_sum_last_3_days',
    'temperature_2m_max_in_last_10_days',
    'temperature_2m_min_in_last_10_days',
    'soil_moisture_0_to_7cm_mean_last_5_days',
    'rainfall_trend_last_5_days',
    'days_with_rain_gt_20mm_last_10_days',
    'consecutive_rainy_days',
    'landslide'
]

# Select only the desired columns for the final dataframe
final_df = final_df[desired_columns]

# Save the combined and cleaned dataset to a new CSV file
final_df.to_csv('combined_landslide_dataset_new.csv', index=False)

print("Combined dataset created successfully and saved as 'combined_landslide_dataset.csv'")
print("\nFinal dataset info:")
final_df.info()
print("\nFirst 5 rows of the combined dataset:")
print(final_df.head())