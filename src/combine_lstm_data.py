import pandas as pd
import numpy as np

def combine_and_prepare_lstm_data(time_series_csv, static_data_csv):
    """
    Loads time-series weather data and static topographical/label data,
    merges them, and reshapes the result into NumPy arrays suitable for an LSTM.

    Args:
        time_series_csv (str): Filepath for the CSV with daily sequential weather data.
        static_data_csv (str): Filepath for the original CSV with lat, lon, elevation, slope, etc., and the 'landslide' label.

    Returns:
        tuple: A tuple containing (X, y) where:
               X is a 3D NumPy array of features (samples, timesteps, features).
               y is a 1D NumPy array of labels.
    """
    try:
        # Step 1: Load the datasets
        print("Loading datasets...")
        time_series_df = pd.read_csv(time_series_csv)
        static_df = pd.read_csv(static_data_csv)
        print("Datasets loaded successfully.")

        # --- Data Validation ---
        if 'event_id' not in time_series_df.columns:
            raise ValueError("Time series CSV must contain an 'event_id' column.")
        if 'landslide' not in static_df.columns:
            raise ValueError("Static data CSV must contain the 'landslide' label column.")

        # Step 2: Prepare the static data by adding a matching event_id
        # The event_id in the time_series data was created sequentially from 1.
        # We replicate that here to create a key for merging.
        static_df['event_id'] = range(1, len(static_df) + 1)
        
        # Select only the necessary static columns for the merge
        static_subset_df = static_df[['event_id', 'elevation', 'slope', 'curvature', 'landslide']]
        print("Prepared static data with event IDs.")

        # Step 3: Merge the two dataframes on 'event_id'
        # This will add the static features and the label to every row of the time-series data.
        print("Merging time-series and static data...")
        merged_df = pd.merge(time_series_df, static_subset_df, on='event_id', how='left')
        
        # Drop rows where a merge was not possible or data is incomplete
        merged_df.dropna(inplace=True)
        print("Merge complete.")

        # Step 4: Reshape the data into sequences for the LSTM
        print("Reshaping data into LSTM format (samples, timesteps, features)...")
        
        # Define which columns are features
        feature_columns = [
            'daily_precip', 'daily_rain', 'daily_snow', 'precip_hours',
            'et0_evapotranspiration', 'max_temp', 'min_temp',
            'avg_soil_moisture_0_7cm', 'avg_soil_moisture_7_28cm',
            'avg_snow_depth', 'avg_relative_humidity', 'elevation', 'slope', 'curvature'
        ]

        sequences = []
        labels = []
        
        # Group by event_id and process each group
        for event_id, group in merged_df.groupby('event_id'):
            # Ensure each event has exactly 10 days of data
            if len(group) == 10:
                # Extract the features for the 10 timesteps
                sequence_features = group[feature_columns].values
                sequences.append(sequence_features)
                
                # The label is the same for all 10 days, so take it from the first row
                labels.append(group['landslide'].iloc[0])

        # Convert the lists to NumPy arrays
        X = np.array(sequences)
        y = np.array(labels)

        print("\n--- Data Preparation Complete ---")
        print(f"Final shape of feature array X: {X.shape}")
        print(f"Final shape of label array y: {y.shape}")
        
        return X, y

    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return None, None

if __name__ == '__main__':
    # Define the file paths for your input datasets
    TIME_SERIES_FILE = 'data/lstm_daily_sequences_extended.csv'
    STATIC_DATA_FILE = 'combined_landslide_dataset_new.csv' # Your original dataset with labels

    # Run the preparation function
    X_lstm, y_lstm = combine_and_prepare_lstm_data(TIME_SERIES_FILE, STATIC_DATA_FILE)

    if X_lstm is not None and y_lstm is not None:
        # --- Save the final arrays for later use ---
        # This is recommended so you don't have to re-run this script every time.
        np.save('data/X_lstm_ready.npy', X_lstm)
        np.save('data/y_lstm_ready.npy', y_lstm)
        print("\nPrepared NumPy arrays have been saved to 'data/X_lstm_ready.npy' and 'data/y_lstm_ready.npy'")
