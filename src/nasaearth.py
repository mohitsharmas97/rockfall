import pandas as pd
import earthaccess
import os
import zipfile
import glob
import math
import rasterio
from tqdm import tqdm

print("✅ Setup complete.")

# ===================================================================
# PART 1: LOAD AND FILTER THE LANDSLIDE CATALOG
# ===================================================================
print("\n--- Part 1: Loading and Filtering Landslide Data ---")
try:
    df = pd.read_csv('landslides_hotspot_with_elevation(1).csv')
    # Define the optimal bounding box for the primary hotspot (Himalayas)
    # Format: (Lon Min, Lat Min, Lon Max, Lat Max)
    optimal_bbox = (60.0, -2.0, 98.0, 39.0)

    # Filter the DataFrame to get only the landslides within this optimal box
    df_hotspot = df[
        (df['longitude'] >= optimal_bbox[0]) & (df['longitude'] <= optimal_bbox[2]) &
        (df['latitude'] >= optimal_bbox[1]) & (df['latitude'] <= optimal_bbox[3])
    ].copy()

    percentage = (len(df_hotspot) / len(df)) * 100
    print(f"Loaded {len(df)} total landslides.")
    print(f"Filtered to {len(df_hotspot)} landslides in the optimal boundary ({percentage:.2f}% of total).")

except FileNotFoundError:
    print("❌ ERROR: 'Global_Landslide_Catalog_Export_rows.csv' not found. Please upload the file.")
    df_hotspot = pd.DataFrame() # Create empty dataframe to prevent further errors

# ===================================================================
# PART 2: DOWNLOAD ELEVATION DATA FOR THE BOUNDARY
# ===================================================================
if not df_hotspot.empty:
    print("\n--- Part 2: Downloading Elevation Data ---")
    auth = earthaccess.login(strategy="interactive", persist=True)

    if auth.authenticated:
        print("Searching for elevation data for the hotspot...")
        granules = earthaccess.search_data(
            short_name="NASADEM_HGT",
            version="001",
            bounding_box=optimal_bbox,
            count=-1
        )

        if granules:
            print(f"Found {len(granules)} elevation tiles to download.")
            output_dir = "nasadem_data"
            os.makedirs(output_dir, exist_ok=True)
            files = earthaccess.download(granules, local_path=output_dir)
        else:
            print("No elevation data found for this area.")
            output_dir = None
    else:
        output_dir = None
else:
    output_dir = None

# ===================================================================
# PART 3: UNZIP DOWNLOADED FILES
# ===================================================================
if output_dir and os.path.exists(output_dir):
    print("\n--- Part 3: Unzipping Elevation Files ---")
    zip_files = glob.glob(os.path.join(output_dir, '*.zip'))

    for file_path in tqdm(zip_files, desc="Unzipping files"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(file_path) # Clean up zip to save space

    print("Unzipping complete.")
else:
    print("\nSkipping Part 3: No data was downloaded.")

# ===================================================================
# PART 4: EXTRACT ELEVATION FOR EACH LANDSLIDE
# # ===================================================================
# if output_dir and os.path.exists(output_dir):
#     print("\n--- Part 4: Extracting Elevation from Tiles ---")
#     raster_cache = {}

#     def get_elevation_from_tile(lat, lon, data_dir):
#         if pd.isna(lat) or pd.isna(lon): return None

#         lat_hemisphere = 'n' if lat >= 0 else 's'
#         lon_hemisphere = 'e' if lon >= 0 else 'w'
#         lat_int, lon_int = math.floor(abs(lat)), math.floor(abs(lon))

#         tile_filename = f"{lat_hemisphere}{lat_int:02d}{lon_hemisphere}{lon_int:03d}.hgt"
#         tile_path = os.path.join(data_dir, tile_filename)

#         if not os.path.exists(tile_path): return None

#         try:
#             if tile_path not in raster_cache:
#                 raster_cache[tile_path] = rasterio.open(tile_path)
#             dem_file = raster_cache[tile_path]
#             value = next(dem_file.sample([(lon, lat)]))[0]
#             return value if value > -1000 else None
#         except Exception:
#             return None

#     # Apply the function to get elevations
#     tqdm.pandas(desc="Getting Elevations")
#     df_hotspot['elevation'] = df_hotspot.progress_apply(
#         lambda row: get_elevation_from_tile(row['latitude'], row['longitude'], output_dir),
#         axis=1
#     )

#     for file in raster_cache.values(): file.close()

#     print("Elevation extraction complete.")

#     # ===================================================================
#     # PART 5: DISPLAY AND SAVE FINAL RESULTS
#     # ===================================================================
#     print("\n--- Part 5: Final Results ---")
#     # Display the first few rows with the new elevation data
#     print("Sample of the final data:")
#     print(df_hotspot[['event_title', 'latitude', 'longitude', 'elevation']].head())

#     # Save your enriched data to a new CSV file
#     output_csv_path = 'landslides_hotspot_with_elevation.csv'
#     df_hotspot.to_csv(output_csv_path, index=False)

#     print(f"\n✅ All steps complete! Final data saved to '{output_csv_path}'")
# else:
#     print("\nSkipping Parts 4 & 5: Process could not be completed.")