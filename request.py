import requests

def get_elevation(lat, lon, dataset="srtm90m"):
    """
    Fetches elevation from the Open Topo Data API for a given lat/lon.
    
    Args:
        lat (float): Latitude in WGS-84 format.
        lon (float): Longitude in WGS-84 format.
        dataset (str): The dataset to query, e.g., "srtm90m".

    Returns:
        float: The elevation in meters, or None if an error occurs.
    """
    # The API endpoint structure is /v1/<dataset_name>
    url = f"https://api.opentopodata.org/v1/{dataset}"
    
    # The 'locations' parameter is required.
    # The format is "latitude,longitude".
    params = {
        "locations": f"{lat},{lon}"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # As per the docs, check for "OK" status.
        if data.get('status') == 'OK' and data.get('results'):
            # The elevation is in the first item of the 'results' list.
            return data['results'][0]['elevation']
        else:
            print(f"API returned an error: {data.get('error', 'Unknown error')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"A network error occurred: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing the API response: {e}")
        print("Received data:", response.text)
        return None

# --- Example Usage ---
# Use one of the coordinates from the documentation's example
latitude = 32.5625
longitude = 107.45

elevation = get_elevation(latitude, longitude)

if elevation is not None:
    print(f"✅ Success! The elevation at (Lat: {latitude}, Lon: {longitude}) is {elevation} meters.")