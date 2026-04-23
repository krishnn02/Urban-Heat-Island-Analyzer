import sys
import os
sys.path.append('src')
from stac_extractor import extract_real_data

# Use a small area for testing
lat, lon = 28.6139, 77.2090 # New Delhi
print("Testing extract_real_data columns...")
df = extract_real_data(lat, lon, radius_km=1, resolution=100, use_cache=False)

if df is not None:
    print(f"Columns found: {df.columns.tolist()}")
    if 'ndwi' in df.columns:
        print("SUCCESS: ndwi column present.")
    else:
        print("FAILURE: ndwi column missing.")
else:
    print("FAILURE: No data returned.")
