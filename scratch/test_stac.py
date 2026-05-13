
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath('.'))

try:
    from src.stac_extractor import extract_real_data
    import pandas as pd
    
    lat, lon = 28.6139, 77.2090 # New Delhi
    print("Testing STAC extraction for New Delhi...")
    df = extract_real_data(lat, lon, radius_km=2, use_cache=False)
    if df is not None:
        print(f"Success! Fetched {len(df)} points.")
        print(df.head())
    else:
        print("Failed to fetch data (returned None).")
except Exception as e:
    print(f"Error caught: {e}")
    import traceback
    traceback.print_exc()
