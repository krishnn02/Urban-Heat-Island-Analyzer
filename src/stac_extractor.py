import pandas as pd
import numpy as np
from pystac_client import Client
from odc.stac import stac_load
import logging
from pyproj import Proj
import os
import hashlib
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_bounding_box(lat, lon, radius_km=5):
    """
    Calculate a rough bounding box for a given latitude, longitude, and radius.
    """
    # 1 degree latitude is approx 111 km
    lat_delta = radius_km / 111.0
    # 1 degree longitude is approx 111 * cos(latitude) km
    lon_delta = radius_km / (111.0 * np.cos(np.radians(lat)))
    
    min_lon = lon - lon_delta
    min_lat = lat - lat_delta
    max_lon = lon + lon_delta
    max_lat = lat + lat_delta
    
    return [min_lon, min_lat, max_lon, max_lat]

def get_cache_path(lat, lon, radius_km, resolution, date_start, date_end):
    """
    Generates a unique filename for the data cache based on request parameters.
    """
    # Create a unique key from parameters (Versioned to force re-fetch on logic change)
    key = f"v3_{lat:.4f}_{lon:.4f}_{radius_km}_{resolution}_{date_start}_{date_end}"
    hash_key = hashlib.md5(key.encode()).hexdigest()

    
    cache_dir = "data/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        
    return os.path.join(cache_dir, f"stac_cache_{hash_key}.csv")

def extract_real_data(lat, lon, radius_km=5, date_start='2023-04-01', date_end='2023-05-31', reference_grid=None, resolution=60, use_cache=True):
    """
    Fetches real Sentinel-2 (NDVI) and Landsat-8 (LST) data from AWS STAC.
    Optimized for speed by using a configurable resolution (default 60m).
    Includes local CSV caching to avoid redundant API hits.
    """
    if lat is None or lon is None:
        return None
        
    cache_path = get_cache_path(lat, lon, radius_km, resolution, date_start, date_end)
    
    if use_cache and os.path.exists(cache_path):
        logger.info(f"Loading data from local cache: {cache_path}")
        return pd.read_csv(cache_path)

    try:
        bbox = get_bounding_box(lat, lon, radius_km)
        time_range = f"{date_start}/{date_end}"
        
        # Microsoft Planetary Computer STAC API with Retry Logic and API Key support
        max_retries = 3
        catalog = None
        api_key = os.environ.get("PC_SDK_SUBSCRIPTION_KEY")
        
        for attempt in range(max_retries):
            try:
                if api_key:
                    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", headers={"Ocp-Apim-Subscription-Key": api_key})
                else:
                    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt+1} failed, retrying in 2s...")
                    time.sleep(2)
                else:
                    raise e

        
        import planetary_computer
        
        logger.info(f"Searching STAC for {time_range} at bbox {bbox}")
        
        # 1. Fetch Sentinel-2 for NDVI (Red and NIR bands)
        s2_items = []
        for attempt in range(max_retries):
            try:
                s2_search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=time_range,
                    query={"eo:cloud_cover": {"lt": 20}},
                    max_items=10
                )
                s2_items = list(s2_search.items())
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"S2 Search attempt {attempt+1} failed, retrying...")
                    time.sleep(1)
                else:
                    raise e
        
        if not s2_items:
            logger.warning("No Sentinel-2 items found.")
            return None
            
        logger.info(f"Found {len(s2_items)} Sentinel-2 items.")
        
        # Sign items for Planetary Computer access
        s2_items = [planetary_computer.sign(item) for item in s2_items]
        
        # Load Sentinel-2 data
        # If reference_grid is provided, use it to define the spatial grid
        s2_load_params = {
            "items": s2_items,
            "bands": ["red", "green", "nir"],
            "bbox": bbox,
            "chunks": {"x": 2048, "y": 2048, "time": 1}
        }
        
        if reference_grid is not None:
            s2_load_params["like"] = reference_grid
        else:
            s2_load_params["resolution"] = resolution # Use optimized resolution
            
        s2_data = stac_load(**s2_load_params)
        
        # Calculate median over time (Lazy)
        s2_median = s2_data.median(dim="time")
        
        # Calculate NDVI and NDWI
        red = s2_median["red"].astype(float)
        green = s2_median["green"].astype(float)
        nir = s2_median["nir"].astype(float)
        
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndwi = (green - nir) / (green + nir + 1e-8)
        
        # 2. Fetch Landsat 8/9 for LST (Thermal band) with Retry Logic
        ls_items = []
        for attempt in range(max_retries):
            try:
                ls_search = catalog.search(
                    collections=["landsat-c2-l2"],
                    bbox=bbox,
                    datetime=time_range,
                    query={
                        "eo:cloud_cover": {"lt": 20},
                        "platform": {"in": ["landsat-8", "landsat-9"]}
                    },
                    max_items=5
                )
                ls_items = list(ls_search.items())
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"LS Search attempt {attempt+1} failed, retrying...")
                    time.sleep(1)
                else:
                    raise e
        
        if not ls_items:
            logger.warning("No Landsat items found.")
            return None
            
        logger.info(f"Found {len(ls_items)} Landsat items.")
        
        ls_items = [planetary_computer.sign(item) for item in ls_items]
        
        # Load Landsat thermal data (lwir11 is band 10 in Landsat 8/9 ST)
        ls_data = stac_load(
            ls_items,
            bands=["lwir11"],
            like=s2_median, # This aligns the grids perfectly!
            chunks={"x": 2048, "y": 2048, "time": 1}
        )
        
        ls_median = ls_data.median(dim="time")
        
        # Landsat Collection 2 surface temperature scale factor
        # ST = (DN * 0.00341802 + 149.0) - 273.15 to get Celsius
        # Fill NaN values with a reasonable background if needed, but let's keep them as NaN and drop
        lwir = ls_median["lwir11"].astype(float)
        temperature = (lwir * 0.00341802 + 149.0) - 273.15
        
        # Combine into a single lazy Dataset
        import xarray as xr
        ds = xr.Dataset({
            "ndvi": ndvi,
            "ndwi": ndwi,
            "land_surface_temperature": temperature
        })
        
        # Optimization: Limit pixels before compute
        # If the grid is too large, we downsample
        max_pixels = 1000000 # 1 Million pixels safety budget
        total_pixels = ds.x.size * ds.y.size
        
        if total_pixels > max_pixels:
            factor = int(np.sqrt(total_pixels / max_pixels)) + 1
            logger.info(f"Budget exceeded ({total_pixels} px). Downsampling by factor {factor} for memory safety...")
            ds = ds.coarsen(x=factor, y=factor, boundary="trim").mean()

            
        # Final Compute - This triggers all the dask graph
        logger.info(f"Triggering final compute for {ds.x.size * ds.y.size} pixels...")
        df = ds.to_dataframe().reset_index()
        
        # Reproject x, y (which are usually in a UTM projection based on the STAC item's EPSG)
        # odc-stac assigns a crs to the xarray dataset under the .odc accessor
        crs = None
        try:
            crs = s2_data.odc.geobox.crs
        except AttributeError:
            logger.warning("Could not find CRS via odc.geobox.")
        
        if crs:
            # Simple conversion using pyproj
            in_proj = Proj(crs)
            out_proj = Proj("EPSG:4326") # WGS84 (Lat/Lon)
            # Use always_xy=True to ensure consistency in projection order
            from pyproj import Transformer
            transformer = Transformer.from_proj(in_proj, out_proj, always_xy=True)
            lon_arr, lat_arr = transformer.transform(df['x'].values, df['y'].values)
            df['longitude'] = lon_arr
            df['latitude'] = lat_arr
        else:
            # Fallback if CRS isn't found (unlikely, but safe)
            logger.warning("CRS not found, using raw x/y as lon/lat (might be wrong)")
            df['longitude'] = df['x']
            df['latitude'] = df['y']
            
        # Removed mock features for Phase 1: Honesty.
        # Real landcover and climate data will be integrated in later phases.
        
        # Filter out water bodies (NDWI > 0 usually indicates water)
        # We'll use > 0.0 as a safe threshold
        initial_count = len(df)
        df = df[df['ndwi'] <= 0.0]
        logger.info(f"Filtered out {initial_count - len(df)} water pixels.")
        
        # Clean up columns - Only using real spectral/thermal data
        final_df = df[['latitude', 'longitude', 'ndvi', 'ndwi', 'land_surface_temperature']]
        
        logger.info(f"Successfully extracted raw data with {len(final_df)} points.")
        
        # Save to cache
        try:
            final_df.to_csv(cache_path, index=False)
            logger.info(f"Saved extracted data to cache: {cache_path}")
        except Exception as cache_err:
            logger.warning(f"Failed to save to cache: {cache_err}")
            
        return final_df
        
    except Exception as e:
        logger.error(f"Error fetching STAC data: {e}")
        return None

if __name__ == "__main__":
    # Test script
    lat, lon = 28.6139, 77.2090 # New Delhi
    print("Fetching real data for New Delhi...")
    df = extract_real_data(lat, lon, radius_km=2)
    if df is not None:
        print(f"Success! Fetched {len(df)} points.")
        print(df.head())
    else:
        print("Failed to fetch data.")
