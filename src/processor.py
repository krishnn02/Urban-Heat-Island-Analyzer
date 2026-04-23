import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clamps NDVI values to [-1, 1] and adds a min-max normalized column [0, 1].
    """
    if 'ndvi' not in df.columns:
        return df
        
    # Clamp to physical limits
    df['ndvi'] = df['ndvi'].clip(-1.0, 1.0)
    
    # Feature scaling for models (0 to 1 range)
    # NDVI typically ranges from -0.2 to 0.9 in most areas
    # but we'll use the full physical range for normalization
    df['ndvi_normalized'] = (df['ndvi'] + 1.0) / 2.0
    
    logger.info("NDVI normalization complete.")
    return df

def clean_temperature(df: pd.DataFrame, min_temp: float = 5.0, max_temp: float = 65.0) -> pd.DataFrame:
    """
    Removes physically impossible temperature values and handles outliers.
    """
    if 'land_surface_temperature' not in df.columns:
        return df
        
    initial_count = len(df)
    
    # Filter by absolute range
    df = df[(df['land_surface_temperature'] >= min_temp) & (df['land_surface_temperature'] <= max_temp)]
    
    # Statistical outlier removal (Z-score > 3)
    mean_temp = df['land_surface_temperature'].mean()
    std_temp = df['land_surface_temperature'].std()
    
    if std_temp > 0:
        df = df[np.abs(df['land_surface_temperature'] - mean_temp) <= (3 * std_temp)]
    
    logger.info(f"Temperature cleaning complete. Removed {initial_count - len(df)} outliers.")
    return df

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes NaNs, Infs, and ensures data types are correct.
    """
    initial_count = len(df)
    
    # Replace inf with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with any nan in critical columns
    critical_cols = ['ndvi', 'ndwi', 'land_surface_temperature', 'latitude', 'longitude']
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])

    
    logger.info(f"Missing data handling complete. Removed {initial_count - len(df)} invalid rows.")
    return df

def calculate_temporal_change(df_baseline: pd.DataFrame, df_current: pd.DataFrame) -> pd.DataFrame:
    """
    Compares two dataframes from different years.
    Returns a dataframe with Delta NDVI and Delta Temperature.
    """
    if df_baseline is None or df_current is None:
        return None
        
    # Merge on coordinates
    # Using a small tolerance for lat/lon if they are not perfectly identical
    # But if they were aligned correctly in extraction, they should match
    df_baseline = df_baseline.rename(columns={
        'ndvi': 'ndvi_baseline',
        'land_surface_temperature': 'temp_baseline'
    })
    
    # We only need the core columns for comparison
    df_base_sub = df_baseline[['latitude', 'longitude', 'ndvi_baseline', 'temp_baseline']]
    
    # Merge
    merged = pd.merge(df_current, df_base_sub, on=['latitude', 'longitude'], how='inner')
    
    if merged.empty:
        logger.warning("Temporal merge resulted in empty dataframe. Spatial alignment may have failed.")
        return None
        
    # Calculate Deltas
    merged['delta_ndvi'] = merged['ndvi'] - merged['ndvi_baseline']
    merged['delta_temp'] = merged['land_surface_temperature'] - merged['temp_baseline']
    
    # Identify "Urbanization Hotspots" (NDVI decrease AND Temp increase)
    merged['is_urban_expansion'] = (merged['delta_ndvi'] < -0.1) & (merged['delta_temp'] > 1.0)
    
    logger.info(f"Temporal change calculation complete. Merged {len(merged)} points.")
    return merged

def process_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to clean and normalize data for the ML pipeline.
    """
    if df is None or df.empty:
        logger.warning("Empty dataframe passed to processor.")
        return df
        
    df = handle_missing_data(df)
    df = normalize_ndvi(df)
    df = clean_temperature(df)
    
    # urban_density clipping removed for Phase 4

        
    logger.info(f"Final processed dataframe has {len(df)} records.")
    return df

if __name__ == "__main__":
    # Test logic
    test_data = pd.DataFrame({
        'ndvi': [0.5, 1.2, -0.3, np.nan, 0.8],
        'land_surface_temperature': [30.0, 100.0, 25.0, 32.0, -10.0],
        'latitude': [28.6, 28.6, 28.6, 28.6, 28.6],
        'longitude': [77.2, 77.2, 77.2, 77.2, 77.2]
    })
    
    print("Original Test Data:")
    print(test_data)
    
    processed = process_for_modeling(test_data)
    print("\nProcessed Data:")
    print(processed)
