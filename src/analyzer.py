import pandas as pd
import numpy as np
from scipy import stats
def calculate_correlations(df: pd.DataFrame):
    """
    Calculates Pearson and Spearman correlations between NDVI and Temperature.
    Returns a dictionary of stats.
    """
    if 'ndvi' not in df.columns or 'land_surface_temperature' not in df.columns:
        return None
        
    try:
        if len(df) < 2:
            return None
            
        # Pearson Correlation (Linear relationship)
        pearson_coef, p_value = stats.pearsonr(df['ndvi'], df['land_surface_temperature'])
        
        # Spearman Correlation (Monotonic relationship)
        spearman_coef, _ = stats.spearmanr(df['ndvi'], df['land_surface_temperature'])
        
        # Linear Regression for the trendline
        slope, intercept, r_value, p_value_reg, std_err = stats.linregress(df['ndvi'], df['land_surface_temperature'])
        
        return {
            'pearson': pearson_coef,
            'spearman': spearman_coef,
            'p_value': p_value,
            'r_squared': r_value**2,
            'slope': slope,
            'intercept': intercept,
            'equation': f"Temp = {slope:.2f} * NDVI + {intercept:.2f}"
        }
    except Exception:
        return None

def calculate_water_influence(df: pd.DataFrame):
    """
    Analyzes the cooling effect of blue spaces (water) using NDWI.
    """
    if 'ndwi' not in df.columns or 'land_surface_temperature' not in df.columns:
        return None
        
    # Correlation between NDWI and Temp
    # Note: Higher NDWI (water) should strongly correlate with lower Temp
    pearson_coef, _ = stats.pearsonr(df['ndwi'], df['land_surface_temperature'])
    
    # Cooling near water bodies (NDWI > 0)
    avg_temp_water_adjacent = df[df['ndwi'] > 0]['land_surface_temperature'].mean()
    avg_temp_dry = df[df['ndwi'] <= 0]['land_surface_temperature'].mean()
    
    return {
        'water_correlation': pearson_coef,
        'water_cooling_impact': avg_temp_dry - avg_temp_water_adjacent if not pd.isna(avg_temp_water_adjacent) else 0
    }

def get_spatial_insights(df: pd.DataFrame):
    """
    Identifies Heat Hotspots and Cool Havens using quantile analysis.
    """
    if df is None or df.empty:
        return None
        
    # Identify Hotspots: Top 10% Temperature AND Bottom 10% NDVI
    temp_threshold = df['land_surface_temperature'].quantile(0.9)
    ndvi_threshold = df['ndvi'].quantile(0.1)
    
    hotspots = df[(df['land_surface_temperature'] >= temp_threshold) & (df['ndvi'] <= ndvi_threshold)]
    
    # Identify Cool Havens: Bottom 10% Temperature AND Top 10% NDVI
    cool_threshold = df['land_surface_temperature'].quantile(0.1)
    ndvi_high_threshold = df['ndvi'].quantile(0.9)
    
    havens = df[(df['land_surface_temperature'] <= cool_threshold) & (df['ndvi'] >= ndvi_high_threshold)]
    
    # Calculate the "Cooling Dividend"
    # How much cooler are the greenest areas compared to the least green?
    avg_temp_green = df[df['ndvi'] > df['ndvi'].quantile(0.8)]['land_surface_temperature'].mean()
    avg_temp_low_ndvi = df[df['ndvi'] < df['ndvi'].quantile(0.2)]['land_surface_temperature'].mean()
    
    cooling_dividend = avg_temp_low_ndvi - avg_temp_green
    
    # Blue Space Impact (if ndwi exists)
    blue_cooling = 0
    if 'ndwi' in df.columns:
        avg_temp_blue = df[df['ndwi'] > 0.1]['land_surface_temperature'].mean()
        if not pd.isna(avg_temp_blue):
            blue_cooling = avg_temp_low_ndvi - avg_temp_blue
            
    return {
        'num_hotspots': len(hotspots),
        'num_havens': len(havens),
        'avg_temp_green': avg_temp_green,
        'avg_temp_low_ndvi': avg_temp_low_ndvi,
        'cooling_dividend': cooling_dividend,
        'blue_cooling_index': blue_cooling,

        'hotspot_df': hotspots.head(10), # Return a sample
        'haven_df': havens.head(10)
    }

def get_binned_analysis(df: pd.DataFrame, bins=5):
    """
    Groups data into NDVI bins and calculates mean temperature for each.
    This is the most visually intuitive way to prove the relationship.
    """
    df_bins = df.copy()
    df_bins['ndvi_bin'] = pd.qcut(df_bins['ndvi'], q=bins, labels=[f"Q{i+1}" for i in range(bins)])
    
    binned_stats = df_bins.groupby('ndvi_bin', observed=True).agg({
        'land_surface_temperature': 'mean',
        'ndvi': 'mean'
    }).reset_index()
    
    return binned_stats

if __name__ == "__main__":
    # Test logic
    test_df = pd.DataFrame({
        'ndvi': np.linspace(0, 1, 100),
        'land_surface_temperature': 40 - (10 * np.linspace(0, 1, 100)) + np.random.normal(0, 1, 100)
    })
    
    c = calculate_correlations(test_df)
    print("Correlations:", c)
    
    s = get_spatial_insights(test_df)
    print("\nCooling Dividend:", s['cooling_dividend'])
    
    b = get_binned_analysis(test_df)
    print("\nBinned Analysis:")
    print(b)
