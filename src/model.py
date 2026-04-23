import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans

def train_predictive_model(df: pd.DataFrame):
    """
    Trains a Random Forest Regressor to predict LST based on NDVI and Location.
    Returns the trained model and test metrics.
    """

    if df is None or df.empty:
        return None
        
    # Core features for spatial heat modeling (Real Satellite Data Only)
    # Added ndwi for improved performance (water cooling effect)
    features = ['ndvi', 'ndwi', 'latitude', 'longitude']
    
    # Check if all features exist
    if not all(col in df.columns for col in features):
        return None
        
    X = df[features]
    y = df['land_surface_temperature']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optimized Random Forest: 
    # Increased estimators and optimized depth for production reliability
    rf_model = RandomForestRegressor(
        n_estimators=300, 
        max_depth=None, 
        min_samples_leaf=10, 
        max_features='sqrt', 
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Robust Evaluation with Cross-Validation
    cv_scores = cross_val_score(rf_model, X, y, cv=3)
    r2_cv = cv_scores.mean()
    
    # Evaluate
    predictions = rf_model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    return {
        'model': rf_model,
        'features': features,
        'r2_score': r2,
        'r2_cv': r2_cv,
        'mae': mae,
        'rmse': rmse,
        'feature_importances': dict(zip(features, rf_model.feature_importances_))
    }

def calculate_heat_risk_score(df: pd.DataFrame, config: dict):
    """
    Calculates a heat risk score and categorizes zones into Low, Medium, High.
    """
    # ... (keeping clustering logic as is, it's effective for risk zones)
    features_for_clustering = df[['ndvi', 'land_surface_temperature']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_label'] = kmeans.fit_predict(features_for_clustering)
    
    centers = kmeans.cluster_centers_
    sorted_idx = np.argsort(centers[:, 1]) 
    
    label_mapping = {
        sorted_idx[0]: 'Low',
        sorted_idx[1]: 'Medium',
        sorted_idx[2]: 'High'
    }
    
    df['heat_risk_zone'] = df['cluster_label'].map(label_mapping)
    df = df.drop(columns=['cluster_label'])
    
    # Normalized continuous risk score
    t_min, t_max = df['land_surface_temperature'].min(), df['land_surface_temperature'].max()
    n_min, n_max = df['ndvi'].min(), df['ndvi'].max()
    
    temp_norm = (df['land_surface_temperature'] - t_min) / (t_max - t_min + 1e-8)
    ndvi_inv_norm = 1.0 - ((df['ndvi'] - n_min) / (n_max - n_min + 1e-8))
    
    df['heat_risk_score'] = (temp_norm * 0.6 + ndvi_inv_norm * 0.4) * 100
    return df

def simulate_scenarios(df: pd.DataFrame, stats: dict, ndvi_increase=0.0, density_decrease=0.0):
    """
    Simulates the impact of changing NDVI and Urban Density on LST.
    Returns the simulated dataframe and impact metrics.
    """
    if df is None or df.empty or stats is None:
        return None, None
        
    sim_df = df.copy()
    
    # Input for simulation
    sim_df['sim_ndvi'] = np.clip(sim_df['ndvi'] + ndvi_increase, -0.1, 0.9)
    # urban_density and humidity removed for Phase 1
    
    # Predict using the model
    # Features must match: ['ndvi', 'ndwi', 'latitude', 'longitude']
    X_sim = sim_df[['sim_ndvi', 'ndwi', 'latitude', 'longitude']].rename(
        columns={'sim_ndvi': 'ndvi'}
    )
    
    sim_df['simulated_lst'] = stats['model'].predict(X_sim)
    sim_df['cooling_potential'] = sim_df['land_surface_temperature'] - sim_df['simulated_lst']
    
    # Calculate Impact Metrics
    avg_cooling = sim_df['cooling_potential'].mean()
    
    # Cooling per 0.1 NDVI (ROI)
    # Avoid division by zero
    cooling_roi = (avg_cooling / (ndvi_increase + 1e-8)) * 0.1 if ndvi_increase > 0 else 0
    
    # Impact on Hotspots (Top 20% Hottest areas)
    hotspot_threshold = df['land_surface_temperature'].quantile(0.8)
    hotspot_cooling = sim_df[df['land_surface_temperature'] >= hotspot_threshold]['cooling_potential'].mean()
    
    metrics = {
        'avg_cooling': avg_cooling,
        'max_cooling': sim_df['cooling_potential'].max(),
        'cooling_roi': cooling_roi,
        'hotspot_cooling': hotspot_cooling,
        'ndvi_increase': ndvi_increase
    }
    
    return sim_df, metrics
