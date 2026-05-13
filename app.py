import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Urban Heat Island Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our custom modules
from src.stac_extractor import extract_real_data, get_cache_path
from src.processor import process_for_modeling, calculate_temporal_change
from src.analyzer import calculate_correlations, get_spatial_insights, get_binned_analysis
from src.model import train_predictive_model, calculate_heat_risk_score, simulate_scenarios
from src.planning import generate_municipal_plan
from src.report_generator import generate_pdf_report, generate_csv_report

# Load config
@st.cache_resource
def get_config():
    # Simple hardcoded config since data_mocker is removed
    return {
        'risk_thresholds': {'low': 32, 'medium': 38, 'high': 42},
        'sim_params': {'ndvi_cooling_factor': 5.0}
    }

config = get_config()

# Cached Model Training (Production Requirement)
@st.cache_resource
def get_trained_model(_df, data_key):
    """
    Caches the trained model using a string key derived from request parameters.
    The _df parameter is prefixed with underscore so Streamlit skips hashing it.
    """
    if _df is not None and not _df.empty:
        return train_predictive_model(_df)
    return None


# City Context Database
CITY_METADATA = {
    "delhi": {"temp": "42°C - 48°C", "climate": "Semi-arid", "trees": "Neem, Peepal, Pilkhan", "issue": "Severe concrete heat retention and loss of Aravalli green cover."},
    "mumbai": {"temp": "32°C - 38°C", "climate": "Tropical wet and dry", "trees": "Banyan, Indian Almond, Karanj", "issue": "High humidity exacerbating apparent temperature and loss of mangroves."},
    "bangalore": {"temp": "30°C - 36°C", "climate": "Tropical savanna", "trees": "Gulmohar, Tabebuia, Mahogany", "issue": "Rapid concrete expansion replacing traditional lakes and canopy cover."},
    "chennai": {"temp": "35°C - 42°C", "climate": "Tropical wet and dry", "trees": "Tamarind, Palm, Arjuna", "issue": "Coastal humidity combined with high pavement density reducing evening cooling."},
    "ahmedabad": {"temp": "40°C - 46°C", "climate": "Hot semi-arid", "trees": "Babul, Shisham, Neem", "issue": "Extreme dry heatwaves striking dense unshaded urban grids."}
}

def get_city_context(city_name):
    city_lower = city_name.lower()
    for key, data in CITY_METADATA.items():
        if key in city_lower:
            return data
    # Fallback for generic Indian city
    return {"temp": "~40°C (Summer Peak)", "climate": "Varied Subcontinental", "trees": "Native regional shade-providing species (e.g., Neem, Ficus)", "issue": "Rapid urbanization outpacing green infrastructure development."}

# Geocoder setup with longer timeout for stability
geolocator = Nominatim(user_agent="urban_heat_analysis_app", timeout=10)

@st.cache_data(show_spinner=False)
def geocode_city(city_name):
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude, location.address
        return None, None, None
    except GeocoderTimedOut:
        return None, None, None

@st.cache_data(show_spinner=False)
def load_data(lat, lon, radius=2, resolution=60):
    try:
        # 1. Extraction from STAC
        progress_bar = st.progress(0, text="Initializing satellite connection...")
        
        with st.status("📡 Fetching and processing satellite data...", expanded=True) as status:
            date_start, date_end = '2023-04-01', '2023-05-31'
            cache_path = get_cache_path(lat, lon, radius, resolution, date_start, date_end)
            
            if os.path.exists(cache_path):
                status.write("📂 Found data in local cache. Loading...")
                progress_bar.progress(50, text="Loading from cache...")
            else:
                status.write("📡 Connecting to Microsoft Planetary Computer...")
                status.write("⏳ This may take a minute for new regions...")
                progress_bar.progress(20, text="Connecting to STAC API...")
                
            real_df = extract_real_data(lat, lon, radius_km=radius, resolution=resolution)
            
            if real_df is not None and len(real_df) > 0:
                progress_bar.progress(60, text="Cleaning and normalizing data...")
                status.write("🧹 Cleaning and normalizing data...")
                real_df = process_for_modeling(real_df)
                
                progress_bar.progress(80, text="Calculating heat risk zones...")
                status.write("📊 Calculating heat risk zones...")
                real_df = calculate_heat_risk_score(real_df, config)
                
                # Downsample for dashboard performance if needed
                if len(real_df) > 10000:
                    status.write("🚀 Optimizing for dashboard performance...")
                    real_df = real_df.sample(10000, random_state=42)
                
                progress_bar.progress(100, text="Analysis Complete!")
                status.update(label="✅ Data successfully loaded!", state="complete", expanded=False)
                st.session_state['is_simulated'] = False
                
                # Concurrency-safe cache save (only if not exists)
                if not os.path.exists(cache_path):
                    try:
                        real_df.to_csv(cache_path, index=False)
                    except Exception as cache_err:
                        logger.warning(f"Failed to save cache: {cache_err}")
                
                return real_df

            else:
                status.update(label="❌ No imagery found.", state="error")
                st.warning("No satellite imagery found for this region/date. Try a different location or radius.")
    except Exception as e:
        st.error(f"🛰️ Satellite Connection Error: {e}")
        st.info("💡 **Troubleshooting:** This often happens due to temporary network issues or API rate limits. Please try again in a few seconds.")
        if st.button("🔄 Retry Connection"):
            st.rerun()
        st.stop()
    
    return None

# Modern Premium CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    :root {
        --primary: #2a9d8f;
        --secondary: #264653;
        --accent: #e9c46a;
        --danger: #e76f51;
        --bg-glass: rgba(255, 255, 255, 0.85);
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%);
        padding: 4rem 2rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: rotate 20s linear infinite;
        z-index: 0;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .main-header h1 {
        font-weight: 800;
        margin-bottom: 1.5rem;
        font-size: 4rem;
        color: #ffffff !important;
        text-shadow: 0 10px 20px rgba(0,0,0,0.2);
        z-index: 1;
        position: relative;
        letter-spacing: -1px;
    }
    
    .main-header p {
        font-size: 1.4rem;
        opacity: 0.9;
        font-weight: 300;
        max-width: 900px;
        margin: 0 auto;
        line-height: 1.6;
        z-index: 1;
        position: relative;
    }

    /* Glassmorphism Cards */
    .stMetric, .insight-box, .metric-container {
        background: var(--bg-glass) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.07) !important;
        transition: all 0.3s ease !important;
    }
    
    .metric-container:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 45px rgba(31, 38, 135, 0.12) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    [data-testid="stSidebarNav"] {
        background: transparent;
    }

    /* Custom Radio Buttons */
    [data-testid="stSidebar"] .stRadio label {
        padding: 15px 20px;
        background: white;
        border-radius: 15px;
        margin-bottom: 10px;
        border: 1px solid transparent;
        transition: 0.2s;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        border-color: var(--primary);
        background: #f0fdfa;
    }

    /* Tabs Decoration */
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9;
        padding: 8px;
        border-radius: 16px;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        background: transparent;
        border: none;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: var(--primary) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* Hide standard Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} - Removed to keep sidebar toggle visible */
    
    /* Enhanced Sidebar Toggle Visibility */
    [data-testid="stSidebarCollapseButton"] {
        background-color: var(--primary);
        color: white !important;
        border-radius: 50%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        padding: 5px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebarCollapseButton"]:hover {
        transform: scale(1.1);
        background-color: var(--secondary);
    }

</style>
""", unsafe_allow_html=True)

# Custom Header Section
st.markdown("""
<div class="main-header">
    <h1>Urban Heat Island Spatial Analysis</h1>
    <p>A data-driven platform for analyzing the relationship between vegetation loss and urban heat retention. Powered by NASA and ESA satellite imagery.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 About This Project (Click to read)", expanded=False):
    st.markdown("""
    **Welcome to the Urban Heat Island Analyzer!**
    
    🏙️ **What is an Urban Heat Island (UHI)?**
    Indian cities are often significantly warmer than surrounding rural areas. This happens because dense concentrations of concrete, asphalt, and buildings absorb heat, especially during extreme summer heatwaves, while natural vegetation (like older maidans and urban forests) is sparse.
    
    🌳 **Why does this matter in India?**
    With temperatures regularly crossing 45°C in cities like Delhi, Ahmedabad, and Chennai, extreme heat is a severe public health risk. By understanding the relationship between **Vegetation (NDVI)** and **Concrete (Urban Density)**, urban planners and civic bodies can make targeted decisions to cool neighborhoods down.
    
    🔍 **How to use this tool:**
    1. **Search for your city** below.
    2. Explore the **Navigation Menu** in the sidebar to switch between maps, trends, and simulations.
    """)

# Main Location Selector
st.markdown("### 📍 Select a Location to Analyze")
col_loc1, col_loc2 = st.columns([2, 1])
with col_loc1:
    city_input_raw = st.text_input("Enter an Indian City Name (e.g., New Delhi, Mumbai, Bangalore, Chennai):", value="New Delhi")
    city_input = city_input_raw.strip() # Clean whitespace
    st.caption("⚠️ **Note:** Analyzing a new region for the first time may take 1-2 minutes to download satellite imagery.")

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 25px; padding: 25px 15px; background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.05);">
        <h2 style="margin-bottom: 5px; color: #1d3557; font-weight: 800; font-size: 1.6rem; letter-spacing: -0.5px;">🌍 UHI Analyzer</h2>
        <div style="height: 3px; width: 40px; background: #2a9d8f; margin: 10px auto; border-radius: 2px;"></div>
        <p style="color: #457b9d; font-size: 0.95rem; font-weight: 500; margin-bottom: 0; opacity: 0.8;">Urban Intelligence Engine</p>
        <p style="color: #888; font-size: 0.7rem; margin-top: 10px;">Tip: Use the arrow at the top to collapse/expand menu</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Analysis Settings")
    radius = st.slider("Search Radius (km):", min_value=1, max_value=10, value=2, help="Larger radius takes more time to process.")
    
    # NEW: Speed vs Detail Control
    detail_level = st.select_slider(
        "Search Detail (Resolution):",
        options=["Ultra Fast (100m)", "Fast (60m)", "High Detail (30m)"],
        value="Fast (60m)",
        help="Higher detail takes longer. We automatically downsample if the area exceeds 1 million pixels for memory safety."
    )

    
    # Map detail level to resolution in meters
    res_map = {"Ultra Fast (100m)": 100, "Fast (60m)": 60, "High Detail (30m)": 30}
    selected_res = res_map[detail_level]
    
    # Pixel Budget Guardrail (Production Requirement)
    # Area (m^2) / Resolution (m^2)
    area_m2 = (radius * 1000 * 2) ** 2
    pixel_count = area_m2 / (selected_res ** 2)
    
    if pixel_count > 1000000:
        st.warning(f"⚠️ **Budget Warning:** Your current settings will generate ~{pixel_count/1e6:.1f}M pixels. The system will automatically downsample for memory safety.")
    
    st.markdown("### 🧭 Main Navigation")
    page = st.radio("Go to:", ["🗺️ Spatial Analysis", "🕒 Temporal Change", "🌳 Scenario Simulator", "📋 Future Plans & Reports"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: #666; font-size: 0.8rem;'>
        <b>📡 Data Provenance</b><br>
        Landsat-8/9 (NASA/USGS)<br>
        Sentinel-2 L2A (ESA)<br>
        via Microsoft Planetary Computer
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 0.8rem;'>Analysis Engine v1.1.0<br>Scientific Research Tool</p>", unsafe_allow_html=True)
    
    with st.sidebar.expander("🔬 Methodology & Limitations"):
        st.markdown("""
        **Methodology:**
        - **LST**: Derived from Landsat-8/9 TIRS bands using the Mono-Window Algorithm.
        - **NDVI**: Calculated from Sentinel-2 MSI (NIR/Red).
        - **Model**: Advanced Spatial-Aware Random Forest Regressor.
        
        **Limitations:**
        - **Snapshots**: Satellite data represents a single point in time (10:30 AM local).
        - **Surface vs Air**: This tool measures *Surface* Temperature, which can be 10-15°C higher than ambient air temperature.
        - **Resolution**: 30m to 100m pixel spatial resolution.
        """)

lat, lon, address = geocode_city(city_input)

if lat and lon:
    st.success(f"Successfully Located: {address}")
    df = load_data(lat=lat, lon=lon, radius=radius, resolution=selected_res)
    
    # Display City Analytics Context
    context = get_city_context(city_input)
    st.markdown("### 📊 Local City Analytics")
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Summer Peak:** {context['temp']}\n\n**Climate:** {context['climate']}")
    c2.success(f"**Recommended Flora:**\n\n{context['trees']}")
    c3.warning(f"**Key Challenge:**\n\n{context['issue']}")
    # Extract base temp from string (e.g., "42°C - 48°C" -> 42.0)
    try:
        base_t = float(context['temp'].split('°')[0].strip()) - 5.0 # baseline
    except:
        base_t = 35.0
        
    # history_df and historical generation removed for Phase 4 (Honesty)
    history_df = pd.DataFrame() # Empty for compatibility
    plan_markdown = generate_municipal_plan(city_input, context)
    
else:
    st.error("📍 City not found. Please check the spelling or try a more specific location (e.g., 'Mumbai, India').")
    st.stop()

# Fail State Handling: Ensure data exists before proceeding
if df is None or df.empty:
    st.error("🚨 Data fetch failed. The STAC API may be down or no imagery exists for this region/date.")
    st.info("Try refreshing the page or searching for a different city.")
    st.stop()

# Train or fetch cached Random Forest model
data_key = f"{lat}_{lon}_{radius}_{selected_res}"
stats = get_trained_model(df, data_key)

if stats is None:
    st.error("Could not train the predictive model. Please ensure the data is loaded correctly.")
    st.stop()

# Extra CSS for components
st.markdown("""
<style>
    .metric-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #2a9d8f;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        color: #1d3557;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #e9c46a;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #1d3557;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .insight-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .insight-box h4 {
        color: #264653;
        margin-bottom: 15px;
        font-weight: 600;
    }
    .stSlider > div > div > div > div {
        background-color: #2a9d8f !important;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0px 0px;
        gap: 8px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e9ecef;
        border-bottom-color: #2a9d8f !important;
        border-bottom-width: 3px !important;
        color: #2a9d8f !important;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Layout (100% Data-Driven)
st.markdown("### 📊 Region Analytics & Mathematical Proof")
col_d1, col_d2, col_d3 = st.columns(3)

insights = get_spatial_insights(df)

if insights:
    with col_d1:
        st.markdown('<div class="metric-container" style="border-top: 5px solid #e76f51;">', unsafe_allow_html=True)
        st.metric(label="🌡️ Avg Surface Temp", value=f"{df['land_surface_temperature'].mean():.1f}°C")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d2:
        st.markdown('<div class="metric-container" style="border-top: 5px solid #2a9d8f;">', unsafe_allow_html=True)
        st.metric(label="❄️ Cooling Dividend", value=f"{insights['cooling_dividend']:.1f}°C", help="Temperature difference between high-vegetation and low-vegetation areas.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d3:
        st.markdown('<div class="metric-container" style="border-top: 5px solid #457b9d;">', unsafe_allow_html=True)
        blue_val = insights.get('blue_cooling_index', 0.0)
        st.metric(label="💧 Blue Cooling Index", value=f"{blue_val:.1f}°C", help="Temperature reduction benefit provided by water bodies.")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Spatial insights could not be calculated for this dataset.")



st.markdown("<br>", unsafe_allow_html=True)

with st.expander("🤖 View Predictive Model Performance (Random Forest)", expanded=False):
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(label="Test R² Score", value=f"{stats['r2_score']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_m2:
        st.markdown('<div class="metric-container" style="border-left-color: #fca311;">', unsafe_allow_html=True)
        st.metric(label="Mean Absolute Error (MAE)", value=f"±{stats['mae']:.2f}°C")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_m3:
        st.markdown('<div class="metric-container" style="border-left-color: #e63946;">', unsafe_allow_html=True)
        st.metric(label="RMSE", value=f"±{stats['rmse']:.2f}°C")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("#### 🔑 Why this model works")
    st.info("This model focuses on **Vegetation (NDVI)** and **Spatial Location** (Micro-climates) to predict heat patterns. In Phase 2, we will add real Landcover and Building Density data.")
    
    # Feature Importance Chart
    importance_df = pd.DataFrame({
        'Feature': list(stats['feature_importances'].keys()),
        'Importance': list(stats['feature_importances'].values())
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                            title="Feature Importance: What drives the temperature?",
                            color='Importance', color_continuous_scale='Viridis')
    fig_importance.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_importance, width='stretch')

st.markdown("---")

# Main visualizations
st.markdown("---")

color_discrete_map = {'High': '#e76f51', 'Medium': '#e9c46a', 'Low': '#2a9d8f'}

if page == "🗺️ Spatial Analysis":
    st.header("Urban Heat and Vegetation Maps")
    st.markdown("Explore the spatial distribution of temperature, vegetation, and calculated risk across the city grid side-by-side.")
    
    tab_maps, tab_science = st.tabs(["🗺️ Distribution Maps", "🧪 Scientific Proof"])
    
    with tab_maps:
        col_map1, col_map2, col_map3 = st.columns(3)
        
        with col_map1:
            st.markdown('<div class="insight-box" style="padding: 10px; margin-top: 0; margin-bottom: 15px; border-left-color: #ef476f;">', unsafe_allow_html=True)
            st.subheader("🌡️ LST (°C)")
            st.markdown('</div>', unsafe_allow_html=True)
            fig_temp = px.scatter_map(df, lat="latitude", lon="longitude", color="land_surface_temperature",
                                         color_continuous_scale="Inferno", size_max=15, zoom=10,
                                         map_style="carto-positron", opacity=0.7)
            fig_temp.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
            st.plotly_chart(fig_temp, width='stretch')

        with col_map2:
            st.markdown('<div class="insight-box" style="padding: 10px; margin-top: 0; margin-bottom: 15px; border-left-color: #06d6a0;">', unsafe_allow_html=True)
            st.subheader("🌿 NDVI")
            st.markdown('</div>', unsafe_allow_html=True)
            fig_ndvi = px.scatter_map(df, lat="latitude", lon="longitude", color="ndvi",
                                         color_continuous_scale="YlGn", size_max=15, zoom=10,
                                         map_style="carto-positron", opacity=0.7)
            fig_ndvi.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
            st.plotly_chart(fig_ndvi, width='stretch')
            
        with col_map3:
            st.markdown('<div class="insight-box" style="padding: 10px; margin-top: 0; margin-bottom: 15px; border-left-color: #e9c46a;">', unsafe_allow_html=True)
            st.subheader("⚠️ Risk Zones")
            st.markdown('</div>', unsafe_allow_html=True)
            fig_risk = px.scatter_map(df, lat="latitude", lon="longitude", color="heat_risk_zone",
                                         color_discrete_map=color_discrete_map, size_max=15, zoom=10,
                                         map_style="carto-positron", opacity=0.8)
            fig_risk.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
            st.plotly_chart(fig_risk, width='stretch')

    with tab_science:
        st.subheader("Mathematical Correlation & Insights")
        analysis = calculate_correlations(df)
        st.session_state['analysis_results'] = analysis
        spatial = get_spatial_insights(df)
        
        if analysis and spatial:
            col_proof1, col_proof2 = st.columns([1, 2])
            
            with col_proof1:
                st.markdown(f"""
                <div class="insight-box" style="border-left-color: #2a9d8f;">
                    <h4>The Proof</h4>
                    <p style="font-size: 1.2rem; font-weight: 800; color: #e76f51;">
                        NDVI vs Temperature = {analysis['pearson']:.2f}
                    </p>
                    <p><b>R² Score:</b> {analysis['r_squared']:.2f}</p>
                    <p><b>P-Value:</b> {analysis['p_value']:.4e}</p>
                    <p><i>Statistical Significance: { "High" if analysis['p_value'] < 0.05 else "Low" }</i></p>
                </div>
                
                <div class="insight-box" style="border-left-color: #118ab2;">
                    <h4>Spatial Insights</h4>
                    <p><b>Cooling Dividend:</b> {spatial['cooling_dividend']:.1f}°C</p>
                    <p><small>The temperature difference between the top 20% greenest and bottom 20% greenest areas.</small></p>
                    <p><b>Hotspots Found:</b> {spatial['num_hotspots']}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_proof2:
                # Advanced Scatter with Marginals
                fig_advanced = px.scatter(df, x="ndvi", y="land_surface_temperature", 
                                         color="land_surface_temperature", marginal_x="histogram", marginal_y="violin",
                                         trendline="ols", trendline_color_override="red",
                                         title="Scientific Relationship: NDVI vs Surface Temp",
                                         labels={"ndvi": "Vegetation (NDVI)", "land_surface_temperature": "Temp (°C)"})
                st.plotly_chart(fig_advanced, width='stretch')
                
            # Density Heatmap
            st.subheader("Density Heatmap: Where do most urban pixels fall?")
            fig_density = px.density_heatmap(df, x="ndvi", y="land_surface_temperature", 
                                           nbinsx=30, nbinsy=30, color_continuous_scale="Viridis",
                                           title="Point Density (NDVI vs Temp)")
            st.plotly_chart(fig_density, width='stretch')
            
            st.markdown("""
            <div class="insight-box" style="border-left-color: #264653; background-color: #f8f9fa;">
                <h4>🧪 Thermal Physics Diagnosis: Why is this area hot?</h4>
                <p>The observed heat patterns are driven by three primary physical factors:</p>
                <ul>
                    <li><b>Thermal Inertia:</b> Dense urban materials (concrete, asphalt) have high heat capacity, absorbing solar radiation during the day and re-radiating it at night.</li>
                    <li><b>Low Albedo:</b> Dark urban surfaces absorb more sunlight compared to natural landscapes, leading to higher surface temperatures.</li>
                    <li><b>Evapotranspiration Deficit:</b> Vegetated areas cool the air through moisture release. In low-NDVI zones, this natural cooling mechanism is absent, creating "Heat Islands."</li>
                </ul>
                <p><small><i>Note: Statistical significance is verified via Pearson correlation (shown above).</i></small></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Insufficient data to perform scientific analysis. Try a larger search radius.")

# Historical Trends page removed for Phase 4

elif page == "🕒 Temporal Change":
    st.header("🕒 Satellite Time Machine: Multi-Year Analysis")
    st.markdown("Compare actual satellite data across different years to visualize urban expansion and its direct thermal impact.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        base_year = st.selectbox("Select Baseline Year:", [2018, 2019, 2020, 2021], index=0)
    with col_t2:
        current_year = st.selectbox("Select Current Year:", [2022, 2023, 2024], index=1)
        
    if st.button("🚀 Run Comparative Analysis"):
        with st.status("Analyzing temporal changes...") as status:
            st.write(f"📡 Fetching Baseline Data ({base_year})...")
            df_base = extract_real_data(lat, lon, radius_km=radius, date_start=f"{base_year}-04-01", date_end=f"{base_year}-05-31", resolution=selected_res)
            
            st.write(f"📡 Fetching Current Data ({current_year})...")
            # We align the current year to the baseline grid
            df_curr = extract_real_data(lat, lon, radius_km=radius, date_start=f"{current_year}-04-01", date_end=f"{current_year}-05-31", resolution=selected_res)
            
            if df_base is not None and df_curr is not None:
                st.write("🧹 Processing and Aligning Grids...")
                df_base = process_for_modeling(df_base)
                df_curr = process_for_modeling(df_curr)
                
                st.write("📊 Calculating Pixel-Level Deltas...")
                delta_df = calculate_temporal_change(df_base, df_curr)
                
                if delta_df is not None:
                    st.session_state.delta_df = delta_df
                    status.update(label="Temporal Analysis Complete!", state="complete", expanded=False)
                else:
                    st.error("Failed to align datasets. Ensure the search radius is sufficient.")
            else:
                st.error("Could not fetch real data for one or both years. Check internet or region coverage.")

    if 'delta_df' in st.session_state:
        ddf = st.session_state.delta_df
        
        st.markdown(f"""
        <div class="insight-box" style="border-left-color: #2a9d8f;">
            <h4>Analysis Summary ({base_year} vs {current_year})</h4>
            <p><b>Avg Temp Change:</b> {ddf['delta_temp'].mean():+.1f}°C</p>
            <p><b>Avg NDVI Change:</b> {ddf['delta_ndvi'].mean():+.2f}</p>
            <p><i>The "Satellite Time Machine" has analyzed {len(ddf)} unique urban points across both years.</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        col_dmap1, col_dmap2 = st.columns(2)
        
        with col_dmap1:
            st.subheader("🌿 Change in Vegetation (Δ NDVI)")
            st.markdown("Red indicates vegetation loss; Green indicates gain.")
            fig_d_ndvi = px.scatter_map(ddf, lat="latitude", lon="longitude", color="delta_ndvi",
                                         color_continuous_scale="RdYlGn", size_max=15, zoom=10,
                                         map_style="carto-positron", opacity=0.8,
                                         range_color=[-0.3, 0.3])
            fig_d_ndvi.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_d_ndvi, width='stretch')
            
        with col_dmap2:
            st.subheader("🌡️ Change in Temperature (Δ Temp)")
            st.markdown("Red indicates heating; Blue indicates cooling.")
            fig_d_temp = px.scatter_map(ddf, lat="latitude", lon="longitude", color="delta_temp",
                                         color_continuous_scale="RdBu_r", size_max=15, zoom=10,
                                         map_style="carto-positron", opacity=0.8,
                                         range_color=[-5, 5])
            fig_d_temp.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_d_temp, width='stretch')
            
        st.markdown("---")
        st.subheader("📉 The Smoking Gun: NDVI Loss vs. Temperature Gain")
        fig_d_scatter = px.scatter(ddf, x="delta_ndvi", y="delta_temp", 
                                  color="delta_temp", color_continuous_scale="Viridis",
                                  trendline="ols", trendline_color_override="red",
                                  title="Proving Urbanization Impact: ΔNDVI vs ΔTemp",
                                  labels={"delta_ndvi": "Change in Vegetation (NDVI)", "delta_temp": "Change in Temp (°C)"})
        st.plotly_chart(fig_d_scatter, width='stretch')

elif page == "🌳 Scenario Simulator":
    st.header("Multi-Variate Scenario Simulator")
    st.markdown("Use our Random Forest model to predict cooling effects from urban planning policies.")
    
    col_slider1, col_slider2 = st.columns(2)
    
    if 'ndvi_inc' not in st.session_state:
        st.session_state.ndvi_inc = 0.1
    if 'density_dec' not in st.session_state:
        st.session_state.density_dec = 0.0
        
    with col_slider1:
        st.session_state.ndvi_inc = st.slider("Simulate average NDVI increase (Planting trees):", min_value=0.0, max_value=0.4, value=st.session_state.ndvi_inc, step=0.05, help="NDVI represents green vegetation. An increase of 0.1 is roughly equivalent to planting dense street trees in an empty block.")
    
    # Simulation now only uses NDVI (Honest approach)
    sim_df, sim_metrics = simulate_scenarios(df, stats, ndvi_increase=st.session_state.ndvi_inc)
    
    if sim_df is not None:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #2a9d8f 0%, #264653 100%); padding: 25px; border-radius: 15px; color: white; margin-bottom: 30px; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.1);">
            <h3 style="margin: 0; color: white;">📈 Model Metric: Cooling Coefficient</h3>
            <p style="font-size: 2.2rem; font-weight: 800; margin: 10px 0; color: #e9c46a;">
                +0.1 NDVI → -{sim_metrics['cooling_roi']:.1f}°C
            </p>
            <p style="opacity: 0.9; margin: 0;">Predicted cooling rate based on the current city's spatial characteristics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_sim_stats1, col_sim_stats2 = st.columns(2)
        
        with col_sim_stats1:
            st.markdown(f"""
            <div class="insight-box" style="border-left-color: #e76f51;">
                <h4>Impact Analysis</h4>
                <p><b>Average Cooling:</b> {sim_metrics['avg_cooling']:.1f}°C</p>
                <p><b>Hotspot Relief:</b> {sim_metrics['hotspot_cooling']:.1f}°C (Cooling in the hottest 20% of areas)</p>
                <p><b>Max Local Cooling:</b> {sim_metrics['max_cooling']:.1f}°C</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_sim_stats2:
            st.markdown(f"""
            <div class="insight-box" style="border-left-color: #e9c46a;">
                <h4>Policy Recommendation</h4>
                <p>Based on this simulation, increasing the green canopy by <b>{st.session_state.ndvi_inc*100:.0f}%</b> could significantly reduce heat-related health risks in this region.</p>
                <p>Priority zones for tree plantation have been identified in the "Risk Zones" map.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader("🖼️ Side-by-Side Visual Proof")
        col_map_current, col_map_sim = st.columns(2)
        
        with col_map_current:
            st.markdown("##### Current Temperature")
            fig_curr = px.scatter_map(df, lat="latitude", lon="longitude", color="land_surface_temperature",
                                         color_continuous_scale="Inferno", size_max=15, zoom=10,
                                         map_style="carto-positron", opacity=0.7)
            fig_curr.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_showscale=False)
            st.plotly_chart(fig_curr, width='stretch')
            
        with col_map_sim:
            st.markdown(f"##### Simulated (After +{st.session_state.ndvi_inc} NDVI)")
            # Use same color scale range for fair comparison
            t_min, t_max = df['land_surface_temperature'].min(), df['land_surface_temperature'].max()
            fig_sim = px.scatter_map(sim_df, lat="latitude", lon="longitude", color="simulated_lst",
                                         color_continuous_scale="Inferno", size_max=15, zoom=10,
                                         map_style="carto-positron", opacity=0.7,
                                         range_color=[t_min, t_max])
            fig_sim.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_sim, width='stretch')
    else:
        st.warning("Please load data to run the simulation.")

elif page == "📋 Future Plans & Reports":
    st.header("Municipal Action Plan & Reporting")
    st.markdown("Use this structured strategy and export the data to drive policy decisions within the Municipal Corporation.")
    
    st.markdown(plan_markdown)
    
    st.markdown("---")
    st.subheader("📥 Export Data & Reports")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # Retrieve analysis results from session state or calculate on the fly
        if 'analysis_results' in st.session_state:
            analysis = st.session_state['analysis_results']
        else:
            analysis = calculate_correlations(df)
            st.session_state['analysis_results'] = analysis

        current_stats = {
            'avg_temp': df['land_surface_temperature'].mean(),
            'avg_ndvi': df['ndvi'].mean(),
            'correlation': analysis['pearson'] if analysis else 0.0
        }

        
        # Pull simulation values from session state if available, else defaults
        sim_ndvi = st.session_state.get('ndvi_inc', 0.1)
        
        sim_df, sim_metrics = simulate_scenarios(df, stats, ndvi_increase=sim_ndvi)
        
        sim_stats = {
            'ndvi_increase': sim_ndvi,
            'avg_cooling': sim_metrics['avg_cooling'] if sim_metrics else 0,
            'max_cooling': sim_metrics['max_cooling'] if sim_metrics else 0
        }
        pdf_bytes = generate_pdf_report(city_input if lat else "New Delhi", context, history_df, plan_markdown, df, current_stats, sim_stats)

        st.download_button(
            label="📄 Download Municipal PDF Report",
            data=pdf_bytes,
            file_name=f"UHI_Report_{city_input if lat else 'NewDelhi'}.pdf",
            mime="application/pdf"
        )
        
    with col_dl2:
        st.info("CSV Export currently disabled as mock historical data has been removed for scientific integrity. Real historical CSV export coming in v1.2.0.")

