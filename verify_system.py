import sys
import os
import warnings
import pandas as pd
import numpy as np

# Add src to path for module resolution
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Top-level imports for IDE resolution
try:
    from src.stac_extractor import extract_real_data, get_cache_path
    from src.processor import process_for_modeling
    from src.model import train_predictive_model, simulate_scenarios, calculate_heat_risk_score
    from src.analyzer import calculate_correlations, get_spatial_insights, get_binned_analysis
    from src.report_generator import generate_pdf_report, generate_csv_report
    from src.planning import generate_municipal_plan
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False

# Suppress sklearn warnings for tiny test datasets
warnings.filterwarnings('ignore', category=UserWarning)


def verify_system():
    print("=" * 60)
    print("  URBAN HEAT ISLAND - SYSTEM VERIFICATION")
    print("=" * 60)
    results = []

    # 1. Check Imports
    print("\n[1/5] Testing imports...")
    if IMPORTS_OK:
        print("  OK - All 6 modules imported.")
        results.append(("Imports", "PASS"))
    else:
        print("  FAIL - Could not import modules from src/ directory.")
        results.append(("Imports", "FAIL: ImportError"))
        return _print_summary(results)

    # 2. Data Processing Pipeline
    print("\n[2/5] Testing data pipeline...")
    try:
        # Use 50 rows so train_test_split and cross-validation work properly
        np.random.seed(42)
        n = 50
        ndvi_vals = np.random.uniform(0.0, 0.9, n)
        test_df = pd.DataFrame({
            'latitude': np.random.uniform(28.5, 28.7, n),
            'longitude': np.random.uniform(77.1, 77.3, n),
            'ndvi': ndvi_vals,
            'ndwi': np.random.uniform(-0.3, 0.1, n),
            'land_surface_temperature': 45 - (10 * ndvi_vals) + np.random.normal(0, 1, n)
        })

        processed_df = process_for_modeling(test_df)
        assert len(processed_df) > 0, "Processed dataframe is empty"
        assert 'ndvi_normalized' in processed_df.columns, "Missing ndvi_normalized column"
        print(f"  OK - Processed {len(processed_df)} records. Columns: {processed_df.columns.tolist()}")
        results.append(("Data Pipeline", "PASS"))
    except Exception as e:
        print(f"  FAIL - {e}")
        results.append(("Data Pipeline", f"FAIL: {e}"))
        return _print_summary(results)

    # 3. Model Training
    print("\n[3/5] Testing model training...")
    try:
        stats = train_predictive_model(processed_df)
        assert stats is not None, "train_predictive_model returned None"
        assert 'model' in stats, "Missing 'model' key in stats"
        assert 'r2_score' in stats, "Missing 'r2_score' key"
        assert not np.isnan(stats['r2_score']), "R2 score is NaN"
        print(f"  OK - R2: {stats['r2_score']:.3f}, MAE: {stats['mae']:.2f}C, RMSE: {stats['rmse']:.2f}C")
        print(f"       Features: {stats['features']}")
        results.append(("Model Training", "PASS"))
    except Exception as e:
        print(f"  FAIL - {e}")
        results.append(("Model Training", f"FAIL: {e}"))
        return _print_summary(results)

    # 4. Simulation
    print("\n[4/5] Testing scenario simulation...")
    try:
        sim_df, metrics = simulate_scenarios(processed_df, stats, ndvi_increase=0.1)
        assert sim_df is not None, "Simulation returned None"
        assert metrics is not None, "Simulation metrics are None"
        assert 'avg_cooling' in metrics, "Missing avg_cooling"
        print(f"  OK - Avg Cooling: {metrics['avg_cooling']:.2f}C, Max: {metrics['max_cooling']:.2f}C")
        results.append(("Simulation", "PASS"))
    except Exception as e:
        print(f"  FAIL - {e}")
        results.append(("Simulation", f"FAIL: {e}"))

    # 5. Report Generation
    print("\n[5/5] Testing PDF report...")
    try:
        context = {"climate": "Semi-arid", "temp": "42C", "issue": "Heat retention", "trees": "Neem, Peepal"}
        history_df = pd.DataFrame()
        current_stats = {'avg_temp': 40.5, 'avg_ndvi': 0.35, 'correlation': -0.72}
        pdf_bytes = generate_pdf_report(
            "Test City", context, history_df, "Test municipal plan text.",
            processed_df, current_stats, metrics
        )
        assert len(pdf_bytes) > 100, f"PDF too small: {len(pdf_bytes)} bytes"
        print(f"  OK - PDF generated: {len(pdf_bytes)} bytes")
        results.append(("PDF Report", "PASS"))
    except Exception as e:
        print(f"  FAIL - {e}")
        results.append(("PDF Report", f"FAIL: {e}"))

    return _print_summary(results)


def _print_summary(results):
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, status in results:
        icon = "PASS" if "PASS" in status else "FAIL"
        if icon == "FAIL":
            all_pass = False
        print(f"  [{icon}] {name}")

    if all_pass:
        print("\n  System is 100% operational and data-driven.")
    else:
        print("\n  Some checks failed. Review output above.")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    success = verify_system()
    if not success:
        sys.exit(1)
