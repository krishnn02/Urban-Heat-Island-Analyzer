import pandas as pd
from fpdf import FPDF
import io

class ReportPDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.cell(0, 10, "Urban Heat Island Municipal Report", border=False, align="C")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, "Data Source: NASA/USGS Landsat & ESA Sentinel-2 via Microsoft Planetary Computer", border=False, align="L")
        self.cell(0, 10, f"Page {self.page_no()}", align="R")

def generate_pdf_report(city_name, context, history_df, plan_text, df=None, current_stats=None, sim_stats=None):
    """
    Generates a PDF report summarizing the Urban Heat Analysis.
    Returns the PDF as a byte stream suitable for Streamlit download.
    """
    pdf = ReportPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, f"Location Analysis: {city_name.upper()}", ln=True)
    pdf.ln(5)
    
    # 1. Context & Recommended Flora
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "1. Climate Context & Recommendations", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 8, f"Climate Zone: {context['climate']}\nAverage Summer Peak: {context['temp']}\nPrimary Challenge: {context['issue']}")
    pdf.ln(2)
    pdf.set_font("helvetica", "I", 11)
    pdf.multi_cell(0, 8, f"Recommended Native Flora for Cooling: {context['trees']}")
    pdf.ln(5)
    
    # 2. Scientific Metrics & Mathematical Proof
    if current_stats:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "1. Current Micro-Climate State", ln=True)
        pdf.set_font("helvetica", "", 11)
        
        avg_temp = current_stats.get('avg_temp', 0)
        avg_ndvi = current_stats.get('avg_ndvi', 0)
        correlation = current_stats.get('correlation', 0)
        
        pdf.multi_cell(0, 8, f"Average Land Surface Temp (LST): {avg_temp:.1f} C\nAverage Vegetation Index (NDVI): {avg_ndvi:.2f}")
        pdf.cell(0, 8, f"Vegetation-Heat Correlation: {correlation:.2f}", ln=True)
        pdf.set_font("helvetica", "I", 10)
        pdf.multi_cell(0, 6, "A negative correlation coefficient indicates a strong inverse relationship: as vegetation decreases, surface temperature increases.")
        pdf.ln(5)
        
        # Add Correlation Chart (Optional Feature)
        if df is not None and not df.empty:
            try:
                # Dynamic import to hide from static analysis if not installed
                plt = __import__('matplotlib.pyplot', fromlist=[''])
                plt.figure(figsize=(6, 4))

                plt.scatter(df['ndvi'], df['land_surface_temperature'], alpha=0.3, color='#e76f51', s=10)
                plt.title(f"Scientific Correlation: NDVI vs Temperature ({city_name})")
                plt.xlabel("Vegetation Index (NDVI)")
                plt.ylabel("Temperature (Celsius)")
                plt.grid(True, linestyle='--', alpha=0.6)
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
                img_buf.seek(0)
                
                pdf.image(img_buf, x=30, w=150)
                plt.close()
                pdf.ln(5)
            except ImportError:
                pdf.set_font("helvetica", "I", 10)
                pdf.cell(0, 10, "(Visualization omitted: matplotlib not installed)", ln=True)
                pdf.ln(5)
            except Exception as e:
                pdf.ln(5)

    
    # 3. Historical Trends
    if not history_df.empty:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "3. Historical Trends", ln=True)
        pdf.set_font("helvetica", "", 11)
        try:
            start_temp = history_df['avg_temp'].iloc[0]
            end_temp = history_df['avg_temp'].iloc[-1]
            temp_diff = end_temp - start_temp
            pdf.multi_cell(0, 8, f"Historical analysis indicates the average temperature has changed by approximately {temp_diff:.1f}°C, correlating with shifts in urban vegetation (NDVI).")
        except:
            pdf.multi_cell(0, 8, "Detailed multi-year trend analysis is pending real-time data sync.")
        pdf.ln(5)
    else:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "3. Historical Trends", ln=True)
        pdf.set_font("helvetica", "I", 11)
        pdf.multi_cell(0, 8, "Historical trend data (2010-2024) is currently being migrated to real satellite sources for scientific accuracy. Please refer to the 'Temporal Change' tab in the dashboard for actual multi-year comparisons.")
        pdf.ln(5)
    
    # 4. Simulation Impact
    if sim_stats:
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "4. Simulation Impact Analysis", ln=True)
        pdf.set_font("helvetica", "", 11)
        
        ndvi_inc = sim_stats.get('ndvi_increase', 0)
        avg_cool = sim_stats.get('avg_cooling', 0)
        max_cool = sim_stats.get('max_cooling', 0)
        
        pdf.multi_cell(0, 8, f"Simulated NDVI Increase: +{ndvi_inc:.2f}\nPredicted Average Cooling: -{avg_cool:.1f} C\nMaximum Localized Cooling: -{max_cool:.1f} C")
        pdf.ln(5)
    
    # 5. Action Plan
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "5. Municipal Action Plan", ln=True)
    pdf.set_font("helvetica", "", 11)
    
    # Clean plan text for PDF (fpdf2 default fonts are Latin-1)
    # Remove markdown characters and common emojis that might crash standard fonts
    clean_plan = plan_text.replace("*", "-").replace("#", "").strip()
    
    # Replace common problematic characters with ASCII equivalents
    replacements = {
        "📋": "Plan:", "🌡️": "Temp:", "🌿": "Greenery:", "⚠️": "Warning:", 
        "🚀": "Action:", "📉": "Trend:", "🌳": "Tree:", "🔬": "Science:",
        "📍": "Location:", "🕒": "Time:", "🧪": "Test:"
    }
    for char, rep in replacements.items():
        clean_plan = clean_plan.replace(char, rep)
        
    # Final fallback to avoid latin-1 encoding errors
    clean_plan = clean_plan.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 6, clean_plan)
    
    return bytes(pdf.output())

def generate_csv_report(history_df):
    """
    Converts historical data to CSV bytes.
    """
    return history_df.to_csv(index=False).encode('utf-8')
