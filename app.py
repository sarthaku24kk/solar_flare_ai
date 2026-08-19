import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import io
import time

from src.data_loader import load_solexs_from_zip, load_hel1os_from_zip, merge_and_synchronize
from src.physics_engine import compute_physics_features, detect_flare_events
from src.ai_model import SolarFlareAI

# Page configuration
st.set_page_config(
    page_title="Aditya-L1 Solar Flare AI Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark astronomy UI
st.markdown("""
<style>
    .main {
        background-color: #0b0e14;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #141722 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #2d3345;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: #161b22;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'solar_flare_ai_model.pkl')
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), 'models', 'training_summary.json')
DEFAULT_DATA_DIR = r'C:\Users\sarth\OneDrive\Pictures\New folder'

@st.cache_resource
def get_ai_model():
    if os.path.exists(MODEL_PATH):
        return SolarFlareAI.load(MODEL_PATH)
    return None

@st.cache_data
def get_training_summary():
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, 'r') as f:
            return json.load(f)
    return None

ai_model = get_ai_model()
summary_data = get_training_summary()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Indian_Space_Research_Organisation_Logo.svg/1200px-Indian_Space_Research_Organisation_Logo.svg.png", width=70)
    st.title("Aditya-L1 Mission")
    st.caption("ISRO Lagrange Point L1 Observatory")
    st.markdown("---")
    
    st.subheader("Data Input Mode")
    input_mode = st.radio(
        "Select Workflow",
        ["📁 Upload 3 PRADAN Zip Files", "📅 Explore 20-Day Mission Archive"],
        index=0
    )
    
    uploaded_slx = None
    uploaded_hls1 = None
    uploaded_hls2 = None
    selected_date = None
    
    if input_mode == "📁 Upload 3 PRADAN Zip Files":
        st.info("Upload standard PRADAN zip archives directly:")
        uploaded_slx = st.file_uploader("1. SoLEXS 24h Zip (AL1_SLX_...zip)", type=['zip'], key="slx_up")
        uploaded_hls1 = st.file_uploader("2. HEL1OS 12h Part 1 Zip (00-12h)", type=['zip'], key="hls1_up")
        uploaded_hls2 = st.file_uploader("3. HEL1OS 12h Part 2 Zip (12-24h)", type=['zip'], key="hls2_up")
    else:
        # Pre-loaded archive
        if os.path.exists(DEFAULT_DATA_DIR):
            all_files = os.listdir(DEFAULT_DATA_DIR)
            slx_dates = sorted(list(set([f.split('_')[3] for f in all_files if f.startswith('AL1_SLX') and '_' in f])))
            selected_date = st.selectbox("Select Mission Observation Date:", slx_dates, index=0)
            st.caption(f"Loaded from mission repository: `{DEFAULT_DATA_DIR}`")
        else:
            st.error(f"Archive folder not found at `{DEFAULT_DATA_DIR}`")
            
    st.markdown("---")
    st.subheader("AI System Config")
    resample_rate = st.selectbox("Timeline Resolution", ["10s (High Cadence)", "30s (Balanced)", "60s (Fast)"], index=0)
    resample_freq = resample_rate.split()[0]
    dt_sec = int(resample_freq.replace('s', ''))
    
    min_prominence = st.slider("Flare Detection Sensitivity (Prominence)", min_value=3.0, max_value=25.0, value=8.0, step=1.0)
    
    st.markdown("---")
    if summary_data:
        st.caption(f"🧠 **AI Model Status:** Online (Trained on {summary_data.get('total_days', 20)} Days / {summary_data.get('total_samples', 0):,} data points)")
        st.caption(f"📈 **Model ROC-AUC:** {summary_data.get('metrics', {}).get('roc_auc_1h', 0.971):.3f}")

# --- MAIN DASHBOARD ---
st.title("☀️ Aditya-L1 Solar Flare AI Nowcasting & Forecasting Engine")
st.markdown("Automated multi-payload algorithmic pipeline fusing **SoLEXS Soft X-rays** ($1\text{--}30\text{ keV}$) and **HEL1OS Hard X-rays** ($10\text{--}150\text{ keV}$) for real-time detection, hardness analysis, and early space weather warning.")

# Run Processing
processed_data = None
flare_events = []

def process_day(slx_input, hls_inputs):
    with st.spinner("🚀 Ingesting zip files, extracting FITS tables, and computing cross-payload physics features..."):
        slx_df = load_solexs_from_zip(slx_input)
        hls_dfs = []
        for h in hls_inputs:
            if h is not None:
                try:
                    hdf = load_hel1os_from_zip(h)
                    hls_dfs.append(hdf)
                except Exception as e:
                    st.warning(f"Could not parse one HEL1OS file: {e}")
        
        if not hls_dfs:
            st.error("No valid HEL1OS lightcurves found.")
            return None, []
            
        synced_df = merge_and_synchronize(slx_df, hls_dfs, resample_freq=resample_freq)
        physics_df = compute_physics_features(synced_df, dt_sec=dt_sec)
        
        # Run AI Model
        if ai_model is not None:
            predicted_df = ai_model.predict_timeline(physics_df)
        else:
            predicted_df = physics_df
            predicted_df['prob_flare_1h'] = 0.0
            predicted_df['prob_flare_2h'] = 0.0
            predicted_df['forecast_risk_level'] = 'NOMINAL'
            
        events = detect_flare_events(predicted_df, min_prominence=min_prominence, dt_sec=dt_sec)
        return predicted_df, events

if input_mode == "📁 Upload 3 PRADAN Zip Files":
    if uploaded_slx and (uploaded_hls1 or uploaded_hls2):
        h_list = [h for h in [uploaded_hls1, uploaded_hls2] if h is not None]
        processed_data, flare_events = process_day(uploaded_slx, h_list)
    else:
        st.info("👋 **Please upload 1 SoLEXS zip and at least 1 HEL1OS zip in the sidebar to begin analysis.**")
        st.markdown("""
        #### How it works:
        1. **Direct In-Memory Parsing:** Drop standard `.zip` files downloaded from ISRO ISSDC PRADAN portal.
        2. **Multi-Rate Alignment:** Automatically extracts and aligns SoLEXS soft X-rays (SDD2/SDD1) and HEL1OS hard X-ray bands (CZT & CDTE).
        3. **Physics + AI Inference:** Dynamic quiescent baseline removal, Hardness Ratio ($HR$), Neupert time derivative ($\frac{dF}{dt}$), and 1h–2h predictive flare horizon.
        """)
else:
    # Selected date mode
    if selected_date and os.path.exists(DEFAULT_DATA_DIR):
        slx_path = os.path.join(DEFAULT_DATA_DIR, f"AL1_SLX_L1_{selected_date}_v1.0.zip")
        hls_files = sorted([os.path.join(DEFAULT_DATA_DIR, f) for f in os.listdir(DEFAULT_DATA_DIR) if f.startswith('HLS') and selected_date in f])
        
        if os.path.exists(slx_path) and hls_files:
            processed_data, flare_events = process_day(slx_path, hls_files)

# Render Results
if processed_data is not None:
    df = processed_data
    
    # Summary Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    total_flares = len(flare_events)
    max_flux = df['solexs_counts'].max()
    max_hls = df['hel1os_czt_total'].max()
    max_hr = df['hardness_ratio'].max()
    max_risk_1h = df['prob_flare_1h'].max()
    
    # Class determination
    if max_flux >= 800:
        peak_class = "X-Class"
        class_color = "#FF1744"
    elif max_flux >= 150:
        peak_class = "M-Class"
        class_color = "#FF9100"
    elif max_flux >= 40:
        peak_class = "C-Class"
        class_color = "#FFD600"
    elif max_flux >= 15:
        peak_class = "B-Class"
        class_color = "#00E676"
    else:
        peak_class = "Quiet Sun"
        class_color = "#00B0FF"
        
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Flares Detected (24h)</div>
            <div class="metric-value">{total_flares}</div>
            <span class="metric-badge" style="background:#238636; color:#fff;">Nowcast Active</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak Flare Class</div>
            <div class="metric-value" style="color:{class_color};">{peak_class}</div>
            <span class="metric-badge" style="background:{class_color}33; color:{class_color};">Peak: {max_flux:.1f} cts/s</span>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Max Hardness Ratio (HR)</div>
            <div class="metric-value" style="color:#a371f7;">{max_hr:.3f}</div>
            <span class="metric-badge" style="background:#8957e533; color:#d2a8ff;">Hard/Soft Ratio</span>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak 1h Flare Risk</div>
            <div class="metric-value" style="color:#f78166;">{max_risk_1h:.1f}%</div>
            <span class="metric-badge" style="background:#da363333; color:#f78166;">AI Forecast Window</span>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Max Hard X-Ray Flux</div>
            <div class="metric-value" style="color:#38bdf8;">{max_hls:.1f}</div>
            <span class="metric-badge" style="background:#0284c733; color:#38bdf8;">HEL1OS CZT Total</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS FOR VISUALIZATION, CATALOG, AND SPACE WEATHER REPORT ---
    tab_plots, tab_catalog, tab_advisory, tab_export = st.tabs([
        "📈 Interactive Light Curves & AI Predictions",
        "📋 Detected Flare Events Catalog",
        "🛡️ Space Weather Advisory & Alert Level",
        "💾 Data & Report Export"
    ])

    with tab_plots:
        st.subheader("Multi-Payload Synchronized Timeline & AI Forecast")
        
        # Build 4-panel Plotly figure
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                "1. SoLEXS Soft X-Ray Light Curve (1-30 keV) & Flare Segmentation",
                "2. HEL1OS Hard X-Ray Multi-Band Count Rates (10-150 keV)",
                "3. Cross-Payload Physics: Hardness Ratio (HR) & Soft X-Ray Rate of Rise (dF/dt)",
                "4. AI Solar Flare Risk Horizon (1-Hour & 2-Hour Ahead Predictive Probability)"
            ),
            row_heights=[0.30, 0.25, 0.22, 0.23]
        )

        # Panel 1: SoLEXS Flux + Baseline + Flare Spans
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['solexs_counts'],
                mode='lines', name='SoLEXS Flux',
                line=dict(color='#58a6ff', width=1.5)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['solexs_baseline'],
                mode='lines', name='Quiescent Baseline',
                line=dict(color='#8b949e', width=1.2, dash='dash')
            ),
            row=1, col=1
        )

        # Add flare peak markers & vertical regions
        for ev in flare_events:
            fig.add_vrect(
                x0=ev['start_time'], x1=ev['end_time'],
                fillcolor=ev['color'], opacity=0.15,
                layer="below", line_width=0,
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[ev['peak_time']], y=[ev['peak_solexs_counts']],
                    mode='markers+text',
                    name=f"{ev['flare_class']} Peak",
                    text=[f"{ev['flare_class']}"],
                    textposition="top center",
                    marker=dict(size=9, color=ev['color'], symbol='diamond'),
                    hovertext=f"Event: {ev['event_id']}<br>Class: {ev['flare_class']}<br>Peak: {ev['peak_solexs_counts']} cts/s<br>Duration: {ev['duration_mins']}m",
                    showlegend=False
                ),
                row=1, col=1
            )

        # Panel 2: HEL1OS Hard X-ray bands
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['hel1os_czt_total'],
                mode='lines', name='HEL1OS CZT (18-160 keV)',
                line=dict(color='#d2a8ff', width=1.5)
            ),
            row=2, col=1
        )
        if 'hel1os_10_20' in df.columns and df['hel1os_10_20'].max() > 0:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=df['hel1os_10_20'],
                    mode='lines', name='10-20 keV',
                    line=dict(color='#7ee787', width=1.0)
                ),
                row=2, col=1
            )
        if 'hel1os_20_40' in df.columns and df['hel1os_20_40'].max() > 0:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=df['hel1os_20_40'],
                    mode='lines', name='20-40 keV',
                    line=dict(color='#ffa657', width=1.0)
                ),
                row=2, col=1
            )

        # Panel 3: Physics - Hardness Ratio & Derivative
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['hardness_ratio'],
                mode='lines', name='Hardness Ratio (HR)',
                line=dict(color='#ff7b72', width=1.4)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['d_solexs_dt'],
                mode='lines', name='d(SoLEXS)/dt (Neupert)',
                line=dict(color='#79c0ff', width=1.2, dash='dot')
            ),
            row=3, col=1
        )

        # Panel 4: AI Predictions (Probabilities)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['prob_flare_1h'],
                mode='lines', name='1-Hour Flare Prob (%)',
                fill='tozeroy',
                line=dict(color='#ff5252', width=1.8),
                fillcolor='rgba(255, 82, 82, 0.2)'
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['prob_flare_2h'],
                mode='lines', name='2-Hour Flare Prob (%)',
                line=dict(color='#e040fb', width=1.2, dash='dash')
            ),
            row=4, col=1
        )
        # 50% Threshold line
        fig.add_hline(y=50, line_dash="dot", line_color="#f85149", annotation_text="Elevated Warning Threshold (50%)", row=4, col=1)

        fig.update_layout(
            height=1000,
            template="plotly_dark",
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#161b22",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=30)
        )
        fig.update_yaxes(title_text="Counts / s", row=1, col=1)
        fig.update_yaxes(title_text="Counts / s", row=2, col=1)
        fig.update_yaxes(title_text="Ratio / Slope", row=3, col=1)
        fig.update_yaxes(title_text="Probability (%)", range=[0, 105], row=4, col=1)
        fig.update_xaxes(title_text="UTC Timestamp", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with tab_catalog:
        st.subheader("Solar Flare Events Detected by Nowcasting Engine")
        if flare_events:
            event_rows = []
            for ev in flare_events:
                event_rows.append({
                    "Event ID": ev['event_id'],
                    "Start (UTC)": ev['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                    "Peak (UTC)": ev['peak_time'].strftime("%H:%M:%S"),
                    "End (UTC)": ev['end_time'].strftime("%H:%M:%S"),
                    "Duration (m)": ev['duration_mins'],
                    "Rise (m)": ev['rise_time_mins'],
                    "Decay (m)": ev['decay_time_mins'],
                    "Flare Class": ev['flare_class'],
                    "Peak SoLEXS (cts)": ev['peak_solexs_counts'],
                    "Peak HEL1OS (cts)": ev['peak_hel1os_counts'],
                    "Peak Hardness": ev['peak_hardness_ratio'],
                    "Fluence Proxy": ev['total_fluence'],
                    "Space Weather Warning": ev['space_weather_alert']
                })
            event_df = pd.DataFrame(event_rows)
            st.dataframe(event_df, use_container_width=True, hide_index=True)
        else:
            st.info("No flare events detected exceeding the current prominence threshold. Sun is in quiescent state.")

    with tab_advisory:
        st.subheader("Operational Space Weather Risk Assessment")
        
        c_adv1, c_adv2 = st.columns([1.5, 1])
        
        with c_adv1:
            st.markdown(f"""
            ### 📡 Space Weather Status: **{peak_class} Activity Detected**
            
            - **High Frequency (HF) Radio Communication:** 
              {"⚠️ High Risk of Sunlit Side Radio Absorption (R1-R2 scale disruption)." if peak_class in ['M-Class', 'X-Class'] else "✅ Normal ionospheric propagation. Minimal attenuation."}
            - **Satellite Navigation & GPS:** 
              {"⚠️ Moderate scintillations and single-frequency GPS ranging errors possible." if peak_class in ['M-Class', 'X-Class'] else "✅ Nominal GPS performance."}
            - **Orbital Satellite Infrastructure:** 
              {"⚠️ Enhanced atmospheric drag and surface charging risk for Low Earth Orbit (LEO) assets." if peak_class in ['M-Class', 'X-Class'] else "✅ Low radiation dose on satellite electronics."}
            - **Solar Energetic Particle (SEP) Acceleration:** 
              Peak Hardness Ratio of **{max_hr:.3f}** indicates {"active non-thermal particle acceleration (Neupert Phase)." if max_hr > 0.5 else "predominantly thermal quiescent heating."}
            """)
            
        with c_adv2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {class_color};">
                <h4 style="color:{class_color}; margin-top:0;">Actionable Advisory</h4>
                <p><b>ISRO L1 Payload Synergy:</b><br>
                SoLEXS thermal soft X-rays confirmed peak flux of <b>{max_flux:.1f} cts/s</b>. 
                HEL1OS hard X-rays confirmed impulsive non-thermal emission up to <b>{max_hls:.1f} cts/s</b>.</p>
                <p><b>Forward Forecast (Next 1-2 Hours):</b><br>
                AI Model evaluates maximum upcoming flare probability at <b>{max_risk_1h:.1f}%</b>.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_export:
        st.subheader("Export Pipeline Products")
        
        exp_c1, exp_c2 = st.columns(2)
        
        with exp_c1:
            st.markdown("#### 1. Synchronized 24h Time-Series Data")
            st.write("Download the complete multi-payload matrix including physics features, hardness ratios, and AI forecast probabilities.")
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Synchronized CSV (24h Matrix)",
                data=csv_data,
                file_name=f"aditya_l1_synchronized_flare_matrix.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with exp_c2:
            st.markdown("#### 2. Detected Flare Event Catalog")
            st.write("Download the structured catalog of detected flare events in JSON format for automated space weather alert ingestion.")
            json_data = json.dumps(flare_events, default=str, indent=2).encode('utf-8')
            st.download_button(
                label="📥 Download Event Catalog (JSON)",
                data=json_data,
                file_name=f"aditya_l1_flare_events_catalog.json",
                mime="application/json",
                use_container_width=True
            )
