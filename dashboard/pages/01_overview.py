"""
⚡ Page 1: Executive Overview & Benchmark Leaderboard
Landing page with headline KPIs, full 4-generation leaderboard, and radar chart.
"""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    apply_custom_css,
    get_forecast_df,
    get_metrics_summary,
    render_sidebar,
)
from src.visualization.plots import create_radar_chart

st.set_page_config(
    page_title="Executive Overview | Grid Intelligence",
    page_icon="⚡",
    layout="wide",
)

apply_custom_css()
render_sidebar()

# 1. Hero Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">⚡ Grid Intelligence: Probabilistic Demand Forecasting</div>
        <div class="hero-subtitle">
            Rigorous empirical benchmark across <b>4 Generations of Time-Series Architecture</b> on the 
            PJM Interconnection (Eastern US Grid). Evaluated with rolling-origin expanding window backtesting (5 folds) 
            and calibrated quantile uncertainty intervals (80% CI).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics = get_metrics_summary()
df = get_forecast_df()

# 2. KPI Cards Row (4 Generations)
col1, col2, col3, col4 = st.columns(4)

# Gen 4: Chronos-T5
c_met = metrics.get("chronos_primary_24h", {})
with col1:
    st.markdown(
        f"""
        <div class="scada-card">
            <span class="scada-badge badge-gen4">GEN 4 • FOUNDATION</span>
            <div class="scada-metric-title">Amazon Chronos-T5</div>
            <div class="scada-metric-value">{c_met.get('mape', 5.24):.2f}% <span style="font-size: 0.9rem; color: #9ca3af;">MAPE</span></div>
            <div class="scada-metric-sub">
                <span>RMSE: <b>{c_met.get('rmse', 3351.5):,.0f} MW</b></span>
                <span style="margin-left: auto; color: #00d4ff;">80% Cov: {c_met.get('coverage_80', 0.767)*100:.1f}%</span>
            </div>
            <div style="margin-top: 8px; font-size: 0.72rem; color: #6b7280; font-family: 'JetBrains Mono', monospace;">
                Zero-shot • No feature eng
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Gen 3: LSTM
l_met = metrics.get("lstm_primary_24h", {})
with col2:
    st.markdown(
        f"""
        <div class="scada-card">
            <span class="scada-badge badge-gen3">GEN 3 • DEEP LEARNING</span>
            <div class="scada-metric-title">Multivariate LSTM</div>
            <div class="scada-metric-value" style="color: #a78bfa;">{l_met.get('mape', 4.95):.2f}% <span style="font-size: 0.9rem; color: #9ca3af;">MAPE</span></div>
            <div class="scada-metric-sub">
                <span>RMSE: <b>{l_met.get('rmse', 2954.1):,.0f} MW</b></span>
                <span style="margin-left: auto; color: #a78bfa;">80% Cov: {l_met.get('coverage_80', 0.875)*100:.1f}%</span>
            </div>
            <div style="margin-top: 8px; font-size: 0.72rem; color: #6b7280; font-family: 'JetBrains Mono', monospace;">
                MC Dropout Uncertainty
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Gen 2: XGBoost
x_met = metrics.get("xgboost_primary_24h", {})
with col3:
    st.markdown(
        f"""
        <div class="scada-card">
            <span class="scada-badge badge-gen2">GEN 2 • BOOSTED TREES</span>
            <div class="scada-metric-title">Quantile XGBoost</div>
            <div class="scada-metric-value" style="color: #fbbf24;">{x_met.get('mape', 5.42):.2f}% <span style="font-size: 0.9rem; color: #9ca3af;">MAPE</span></div>
            <div class="scada-metric-sub">
                <span>RMSE: <b>{x_met.get('rmse', 2658.2):,.0f} MW</b></span>
                <span style="margin-left: auto; color: #fbbf24;">80% Cov: {x_met.get('coverage_80', 0.858)*100:.1f}%</span>
            </div>
            <div style="margin-top: 8px; font-size: 0.72rem; color: #6b7280; font-family: 'JetBrains Mono', monospace;">
                35+ Features • Best Sharpness
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Gen 1: SARIMAX
s_met = metrics.get("sarimax_primary_24h", {})
with col4:
    st.markdown(
        f"""
        <div class="scada-card">
            <span class="scada-badge badge-gen1">GEN 1 • CLASSICAL</span>
            <div class="scada-metric-title">SARIMAX Baseline</div>
            <div class="scada-metric-value" style="color: #d1d5db;">{s_met.get('mape', 13.71):.2f}% <span style="font-size: 0.9rem; color: #9ca3af;">MAPE</span></div>
            <div class="scada-metric-sub">
                <span>RMSE: <b>{s_met.get('rmse', 5505.7):,.0f} MW</b></span>
                <span style="margin-left: auto; color: #9ca3af;">80% Cov: {s_met.get('coverage_80', 0.225)*100:.1f}%</span>
            </div>
            <div style="margin-top: 8px; font-size: 0.72rem; color: #6b7280; font-family: 'JetBrains Mono', monospace;">
                (2,1,2)(1,1,1)7 Baseline
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 3. Leaderboard and Multi-Metric Radar Chart
left_col, right_col = st.columns([1.3, 1.0])

with left_col:
    st.markdown("### 🏆 24-Hour Benchmark Leaderboard (5-Fold Mean)")
    
    horizon_choice = st.radio(
        "Forecast Evaluation Horizon",
        options=["primary_24h", "extended_168h"],
        format_func=lambda x: "Primary 24-Hour (Day-Ahead)" if "24h" in x else "Extended 168-Hour (7-Day Week Ahead)",
        horizontal=True,
    )

    models_list = ["lstm", "chronos", "xgboost", "sarimax"]
    rows = []
    for m in models_list:
        k = f"{m}_{horizon_choice}"
        dat = metrics.get(k, {})
        gen_tag = {
            "chronos": "Gen 4 Foundation",
            "lstm": "Gen 3 Deep Learning",
            "xgboost": "Gen 2 Boosted Trees",
            "sarimax": "Gen 1 Classical",
        }.get(m, m)
        rows.append({
            "Model": m.upper() if m != "chronos" else "Chronos-T5",
            "Architecture": gen_tag,
            "MAE (MW)": dat.get("mae", 0.0),
            "RMSE (MW)": dat.get("rmse", 0.0),
            "MAPE (%)": f"{dat.get('mape', 0.0):.2f}%",
            "Winkler Score": f"{dat.get('winkler_80', 0.0):,.1f}",
            "80% Coverage": f"{dat.get('coverage_80', 0.0)*100:.1f}%",
        })

    leader_df = pd.DataFrame(rows).sort_values("MAE (MW)")
    leader_df.insert(0, "Rank", [f"#{i+1}" for i in range(len(leader_df))])

    st.dataframe(
        leader_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 4px;">
            * <b>Winkler Score</b> evaluates interval sharpness and penalizes quantile misses (proper scoring rule). Lower is superior.
            * <b>80% Coverage</b> represents empirical observation containment within the [q10, q90] prediction envelope.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown("### 🕸️ Multi-Dimensional Comparison")
    radar_fig = create_radar_chart(metrics, horizon_key=horizon_choice)
    st.plotly_chart(radar_fig, use_container_width=True)

# 4. Dataset Provenance Card
st.markdown("---")
st.markdown("### 🗄️ Dataset Provenance & Architectural Boundaries")
info_c1, info_c2, info_c3, info_c4 = st.columns(4)

with info_c1:
    st.markdown(
        """
        <div class="telemetry-box">
            <b>GRID NODE</b><br>
            PJM Interconnection (PJME)<br>
            Eastern US Regional Grid
        </div>
        """,
        unsafe_allow_html=True,
    )
with info_c2:
    st.markdown(
        """
        <div class="telemetry-box">
            <b>DATA RANGE</b><br>
            2002-01-01 to 2018-08-03<br>
            140,256 Continuous Hours
        </div>
        """,
        unsafe_allow_html=True,
    )
with info_c3:
    st.markdown(
        """
        <div class="telemetry-box">
            <b>BACKTEST PROTOCOL</b><br>
            5-Fold Expanding Window<br>
            Zero Temporal Leakage (Shift ≥24h)
        </div>
        """,
        unsafe_allow_html=True,
    )
with info_c4:
    st.markdown(
        """
        <div class="telemetry-box">
            <b>DEPLOYMENT FOOTPRINT</b><br>
            Bridge Deliverables: &lt; 300 KB<br>
            Client Load Time: &lt; 50 ms
        </div>
        """,
        unsafe_allow_html=True,
    )
