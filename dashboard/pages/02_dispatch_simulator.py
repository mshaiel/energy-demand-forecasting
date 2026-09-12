"""
⚡ Page 2: Interactive Grid Stress-Tester & Dispatch Simulator
Interactive decision-support sandbox for power grid operators and energy analysts.
Simulates extreme weather anomalies, renewable deficit, and peak shaving demand response.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    apply_custom_css,
    get_forecast_df,
    get_fold_metadata,
    render_sidebar,
)
from src.visualization.plots import create_dispatch_simulation_plot

st.set_page_config(
    page_title="Dispatch Simulator | Grid Intelligence",
    page_icon="⚡",
    layout="wide",
)

apply_custom_css()
render_sidebar()

# Header
st.markdown(
    """
    <div class="console-header">
        <div>
            <div class="console-title">⚡ Interactive Grid Stress-Tester & Dispatch Simulator</div>
            <div class="console-subtitle">Simulate real-time heatwave anomalies, peak capacity shortfalls, and financial savings from commercial demand response.</div>
        </div>
        <div>
            <span class="chip chip-cyan"><span class="pulse-dot"></span> SIMULATOR ENGINE ACTIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

df = get_forecast_df()
folds = get_fold_metadata()
df_24h = df[df["horizon"] == "primary"].copy()

# 1. Interactive Control Panel
with st.container():
    st.markdown("#### 🎛️ Scenario Configuration & Stress Parameters")
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(4)

    with ctrl_c1:
        selected_fold = st.selectbox(
            "Backtest Evaluation Period",
            options=[1, 2, 3, 4, 5],
            index=4,
            format_func=lambda x: f"Fold {x} (Peak Summer Horizon)" if x == 5 else f"Fold {x}",
        )

    with ctrl_c2:
        selected_model = st.selectbox(
            "Forecasting Core Model",
            options=["chronos", "lstm", "xgboost", "sarimax"],
            index=0,
            format_func=lambda x: {
                "chronos": "Chronos-T5 (Gen 4)",
                "lstm": "LSTM (Gen 3)",
                "xgboost": "XGBoost (Gen 2)",
                "sarimax": "SARIMAX (Gen 1)",
            }.get(x, x),
        )

    with ctrl_c3:
        temp_shock = st.slider(
            "🌡️ Heatwave Anomaly (°C)",
            min_value=0.0,
            max_value=6.0,
            value=3.5,
            step=0.5,
            help="Simulates sudden heatwave. Each +1°C drives ~3.8% cooling load increase across PJM.",
        )

    with ctrl_c4:
        peak_shaving_pct = st.slider(
            "📉 Demand Response Shaving (%)",
            min_value=0,
            max_value=15,
            value=8,
            step=1,
            help="Targeted peak-hour commercial curtailment applied during 14:00-19:00.",
        )

# Filter baseline data
sub_df = df_24h[(df_24h["fold_id"] == selected_fold) & (df_24h["model"] == selected_model)].sort_values("timestamp")
if sub_df.empty:
    sub_df = df_24h[df_24h["model"] == selected_model].iloc[:24].sort_values("timestamp")

timestamps = sub_df["timestamp"]
baseline_actual = sub_df["actual"].values
baseline_q50 = sub_df["q50"].values

# 2. Simulation Math Engine
# Cooling load factor: peak hours (11:00 to 20:00) sensitive to temp anomaly
hours = timestamps.dt.hour.values
is_peak_cooling = ((hours >= 11) & (hours <= 20)).astype(float)
cooling_multiplier = 1.0 + (temp_shock * 0.038) * (0.5 + 0.5 * is_peak_cooling)
simulated_q50 = baseline_q50 * cooling_multiplier

# Demand Response peak shaving (applied between 14:00 and 19:00)
is_dr_window = ((hours >= 14) & (hours <= 19)).astype(float)
peak_shaved_q50 = simulated_q50 * (1.0 - (peak_shaving_pct / 100.0) * is_dr_window)

# Grid Capacity Thresholds (PJM Eastern zone approx capacity: 48,000 MW)
grid_capacity_mw = 48000.0
base_peak = float(np.max(baseline_q50))
stressed_peak = float(np.max(simulated_q50))
shaved_peak = float(np.max(peak_shaved_q50))
peak_reduction_mw = stressed_peak - shaved_peak

# Reserve Margin: (Capacity - Stressed Peak) / Stressed Peak
reserve_margin_stressed = ((grid_capacity_mw - stressed_peak) / stressed_peak) * 100.0
reserve_margin_shaved = ((grid_capacity_mw - shaved_peak) / shaved_peak) * 100.0

# Financial Savings Calculation:
# PJM On-Peak LMP wholesale electricity rate estimated at ~$85 / MWh
# Peak shaving occurs across 6 hours
mwh_saved = float(np.sum(simulated_q50 - peak_shaved_q50))
cost_avoided_usd = mwh_saved * 85.0

# 3. Telemetry KPI Cards
st.markdown("<br>", unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    delta_mw = stressed_peak - base_peak
    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">
                <span>STRESSED PEAK LOAD</span>
                <span class="chip chip-warning">+{delta_mw:,.0f} MW</span>
            </div>
            <div class="telemetry-value" style="color: #f59e0b;">{stressed_peak:,.0f} <span style="font-size: 0.9rem; color: #94a3b8;">MW</span></div>
            <div class="telemetry-meta">
                <span>Base: <b>{base_peak:,.0f} MW</b></span>
                <span style="margin-left: auto;">+{temp_shock}°C shock</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    if reserve_margin_stressed < 8.0:
        status_html = '<span class="chip chip-critical"><span class="pulse-dot"></span> EMERGENCY</span>'
        res_color = "#f43f5e"
    elif reserve_margin_stressed < 14.0:
        status_html = '<span class="chip chip-warning"><span class="pulse-dot"></span> PEAK ALERT</span>'
        res_color = "#f59e0b"
    else:
        status_html = '<span class="chip chip-normal"><span class="pulse-dot"></span> NORMAL</span>'
        res_color = "#10b981"

    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">
                <span>NET RESERVE MARGIN</span>
                {status_html}
            </div>
            <div class="telemetry-value" style="color: {res_color};">{reserve_margin_stressed:.1f}%</div>
            <div class="telemetry-meta">
                <span>Safe Threshold: <b>15.0%</b></span>
                <span style="margin-left: auto;">Cap: 48,000 MW</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">
                <span>PEAK CAPACITY SHAVED</span>
                <span class="chip chip-normal">-{peak_shaving_pct}% DR</span>
            </div>
            <div class="telemetry-value" style="color: #10b981;">{peak_reduction_mw:,.0f} <span style="font-size: 0.9rem; color: #94a3b8;">MW</span></div>
            <div class="telemetry-meta">
                <span>Post-DR Reserve: <b>{reserve_margin_shaved:.1f}%</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">
                <span>ESTIMATED COST AVOIDED</span>
                <span class="chip chip-cyan">PJM LMP @ $85</span>
            </div>
            <div class="telemetry-value" style="color: #00d4ff;">${cost_avoided_usd:,.0f}</div>
            <div class="telemetry-meta">
                <span>Energy Curtailed: <b>{mwh_saved:,.0f} MWh</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 4. Main Simulation Chart
sim_fig = create_dispatch_simulation_plot(
    timestamps=timestamps,
    actual=baseline_actual,
    baseline_q50=baseline_q50,
    simulated_q50=simulated_q50,
    peak_shaved_q50=peak_shaved_q50,
    grid_capacity_mw=grid_capacity_mw,
)
st.plotly_chart(sim_fig, use_container_width=True)

# 5. Interactive Hourly Schedule & CSV Export
st.markdown("---")
table_c1, table_c2 = st.columns([3, 1])

with table_c1:
    st.markdown("#### 📋 24-Hour Operational Dispatch Table (Simulated)")
    sim_export_df = pd.DataFrame({
        "Timestamp": timestamps.dt.strftime("%Y-%m-%d %H:00"),
        "Hour": hours,
        "Base Forecast (MW)": np.round(baseline_q50, 0).astype(int),
        "Stressed Load (MW)": np.round(simulated_q50, 0).astype(int),
        "Post-DR Shaved (MW)": np.round(peak_shaved_q50, 0).astype(int),
        "Capacity Reserve (MW)": np.round(grid_capacity_mw - peak_shaved_q50, 0).astype(int),
    })
    st.dataframe(sim_export_df, use_container_width=True, hide_index=True)

with table_c2:
    st.markdown("#### 📥 Data Export")
    st.markdown("Download this simulated hourly dispatch schedule directly for power trading or grid operations review.")
    csv_data = sim_export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Schedule (CSV)",
        data=csv_data,
        file_name=f"pjm_dispatch_simulation_fold{selected_fold}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 10px;">
            * Calculations based on PJM wholesale day-ahead market settlement standards.
        </div>
        """,
        unsafe_allow_html=True,
    )
