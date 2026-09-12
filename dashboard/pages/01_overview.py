"""
⚡ Page 1: Executive Overview & Benchmark Leaderboard
Institutional SCADA landing console with clean telemetry cards, 4-generation leaderboard, and radar comparison.
"""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

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

# Console Header
st.markdown(
    """
    <div class="console-header">
        <div>
            <div class="console-title">⚡ PJM Interconnection: Probabilistic Demand Forecasting</div>
            <div class="console-subtitle">4-Generation architectural benchmark on Eastern US regional grid demand (140k+ hours, 2002–2018).</div>
        </div>
        <div>
            <span class="chip chip-normal"><span class="pulse-dot"></span> 5-FOLD EXPANDING WINDOW</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics = get_metrics_summary()
df = get_forecast_df()

# 1. High-Density Telemetry Metric Row
c1, c2, c3, c4 = st.columns(4)

# Gen 4: Chronos-T5
c_met = metrics.get("chronos_primary_24h", {})
with c1:
    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">
                <span>AMAZON CHRONOS-T5</span>
                <span class="chip chip-cyan">GEN 4 FOUNDATION</span>
            </div>
            <div class="telemetry-value" style="color: #00d4ff;">{c_met.get('mape', 5.24):.2f}% <span style="font-size: 0.85rem; color: #64748b;">MAPE</span></div>
            <div class="telemetry-meta">
                <span>RMSE: <b>{c_met.get('rmse', 3351.5):,.0f} MW</b></span>
                <span style="margin-left: auto;">80% Cov: <b>{c_met.get('coverage_80', 0.767)*100:.1f}%</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Gen 3: LSTM
l_met = metrics.get("lstm_primary_24h", {})
with c2:
    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">
                <span>MULTIVARIATE LSTM</span>
                <span class="chip" style="background: rgba(139,92,246,0.15); color: #a78bfa; border: 1px solid rgba(139,92,246,0.3);">GEN 3 DEEP LEARNING</span>
            </div>
            <div class="telemetry-value" style="color: #a78bfa;">{l_met.get('mape', 4.95):.2f}% <span style="font-size: 0.85rem; color: #64748b;">MAPE</span></div>
            <div class="telemetry-meta">
                <span>RMSE: <b>{l_met.get('rmse', 2954.1):,.0f} MW</b></span>
                <span style="margin-left: auto;">80% Cov: <b>{l_met.get('coverage_80', 0.875)*100:.1f}%</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Gen 2: XGBoost
x_met = metrics.get("xgboost_primary_24h", {})
with c3:
    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">
                <span>QUANTILE XGBOOST</span>
                <span class="chip chip-warning">GEN 2 BOOSTING</span>
            </div>
            <div class="telemetry-value" style="color: #f59e0b;">{x_met.get('mape', 5.42):.2f}% <span style="font-size: 0.85rem; color: #64748b;">MAPE</span></div>
            <div class="telemetry-meta">
                <span>RMSE: <b>{x_met.get('rmse', 2658.2):,.0f} MW</b></span>
                <span style="margin-left: auto;">80% Cov: <b>{x_met.get('coverage_80', 0.858)*100:.1f}%</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Gen 1: SARIMAX
s_met = metrics.get("sarimax_primary_24h", {})
with c4:
    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">
                <span>SARIMAX BASELINE</span>
                <span class="chip" style="background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3);">GEN 1 CLASSICAL</span>
            </div>
            <div class="telemetry-value" style="color: #94a3b8;">{s_met.get('mape', 13.71):.2f}% <span style="font-size: 0.85rem; color: #64748b;">MAPE</span></div>
            <div class="telemetry-meta">
                <span>RMSE: <b>{s_met.get('rmse', 5505.7):,.0f} MW</b></span>
                <span style="margin-left: auto;">80% Cov: <b>{s_met.get('coverage_80', 0.225)*100:.1f}%</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 2. Benchmark Leaderboard & Radar Chart
col_table, col_radar = st.columns([1.35, 1.0])

with col_table:
    st.markdown("#### 🏆 Out-of-Sample Leaderboard (5-Fold Expanding Window)")
    
    horizon_opt = st.radio(
        "Evaluation Horizon",
        options=["primary_24h", "extended_168h"],
        format_func=lambda x: "Primary 24-Hour (Day-Ahead)" if "24h" in x else "Extended 168-Hour (7-Day Week Ahead)",
        horizontal=True,
    )

    models_list = ["lstm", "chronos", "xgboost", "sarimax"]
    rows = []
    for m in models_list:
        k = f"{m}_{horizon_opt}"
        dat = metrics.get(k, {})
        gen_tag = {
            "chronos": "Gen 4 Foundation",
            "lstm": "Gen 3 Deep Learning",
            "xgboost": "Gen 2 Boosted Trees",
            "sarimax": "Gen 1 Classical Baseline",
        }.get(m, m)
        rows.append({
            "Model": m.upper() if m != "chronos" else "Chronos-T5",
            "Architecture": gen_tag,
            "MAE (MW)": round(dat.get("mae", 0.0), 1),
            "RMSE (MW)": round(dat.get("rmse", 0.0), 1),
            "MAPE": f"{dat.get('mape', 0.0):.2f}%",
            "Winkler Score": f"{dat.get('winkler_80', 0.0):,.0f}",
            "80% Coverage": f"{dat.get('coverage_80', 0.0)*100:.1f}%",
        })

    leader_df = pd.DataFrame(rows).sort_values("MAE (MW)")
    leader_df.insert(0, "Rank", [f"#{i+1}" for i in range(len(leader_df))])

    st.dataframe(leader_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 6px; line-height: 1.4;">
            • <b>Winkler Score</b> evaluates proper interval scoring: sharp intervals with minimal tail breach penalties.<br>
            • <b>80% Coverage</b> measures percentage of observations strictly within the [q10, q90] prediction band.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_radar:
    st.markdown("#### 🕸️ Multi-Attribute Tradeoff Surface")
    radar_fig = create_radar_chart(metrics, horizon_key=horizon_opt)
    st.plotly_chart(radar_fig, use_container_width=True)

# 3. Interactive Call-to-Action to Simulator
st.markdown("---")
st.markdown(
    """
    <div style="background: #0e1526; border: 1px solid #1e293b; border-radius: 8px; padding: 18px 22px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
        <div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9;">⚡ Want to test extreme scenarios?</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">Simulate real-time heatwave anomalies, peak capacity deficits, and commercial demand response in the Dispatch Simulator.</div>
        </div>
        <div>
            <a href="/dispatch_simulator" target="_self" style="text-decoration: none;">
                <span style="background: #00d4ff; color: #080c14; font-weight: 600; font-size: 0.85rem; padding: 8px 16px; border-radius: 6px; display: inline-block;">
                    Open Dispatch Simulator →
                </span>
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
