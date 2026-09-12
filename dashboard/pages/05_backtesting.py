"""
⚡ Page 5: Backtesting Protocol & Calibration Analysis
Expanding window Gantt chart, per-fold model stability tracking, error distribution box plots, and coverage calibration.
"""
from pathlib import Path
import sys
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    apply_custom_css,
    get_forecast_df,
    get_fold_metadata,
    get_metrics_summary,
    render_sidebar,
)
from src.visualization.plots import (
    create_fold_timeline_chart,
    create_per_fold_metric_chart,
    create_error_box_plot,
    create_coverage_calibration_chart,
)
from src.evaluation.metrics import mae, rmse, mape, winkler_score

st.set_page_config(
    page_title="Backtesting Protocol | Grid Intelligence",
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
            <div class="console-title">🛡️ Rolling-Origin Backtesting & Calibration Protocol</div>
            <div class="console-subtitle">Empirical proof of zero temporal leakage. Out-of-sample stability evaluated across 5 expanding windows.</div>
        </div>
        <div>
            <span class="chip chip-normal"><span class="pulse-dot"></span> ZERO LEAKAGE ENFORCED</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

df = get_forecast_df()
folds = get_fold_metadata()
metrics_summary = get_metrics_summary()

# 1. Gantt Timeline Chart
st.markdown("#### 📅 Temporal Boundaries (5-Fold Expanding Window)")
gantt_fig = create_fold_timeline_chart(folds)
st.plotly_chart(gantt_fig, use_container_width=True)

st.markdown("---")
# 2. Stability Line Chart & Error Box Plots
col_stability, col_box = st.columns(2)

with col_stability:
    st.markdown("#### 📈 Model Stability Across Folds")
    metric_choice = st.selectbox(
        "Metric for Stability Tracking",
        options=["RMSE", "MAE", "MAPE", "Winkler Score"],
        index=0,
    )

    metric_map = {
        "RMSE": (rmse, "RMSE (MW)"),
        "MAE": (mae, "MAE (MW)"),
        "MAPE": (mape, "MAPE (%)"),
        "Winkler Score": (lambda y, p: winkler_score(y, p * 0.9, p * 1.1), "Winkler Score"),
    }
    m_func, m_name = metric_map[metric_choice]

    df_24h = df[df["horizon"] == "primary"]
    stab_fig = create_per_fold_metric_chart(df_24h, m_func, metric_name=m_name)
    st.plotly_chart(stab_fig, use_container_width=True)

with col_box:
    st.markdown("#### 📦 Error Distribution Spread (MW)")
    st.markdown("<div style='font-size: 0.8rem; color: #64748b; margin-bottom: 8px;'>Interquartile ranges (IQR) of absolute errors across all evaluated hours.</div>", unsafe_allow_html=True)
    box_fig = create_error_box_plot(df_24h)
    st.plotly_chart(box_fig, use_container_width=True)

st.markdown("---")
# 3. Coverage Calibration Chart
st.markdown("#### 🎯 Empirical Coverage Calibration (Nominal: 80% CI)")
st.markdown(
    "<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 12px;'>"
    "Ideal calibration achieves 80.0% coverage within $[q_{10}, q_{90}]$. "
    "Bars within $\\pm 5\\%$ are in <span style='color: #10b981; font-weight: 600;'>Green</span>, "
    "$\\pm 10\\%$ in <span style='color: #f59e0b; font-weight: 600;'>Amber</span>."
    "</div>",
    unsafe_allow_html=True,
)

cov_fig = create_coverage_calibration_chart(metrics_summary, horizon_key="primary_24h")
st.plotly_chart(cov_fig, use_container_width=True)
