"""
⚡ Page 4: Backtesting & Methodological Validation
5-Fold expanding window timeline, per-fold stability, error distributions, and calibration bars.
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

st.markdown("## 🛡️ Expanding-Window Backtesting Protocol")
st.markdown(
    "Demonstrating zero temporal leakage via rolling-origin evaluation. The training window expands continuously while contiguous, non-overlapping test windows evaluate out-of-sample generalization."
)

df = get_forecast_df()
folds = get_fold_metadata()
metrics_summary = get_metrics_summary()

# 1. Gantt Timeline Chart
st.markdown("### 📅 Temporal Window Architecture (No Lookahead)")
gantt_fig = create_fold_timeline_chart(folds)
st.plotly_chart(gantt_fig, use_container_width=True)

st.markdown("---")
# 2. Stability Line Chart & Error Box Plots
col_stability, col_box = st.columns(2)

with col_stability:
    st.markdown("### 📈 Model Stability Across Folds")
    metric_choice = st.selectbox(
        "Select Metric for Stability Tracking",
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
    st.markdown("### 📦 Error Distribution Spread (MW)")
    st.markdown("Interquartile ranges (IQR) of absolute errors across all backtested timestamps.")
    box_fig = create_error_box_plot(df_24h)
    st.plotly_chart(box_fig, use_container_width=True)

st.markdown("---")
# 3. Coverage Calibration Chart
st.markdown("### 🎯 Empirical Coverage Calibration (Nominal: 80%)")
st.markdown(
    "Well-calibrated models produce intervals where approximately 80% of ground truth values fall within $[q_{10}, q_{90}]$. "
    "Bars within $\\pm 5\\%$ are highlighted in <span style='color: #10b981; font-weight: 600;'>Green</span>.",
    unsafe_allow_html=True,
)

cov_fig = create_coverage_calibration_chart(metrics_summary, horizon_key="primary_24h")
st.plotly_chart(cov_fig, use_container_width=True)
