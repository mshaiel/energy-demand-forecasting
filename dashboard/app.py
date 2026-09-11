"""
⚡ Grid Intelligence: Probabilistic Energy Demand Forecasting
Main Streamlit Entrypoint
"""
from pathlib import Path
import sys
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    apply_custom_css,
    get_forecast_df,
    get_metrics_summary,
    get_feature_importance,
    get_fold_metadata,
)

st.set_page_config(
    page_title="Grid Intelligence | Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Pre-warm cached datasets into session state
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = get_forecast_df()
if "metrics" not in st.session_state:
    st.session_state.metrics = get_metrics_summary()
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = get_feature_importance()
if "fold_metadata" not in st.session_state:
    st.session_state.fold_metadata = get_fold_metadata()

# Forward directly to Page 1: Overview
try:
    st.switch_page("pages/01_overview.py")
except Exception:
    # Fallback if switch_page is in older context
    st.markdown(
        """
        <div style="padding: 20px; text-align: center;">
            <h1 style="color: #00d4ff;">⚡ Grid Intelligence SCADA Forecast OS</h1>
            <p style="color: #9ca3af;">Please select a module from the sidebar navigation to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
