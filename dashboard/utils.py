"""Shared utilities and data caching for Streamlit SCADA dashboard."""
from pathlib import Path
import sys
import json
import pandas as pd
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STYLE_PATH = Path(__file__).resolve().parent / "assets" / "style.css"
BRIDGE_DIR = PROJECT_ROOT / "data" / "bridge"


def apply_custom_css():
    """Inject custom SCADA dark-mode CSS overrides."""
    if STYLE_PATH.exists():
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def get_forecast_df() -> pd.DataFrame:
    """Cached loader for core forecast results DataFrame."""
    csv_path = BRIDGE_DIR / "forecast_results.csv"
    if not csv_path.exists():
        st.error(f"forecast_results.csv not found in {BRIDGE_DIR.resolve()}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df


@st.cache_data
def get_metrics_summary() -> dict:
    """Cached loader for aggregate metrics dictionary."""
    path = BRIDGE_DIR / "metrics_summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_feature_importance() -> dict:
    """Cached loader for XGBoost feature gain and SHAP scores."""
    path = BRIDGE_DIR / "feature_importance.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_fold_metadata() -> list:
    """Cached loader for 5-fold temporal boundaries."""
    path = BRIDGE_DIR / "fold_metadata.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_sidebar():
    """Render common sidebar telemetry and navigation context."""
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.5rem;">⚡</span>
                <div>
                    <div style="font-weight: 800; font-size: 1.1rem; color: #00d4ff; letter-spacing: -0.02em;">GRID INTELLIGENCE</div>
                    <div style="font-size: 0.72rem; color: #9ca3af; font-family: 'JetBrains Mono', monospace;">SCADA FORECAST OS v1.0</div>
                </div>
            </div>
            <div style="margin-top: 12px;">
                <span class="status-pill status-live">TELEMETRY ONLINE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 Grid Context")
    st.sidebar.markdown(
        """
        * **Regional Entity**: PJM Interconnection (Eastern US)
        * **Dataset**: `PJME_hourly.csv` (140k+ obs)
        * **Evaluation**: 5-Fold Expanding Window
        * **Nominal Coverage**: 80.0% ($q_{10}$ to $q_{90}$)
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace; line-height: 1.6;">
            <b>Operational Modules</b><br>
            01 • Executive Overview<br>
            02 • Dispatch Simulator<br>
            03 • 24h Day-Ahead Forecast<br>
            04 • 168h Extended Horizon<br>
            05 • Backtesting Protocol<br>
            06 • Feature Intelligence<br>
            07 • Mathematical Theory
        </div>
        """,
        unsafe_allow_html=True,
    )
