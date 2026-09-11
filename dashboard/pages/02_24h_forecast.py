"""
⚡ Page 2: 24-Hour Day-Ahead Forecast Explorer
Interactive fold-by-fold inspection of predictions and uncertainty envelopes.
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
    get_fold_metadata,
    render_sidebar,
)
from src.visualization.plots import create_forecast_plot
from src.evaluation.metrics import compute_all_metrics

st.set_page_config(
    page_title="24h Forecast Explorer | Grid Intelligence",
    page_icon="⚡",
    layout="wide",
)

apply_custom_css()
render_sidebar()

st.markdown("## 🔍 24-Hour Day-Ahead Forecast Explorer")
st.markdown(
    "Explore individual fold predictions, evaluate point forecasts ($q_{50}$), and inspect the calibrated 80% confidence interval ribbons ($q_{10}$ to $q_{90}$)."
)

df = get_forecast_df()
folds = get_fold_metadata()

# Filter to primary 24h horizon
df_24h = df[df["horizon"] == "primary"].copy()

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Display Controls")
fold_id = st.sidebar.selectbox(
    "Select Backtest Fold",
    options=[1, 2, 3, 4, 5],
    index=4,  # default to Fold 5
    format_func=lambda x: f"Fold {x} (Final Test Period)" if x == 5 else f"Fold {x}",
)

models_available = ["chronos", "lstm", "xgboost", "sarimax"]
selected_models = st.sidebar.multiselect(
    "Models to Overlay",
    options=models_available,
    default=models_available,
    format_func=lambda x: {
        "chronos": "Chronos-T5 (Gen 4)",
        "lstm": "LSTM (Gen 3)",
        "xgboost": "XGBoost (Gen 2)",
        "sarimax": "SARIMAX (Gen 1)",
    }.get(x, x),
)

show_uncertainty = st.sidebar.toggle("Show Uncertainty Bands (80% CI)", value=True)
show_actual = st.sidebar.toggle("Show Ground Truth Demand", value=True)

# Slicing for selected fold
sub_df = df_24h[df_24h["fold_id"] == fold_id]

# Extract Fold Date Range Info
fold_info = next((f for f in folds if f["fold_id"] == fold_id), None)
date_str = ""
if fold_info:
    date_str = f"Evaluation Window: <b>{fold_info['test_start'][:10]}</b> to <b>{fold_info['test_end'][:10]}</b> | Train Cutoff: <b>{fold_info['train_end'][:10]}</b>"
    st.markdown(f"<div style='margin-bottom: 12px; color: #9ca3af; font-size: 0.9rem;'>{date_str}</div>", unsafe_allow_html=True)

# Main Plotly Forecast Figure
fig = create_forecast_plot(
    sub_df,
    models=selected_models,
    show_actual=show_actual,
    show_uncertainty=show_uncertainty,
    title=f"Fold {fold_id} Day-Ahead Hourly Demand Profile (MW)",
)
st.plotly_chart(fig, use_container_width=True)

# Per-Fold Performance Metrics Breakdown
st.markdown(f"### 📊 Fold {fold_id} Model Performance Scorecard")

fold_metrics = []
for m in selected_models:
    m_data = sub_df[sub_df["model"] == m]
    if not m_data.empty:
        met = compute_all_metrics(m_data["actual"], m_data["q10"], m_data["q50"], m_data["q90"])
        fold_metrics.append({
            "Model": m.upper() if m != "chronos" else "Chronos-T5",
            "MAE (MW)": round(met["mae"], 1),
            "RMSE (MW)": round(met["rmse"], 1),
            "MAPE (%)": f"{met['mape']:.2f}%",
            "Winkler Score": round(met["winkler_80"], 1),
            "80% Coverage": f"{met['coverage_80']*100:.1f}%",
            "_raw_rmse": met["rmse"],
        })

if fold_metrics:
    score_df = pd.DataFrame(fold_metrics).sort_values("_raw_rmse")
    best_model = score_df.iloc[0]["Model"]
    score_df = score_df.drop(columns=["_raw_rmse"])

    col_score, col_badge = st.columns([3, 1])
    with col_score:
        st.dataframe(score_df, use_container_width=True, hide_index=True)
    with col_badge:
        st.markdown(
            f"""
            <div class="scada-card" style="text-align: center; border-color: #00d4ff;">
                <span class="scada-badge badge-gen4">FOLD {fold_id} WINNER</span>
                <div style="font-size: 1.4rem; font-weight: 800; color: #00d4ff; margin: 10px 0;">{best_model}</div>
                <div style="font-size: 0.8rem; color: #9ca3af;">Lowest overall RMSE & highest interval sharpness on this window.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
