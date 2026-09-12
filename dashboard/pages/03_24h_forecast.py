"""
⚡ Page 3: 24-Hour Day-Ahead Forecast Explorer
Interactive fold-by-fold inspection of predictions, uncertainty envelopes, and peak load timestamps.
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

# Console Header
st.markdown(
    """
    <div class="console-header">
        <div>
            <div class="console-title">🔍 24-Hour Day-Ahead Forecast Explorer</div>
            <div class="console-subtitle">Out-of-sample hourly predictions, 80% confidence ribbons [q10, q90], and peak-hour safety margins.</div>
        </div>
        <div>
            <span class="chip chip-cyan">PRIMARY MARKET HORIZON</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

df = get_forecast_df()
folds = get_fold_metadata()
df_24h = df[df["horizon"] == "primary"].copy()

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Display Controls")
fold_id = st.sidebar.selectbox(
    "Select Backtest Fold",
    options=[1, 2, 3, 4, 5],
    index=4,
    format_func=lambda x: f"Fold {x} (Peak Summer Horizon)" if x == 5 else f"Fold {x}",
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

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    show_uncertainty = st.toggle("Uncertainty (80% CI)", value=True)
with col_t2:
    show_actual = st.toggle("Actual Demand", value=True)

# Slicing for selected fold
sub_df = df_24h[df_24h["fold_id"] == fold_id]

# Fold date info
fold_info = next((f for f in folds if f["fold_id"] == fold_id), None)
if fold_info:
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; margin-bottom: 8px;">
            EVALUATION PERIOD: {fold_info['test_start'][:10]} to {fold_info['test_end'][:10]} | TRAINING CUTOFF: {fold_info['train_end'][:10]} (Expanding Window)
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main Forecast Chart (Clean bottom legend, zero overlap)
fig = create_forecast_plot(
    sub_df,
    models=selected_models,
    show_actual=show_actual,
    show_uncertainty=show_uncertainty,
    title=f"Fold {fold_id} Day-Ahead Hourly Demand Profile (MW)",
)
st.plotly_chart(fig, use_container_width=True)

# Performance Breakdown & Peak Hour Inspection
st.markdown("---")
score_col, peak_col = st.columns([1.6, 1.0])

with score_col:
    st.markdown(f"#### 📊 Fold {fold_id} Accuracy Scorecard")
    fold_metrics = []
    for m in selected_models:
        m_data = sub_df[sub_df["model"] == m]
        if not m_data.empty:
            met = compute_all_metrics(m_data["actual"], m_data["q10"], m_data["q50"], m_data["q90"])
            fold_metrics.append({
                "Model": m.upper() if m != "chronos" else "Chronos-T5",
                "MAE (MW)": round(met["mae"], 1),
                "RMSE (MW)": round(met["rmse"], 1),
                "MAPE": f"{met['mape']:.2f}%",
                "Winkler Score": round(met["winkler_80"], 0),
                "80% Coverage": f"{met['coverage_80']*100:.1f}%",
            })

    if fold_metrics:
        score_df = pd.DataFrame(fold_metrics).sort_values("RMSE (MW)")
        st.dataframe(score_df, use_container_width=True, hide_index=True)

with peak_col:
    st.markdown("#### ⚡ Peak Hour Load Analysis")
    actual_max_row = sub_df.loc[sub_df["actual"].idxmax()]
    actual_peak_mw = actual_max_row["actual"]
    actual_peak_time = actual_max_row["timestamp"].strftime("%H:00")

    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-label">
                <span>ACTUAL GRID PEAK</span>
                <span class="chip chip-warning">AT {actual_peak_time}</span>
            </div>
            <div class="telemetry-value" style="color: #f59e0b;">{actual_peak_mw:,.0f} <span style="font-size: 0.9rem; color: #94a3b8;">MW</span></div>
            <div class="telemetry-meta">
                <span>Timestamp: <b>{actual_max_row['timestamp'].strftime('%Y-%m-%d %H:00')}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1-Click CSV Download
    csv_bytes = sub_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Fold Predictions (CSV)",
        data=csv_bytes,
        file_name=f"pjm_24h_forecast_fold{fold_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )
