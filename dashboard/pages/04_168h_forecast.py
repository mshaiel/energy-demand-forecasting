"""
⚡ Page 4: 168-Hour Extended Forecast Explorer (7-Day Horizon)
Multi-day forecast trajectory, diurnal demand decomposition, and uncertainty envelope width analysis.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
from src.visualization.plots import (
    CHART_THEME,
    MODEL_COLORS,
    create_forecast_plot,
)

st.set_page_config(
    page_title="168h Extended Forecast | Grid Intelligence",
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
            <div class="console-title">📅 168-Hour Extended Forecast Trajectory (7-Day Operating Week)</div>
            <div class="console-subtitle">Observe long-horizon stability: XGBoost autoregressive lags (168h/336h) anchor weekly seasonality, while zero-shot models experience drift.</div>
        </div>
        <div>
            <span class="chip chip-cyan">WEEK-AHEAD DISPATCH</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

df = get_forecast_df()
folds = get_fold_metadata()
df_168h = df[df["horizon"] == "extended"].copy()

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Extended Controls")
fold_id = st.sidebar.selectbox(
    "Select Backtest Fold",
    options=[1, 2, 3, 4, 5],
    index=4,
    format_func=lambda x: f"Fold {x} (Peak Summer Horizon)" if x == 5 else f"Fold {x}",
)

models_available = ["xgboost", "chronos", "lstm", "sarimax"]
selected_models = st.sidebar.multiselect(
    "Models to Overlay",
    options=models_available,
    default=["xgboost", "chronos", "lstm"],
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

sub_df = df_168h[df_168h["fold_id"] == fold_id]

# Main 7-Day Chart (No collision, bottom legend)
fig = create_forecast_plot(
    sub_df,
    models=selected_models,
    show_actual=show_actual,
    show_uncertainty=show_uncertainty,
    title=f"Fold {fold_id} Full 7-Day Out-of-Sample Horizon (168 Hours)",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col_heat, col_decomp = st.columns(2)

with col_heat:
    st.markdown("#### 📈 Uncertainty Spread by Time of Day")
    st.markdown("<div style='font-size: 0.8rem; color: #64748b; margin-bottom: 8px;'>Average envelope width (q90 - q10) across 24 hours of the diurnal cycle.</div>", unsafe_allow_html=True)

    sub_df_copy = sub_df.copy()
    sub_df_copy["hour"] = sub_df_copy["timestamp"].dt.hour
    sub_df_copy["band_width"] = sub_df_copy["q90"] - sub_df_copy["q10"]

    spread_fig = go.Figure()
    for m in selected_models:
        m_sub = sub_df_copy[sub_df_copy["model"] == m]
        if not m_sub.empty:
            hourly_spread = m_sub.groupby("hour")["band_width"].mean()
            color = MODEL_COLORS.get(m, "#00d4ff")
            spread_fig.add_trace(
                go.Scatter(
                    x=hourly_spread.index,
                    y=hourly_spread.values,
                    mode="lines+markers",
                    name=m.upper(),
                    line=dict(color=color, width=2.2),
                    marker=dict(size=5),
                )
            )

    spread_fig.update_layout(
        height=340,
        xaxis=dict(title="Hour of Day (00:00 to 23:00)", tickmode="linear", dtick=3, **CHART_THEME["xaxis"]),
        yaxis=dict(title="Spread (MW)", **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10, color="#cbd5e1")),
        margin=dict(l=55, r=25, t=30, b=65),
    )
    st.plotly_chart(spread_fig, use_container_width=True)

with col_decomp:
    st.markdown("#### 📊 Diurnal Profile (Weekday vs Weekend)")
    st.markdown("<div style='font-size: 0.8rem; color: #64748b; margin-bottom: 8px;'>Structural load difference captured by calendar features and Fourier sine/cosine terms.</div>", unsafe_allow_html=True)

    sub_df_copy["is_weekend"] = sub_df_copy["timestamp"].dt.dayofweek >= 5
    weekday_mean = sub_df_copy[~sub_df_copy["is_weekend"]].groupby("hour")["actual"].mean()
    weekend_mean = sub_df_copy[sub_df_copy["is_weekend"]].groupby("hour")["actual"].mean()

    decomp_fig = go.Figure()
    decomp_fig.add_trace(
        go.Scatter(
            x=weekday_mean.index,
            y=weekday_mean.values,
            mode="lines",
            name="Weekday (Mon–Fri)",
            line=dict(color="#00d4ff", width=2.5),
        )
    )
    decomp_fig.add_trace(
        go.Scatter(
            x=weekend_mean.index,
            y=weekend_mean.values,
            mode="lines",
            name="Weekend (Sat–Sun)",
            line=dict(color="#f59e0b", width=2.5, dash="dash"),
        )
    )

    decomp_fig.update_layout(
        height=340,
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=3, **CHART_THEME["xaxis"]),
        yaxis=dict(title="Mean Demand (MW)", **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=10, color="#cbd5e1")),
        margin=dict(l=55, r=25, t=30, b=65),
    )
    st.plotly_chart(decomp_fig, use_container_width=True)
