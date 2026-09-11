"""
Reusable Plotly figure factories adhering to the dark-mode SCADA aesthetic.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# SCADA Palette Tokens
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.75)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#f9fafb", size=12),
    xaxis=dict(gridcolor="#1f2937", zerolinecolor="#374151", linecolor="#374151"),
    yaxis=dict(gridcolor="#1f2937", zerolinecolor="#374151", linecolor="#374151"),
)

MODEL_COLORS = {
    "sarimax": "#9ca3af",   # Muted grey (Gen 1)
    "xgboost": "#f59e0b",   # Amber / Orange (Gen 2)
    "lstm": "#8b5cf6",      # Purple (Gen 3)
    "chronos": "#00d4ff",   # Electric Cyan (Gen 4 Foundation)
}

MODEL_DISPLAY_NAMES = {
    "sarimax": "Gen 1: SARIMAX",
    "xgboost": "Gen 2: XGBoost",
    "lstm": "Gen 3: LSTM (MC Dropout)",
    "chronos": "Gen 4: Chronos-T5 (Zero-Shot)",
}


def hex_to_rgba(hex_code: str, opacity: float = 0.2) -> str:
    """Convert hex color string to rgba CSS string."""
    hex_code = hex_code.lstrip("#")
    if len(hex_code) == 6:
        r, g, b = tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{opacity})"
    return f"rgba(0,212,255,{opacity})"


def create_forecast_plot(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    show_actual: bool = True,
    show_uncertainty: bool = True,
    title: str = "PJM Energy Demand Forecast vs Actual",
) -> go.Figure:
    """
    Generate interactive forecast comparison chart with median prediction (q50)
    and shaded uncertainty bands (q10 to q90).
    """
    fig = go.Figure()
    if models is None:
        models = list(MODEL_COLORS.keys())

    # Add ground truth trace first
    if show_actual and not df.empty:
        # Sort by timestamp
        sub_actual = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=sub_actual["timestamp"],
                y=sub_actual["actual"],
                mode="lines",
                name="Actual Demand",
                line=dict(color="#f9fafb", width=2.5, dash="solid"),
                hovertemplate="<b>Actual</b>: %{y:,.0f} MW<br>%{x}<extra></extra>",
            )
        )

    # Add predictions for each requested model
    for m in models:
        m_df = df[df["model"] == m].sort_values("timestamp")
        if m_df.empty:
            continue

        color = MODEL_COLORS.get(m, "#00d4ff")
        rgba_fill = hex_to_rgba(color, 0.18)
        disp_name = MODEL_DISPLAY_NAMES.get(m, m.upper())

        # Uncertainty bands: q10 and q90
        if show_uncertainty and "q10" in m_df.columns and "q90" in m_df.columns:
            # Upper bound (q90)
            fig.add_trace(
                go.Scatter(
                    x=m_df["timestamp"],
                    y=m_df["q90"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Lower bound (q10) filled to upper bound
            fig.add_trace(
                go.Scatter(
                    x=m_df["timestamp"],
                    y=m_df["q10"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=rgba_fill,
                    name=f"{disp_name} (80% CI)",
                    hoverinfo="skip",
                )
            )

        # Median forecast (q50)
        fig.add_trace(
            go.Scatter(
                x=m_df["timestamp"],
                y=m_df["q50"],
                mode="lines",
                name=f"{disp_name} (q50)",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{disp_name}</b>: %{{y:,.0f}} MW<br>%{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f9fafb")),
        xaxis=dict(title="Timestamp (Hourly UTC)", showgrid=True, **CHART_THEME["xaxis"]),
        yaxis=dict(title="Grid Demand (MW)", showgrid=True, tickformat=",.0f", **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="#374151",
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=70, b=50),
        hovermode="x unified",
    )
    return fig


def create_radar_chart(
    metrics_summary: Dict[str, Any],
    horizon_key: str = "primary_24h",
) -> go.Figure:
    """
    Generate normalized Radar / Spider chart comparing models across dimensions.
    Dimensions: MAE, RMSE, MAPE, Winkler Score, Coverage Balance.
    """
    categories = ["1 / MAE", "1 / RMSE", "1 / MAPE", "1 / Winkler", "Coverage (80%)"]
    fig = go.Figure()

    raw_vals: Dict[str, List[float]] = {}
    models_to_eval = ["sarimax", "xgboost", "lstm", "chronos"]

    for m in models_to_eval:
        key = f"{m}_{horizon_key}"
        if key in metrics_summary:
            rec = metrics_summary[key]
            # Inverse of error so higher is better on the radar
            raw_vals[m] = [
                1.0 / max(rec.get("mae", 1.0), 1e-4),
                1.0 / max(rec.get("rmse", 1.0), 1e-4),
                1.0 / max(rec.get("mape", 1.0), 1e-4),
                1.0 / max(rec.get("winkler_80", 1.0), 1e-4),
                1.0 - abs(rec.get("coverage_80", 0.8) - 0.8) * 5.0,  # Peak score at 0.80
            ]

    # Normalize each category to 0-1 scale across available models
    if raw_vals:
        arr = np.array(list(raw_vals.values()))
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)
        rng = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)

        for m, vals in raw_vals.items():
            norm_vals = ((np.array(vals) - min_vals) / rng) * 0.75 + 0.25
            # Close polygon loop
            r_vals = norm_vals.tolist() + [norm_vals[0]]
            cat_loop = categories + [categories[0]]
            color = MODEL_COLORS.get(m, "#00d4ff")

            fig.add_trace(
                go.Scatterpolar(
                    r=r_vals,
                    theta=cat_loop,
                    fill="toself",
                    fillcolor=hex_to_rgba(color, 0.15),
                    line=dict(color=color, width=2),
                    name=MODEL_DISPLAY_NAMES.get(m, m.upper()),
                )
            )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False, gridcolor="#374151"),
            angularaxis=dict(gridcolor="#374151", linecolor="#4b5563"),
            bgcolor="rgba(17,24,39,0.6)",
        ),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=40, t=30, b=50),
    )
    return fig


def create_fold_timeline_chart(fold_metadata: List[Dict[str, Any]]) -> go.Figure:
    """
    Gantt chart displaying 5-fold expanding training windows and contiguous test windows.
    Visually proves no temporal leakage.
    """
    fig = go.Figure()

    for item in fold_metadata:
        fold_id = item["fold_id"]
        # Training segment
        fig.add_trace(
            go.Scatter(
                x=[item["train_start"], item["train_end"]],
                y=[f"Fold {fold_id}", f"Fold {fold_id}"],
                mode="lines",
                line=dict(color="#4b5563", width=14),
                name="Train Window (Expanding)" if fold_id == 1 else None,
                showlegend=(fold_id == 1),
                hoverinfo="text",
                hovertext=f"Fold {fold_id} Train: {item['train_start']} to {item['train_end']}",
            )
        )
        # Test segment
        fig.add_trace(
            go.Scatter(
                x=[item["test_start"], item["test_end"]],
                y=[f"Fold {fold_id}", f"Fold {fold_id}"],
                mode="lines",
                line=dict(color="#00d4ff", width=14),
                name="Test Window (Evaluation)" if fold_id == 1 else None,
                showlegend=(fold_id == 1),
                hoverinfo="text",
                hovertext=f"Fold {fold_id} Test: {item['test_start']} to {item['test_end']}",
            )
        )

    fig.update_layout(
        title=dict(text="5-Fold Expanding Window Backtesting Protocol (No Leakage)", font=dict(size=15)),
        xaxis=dict(title="Timeline", **CHART_THEME["xaxis"]),
        yaxis=dict(autorange="reversed", **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
    )
    return fig


def create_per_fold_metric_chart(
    df: pd.DataFrame,
    metric_func: Any,
    metric_name: str = "RMSE (MW)",
) -> go.Figure:
    """
    Line chart tracking stability of each model across the 5 backtesting folds.
    """
    fig = go.Figure()
    folds = sorted(df["fold_id"].unique())

    for m in df["model"].unique():
        scores = []
        for f in folds:
            sub = df[(df["model"] == m) & (df["fold_id"] == f)]
            if not sub.empty:
                scores.append(metric_func(sub["actual"], sub["q50"]))
            else:
                scores.append(np.nan)

        color = MODEL_COLORS.get(m, "#00d4ff")
        disp_name = MODEL_DISPLAY_NAMES.get(m, m.upper())

        fig.add_trace(
            go.Scatter(
                x=[f"Fold {f}" for f in folds],
                y=scores,
                mode="lines+markers",
                name=disp_name,
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color),
                hovertemplate=f"<b>{disp_name}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=f"Model Stability: {metric_name} Across Folds", font=dict(size=15)),
        xaxis=dict(title="Backtest Fold", **CHART_THEME["xaxis"]),
        yaxis=dict(title=metric_name, **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig


def create_error_box_plot(df: pd.DataFrame) -> go.Figure:
    """
    Distribution of absolute errors per model across all backtest folds.
    """
    fig = go.Figure()
    data_with_err = df.copy()
    data_with_err["abs_error"] = np.abs(data_with_err["actual"] - data_with_err["q50"])

    for m in data_with_err["model"].unique():
        sub = data_with_err[data_with_err["model"] == m]
        color = MODEL_COLORS.get(m, "#00d4ff")
        disp = MODEL_DISPLAY_NAMES.get(m, m.upper())
        fig.add_trace(
            go.Box(
                y=sub["abs_error"],
                name=disp,
                marker_color=color,
                boxmean=True,
            )
        )

    fig.update_layout(
        title=dict(text="Absolute Error Distribution (MW)", font=dict(size=15)),
        yaxis=dict(title="Absolute Error (MW)", **CHART_THEME["yaxis"]),
        xaxis=dict(**CHART_THEME["xaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig


def create_coverage_calibration_chart(metrics_summary: Dict[str, Any], horizon_key: str = "primary_24h") -> go.Figure:
    """
    Empirical coverage bar chart vs the nominal 80% confidence interval line.
    """
    models = ["sarimax", "xgboost", "lstm", "chronos"]
    labels = []
    coverages = []
    bar_colors = []

    for m in models:
        key = f"{m}_{horizon_key}"
        cov = metrics_summary.get(key, {}).get("coverage_80", 0.0)
        labels.append(MODEL_DISPLAY_NAMES.get(m, m))
        coverages.append(cov * 100.0)

        # Color: green within +-5%, amber within +-10%, red otherwise
        err = abs(cov - 0.80)
        if err <= 0.05:
            bar_colors.append("#10b981")  # Green
        elif err <= 0.10:
            bar_colors.append("#f59e0b")  # Amber
        else:
            bar_colors.append("#ef4444")  # Red

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=coverages,
            marker_color=bar_colors,
            text=[f"{c:.1f}%" for c in coverages],
            textposition="auto",
        )
    )

    # Ideal 80% line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(models) - 0.5,
        y0=80,
        y1=80,
        line=dict(color="#00d4ff", width=2.5, dash="dash"),
    )
    fig.add_annotation(
        x=len(models) - 0.6,
        y=82,
        text="Nominal 80% CI",
        showarrow=False,
        font=dict(color="#00d4ff", size=11),
    )

    fig.update_layout(
        title=dict(text="Empirical Coverage Calibration vs Nominal 80%", font=dict(size=15)),
        yaxis=dict(title="Observed Coverage (%)", range=[0, 105], **CHART_THEME["yaxis"]),
        xaxis=dict(**CHART_THEME["xaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig


def create_shap_bar_chart(feature_importance: Dict[str, Any], top_n: int = 20) -> go.Figure:
    """
    Horizontal bar chart of top features by mean |SHAP| value, color-coded by feature category.
    """
    shap_data = feature_importance.get("shap_mean", {})
    if not shap_data:
        shap_data = feature_importance.get("gain", {})

    sorted_feats = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not sorted_feats:
        return go.Figure()

    feats = [f[0] for f in sorted_feats][::-1]
    vals = [f[1] for f in sorted_feats][::-1]

    # Category colors: lag=amber, cyclical=cyan, rolling=purple, calendar=grey
    colors = []
    for f in feats:
        if "lag" in f or "shift" in f:
            colors.append("#f59e0b")
        elif "sin" in f or "cos" in f:
            colors.append("#00d4ff")
        elif "rolling" in f or "ewm" in f:
            colors.append("#8b5cf6")
        else:
            colors.append("#9ca3af")

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=feats,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"Top {top_n} Features by Predictive Importance", font=dict(size=15)),
        xaxis=dict(title="Importance Score", **CHART_THEME["xaxis"]),
        yaxis=dict(**CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=dict(l=120, r=30, t=50, b=40),
    )
    return fig


def create_cyclical_polar_chart() -> go.Figure:
    """
    Unit circle demonstrating continuous Fourier representation of hour-of-day.
    Solves hour 23 -> hour 0 boundary discontinuity.
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.sin(theta)
    y = np.cos(theta)

    key_hours = [0, 6, 12, 18, 23]
    key_x = [np.sin(2 * np.pi * h / 24) for h in key_hours]
    key_y = [np.cos(2 * np.pi * h / 24) for h in key_hours]
    key_labels = [f"Hour {h}:00" for h in key_hours]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="#374151", dash="dash", width=1.5),
            name="24h Cycle",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=key_x,
            y=key_y,
            mode="markers+text",
            marker=dict(size=12, color="#00d4ff"),
            text=key_labels,
            textposition="top center",
            name="Key Hours",
            hovertemplate="<b>%{text}</b><br>sin: %{x:.2f}<br>cos: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Continuous Cyclical Fourier Space (hour_sin, hour_cos)", font=dict(size=15)),
        xaxis=dict(title="hour_sin", range=[-1.3, 1.3], **CHART_THEME["xaxis"]),
        yaxis=dict(title="hour_cos", range=[-1.3, 1.3], scaleanchor="x", scaleratio=1, **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
