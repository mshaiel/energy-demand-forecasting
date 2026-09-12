"""
Reusable Plotly figure factories adhering to the institutional SCADA dark aesthetic.
Guarantees zero text collisions, proper margins, and crisp telemetry formatting.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Institutional Theme Tokens
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,21,38,0.7)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155", linecolor="#334155", tickfont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155", linecolor="#334155", tickfont=dict(color="#94a3b8")),
)

MODEL_COLORS = {
    "sarimax": "#64748b",   # Slate Muted
    "xgboost": "#f59e0b",   # Amber
    "lstm": "#8b5cf6",      # Purple
    "chronos": "#00d4ff",   # Electric Cyan
}

MODEL_DISPLAY_NAMES = {
    "sarimax": "SARIMAX (Gen 1)",
    "xgboost": "XGBoost (Gen 2)",
    "lstm": "LSTM (Gen 3)",
    "chronos": "Chronos-T5 (Gen 4)",
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
    title: str = "PJM Grid Demand: 24-Hour Horizon vs Ground Truth",
) -> go.Figure:
    """
    Interactive forecast comparison chart with generous margins and bottom legend
    to guarantee zero collision with titles or header text.
    """
    fig = go.Figure()
    if models is None:
        models = list(MODEL_COLORS.keys())

    # Actual demand trace
    if show_actual and not df.empty:
        sub_actual = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=sub_actual["timestamp"],
                y=sub_actual["actual"],
                mode="lines",
                name="Actual Load",
                line=dict(color="#f8fafc", width=2.5),
                hovertemplate="<b>Actual</b>: %{y:,.0f} MW<br>%{x|%b %d, %H:00}<extra></extra>",
            )
        )

    # Models traces
    for m in models:
        m_df = df[df["model"] == m].sort_values("timestamp")
        if m_df.empty:
            continue

        color = MODEL_COLORS.get(m, "#00d4ff")
        rgba_fill = hex_to_rgba(color, 0.15)
        disp_name = MODEL_DISPLAY_NAMES.get(m, m.upper())

        # Uncertainty ribbons
        if show_uncertainty and "q10" in m_df.columns and "q90" in m_df.columns:
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
            fig.add_trace(
                go.Scatter(
                    x=m_df["timestamp"],
                    y=m_df["q10"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=rgba_fill,
                    name=f"{disp_name} [80% CI]",
                    hoverinfo="skip",
                )
            )

        # Median forecast (q50)
        fig.add_trace(
            go.Scatter(
                x=m_df["timestamp"],
                y=m_df["q50"],
                mode="lines+markers",
                marker=dict(size=4),
                name=disp_name,
                line=dict(color=color, width=2.2),
                hovertemplate=f"<b>{disp_name}</b>: %{{y:,.0f}} MW<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color="#f1f5f9", family="Inter, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        height=460,
        xaxis=dict(
            title=None,
            showgrid=True,
            **CHART_THEME["xaxis"],
        ),
        yaxis=dict(
            title="Demand (MW)",
            showgrid=True,
            tickformat=",.0f",
            **CHART_THEME["yaxis"],
        ),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(14,21,38,0.85)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(size=11, color="#cbd5e1"),
        ),
        margin=dict(l=65, r=30, t=50, b=75),
        hovermode="x unified",
    )
    return fig


def create_radar_chart(
    metrics_summary: Dict[str, Any],
    horizon_key: str = "primary_24h",
) -> go.Figure:
    """Radar comparison chart with non-overlapping geometry."""
    categories = ["Accuracy (1/MAE)", "Peak Fit (1/RMSE)", "Percentage (1/MAPE)", "Sharpness (1/Winkler)", "Calibration"]
    fig = go.Figure()

    raw_vals: Dict[str, List[float]] = {}
    models_to_eval = ["chronos", "lstm", "xgboost", "sarimax"]

    for m in models_to_eval:
        key = f"{m}_{horizon_key}"
        if key in metrics_summary:
            rec = metrics_summary[key]
            raw_vals[m] = [
                1.0 / max(rec.get("mae", 1.0), 1e-4),
                1.0 / max(rec.get("rmse", 1.0), 1e-4),
                1.0 / max(rec.get("mape", 1.0), 1e-4),
                1.0 / max(rec.get("winkler_80", 1.0), 1e-4),
                1.0 - min(abs(rec.get("coverage_80", 0.8) - 0.8) * 3.5, 0.8),
            ]

    if raw_vals:
        arr = np.array(list(raw_vals.values()))
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)
        rng = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)

        for m, vals in raw_vals.items():
            norm_vals = ((np.array(vals) - min_vals) / rng) * 0.7 + 0.3
            r_vals = norm_vals.tolist() + [norm_vals[0]]
            cat_loop = categories + [categories[0]]
            color = MODEL_COLORS.get(m, "#00d4ff")

            fig.add_trace(
                go.Scatterpolar(
                    r=r_vals,
                    theta=cat_loop,
                    fill="toself",
                    fillcolor=hex_to_rgba(color, 0.12),
                    line=dict(color=color, width=2),
                    name=MODEL_DISPLAY_NAMES.get(m, m.upper()),
                )
            )

    fig.update_layout(
        height=380,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False, gridcolor="#1e293b"),
            angularaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(size=10, color="#94a3b8")),
            bgcolor="rgba(14,21,38,0.5)",
        ),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#cbd5e1"),
        ),
        margin=dict(l=55, r=55, t=35, b=65),
    )
    return fig


def create_dispatch_simulation_plot(
    timestamps: pd.Series,
    actual: np.ndarray,
    baseline_q50: np.ndarray,
    simulated_q50: np.ndarray,
    peak_shaved_q50: Optional[np.ndarray] = None,
    grid_capacity_mw: float = 45000.0,
) -> go.Figure:
    """
    Operational Stress-Test & Peak Dispatch Simulator Chart.
    Visualizes base forecast vs weather/demand shock vs peak demand response shaving.
    """
    fig = go.Figure()

    # Grid Critical Capacity Limit line
    fig.add_hline(
        y=grid_capacity_mw,
        line=dict(color="#f43f5e", width=1.8, dash="dash"),
        annotation_text=f"Total Available Thermal Capacity ({grid_capacity_mw:,.0f} MW)",
        annotation_position="top left",
        annotation_font=dict(color="#f43f5e", size=11),
    )

    # 1. Baseline Day-Ahead Dispatch
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=baseline_q50,
            mode="lines",
            name="Baseline Day-Ahead Forecast",
            line=dict(color="#3b82f6", width=2, dash="dot"),
            hovertemplate="<b>Base Forecast</b>: %{y:,.0f} MW<extra></extra>",
        )
    )

    # 2. Simulated Shock Forecast (Heatwave / Grid Strain)
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=simulated_q50,
            mode="lines+markers",
            marker=dict(size=4),
            name="Stressed Grid Scenario",
            line=dict(color="#f59e0b", width=2.5),
            hovertemplate="<b>Stressed Load</b>: %{y:,.0f} MW<extra></extra>",
        )
    )

    # 3. Peak-Shaved Demand Response
    if peak_shaved_q50 is not None:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=peak_shaved_q50,
                mode="lines",
                name="Post-Demand Response (Peak Shaved)",
                line=dict(color="#10b981", width=2.5),
                hovertemplate="<b>Shaved Load</b>: %{y:,.0f} MW<extra></extra>",
            )
        )

    # Fill risk region where stressed demand exceeds safe reserve threshold
    safe_reserve_threshold = grid_capacity_mw * 0.90
    fig.add_hline(
        y=safe_reserve_threshold,
        line=dict(color="#f59e0b", width=1, dash="dot"),
        annotation_text=f"90% Reserve Alert Threshold ({safe_reserve_threshold:,.0f} MW)",
        annotation_position="bottom left",
        annotation_font=dict(color="#f59e0b", size=10),
    )

    fig.update_layout(
        title=dict(
            text="Simulated Hourly Load Trajectory vs Capacity Safety Envelope",
            font=dict(size=14, color="#f1f5f9"),
            x=0.01,
            y=0.96,
        ),
        height=480,
        xaxis=dict(showgrid=True, **CHART_THEME["xaxis"]),
        yaxis=dict(title="Grid Demand (MW)", tickformat=",.0f", showgrid=True, **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(14,21,38,0.85)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(size=11, color="#cbd5e1"),
        ),
        margin=dict(l=65, r=30, t=55, b=75),
        hovermode="x unified",
    )
    return fig


def create_fold_timeline_chart(fold_metadata: List[Dict[str, Any]]) -> go.Figure:
    """Gantt chart displaying expanding window boundaries with clean padding."""
    fig = go.Figure()

    for item in fold_metadata:
        fold_id = item["fold_id"]
        fig.add_trace(
            go.Scatter(
                x=[item["train_start"], item["train_end"]],
                y=[f"Fold {fold_id}", f"Fold {fold_id}"],
                mode="lines",
                line=dict(color="#334155", width=12),
                name="Train History (Expanding)" if fold_id == 1 else None,
                showlegend=(fold_id == 1),
                hoverinfo="text",
                hovertext=f"Fold {fold_id} Train: {item['train_start'][:10]} to {item['train_end'][:10]}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[item["test_start"], item["test_end"]],
                y=[f"Fold {fold_id}", f"Fold {fold_id}"],
                mode="lines",
                line=dict(color="#00d4ff", width=12),
                name="Out-of-Sample Test Window" if fold_id == 1 else None,
                showlegend=(fold_id == 1),
                hoverinfo="text",
                hovertext=f"Fold {fold_id} Test: {item['test_start'][:10]} to {item['test_end'][:10]}",
            )
        )

    fig.update_layout(
        title=dict(text="5-Fold Expanding Window Protocol (Zero Temporal Leakage)", font=dict(size=13, color="#f1f5f9"), x=0.01),
        height=320,
        xaxis=dict(title=None, **CHART_THEME["xaxis"]),
        yaxis=dict(autorange="reversed", **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=11, color="#cbd5e1")),
        margin=dict(l=70, r=30, t=45, b=65),
    )
    return fig


def create_per_fold_metric_chart(
    df: pd.DataFrame,
    metric_func: Any,
    metric_name: str = "RMSE (MW)",
) -> go.Figure:
    """Stability line chart across backtest folds."""
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
                line=dict(color=color, width=2.2),
                marker=dict(size=6, color=color),
                hovertemplate=f"<b>{disp_name}</b>: %{{y:,.1f}}<extra></extra>",
            )
        )

    fig.update_layout(
        height=360,
        xaxis=dict(title=None, **CHART_THEME["xaxis"]),
        yaxis=dict(title=metric_name, **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font=dict(size=10, color="#cbd5e1")),
        margin=dict(l=60, r=25, t=35, b=65),
    )
    return fig


def create_error_box_plot(df: pd.DataFrame) -> go.Figure:
    """Distribution of absolute errors per model."""
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
        height=360,
        yaxis=dict(title="Absolute Error (MW)", **CHART_THEME["yaxis"]),
        xaxis=dict(**CHART_THEME["xaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=dict(l=55, r=25, t=35, b=45),
    )
    return fig


def create_coverage_calibration_chart(metrics_summary: Dict[str, Any], horizon_key: str = "primary_24h") -> go.Figure:
    """Empirical coverage calibration chart."""
    models = ["sarimax", "xgboost", "lstm", "chronos"]
    labels = []
    coverages = []
    bar_colors = []

    for m in models:
        key = f"{m}_{horizon_key}"
        cov = metrics_summary.get(key, {}).get("coverage_80", 0.0)
        labels.append(MODEL_DISPLAY_NAMES.get(m, m))
        coverages.append(cov * 100.0)

        err = abs(cov - 0.80)
        if err <= 0.05:
            bar_colors.append("#10b981")
        elif err <= 0.10:
            bar_colors.append("#f59e0b")
        else:
            bar_colors.append("#f43f5e")

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

    fig.add_hline(
        y=80,
        line=dict(color="#00d4ff", width=2, dash="dash"),
        annotation_text="Ideal Nominal 80% CI",
        annotation_position="top right",
        annotation_font=dict(color="#00d4ff", size=11),
    )

    fig.update_layout(
        height=340,
        yaxis=dict(title="Observed Coverage (%)", range=[0, 105], **CHART_THEME["yaxis"]),
        xaxis=dict(**CHART_THEME["xaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=dict(l=50, r=30, t=40, b=40),
    )
    return fig


def create_shap_bar_chart(feature_importance: Dict[str, Any], top_n: int = 15) -> go.Figure:
    """Horizontal bar chart for SHAP importances with generous label margins."""
    shap_data = feature_importance.get("shap_mean", {})
    if not shap_data:
        shap_data = feature_importance.get("gain", {})

    sorted_feats = sorted(shap_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not sorted_feats:
        return go.Figure()

    feats = [f[0] for f in sorted_feats][::-1]
    vals = [f[1] for f in sorted_feats][::-1]

    colors = []
    for f in feats:
        if "lag" in f or "shift" in f:
            colors.append("#f59e0b")
        elif "sin" in f or "cos" in f:
            colors.append("#00d4ff")
        elif "rolling" in f or "ewm" in f:
            colors.append("#8b5cf6")
        else:
            colors.append("#64748b")

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=feats,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b>: %{x:,.1f} MW impact<extra></extra>",
        )
    )

    fig.update_layout(
        height=440,
        xaxis=dict(title="Mean |SHAP Value| (MW Impact on Demand)", **CHART_THEME["xaxis"]),
        yaxis=dict(**CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=dict(l=140, r=25, t=30, b=50),
    )
    return fig


def create_cyclical_polar_chart() -> go.Figure:
    """Continuous circular Fourier projection of 24h diurnal cycle."""
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
            line=dict(color="#334155", dash="dash", width=1.5),
            name="24h Continuous Cycle",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=key_x,
            y=key_y,
            mode="markers+text",
            marker=dict(size=10, color="#00d4ff"),
            text=key_labels,
            textposition="top center",
            name="Key Diurnal Steps",
            hovertemplate="<b>%{text}</b><br>sin: %{x:.2f}<br>cos: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=380,
        xaxis=dict(title="hour_sin", range=[-1.4, 1.4], **CHART_THEME["xaxis"]),
        yaxis=dict(title="hour_cos", range=[-1.4, 1.4], scaleanchor="x", scaleratio=1, **CHART_THEME["yaxis"]),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        margin=dict(l=40, r=40, t=30, b=40),
    )
    return fig
