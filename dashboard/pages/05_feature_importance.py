"""
⚡ Page 5: Feature Intelligence & Model Interpretability
SHAP TreeExplainer values, gain rankings, category attribution, and continuous Fourier polar circle.
"""
from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    apply_custom_css,
    get_feature_importance,
    render_sidebar,
)
from src.visualization.plots import (
    CHART_THEME,
    create_shap_bar_chart,
    create_cyclical_polar_chart,
)

st.set_page_config(
    page_title="Feature Intelligence | Grid Intelligence",
    page_icon="⚡",
    layout="wide",
)

apply_custom_css()
render_sidebar()

st.markdown("## 🧠 Feature Intelligence & Interpretability")
st.markdown(
    "In-depth breakdown of the 35+ engineered temporal, cyclical, and autoregressive features for **Gen 2: XGBoost**. "
    "Features are evaluated using model-agnostic **SHAP (SHapley Additive exPlanations)** and split-gain metrics."
)

feat_data = get_feature_importance()

col_bar, col_donut = st.columns([1.4, 1.0])

with col_bar:
    st.markdown("### 🏆 Top 20 Features by Mean |SHAP Value| (MW Impact)")
    top_n = st.slider("Number of features to display", min_value=10, max_value=25, value=15)
    shap_fig = create_shap_bar_chart(feat_data, top_n=top_n)
    st.plotly_chart(shap_fig, use_container_width=True)

with col_donut:
    st.markdown("### 🥧 Importance Attribution by Category")
    shap_dict = feat_data.get("shap_mean", {})
    
    # Categorize features
    cats = {"Autoregressive Lags": 0.0, "Cyclical Fourier": 0.0, "Rolling Statistics": 0.0, "Calendar & Interaction": 0.0}
    for k, v in shap_dict.items():
        if "lag" in k:
            cats["Autoregressive Lags"] += v
        elif "sin" in k or "cos" in k:
            cats["Cyclical Fourier"] += v
        elif "rolling" in k or "ewm" in k:
            cats["Rolling Statistics"] += v
        else:
            cats["Calendar & Interaction"] += v

    donut_fig = go.Figure(
        data=[
            go.Pie(
                labels=list(cats.keys()),
                values=list(cats.values()),
                hole=0.6,
                marker=dict(colors=["#f59e0b", "#00d4ff", "#8b5cf6", "#9ca3af"]),
                textinfo="label+percent",
            )
        ]
    )
    donut_fig.update_layout(
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=30, b=50),
    )
    st.plotly_chart(donut_fig, use_container_width=True)

st.markdown("---")
col_polar, col_compare = st.columns([1.1, 1.3])

with col_polar:
    st.markdown("### 🔄 Cyclical Fourier Encoding Mechanics")
    st.markdown(
        "Standard integer representations treat Hour 23 and Hour 0 as extreme opposites ($23 - 0 = 23$). "
        "Sine/cosine Fourier projection maps time onto a continuous unit circle, eliminating boundary distortion."
    )
    polar_fig = create_cyclical_polar_chart()
    st.plotly_chart(polar_fig, use_container_width=True)

with col_compare:
    st.markdown("### ⚖️ SHAP vs Split Gain Rank Divergence")
    st.markdown("Comparison showing how Gain over-indexes on high-cardinality features while SHAP measures true marginal impact.")

    gains = feat_data.get("gain", {})
    shaps = feat_data.get("shap_mean", {})

    sorted_gain_feats = sorted(gains.keys(), key=lambda x: gains[x], reverse=True)[:10]
    sorted_shap_feats = sorted(shaps.keys(), key=lambda x: shaps[x], reverse=True)[:10]

    comp_rows = []
    for rank in range(10):
        comp_rows.append({
            "Rank": f"#{rank+1}",
            "Top Feature by SHAP": sorted_shap_feats[rank] if rank < len(sorted_shap_feats) else "-",
            "SHAP (MW)": round(shaps.get(sorted_shap_feats[rank], 0), 1) if rank < len(sorted_shap_feats) else 0,
            "Top Feature by Split Gain": sorted_gain_feats[rank] if rank < len(sorted_gain_feats) else "-",
        })

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
