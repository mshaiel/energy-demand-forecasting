"""
⚡ Page 6: Feature Intelligence & Interpretability
SHAP TreeExplainer values, category attribution donut, and continuous Fourier polar circle.
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

# Console Header
st.markdown(
    """
    <div class="console-header">
        <div>
            <div class="console-title">🧠 Feature Intelligence & SHAP Interpretability</div>
            <div class="console-subtitle">Explaining XGBoost decisions with model-agnostic SHapley Additive exPlanations (SHAP) across 35+ engineered features.</div>
        </div>
        <div>
            <span class="chip chip-warning">TREEEXPLAINER ATTRIBUTION</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

feat_data = get_feature_importance()

col_bar, col_donut = st.columns([1.35, 1.0])

with col_bar:
    st.markdown("#### 🏆 Top Features by Mean |SHAP| (MW Impact on Demand)")
    top_n = st.slider("Features to Display", min_value=10, max_value=25, value=15)
    shap_fig = create_shap_bar_chart(feat_data, top_n=top_n)
    st.plotly_chart(shap_fig, use_container_width=True)

with col_donut:
    st.markdown("#### 🥧 Importance Attribution by Category")
    shap_dict = feat_data.get("shap_mean", {})
    
    cats = {
        "Autoregressive Lags": 0.0,
        "Cyclical Fourier": 0.0,
        "Rolling Statistics": 0.0,
        "Calendar & Interaction": 0.0,
    }
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
                marker=dict(colors=["#f59e0b", "#00d4ff", "#8b5cf6", "#64748b"]),
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )
    donut_fig.update_layout(
        height=380,
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font=dict(size=10, color="#cbd5e1")),
        margin=dict(l=30, r=30, t=30, b=50),
    )
    st.plotly_chart(donut_fig, use_container_width=True)

st.markdown("---")
col_polar, col_compare = st.columns([1.1, 1.3])

with col_polar:
    st.markdown("#### 🔄 Continuous Fourier Encoding Mechanics")
    st.markdown(
        "<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 10px;'>"
        "Integer hour representations create artificial boundary step-jumps (Hour 23 &ne; Hour 0). "
        "Projecting onto a circular Fourier basis ensures smooth, continuous periodic transitions."
        "</div>",
        unsafe_allow_html=True,
    )
    polar_fig = create_cyclical_polar_chart()
    st.plotly_chart(polar_fig, use_container_width=True)

with col_compare:
    st.markdown("#### ⚖️ SHAP vs Split Gain Rank Divergence")
    st.markdown(
        "<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 10px;'>"
        "Split Gain over-indexes on high-cardinality continuous splits, whereas SHAP measures true marginal impact."
        "</div>",
        unsafe_allow_html=True,
    )

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
