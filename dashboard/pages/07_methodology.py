"""
⚡ Page 7: Mathematical Methodology & Formal Specification
Theoretical foundations: proper scoring rules, Winkler score formulas, and leakage isolation.
"""
from pathlib import Path
import sys
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    apply_custom_css,
    render_sidebar,
)

st.set_page_config(
    page_title="Methodology & Theory | Grid Intelligence",
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
            <div class="console-title">📜 Mathematical Methodology & Theoretical Foundations</div>
            <div class="console-subtitle">Formal definitions of proper scoring rules, interval sharpness penalties, and zero-leakage constraints.</div>
        </div>
        <div>
            <span class="chip chip-cyan">RESEARCH ABSTRACT</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("1. 🛡️ Rolling-Origin Expanding Window Protocol", expanded=True):
    st.markdown(
        """
        Standard cross-validation ($k$-fold) shuffles timestamps, causing severe lookahead bias in time series. 
        Instead, we employ an **Expanding Window Rolling-Origin Protocol** over 5 folds:
        
        * **Training Boundaries**: Always begins at $t_0 = \\text{2002-01-01}$, expanding by $168$ hours (1 week) on each subsequent fold.
        * **Test Boundaries**: Contiguous, non-overlapping evaluation slices covering the final $5 \\times 168 = 840$ hours (~35 days) of the grid history.
        * **Buffer Gap**: An optional gap parameter $\\delta \\ge 0$ prevents same-timestamp boundary overlap.
        """
    )

with st.expander("2. 📐 Proper Scoring Rules for Probabilistic Forecasting", expanded=True):
    st.markdown("#### A. Winkler Score (Interval Sharpness & Calibration)")
    st.markdown(
        "For a $(1 - \\alpha)$ prediction interval $[L, U]$ and ground truth observation $y$, the Winkler Score penalizes both interval width and boundary failures:"
    )
    st.latex(
        r"W_\alpha(L, U, y) = (U - L) + \frac{2}{\alpha}(L - y)\mathbb{I}(y < L) + \frac{2}{\alpha}(y - U)\mathbb{I}(y > U)"
    )
    st.markdown(
        """
        * If $y$ falls within $[L, U]$, the score equals the interval width $(U - L)$.
        * If $y$ misses the lower or upper boundary, it is penalized with a steep slope of $\\frac{2}{\alpha}$ (for an 80% CI where $\\alpha = 0.20$, penalty multiplier = $10\\times$).
        * Lower Winkler score indicates narrower, well-calibrated intervals.
        """
    )

    st.markdown("#### B. Pinball (Quantile) Loss")
    st.markdown("Measures the calibration of an individual quantile forecast $q \\in (0, 1)$:")
    st.latex(
        r"L_q(y, \hat{y}) = \max(q(y - \hat{y}), (q - 1)(y - \hat{y}))"
    )

with st.expander("3. 🤖 Amazon Chronos-T5 Zero-Shot Foundation Architecture", expanded=True):
    st.markdown(
        """
        **Amazon Chronos** treats univariate time series as a tokenized language:
        1. **Scaling**: Historical context is mean-scaled to remove arbitrary magnitude differences.
        2. **Quantization**: Real-valued points are mapped into discrete vocabulary tokens.
        3. **Encoder-Decoder**: Pretrained on billions of synthetic and real data points using the T5 transformer backbone.
        4. **Zero-Shot Advantage**: **Zero feature engineering, zero parameter tuning**. The raw 512-hour lookback context is directly converted into autoregressive forecast distributions via ancestral sampling.
        """
    )

with st.expander("4. 🔒 Zero-Leakage Feature Engineering Checklist", expanded=True):
    st.markdown(
        """
        To guarantee zero data contamination into test windows, all temporal transformations obey strict causal constraints:
        
        1. **Minimum Lag Shift**: Any autoregressive feature uses $\\text{shift}(k)$ where $k \\ge \\text{horizon}$ (minimum 24 hours for day-ahead dispatch).
        2. **Right-Aligned Rolling Statistics**: Rolling means, standard deviations, and extrema are calculated on pre-shifted series:
        """
    )
    st.code(
        "df['rolling_mean_24h'] = df['PJME_MW'].shift(24).rolling(24).mean()",
        language="python",
    )
    st.markdown(
        """
        3. **Out-of-Fold Early Stopping**: XGBoost early stopping splits use only the trailing 10% of the training window, strictly isolated from the test horizon.
        """
    )

with st.expander("5. 📚 Academic References & Citations", expanded=False):
    st.markdown(
        """
        * **Amazon Chronos**: Ansari et al., *"Chronos: Learning the Language of Time Series"*, arXiv:2403.07815 (2024).
        * **Quantile Regression Gradient Boosting**: Chen & Guestrin, *"XGBoost: A Scalable Tree Boosting System"*, KDD (2016).
        * **Monte Carlo Dropout Uncertainty**: Gal & Ghahramani, *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"*, ICML (2016).
        * **Interval Evaluation**: Winkler, R. L., *"A Decision-Theoretic Approach to Interval Estimation"*, JASA (1972).
        * **Dataset**: PJM Interconnection, *Hourly Energy Consumption Data (PJME)*, Rob Mulla / PJM LLC.
        """
    )
