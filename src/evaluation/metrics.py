"""
Evaluation metrics for point forecasting and probabilistic uncertainty evaluation.
Includes proper scoring rules (Winkler Score, Pinball Loss) and empirical coverage.
"""
from typing import Any, Dict, Union
import numpy as np
import pandas as pd


def _to_numpy(arr: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    if isinstance(arr, pd.Series):
        return arr.to_numpy()
    return np.asarray(arr, dtype=np.float64)


def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Mean Absolute Error (MW)."""
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Root Mean Squared Error (MW)."""
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error (%)."""
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean(np.abs((yt - yp) / (yt + eps))) * 100.0)


def pinball_loss(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    quantile: float,
) -> float:
    """
    Pinball (Quantile) Loss for a specific quantile in (0, 1).
    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"Quantile must be in (0, 1), got {quantile}")
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    diff = yt - yp
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def winkler_score(
    y_true: Union[np.ndarray, pd.Series],
    q_low: Union[np.ndarray, pd.Series],
    q_high: Union[np.ndarray, pd.Series],
    alpha: float = 0.2,
) -> float:
    """
    Winkler Score for prediction interval [q_low, q_high] at nominal coverage (1 - alpha).
    Standard proper scoring rule for prediction intervals. Lower is better.
    
    Penalizes:
    - Interval width: (q_high - q_low)
    - Lower boundary misses: (2 / alpha) * (q_low - y) when y < q_low
    - Upper boundary misses: (2 / alpha) * (y - q_high) when y > q_high
    """
    yt = _to_numpy(y_true)
    ql = _to_numpy(q_low)
    qh = _to_numpy(q_high)

    interval_width = qh - ql
    penalty_low = (2.0 / alpha) * np.maximum(ql - yt, 0.0)
    penalty_high = (2.0 / alpha) * np.maximum(yt - qh, 0.0)

    score = interval_width + penalty_low + penalty_high
    return float(np.mean(score))


def coverage(
    y_true: Union[np.ndarray, pd.Series],
    q_low: Union[np.ndarray, pd.Series],
    q_high: Union[np.ndarray, pd.Series],
) -> float:
    """
    Empirical coverage rate: proportion of observations that fall within [q_low, q_high].
    For an 80% prediction interval (q10 to q90), ideal coverage is 0.80 (80%).
    """
    yt = _to_numpy(y_true)
    ql = _to_numpy(q_low)
    qh = _to_numpy(q_high)
    inside = (yt >= ql) & (yt <= qh)
    return float(np.mean(inside))


def compute_all_metrics(
    y_true: Union[np.ndarray, pd.Series],
    q10: Union[np.ndarray, pd.Series],
    q50: Union[np.ndarray, pd.Series],
    q90: Union[np.ndarray, pd.Series],
    alpha: float = 0.2,
) -> Dict[str, float]:
    """
    Compute full suite of point and probabilistic metrics.
    """
    return {
        "mae": mae(y_true, q50),
        "rmse": rmse(y_true, q50),
        "mape": mape(y_true, q50),
        "pinball_q10": pinball_loss(y_true, q10, 0.1),
        "pinball_q90": pinball_loss(y_true, q90, 0.9),
        "winkler_80": winkler_score(y_true, q10, q90, alpha=alpha),
        "coverage_80": coverage(y_true, q10, q90),
        "n_samples": float(len(y_true)),
    }
