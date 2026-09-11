"""Tests for forecasting and probabilistic evaluation metrics."""
import numpy as np
import pytest

from src.evaluation.metrics import (
    mae,
    rmse,
    mape,
    pinball_loss,
    winkler_score,
    coverage,
    compute_all_metrics,
)


def test_point_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 315.0])

    # MAE = (|10| + |-10| + |15|) / 3 = 35 / 3 = 11.6667
    assert pytest.approx(mae(y_true, y_pred), rel=1e-3) == 11.6667

    # RMSE = sqrt((100 + 100 + 225) / 3) = sqrt(141.6667) = 11.9023
    assert pytest.approx(rmse(y_true, y_pred), rel=1e-3) == 11.9023

    # MAPE = (10/100 + 10/200 + 15/300)/3 * 100 = (0.1 + 0.05 + 0.05)/3 * 100 = 6.6667%
    assert pytest.approx(mape(y_true, y_pred), rel=1e-3) == 6.6667


def test_pinball_loss():
    y_true = np.array([100.0])
    # Under-prediction: y_pred = 80 -> error = 20 > 0
    # For q = 0.9: loss = 0.9 * 20 = 18
    assert pytest.approx(pinball_loss(y_true, np.array([80.0]), quantile=0.9)) == 18.0

    # Over-prediction: y_pred = 120 -> error = -20 < 0
    # For q = 0.1: loss = (0.1 - 1) * (-20) = 0.9 * 20 = 18
    assert pytest.approx(pinball_loss(y_true, np.array([120.0]), quantile=0.1)) == 18.0

    with pytest.raises(ValueError):
        pinball_loss(y_true, y_true, quantile=1.5)


def test_winkler_score_and_coverage():
    y_true = np.array([100.0, 150.0, 200.0])
    q_low = np.array([90.0, 140.0, 190.0])
    q_high = np.array([110.0, 160.0, 210.0])

    # All points inside [q_low, q_high]
    # Coverage = 100%
    assert coverage(y_true, q_low, q_high) == 1.0

    # Winkler Score: interval_width = 20 for all 3, penalty = 0
    # Winkler = 20.0
    assert winkler_score(y_true, q_low, q_high, alpha=0.2) == 20.0

    # Case with miss below q_low
    y_miss = np.array([80.0])  # 10 below q_low=90
    # width = 20, penalty = (2 / 0.2) * (90 - 80) = 10 * 10 = 100 -> score = 120
    assert pytest.approx(winkler_score(y_miss, np.array([90.0]), np.array([110.0]), alpha=0.2)) == 120.0


def test_compute_all_metrics():
    y = np.array([100.0, 200.0, 300.0])
    q10 = np.array([80.0, 180.0, 280.0])
    q50 = np.array([100.0, 200.0, 300.0])
    q90 = np.array([120.0, 220.0, 320.0])

    res = compute_all_metrics(y, q10, q50, q90)
    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
    assert res["coverage_80"] == 1.0
    assert res["winkler_80"] == 40.0
