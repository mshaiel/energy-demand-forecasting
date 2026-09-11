"""Tests for feature engineering logic and leakage prevention."""
import numpy as np
import pandas as pd
import pytest

from src.features.temporal import (
    add_calendar_features,
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
    add_ewm_features,
    add_interaction_features,
    build_feature_matrix,
)


@pytest.fixture
def synthetic_ts():
    """Create 200 hours of synthetic hourly demand series."""
    dates = pd.date_range("2024-01-01 00:00:00", periods=600, freq="h")
    # Base pattern: 30,000 MW + sinusoidal daily pattern
    hours = np.arange(len(dates))
    values = 30000 + 5000 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 50, len(dates))
    df = pd.DataFrame({"PJME_MW": values}, index=dates)
    return df


def test_calendar_features(synthetic_ts):
    df = add_calendar_features(synthetic_ts)
    expected_cols = [
        "hour",
        "dayofweek",
        "month",
        "quarter",
        "dayofyear",
        "weekofyear",
        "is_weekend",
        "is_business_hour",
    ]
    for c in expected_cols:
        assert c in df.columns
    assert df["hour"].min() >= 0 and df["hour"].max() <= 23
    assert df["dayofweek"].min() >= 0 and df["dayofweek"].max() <= 6
    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_cyclical_features(synthetic_ts):
    df = add_cyclical_features(synthetic_ts)
    for prefix in ["hour", "dow", "month", "doy"]:
        sin_col = f"{prefix}_sin"
        cos_col = f"{prefix}_cos"
        assert sin_col in df.columns
        assert cos_col in df.columns
        # Sine and cosine must be within [-1.0, 1.0]
        assert df[sin_col].min() >= -1.0 - 1e-6 and df[sin_col].max() <= 1.0 + 1e-6
        assert df[cos_col].min() >= -1.0 - 1e-6 and df[cos_col].max() <= 1.0 + 1e-6
        # Pythagorean identity for hour: sin^2 + cos^2 == 1
        np.testing.assert_allclose(df["hour_sin"] ** 2 + df["hour_cos"] ** 2, 1.0, atol=1e-5)


def test_lag_leakage_prevention(synthetic_ts):
    # Setting a lag smaller than horizon must raise ValueError to prevent leakage
    with pytest.raises(ValueError, match="causing leakage"):
        add_lag_features(synthetic_ts, horizon_hours=24, lags=[12, 24])

    df = add_lag_features(synthetic_ts, horizon_hours=24, lags=[24, 48])
    assert "lag_24h" in df.columns
    assert "lag_48h" in df.columns
    # Check that lag_24h matches PJME_MW shifted by 24
    assert df["lag_24h"].iloc[24] == synthetic_ts["PJME_MW"].iloc[0]


def test_rolling_and_ewm(synthetic_ts):
    df = add_rolling_features(synthetic_ts, horizon_hours=24, windows=[24])
    assert "rolling_mean_24h" in df.columns
    # Rolling feature at step 24 must be NaN because shift(24) only has 1 valid value
    assert np.isnan(df["rolling_mean_24h"].iloc[24])

    df_ewm = add_ewm_features(synthetic_ts, horizon_hours=24, alphas=[0.1])
    assert "ewm_alpha_0_1" in df_ewm.columns


def test_build_feature_matrix(synthetic_ts):
    X, y = build_feature_matrix(synthetic_ts, horizon_hours=24, drop_na=True, include_long_lags=False)
    assert len(X) == len(y)
    assert len(X) > 0
    assert "PJME_MW" not in X.columns
    assert not X.isnull().any().any()
