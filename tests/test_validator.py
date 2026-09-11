"""Tests for bridge data schema validation."""
from pathlib import Path
import pandas as pd
import pytest

from src.data.validator import (
    validate_forecast_results_df,
    validate_metrics_summary,
    validate_bridge_directory,
)


def test_validate_forecast_results_df_valid():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
        "actual": [30000] * 10,
        "q10": [28000] * 10,
        "q50": [30000] * 10,
        "q90": [32000] * 10,
        "fold_id": [1] * 10,
        "model": ["chronos"] * 10,
        "horizon": ["primary"] * 10,
    })
    is_valid, issues = validate_forecast_results_df(df)
    # Valid schema, though only chronos present
    assert any("Missing expected models" in iss for iss in issues)


def test_validate_crossing_quantiles():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "actual": [30000] * 5,
        "q10": [35000] * 5,  # Crossing! q10 > q50
        "q50": [30000] * 5,
        "q90": [32000] * 5,
        "fold_id": [1] * 5,
        "model": ["xgboost"] * 5,
        "horizon": ["primary"] * 5,
    })
    is_valid, issues = validate_forecast_results_df(df)
    assert not is_valid
    assert any("q10 > q50" in iss for iss in issues)


def test_validate_metrics_summary():
    valid_summary = {
        "chronos_primary_24h": {
            "mae": 1200.0,
            "rmse": 1500.0,
            "mape": 3.5,
            "pinball_q10": 150.0,
            "pinball_q90": 160.0,
            "winkler_80": 2500.0,
            "coverage_80": 0.81,
        }
    }
    is_valid, issues = validate_metrics_summary(valid_summary)
    assert is_valid
    assert len(issues) == 0

    invalid_summary = {
        "chronos_primary_24h": {
            "mae": 1200.0
            # missing rest
        }
    }
    is_valid, issues = validate_metrics_summary(invalid_summary)
    assert not is_valid
    assert len(issues) > 0
