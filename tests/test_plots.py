"""Tests for Plotly visualizer factories."""
import pandas as pd
import pytest

from src.visualization.plots import (
    create_forecast_plot,
    create_radar_chart,
    create_fold_timeline_chart,
    create_per_fold_metric_chart,
    create_error_box_plot,
    create_coverage_calibration_chart,
    create_shap_bar_chart,
    create_cyclical_polar_chart,
)
from src.evaluation.metrics import rmse


@pytest.fixture
def mock_forecast_df():
    ts = pd.date_range("2024-01-01", periods=48, freq="h")
    records = []
    for m in ["sarimax", "xgboost", "lstm", "chronos"]:
        for t in ts:
            records.append({
                "timestamp": t,
                "actual": 30000.0,
                "q10": 28000.0,
                "q50": 30100.0,
                "q90": 32000.0,
                "fold_id": 1,
                "model": m,
                "horizon": "primary",
            })
    return pd.DataFrame(records)


def test_plots_generation(mock_forecast_df):
    fig_fc = create_forecast_plot(mock_forecast_df)
    assert fig_fc is not None

    summary = {
        "chronos_primary_24h": {"mae": 100, "rmse": 150, "mape": 2.0, "winkler_80": 200, "coverage_80": 0.8},
        "xgboost_primary_24h": {"mae": 120, "rmse": 170, "mape": 2.5, "winkler_80": 220, "coverage_80": 0.78},
    }
    fig_radar = create_radar_chart(summary)
    assert fig_radar is not None

    fig_cov = create_coverage_calibration_chart(summary)
    assert fig_cov is not None

    fold_meta = [{"fold_id": 1, "train_start": "2020-01-01", "train_end": "2023-12-01", "test_start": "2023-12-02", "test_end": "2023-12-09"}]
    fig_gantt = create_fold_timeline_chart(fold_meta)
    assert fig_gantt is not None

    fig_pf = create_per_fold_metric_chart(mock_forecast_df, rmse)
    assert fig_pf is not None

    fig_box = create_error_box_plot(mock_forecast_df)
    assert fig_box is not None

    fig_shap = create_shap_bar_chart({"shap_mean": {"lag_24h": 1.5, "hour_sin": 0.8}})
    assert fig_shap is not None

    fig_polar = create_cyclical_polar_chart()
    assert fig_polar is not None
