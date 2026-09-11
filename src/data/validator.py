"""Validation functions for bridge data schemas and values."""
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import pandas as pd

EXPECTED_FORECAST_COLUMNS = {
    "timestamp",
    "actual",
    "q10",
    "q50",
    "q90",
    "fold_id",
    "model",
    "horizon",
}

EXPECTED_MODELS = {"sarimax", "xgboost", "lstm", "chronos"}
EXPECTED_HORIZONS = {"primary", "extended"}


def validate_forecast_results_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the structure and integrity of forecast_results DataFrame.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors_or_warnings)
    """
    issues: List[str] = []

    # Check column set
    missing_cols = EXPECTED_FORECAST_COLUMNS - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {sorted(list(missing_cols))}")
        return False, issues

    # Null checks
    null_counts = df[list(EXPECTED_FORECAST_COLUMNS)].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        issues.append(f"Null values detected: {cols_with_nulls.to_dict()}")

    # Models check
    models_found = set(df["model"].unique())
    missing_models = EXPECTED_MODELS - models_found
    if missing_models:
        issues.append(f"Missing expected models: {sorted(list(missing_models))}")

    # Quantile ordering sanity check: q10 <= q50 <= q90
    cross_q10_q50 = (df["q10"] > df["q50"]).sum()
    cross_q50_q90 = (df["q50"] > df["q90"]).sum()
    if cross_q10_q50 > 0:
        issues.append(f"Detected {cross_q10_q50} rows where q10 > q50 (quantile crossing)")
    if cross_q50_q90 > 0:
        issues.append(f"Detected {cross_q50_q90} rows where q50 > q90 (quantile crossing)")

    # Folds check
    folds = sorted(df["fold_id"].unique())
    if len(folds) < 1:
        issues.append("No folds found in forecast results.")

    return len(issues) == 0, issues


def validate_metrics_summary(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate metrics_summary dictionary schema.
    """
    issues: List[str] = []
    required_metric_keys = {
        "mae",
        "rmse",
        "mape",
        "pinball_q10",
        "pinball_q90",
        "winkler_80",
        "coverage_80",
    }

    if not metrics:
        return False, ["metrics_summary is empty."]

    for key, val in metrics.items():
        if not isinstance(val, dict):
            issues.append(f"Entry {key} is not a dictionary.")
            continue
        missing = required_metric_keys - set(val.keys())
        if missing:
            issues.append(f"Entry {key} missing metrics: {sorted(list(missing))}")

    return len(issues) == 0, issues


def validate_bridge_directory(bridge_dir: Path, max_size_mb: float = 5.0) -> Tuple[bool, List[str]]:
    """
    Check that all 4 bridge files exist and do not exceed the target size budget.
    """
    issues: List[str] = []
    required_files = [
        "forecast_results.csv",
        "metrics_summary.json",
        "feature_importance.json",
        "fold_metadata.json",
    ]

    total_size = 0.0
    for fname in required_files:
        fpath = bridge_dir / fname
        if not fpath.exists():
            issues.append(f"Bridge file missing: {fname}")
        else:
            size_mb = fpath.stat().st_size / (1024 * 1024)
            total_size += size_mb

    if total_size > max_size_mb:
        issues.append(f"Total bridge directory size ({total_size:.2f} MB) exceeds limit ({max_size_mb} MB)")

    return len(issues) == 0, issues
