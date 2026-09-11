"""Data loading utilities for raw PJM time series and bridge files."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import pandas as pd


def load_raw_pjme(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw PJME hourly demand CSV, format timestamps, and resolve duplicates.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to PJME_hourly.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by UTC/naive DatetimeIndex with 'PJME_MW' target column.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path.resolve()}")

    df = pd.read_csv(path)
    # Standardize column naming
    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), "Datetime")
    val_col = next((c for c in df.columns if "pjme" in c.lower() or "mw" in c.lower()), "PJME_MW")

    df = df.rename(columns={date_col: "Datetime", val_col: "PJME_MW"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"])
    df = df.set_index("Datetime")

    # Interpolate missing hourly steps if any exist
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    if len(full_idx) != len(df):
        df = df.reindex(full_idx)
        df["PJME_MW"] = df["PJME_MW"].interpolate(method="time")

    return df


def load_forecast_results(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the core bridge forecast_results.csv with parsed timestamps.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Bridge file forecast_results.csv not found at {path.resolve()}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def load_metrics_summary(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metrics_summary.json containing aggregate metrics.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Bridge file metrics_summary.json not found at {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_importance(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load feature_importance.json containing XGBoost gain & SHAP metrics.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Bridge file feature_importance.json not found at {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fold_metadata(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load fold_metadata.json containing fold timeline boundaries.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Bridge file fold_metadata.json not found at {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
