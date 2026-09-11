"""
Temporal, cyclical, lag, rolling, and interaction feature engineering.
Strictly zero-leakage: all historical aggregations are shifted by at least horizon_hours.
"""
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar and business-time indicator features.
    Assumes df.index is a DatetimeIndex.
    """
    df = df.copy()
    idx = df.index

    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["dayofyear"] = idx.dayofyear
    df["weekofyear"] = idx.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_business_hour"] = (
        (df["hour"] >= 8) & (df["hour"] <= 18) & (df["is_weekend"] == 0)
    ).astype(int)

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical (Fourier) sine/cosine transformations to avoid integer boundary discontinuities.
    """
    df = df.copy()
    idx = df.index

    # Hour of day (period = 24)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)

    # Day of week (period = 7)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)

    # Month of year (period = 12)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12.0)

    # Day of year (period = 365.25 for annual seasonality)
    df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    horizon_hours: int = 24,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add autoregressive lag features with strict horizon_hours minimum shift
    to prevent target leakage.
    """
    df = df.copy()
    if lags is None:
        # 24h, 48h, 168h (1 week), 336h (2 weeks), 8760h (~1 year)
        lags = [horizon_hours, 48, 168, 336, 8760]

    for lag in lags:
        if lag < horizon_hours:
            raise ValueError(f"Lag {lag} is smaller than horizon {horizon_hours}, causing leakage!")
        df[f"lag_{lag}h"] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    horizon_hours: int = 24,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add right-aligned rolling statistics shifted by horizon_hours.
    """
    df = df.copy()
    if windows is None:
        windows = [24, 48, 168]

    # Pre-shift series by horizon to strictly forbid lookahead
    shifted_target = df[target_col].shift(horizon_hours)

    for w in windows:
        df[f"rolling_mean_{w}h"] = shifted_target.rolling(w).mean()
        df[f"rolling_std_{w}h"] = shifted_target.rolling(w).std()
        df[f"rolling_max_{w}h"] = shifted_target.rolling(w).max()
        df[f"rolling_min_{w}h"] = shifted_target.rolling(w).min()

    return df


def add_ewm_features(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    horizon_hours: int = 24,
    alphas: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Add Exponential Weighted Mean features shifted by horizon_hours.
    """
    df = df.copy()
    if alphas is None:
        alphas = [0.3, 0.1, 0.05]

    shifted_target = df[target_col].shift(horizon_hours)

    for alpha in alphas:
        col_name = f"ewm_alpha_{str(alpha).replace('.', '_')}"
        df[col_name] = shifted_target.ewm(alpha=alpha).mean()

    return df


def add_interaction_features(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """
    Add difference and interaction features.
    """
    df = df.copy()

    # Trend delta: difference between horizon and previous day
    df["delta_24h"] = df[target_col].shift(horizon_hours) - df[target_col].shift(horizon_hours + 24)

    # Cross interaction between hour and day of week
    if "hour" in df.columns and "dayofweek" in df.columns:
        df["hour_x_dow"] = df["hour"] * df["dayofweek"]

    # Business hour flag × lag_24h (amplifies workday morning jump)
    lag_col = f"lag_{horizon_hours}h"
    if "is_business_hour" in df.columns and lag_col in df.columns:
        df["biz_x_lag24"] = df["is_business_hour"] * df[lag_col]

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    horizon_hours: int = 24,
    drop_na: bool = True,
    include_long_lags: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct complete 35-40 feature matrix from raw hourly time series.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) feature matrix and target series.
    """
    data = df.copy()
    data = add_calendar_features(data)
    data = add_cyclical_features(data)

    lags = [lag for lag in [horizon_hours, 48, 168, 336] if lag < len(data) - 10]
    if include_long_lags and len(data) > 9000:
        lags.append(8760)  # 1 year lag if enough data

    data = add_lag_features(data, target_col=target_col, horizon_hours=horizon_hours, lags=lags)
    data = add_rolling_features(data, target_col=target_col, horizon_hours=horizon_hours)
    data = add_ewm_features(data, target_col=target_col, horizon_hours=horizon_hours)
    data = add_interaction_features(data, target_col=target_col, horizon_hours=horizon_hours)

    if drop_na:
        data = data.dropna()

    y = data[target_col]
    X = data.drop(columns=[target_col])
    return X, y
