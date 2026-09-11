"""
Utility script to generate realistic benchmark bridge files.
Used for local testing and as an immediate high-fidelity offline demonstration
mirroring the output produced by the Colab master training pipeline.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_all_metrics
from src.data.validator import validate_bridge_directory, validate_forecast_results_df, validate_metrics_summary

BRIDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "bridge"
BRIDGE_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Generate 5 realistic expanding-window folds
# Dataset end: 2018-08-03
# Folds cover 5 x 168h = 840 hours (~35 days)
base_end = pd.Timestamp("2018-08-03 00:00:00")
folds_meta = []
for f in range(5):
    f_id = f + 1
    # Test window: 168h
    test_start = base_end - pd.Timedelta(hours=(5 - f) * 168)
    test_end = test_start + pd.Timedelta(hours=168) - pd.Timedelta(hours=1)
    train_start = pd.Timestamp("2002-01-01 00:00:00")
    train_end = test_start - pd.Timedelta(hours=1)
    folds_meta.append({
        "fold_id": f_id,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "test_start": str(test_start),
        "test_end": str(test_end),
    })

# Save fold_metadata.json
with open(BRIDGE_DIR / "fold_metadata.json", "w", encoding="utf-8") as fp:
    json.dump(folds_meta, fp, indent=2)

# Generate predictions for all models across all 5 folds and horizons
rows = []
models = ["chronos", "xgboost", "lstm", "sarimax"]
horizons = [("primary", 24), ("extended", 168)]

# Base performance factors (Chronos best, then XGBoost, LSTM, SARIMAX)
noise_factors = {
    "chronos": 0.035,   # ~3.5% MAPE
    "xgboost": 0.042,   # ~4.2% MAPE
    "lstm": 0.051,      # ~5.1% MAPE
    "sarimax": 0.078,   # ~7.8% MAPE
}

width_factors = {
    "chronos": 0.075,
    "xgboost": 0.085,
    "lstm": 0.105,
    "sarimax": 0.145,
}

for fold_info in folds_meta:
    f_id = fold_info["fold_id"]
    t_start = pd.Timestamp(fold_info["test_start"])
    
    for h_name, h_len in horizons:
        times = pd.date_range(t_start, periods=h_len, freq="h")
        # Realistic grid demand: 26,000 - 42,000 MW with daily and weekly curve
        hour = times.hour.values
        dow = times.dayofweek.values
        base_demand = (
            31000
            + 6500 * np.sin(2 * np.pi * (hour - 6) / 24)
            - 3000 * (dow >= 5).astype(int)
            + np.random.normal(0, 400, h_len)
        )

        for m in models:
            err_sd = noise_factors[m] * base_demand
            # q50 prediction
            bias = 150 if m == "sarimax" else 0
            q50 = base_demand + np.random.normal(bias, err_sd, h_len)
            
            # Interval half-width
            hw = width_factors[m] * base_demand
            q10 = q50 - hw + np.random.normal(0, 50, h_len)
            q90 = q50 + hw + np.random.normal(0, 50, h_len)
            
            # Enforce monotonicity
            q10 = np.minimum(q10, q50 - 50)
            q90 = np.maximum(q90, q50 + 50)

            for i in range(h_len):
                rows.append({
                    "timestamp": times[i],
                    "actual": round(float(base_demand[i]), 1),
                    "q10": round(float(q10[i]), 1),
                    "q50": round(float(q50[i]), 1),
                    "q90": round(float(q90[i]), 1),
                    "fold_id": f_id,
                    "model": m,
                    "horizon": h_name,
                })

results_df = pd.DataFrame(rows)
results_df.to_csv(BRIDGE_DIR / "forecast_results.csv", index=False)

# Compute metrics_summary.json
metrics_dict = {}
for m in models:
    for h_name, h_len in horizons:
        key_name = f"{m}_{h_name}_{h_len}h"
        sub = results_df[(results_df["model"] == m) & (results_df["horizon"] == h_name)]
        all_met = compute_all_metrics(sub["actual"], sub["q10"], sub["q50"], sub["q90"])
        metrics_dict[key_name] = {k: round(v, 4) for k, v in all_met.items()}

with open(BRIDGE_DIR / "metrics_summary.json", "w", encoding="utf-8") as fp:
    json.dump(metrics_dict, fp, indent=2)

# Generate feature_importance.json (XGBoost)
features = [
    ("lag_24h", 0.2850, 1850.4),
    ("lag_168h", 0.1920, 1240.2),
    ("rolling_mean_24h", 0.1150, 890.1),
    ("hour_sin", 0.0840, 620.3),
    ("hour_cos", 0.0760, 580.9),
    ("biz_x_lag24", 0.0520, 410.5),
    ("delta_24h", 0.0410, 320.4),
    ("dow_sin", 0.0310, 240.6),
    ("dow_cos", 0.0280, 210.2),
    ("ewm_alpha_0_3", 0.0250, 190.8),
    ("rolling_std_24h", 0.0180, 140.3),
    ("hour_x_dow", 0.0150, 115.0),
    ("is_business_hour", 0.0120, 95.2),
    ("is_weekend", 0.0110, 88.4),
    ("lag_48h", 0.0090, 72.1),
    ("rolling_max_24h", 0.0080, 65.0),
    ("month_sin", 0.0070, 55.3),
    ("month_cos", 0.0060, 48.2),
    ("doy_sin", 0.0050, 41.0),
    ("doy_cos", 0.0040, 32.1),
]

feat_imp = {
    "shap_mean": {name: shap for name, shap, _ in features},
    "gain": {name: gain for name, _, gain in features},
}

with open(BRIDGE_DIR / "feature_importance.json", "w", encoding="utf-8") as fp:
    json.dump(feat_imp, fp, indent=2)

# Run validation checks
is_valid_bridge, b_issues = validate_bridge_directory(BRIDGE_DIR)
is_valid_df, df_issues = validate_forecast_results_df(results_df)
is_valid_m, m_issues = validate_metrics_summary(metrics_dict)

print("Validation Results:")
print(f"  Bridge Dir Valid: {is_valid_bridge} ({b_issues})")
print(f"  DataFrame Valid:  {is_valid_df} ({df_issues})")
print(f"  Metrics Valid:    {is_valid_m} ({m_issues})")
print(f"Bridge files generated at: {BRIDGE_DIR.resolve()}")
