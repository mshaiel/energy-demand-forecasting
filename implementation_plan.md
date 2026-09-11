# Energy Demand Forecasting — ML Portfolio Project: Full Execution Architecture

> **Domain**: Probabilistic Time-Series Energy Demand Forecasting  
> **Dataset**: PJME_hourly.csv (PJM Interconnection, Eastern US grid)  
> **Narrative**: 4-Generation model benchmark with rolling-origin backtesting and quantile outputs  
> **Horizon**: 24h (primary) + 168h (extended)  
> **Backtesting**: 5-fold expanding window  
> **Foundation Model**: Amazon Chronos-T5  
> **Dashboard**: Dark-mode SCADA-style Streamlit app reading pre-computed CSVs

---

## 1. Git Repository Structure

```
energy-demand-forecasting/
│
├── README.md                         ← Landing page (badges, GIF demo, benchmark table)
├── LICENSE
├── .gitignore
├── requirements_local.txt            ← Streamlit + lightweight libs only
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb     ← [Colab] EDA, stationarity tests, ACF/PACF plots
│   ├── 02_feature_engineering.ipynb  ← [Colab] All temporal + cyclical feature construction
│   ├── 03_backtesting_framework.ipynb← [Colab] Expanding-window fold generator
│   ├── 04_gen1_sarimax.ipynb         ← [Colab] Classical baseline
│   ├── 05_gen2_xgboost.ipynb         ← [Colab] Gradient boosting w/ feature importance
│   ├── 06_gen3_lstm.ipynb            ← [Colab] Multivariate LSTM + dropout uncertainty
│   ├── 07_gen4_chronos.ipynb         ← [Colab] Zero-shot Chronos-T5 inference
│   └── 08_results_export.ipynb       ← [Colab] Bridge file assembly & validation
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 ← CSV ingestion, dtype enforcement, UTC alignment
│   │   └── validator.py              ← Schema checks for bridge CSVs on local load
│   ├── features/
│   │   ├── __init__.py
│   │   └── temporal.py               ← Feature engineering logic (mirrors Colab notebook)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                ← MAE, RMSE, MAPE, CRPS, WQL implementations
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                  ← Reusable Plotly figure factories
│
├── dashboard/
│   ├── app.py                        ← Streamlit entry point
│   ├── pages/
│   │   ├── 01_overview.py            ← Benchmark leaderboard + headline metrics
│   │   ├── 02_24h_forecast.py        ← 24h forecast explorer
│   │   ├── 03_168h_forecast.py       ← 168h extended forecast explorer
│   │   ├── 04_backtesting.py         ← Fold-by-fold performance decomposition
│   │   ├── 05_feature_importance.py  ← XGBoost SHAP + LSTM attention weights
│   │   └── 06_methodology.py         ← Static methodology explainer (LaTeX equations)
│   └── assets/
│       └── style.css                 ← Custom dark-mode CSS overrides
│
├── data/
│   ├── raw/                          ← .gitignore'd — downloaded by Colab
│   │   └── PJME_hourly.csv
│   └── bridge/                       ← ✅ Committed to Git (< 5 MB total)
│       ├── forecast_results.csv      ← Core bridge: predictions per fold per model
│       ├── metrics_summary.json      ← Aggregated metrics per model per horizon
│       ├── feature_importance.json   ← XGBoost gain + SHAP values
│       └── fold_metadata.json        ← Fold date boundaries for timeline plot
│
├── configs/
│   ├── model_config.yaml             ← Hyperparameters for all 4 models
│   └── backtest_config.yaml          ← Fold definitions, horizon settings
│
└── tests/
    ├── test_features.py
    ├── test_metrics.py
    └── test_validator.py
```

> **Key principle**: `data/raw/` is in `.gitignore`. Only `data/bridge/` is committed. GitHub repo stays clean and recruiters can clone + run `streamlit run dashboard/app.py` immediately with no Kaggle credentials.

---

## 2. Temporal & Cyclical Feature Engineering (XGBoost Baseline)

The XGBoost model's competitiveness hinges entirely on feature quality. The PJME dataset is purely univariate, so features substitute for structural signals captured implicitly by LSTM and Chronos.

### 2.1 Calendar Features (Raw Categorical)

| Feature | Derivation | Notes |
|---|---|---|
| `hour` | `df.index.hour` | 0–23, kept as integer |
| `dayofweek` | `df.index.dayofweek` | 0=Monday, 6=Sunday |
| `month` | `df.index.month` | 1–12 |
| `quarter` | `df.index.quarter` | 1–4 |
| `dayofyear` | `df.index.dayofyear` | 1–366 |
| `weekofyear` | `df.index.isocalendar().week` | ISO week |
| `is_weekend` | `dayofweek >= 5` | Binary |
| `is_business_hour` | `hour in [8..18] & !is_weekend` | Binary |

### 2.2 Cyclical (Fourier) Encodings

**Critical**: Raw integer hour/month features create artificial discontinuities (hour 23 ≠ hour 0 in integer space, but they are adjacent in real time). Sine/cosine encoding wraps the cycle correctly.

```python
import numpy as np

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    # Hour of day (period = 24)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Day of week (period = 7)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # Month of year (period = 12)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Day of year — captures annual seasonality (period = 365.25)
    df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    return df
```

### 2.3 Lag Features (Autoregressive Signal)

These are the most predictive features. **Strict rule**: lags must always be ≥ the forecast horizon to prevent leakage. For 24h horizon, minimum lag = 24h.

```
lag_24h     ← Same hour, previous day (strongest single feature)
lag_48h     ← Same hour, 2 days ago
lag_168h    ← Same hour, previous week (dominant weekly pattern)
lag_336h    ← Same hour, 2 weeks ago
lag_8760h   ← Same hour, previous year (strong annual baseline)
```

### 2.4 Rolling Window Statistics

Computed on a **right-aligned, closed window** using only past values. Window must be shifted by the forecast horizon to avoid leakage.

```python
# Example for 24h horizon — shift ensures no look-ahead
for window in [24, 48, 168]:
    df[f'rolling_mean_{window}h'] = (
        df['PJME_MW'].shift(24).rolling(window).mean()
    )
    df[f'rolling_std_{window}h'] = (
        df['PJME_MW'].shift(24).rolling(window).std()
    )
    df[f'rolling_max_{window}h'] = (
        df['PJME_MW'].shift(24).rolling(window).max()
    )
```

### 2.5 Exponential Weighted Mean (EWM)

Captures recency-weighted trend without a hard window cutoff:

```python
for alpha in [0.3, 0.1, 0.05]:
    df[f'ewm_alpha_{alpha}'] = (
        df['PJME_MW'].shift(24).ewm(alpha=alpha).mean()
    )
```

### 2.6 Interaction & Difference Features

```python
# Trend signal: difference from same hour yesterday
df['delta_24h'] = df['PJME_MW'].shift(24) - df['PJME_MW'].shift(48)

# Hour × Weekday interaction (captures "Monday 9am" vs "Sunday 9am")
df['hour_x_dow'] = df.index.hour * df.index.dayofweek

# Business hour flag × lag (amplify weekday morning signal)
df['biz_x_lag24'] = df['is_business_hour'] * df['lag_24h']
```

### 2.7 Probabilistic Output from XGBoost

XGBoost natively supports quantile regression via `objective='reg:quantileerror'` (XGBoost ≥ 2.0). Train three separate models per fold:

```python
models = {}
for q in [0.1, 0.5, 0.9]:
    model = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=q,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models[q] = model
```

**Final feature count**: ~35–40 features. Use SHAP (`shap.TreeExplainer`) post-training for the dashboard's feature importance page.

---

## 3. Colab Notebook Workflow — Training Loops & Bridge Export

### 3.1 Environment Setup (Top of each notebook)

```python
# Cell 1 — Install (run once per session)
!pip install -q chronos-forecasting xgboost lightgbm shap kaggle

# Cell 2 — Kaggle data download
import os
os.environ['KAGGLE_USERNAME'] = "your_username"   # or use Colab secrets
os.environ['KAGGLE_KEY'] = "your_api_key"
!kaggle datasets download -d robikscube/hourly-energy-consumption -p /content/data --unzip
```

### 3.2 Expanding-Window Backtesting Framework

The fold generator is the methodological backbone. It lives in `03_backtesting_framework.ipynb` and is imported (copy-pasted as a utility cell) into each model notebook.

```python
from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd

@dataclass
class BacktestFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def expanding_window_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    horizon_hours: int = 24,
    min_train_years: int = 2,
    gap_hours: int = 0,           # buffer between train end and test start
) -> Iterator[BacktestFold]:
    """
    Generates n_folds expanding-window folds.
    The test windows are contiguous, non-overlapping, and cover the
    final n_folds * horizon_hours of the dataset.
    
    Leakage prevention: gap_hours ensures no same-timestamp overlap.
    """
    total_hours = len(df)
    test_pool = n_folds * horizon_hours
    
    # Fixed test region starts after minimum training period
    test_region_start_idx = total_hours - test_pool
    
    for fold in range(n_folds):
        test_start_idx = test_region_start_idx + fold * horizon_hours
        test_end_idx   = test_start_idx + horizon_hours
        train_end_idx  = test_start_idx - gap_hours - 1
        
        yield BacktestFold(
            fold_id    = fold + 1,
            train_start = df.index[0],
            train_end   = df.index[train_end_idx],
            test_start  = df.index[test_start_idx],
            test_end    = df.index[test_end_idx - 1],
        )
```

**5-fold configuration for PJME (2002–2018, ~16 years)**:
- Min training size: 2 years
- Test windows: 5 × 168h = last 840 hours (~35 days) of data
- Each fold's training set expands by 168h (1 week) from the previous

### 3.3 Per-Model Training Loop Pattern

All 4 model notebooks follow the same outer structure:

```python
all_fold_results = []

for fold in expanding_window_folds(df, n_folds=5, horizon_hours=168):
    # --- Data Slicing ---
    train_df = df[fold.train_start : fold.train_end]
    test_df  = df[fold.test_start  : fold.test_end]
    
    # --- Model-specific training (see per-model sections below) ---
    q10, q50, q90 = train_and_predict(train_df, test_df)
    
    # --- Standardized Record ---
    fold_df = pd.DataFrame({
        'timestamp':  test_df.index,
        'actual':     test_df['PJME_MW'].values,
        'q10':        q10,
        'q50':        q50,
        'q90':        q90,
        'fold_id':    fold.fold_id,
        'model':      'MODEL_NAME',   # 'sarimax' | 'xgboost' | 'lstm' | 'chronos'
        'horizon':    'primary',       # or 'extended'
    })
    all_fold_results.append(fold_df)

results_df = pd.concat(all_fold_results, ignore_index=True)
```

### 3.4 Generation 1 — SARIMAX

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_and_predict_sarimax(train_df, test_df):
    # Fit on daily resampled data to reduce SARIMAX computational cost
    train_daily = train_df['PJME_MW'].resample('D').mean()
    
    model = SARIMAX(
        train_daily,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),  # 7-day weekly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200)
    
    # Forecast with confidence intervals
    forecast = fitted.get_forecast(steps=len(test_df) // 24)
    pred_mean = forecast.predicted_mean
    ci        = forecast.conf_int(alpha=0.2)  # 80% CI → approx q10/q90
    
    # Upsample back to hourly
    q50 = pred_mean.resample('H').interpolate()[:len(test_df)]
    q10 = ci.iloc[:, 0].resample('H').interpolate()[:len(test_df)]
    q90 = ci.iloc[:, 1].resample('H').interpolate()[:len(test_df)]
    
    return q10.values, q50.values, q90.values
```

### 3.5 Generation 2 — XGBoost (Quantile)

```python
def train_and_predict_xgboost(train_df, test_df):
    X_train, y_train = build_feature_matrix(train_df)
    X_test,  y_test  = build_feature_matrix(test_df)
    
    # Validation split — last 10% of training data (still no leakage)
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]
    
    preds = {}
    for q in [0.1, 0.5, 0.9]:
        model = xgb.XGBRegressor(
            objective='reg:quantileerror', quantile_alpha=q,
            n_estimators=2000, learning_rate=0.03,
            max_depth=7, subsample=0.8, colsample_bytree=0.75,
            min_child_weight=5, gamma=0.1,
            early_stopping_rounds=50, tree_method='hist', device='cuda',
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds[q] = model.predict(X_test)
    
    return preds[0.1], preds[0.5], preds[0.9]
```

> **Note**: `device='cuda'` works on Colab's T4/A100. On your local GTX 1050, change to `device='cpu'` or `tree_method='auto'`.

### 3.6 Generation 3 — Multivariate LSTM

**Architecture**:
- Input: `(batch, seq_len=168, n_features)` — 168h lookback window, multivariate
- Features: `[PJME_MW_normalized, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, is_weekend, is_business_hour]` → 9 channels
- Output: `(batch, horizon=24)` — direct multi-step output

```python
import torch
import torch.nn as nn

class ProbabilisticLSTM(nn.Module):
    """
    LSTM with MC Dropout for uncertainty quantification.
    Dropout layers remain active at inference time to sample 
    a distribution of outputs (Monte Carlo Dropout).
    """
    def __init__(self, n_features, hidden_dim=256, n_layers=2, 
                 dropout=0.3, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, horizon)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last hidden state
        return self.fc(out)

def mc_dropout_predict(model, X_tensor, n_samples=200):
    """
    Run n_samples stochastic forward passes with dropout active.
    Returns q10, q50, q90 from the empirical distribution.
    """
    model.train()  # keep dropout active
    samples = torch.stack([model(X_tensor) for _ in range(n_samples)])
    # samples shape: (n_samples, batch, horizon)
    q10 = torch.quantile(samples, 0.10, dim=0).detach().numpy()
    q50 = torch.quantile(samples, 0.50, dim=0).detach().numpy()
    q90 = torch.quantile(samples, 0.90, dim=0).detach().numpy()
    return q10, q50, q90
```

**Training loop**:
```python
# Loss: Pinball (quantile) loss for q50, L1 for coverage
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
```

### 3.7 Generation 4 — Amazon Chronos-T5 (Zero-Shot)

```python
from chronos import ChronosPipeline
import torch

# Load pretrained model — "small" variant for Colab memory budget
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",   # upgrade to "base" if A100 runtime
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

def predict_chronos_fold(train_df, test_df, horizon=24, n_samples=200):
    """
    Zero-shot inference: NO fine-tuning, NO feature engineering.
    Chronos takes raw univariate history as context.
    """
    context = torch.tensor(train_df['PJME_MW'].values, dtype=torch.float32)
    
    # Returns (n_samples, horizon) — native probabilistic output
    forecast_samples = pipeline.predict(
        context=context.unsqueeze(0),   # shape: (1, context_len)
        prediction_length=horizon,
        num_samples=n_samples,
    )  # shape: (1, n_samples, horizon)
    
    samples = forecast_samples[0].numpy()  # (n_samples, horizon)
    q10 = np.quantile(samples, 0.10, axis=0)
    q50 = np.quantile(samples, 0.50, axis=0)
    q90 = np.quantile(samples, 0.90, axis=0)
    
    return q10, q50, q90
```

> **Critical portfolio note**: In your README and notebook, explicitly call out that Chronos receives **zero context engineering** — no feature extraction, no normalization pipeline. The comparison is intentionally unfair to classical methods and that asymmetry is the point of the Gen 4 benchmark.

### 3.8 Evaluation Metrics

```python
import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def pinball_loss(y_true, y_pred, quantile):
    """Pinball / quantile loss — measures calibration of a single quantile."""
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

def winkler_score(y_true, q_low, q_high, alpha=0.2):
    """
    Winkler Score for interval [q_low, q_high] at level alpha.
    Lower is better. Penalizes both wide intervals and coverage failures.
    """
    interval_width = q_high - q_low
    penalty_low  = (2 / alpha) * np.maximum(q_low - y_true, 0)
    penalty_high = (2 / alpha) * np.maximum(y_true - q_high, 0)
    return np.mean(interval_width + penalty_low + penalty_high)

def coverage(y_true, q_low, q_high):
    """Empirical coverage rate — should be close to (1 - alpha) = 0.80."""
    return np.mean((y_true >= q_low) & (y_true <= q_high))
```

> **Why Winkler Score**: It's the proper scoring rule for interval forecasts, penalizing both overconfidence (misses) and underconfidence (overly wide bands). Including it signals graduate-level understanding of probabilistic evaluation.

### 3.9 Bridge File Export (08_results_export.ipynb)

```python
import json, os

OUTPUT_DIR = '/content/bridge_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Bridge File 1: forecast_results.csv ──────────────────────────────────────
# Schema: timestamp, actual, q10, q50, q90, fold_id, model, horizon
all_results_df.to_csv(f'{OUTPUT_DIR}/forecast_results.csv', index=False)

# ── Bridge File 2: metrics_summary.json ──────────────────────────────────────
metrics = {}
for model_name in ['sarimax', 'xgboost', 'lstm', 'chronos']:
    for horizon in ['primary_24h', 'extended_168h']:
        subset = all_results_df[
            (all_results_df['model'] == model_name) &
            (all_results_df['horizon'] == horizon)
        ]
        key = f'{model_name}_{horizon}'
        metrics[key] = {
            'mae':          float(mae(subset['actual'], subset['q50'])),
            'rmse':         float(rmse(subset['actual'], subset['q50'])),
            'mape':         float(mape(subset['actual'], subset['q50'])),
            'pinball_q10':  float(pinball_loss(subset['actual'], subset['q10'], 0.1)),
            'pinball_q90':  float(pinball_loss(subset['actual'], subset['q90'], 0.9)),
            'winkler_80':   float(winkler_score(subset['actual'], subset['q10'], subset['q90'])),
            'coverage_80':  float(coverage(subset['actual'], subset['q10'], subset['q90'])),
            'n_samples':    int(len(subset)),
        }

with open(f'{OUTPUT_DIR}/metrics_summary.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ── Bridge File 3: feature_importance.json ───────────────────────────────────
# (Populated in XGBoost notebook using SHAP values)
# Schema: {"gain": {feature: value, ...}, "shap_mean": {feature: value, ...}}

# ── Bridge File 4: fold_metadata.json ────────────────────────────────────────
fold_meta = [
    {
        'fold_id': f.fold_id,
        'train_start': str(f.train_start),
        'train_end':   str(f.train_end),
        'test_start':  str(f.test_start),
        'test_end':    str(f.test_end),
    }
    for f in folds
]
with open(f'{OUTPUT_DIR}/fold_metadata.json', 'w') as f:
    json.dump(fold_meta, f, indent=2)

# ── Size validation ───────────────────────────────────────────────────────────
for fname in os.listdir(OUTPUT_DIR):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f'{fname}: {size_mb:.2f} MB {"✅" if size_mb < 2 else "⚠️ TRIM NEEDED"}')
```

Expected sizes:
| File | Estimated Size |
|---|---|
| `forecast_results.csv` | ~2–3 MB |
| `metrics_summary.json` | < 10 KB |
| `feature_importance.json` | < 50 KB |
| `fold_metadata.json` | < 5 KB |
| **Total** | **< 4 MB** ✅ |

---

## 4. Streamlit Dashboard Architecture

### 4.1 Design System (assets/style.css)

**Palette**:
| Token | Hex | Usage |
|---|---|---|
| `--bg-primary` | `#0a0e1a` | Page background |
| `--bg-card` | `#111827` | Card/panel backgrounds |
| `--bg-border` | `#1f2937` | Card borders |
| `--accent-primary` | `#00d4ff` | Cyan — primary highlights |
| `--accent-secondary` | `#7c3aed` | Purple — secondary model |
| `--accent-success` | `#10b981` | Green — good metrics |
| `--accent-warning` | `#f59e0b` | Amber — medium metrics |
| `--accent-danger` | `#ef4444` | Red — poor metrics |
| `--text-primary` | `#f9fafb` | Primary text |
| `--text-muted` | `#6b7280` | Metadata/labels |

**Typography**: Import `Inter` + `JetBrains Mono` from Google Fonts. Inter for body, JetBrains Mono for metric values and code snippets.

**Chart theme** (applied globally to all Plotly figures):
```python
CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.8)',
    font=dict(family='Inter', color='#f9fafb'),
    xaxis=dict(gridcolor='#1f2937', zerolinecolor='#374151'),
    yaxis=dict(gridcolor='#1f2937', zerolinecolor='#374151'),
)
MODEL_COLORS = {
    'sarimax':  '#6b7280',
    'xgboost':  '#f59e0b',
    'lstm':     '#7c3aed',
    'chronos':  '#00d4ff',
}
```

### 4.2 Page-by-Page Component Specification

---

#### Page 1: Overview (01_overview.py) — The Landing Page

**Goal**: Instantly impress a recruiter who has 30 seconds to evaluate.

**Components**:

1. **Hero header**: Large title "⚡ Grid Intelligence: Probabilistic Energy Demand Forecasting" with subtitle line. Cyan gradient text.

2. **KPI Cards Row** (4 columns, one per model): Each card shows:
   - Model name + generation badge ("GEN 4 • FOUNDATION MODEL")
   - Best fold RMSE (primary metric, large font, JetBrains Mono)
   - Delta vs. SARIMAX baseline (green arrow if better)
   - 80% Coverage Rate badge (color-coded: green if 75–85%, amber otherwise)

3. **Benchmark Leaderboard Table**: Sortable table (24h horizon, all 5 folds averaged):
   ```
   | Rank | Model     | MAE ↓ | RMSE ↓ | MAPE ↓ | Winkler ↓ | Coverage |
   |------|-----------|--------|--------|--------|-----------|----------|
   | 🥇 1 | Chronos   | ...    | ...    | ...    | ...       | ...%     |
   | 🥈 2 | XGBoost   | ...    | ...    | ...    | ...       | ...%     |
   | 🥉 3 | LSTM      | ...    | ...    | ...    | ...       | ...%     |
   |    4 | SARIMAX   | ...    | ...    | ...    | ...       | ...%     |
   ```
   Color-coded cells (green/amber/red) using `st.dataframe` with Pandas Styler.

4. **Radar Chart**: Plotly radar/spider chart with axes for each metric (normalized 0–1, inverted for lower-is-better). All 4 models overlaid. This is visually distinctive and immediately shows multi-dimensional comparison.

5. **Dataset provenance card**: Small info box — "PJME Hourly | 2002–2018 | 140,256 observations | PJM Interconnection, Eastern US Grid"

---

#### Page 2: 24h Forecast Explorer (02_24h_forecast.py)

**Goal**: Let the viewer interactively explore model outputs fold-by-fold.

**Controls sidebar**:
- `st.selectbox` — "Forecast Fold" (1–5) — updates timestamp range in header
- `st.multiselect` — "Models to Display" — toggles which model traces appear
- `st.toggle` — "Show Uncertainty Bands" — show/hide q10–q90 shaded regions
- `st.toggle` — "Show Actual Demand" — toggle the ground truth trace

**Main chart** (Plotly line chart):
- Traces: Actual (white dashed), q50 per model (solid, model color), q10/q90 (shaded fill with 20% opacity matching model color)
- X-axis: Timestamp (hourly), hover shows exact MW value
- Y-axis: MW demand, formatted with comma separator
- Annotation: Fold date range, training window size

**Metrics panel below chart** (3 columns):
- MAE, RMSE, MAPE for selected fold (updates dynamically with fold selector)
- Winkler score + empirical coverage rate
- "Best model this fold" badge

---

#### Page 3: 168h Extended Forecast Explorer (03_168h_forecast.py)

Same layout as Page 2, but:
- X-axis tick format changes to Day/Hour
- Add secondary Y-axis toggle for a simple time-of-day heatmap showing uncertainty band width by hour (wider at 2–5am, narrower at peak hours)
- Include a "Week Pattern Decomposition" subplot showing weekday vs. weekend demand profile

---

#### Page 4: Backtesting Analysis (04_backtesting.py)

**Goal**: Demonstrate methodological rigor.

**Components**:

1. **Fold Timeline Chart**: Horizontal Gantt-style chart showing the 5 training + test windows across 2002–2018. Training windows are dark grey bars, test windows are cyan highlighted segments. This visually proves no temporal leakage.

2. **Per-Fold Performance Line Chart**: Line chart with fold_id on X, metric on Y. One line per model. Add `st.selectbox` to switch between MAE / RMSE / MAPE / Winkler. Shows performance stability (or degradation) across time — this is a key signal of model robustness.

3. **Box Plot**: Distribution of errors across all 5 folds, one box per model. Narrower IQR = more consistent. Plotly box with individual points shown.

4. **Coverage Calibration Chart**: Bar chart of empirical coverage vs. ideal 80% line. A horizontal dashed cyan line at 0.80. Each model's bar is colored green if within ±5%, amber if ±10%, red otherwise.

---

#### Page 5: Feature Intelligence (05_feature_importance.py)

**Goal**: Show depth of feature engineering understanding.

**Components**:

1. **SHAP Summary Bar Chart**: Horizontal bar chart of top-20 features by mean |SHAP value|. Color encodes feature category (lag = amber, cyclical = cyan, rolling = purple, calendar = grey).

2. **Feature Category Pie/Donut**: Aggregated SHAP importance by category. Shows "lag features account for 48% of predictive signal" — immediately communicates understanding.

3. **Gain vs. SHAP Comparison Table**: Two-column table showing feature rank by gain vs. rank by SHAP. Highlights discrepancies (gain can be misleading; SHAP is model-agnostic).

4. **Cyclical Encoding Explainer**: Static plotly polar chart showing a unit circle with hour_sin/hour_cos encoding at 4 representative hours (midnight, 6am, noon, 6pm). This is a recruiting conversation starter — it shows you understand *why* cyclical encoding is necessary.

---

#### Page 6: Methodology (06_methodology.py)

**Goal**: Act as an in-app paper abstract for recruiters who want depth.

**Components**:
- `st.expander` blocks for each section (collapses by default, no visual clutter):
  - **Backtesting Protocol**: Expanding window definition, fold dates table
  - **Probabilistic Evaluation**: Winkler Score formula rendered with `st.latex`, Pinball Loss formula
  - **Chronos Zero-Shot**: Brief explanation of T5 pretraining, why no feature engineering is a feature not a bug
  - **Temporal Leakage Prevention**: Numbered checklist of all lag/rolling shift decisions
- GitHub link, Kaggle dataset citation, references to Chronos paper (arXiv:2403.07815)

---

### 4.3 app.py Entry Point

```python
import streamlit as st

st.set_page_config(
    page_title="Grid Intelligence | Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject dark-mode CSS
with open("dashboard/assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Data loading with caching (reads bridge CSVs once per session)
@st.cache_data
def load_forecast_results():
    import pandas as pd
    df = pd.read_csv("data/bridge/forecast_results.csv", parse_dates=['timestamp'])
    return df

@st.cache_data
def load_metrics():
    import json
    with open("data/bridge/metrics_summary.json") as f:
        return json.load(f)

# Make data available to all pages via session state
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = load_forecast_results()
if 'metrics' not in st.session_state:
    st.session_state.metrics = load_metrics()
```

---

## 5. GitHub Repository Presentation Layer

### 5.1 README.md Structure

```markdown
# ⚡ Grid Intelligence: Probabilistic Energy Demand Forecasting

> A rigorous benchmark of four generations of time-series models on PJM 
> grid data, featuring rolling-origin backtesting and quantile uncertainty estimation.

[Live Dashboard](your-streamlit-link) • [Colab Notebook](colab-link) • [Dataset](kaggle-link)

## 📊 Benchmark Results (5-Fold Expanding Window, 24h Horizon)

| Model | Generation | MAE ↓ | RMSE ↓ | MAPE ↓ | 80% Coverage |
|-------|-----------|--------|--------|--------|-------------|
| Chronos-T5 | Foundation (zero-shot) | ... | ... | ... | ... |
| XGBoost | Gradient Boosting | ... | ... | ... | ... |
| LSTM + MC Dropout | Deep Learning | ... | ... | ... | ... |
| SARIMAX | Classical | ... | ... | ... | ... |

## 🏗️ Architecture

[Hybrid workflow diagram image here]

## 🔬 Methodology Highlights
- **No temporal leakage**: Rolling-origin expanding window (5 folds)
- **Probabilistic outputs**: Quantile regression (q10/q50/q90) for all models
- **Proper scoring**: Winkler Score + Pinball Loss, not just point RMSE
- **Foundation model evaluation**: Zero-shot Chronos-T5 inference, no fine-tuning
```

### 5.2 GitHub Badges to Include

```markdown
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)
![Chronos](https://img.shields.io/badge/Chronos--T5-Amazon-yellow)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
```

---

## 6. configs/model_config.yaml

```yaml
xgboost:
  objective: reg:quantileerror
  n_estimators: 2000
  learning_rate: 0.03
  max_depth: 7
  subsample: 0.8
  colsample_bytree: 0.75
  min_child_weight: 5
  gamma: 0.1
  early_stopping_rounds: 50

lstm:
  hidden_dim: 256
  n_layers: 2
  dropout: 0.3
  horizon: 24
  seq_len: 168
  n_features: 9
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001

chronos:
  model_variant: amazon/chronos-t5-small
  num_samples: 200
  torch_dtype: bfloat16

sarimax:
  order: [2, 1, 2]
  seasonal_order: [1, 1, 1, 7]
  resample_freq: D

evaluation:
  quantiles: [0.1, 0.5, 0.9]
  ci_alpha: 0.2

backtest:
  n_folds: 5
  horizons:
    primary: 24
    extended: 168
  min_train_years: 2
  gap_hours: 0
```

---

## 7. Portfolio Positioning Notes

| What you're demonstrating | Where it appears |
|---|---|
| Temporal leakage awareness | Backtesting framework, lag shift code, Page 4 Gantt chart |
| Probabilistic ML literacy | Winkler Score, MC Dropout, Chronos num_samples, Page 2 uncertainty bands |
| Feature engineering depth | 35-feature XGBoost matrix, SHAP analysis, Page 5 cyclical encoding explainer |
| Modern ML landscape awareness | Gen 4 Chronos, zero-shot framing, arXiv citation in Methodology page |
| Software engineering discipline | Modular `src/` layout, typed dataclasses, `@st.cache_data`, `configs/*.yaml` |
| Empirical rigor | 5-fold expanding window, multiple proper scoring rules, no cherry-picked results |
