# ⚡ Notebooks & Colab Execution Guide

This folder contains both modular research notebooks and the unified master training pipeline for Google Colab.

## 📁 Notebook Directory

| Notebook | Purpose | Runtime |
|---|---|---|
| [`01_eda_and_cleaning.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/01_eda_and_cleaning.ipynb) | Exploratory Data Analysis, missing step interpolation, ADF stationarity test | CPU |
| [`02_feature_engineering.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/02_feature_engineering.ipynb) | Cyclical Fourier transforms, lag shifts ($\ge 24$h), rolling & EWM features | CPU |
| [`03_backtesting_framework.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/03_backtesting_framework.ipynb) | 5-Fold expanding window generator (zero lookahead leakage) | CPU |
| [`04_gen1_sarimax.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/04_gen1_sarimax.ipynb) | Classical baseline: SARIMAX $(2,1,2)(1,1,1)_7$ | CPU |
| [`05_gen2_xgboost.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/05_gen2_xgboost.ipynb) | Quantile Gradient Boosting ($q_{10}, q_{50}, q_{90}$) + SHAP TreeExplainer | GPU/CPU |
| [`06_gen3_lstm.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/06_gen3_lstm.ipynb) | Multivariate LSTM with Monte Carlo Dropout uncertainty sampling | GPU |
| [`07_gen4_chronos.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/07_gen4_chronos.ipynb) | Zero-shot foundation model inference (`amazon/chronos-t5-small`) | GPU (T4/A100) |
| [`08_results_export.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/08_results_export.ipynb) | Bridge file assembly and schema validation | CPU |
| **[`master_training_pipeline.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/master_training_pipeline.ipynb)** | **⚡ All-in-one Colab master runner (recommended)** | **GPU (T4)** |

---

## 🚀 How to Run on Google Colab (One-Click)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and choose [`master_training_pipeline.ipynb`](file:///d:/LOCK%20IN/project4/notebooks/master_training_pipeline.ipynb).
3. Select **Runtime** > **Change runtime type** > Select **T4 GPU** (free tier).
4. Click **Runtime** > **Run all** (`Ctrl + F9`).
5. What happens automatically:
   - Ingests `PJME_hourly.csv` (no Kaggle API key required).
   - Generates all cyclical and lag features.
   - Fits all 4 generations across 5 expanding-window folds.
   - Computes Winkler scores, Pinball losses, coverage, and SHAP values.
   - Packages `bridge_output.zip` and triggers an automatic browser download.
6. Unzip the 4 files (`forecast_results.csv`, `metrics_summary.json`, `feature_importance.json`, `fold_metadata.json`) into `data/bridge/`.

> **Note**: A complete, high-fidelity validated set of bridge files is already pre-generated in `data/bridge/` so the dashboard can run immediately offline without waiting for Colab.
