import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Allow imports from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluate import compute_anomaly_metrics, print_results_table, save_metrics
from features import FEATURE_COLS
from models.iforest_model import IsolationForestDetector
from models.lstm_autoencoder import LSTMAnomalyDetector
from models.zscore_model import ZScoreDetector
from visualize import (
    plot_anomaly_comparison,
    plot_confusion_matrices,
    plot_forecast_error_overview,
    plot_reconstruction_error,
    plot_seasonal_anomaly_pattern,
)

# ── 1. Environment ─────────────────────────────────────────────────────────────
load_dotenv()

# ── 2. Load features ───────────────────────────────────────────────────────────
FEATURES_PATH = os.path.join("data", "processed", "features.parquet")
print(f"Loading {FEATURES_PATH} …")
df = pd.read_parquet(FEATURES_PATH)
print(f"  {len(df):,} rows loaded | columns: {list(df.columns)}\n")

y_true = df["is_anomaly"].values
X = df[FEATURE_COLS].values

# ── 3. Z-Score (baseline) ──────────────────────────────────────────────────────
print("=" * 52)
print("Running Z-Score detector …")
zscore = ZScoreDetector(threshold=2.5)
pred_zscore = zscore.fit_predict(df)
score_zscore = zscore.get_scores(df)

# ── 4. Isolation Forest ────────────────────────────────────────────────────────
print("\nRunning Isolation Forest …")
iforest = IsolationForestDetector(contamination=0.05)
pred_iforest = iforest.fit_predict(X)
score_iforest = iforest.get_scores(X)

# ── 5. LSTM Autoencoder ────────────────────────────────────────────────────────
print("\nRunning LSTM Autoencoder …")
lstm = LSTMAnomalyDetector(sequence_len=24, hidden_dim=32)
pred_lstm = lstm.fit_predict(df, FEATURE_COLS)
score_lstm = lstm.get_scores(df[FEATURE_COLS].values)

# ── 6. Metrics ─────────────────────────────────────────────────────────────────
print("\nComputing metrics …")
metrics_zscore = compute_anomaly_metrics(y_true, pred_zscore, score_zscore, "Z-Score (baseline)")
metrics_iforest = compute_anomaly_metrics(y_true, pred_iforest, score_iforest, "Isolation Forest")
metrics_lstm = compute_anomaly_metrics(y_true, pred_lstm, score_lstm, "LSTM Autoencoder")
metrics_list = [metrics_zscore, metrics_iforest, metrics_lstm]

# ── 7. Print comparison table ──────────────────────────────────────────────────
print()
print_results_table(metrics_list)

# ── 8. Save CSV ────────────────────────────────────────────────────────────────
save_metrics(metrics_list, path=os.path.join("results", "metrics", "anomaly_results.csv"))

# ── 9. Plots ───────────────────────────────────────────────────────────────────
print("\nGenerating plots …")
predictions_dict = {
    "Z-Score": pred_zscore,
    "Isolation Forest": pred_iforest,
    "LSTM Autoencoder": pred_lstm,
}

plot_forecast_error_overview(df)
plot_anomaly_comparison(df, predictions_dict)
plot_confusion_matrices(y_true, predictions_dict)
plot_reconstruction_error(score_lstm, df, threshold=lstm.threshold)
plot_seasonal_anomaly_pattern(df, predictions_dict)

# ── 10. Largest missed anomaly per model ──────────────────────────────────────
print("\nLargest missed anomaly per model:")
missed_summary = {}
for name, preds in predictions_dict.items():
    mask = (y_true == 1) & (np.asarray(preds) == 0)
    missed_df = df[mask].copy()
    if missed_df.empty:
        print(f"  {name}: no missed anomalies")
        continue
    worst = missed_df["total_abs_error"].idxmax()
    mw = missed_df.loc[worst, "total_abs_error"]
    missed_summary[name] = (worst, mw)
    print(f"  {name}: {worst} — error: {mw:.0f} MW")

# Event caught by all 3
all_caught_mask = (
    (y_true == 1)
    & (pred_zscore == 1)
    & (pred_iforest == 1)
    & (pred_lstm == 1)
)
all_caught_df = df[all_caught_mask]
if not all_caught_df.empty:
    best_ts = all_caught_df["total_abs_error"].idxmax()
    best_mw = all_caught_df.loc[best_ts, "total_abs_error"]
else:
    best_ts = "N/A"
    best_mw = float("nan")

# ── 11. Final summary ──────────────────────────────────────────────────────────
n_total = len(df)
n_anom = int(y_true.sum())
pct_anom = 100 * n_anom / n_total

sorted_metrics = sorted(metrics_list, key=lambda d: d["f1"], reverse=True)
best = sorted_metrics[0]
baseline_f1 = metrics_zscore["f1"]
improvement = (
    100 * (best["f1"] - baseline_f1) / baseline_f1 if baseline_f1 > 0 else float("nan")
)

line = "=" * 52
thin = "─" * 52

print(f"""
{line}
ANOMALY DETECTION BENCHMARK — RESULTS
entso-renewable-anomaly-detector
{line}
Dataset     : Germany 2023, hourly wind+solar
Total hours : {n_total:,}  |  Anomaly hours: {n_anom} ({pct_anom:.1f}%)
Features    : {len(FEATURE_COLS)} engineered features

Method              Precision  Recall   F1      ROC-AUC
{thin}
Z-Score (baseline)  {metrics_zscore['precision']:.2f}       {metrics_zscore['recall']:.2f}     {metrics_zscore['f1']:.2f}    {metrics_zscore['roc_auc']:.2f}
Isolation Forest    {metrics_iforest['precision']:.2f}       {metrics_iforest['recall']:.2f}     {metrics_iforest['f1']:.2f}    {metrics_iforest['roc_auc']:.2f}
LSTM Autoencoder    {metrics_lstm['precision']:.2f}       {metrics_lstm['recall']:.2f}     {metrics_lstm['f1']:.2f}    {metrics_lstm['roc_auc']:.2f}

Best F1  : {best['model_name']}
Improvement over baseline: +{improvement:.1f}% F1
Largest event caught by all 3: {best_ts} — {best_mw:.0f} MW error
{line}
Plots   → results/plots/
Metrics → results/metrics/anomaly_results.csv
""")
