import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def calibrate_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Select the F1-optimal threshold using validation data only."""
    valid = np.isfinite(scores)
    y_valid = np.asarray(y_true)[valid]
    score_valid = np.asarray(scores)[valid]
    if len(score_valid) == 0 or len(np.unique(y_valid)) < 2:
        raise ValueError("Threshold calibration requires finite scores and both classes")
    precision, recall, thresholds = precision_recall_curve(y_valid, score_valid)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    return float(thresholds[int(np.nanargmax(f1))])


def predictions_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(scores)
    return (np.isfinite(scores) & (scores >= threshold)).astype(int)


def compute_anomaly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    model_name: str,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not (len(y_true) == len(y_pred) == len(scores)):
        raise ValueError("Metric inputs must have equal length")
    # Mask out NaN scores so ROC-AUC doesn't crash on rolling warm-up NaNs
    valid = np.isfinite(scores)
    valid_classes = np.unique(y_true[valid]) if valid.any() else np.array([])
    roc = (
        roc_auc_score(y_true[valid], scores[valid])
        if len(valid_classes) == 2
        else float("nan")
    )
    pr_auc = (
        average_precision_score(y_true[valid], scores[valid])
        if len(valid_classes) == 2
        else float("nan")
    )
    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    duration_days = max(len(y_true) / 24, 1 / 24)

    return {
        "model_name": model_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) == 2 else float("nan"),
        "false_alerts_per_30d": false_positives * 30 / duration_days,
        "n_anomalies_detected": int(y_pred.sum()),
    }


def block_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    *,
    block_size: int = 168,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Moving-block bootstrap intervals that preserve local serial dependence."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)
    if not (len(y_true) == len(y_pred) == len(scores)) or len(y_true) == 0:
        raise ValueError("Bootstrap inputs must be non-empty and have equal length")
    block_size = min(block_size, len(y_true))
    rng = np.random.default_rng(random_state)
    metrics = {"precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": []}
    max_start = len(y_true) - block_size
    n_blocks = int(np.ceil(len(y_true) / block_size))
    for _ in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sampled = np.concatenate([np.arange(start, start + block_size) for start in starts])
        sampled = sampled[: len(y_true)]
        result = compute_anomaly_metrics(
            y_true[sampled], y_pred[sampled], scores[sampled], "bootstrap"
        )
        for name in metrics:
            if np.isfinite(result[name]):
                metrics[name].append(result[name])
    alpha = (1 - confidence) / 2
    return {
        name: (
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1 - alpha)),
        )
        for name, values in metrics.items()
        if values
    }


def save_metrics(
    metrics_list: list[dict],
    path: str = "results/metrics/anomaly_results.csv",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(metrics_list).sort_values("f1", ascending=False)
    df.to_csv(path, index=False)
    logger.info("Metrics saved to %s", path)


def print_results_table(metrics_list: list[dict]) -> None:
    rows = sorted(metrics_list, key=lambda d: d["f1"], reverse=True)

    # Column widths
    W_METHOD = 22
    W_PREC   =  9
    W_REC    =  6
    W_F1     =  4
    W_AUC    =  8
    W_FOUND  = 16

    def _row(method, prec, rec, f1, auc, found, sep="║"):
        return (
            f"{sep} {method:<{W_METHOD}} {sep}"
            f" {prec:^{W_PREC}} {sep}"
            f" {rec:^{W_REC}} {sep}"
            f" {f1:^{W_F1}} {sep}"
            f" {auc:^{W_AUC}} {sep}"
            f" {found:^{W_FOUND}} {sep}"
        )

    top    = "╔" + "══" + "═"*W_METHOD + "═╦" + "═"*(W_PREC+2)   + "╦" + "═"*(W_REC+2)  + "╦" + "═"*(W_F1+2)  + "╦" + "═"*(W_AUC+2)  + "╦" + "═"*(W_FOUND+2) + "╗"
    mid    = "╠" + "══" + "═"*W_METHOD + "═╬" + "═"*(W_PREC+2)   + "╬" + "═"*(W_REC+2)  + "╬" + "═"*(W_F1+2)  + "╬" + "═"*(W_AUC+2)  + "╬" + "═"*(W_FOUND+2) + "╣"
    bottom = "╚" + "══" + "═"*W_METHOD + "═╩" + "═"*(W_PREC+2)   + "╩" + "═"*(W_REC+2)  + "╩" + "═"*(W_F1+2)  + "╩" + "═"*(W_AUC+2)  + "╩" + "═"*(W_FOUND+2) + "╝"

    header = _row("Method", "Precision", "Recall", "F1", "ROC-AUC", "Anomalies Found")

    print(top)
    print(header)
    print(mid)
    for i, r in enumerate(rows):
        line = _row(
            r["model_name"],
            f"{r['precision']:.2f}",
            f"{r['recall']:.2f}",
            f"{r['f1']:.2f}",
            f"{r['roc_auc']:.2f}",
            str(r["n_anomalies_detected"]),
        )
        print(line)
        if i < len(rows) - 1:
            print(mid)
    print(bottom)

    # Footer: improvement of best F1 over baseline (first alphabetically by name containing "Z" or last in sorted list)
    baseline = next(
        (r for r in metrics_list if "score" in r["model_name"].lower() or "z-score" in r["model_name"].lower()),
        metrics_list[-1],
    )
    best = rows[0]
    if best["model_name"] != baseline["model_name"] and baseline["f1"] > 0:
        improvement = 100 * (best["f1"] - baseline["f1"]) / baseline["f1"]
        print(
            f'\nBest F1: {best["model_name"]} '
            f'(+{improvement:.1f}% over {baseline["model_name"]} baseline)'
        )
    else:
        print(f'\nBest F1: {best["model_name"]} ({best["f1"]:.2f})')
