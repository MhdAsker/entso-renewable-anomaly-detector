"""Register the current calibrated artifacts without retraining them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from features import feature_columns_for
from models.iforest_model import IsolationForestDetector
from models.lstm_autoencoder import LSTMAnomalyDetector
from models.robust_model import RobustMADDetector
from models.zscore_model import ZScoreDetector
from registry import DEFAULT_REGISTRY, register_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--features", type=Path, default=ROOT / "data" / "processed" / "features.parquet"
    )
    parser.add_argument(
        "--metrics", type=Path, default=ROOT / "results" / "metrics" / "anomaly_results.csv"
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    features = pd.read_parquet(args.features)
    metrics_frame = pd.read_csv(args.metrics)
    metrics = metrics_frame.to_dict(orient="records")
    feature_cols = feature_columns_for(features)
    fit_end_time = pd.Timestamp(metrics_frame.iloc[0]["train_end"])
    validation_start = pd.Timestamp(metrics_frame.iloc[0]["validation_start"])
    test_start = pd.Timestamp(metrics_frame.iloc[0]["test_start"])
    fit_frame = features.loc[:fit_end_time]
    validation = (features.index >= validation_start) & (features.index < test_start)
    X = features[feature_cols].to_numpy()

    iforest = IsolationForestDetector.load(str(ROOT / "models" / "iforest.pkl"))
    lstm = LSTMAnomalyDetector.load(str(ROOT / "models" / "lstm_autoencoder.pt"))
    score_arrays = {
        "zscore": ZScoreDetector().get_scores(features),
        "robust_mad": RobustMADDetector().get_scores(features),
        "isolation_forest": iforest.get_scores(X),
        "lstm_autoencoder": lstm.get_scores(X),
    }
    metric_thresholds = dict(zip(metrics_frame["model_name"], metrics_frame["threshold"]))
    thresholds = {
        "zscore": float(metric_thresholds["Z-Score (baseline)"]),
        "robust_mad": float(metric_thresholds["Robust MAD"]),
        "isolation_forest": float(iforest.threshold),
        "lstm_autoencoder": float(lstm.threshold),
    }
    reference = {
        model: {
            "p95": float(np.nanquantile(scores[validation], 0.95)),
            "p99": float(np.nanquantile(scores[validation], 0.99)),
            "max": float(np.nanmax(scores[validation])),
        }
        for model, scores in score_arrays.items()
    }
    return register_run(
        feature_path=args.features,
        feature_cols=feature_cols,
        fit_frame=fit_frame,
        thresholds=thresholds,
        score_reference=reference,
        metrics=metrics,
        split={
            "fit_end": str(fit_end_time),
            "validation_start": str(validation_start),
            "test_start": str(test_start),
        },
        label_column="is_anomaly",
        run_id=args.run_id,
        registry_dir=args.registry_dir,
    )


if __name__ == "__main__":
    path = run(parse_args())
    print(f"Registered: {path}")
