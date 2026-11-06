"""Run expanding-window seasonal backtests with validation-only calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from backtest import expanding_window_folds
from evaluate import calibrate_threshold, compute_anomaly_metrics, predictions_from_scores
from features import feature_columns_for
from models.iforest_model import IsolationForestDetector
from models.lstm_autoencoder import LSTMAnomalyDetector
from models.robust_model import RobustMADDetector
from models.zscore_model import ZScoreDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=ROOT / "data" / "processed" / "features.parquet"
    )
    parser.add_argument("--include-lstm", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-column", default="is_anomaly")
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(args.features)
    X = df[feature_columns_for(df)].to_numpy()
    if args.label_column not in df:
        raise ValueError(f"Label column not found: {args.label_column}")
    y = df[args.label_column].to_numpy()
    folds = expanding_window_folds(len(df))
    rows: list[dict] = []

    for fold in folds:
        scores = {
            "Z-Score": ZScoreDetector().get_scores(df),
            "Robust MAD": RobustMADDetector().get_scores(df),
        }
        iforest = IsolationForestDetector(random_state=args.seed)
        iforest.fit(X[fold.fit], save_artifact=False)
        scores["Isolation Forest"] = iforest.get_scores(X)

        if args.include_lstm:
            lstm = LSTMAnomalyDetector(random_state=args.seed)
            lstm.fit(X[fold.fit], epochs=args.epochs, save_artifact=False)
            scores["LSTM Autoencoder"] = lstm.get_scores(X)

        for model_name, model_scores in scores.items():
            threshold = calibrate_threshold(y[fold.validation], model_scores[fold.validation])
            test_scores = model_scores[fold.test]
            predictions = predictions_from_scores(test_scores, threshold)
            metrics = compute_anomaly_metrics(y[fold.test], predictions, test_scores, model_name)
            rows.append(
                {
                    **metrics,
                    "fold": fold.fold,
                    "threshold": threshold,
                    "fit_end": df.index[fold.fit.stop - 1],
                    "validation_start": df.index[fold.validation.start],
                    "test_start": df.index[fold.test.start],
                    "test_end": df.index[fold.test.stop - 1],
                }
            )

    fold_results = pd.DataFrame(rows)
    aggregate = fold_results.groupby("model_name", as_index=False).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
    )
    output_dir = ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(output_dir / "backtest_folds.csv", index=False)
    aggregate.to_csv(output_dir / "backtest_summary.csv", index=False)
    print(fold_results.to_string(index=False))
    print("\nAggregate rolling-origin results:\n", aggregate.to_string(index=False))
    return fold_results, aggregate


if __name__ == "__main__":
    run(parse_args())
