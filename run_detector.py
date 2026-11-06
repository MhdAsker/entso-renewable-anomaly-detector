"""Train and evaluate the anomaly detectors with a chronological holdout."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import (
    calibrate_threshold,
    compute_anomaly_metrics,
    predictions_from_scores,
    print_results_table,
    save_metrics,
)
from features import FEATURE_COLS, feature_columns_for
from models.iforest_model import IsolationForestDetector
from models.lstm_autoencoder import LSTMAnomalyDetector
from models.robust_model import RobustMADDetector
from models.zscore_model import ZScoreDetector
from registry import DEFAULT_REGISTRY, register_run
from visualize import (
    plot_anomaly_comparison,
    plot_confusion_matrices,
    plot_forecast_error_overview,
    plot_reconstruction_error,
    plot_seasonal_anomaly_pattern,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "features.parquet",
        help="Path to the engineered feature parquet file.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Chronological fraction used for unsupervised training (default: 0.70).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
        help="Fraction reserved before the holdout for threshold calibration.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="LSTM training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--label-column",
        default="is_anomaly",
        help="Evaluation label column; use event_label for independently sourced labels.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--run-id", help="Optional model-registry version identifier.")
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--no-register", action="store_true", help="Do not register artifacts.")
    return parser.parse_args()


def validate_features(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLS) | {"is_anomaly", "total_abs_error"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature file is missing columns: {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Feature file must use a DatetimeIndex")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Feature timestamps must be sorted and unique")
    if not np.isfinite(df[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError("Model features contain NaN or infinite values")


def run(args: argparse.Namespace) -> list[dict]:
    if not 0.5 <= args.train_fraction < 0.95:
        raise ValueError("--train-fraction must be in [0.50, 0.95)")
    if not 0.05 <= args.validation_fraction < args.train_fraction:
        raise ValueError("--validation-fraction must be at least 0.05 and smaller than training")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    print(f"Loading {args.features} ...")
    df = pd.read_parquet(args.features)
    validate_features(df)
    if args.label_column not in df.columns:
        raise ValueError(f"Label column not found: {args.label_column}")

    split_idx = int(len(df) * args.train_fraction)
    validation_size = int(len(df) * args.validation_fraction)
    fit_end = split_idx - validation_size
    if fit_end < 24:
        raise ValueError("Fit window is too short")
    train_df = df.iloc[:fit_end]
    validation_df = df.iloc[fit_end:split_idx]
    test_df = df.iloc[split_idx:]
    feature_cols = feature_columns_for(df)
    X_all = df[feature_cols].to_numpy()
    X_train = train_df[feature_cols].to_numpy()
    y_validation = validation_df[args.label_column].to_numpy()
    y_test = test_df[args.label_column].to_numpy()

    print(
        f"Training: {train_df.index.min()} to {train_df.index.max()} "
        f"({len(train_df):,} rows)"
    )
    print(
        f"Validate: {validation_df.index.min()} to {validation_df.index.max()} "
        f"({len(validation_df):,} rows)"
    )
    print(
        f"Holdout : {test_df.index.min()} to {test_df.index.max()} "
        f"({len(test_df):,} rows, {int(y_test.sum())} proxy anomalies)"
    )

    validation = slice(fit_end, split_idx)
    holdout = slice(split_idx, None)

    zscore = ZScoreDetector()
    score_zscore_all = zscore.get_scores(df)
    zscore.threshold = calibrate_threshold(y_validation, score_zscore_all[validation])
    pred_zscore_all = predictions_from_scores(score_zscore_all, zscore.threshold)

    robust = RobustMADDetector()
    score_robust_all = robust.get_scores(df)
    robust.threshold = calibrate_threshold(y_validation, score_robust_all[validation])
    pred_robust_all = predictions_from_scores(score_robust_all, robust.threshold)

    iforest = IsolationForestDetector(contamination=0.05, random_state=args.seed)
    iforest.fit(X_train)
    score_iforest_all = iforest.get_scores(X_all)
    iforest_threshold = calibrate_threshold(y_validation, score_iforest_all[validation])
    iforest.threshold = iforest_threshold
    iforest.save()
    pred_iforest_all = predictions_from_scores(score_iforest_all, iforest_threshold)

    lstm = LSTMAnomalyDetector(sequence_len=24, hidden_dim=32, random_state=args.seed)
    lstm.fit(X_train, epochs=args.epochs)
    score_lstm_all = lstm.get_scores(X_all)
    lstm.threshold = calibrate_threshold(y_validation, score_lstm_all[validation])
    lstm.save()
    pred_lstm_all = predictions_from_scores(score_lstm_all, lstm.threshold)

    holdout_outputs = {
        "Z-Score (baseline)": (
            pred_zscore_all[holdout], score_zscore_all[holdout], zscore.threshold
        ),
        "Robust MAD": (
            pred_robust_all[holdout], score_robust_all[holdout], robust.threshold
        ),
        "Isolation Forest": (
            pred_iforest_all[holdout], score_iforest_all[holdout], iforest_threshold
        ),
        "LSTM Autoencoder": (
            pred_lstm_all[holdout], score_lstm_all[holdout], lstm.threshold
        ),
    }
    metrics = [
        {
            **compute_anomaly_metrics(y_test, pred, score, name),
            "threshold": threshold,
            "evaluation_rows": len(test_df),
            "train_end": str(train_df.index.max()),
            "validation_start": str(validation_df.index.min()),
            "test_start": str(test_df.index.min()),
        }
        for name, (pred, score, threshold) in holdout_outputs.items()
    ]

    print("\nChronological holdout results (agreement with proxy labels):")
    print_results_table(metrics)
    metrics_path = ROOT / "results" / "metrics" / "anomaly_results.csv"
    save_metrics(metrics, path=str(metrics_path))

    if not args.no_register:
        score_arrays = {
            "zscore": score_zscore_all[validation],
            "robust_mad": score_robust_all[validation],
            "isolation_forest": score_iforest_all[validation],
            "lstm_autoencoder": score_lstm_all[validation],
        }
        score_reference = {
            name: {
                "p95": float(np.nanquantile(values, 0.95)),
                "p99": float(np.nanquantile(values, 0.99)),
                "max": float(np.nanmax(values)),
            }
            for name, values in score_arrays.items()
        }
        run_dir = register_run(
            feature_path=args.features,
            feature_cols=feature_cols,
            fit_frame=train_df,
            thresholds={
                "zscore": zscore.threshold,
                "robust_mad": robust.threshold,
                "isolation_forest": iforest_threshold,
                "lstm_autoencoder": lstm.threshold,
            },
            score_reference=score_reference,
            metrics=metrics,
            split={
                "fit_end": str(train_df.index.max()),
                "validation_start": str(validation_df.index.min()),
                "test_start": str(test_df.index.min()),
            },
            label_column=args.label_column,
            run_id=args.run_id,
            registry_dir=args.registry_dir,
        )
        print(f"Registered model version: {run_dir}")

    if not args.skip_plots:
        predictions = {
            "Z-Score": pred_zscore_all[holdout],
            "Robust MAD": pred_robust_all[holdout],
            "Isolation Forest": pred_iforest_all[holdout],
            "LSTM Autoencoder": pred_lstm_all[holdout],
        }
        plot_forecast_error_overview(test_df)
        plot_anomaly_comparison(test_df, predictions)
        plot_confusion_matrices(y_test, predictions)
        plot_reconstruction_error(score_lstm_all[holdout], test_df, threshold=lstm.threshold)
        plot_seasonal_anomaly_pattern(test_df, predictions)

    print(f"\nMetrics saved to {metrics_path}")
    return metrics


if __name__ == "__main__":
    run(parse_args())
