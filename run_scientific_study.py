"""Run the pre-specified scientific benchmark and write auditable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from conformal import conformal_p_values, conformal_predictions
from evaluate import (
    block_bootstrap_confidence_intervals,
    calibrate_threshold,
    compute_anomaly_metrics,
    predictions_from_scores,
)
from event_evaluate import compute_event_metrics
from features import FEATURE_COLS, feature_columns_for
from models.iforest_model import IsolationForestDetector
from models.robust_model import RobustMADDetector
from models.seasonal_quantile_model import SeasonalQuantileDetector
from models.zscore_model import ZScoreDetector
from outcomes import build_operational_stress_label


ABLATION_GROUPS = {
    "errors": FEATURE_COLS[:6],
    "errors+ramps": FEATURE_COLS[:9],
    "errors+ramps+rolling": FEATURE_COLS[:13],
    "errors+ramps+rolling+calendar": FEATURE_COLS[:18],
    "all_core_features": FEATURE_COLS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "features.parquet",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        help="Optional ENTSO-E imbalance outcome parquet; otherwise use is_anomaly proxy.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "scientific")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--bootstrap-block-hours", type=int, default=168)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conformal-alpha", type=float, default=0.05)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "matplotlib", "pyarrow"]
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _validate(frame: pd.DataFrame) -> None:
    required = set(FEATURE_COLS) | {"is_anomaly", "total_abs_error"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Features must use a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Feature timestamps must be sorted and unique")


def _labels(
    frame: pd.DataFrame, outcomes_path: Path | None, fit_end: int
) -> tuple[pd.Series, dict]:
    if outcomes_path is None:
        return frame["is_anomaly"].astype("Int64"), {
            "source": "rolling_forecast_error_proxy",
            "independent_outcome": False,
            "warning": "Construct-coupled proxy; not external ground truth.",
        }
    outcomes = pd.read_parquet(outcomes_path)
    if not isinstance(outcomes.index, pd.DatetimeIndex):
        raise TypeError("Operational outcomes must use a DatetimeIndex")
    if outcomes.index.tz is not None:
        outcomes.index = outcomes.index.tz_convert("UTC").tz_localize(None)
    label, metadata = build_operational_stress_label(
        outcomes.sort_index(), frame.index[fit_end - 1]
    )
    aligned = label.reindex(frame.index)
    metadata.update(
        {
            "source": str(outcomes_path),
            "independent_outcome": True,
            "aligned_coverage_fraction": float(aligned.notna().mean()),
        }
    )
    return aligned, metadata


def _evaluate_output(
    name: str,
    labels: pd.Series,
    scores: np.ndarray,
    predictions: np.ndarray,
    positions: np.ndarray,
    index: pd.DatetimeIndex,
) -> tuple[dict, dict]:
    observed = labels.iloc[positions].notna().to_numpy()
    y_true = labels.iloc[positions].dropna().astype(int).to_numpy()
    eval_scores = np.asarray(scores)[positions][observed]
    eval_predictions = np.asarray(predictions)[positions][observed]
    eval_index = index[positions][observed]
    point = compute_anomaly_metrics(y_true, eval_predictions, eval_scores, name)
    event = {"model_name": name, **compute_event_metrics(y_true, eval_predictions, eval_index)}
    return point, event


def _make_plots(
    benchmark: pd.DataFrame,
    ablation: pd.DataFrame,
    seasonal: pd.DataFrame,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    for name, (recall, precision) in curves.items():
        axis.plot(recall, precision, label=name)
    axis.set(xlabel="Recall", ylabel="Precision", title="Holdout precision-recall curves")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall.png", dpi=180)
    plt.close(fig)

    plot_ablation = ablation.sort_values("f1")
    labels = plot_ablation["feature_set"].str.replace("+", " + ", regex=False)
    labels = labels.str.replace("_", " ", regex=False)
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    bars = axis.barh(labels, plot_ablation["f1"], color="#2878B5")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set(xlabel="Holdout F1", title="Isolation Forest cumulative feature ablation")
    axis.set_xlim(0, max(0.7, float(plot_ablation["f1"].max()) + 0.08))
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_f1.png", dpi=180)
    plt.close(fig)

    if not seasonal.empty:
        pivot = seasonal.pivot(index="model_name", columns="season", values="f1")
        pivot = pivot.rename(columns={0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"})
        pivot.index = pivot.index.str.replace("alpha=", "α=", regex=False)
        fig, axis = plt.subplots(figsize=(8, 4.5))
        pivot.plot(kind="bar", ax=axis)
        axis.set(ylabel="F1", title="Holdout performance by season")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=18)
        axis.legend(title="Season")
        fig.tight_layout()
        fig.savefig(output_dir / "seasonal_f1.png", dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    if not 0.5 <= args.train_fraction < 0.95:
        raise ValueError("--train-fraction must be in [0.50, 0.95)")
    if not 0.05 <= args.validation_fraction < args.train_fraction:
        raise ValueError("Invalid validation fraction")
    if args.bootstrap_resamples < 1 or args.bootstrap_block_hours < 1:
        raise ValueError("Bootstrap settings must be positive")

    np.random.seed(args.seed)
    frame = pd.read_parquet(args.features)
    _validate(frame)
    split = int(len(frame) * args.train_fraction)
    validation_size = int(len(frame) * args.validation_fraction)
    fit_end = split - validation_size
    fit = np.arange(0, fit_end)
    validation = np.arange(fit_end, split)
    holdout = np.arange(split, len(frame))
    labels, label_metadata = _labels(frame, args.outcomes, fit_end)
    if labels.iloc[validation].dropna().nunique() < 2:
        raise ValueError("Observed validation labels must contain both classes")
    if labels.iloc[holdout].dropna().nunique() < 2:
        raise ValueError("Observed holdout labels must contain both classes")

    feature_columns = feature_columns_for(frame)
    X = frame[feature_columns].to_numpy(dtype=float)
    score_map: dict[str, np.ndarray] = {
        "Rolling z-score": ZScoreDetector().get_scores(frame),
        "Robust MAD": RobustMADDetector().get_scores(frame),
    }
    seasonal_model = SeasonalQuantileDetector().fit(frame.iloc[fit])
    score_map["Seasonal quantile"] = seasonal_model.get_scores(frame)
    iforest = IsolationForestDetector(random_state=args.seed).fit(X[fit], save_artifact=False)
    score_map["Isolation Forest"] = iforest.get_scores(X)

    observed_validation = labels.iloc[validation].notna().to_numpy()
    y_validation = labels.iloc[validation].dropna().astype(int).to_numpy()
    prediction_map: dict[str, np.ndarray] = {}
    thresholds = {}
    for name, scores in score_map.items():
        threshold = calibrate_threshold(
            y_validation, scores[validation][observed_validation]
        )
        thresholds[name] = threshold
        prediction_map[name] = predictions_from_scores(scores, threshold)

    robust_scores = score_map["Robust MAD"]
    finite_calibration = robust_scores[validation]
    conformal_name = f"Conformal MAD (alpha={args.conformal_alpha:g})"
    conformal_scores = 1 - conformal_p_values(finite_calibration, robust_scores)
    conformal_predictions_all = conformal_predictions(
        finite_calibration, robust_scores, args.conformal_alpha
    )
    score_map[conformal_name] = conformal_scores
    prediction_map[conformal_name] = conformal_predictions_all
    thresholds[conformal_name] = 1 - args.conformal_alpha

    benchmark_rows = []
    event_rows = []
    interval_rows = []
    seasonal_rows = []
    curves = {}
    for name, scores in score_map.items():
        predictions = prediction_map[name]
        point, event = _evaluate_output(
            name, labels, scores, predictions, holdout, frame.index
        )
        point.update(
            {
                "threshold": thresholds[name],
                "evaluation_rows": int(labels.iloc[holdout].notna().sum()),
                "positive_hours": int(labels.iloc[holdout].fillna(0).sum()),
            }
        )
        benchmark_rows.append(point)
        event_rows.append(event)

        observed = labels.iloc[holdout].notna().to_numpy()
        y_test = labels.iloc[holdout].dropna().astype(int).to_numpy()
        test_scores = scores[holdout][observed]
        test_predictions = predictions[holdout][observed]
        intervals = block_bootstrap_confidence_intervals(
            y_test,
            test_predictions,
            test_scores,
            block_size=args.bootstrap_block_hours,
            n_bootstrap=args.bootstrap_resamples,
            random_state=args.seed,
        )
        for metric, (lower, upper) in intervals.items():
            interval_rows.append(
                {"model_name": name, "metric": metric, "lower": lower, "upper": upper}
            )
        precision, recall, _ = precision_recall_curve(y_test, test_scores)
        curves[name] = (recall, precision)
        test_frame = frame.iloc[holdout].loc[labels.iloc[holdout].notna()]
        for season, group in test_frame.groupby("season"):
            positions = frame.index.get_indexer(group.index)
            seasonal_point, _ = _evaluate_output(
                name, labels, scores, predictions, positions, frame.index
            )
            seasonal_rows.append({**seasonal_point, "season": int(season)})

    ablation_rows = []
    for feature_set, columns in ABLATION_GROUPS.items():
        detector = IsolationForestDetector(random_state=args.seed).fit(
            frame.iloc[fit][columns].to_numpy(dtype=float), save_artifact=False
        )
        scores = detector.get_scores(frame[columns].to_numpy(dtype=float))
        threshold = calibrate_threshold(
            y_validation, scores[validation][observed_validation]
        )
        predictions = predictions_from_scores(scores, threshold)
        point, _ = _evaluate_output(
            f"Isolation Forest: {feature_set}",
            labels,
            scores,
            predictions,
            holdout,
            frame.index,
        )
        ablation_rows.append(
            {**point, "feature_set": feature_set, "n_features": len(columns), "threshold": threshold}
        )

    results = {
        "benchmark": pd.DataFrame(benchmark_rows),
        "confidence_intervals": pd.DataFrame(interval_rows),
        "event_metrics": pd.DataFrame(event_rows),
        "ablation": pd.DataFrame(ablation_rows),
        "seasonal_metrics": pd.DataFrame(seasonal_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        result.to_csv(args.output_dir / f"{name}.csv", index=False)
    _make_plots(
        results["benchmark"], results["ablation"], results["seasonal_metrics"], curves, args.output_dir
    )

    manifest = {
        "protocol": "docs/SCIENTIFIC_PROTOCOL.md",
        "feature_path": str(args.features.resolve()),
        "feature_sha256": _sha256(args.features),
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "random_seed": args.seed,
        "split": {
            "fit": [str(frame.index[fit[0]]), str(frame.index[fit[-1]])],
            "validation": [str(frame.index[validation[0]]), str(frame.index[validation[-1]])],
            "holdout": [str(frame.index[holdout[0]]), str(frame.index[holdout[-1]])],
            "rows": {"fit": len(fit), "validation": len(validation), "holdout": len(holdout)},
        },
        "label": label_metadata,
        "threshold_policy": "F1-optimal on validation labels; conformal MAD is label-free",
        "bootstrap": {
            "method": "moving block",
            "block_hours": args.bootstrap_block_hours,
            "resamples": args.bootstrap_resamples,
            "confidence": 0.95,
        },
        "feature_columns": feature_columns,
        "python": platform.python_version(),
        "packages": _package_versions(),
    }
    (args.output_dir / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(results["benchmark"].to_string(index=False))
    print(f"\nScientific artifacts written to {args.output_dir}")
    return results


if __name__ == "__main__":
    run(parse_args())
