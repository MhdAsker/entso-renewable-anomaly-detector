"""Run Phase 2 robustness, negative-control, and decision studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from backtest import expanding_window_folds
from conformal import conformal_p_values, conformal_predictions
from evaluate import calibrate_threshold, compute_anomaly_metrics, predictions_from_scores
from features import feature_columns_for
from labels import load_event_labels
from models.iforest_model import IsolationForestDetector
from models.robust_model import RobustMADDetector
from models.seasonal_quantile_model import SeasonalQuantileDetector
from models.zscore_model import ZScoreDetector
from outcomes import build_operational_stress_label
from robustness import (
    circular_shift_negative_control,
    corrupt_features,
    holm_adjust,
    paired_block_bootstrap_difference,
    scenario_utility,
)
from study_data import DatasetSpec, load_dataset_catalog, validate_dataset_period


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "config" / "study_datasets.csv"
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        help="Dataset identifier to run; repeat the flag. Default: all active datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase2")
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--negative-control-shifts", type=int, default=500)
    parser.add_argument("--stress-repeats", type=int, default=20)
    parser.add_argument("--block-hours", type=int, default=168)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _split_positions(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = int(n_rows * 0.70)
    validation_size = int(n_rows * 0.10)
    fit_end = split - validation_size
    return (
        np.arange(0, fit_end),
        np.arange(fit_end, split),
        np.arange(split, n_rows),
    )


def _load_labels(
    frame: pd.DataFrame, specification: DatasetSpec, fit_end: int
) -> tuple[pd.Series, dict]:
    if specification.outcome_path is None:
        if "is_anomaly" not in frame:
            raise ValueError("Proxy dataset is missing is_anomaly")
        return frame["is_anomaly"].astype("Int64"), {
            "label_type": specification.label_type,
            "evidence_level": specification.evidence_level,
            "independent_outcome": False,
        }
    if not specification.outcome_path.exists():
        raise FileNotFoundError(specification.outcome_path)
    if specification.label_type == "adjudicated_event":
        aligned = load_event_labels(specification.outcome_path).reindex(frame.index)
        metadata = {
            "definition": "timestamped binary events from the registered outcome file",
            "observed_hours": int(aligned.notna().sum()),
            "positive_hours": int(aligned.fillna(0).sum()),
        }
    elif specification.label_type == "operational_stress_proxy":
        outcomes = pd.read_parquet(specification.outcome_path)
        if outcomes.index.tz is not None:
            outcomes.index = outcomes.index.tz_convert("UTC").tz_localize(None)
        labels, metadata = build_operational_stress_label(
            outcomes.sort_index(), frame.index[fit_end - 1]
        )
        aligned = labels.reindex(frame.index)
    else:
        raise ValueError(f"Unsupported external label type: {specification.label_type}")
    metadata.update(
        {
            "label_type": specification.label_type,
            "evidence_level": specification.evidence_level,
            "independent_outcome": True,
            "aligned_coverage_fraction": float(aligned.notna().mean()),
        }
    )
    return aligned, metadata


def _fit_primary_models(
    frame: pd.DataFrame,
    labels: pd.Series,
    fit: np.ndarray,
    validation: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], IsolationForestDetector]:
    feature_columns = feature_columns_for(frame)
    X = frame[feature_columns].to_numpy(dtype=float)
    score_map = {
        "Rolling z-score": ZScoreDetector().get_scores(frame),
        "Robust MAD": RobustMADDetector().get_scores(frame),
    }
    seasonal = SeasonalQuantileDetector().fit(frame.iloc[fit])
    score_map["Seasonal quantile"] = seasonal.get_scores(frame)
    iforest = IsolationForestDetector(random_state=seed).fit(X[fit], save_artifact=False)
    score_map["Isolation Forest"] = iforest.get_scores(X)

    observed_validation = labels.iloc[validation].notna().to_numpy()
    y_validation = labels.iloc[validation].dropna().astype(int).to_numpy()
    predictions = {}
    thresholds = {}
    for name, scores in score_map.items():
        threshold = calibrate_threshold(
            y_validation, scores[validation][observed_validation]
        )
        thresholds[name] = threshold
        predictions[name] = predictions_from_scores(scores, threshold)

    robust_scores = score_map["Robust MAD"]
    conformal_name = "Conformal MAD (alpha=0.05)"
    score_map[conformal_name] = 1 - conformal_p_values(
        robust_scores[validation], robust_scores
    )
    predictions[conformal_name] = conformal_predictions(
        robust_scores[validation], robust_scores, alpha=0.05
    )
    thresholds[conformal_name] = 0.95
    return score_map, predictions, thresholds, iforest


def _holdout_arrays(
    labels: pd.Series,
    holdout: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = labels.iloc[holdout].notna().to_numpy()
    return (
        labels.iloc[holdout].dropna().astype(int).to_numpy(),
        np.asarray(predictions)[holdout][observed],
        np.asarray(scores)[holdout][observed],
    )


def _benchmark_rows(
    dataset_id: str,
    labels: pd.Series,
    holdout: np.ndarray,
    score_map: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for name, scores in score_map.items():
        y, pred, test_scores = _holdout_arrays(
            labels, holdout, scores, predictions[name]
        )
        rows.append(
            {"dataset_id": dataset_id, **compute_anomaly_metrics(y, pred, test_scores, name)}
        )
    return rows


def _paired_comparisons(
    labels: pd.Series,
    holdout: np.ndarray,
    score_map: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> pd.DataFrame:
    reference = "Seasonal quantile"
    y, reference_pred, reference_scores = _holdout_arrays(
        labels, holdout, score_map[reference], predictions[reference]
    )
    rows = []
    for candidate in score_map:
        if candidate == reference:
            continue
        _, candidate_pred, candidate_scores = _holdout_arrays(
            labels, holdout, score_map[candidate], predictions[candidate]
        )
        for metric in ("f1", "pr_auc"):
            comparison = paired_block_bootstrap_difference(
                y,
                candidate_pred,
                candidate_scores,
                reference_pred,
                reference_scores,
                metric=metric,
                block_size=args.block_hours,
                n_bootstrap=args.bootstrap_resamples,
                random_state=args.seed,
            )
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    **comparison,
                }
            )
    return pd.DataFrame(rows)


def _threshold_sensitivity(
    labels: pd.Series,
    validation: np.ndarray,
    holdout: np.ndarray,
    score_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    quantiles = (0.90, 0.95, 0.975, 0.99, 0.995, 0.999)
    for name, scores in score_map.items():
        if name.startswith("Conformal"):
            continue
        calibration_scores = scores[validation]
        calibration_scores = calibration_scores[np.isfinite(calibration_scores)]
        for quantile in quantiles:
            threshold = float(np.quantile(calibration_scores, quantile))
            all_predictions = predictions_from_scores(scores, threshold)
            y, pred, test_scores = _holdout_arrays(
                labels, holdout, scores, all_predictions
            )
            rows.append(
                {
                    "model_name": name,
                    "validation_score_quantile": quantile,
                    "threshold": threshold,
                    **compute_anomaly_metrics(y, pred, test_scores, name),
                }
            )
    return pd.DataFrame(rows)


def _negative_controls(
    labels: pd.Series,
    holdout: np.ndarray,
    score_map: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for name, scores in score_map.items():
        y, pred, test_scores = _holdout_arrays(labels, holdout, scores, predictions[name])
        for metric in ("f1", "pr_auc"):
            result = circular_shift_negative_control(
                y,
                pred,
                test_scores,
                metric=metric,
                min_shift=min(args.block_hours, (len(y) - 1) // 3),
                n_shifts=args.negative_control_shifts,
                random_state=args.seed,
            )
            rows.append({"model_name": name, "metric": metric, **result})
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(
        result["empirical_p_value"].to_numpy()
    )
    return result


def _utility_scenarios(
    labels: pd.Series,
    holdout: np.ndarray,
    score_map: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for false_positive_cost in (0.01, 0.1, 0.5, 1.0):
        for false_negative_cost in (1.0, 5.0, 10.0):
            for name, scores in score_map.items():
                y, pred, _ = _holdout_arrays(
                    labels, holdout, scores, predictions[name]
                )
                rows.append(
                    {
                        "model_name": name,
                        "true_positive_benefit": 1.0,
                        "false_positive_cost": false_positive_cost,
                        "false_negative_cost": false_negative_cost,
                        **scenario_utility(
                            y,
                            pred,
                            false_positive_cost=false_positive_cost,
                            false_negative_cost=false_negative_cost,
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    result["best_in_scenario"] = result.groupby(
        ["false_positive_cost", "false_negative_cost"]
    )["utility_per_30d"].transform("max").eq(result["utility_per_30d"])
    return result


def _stress_tests(
    frame: pd.DataFrame,
    labels: pd.Series,
    fit: np.ndarray,
    holdout: np.ndarray,
    detector: IsolationForestDetector,
    threshold: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = feature_columns_for(frame)
    X = frame[columns].to_numpy(dtype=float)
    fit_median = np.median(X[fit], axis=0)
    fit_std = np.std(X[fit], axis=0)
    fit_std = np.where(fit_std > 0, fit_std, 1.0)
    observed = labels.iloc[holdout].notna().to_numpy()
    y = labels.iloc[holdout].dropna().astype(int).to_numpy()
    rows = []
    for corruption_index, corruption in enumerate(("missingness", "gaussian_noise")):
        for level_index, level in enumerate((0.0, 0.01, 0.05, 0.10, 0.20)):
            for repeat in range(args.stress_repeats):
                corrupted = corrupt_features(
                    X[holdout],
                    fit_median,
                    fit_std,
                    corruption=corruption,
                    level=level,
                    random_state=(
                        args.seed + corruption_index * 100_000 + level_index * 1_000 + repeat
                    ),
                )
                scores = detector.get_scores(corrupted)[observed]
                pred = predictions_from_scores(scores, threshold)
                rows.append(
                    {
                        "corruption": corruption,
                        "level": level,
                        "repeat": repeat,
                        **compute_anomaly_metrics(y, pred, scores, "Isolation Forest"),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["corruption", "level"], as_index=False).agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        f1_q025=("f1", lambda x: x.quantile(0.025)),
        f1_q975=("f1", lambda x: x.quantile(0.975)),
        pr_auc_mean=("pr_auc", "mean"),
        mcc_mean=("mcc", "mean"),
        false_alerts_per_30d_mean=("false_alerts_per_30d", "mean"),
    )
    return raw, summary


def _rolling_transfer(
    frame: pd.DataFrame, labels: pd.Series, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = feature_columns_for(frame)
    X = frame[feature_columns].to_numpy(dtype=float)
    rows = []
    for fold in expanding_window_folds(len(frame)):
        score_map = {
            "Rolling z-score": ZScoreDetector().get_scores(frame),
            "Robust MAD": RobustMADDetector().get_scores(frame),
        }
        seasonal = SeasonalQuantileDetector().fit(frame.iloc[fold.fit])
        score_map["Seasonal quantile"] = seasonal.get_scores(frame)
        iforest = IsolationForestDetector(random_state=seed).fit(
            X[fold.fit], save_artifact=False
        )
        score_map["Isolation Forest"] = iforest.get_scores(X)
        validation = np.arange(fold.validation.start, fold.validation.stop)
        test = np.arange(fold.test.start, fold.test.stop)
        observed_validation = labels.iloc[validation].notna().to_numpy()
        y_validation = labels.iloc[validation].dropna().astype(int).to_numpy()
        if len(np.unique(y_validation)) < 2:
            continue
        for name, scores in score_map.items():
            threshold = calibrate_threshold(
                y_validation, scores[validation][observed_validation]
            )
            all_predictions = predictions_from_scores(scores, threshold)
            y, pred, test_scores = _holdout_arrays(
                labels, test, scores, all_predictions
            )
            if len(np.unique(y)) < 2:
                continue
            rows.append(
                {
                    "fold": fold.fold,
                    "model_name": name,
                    "threshold": threshold,
                    "test_start": frame.index[test[0]],
                    "test_end": frame.index[test[-1]],
                    **compute_anomaly_metrics(y, pred, test_scores, name),
                }
            )
    folds = pd.DataFrame(rows)
    stability = folds.groupby("model_name", as_index=False).agg(
        threshold_mean=("threshold", "mean"),
        threshold_std=("threshold", "std"),
        threshold_min=("threshold", "min"),
        threshold_max=("threshold", "max"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        f1_min=("f1", "min"),
        pr_auc_mean=("pr_auc", "mean"),
        false_alerts_per_30d_mean=("false_alerts_per_30d", "mean"),
    )
    stability["threshold_cv_abs"] = (
        stability["threshold_std"] / stability["threshold_mean"].abs().replace(0, np.nan)
    )
    return folds, stability


def _make_plots(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    paired = results["paired_comparisons"]
    f1 = paired[paired["metric"] == "f1"].sort_values("observed_difference")
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    xerr = np.vstack(
        [f1["observed_difference"] - f1["lower"], f1["upper"] - f1["observed_difference"]]
    )
    axis.errorbar(
        f1["observed_difference"], f1["candidate"], xerr=xerr, fmt="o", capsize=4
    )
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.set(
        xlabel="Paired holdout F1 difference vs seasonal quantile (95% block CI)",
        title="Paired model comparison",
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_f1_differences.png", dpi=180)
    plt.close(fig)

    sensitivity = results["threshold_sensitivity"]
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for name, group in sensitivity.groupby("model_name"):
        axis.plot(
            group["false_alerts_per_30d"], group["recall"], marker="o", label=name
        )
    axis.set(
        xlabel="False alerts per 30 days",
        ylabel="Recall",
        title="Validation-quantile threshold trade-off",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_tradeoff.png", dpi=180)
    plt.close(fig)

    stress = results["stress_summary"]
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for corruption, group in stress.groupby("corruption"):
        axis.plot(group["level"], group["f1_mean"], marker="o", label=corruption)
        axis.fill_between(
            group["level"], group["f1_q025"], group["f1_q975"], alpha=0.15
        )
    axis.set(
        xlabel="Corruption level",
        ylabel="Isolation Forest holdout F1",
        title="Holdout-only feature-corruption stress test",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stress_robustness.png", dpi=180)
    plt.close(fig)

    utility = results["utility_scenarios"]
    utility = utility[utility["false_negative_cost"] == 5.0]
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for name, group in utility.groupby("model_name"):
        axis.plot(
            group["false_positive_cost"], group["utility_per_30d"], marker="o", label=name
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        xlabel="Assumed false-positive cost",
        ylabel="Scenario utility per 30 days",
        title="Utility sensitivity (false-negative cost = 5)",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "utility_sensitivity.png", dpi=180)
    plt.close(fig)


def run_dataset(
    specification: DatasetSpec, args: argparse.Namespace
) -> tuple[dict[str, pd.DataFrame], dict]:
    frame = pd.read_parquet(specification.feature_path)
    validate_dataset_period(frame, specification)
    fit, validation, holdout = _split_positions(len(frame))
    labels, label_metadata = _load_labels(frame, specification, fit[-1] + 1)
    if labels.iloc[validation].dropna().nunique() < 2:
        raise ValueError(f"{specification.dataset_id}: validation labels need both classes")
    if labels.iloc[holdout].dropna().nunique() < 2:
        raise ValueError(f"{specification.dataset_id}: holdout labels need both classes")
    score_map, predictions, thresholds, iforest = _fit_primary_models(
        frame, labels, fit, validation, args.seed
    )
    benchmark = pd.DataFrame(
        _benchmark_rows(
            specification.dataset_id, labels, holdout, score_map, predictions
        )
    )
    stress_raw, stress_summary = _stress_tests(
        frame,
        labels,
        fit,
        holdout,
        iforest,
        thresholds["Isolation Forest"],
        args,
    )
    rolling_folds, threshold_stability = _rolling_transfer(frame, labels, args.seed)
    results = {
        "benchmark": benchmark,
        "paired_comparisons": _paired_comparisons(
            labels, holdout, score_map, predictions, args
        ),
        "threshold_sensitivity": _threshold_sensitivity(
            labels, validation, holdout, score_map
        ),
        "negative_controls": _negative_controls(
            labels, holdout, score_map, predictions, args
        ),
        "utility_scenarios": _utility_scenarios(
            labels, holdout, score_map, predictions
        ),
        "stress_raw": stress_raw,
        "stress_summary": stress_summary,
        "rolling_folds": rolling_folds,
        "threshold_stability": threshold_stability,
    }
    output_dir = args.output_dir / specification.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        result.to_csv(output_dir / f"{name}.csv", index=False)
    _make_plots(results, output_dir)
    metadata = {
        "dataset_id": specification.dataset_id,
        "feature_path": str(specification.feature_path.resolve()),
        "feature_sha256": _sha256(specification.feature_path),
        "zone": specification.zone,
        "declared_period": [
            str(specification.period_start.date()),
            str(specification.period_end.date()),
        ],
        "observed_period": [str(frame.index.min()), str(frame.index.max())],
        "rows": len(frame),
        "label": label_metadata,
        "split_rows": {
            "fit": len(fit),
            "validation": len(validation),
            "holdout": len(holdout),
        },
        "thresholds": thresholds,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return results, metadata


def run(args: argparse.Namespace) -> dict[str, dict[str, pd.DataFrame]]:
    if min(
        args.bootstrap_resamples,
        args.negative_control_shifts,
        args.stress_repeats,
        args.block_hours,
    ) < 1:
        raise ValueError("All repetition and block settings must be positive")
    specifications = load_dataset_catalog(args.catalog, ROOT)
    if args.dataset_id:
        requested = set(args.dataset_id)
        specifications = [s for s in specifications if s.dataset_id in requested]
        missing = requested - {s.dataset_id for s in specifications}
        if missing:
            raise ValueError(f"Unknown or inactive dataset identifiers: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    metadata = []
    cross_dataset = []
    for specification in specifications:
        print(f"Running Phase 2 studies for {specification.dataset_id} ...")
        results, dataset_metadata = run_dataset(specification, args)
        all_results[specification.dataset_id] = results
        metadata.append(dataset_metadata)
        cross_dataset.append(results["benchmark"])
    pd.concat(cross_dataset, ignore_index=True).to_csv(
        args.output_dir / "cross_dataset_benchmark.csv", index=False
    )
    manifest = {
        "protocol": "docs/PHASE2_PROTOCOL.md",
        "git": _git_state(),
        "random_seed": args.seed,
        "paired_bootstrap": {
            "resamples": args.bootstrap_resamples,
            "block_hours": args.block_hours,
        },
        "negative_control_shifts": args.negative_control_shifts,
        "stress_repeats": args.stress_repeats,
        "datasets": metadata,
        "external_validity_status": {
            "active_dataset_count": len(metadata),
            "has_independent_outcome": any(
                item["label"]["evidence_level"] != "internal_proxy" for item in metadata
            ),
            "criterion_met": (
                len(metadata) >= 2
                and any(
                    item["label"]["evidence_level"] != "internal_proxy"
                    for item in metadata
                )
            ),
        },
        "claim_boundary": (
            "Cross-dataset external validity requires at least two independently "
            "sourced active datasets; current results remain internal when the catalog has one."
        ),
    }
    (args.output_dir / "phase2_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Phase 2 artifacts written to {args.output_dir}")
    return all_results


if __name__ == "__main__":
    run(parse_args())
