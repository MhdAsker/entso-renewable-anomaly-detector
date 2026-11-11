"""Statistical robustness utilities for Phase 2 scientific studies."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def _metric(
    metric: str, y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> float:
    if metric == "f1":
        return float(f1_score(y_true, predictions, zero_division=0))
    if metric == "pr_auc":
        return float(average_precision_score(y_true, scores))
    raise ValueError(f"Unsupported paired metric: {metric}")


def paired_block_bootstrap_difference(
    y_true: np.ndarray,
    candidate_predictions: np.ndarray,
    candidate_scores: np.ndarray,
    reference_predictions: np.ndarray,
    reference_scores: np.ndarray,
    *,
    metric: str = "f1",
    block_size: int = 168,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Paired moving-block interval for candidate minus reference performance."""
    arrays = [
        np.asarray(y_true, dtype=int),
        np.asarray(candidate_predictions, dtype=int),
        np.asarray(candidate_scores, dtype=float),
        np.asarray(reference_predictions, dtype=int),
        np.asarray(reference_scores, dtype=float),
    ]
    if not arrays[0].size or len({len(value) for value in arrays}) != 1:
        raise ValueError("Paired bootstrap inputs must be non-empty and equal length")
    finite = np.isfinite(arrays[2]) & np.isfinite(arrays[4])
    arrays = [value[finite] for value in arrays]
    if not arrays[0].size:
        raise ValueError("Paired bootstrap has no finite paired scores")

    y, candidate_pred, candidate_score, reference_pred, reference_score = arrays
    observed = _metric(metric, y, candidate_pred, candidate_score) - _metric(
        metric, y, reference_pred, reference_score
    )
    block_size = min(block_size, len(y))
    n_blocks = int(np.ceil(len(y) / block_size))
    max_start = len(y) - block_size
    rng = np.random.default_rng(random_state)
    differences = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sampled = np.concatenate(
            [np.arange(start, start + block_size) for start in starts]
        )[: len(y)]
        if metric == "pr_auc" and len(np.unique(y[sampled])) < 2:
            continue
        differences.append(
            _metric(metric, y[sampled], candidate_pred[sampled], candidate_score[sampled])
            - _metric(metric, y[sampled], reference_pred[sampled], reference_score[sampled])
        )
    if not differences:
        raise ValueError("No valid paired bootstrap samples were generated")
    alpha = (1 - confidence) / 2
    values = np.asarray(differences)
    return {
        "observed_difference": float(observed),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "bootstrap_win_fraction": float(np.mean(values > 0)),
        "n_valid_resamples": int(len(values)),
    }


def circular_shift_negative_control(
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    *,
    metric: str,
    min_shift: int = 168,
    n_shifts: int = 500,
    random_state: int = 42,
) -> dict[str, float]:
    """Compare observed performance with autocorrelation-preserving label shifts."""
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not (len(y_true) == len(predictions) == len(scores)) or len(y_true) == 0:
        raise ValueError("Negative-control inputs must be non-empty and equal length")
    if 2 * min_shift >= len(y_true):
        raise ValueError("min_shift leaves no admissible circular shifts")
    observed = _metric(metric, y_true, predictions, scores)
    admissible = np.arange(min_shift, len(y_true) - min_shift + 1)
    rng = np.random.default_rng(random_state)
    offsets = rng.choice(admissible, size=n_shifts, replace=True)
    null_values = np.asarray(
        [
            _metric(metric, np.roll(y_true, int(offset)), predictions, scores)
            for offset in offsets
        ]
    )
    return {
        "observed": float(observed),
        "null_mean": float(null_values.mean()),
        "null_q025": float(np.quantile(null_values, 0.025)),
        "null_q975": float(np.quantile(null_values, 0.975)),
        "empirical_p_value": float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1)),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise-error adjustment that preserves original ordering."""
    values = np.asarray(p_values, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        [(len(values) - rank) * values[index] for rank, index in enumerate(order)]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1.0, adjusted_sorted)
    return adjusted


def scenario_utility(
    y_true: np.ndarray,
    predictions: np.ndarray,
    *,
    true_positive_benefit: float = 1.0,
    false_positive_cost: float = 0.1,
    false_negative_cost: float = 5.0,
) -> dict[str, float]:
    """Return transparent scenario utility; coefficients are assumptions, not costs."""
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if len(y_true) != len(predictions) or len(y_true) == 0:
        raise ValueError("Utility inputs must be non-empty and equal length")
    tp = int(((y_true == 1) & (predictions == 1)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())
    utility = tp * true_positive_benefit - fp * false_positive_cost - fn * false_negative_cost
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "utility": float(utility),
        "utility_per_30d": float(utility * 720 / len(y_true)),
    }


def corrupt_features(
    values: np.ndarray,
    fit_median: np.ndarray,
    fit_std: np.ndarray,
    *,
    corruption: str,
    level: float,
    random_state: int,
) -> np.ndarray:
    """Apply controlled holdout-only missingness or Gaussian measurement noise."""
    if not 0 <= level <= 1:
        raise ValueError("Corruption level must be in [0, 1]")
    result = np.asarray(values, dtype=float).copy()
    median = np.asarray(fit_median, dtype=float)
    scale = np.asarray(fit_std, dtype=float)
    if result.shape[1] != len(median) or len(median) != len(scale):
        raise ValueError("Feature reference vectors do not match input width")
    rng = np.random.default_rng(random_state)
    if corruption == "missingness":
        mask = rng.random(result.shape) < level
        result[mask] = np.broadcast_to(median, result.shape)[mask]
    elif corruption == "gaussian_noise":
        result += rng.normal(0, level, size=result.shape) * scale
    else:
        raise ValueError(f"Unsupported corruption type: {corruption}")
    return result
