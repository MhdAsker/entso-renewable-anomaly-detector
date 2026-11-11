"""Finite-sample split-conformal calibration for anomaly scores."""

from __future__ import annotations

import numpy as np


def conformal_p_values(
    calibration_scores: np.ndarray, test_scores: np.ndarray
) -> np.ndarray:
    """Return upper-tail split-conformal p-values for higher-is-stranger scores."""
    calibration = np.asarray(calibration_scores, dtype=float)
    test = np.asarray(test_scores, dtype=float)
    calibration = calibration[np.isfinite(calibration)]
    if len(calibration) == 0:
        raise ValueError("At least one finite calibration score is required")
    result = np.full(test.shape, np.nan, dtype=float)
    finite = np.isfinite(test)
    result[finite] = (
        1
        + np.asarray(
            [(calibration >= score).sum() for score in test[finite]], dtype=float
        )
    ) / (len(calibration) + 1)
    return result


def conformal_predictions(
    calibration_scores: np.ndarray, test_scores: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Flag test scores with conformal p-value at or below ``alpha``."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    p_values = conformal_p_values(calibration_scores, test_scores)
    return (np.isfinite(p_values) & (p_values <= alpha)).astype(int)
