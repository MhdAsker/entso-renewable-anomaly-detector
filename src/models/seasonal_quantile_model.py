"""Interpretable hour-by-month conditional quantile baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalQuantileDetector:
    """Score absolute errors relative to historical seasonal quantiles."""

    def __init__(self, quantile: float = 0.99, min_group_size: int = 12):
        if not 0.5 < quantile < 1:
            raise ValueError("quantile must be between 0.5 and 1")
        self.quantile = quantile
        self.min_group_size = min_group_size
        self.group_thresholds: pd.Series | None = None
        self.fallback: float | None = None

    def fit(self, frame: pd.DataFrame) -> "SeasonalQuantileDetector":
        required = {"total_abs_error", "month", "hour"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        grouped = frame.groupby(["month", "hour"])["total_abs_error"]
        counts = grouped.size()
        thresholds = grouped.quantile(self.quantile)
        self.group_thresholds = thresholds[counts >= self.min_group_size]
        self.fallback = float(frame["total_abs_error"].quantile(self.quantile))
        return self

    def get_scores(self, frame: pd.DataFrame) -> np.ndarray:
        if self.group_thresholds is None or self.fallback is None:
            raise RuntimeError("Detector must be fitted before scoring")
        keys = pd.MultiIndex.from_arrays([frame["month"], frame["hour"]])
        thresholds = self.group_thresholds.reindex(keys).fillna(self.fallback).to_numpy()
        thresholds = np.maximum(thresholds, np.finfo(float).eps)
        return frame["total_abs_error"].to_numpy() / thresholds
