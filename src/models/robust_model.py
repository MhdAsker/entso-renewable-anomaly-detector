"""Causal rolling median/MAD anomaly baseline."""

import numpy as np
import pandas as pd


class RobustMADDetector:
    """Score absolute forecast errors against a trailing robust distribution."""

    def __init__(self, window: int = 720, min_periods: int = 168, threshold: float = 3.5):
        self.window = window
        self.min_periods = min_periods
        self.threshold = threshold

    def get_scores(self, df: pd.DataFrame) -> np.ndarray:
        values = df["total_abs_error"]
        history = values.shift(1)
        median = history.rolling(self.window, min_periods=self.min_periods).median()
        mad = history.rolling(self.window, min_periods=self.min_periods).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        # 0.67449 makes MAD scores comparable with z-scores for normal data.
        denominator = mad.replace(0, np.nan)
        return (0.67449 * (values - median) / denominator).clip(lower=0).to_numpy()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        scores = self.get_scores(df)
        return (scores > self.threshold).astype(int)

    def set_threshold(self, threshold: float) -> "RobustMADDetector":
        self.threshold = float(threshold)
        return self
