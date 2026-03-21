import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ZScoreDetector:
    """
    Statistical baseline: flags hours where total_error_norm exceeds
    a rolling z-score threshold. Simple, interpretable, no ML.
    """

    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold
        self.predictions: np.ndarray | None = None

    def _rolling_zscore(self, series: pd.Series) -> pd.Series:
        roll = series.rolling(window=168, min_periods=24)
        z = (series - roll.mean()) / roll.std()
        return z

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        z = self._rolling_zscore(df["total_error_norm"])
        preds = (z.abs() > self.threshold).astype(int).values

        n = int(preds.sum())
        pct = 100 * n / len(preds)
        logger.info("ZScore flagged %d anomalies (%.1f%%)", n, pct)

        self.predictions = preds
        return preds

    def get_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw absolute z-scores — higher means more anomalous."""
        return self._rolling_zscore(df["total_error_norm"]).abs().values
