import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)


class IsolationForestDetector:
    """
    Unsupervised anomaly detection using Isolation Forest.
    Anomalies = points that are easy to isolate in feature space.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.predictions: np.ndarray | None = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.fit_transform(X)

        # IsolationForest: -1 = anomaly, 1 = normal
        raw = self.model.fit_predict(X_scaled)
        preds = (raw == -1).astype(int)

        n = int(preds.sum())
        pct = 100 * n / len(preds)
        logger.info("IsolationForest flagged %d anomalies (%.1f%%)", n, pct)

        os.makedirs(_MODELS_DIR, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler},
            os.path.join(_MODELS_DIR, "iforest.pkl"),
        )

        self.predictions = preds
        return preds

    def get_scores(self, X: np.ndarray) -> np.ndarray:
        """Return negated decision function — higher score = more anomalous."""
        X_scaled = self.scaler.transform(X)
        return -self.model.decision_function(X_scaled)
