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
        self.threshold: float | None = None

    def fit(self, X: np.ndarray, save_artifact: bool = True) -> "IsolationForestDetector":
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        if save_artifact:
            self.save()
        return self

    def save(self) -> None:
        os.makedirs(_MODELS_DIR, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "threshold": self.threshold},
            os.path.join(_MODELS_DIR, "iforest.pkl"),
        )

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        artifact = joblib.load(path)
        detector = cls()
        detector.model = artifact["model"]
        detector.scaler = artifact["scaler"]
        detector.threshold = artifact.get("threshold")
        return detector

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)

        # IsolationForest: -1 = anomaly, 1 = normal
        if self.threshold is None:
            raw = self.model.predict(X_scaled)
            preds = (raw == -1).astype(int)
        else:
            preds = (-self.model.decision_function(X_scaled) >= self.threshold).astype(int)

        n = int(preds.sum())
        pct = 100 * n / len(preds)
        logger.info("IsolationForest flagged %d anomalies (%.1f%%)", n, pct)

        self.predictions = preds
        return preds

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Compatibility helper. Prefer separate fit/predict for evaluation."""
        return self.fit(X).predict(X)

    def get_scores(self, X: np.ndarray) -> np.ndarray:
        """Return negated decision function — higher score = more anomalous."""
        X_scaled = self.scaler.transform(X)
        return -self.model.decision_function(X_scaled)
