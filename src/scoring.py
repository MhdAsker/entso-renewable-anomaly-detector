"""Load a registered model version and score engineered hourly features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import predictions_from_scores
from models.iforest_model import IsolationForestDetector
from models.lstm_autoencoder import LSTMAnomalyDetector
from models.robust_model import RobustMADDetector
from models.zscore_model import ZScoreDetector
from registry import DEFAULT_REGISTRY, resolve_run


class OperationalScorer:
    def __init__(
        self, version: str = "latest", registry_dir: str | Path = DEFAULT_REGISTRY
    ):
        self.run_dir, self.manifest = resolve_run(version, registry_dir)
        artifacts = self.manifest["artifacts"]
        self.iforest = IsolationForestDetector.load(
            str(self.run_dir / artifacts["isolation_forest"]["file"])
        )
        self.lstm = LSTMAnomalyDetector.load(
            str(self.run_dir / artifacts["lstm_autoencoder"]["file"])
        )

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        feature_cols = self.manifest["feature_columns"]
        missing = sorted(set(feature_cols) - set(features.columns))
        if missing:
            raise ValueError(f"Scoring data is missing registered features: {missing}")
        X = features[feature_cols].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("Scoring features contain NaN or infinite values")

        thresholds = self.manifest["thresholds"]
        score_map = {
            "zscore": ZScoreDetector().get_scores(features),
            "robust_mad": RobustMADDetector().get_scores(features),
            "isolation_forest": self.iforest.get_scores(X),
            "lstm_autoencoder": self.lstm.get_scores(X),
        }
        rows = []
        for model, scores in score_map.items():
            threshold = float(thresholds[model])
            predictions = predictions_from_scores(scores, threshold)
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": features.index,
                        "model": model,
                        "score": scores,
                        "threshold": threshold,
                        "is_anomaly": predictions,
                        "model_version": self.manifest["run_id"],
                    }
                )
            )
        result = pd.concat(rows, ignore_index=True)
        return result.sort_values(["timestamp", "model"]).reset_index(drop=True)
