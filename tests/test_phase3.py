import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alerts import build_alert_events, record_feedback, upsert_alerts
from storage import atomic_upsert_parquet, atomic_upsert_scores, read_state, write_state


class Phase3Tests(unittest.TestCase):
    def test_parquet_upsert_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            index = pd.date_range("2023-01-01", periods=2, freq="h")
            frame = pd.DataFrame({"value": [1, 2]}, index=index)
            atomic_upsert_parquet(path, frame)
            result = atomic_upsert_parquet(path, frame)
            self.assertEqual(len(result), 2)

    def test_score_upsert_uses_composite_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.parquet"
            scores = pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2023-01-01")],
                    "model": ["zscore"],
                    "model_version": ["v1"],
                    "score": [1.0],
                }
            )
            atomic_upsert_scores(path, scores)
            scores["score"] = 2.0
            result = atomic_upsert_scores(path, scores)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["score"], 2.0)

    def test_alerts_merge_across_batch_boundary_and_accept_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alerts.csv"
            first = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00"]),
                    "model": ["zscore", "zscore"],
                    "score": [4.0, 5.0],
                    "threshold": [3.0, 3.0],
                    "is_anomaly": [1, 1],
                    "model_version": ["v1", "v1"],
                }
            )
            second = first.iloc[[1]].copy()
            second["timestamp"] = pd.Timestamp("2023-01-01 02:00")
            events1 = build_alert_events(first, {"zscore": {"p99": 4.5, "max": 6}})
            events2 = build_alert_events(second, {"zscore": {"p99": 4.5, "max": 6}})
            upsert_alerts(path, events1)
            merged = upsert_alerts(path, events2)
            self.assertEqual(len(merged), 1)
            event_id = merged.iloc[0]["event_id"]
            record_feedback(path, event_id, "true_positive", "confirmed")
            reviewed = pd.read_csv(path)
            self.assertEqual(reviewed.iloc[0]["status"], "reviewed")

    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            write_state(path, {"last_scored_at": "2023-01-01T00:00:00"})
            self.assertEqual(read_state(path)["last_scored_at"], "2023-01-01T00:00:00")


if __name__ == "__main__":
    unittest.main()
