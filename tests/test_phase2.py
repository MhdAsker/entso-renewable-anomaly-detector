import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import expanding_window_folds
from capacity import capacities_for_index
from evaluate import calibrate_threshold, predictions_from_scores
from fetch import ENTSOFetcher
from labels import load_event_labels
from models.robust_model import RobustMADDetector


class Phase2Tests(unittest.TestCase):
    def test_fetch_retry_recovers_from_transient_failure(self):
        fetcher = ENTSOFetcher.__new__(ENTSOFetcher)
        fetcher.max_attempts = 3
        fetcher.retry_delay = 0
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("temporary")
            return "ok"

        self.assertEqual(fetcher._fetch_with_retry(flaky), "ok")
        self.assertEqual(len(attempts), 3)

    def test_date_effective_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capacity.csv"
            pd.DataFrame(
                {
                    "effective_date": ["2023-01-01", "2023-07-01"],
                    "wind_mw": [10, 20],
                    "solar_mw": [30, 40],
                }
            ).to_csv(path, index=False)
            index = pd.to_datetime(["2023-06-30", "2023-07-01"])
            result = capacities_for_index(index, path)
            self.assertEqual(result["wind_mw"].tolist(), [10, 20])

    def test_validation_threshold_calibration(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        threshold = calibrate_threshold(y, scores)
        np.testing.assert_array_equal(
            predictions_from_scores(scores, threshold), np.array([0, 0, 1, 1])
        )

    def test_expanding_folds_are_ordered_and_tests_do_not_overlap(self):
        folds = expanding_window_folds(1000)
        self.assertGreaterEqual(len(folds), 3)
        for fold in folds:
            self.assertLessEqual(fold.fit.stop, fold.validation.start)
            self.assertLessEqual(fold.validation.stop, fold.test.start)
        for previous, current in zip(folds, folds[1:]):
            self.assertLessEqual(previous.test.stop, current.test.start)

    def test_robust_mad_ranks_large_outlier_highest(self):
        rng = np.random.default_rng(42)
        values = np.abs(rng.normal(100, 10, 250))
        values[-1] = 1000
        frame = pd.DataFrame({"total_abs_error": values})
        scores = RobustMADDetector(window=200, min_periods=50).get_scores(frame)
        self.assertEqual(int(np.nanargmax(scores)), len(values) - 1)

    def test_external_event_label_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame(
                {"timestamp": ["2023-01-01T00:00:00Z"], "is_event": [1]}
            ).to_csv(path, index=False)
            labels = load_event_labels(path)
            self.assertEqual(labels.iloc[0], 1)
            self.assertIsNone(labels.index.tz)


if __name__ == "__main__":
    unittest.main()
