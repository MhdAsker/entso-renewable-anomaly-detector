import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal import conformal_p_values, conformal_predictions
from evaluate import block_bootstrap_confidence_intervals
from event_evaluate import binary_events, compute_event_metrics
from models.seasonal_quantile_model import SeasonalQuantileDetector
from outcomes import OperationalOutcomeFetcher, _safe_error, build_operational_stress_label


class ScientificPhase1Tests(unittest.TestCase):
    def test_event_metrics_use_one_to_one_matching(self):
        index = pd.date_range("2023-01-01", periods=8, freq="h")
        truth = np.array([0, 1, 1, 0, 0, 1, 0, 0])
        predictions = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        metrics = compute_event_metrics(truth, predictions, index)
        self.assertEqual(metrics["n_true_events"], 2)
        self.assertEqual(metrics["n_predicted_events"], 3)
        self.assertEqual(metrics["matched_events"], 2)
        self.assertAlmostEqual(metrics["event_recall"], 1.0)
        self.assertAlmostEqual(metrics["event_precision"], 2 / 3)

    def test_event_segmentation_breaks_across_missing_hours(self):
        index = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 03:00"])
        self.assertEqual(len(binary_events(np.array([1, 1]), index)), 2)

    def test_block_bootstrap_is_reproducible(self):
        y = np.array([0, 0, 1, 1] * 20)
        predictions = np.array([0, 1, 1, 1] * 20)
        scores = np.linspace(0, 1, len(y))
        first = block_bootstrap_confidence_intervals(
            y, predictions, scores, block_size=8, n_bootstrap=20, random_state=7
        )
        second = block_bootstrap_confidence_intervals(
            y, predictions, scores, block_size=8, n_bootstrap=20, random_state=7
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first["f1"][0], first["f1"][1])

    def test_seasonal_quantile_uses_fallback_for_unseen_group(self):
        training = pd.DataFrame(
            {"total_abs_error": np.arange(1, 21), "month": 1, "hour": 0}
        )
        detector = SeasonalQuantileDetector(quantile=0.9, min_group_size=5).fit(training)
        test = pd.DataFrame({"total_abs_error": [10.0], "month": [2], "hour": [3]})
        self.assertAlmostEqual(
            detector.get_scores(test)[0], 10 / training["total_abs_error"].quantile(0.9)
        )

    def test_conformal_values_have_finite_sample_correction(self):
        calibration = np.array([1.0, 2.0, 3.0])
        test = np.array([4.0, 2.0])
        np.testing.assert_allclose(conformal_p_values(calibration, test), [0.25, 0.75])
        np.testing.assert_array_equal(
            conformal_predictions(calibration, test, alpha=0.25), [1, 0]
        )

    def test_outcome_label_preserves_unknown_hours(self):
        index = pd.date_range("2023-01-01", periods=8, freq="h")
        outcomes = pd.DataFrame(
            {
                "imbalance_price_value": [1, 2, 3, 4, np.nan, 5, 100, 6],
                "imbalance_volume_value": [np.nan] * 8,
            },
            index=index,
        )
        label, metadata = build_operational_stress_label(outcomes, index[3], quantile=0.75)
        self.assertTrue(pd.isna(label.iloc[4]))
        self.assertEqual(label.iloc[6], 1)
        self.assertAlmostEqual(metadata["thresholds"]["price_abs_quantile"], 3.25)
        self.assertEqual(metadata["unavailable_components"], ["volume"])

    def test_api_errors_redact_tokens(self):
        message = _safe_error(
            RuntimeError("https://example.test?securityToken=secret-value&documentType=A85")
        )
        self.assertNotIn("secret-value", message)
        self.assertIn("<redacted>", message)

    def test_outcome_fetch_rejects_reversed_range_before_api_call(self):
        fetcher = OperationalOutcomeFetcher.__new__(OperationalOutcomeFetcher)
        with self.assertRaises(ValueError):
            fetcher.fetch("20240102", "20240101")


if __name__ == "__main__":
    unittest.main()
