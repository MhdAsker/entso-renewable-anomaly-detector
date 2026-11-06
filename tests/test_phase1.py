import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import compute_anomaly_metrics
from features import build_features
from models.lstm_autoencoder import LSTMAnomalyDetector
from models.zscore_model import ZScoreDetector


class FeatureTests(unittest.TestCase):
    def make_raw(self, rows=800):
        index = pd.date_range("2023-01-01", periods=rows, freq="h")
        forecast_wind = np.full(rows, 10_000.0)
        forecast_solar = np.full(rows, 5_000.0)
        error = np.linspace(-500, 500, rows)
        actual_wind = forecast_wind + error * 0.6
        actual_solar = forecast_solar + error * 0.4
        return pd.DataFrame(
            {
                "actual_wind_total_mw": actual_wind,
                "forecast_wind_total_mw": forecast_wind,
                "actual_solar_mw": actual_solar,
                "forecast_solar_mw": forecast_solar,
                "actual_renewable_mw": actual_wind + actual_solar,
                "forecast_renewable_mw": forecast_wind + forecast_solar,
            },
            index=index,
        )

    def test_label_reference_window_is_strictly_historical(self):
        raw = self.make_raw()
        result = build_features(raw, output_path=ROOT / "tests" / "_features.parquet")
        ts = raw.index[300]
        previous_abs_errors = (
            raw["actual_renewable_mw"] - raw["forecast_renewable_mw"]
        ).abs().iloc[:300]
        self.assertAlmostEqual(result.loc[ts, "rolling_mean"], previous_abs_errors.mean())

    def test_ramp_error_uses_total_renewable_ramp(self):
        raw = self.make_raw()
        result = build_features(raw, output_path=ROOT / "tests" / "_features.parquet")
        expected = raw["actual_renewable_mw"].diff() - raw["forecast_renewable_mw"].diff()
        ts = result.index[10]
        self.assertAlmostEqual(result.loc[ts, "ramp_error_mw"], expected.loc[ts])

    @classmethod
    def tearDownClass(cls):
        path = ROOT / "tests" / "_features.parquet"
        if path.exists():
            path.unlink()


class DetectorTests(unittest.TestCase):
    def test_zscore_excludes_current_value_from_history(self):
        values = pd.Series([0.0] * 24 + [100.0])
        score = ZScoreDetector()._rolling_zscore(values)
        self.assertTrue(np.isinf(score.iloc[-1]) or np.isnan(score.iloc[-1]))

    def test_lstm_sequence_count_and_alignment(self):
        detector = LSTMAnomalyDetector(sequence_len=4)
        sequences = detector._make_sequences(np.arange(12).reshape(6, 2))
        self.assertEqual(tuple(sequences.shape), (3, 4, 2))
        np.testing.assert_array_equal(sequences[-1, -1].numpy(), np.array([10, 11]))

    def test_metrics_handle_single_class(self):
        metrics = compute_anomaly_metrics(
            np.zeros(4, dtype=int), np.zeros(4, dtype=int), np.arange(4), "test"
        )
        self.assertTrue(np.isnan(metrics["roc_auc"]))


if __name__ == "__main__":
    unittest.main()
