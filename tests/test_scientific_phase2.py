import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustness import (
    circular_shift_negative_control,
    corrupt_features,
    holm_adjust,
    paired_block_bootstrap_difference,
    scenario_utility,
)
from study_data import load_dataset_catalog, validate_dataset_period


class ScientificPhase2Tests(unittest.TestCase):
    def test_paired_bootstrap_detects_identical_models(self):
        y = np.array([0, 0, 1, 1] * 30)
        predictions = np.array([0, 0, 1, 1] * 30)
        scores = np.linspace(0, 1, len(y))
        result = paired_block_bootstrap_difference(
            y,
            predictions,
            scores,
            predictions,
            scores,
            metric="f1",
            block_size=12,
            n_bootstrap=25,
        )
        self.assertEqual(result["observed_difference"], 0)
        self.assertEqual(result["lower"], 0)
        self.assertEqual(result["upper"], 0)

    def test_negative_control_is_reproducible(self):
        y = np.array([0] * 20 + [1] * 5 + [0] * 20 + [1] * 5)
        predictions = y.copy()
        scores = predictions.astype(float)
        first = circular_shift_negative_control(
            y, predictions, scores, metric="f1", min_shift=5, n_shifts=20, random_state=9
        )
        second = circular_shift_negative_control(
            y, predictions, scores, metric="f1", min_shift=5, n_shifts=20, random_state=9
        )
        self.assertEqual(first, second)
        self.assertGreater(first["observed"], first["null_mean"])

    def test_holm_adjustment_is_monotonic_in_rank(self):
        p_values = np.array([0.04, 0.001, 0.02])
        adjusted = holm_adjust(p_values)
        np.testing.assert_allclose(adjusted, [0.04, 0.003, 0.04])

    def test_scenario_utility_uses_declared_coefficients(self):
        result = scenario_utility(
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0]),
            false_positive_cost=0.5,
            false_negative_cost=2,
        )
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertAlmostEqual(result["utility"], -1.5)

    def test_missingness_corruption_uses_fit_median(self):
        values = np.ones((4, 2))
        result = corrupt_features(
            values,
            np.array([10.0, 20.0]),
            np.ones(2),
            corruption="missingness",
            level=1.0,
            random_state=42,
        )
        np.testing.assert_array_equal(result, np.tile([10.0, 20.0], (4, 1)))

    def test_gaussian_noise_zero_level_is_identity(self):
        values = np.arange(8, dtype=float).reshape(4, 2)
        result = corrupt_features(
            values,
            np.zeros(2),
            np.ones(2),
            corruption="gaussian_noise",
            level=0,
            random_state=42,
        )
        np.testing.assert_array_equal(result, values)

    def test_catalog_and_period_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.csv"
            pd.DataFrame(
                {
                    "dataset_id": ["sample"],
                    "feature_path": ["features.parquet"],
                    "outcome_path": [""],
                    "zone": ["TEST"],
                    "period_start": ["2023-01-01"],
                    "period_end": ["2023-12-31"],
                    "label_type": ["rolling_forecast_error_proxy"],
                    "evidence_level": ["internal_proxy"],
                    "active": [True],
                }
            ).to_csv(catalog, index=False)
            specification = load_dataset_catalog(catalog, root)[0]
            frame = pd.DataFrame(
                {"value": [1]}, index=pd.to_datetime(["2023-06-01"])
            )
            validate_dataset_period(frame, specification)
            self.assertEqual(specification.dataset_id, "sample")
            self.assertIsNone(specification.outcome_path)

    def test_catalog_rejects_unknown_evidence_level(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.csv"
            pd.DataFrame(
                {
                    "dataset_id": ["sample"],
                    "feature_path": ["features.parquet"],
                    "outcome_path": [""],
                    "zone": ["TEST"],
                    "period_start": ["2023-01-01"],
                    "period_end": ["2023-12-31"],
                    "label_type": ["proxy"],
                    "evidence_level": ["unsupported"],
                    "active": [True],
                }
            ).to_csv(catalog, index=False)
            with self.assertRaises(ValueError):
                load_dataset_catalog(catalog, root)

    def test_external_evidence_requires_outcome_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.csv"
            pd.DataFrame(
                {
                    "dataset_id": ["external"],
                    "feature_path": ["features.parquet"],
                    "outcome_path": [""],
                    "zone": ["TEST"],
                    "period_start": ["2023-01-01"],
                    "period_end": ["2023-12-31"],
                    "label_type": ["operational_stress_proxy"],
                    "evidence_level": ["external_proxy"],
                    "active": [True],
                }
            ).to_csv(catalog, index=False)
            with self.assertRaises(ValueError):
                load_dataset_catalog(catalog, root)


if __name__ == "__main__":
    unittest.main()
