"""Tests for matched-environment failure-rate scale sensitivity."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.spatial_risk import build_spatial_correlation_matrix
from tools.analyze_conditional_joint_reliability import (
    calculate_conditional_joint_reliability,
)
from tools.analyze_failure_rate_scale_sensitivity import (
    _build_coverage,
    calculate_failure_rate_scale_sensitivity,
    run_diagnostic,
)


class FailureRateScaleSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.base_rates = np.array([0.001, 0.002, 0.003])
        self.frequencies = np.array([10.0, 12.0, 15.0])
        self.demands = np.array([10.0, 40.0, 90.0, 100.0])
        self.distance_matrix = np.array(
            [[0.0, 0.3, 1.0], [0.3, 0.0, 0.8], [1.0, 0.8, 0.0]]
        )
        self.correlation_matrix = build_spatial_correlation_matrix(
            self.distance_matrix, correlation_length_km=0.5
        )
        self.scales = (1.0, 3.0, 5.0)

    def _sensitivity(self, **overrides):
        options = {
            "base_failure_rates": self.base_rates,
            "processing_frequencies": self.frequencies,
            "computation_demands": self.demands,
            "correlation_matrix": self.correlation_matrix,
            "failure_rate_scales": self.scales,
            "episodes": 4,
            "beta_p": 0.8,
            "seed": 2026,
        }
        options.update(overrides)
        return calculate_failure_rate_scale_sensitivity(**options)

    def test_scale_one_matches_existing_conditional_diagnostic(self):
        baseline = calculate_conditional_joint_reliability(
            self.base_rates,
            self.frequencies,
            self.demands,
            self.correlation_matrix,
            episodes=4,
            beta_p=0.8,
            seed=2026,
        )
        result = self._sensitivity()
        scale_one = result["data_by_scale"][1.0]
        for key in ("spatial_fields", "effective_rates", "failure_probabilities", "joint_failure", "joint_success"):
            self.assertTrue(np.array_equal(scale_one[key], baseline[key]))

    def test_matched_fields_and_probability_monotonicity(self):
        result = self._sensitivity()
        fields = result["spatial_fields"]
        self.assertEqual(fields.shape, (4, 3))
        for scale in self.scales:
            self.assertTrue(np.array_equal(result["data_by_scale"][scale]["spatial_fields"], fields))

        for lower, higher in zip(self.scales, self.scales[1:]):
            lower_data = result["data_by_scale"][lower]
            higher_data = result["data_by_scale"][higher]
            self.assertTrue(
                np.all(higher_data["failure_probabilities"] >= lower_data["failure_probabilities"])
            )
            self.assertTrue(np.all(higher_data["joint_failure"] >= lower_data["joint_failure"]))
            self.assertTrue(np.all(higher_data["joint_success"] <= lower_data["joint_success"]))

    def test_coverage_is_non_increasing_with_requirement_and_reproducible(self):
        first = self._sensitivity()
        second = self._sensitivity()
        requirements = (0.9, 0.99, 0.999, 0.9999)
        first_coverage = _build_coverage(first["data_by_scale"], requirements)
        second_coverage = _build_coverage(second["data_by_scale"], requirements)
        self.assertTrue(np.array_equal(first["spatial_fields"], second["spatial_fields"]))
        self.assertTrue(np.array_equal(first_coverage.to_numpy(), second_coverage.to_numpy()))
        for scale in self.scales:
            values = first_coverage.loc[
                first_coverage["failure_rate_scale"] == scale, "success_rate"
            ].to_numpy()
            self.assertTrue(np.all(np.diff(values) <= 0.0))

    def test_run_writes_expected_outputs_and_sample_counts(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            server_path = root / "server_info.xlsx"
            task_path = root / "task_parameters.xlsx"
            output_dir = root / "diagnostics"
            pd.DataFrame(
                {
                    "Server_ID": [1, 2, 3],
                    "Server_Type": ["Edge", "Edge", "Cloud"],
                    "Processing_Frequency": self.frequencies,
                    "Failure_Rate": self.base_rates,
                    "Latitude": [-37.80, -37.81, -37.82],
                    "Longitude": [144.95, 144.96, 144.97],
                }
            ).to_excel(server_path, index=False)
            pd.DataFrame(
                {"Task_ID": [1, 2, 3, 4], "Computation_Demand": self.demands}
            ).to_excel(task_path, index=False)

            result = run_diagnostic(
                server_path=server_path,
                task_path=task_path,
                output_dir=output_dir,
                failure_rate_scales=self.scales,
                reliability_requirements=(0.9, 0.99, 0.999, 0.9999),
                episodes=3,
                tasks_per_episode=4,
                beta_p=0.8,
                seed=2026,
            )

            coverage = result["coverage"]
            self.assertEqual(len(coverage), 12)
            self.assertTrue(np.all(coverage["total_samples"] == 36))
            self.assertTrue(np.all(coverage["success_samples"] + (coverage["failure_rate"] * 36).round().astype(int) == 36))
            self.assertEqual(
                set(result["outputs"]), {"coverage", "summary", "threshold_matrix"}
            )
            self.assertEqual(
                set(result["threshold_matrix"].columns),
                {"failure_rate_scale", "R_0.9", "R_0.99", "R_0.999", "R_0.9999"},
            )
            self.assertTrue(all(path.is_file() for path in result["outputs"].values()))


if __name__ == "__main__":
    unittest.main()
