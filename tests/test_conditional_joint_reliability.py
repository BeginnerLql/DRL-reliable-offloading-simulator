"""Focused tests for the offline conditional joint-reliability diagnostic."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

from core.spatial_risk import build_spatial_correlation_matrix
from tools.analyze_conditional_joint_reliability import (
    calculate_conditional_joint_reliability,
    run_diagnostic,
)


class ConditionalJointReliabilityTests(unittest.TestCase):
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

    def _calculation(self, **overrides):
        options = {
            "base_failure_rates": self.base_rates,
            "processing_frequencies": self.frequencies,
            "computation_demands": self.demands,
            "correlation_matrix": self.correlation_matrix,
            "episodes": 4,
            "beta_p": 0.8,
            "seed": 2026,
        }
        options.update(overrides)
        return calculate_conditional_joint_reliability(**options)

    def test_joint_failure_and_success_are_bounded_and_exact(self):
        result = self._calculation()
        failure_probabilities = result["failure_probabilities"]
        joint_failure = result["joint_failure"]
        joint_success = result["joint_success"]

        self.assertTrue(
            np.all((failure_probabilities >= 0.0) & (failure_probabilities <= 1.0))
        )
        self.assertTrue(np.all((joint_failure >= 0.0) & (joint_failure <= 1.0)))
        self.assertTrue(np.all((joint_success >= 0.0) & (joint_success <= 1.0)))
        self.assertTrue(np.allclose(joint_success, 1.0 - joint_failure))

        for pair_position, (first_server, second_server) in enumerate(
            result["pair_indices"]
        ):
            expected = (
                failure_probabilities[:, :, first_server]
                * failure_probabilities[:, :, second_server]
            )
            self.assertTrue(np.allclose(joint_failure[:, :, pair_position], expected))

    def test_one_spatial_field_is_fixed_within_episode_and_changes_between_episodes(self):
        result = self._calculation()
        self.assertEqual(result["spatial_fields"].shape, (4, len(self.base_rates)))
        self.assertTrue(np.all(result["effective_rates"] > 0.0))
        service_times = self.demands[:, None] / self.frequencies[None, :]
        reconstructed_effective_rates = -np.log1p(-result["failure_probabilities"])
        reconstructed_effective_rates /= service_times[None, :, :]

        for server_index in range(len(self.base_rates)):
            expected = result["effective_rates"][:, server_index, None]
            self.assertTrue(
                np.allclose(
                    reconstructed_effective_rates[:, :, server_index], expected
                )
            )

        self.assertFalse(
            np.array_equal(result["spatial_fields"][0], result["spatial_fields"][1])
        )

    def test_fixed_seed_is_reproducible_and_beta_zero_recovers_base_rates(self):
        first = self._calculation()
        second = self._calculation()
        for key in (
            "spatial_fields",
            "effective_rates",
            "failure_probabilities",
            "joint_failure",
            "joint_success",
        ):
            self.assertTrue(np.array_equal(first[key], second[key]))

        beta_zero = self._calculation(beta_p=0.0)
        expected_rates = np.broadcast_to(self.base_rates, (4, len(self.base_rates)))
        self.assertTrue(np.array_equal(beta_zero["effective_rates"], expected_rates))

    def test_offline_default_scale_is_independent_of_runtime_config(self):
        from config.params import params
        with patch.object(params, "FAILURE_RATE_SCALE", 10.0):
            runtime_config_result = self._calculation()
        expected = np.vstack([
            self.base_rates * np.exp(
                0.8 * field - 0.8**2 / 2.0
            )
            for field in runtime_config_result["spatial_fields"]
        ])
        self.assertTrue(np.allclose(runtime_config_result["effective_rates"], expected))
        explicit_scale_result = self._calculation(failure_rate_scale=10.0)
        expected_explicit = np.vstack([
            10.0 * self.base_rates * np.exp(
                0.8 * field - 0.8**2 / 2.0
            )
            for field in explicit_scale_result["spatial_fields"]
        ])
        self.assertTrue(np.allclose(explicit_scale_result["effective_rates"], expected_explicit))

    def test_correlation_matrix_is_symmetric_with_unit_diagonal(self):
        self.assertTrue(
            np.allclose(self.correlation_matrix, self.correlation_matrix.T)
        )
        self.assertTrue(np.allclose(np.diag(self.correlation_matrix), 1.0))

    def test_run_diagnostic_writes_conditional_summary_and_samples(self):
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
                episodes=3,
                tasks_per_episode=4,
                beta_p=0.8,
                correlation_length_km=0.5,
                seed=2026,
                sample_rows=20,
            )

            summary = result["summary"]
            self.assertEqual(
                set(summary["metric"]),
                {"joint_failure_probability", "joint_success_probability"},
            )
            self.assertTrue(np.all(summary["count"] == 36))
            self.assertTrue(np.all(summary["min"] >= 0.0))
            self.assertTrue(np.all(summary["max"] <= 1.0))

            by_pair = result["by_pair"]
            self.assertEqual(len(by_pair), 3)
            self.assertTrue(np.all(by_pair["server_j"] < by_pair["server_k"]))
            self.assertEqual(len(result["samples"]), 20)
            self.assertTrue(
                np.allclose(
                    result["samples"]["joint_success"],
                    1.0 - result["samples"]["joint_failure"],
                )
            )

            expected_outputs = {
                "conditional_joint_reliability_summary.csv",
                "conditional_joint_reliability_by_pair.csv",
                "conditional_joint_reliability_samples.csv",
                "conditional_joint_reliability_by_task_load.csv",
            }
            self.assertEqual(
                {path.name for path in result["outputs"].values()}, expected_outputs
            )
            self.assertTrue(all(path.is_file() for path in result["outputs"].values()))


if __name__ == "__main__":
    unittest.main()
