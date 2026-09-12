import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.spatial_risk import sample_spatial_risk_fields
from tools.analyze_pair_failure_correlation import (
    BETA_VALUES,
    INDEPENDENT_BASELINE_TYPE,
    compute_failure_probability_samples,
    compute_hazard_multipliers,
    pair_failure_statistics,
    run_diagnostic,
    theoretical_hazard_correlation,
)


class PairFailureCorrelationTests(unittest.TestCase):
    @staticmethod
    def _pair_probabilities(rho, beta=0.8, sample_count=40_000, seed=2026):
        correlation = np.array([[1.0, rho], [rho, 1.0]])
        spatial_z = sample_spatial_risk_fields(
            correlation,
            sample_count,
            rng=np.random.default_rng(seed),
        )
        base_rates = np.array([0.002, 0.002])
        service_times = np.array([5.0, 5.0])
        spatial_rates = base_rates[np.newaxis, :] * compute_hazard_multipliers(
            spatial_z,
            beta,
        )
        return compute_failure_probability_samples(spatial_rates, service_times)

    def test_beta_zero_is_exact_independent_control(self):
        spatial = self._pair_probabilities(rho=0.8, beta=0.0)
        metrics = pair_failure_statistics(spatial[:, 0], spatial[:, 1])

        self.assertEqual(metrics["joint_spatial"], metrics["joint_independent"])
        self.assertEqual(metrics["joint_failure_amplification"], 1.0)
        self.assertEqual(metrics["excess_joint_failure"], 0.0)
        self.assertEqual(metrics["binary_failure_correlation"], 0.0)

    def test_zero_latent_correlation_gives_marginal_product(self):
        spatial = self._pair_probabilities(rho=0.0, beta=0.8)
        metrics = pair_failure_statistics(spatial[:, 0], spatial[:, 1])

        self.assertAlmostEqual(
            metrics["joint_spatial"],
            metrics["marginal_j"] * metrics["marginal_k"],
            delta=0.002,
        )
        self.assertAlmostEqual(
            metrics["joint_failure_amplification"],
            1.0,
            delta=0.03,
        )
        self.assertLess(abs(metrics["binary_failure_correlation"]), 0.02)

    def test_positive_latent_correlation_increases_joint_failure(self):
        spatial = self._pair_probabilities(rho=0.8, beta=0.8)
        metrics = pair_failure_statistics(spatial[:, 0], spatial[:, 1])

        self.assertGreater(
            metrics["joint_spatial"],
            metrics["marginal_j"] * metrics["marginal_k"],
        )
        self.assertGreater(metrics["joint_failure_amplification"], 1.0)
        self.assertGreater(metrics["binary_failure_correlation"], 0.0)

    def test_equal_node_joint_failure_increases_with_rho(self):
        joints = []
        for rho in (0.0, 0.3, 0.8):
            spatial = self._pair_probabilities(rho=rho, beta=0.8)
            metrics = pair_failure_statistics(spatial[:, 0], spatial[:, 1])
            joints.append(metrics["joint_spatial"])

        self.assertLess(joints[0], joints[1])
        self.assertLess(joints[1], joints[2])

    def test_empirical_latent_correlation_matches_target(self):
        rho = 0.6
        samples = sample_spatial_risk_fields(
            np.array([[1.0, rho], [rho, 1.0]]),
            50_000,
            rng=np.random.default_rng(2026),
        )
        empirical = np.corrcoef(samples, rowvar=False)[0, 1]
        self.assertAlmostEqual(empirical, rho, delta=0.02)

    def test_empirical_hazard_correlation_matches_theory(self):
        rho = 0.6
        beta = 0.8
        samples = sample_spatial_risk_fields(
            np.array([[1.0, rho], [rho, 1.0]]),
            50_000,
            rng=np.random.default_rng(2026),
        )
        multipliers = compute_hazard_multipliers(samples, beta)
        empirical = np.corrcoef(multipliers, rowvar=False)[0, 1]
        theoretical = theoretical_hazard_correlation(beta, rho)
        self.assertAlmostEqual(empirical, theoretical, delta=0.02)

    def test_matched_marginal_baseline_is_exact(self):
        spatial = self._pair_probabilities(rho=0.8, beta=0.8)
        metrics = pair_failure_statistics(spatial[:, 0], spatial[:, 1])

        self.assertEqual(
            metrics["joint_independent"],
            metrics["marginal_j"] * metrics["marginal_k"],
        )
        self.assertEqual(metrics["marginal_j_independent"], metrics["marginal_j"])
        self.assertEqual(metrics["marginal_k_independent"], metrics["marginal_k"])
        self.assertEqual(metrics["absolute_marginal_difference_j"], 0.0)
        self.assertEqual(metrics["absolute_marginal_difference_k"], 0.0)
        self.assertEqual(
            metrics["independent_baseline_type"],
            "matched_marginal_product",
        )

    def test_reliability_overestimation_equals_excess_joint_failure(self):
        spatial = self._pair_probabilities(rho=0.8, beta=0.8)
        metrics = pair_failure_statistics(spatial[:, 0], spatial[:, 1])

        self.assertEqual(
            metrics["reliability_overestimation"],
            metrics["excess_joint_failure"],
        )

    def test_diagnostic_outputs_only_distinct_server_pairs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            server_path = root / "server_info.xlsx"
            task_path = root / "task_parameters.xlsx"
            output_dir = root / "diagnostics"
            pd.DataFrame({
                "Server_ID": [9, 2, 5],
                "Processing_Frequency": [10.0, 12.0, 14.0],
                "Failure_Rate": [0.001, 0.002, 0.0005],
                "Latitude": [-37.814395, -37.820910, -37.812390],
                "Longitude": [144.963537, 144.955155, 144.971200],
            }).to_excel(server_path, index=False)
            pd.DataFrame({
                "Task_ID": [1, 2],
                "Computation_Demand": [10.0, 20.0],
            }).to_excel(task_path, index=False)

            detail, pair_summary, global_summary = run_diagnostic(
                server_path=server_path,
                task_path=task_path,
                output_dir=output_dir,
                sample_count=1000,
            )

            self.assertEqual(len(detail), len(BETA_VALUES) * 2 * 3)
            self.assertEqual(len(pair_summary), len(BETA_VALUES) * 3)
            self.assertEqual(len(global_summary), len(BETA_VALUES))
            self.assertTrue((detail["Server_J"] < detail["Server_K"]).all())
            self.assertFalse((detail["Server_J"] == detail["Server_K"]).any())
            self.assertTrue(
                detail["Marginal_Failure_J_Independent"].equals(
                    detail["Marginal_Failure_J_Spatial"]
                )
            )
            self.assertTrue(
                detail["Marginal_Failure_K_Independent"].equals(
                    detail["Marginal_Failure_K_Spatial"]
                )
            )
            self.assertTrue((detail["Absolute_Marginal_Difference_J"] == 0.0).all())
            self.assertTrue((detail["Absolute_Marginal_Difference_K"] == 0.0).all())
            self.assertTrue(
                (pair_summary["Max_Absolute_Marginal_Difference"] == 0.0).all()
            )
            self.assertTrue(
                (global_summary["Mean_Absolute_Marginal_Difference"] == 0.0).all()
            )
            self.assertTrue(
                (global_summary["Max_Absolute_Marginal_Difference"] == 0.0).all()
            )

            output_frames = {
                "task_pair_joint_failure.csv": detail,
                "pair_failure_correlation_summary.csv": pair_summary,
                "pair_failure_global_summary.csv": global_summary,
            }
            for filename, expected_frame in output_frames.items():
                output_path = output_dir / filename
                self.assertTrue(output_path.exists())
                written_frame = pd.read_csv(output_path)
                self.assertEqual(len(written_frame), len(expected_frame))
                self.assertEqual(
                    set(written_frame["Independent_Baseline_Type"]),
                    {INDEPENDENT_BASELINE_TYPE},
                )
                if filename == "task_pair_joint_failure.csv":
                    self.assertTrue(
                        written_frame["Marginal_Failure_J_Independent"].equals(
                            written_frame["Marginal_Failure_J_Spatial"]
                        )
                    )
                    self.assertTrue(
                        written_frame["Marginal_Failure_K_Independent"].equals(
                            written_frame["Marginal_Failure_K_Spatial"]
                        )
                    )
                    self.assertTrue(
                        (written_frame["Absolute_Marginal_Difference_J"] == 0.0).all()
                    )
                    self.assertTrue(
                        (written_frame["Absolute_Marginal_Difference_K"] == 0.0).all()
                    )


if __name__ == "__main__":
    unittest.main()
