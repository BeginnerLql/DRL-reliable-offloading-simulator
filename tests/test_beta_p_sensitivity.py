import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.analyze_beta_p_sensitivity import (
    BETA_VALUES,
    compute_hazard_multiplier,
    run_diagnostic,
    theoretical_multiplier_quantile,
)


class BetaPSensitivityTests(unittest.TestCase):
    def test_beta_zero_multiplier_is_exactly_one(self):
        z_samples = np.array([[-2.0, 0.0], [0.5, 3.0]])
        multiplier = compute_hazard_multiplier(z_samples, 0.0)
        self.assertTrue(np.array_equal(multiplier, np.ones_like(z_samples)))

    def test_multiplier_uses_the_configured_formula(self):
        z_samples = np.array([-1.0, 0.0, 1.0])
        beta = 0.5
        expected = np.exp(beta * z_samples - 0.5 * beta**2)
        self.assertTrue(
            np.allclose(compute_hazard_multiplier(z_samples, beta), expected)
        )

    def test_fixed_seed_is_reproducible(self):
        first_z = np.random.default_rng(2026).normal(size=(1000, 3))
        second_z = np.random.default_rng(2026).normal(size=(1000, 3))
        self.assertTrue(np.array_equal(first_z, second_z))
        self.assertTrue(
            np.array_equal(
                compute_hazard_multiplier(first_z, 0.5),
                compute_hazard_multiplier(second_z, 0.5),
            )
        )

    def test_all_betas_use_the_same_z_realizations(self):
        z_samples = np.random.default_rng(2026).normal(size=(1000, 3))
        multipliers = {
            beta: compute_hazard_multiplier(z_samples, beta)
            for beta in BETA_VALUES
        }
        for beta in BETA_VALUES:
            expected = np.exp(beta * z_samples - 0.5 * beta**2)
            self.assertTrue(np.array_equal(multipliers[beta], expected))
        self.assertEqual(multipliers[0.0].shape, z_samples.shape)

    def test_mean_multiplier_is_near_one_for_nonzero_betas(self):
        z_samples = np.random.default_rng(2026).normal(size=100_000)
        for beta in (0.2, 0.5, 0.8):
            with self.subTest(beta=beta):
                mean_multiplier = compute_hazard_multiplier(
                    z_samples,
                    beta,
                ).mean()
                self.assertLess(abs(mean_multiplier - 1.0), 0.02)

    def test_multiplier_std_increases_with_beta(self):
        z_samples = np.random.default_rng(2026).normal(size=100_000)
        stds = [
            compute_hazard_multiplier(z_samples, beta).std()
            for beta in (0.2, 0.5, 0.8)
        ]
        self.assertLess(stds[0], stds[1])
        self.assertLess(stds[1], stds[2])

    def test_empirical_median_matches_theoretical_median(self):
        z_samples = np.random.default_rng(2026).normal(size=100_000)
        for beta in (0.2, 0.5, 0.8):
            with self.subTest(beta=beta):
                empirical = np.median(compute_hazard_multiplier(z_samples, beta))
                theoretical = np.exp(-beta**2 / 2.0)
                self.assertLess(abs(empirical - theoretical), 0.02)

    def test_theoretical_quantile_helper(self):
        beta = 0.5
        expected = np.exp(beta * (-1.6448536269514729) - 0.5 * beta**2)
        self.assertAlmostEqual(
            theoretical_multiplier_quantile(beta, 0.05),
            expected,
            places=12,
        )
        self.assertAlmostEqual(
            theoretical_multiplier_quantile(beta, 0.5),
            np.exp(-beta**2 / 2.0),
            places=12,
        )

    def test_run_diagnostic_writes_summary_and_server_quantiles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            input_path = temporary_directory / "server_info.xlsx"
            output_dir = temporary_directory / "diagnostics"
            pd.DataFrame({
                "Server_ID": [9, 2, 5],
                "Failure_Rate": [0.001, 0.002, 0.0005],
                "Latitude": [-37.814395, -37.820910, -37.812390],
                "Longitude": [144.963537, 144.955155, 144.971200],
            }).to_excel(input_path, index=False)

            summary, server_quantiles = run_diagnostic(
                input_path=input_path,
                output_dir=output_dir,
                sample_count=1000,
            )

            self.assertEqual(summary["Beta_p"].tolist(), list(BETA_VALUES))
            self.assertTrue((summary["Sample_Count"] == 1000).all())
            self.assertEqual(len(server_quantiles), 3 * len(BETA_VALUES))
            self.assertTrue(
                (output_dir / "beta_p_sensitivity_summary.csv").exists()
            )
            self.assertTrue(
                (output_dir / "beta_p_server_effective_rate_quantiles.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
