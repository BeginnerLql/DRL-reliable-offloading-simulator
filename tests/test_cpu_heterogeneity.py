import itertools
import unittest

import numpy as np
import pandas as pd

from tools.analyze_backup_ranking_competition import rank_backup_candidates
from tools.analyze_correlation_aware_backup_selection import (
    MATCHED_BASELINE_TYPE,
    compare_backup_selections,
)
from tools.analyze_cpu_heterogeneity import (
    build_cpu_scenarios,
    contract_cpu_capacities,
    cpu_statistics,
    decorate_ranking_competition,
    decorate_selection_comparison,
    enforce_homogeneous_marginals,
    summarize_sensitivity,
    validate_beta_zero_controls,
    validate_c3_against_h0,
)


class CPUHeterogeneityTests(unittest.TestCase):
    ORIGINAL_CPUS = np.array([10.0, 20.0, 40.0, 50.0])

    @staticmethod
    def _pair_frame(
        marginals,
        beta_p=0.8,
        amplifications=None,
        include_service_times=False,
    ):
        marginals = np.asarray(marginals, dtype=float)
        if amplifications is None:
            amplifications = {(1, 2): 2.0}
        rows = []
        for server_j, server_k in itertools.combinations(range(1, 5), 2):
            marginal_j = marginals[server_j - 1]
            marginal_k = marginals[server_k - 1]
            independent = marginal_j * marginal_k
            amplification = amplifications.get((server_j, server_k), 1.0)
            row = {
                "Beta_p": beta_p,
                "Independent_Baseline_Type": MATCHED_BASELINE_TYPE,
                "Task_ID": 101,
                "Server_J": server_j,
                "Server_K": server_k,
                "Distance_km": float(server_k - server_j),
                "Rho_phy": 0.1 * (server_k - server_j),
                "Marginal_Failure_J_Spatial": marginal_j,
                "Marginal_Failure_K_Spatial": marginal_k,
                "Joint_Failure_Spatial": independent * amplification,
                "Joint_Failure_Independent": independent,
            }
            if include_service_times:
                row["Service_Time_J"] = 2.0
                row["Service_Time_K"] = 2.0
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _decorate(gamma, marginals, beta_p=0.8, amplifications=None):
        pair_frame = CPUHeterogeneityTests._pair_frame(
            marginals,
            beta_p=beta_p,
            amplifications=amplifications,
        )
        base_comparison = compare_backup_selections(pair_frame)
        base_ranking = rank_backup_candidates(pair_frame)
        cpus = contract_cpu_capacities(
            CPUHeterogeneityTests.ORIGINAL_CPUS,
            gamma,
        )
        mean_cpu, std_cpu, cv_cpu = cpu_statistics(cpus)
        cpu_by_server = {
            server_id: float(cpu)
            for server_id, cpu in enumerate(cpus, start=1)
        }
        comparison = decorate_selection_comparison(
            base_comparison,
            gamma,
            "synthetic",
            cpu_by_server,
            mean_cpu,
            std_cpu,
            cv_cpu,
        )
        ranking = decorate_ranking_competition(
            base_ranking,
            base_comparison,
            gamma,
            "synthetic",
            mean_cpu,
            std_cpu,
            cv_cpu,
        )
        return base_comparison, comparison, ranking

    def test_gamma_endpoints_and_mean_invariance(self):
        mean_cpu = self.ORIGINAL_CPUS.mean()
        np.testing.assert_array_equal(
            contract_cpu_capacities(self.ORIGINAL_CPUS, 0.0),
            np.full_like(self.ORIGINAL_CPUS, mean_cpu),
        )
        np.testing.assert_array_equal(
            contract_cpu_capacities(self.ORIGINAL_CPUS, 1.0),
            self.ORIGINAL_CPUS,
        )
        for gamma in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
            self.assertAlmostEqual(
                contract_cpu_capacities(self.ORIGINAL_CPUS, gamma).mean(),
                mean_cpu,
                places=14,
            )

    def test_std_and_cv_scale_linearly(self):
        original_mean, original_std, original_cv = cpu_statistics(
            self.ORIGINAL_CPUS
        )
        for gamma in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
            mean_cpu, std_cpu, cv_cpu = cpu_statistics(
                contract_cpu_capacities(self.ORIGINAL_CPUS, gamma)
            )
            self.assertAlmostEqual(mean_cpu, original_mean, places=14)
            self.assertAlmostEqual(std_cpu, gamma * original_std, places=14)
            self.assertAlmostEqual(cv_cpu, gamma * original_cv, places=14)

        table = build_cpu_scenarios(
            [1, 2, 3, 4],
            self.ORIGINAL_CPUS,
            fixed_failure_rate=0.01,
        )
        self.assertEqual(len(table), 16)
        self.assertEqual(
            set(table["CPU_Heterogeneity_Scenario"]),
            {"C0", "C1", "C2", "C3"},
        )

    def test_invalid_gamma_cpu_and_fixed_rate_are_rejected(self):
        for gamma in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(gamma=gamma):
                with self.assertRaisesRegex(ValueError, "gamma"):
                    contract_cpu_capacities(self.ORIGINAL_CPUS, gamma)

        for cpus in ([10.0, 0.0], [10.0, -1.0], [10.0, np.nan], [10.0, np.inf]):
            with self.subTest(cpus=cpus):
                with self.assertRaisesRegex(ValueError, "CPU capacities"):
                    contract_cpu_capacities(cpus, 0.5)

        for rate in (-0.01, np.nan, np.inf):
            with self.subTest(rate=rate):
                with self.assertRaisesRegex(ValueError, "fixed failure rate"):
                    build_cpu_scenarios([1, 2], [10.0, 20.0], rate)

    def test_gamma_zero_has_equal_service_time_and_marginal_failure(self):
        cpus = contract_cpu_capacities(self.ORIGINAL_CPUS, 0.0)
        service_times = 60.0 / cpus
        marginals = -np.expm1(-0.01 * service_times)
        np.testing.assert_array_equal(service_times, np.full(4, 2.0))
        np.testing.assert_array_equal(marginals, np.full(4, marginals[0]))

        noisy = self._pair_frame(
            [0.0199, 0.0201, 0.0200, 0.0202],
            include_service_times=True,
        )
        pooled = enforce_homogeneous_marginals(noisy)
        unique_marginals = np.unique(np.concatenate([
            pooled["Marginal_Failure_J_Spatial"],
            pooled["Marginal_Failure_K_Spatial"],
        ]))
        self.assertEqual(len(unique_marginals), 1)

    def test_beta_zero_control_and_backup_differs_from_primary(self):
        _, comparison, ranking = self._decorate(
            0.0,
            np.full(4, 0.02),
            beta_p=0.0,
            amplifications={},
        )
        validate_beta_zero_controls(comparison, ranking)
        self.assertFalse(comparison["Selection_Changed"].any())
        self.assertTrue((comparison["Absolute_Risk_Reduction"] == 0.0).all())
        for column in (
            "Independent_Selected_Backup",
            "Correlation_Aware_Selected_Backup",
        ):
            self.assertFalse((comparison[column] == comparison["Primary_Server"]).any())

    def test_homogeneous_cpu_can_have_correlation_driven_change(self):
        _, comparison, ranking = self._decorate(
            0.0,
            np.full(4, 0.02),
            beta_p=0.8,
            amplifications={(1, 2): 2.0},
        )
        primary_one = comparison[comparison["Primary_Server"] == 1].iloc[0]
        self.assertEqual(primary_one["Independent_Selected_Backup"], 2)
        self.assertEqual(primary_one["Correlation_Aware_Selected_Backup"], 3)
        self.assertTrue(primary_one["Selection_Changed"])
        self.assertGreater(primary_one["Relative_Risk_Reduction"], 0.0)
        self.assertTrue(ranking.loc[
            ranking["Primary_Server"] == 1,
            "Actual_Reversal",
        ].iloc[0])

    def test_zero_denominators_do_not_produce_infinity(self):
        _, comparison, ranking = self._decorate(
            0.0,
            np.zeros(4),
            beta_p=0.8,
            amplifications={},
        )
        summary = summarize_sensitivity(comparison, ranking)
        for frame in (comparison, ranking, summary):
            numeric = frame.select_dtypes(include=[np.number]).to_numpy()
            self.assertFalse(np.isinf(numeric).any())

    def test_gamma_one_cross_validation_matches_h0(self):
        base_comparison, comparison, ranking = self._decorate(
            1.0,
            np.array([0.01, 0.02, 0.03, 0.04]),
        )
        h0_comparison = base_comparison.copy()
        h0_comparison["Alpha"] = 0.0
        h0_ranking = ranking[[
            "Beta_p", "Task_ID", "Primary_Server", "Actual_Reversal"
        ]].copy()
        h0_ranking["Alpha"] = 0.0
        validate_c3_against_h0(
            comparison,
            ranking,
            h0_comparison,
            h0_ranking,
        )


if __name__ == "__main__":
    unittest.main()
