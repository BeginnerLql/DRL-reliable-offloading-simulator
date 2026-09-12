import itertools
import unittest

import numpy as np
import pandas as pd

from tools.analyze_backup_ranking_competition import rank_backup_candidates
from tools.analyze_correlation_aware_backup_selection import (
    MATCHED_BASELINE_TYPE,
    compare_backup_selections,
)
from tools.analyze_failure_rate_heterogeneity import (
    build_failure_rate_scenarios,
    contract_failure_rates,
    decorate_ranking_competition,
    decorate_selection_comparison,
    failure_rate_statistics,
    summarize_sensitivity,
    validate_beta_zero_controls,
    validate_h3_against_step3a,
)


class FailureRateHeterogeneityTests(unittest.TestCase):
    ORIGINAL_RATES = np.array([0.01, 0.03, 0.04, 0.02])

    @staticmethod
    def _pair_frame(rates, beta_p=0.8, amplifications=None):
        rates = np.asarray(rates, dtype=float)
        if amplifications is None:
            amplifications = {(1, 4): 2.0}
        rows = []
        for server_j, server_k in itertools.combinations(range(1, 5), 2):
            marginal_j = rates[server_j - 1]
            marginal_k = rates[server_k - 1]
            independent_joint = marginal_j * marginal_k
            amplification = amplifications.get((server_j, server_k), 1.0)
            rows.append({
                "Beta_p": beta_p,
                "Independent_Baseline_Type": MATCHED_BASELINE_TYPE,
                "Task_ID": 101,
                "Server_J": server_j,
                "Server_K": server_k,
                "Distance_km": float(server_k - server_j),
                "Rho_phy": 0.1 * (server_k - server_j),
                "Marginal_Failure_J_Spatial": marginal_j,
                "Marginal_Failure_K_Spatial": marginal_k,
                "Joint_Failure_Spatial": independent_joint * amplification,
                "Joint_Failure_Independent": independent_joint,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _decorate(alpha, rates, beta_p=0.8, amplifications=None):
        pair_frame = FailureRateHeterogeneityTests._pair_frame(
            rates,
            beta_p=beta_p,
            amplifications=amplifications,
        )
        base_comparison = compare_backup_selections(pair_frame)
        base_ranking = rank_backup_candidates(pair_frame)
        mean_rate, std_rate, cv_rate = failure_rate_statistics(rates)
        rate_by_server = {
            server_id: float(rate)
            for server_id, rate in enumerate(rates, start=1)
        }
        comparison = decorate_selection_comparison(
            base_comparison,
            alpha,
            "synthetic",
            rate_by_server,
            mean_rate,
            std_rate,
            cv_rate,
        )
        ranking = decorate_ranking_competition(
            base_ranking,
            base_comparison,
            alpha,
            "synthetic",
        )
        return base_comparison, comparison, ranking

    def test_alpha_endpoints_preserve_mean(self):
        original = self.ORIGINAL_RATES
        mean_rate = original.mean()
        homogeneous = contract_failure_rates(original, 0.0)
        restored = contract_failure_rates(original, 1.0)

        np.testing.assert_array_equal(
            homogeneous,
            np.full_like(original, mean_rate),
        )
        np.testing.assert_array_equal(restored, original)
        for alpha in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
            scenario_rates = contract_failure_rates(original, alpha)
            self.assertAlmostEqual(scenario_rates.mean(), mean_rate, places=15)

    def test_std_and_cv_scale_linearly_with_alpha(self):
        original_mean, original_std, original_cv = failure_rate_statistics(
            self.ORIGINAL_RATES
        )
        for alpha in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
            mean_rate, std_rate, cv_rate = failure_rate_statistics(
                contract_failure_rates(self.ORIGINAL_RATES, alpha)
            )
            self.assertAlmostEqual(mean_rate, original_mean, places=15)
            self.assertAlmostEqual(std_rate, alpha * original_std, places=15)
            self.assertAlmostEqual(cv_rate, alpha * original_cv, places=15)

        table = build_failure_rate_scenarios(
            [1, 2, 3, 4],
            self.ORIGINAL_RATES,
        )
        self.assertEqual(len(table), 16)
        self.assertEqual(
            set(table["Heterogeneity_Scenario"]),
            {"H0", "H1", "H2", "H3"},
        )

    def test_invalid_alpha_and_failure_rates_are_rejected(self):
        for alpha in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    contract_failure_rates(self.ORIGINAL_RATES, alpha)

        invalid_rates = (
            [0.01, -0.02],
            [0.01, np.nan],
            [0.01, np.inf],
        )
        for rates in invalid_rates:
            with self.subTest(rates=rates):
                with self.assertRaisesRegex(ValueError, "failure rates"):
                    contract_failure_rates(rates, 0.5)

    def test_low_heterogeneity_can_reverse_while_high_does_not(self):
        low_rates = contract_failure_rates(self.ORIGINAL_RATES, 0.0)
        high_rates = contract_failure_rates(self.ORIGINAL_RATES, 1.0)

        low_comparison = compare_backup_selections(
            self._pair_frame(low_rates)
        )
        high_comparison = compare_backup_selections(
            self._pair_frame(high_rates)
        )
        low_case = low_comparison[
            low_comparison["Primary_Server"] == 4
        ].iloc[0]
        high_case = high_comparison[
            high_comparison["Primary_Server"] == 4
        ].iloc[0]

        self.assertEqual(low_case["Independent_Selected_Backup"], 1)
        self.assertEqual(low_case["Correlation_Aware_Selected_Backup"], 2)
        self.assertTrue(low_case["Selection_Changed"])
        self.assertGreater(low_case["Relative_Risk_Reduction"], 0.0)

        self.assertEqual(high_case["Independent_Selected_Backup"], 1)
        self.assertEqual(high_case["Correlation_Aware_Selected_Backup"], 1)
        self.assertFalse(high_case["Selection_Changed"])

        low_ranking = rank_backup_candidates(self._pair_frame(low_rates))
        high_ranking = rank_backup_candidates(self._pair_frame(high_rates))
        self.assertGreater(
            low_ranking.loc[
                low_ranking["Primary_Server"] == 4,
                "Reversal_Margin",
            ].iloc[0],
            1.0,
        )
        self.assertLess(
            high_ranking.loc[
                high_ranking["Primary_Server"] == 4,
                "Reversal_Margin",
            ].iloc[0],
            1.0,
        )
        for column in (
            "Independent_Selected_Backup",
            "Correlation_Aware_Selected_Backup",
        ):
            self.assertFalse(
                (low_comparison[column] == low_comparison["Primary_Server"]).any()
            )

    def test_beta_zero_controls_hold_for_every_alpha(self):
        comparisons = []
        rankings = []
        for alpha in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
            rates = contract_failure_rates(self.ORIGINAL_RATES, alpha)
            _, comparison, ranking = self._decorate(
                alpha,
                rates,
                beta_p=0.0,
                amplifications={},
            )
            comparisons.append(comparison)
            rankings.append(ranking)

        comparison = pd.concat(comparisons, ignore_index=True)
        ranking = pd.concat(rankings, ignore_index=True)
        validate_beta_zero_controls(comparison, ranking)
        self.assertFalse(comparison["Selection_Changed"].any())
        self.assertTrue((comparison["Absolute_Risk_Reduction"] == 0.0).all())
        self.assertFalse(ranking["Actual_Reversal"].any())

    def test_zero_denominators_do_not_produce_infinity(self):
        rates = np.zeros(4)
        _, comparison, ranking = self._decorate(
            0.0,
            rates,
            beta_p=0.8,
            amplifications={},
        )
        summary = summarize_sensitivity(comparison, ranking)

        for frame in (comparison, ranking, summary):
            numeric = frame.select_dtypes(include=[np.number]).to_numpy()
            self.assertFalse(np.isinf(numeric).any())

    def test_h3_cross_validation_matches_step3a(self):
        rates = contract_failure_rates(self.ORIGINAL_RATES, 1.0)
        base_comparison, comparison, ranking = self._decorate(
            1.0,
            rates,
        )

        validate_h3_against_step3a(comparison, base_comparison)
        summary = summarize_sensitivity(comparison, ranking)
        self.assertEqual(summary.iloc[0]["Case_Count"], 4)


if __name__ == "__main__":
    unittest.main()
