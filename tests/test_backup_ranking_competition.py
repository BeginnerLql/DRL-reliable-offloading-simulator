import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.analyze_backup_ranking_competition import (
    cross_validate_best_backups,
    rank_backup_candidates,
    run_diagnostic,
    summarize_ranking_competition,
)
from tools.analyze_correlation_aware_backup_selection import (
    MATCHED_BASELINE_TYPE,
    compare_backup_selections,
)


class BackupRankingCompetitionTests(unittest.TestCase):
    @staticmethod
    def _pair_frame(
        marginals=None,
        amplifications=None,
        beta_p=0.8,
    ):
        if marginals is None:
            marginals = {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04}
        if amplifications is None:
            amplifications = {(1, 2): 2.0}

        rows = []
        for server_j, server_k in itertools.combinations(sorted(marginals), 2):
            independent_joint = marginals[server_j] * marginals[server_k]
            amplification = amplifications.get((server_j, server_k), 1.0)
            rows.append({
                "Beta_p": beta_p,
                "Independent_Baseline_Type": MATCHED_BASELINE_TYPE,
                "Task_ID": 101,
                "Server_J": server_j,
                "Server_K": server_k,
                "Distance_km": float(server_k - server_j),
                "Rho_phy": 0.1 * (server_k - server_j),
                "Marginal_Failure_J_Spatial": marginals[server_j],
                "Marginal_Failure_K_Spatial": marginals[server_k],
                "Joint_Failure_Spatial": independent_joint * amplification,
                "Joint_Failure_Independent": independent_joint,
            })
        return pd.DataFrame(rows)

    def test_ranking_selects_top_three_and_handles_both_pair_orientations(self):
        ranking = rank_backup_candidates(self._pair_frame())

        primary_is_server_j = ranking[ranking["Primary_Server"] == 1].iloc[0]
        self.assertEqual(
            [
                primary_is_server_j["Best_Backup"],
                primary_is_server_j["Second_Backup"],
                primary_is_server_j["Third_Backup"],
            ],
            [2, 3, 4],
        )
        self.assertEqual(primary_is_server_j["Best_Marginal"], 0.02)
        self.assertEqual(primary_is_server_j["Second_Marginal"], 0.03)
        self.assertEqual(primary_is_server_j["Third_Marginal"], 0.04)

        primary_is_server_k = ranking[ranking["Primary_Server"] == 2].iloc[0]
        self.assertEqual(primary_is_server_k["Best_Backup"], 1)
        self.assertEqual(primary_is_server_k["Best_Marginal"], 0.01)

        for column in ("Best_Backup", "Second_Backup", "Third_Backup"):
            self.assertFalse((ranking[column] == ranking["Primary_Server"]).any())

    def test_gap_amplification_and_reversal_formulas(self):
        ranking = rank_backup_candidates(self._pair_frame())
        case = ranking[ranking["Primary_Server"] == 1].iloc[0]

        self.assertAlmostEqual(case["Marginal_Gap_1_2"], 0.01)
        self.assertAlmostEqual(case["Marginal_Gap_Ratio_1_2"], 1.5)
        self.assertAlmostEqual(case["Best_Amplification"], 2.0)
        self.assertAlmostEqual(case["Second_Amplification"], 1.0)
        self.assertAlmostEqual(
            case["Required_Amplification_Ratio_For_Reversal"],
            1.5,
        )
        self.assertAlmostEqual(case["Observed_Amplification_Ratio"], 2.0)
        self.assertAlmostEqual(case["Reversal_Margin"], 2.0 / 1.5)
        self.assertGreater(case["Reversal_Margin"], 1.0)
        self.assertLess(case["Joint_Risk_Ratio_1_2"], 1.0)
        self.assertLess(case["Second_Joint_Risk"], case["Best_Joint_Risk"])

    def test_ties_are_resolved_by_smallest_server_id(self):
        frame = self._pair_frame(
            marginals={1: 0.01, 2: 0.01, 3: 0.02, 4: 0.04},
            amplifications={},
        )
        ranking = rank_backup_candidates(frame)
        case = ranking[ranking["Primary_Server"] == 4].iloc[0]

        self.assertEqual(case["Best_Backup"], 1)
        self.assertEqual(case["Second_Backup"], 2)
        self.assertEqual(case["Third_Backup"], 3)

    def test_malformed_baseline_is_rejected(self):
        frame = self._pair_frame()
        frame.loc[0, "Independent_Baseline_Type"] = "separate_monte_carlo"
        with self.assertRaisesRegex(ValueError, "matched_marginal_product"):
            rank_backup_candidates(frame)

    def test_same_server_pair_is_rejected(self):
        frame = self._pair_frame()
        frame.loc[0, "Server_K"] = frame.loc[0, "Server_J"]
        with self.assertRaisesRegex(ValueError, "distinct servers"):
            rank_backup_candidates(frame)

    def test_zero_denominators_produce_nan_without_infinity(self):
        frame = self._pair_frame(
            marginals={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            amplifications={},
        )
        ranking = rank_backup_candidates(frame)

        ratio_columns = [
            "Marginal_Gap_Ratio_1_2",
            "Best_Amplification",
            "Second_Amplification",
            "Observed_Amplification_Ratio",
            "Reversal_Margin",
            "Joint_Risk_Ratio_1_2",
        ]
        self.assertTrue(ranking[ratio_columns].isna().all().all())
        self.assertFalse(
            np.isinf(ranking.select_dtypes(include=[np.number]).to_numpy()).any()
        )
        self.assertFalse(ranking["Near_Tie_5pct"].any())
        self.assertFalse(ranking["Near_Tie_10pct"].any())

    def test_summary_cross_validation_and_output_files(self):
        frame = self._pair_frame()
        ranking = rank_backup_candidates(frame)
        summary = summarize_ranking_competition(ranking)

        self.assertEqual(len(ranking), 4)
        self.assertEqual(len(summary), 1)
        self.assertGreater(summary.iloc[0]["Max_Reversal_Margin"], 1.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pair_path = root / "task_pair_joint_failure.csv"
            comparison_path = root / "backup_selection_comparison.csv"
            output_dir = root / "diagnostics"
            frame.to_csv(pair_path, index=False)
            compare_backup_selections(frame).to_csv(comparison_path, index=False)

            cross_validate_best_backups(ranking, comparison_path)
            written_ranking, written_summary = run_diagnostic(
                pair_input=pair_path,
                selection_input=comparison_path,
                output_dir=output_dir,
            )

            self.assertEqual(len(written_ranking), 4)
            self.assertEqual(len(written_summary), 1)
            self.assertTrue(
                (output_dir / "backup_ranking_competition.csv").exists()
            )
            self.assertTrue(
                (output_dir / "backup_ranking_competition_summary.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
