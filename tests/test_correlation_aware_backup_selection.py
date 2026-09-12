import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.analyze_correlation_aware_backup_selection import (
    MATCHED_BASELINE_TYPE,
    build_primary_candidates,
    compare_backup_selections,
    run_diagnostic,
    summarize_backup_selections,
    validate_pair_failure_data,
)


class CorrelationAwareBackupSelectionTests(unittest.TestCase):
    @staticmethod
    def _pair_frame(
        betas=(0.0, 0.8),
        server_count=8,
        task_id=101,
        zero_marginals=False,
    ):
        rows = []
        marginals = {
            server_id: 0.0 if zero_marginals else server_id / 100.0
            for server_id in range(1, server_count + 1)
        }
        for beta_p in betas:
            for server_j, server_k in itertools.combinations(
                range(1, server_count + 1),
                2,
            ):
                rho = 0.8 if (server_j, server_k) == (1, 2) else 0.05
                independent_joint = (
                    marginals[server_j] * marginals[server_k]
                )
                spatial_joint = independent_joint * (
                    1.0 if beta_p == 0.0 else 1.0 + rho
                )
                rows.append({
                    "Beta_p": beta_p,
                    "Independent_Baseline_Type": MATCHED_BASELINE_TYPE,
                    "Task_ID": task_id,
                    "Server_J": server_j,
                    "Server_K": server_k,
                    "Distance_km": float(server_k - server_j),
                    "Rho_phy": rho,
                    "Marginal_Failure_J_Spatial": marginals[server_j],
                    "Marginal_Failure_K_Spatial": marginals[server_k],
                    "Joint_Failure_Spatial": spatial_joint,
                    "Joint_Failure_Independent": independent_joint,
                })
        return pd.DataFrame(rows)

    def test_candidates_include_both_pair_orientations_and_seven_backups(self):
        candidates = build_primary_candidates(self._pair_frame())
        fixed_primary = candidates[
            (candidates["Beta_p"] == 0.8)
            & (candidates["Task_ID"] == 101)
            & (candidates["Primary_Server"] == 4)
        ]

        self.assertEqual(len(fixed_primary), 7)
        self.assertEqual(set(fixed_primary["Backup_Server"]), {1, 2, 3, 5, 6, 7, 8})
        self.assertFalse(
            (fixed_primary["Primary_Server"] == fixed_primary["Backup_Server"]).any()
        )

        primary_was_server_k = fixed_primary[
            fixed_primary["Backup_Server"] == 2
        ].iloc[0]
        self.assertEqual(primary_was_server_k["Primary_Marginal_Failure"], 0.04)
        self.assertEqual(primary_was_server_k["Backup_Marginal_Failure"], 0.02)

        primary_was_server_j = fixed_primary[
            fixed_primary["Backup_Server"] == 7
        ].iloc[0]
        self.assertEqual(primary_was_server_j["Primary_Marginal_Failure"], 0.04)
        self.assertEqual(primary_was_server_j["Backup_Marginal_Failure"], 0.07)

    def test_selection_criteria_can_differ_and_reduce_actual_risk(self):
        comparison = compare_backup_selections(self._pair_frame())
        case = comparison[
            (comparison["Beta_p"] == 0.8)
            & (comparison["Task_ID"] == 101)
            & (comparison["Primary_Server"] == 1)
        ].iloc[0]

        self.assertEqual(case["Independent_Selected_Backup"], 2)
        self.assertEqual(case["Correlation_Aware_Selected_Backup"], 3)
        self.assertTrue(case["Selection_Changed"])
        self.assertGreater(case["Absolute_Risk_Reduction"], 0.0)
        self.assertGreater(case["Relative_Risk_Reduction"], 0.0)
        self.assertEqual(
            case["Independent_Assumed_Joint_Risk"],
            case["Primary_Marginal_Failure"]
            * case["Independent_Backup_Marginal_Failure"],
        )
        self.assertEqual(
            case["Independent_Assumption_Underestimation"],
            case["Independent_Actual_Joint_Risk"]
            - case["Independent_Assumed_Joint_Risk"],
        )

    def test_beta_zero_control_and_output_row_counts(self):
        frame = self._pair_frame()
        comparison = compare_backup_selections(frame)
        summary = summarize_backup_selections(comparison)
        beta_zero_cases = comparison[comparison["Beta_p"] == 0.0]
        beta_zero_summary = summary[summary["Beta_p"] == 0.0].iloc[0]

        self.assertEqual(len(beta_zero_cases), 8)
        self.assertFalse(beta_zero_cases["Selection_Changed"].any())
        self.assertTrue(
            (beta_zero_cases["Absolute_Risk_Reduction"] == 0.0).all()
        )
        self.assertEqual(beta_zero_summary["Selection_Change_Count"], 0)
        self.assertEqual(beta_zero_summary["Selection_Change_Rate"], 0.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "task_pair_joint_failure.csv"
            output_dir = root / "diagnostics"
            frame.to_csv(input_path, index=False)

            written_comparison, written_summary = run_diagnostic(
                input_path=input_path,
                output_dir=output_dir,
            )

            self.assertEqual(len(written_comparison), 16)
            self.assertEqual(len(written_summary), 2)
            self.assertTrue((output_dir / "backup_selection_comparison.csv").exists())
            self.assertTrue((output_dir / "backup_selection_summary.csv").exists())

    def test_ties_choose_smallest_backup_server_id(self):
        frame = self._pair_frame(
            betas=(0.8,),
            server_count=3,
        )
        for column in (
            "Marginal_Failure_J_Spatial",
            "Marginal_Failure_K_Spatial",
        ):
            frame[column] = 0.02
        frame["Joint_Failure_Independent"] = 0.0004
        frame["Joint_Failure_Spatial"] = 0.0004

        comparison = compare_backup_selections(frame)
        case = comparison[comparison["Primary_Server"] == 3].iloc[0]
        self.assertEqual(case["Independent_Selected_Backup"], 1)
        self.assertEqual(case["Correlation_Aware_Selected_Backup"], 1)

    def test_same_server_pair_is_rejected(self):
        frame = self._pair_frame()
        frame.loc[0, "Server_K"] = frame.loc[0, "Server_J"]
        with self.assertRaisesRegex(ValueError, "distinct servers"):
            validate_pair_failure_data(frame)

    def test_malformed_baseline_type_is_rejected(self):
        frame = self._pair_frame()
        frame.loc[0, "Independent_Baseline_Type"] = "separate_monte_carlo"
        with self.assertRaisesRegex(ValueError, "matched_marginal_product"):
            validate_pair_failure_data(frame)

    def test_zero_denominators_produce_nan_instead_of_infinity(self):
        comparison = compare_backup_selections(
            self._pair_frame(
                betas=(0.8,),
                server_count=3,
                zero_marginals=True,
            )
        )

        self.assertTrue(comparison["Relative_Risk_Reduction"].isna().all())
        self.assertTrue(
            comparison[
                "Independent_Assumption_Underestimation_Ratio"
            ].isna().all()
        )
        numeric = comparison.select_dtypes(include=[np.number]).to_numpy()
        self.assertFalse(np.isinf(numeric).any())


if __name__ == "__main__":
    unittest.main()
