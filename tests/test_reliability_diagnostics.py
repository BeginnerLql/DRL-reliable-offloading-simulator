import unittest

import pandas as pd

from io_utils.save_parameters_and_logs import (
    build_pair_diagnostics,
    build_reliability_diagnostics,
)


class ReliabilityDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _rows():
        rows = []
        task_id = 1
        for requirement in (0.9, 0.99, 0.999, 0.9999):
            for index in range(4):
                execution = 0.995 if index < 3 else 0.9995
                margin = execution - requirement
                rows.append({
                    "episode": 1,
                    "task_id": task_id,
                    "Primary": 1 if index < 3 else 2,
                    "Backup": 2 if index < 3 else 3,
                    "Z": 1 if index == 3 else 0,
                    "Reliability_Requirement": requirement,
                    "Execution_Reliability": execution,
                    "Reliability_Satisfied": execution >= requirement,
                    "Final_status": "success" if execution >= requirement else "failure",
                    "Joint_Failure_Probability": 1.0 - execution,
                    "Task_Reward": 10.0 + index,
                    "Task_Delay": 1.0 + index,
                    "Reliability_Margin": margin,
                    "Reliability_Shortfall": max(-margin, 0.0),
                    "Reliability_Excess": max(margin, 0.0),
                })
                task_id += 1
        return pd.DataFrame(rows)

    def test_margin_shortfall_excess_are_preserved(self):
        frame = self._rows()
        row = frame.iloc[0]
        self.assertAlmostEqual(row["Reliability_Margin"], 0.095)
        row = frame.iloc[4]
        self.assertAlmostEqual(row["Reliability_Margin"], 0.005)
        row = frame.iloc[8]
        self.assertAlmostEqual(row["Reliability_Margin"], -0.004)
        self.assertAlmostEqual(row["Reliability_Shortfall"], 0.004)
        self.assertEqual(row["Reliability_Excess"], 0.0)

    def test_reliability_diagnostics_grouping_and_actual_reward(self):
        diagnostics = build_reliability_diagnostics(self._rows())
        self.assertEqual(len(diagnostics), 4)
        self.assertEqual(diagnostics["Task_Count"].tolist(), [4, 4, 4, 4])
        self.assertEqual(diagnostics.loc[0, "Success_Rate"], 1.0)
        self.assertEqual(diagnostics.loc[2, "Success_Rate"], 0.25)
        self.assertAlmostEqual(diagnostics.loc[0, "Mean_Task_Reward"], 11.5)
        self.assertEqual(diagnostics.loc[0, "Parallel_Mode_Count"], 1)
        self.assertEqual(diagnostics.loc[0, "Parallel_Mode_Rate"], 0.25)

    def test_pair_selection_share_is_within_requirement(self):
        frame = self._rows()
        diagnostics = build_pair_diagnostics(frame[frame["Reliability_Requirement"] == 0.9])
        self.assertEqual(len(diagnostics), 2)
        shares = sorted(diagnostics["Selection_Share_Within_Requirement"].tolist())
        self.assertEqual(shares, [0.25, 0.75])
        self.assertEqual(sorted(diagnostics["Selection_Count"].tolist()), [1, 3])

    def test_final_success_source_is_reliability_threshold(self):
        frame = self._rows()
        frame["Primary_Status"] = "success"
        frame["Backup_Status"] = "success"
        diagnostics = build_reliability_diagnostics(frame)
        self.assertEqual(diagnostics.loc[2, "Success_Count"], 1)
        self.assertEqual(diagnostics.loc[2, "Failure_Count"], 3)


if __name__ == "__main__":
    unittest.main()
