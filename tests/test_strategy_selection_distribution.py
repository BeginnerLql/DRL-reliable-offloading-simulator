import unittest

import pandas as pd

from io_utils.post_process_results import compute_distributions


class StrategySelectionDistributionTests(unittest.TestCase):
    @staticmethod
    def _servers_df():
        return pd.DataFrame({
            "Server_ID": [1, 2],
            "Server_Type": ["Edge", "Cloud"],
        })

    @staticmethod
    def _tasks_df():
        rows = []

        # Retry: most backups never start, but the strategy was still selected.
        for task_id in range(20):
            rows.append({
                "task_id": task_id,
                "episode": 1,
                "Primary": 1,
                "Backup": 1,
                "Z": 0,
                "Backup_Start": 1.0 if task_id < 3 else None,
            })

        # Recovery Block: most backups never start, but the strategy was selected.
        for task_id in range(20, 60):
            rows.append({
                "task_id": task_id,
                "episode": 1,
                "Primary": 1,
                "Backup": 2,
                "Z": 0,
                "Backup_Start": 1.0 if task_id < 27 else None,
            })

        # First Result: all rows have a started backup in this fixture.
        for task_id in range(60, 100):
            rows.append({
                "task_id": task_id,
                "episode": 1,
                "Primary": 1,
                "Backup": 2,
                "Z": 1,
                "Backup_Start": 1.0,
            })

        return pd.DataFrame(rows)

    def test_strategy_selection_uses_all_episode_decisions(self):
        _, strategy_df = compute_distributions(self._servers_df(), self._tasks_df())
        percentages = dict(zip(strategy_df["Strategy"], strategy_df["Percentage"]))

        self.assertAlmostEqual(percentages["Retry"], 20.0)
        self.assertAlmostEqual(percentages["Recovery Block"], 40.0)
        self.assertAlmostEqual(percentages["First Result"], 40.0)
        self.assertAlmostEqual(sum(percentages.values()), 100.0)

    def test_z0_primary_success_without_backup_start_is_counted(self):
        tasks_df = pd.DataFrame([
            {
                "task_id": task_id,
                "episode": 1,
                "Primary": 1,
                "Backup": 1,
                "Z": 0,
                "Backup_Start": None,
            }
            for task_id in range(5)
        ])

        _, strategy_df = compute_distributions(self._servers_df(), tasks_df)
        percentages = dict(zip(strategy_df["Strategy"], strategy_df["Percentage"]))

        self.assertAlmostEqual(percentages["Retry"], 100.0)
        self.assertAlmostEqual(percentages["Recovery Block"], 0.0)
        self.assertAlmostEqual(percentages["First Result"], 0.0)


if __name__ == "__main__":
    unittest.main()
