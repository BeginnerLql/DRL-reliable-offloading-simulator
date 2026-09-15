import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config.params import params
from core.main_loop import MainLoop
from io_utils.save_parameters_and_logs import save_params_and_logs


class _MockPPO:
    def __init__(self):
        self.assigned_rewards = []

    def select_action(self, state, epsilon):
        return 0

    def store_transition(self, *args, **kwargs):
        return None

    def assign_task_reward(self, task_id, reward):
        self.assigned_rewards.append((task_id, reward))

    def train_step(self):
        return None


class ReplicaCompletionLogPersistenceTests(unittest.TestCase):
    def test_replica_log_accumulates_across_episodes_and_matches_pairs(self):
        model = _MockPPO()
        with patch.multiple(
            params,
            model_summary="ppo",
            TASK_ARRIVAL_RATE=20.0,
            SPATIAL_RISK_ENABLED=False,
        ):
            loop = MainLoop(
                model=model,
                total_episodes=2,
                maxtaskno=2,
                num_states=params.num_states,
                num_actions=params.num_actions,
            )
            loop.EP()

        self.assertEqual(len(loop.replica_completion_log), 8)
        log_df = pd.DataFrame(loop.replica_completion_log)
        self.assertEqual(log_df["episode"].value_counts().to_dict(), {1: 4, 2: 4})
        self.assertEqual(loop.env_state.num_resolved_tasks, 2)
        self.assertEqual(loop.env_state.num_completed_replicas, 4)
        self.assertEqual(len(model.assigned_rewards), 4)

        grouped = log_df.groupby(["episode", "task_id"], sort=True)
        self.assertTrue((grouped.size() == 2).all())
        for (episode, task_id), group in grouped:
            self.assertEqual(group["action_index"].nunique(dropna=False), 1)
            self.assertEqual(group["server_id"].nunique(), 2)
            self.assertEqual(group["replica_label"].nunique(), 2)
            action_index = int(group["action_index"].iloc[0])
            expected_pair = set(loop.action_pairs[action_index])
            self.assertEqual(set(group["server_id"]), expected_pair)
            self.assertTrue((pd.to_numeric(group["finish_time"]) >= 0.0).all())
            self.assertTrue((pd.to_numeric(group["service_time"]) > 0.0).all())
            self.assertTrue(
                (
                    pd.to_numeric(group["finish_time"])
                    >= pd.to_numeric(group["service_time"])
                ).all()
            )

        self.assertEqual(
            log_df[["episode", "task_id", "replica_label"]]
            .duplicated()
            .sum(),
            0,
        )

    def test_save_writes_sorted_replica_completions_sheet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            pd.DataFrame({"Server_ID": [1]}).to_excel(
                root / "data" / "server_info.xlsx", index=False
            )
            pd.DataFrame({"Task_ID": [1]}).to_excel(
                root / "data" / "task_parameters.xlsx", index=False
            )
            replica_log = [
                {
                    "episode": 2,
                    "task_id": 1,
                    "action_index": 26,
                    "server_id": 8,
                    "replica_label": "backup",
                    "finish_time": 14.7,
                    "service_time": 2.0,
                },
                {
                    "episode": 1,
                    "task_id": 1,
                    "action_index": 26,
                    "server_id": 6,
                    "replica_label": "primary",
                    "finish_time": 12.3,
                    "service_time": 1.5,
                },
            ]
            with patch("io_utils.save_parameters_and_logs.DATA_DIR", str(root / "data")), \
                 patch("io_utils.save_parameters_and_logs.RESULTS_DIR", str(root / "results")), \
                 patch.object(params, "model_summary", "ppo"):
                save_params_and_logs(
                    params,
                    [],
                    [],
                    replica_completion_log=replica_log,
                )

            output = root / "results" / "fixed_rate_results" / "ppo_results.xlsx"
            self.assertTrue(output.exists())
            workbook = pd.ExcelFile(output)
            self.assertIn("ReplicaCompletions", workbook.sheet_names)
            replica_df = pd.read_excel(output, sheet_name="ReplicaCompletions")
            self.assertEqual(
                list(replica_df.columns),
                [
                    "Episode",
                    "Task_ID",
                    "Action_Index",
                    "Server_ID",
                    "Replica_Label",
                    "Finish_Time",
                    "Service_Time",
                ],
            )
            self.assertEqual(replica_df[["Episode", "Task_ID"]].values.tolist(), [[1, 1], [2, 1]])
            self.assertEqual(replica_df.loc[0, "Action_Index"], 26)

    def test_none_replica_log_writes_empty_fixed_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            pd.DataFrame({"Server_ID": [1]}).to_excel(
                root / "data" / "server_info.xlsx", index=False
            )
            pd.DataFrame({"Task_ID": [1]}).to_excel(
                root / "data" / "task_parameters.xlsx", index=False
            )
            with patch("io_utils.save_parameters_and_logs.DATA_DIR", str(root / "data")), \
                 patch("io_utils.save_parameters_and_logs.RESULTS_DIR", str(root / "results")), \
                 patch.object(params, "model_summary", "ppo"):
                save_params_and_logs(params, [], [])
            output = root / "results" / "fixed_rate_results" / "ppo_results.xlsx"
            replica_df = pd.read_excel(output, sheet_name="ReplicaCompletions")
            self.assertEqual(
                list(replica_df.columns),
                [
                    "Episode",
                    "Task_ID",
                    "Action_Index",
                    "Server_ID",
                    "Replica_Label",
                    "Finish_Time",
                    "Service_Time",
                ],
            )
            self.assertTrue(replica_df.empty)


if __name__ == "__main__":
    unittest.main()
