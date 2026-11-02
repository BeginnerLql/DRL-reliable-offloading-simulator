import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agents.corr_delay_greedy_agent import CorrDelayGreedyAgent
from config.params import params
from config.paths import EVALUATION_TASK_PATH
from core.main_loop import MainLoop


class CorrDelayGreedyTests(unittest.TestCase):
    def test_upload_overlaps_cpu_backlog_and_mask_excludes_faster_pairs(self):
        task = SimpleNamespace(env=SimpleNamespace(now=0), task_size=10,
                               computation_demand=1)
        servers = {
            1: SimpleNamespace(transmission_rate=1, processing_frequency=1),
            2: SimpleNamespace(transmission_rate=10, processing_frequency=1),
            3: SimpleNamespace(transmission_rate=10 / 9, processing_frequency=2),
        }
        state = SimpleNamespace(
            servers=servers, get_server_by_id=servers.__getitem__,
            get_server_backlog_components=lambda server_id, now:
                (3, 5) if server_id == 2 else (0, 0),
        )
        agent = CorrDelayGreedyAgent()
        pairs = [(1, 2), (1, 3), (2, 3)]
        # Predicted times: 11, 9, 9.5. Adding upload to backlog would
        # incorrectly prefer (1, 3) over (1, 2).
        self.assertEqual(agent.select_action(task, state, pairs,
                                             action_mask=[True, True, False]), 0)
        self.assertEqual(agent.select_action(task, state, pairs,
                                             action_mask=[False, True, False]), 1)

    def test_ties_prefer_second_completion_then_server_ids(self):
        task = SimpleNamespace(env=SimpleNamespace(now=0), task_size=1,
                               computation_demand=1)
        servers = {index: SimpleNamespace(transmission_rate=1, processing_frequency=1)
                   for index in range(1, 5)}
        backlog = {1: 0, 2: 4, 3: 2, 4: 2}
        state = SimpleNamespace(
            servers=servers, get_server_by_id=servers.__getitem__,
            get_server_backlog_components=lambda server_id, now: (backlog[server_id], 0),
        )
        # All pairs finish first at t=2. Servers 3 and 4 both finish at
        # t=3, so (1, 3) wins even when it appears after (1, 4).
        self.assertEqual(CorrDelayGreedyAgent().select_action(
            task, state, [(1, 4), (1, 2), (1, 3)], action_mask=[True] * 3,
        ), 2)

    def test_safe_set_and_maximum_reliability_fallback_with_ppo_mask_off(self):
        with patch.object(params, "model_summary", "Corr-delayGreedy"), \
                patch.object(params, "SPATIAL_RISK_ENABLED", True), \
                patch.object(params, "RELIABILITY_ACTION_MASK_ENABLED", False):
            loop = MainLoop(CorrDelayGreedyAgent(), 1, 1,
                            params.num_states, params.num_actions)
            reliabilities = [0.90, 0.92, 0.92] + [0.80] * (params.num_actions - 3)
            loop.extract_parameters_from_index = lambda index: (index, None)
            loop._get_pair_reliability = lambda task, index, unused: SimpleNamespace(
                correlated=reliabilities[index],
            )
            task = SimpleNamespace(reliability_requirement=0.99)
            mask = loop._get_reliability_action_mask(task)
            np.testing.assert_array_equal(np.flatnonzero(mask), [1, 2])
            self.assertEqual(task.feasible_action_count, 0)
            self.assertTrue(task.mask_fallback_used)

            task.reliability_requirement = 0.91
            np.testing.assert_array_equal(
                np.flatnonzero(loop._get_reliability_action_mask(task)), [1, 2],
            )
            self.assertEqual(task.feasible_action_count, 2)
            self.assertFalse(task.mask_fallback_used)

            # Existing PPO behavior still honors its disabled mask switch.
            loop.model_name = "ppo"
            task.reliability_requirement = 0.99
            self.assertIsNone(loop._get_reliability_action_mask(task))
            self.assertFalse(task.mask_fallback_used)

    def test_two_episode_simulation_records_outcomes_without_training(self):
        with patch.object(params, "model_summary", "Corr-delayGreedy"), \
                patch.object(params, "SPATIAL_RISK_ENABLED", True), \
                patch.object(params, "RELIABILITY_ACTION_MASK_ENABLED", False), \
                redirect_stdout(io.StringIO()):
            from Project_main import build_model

            loop = MainLoop(build_model(), 2, 200, params.num_states, params.num_actions,
                            task_parameters_path=EVALUATION_TASK_PATH)
            loop.EP()
            self.assertEqual(len(loop.task_results), 400)
            self.assertEqual(len(loop.log_data), 2)
            self.assertEqual(loop.tempbuffer, {})
            self.assertEqual(loop.pendingList, [])
            self.assertTrue(any(row["mask_fallback_used"] for row in loop.task_results))
            for row in loop.task_results:
                self.assertEqual(row["mask_fallback_used"], row["feasible_action_count"] == 0)
                self.assertEqual(row["requirement_satisfied_correlated"],
                                 row["feasible_action_count"] > 0)
                self.assertEqual(row["task_completion_time"],
                                 min(row["replica_A_finish_time"], row["replica_B_finish_time"]))
                self.assertGreaterEqual(row["task_delay"], 0)
            for episode, reward, delay, rsr, _, fallback in loop.log_data:
                rows = [row for row in loop.task_results if row["episode"] == episode]
                self.assertAlmostEqual(reward, -sum(row["task_delay"] for row in rows))
                self.assertAlmostEqual(delay, -reward / 200)
                self.assertAlmostEqual(rsr + fallback, 1)


if __name__ == "__main__":
    unittest.main()
