"""Frozen policy inference must be logged without creating training transitions."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compare_ppo_greedy import (
    build_agent, evaluate_episode, frozen_snapshot, masked_evaluation, same_tree,
)
from compare_tore import cache_evaluation_profiles
from config.params import params
from core.main_loop import MainLoop


class PPOEvaluationLoggingTests(unittest.TestCase):
    def test_policy_and_fallback_are_distinct_without_changing_execution(self):
        profiles = {sid: dict(Site_ID=str(sid), Processing_Frequency=4e9,
                             Base_Failure_Rate=.01, Transmission_Rate=20.,
                             Latitude=0., Longitude=sid/100.) for sid in range(1, 9)}
        tasks = {tid: dict(Task_Size=5., Computation_Demand=2e9,
                           Reliability_Requirement=requirement, Deadline=3.)
                 for tid, requirement in ((1, .9), (2, 1.))}
        tasks = cache_evaluation_profiles(profiles, tasks, failure_rate_scale=1., ell_ratio=1.)
        agent = build_agent()
        snapshot = frozen_snapshot(agent)
        basis = np.array([.1, .1])
        # Logging must also work with an evaluation-only context, without training settings.
        context = SimpleNamespace(this_episode=1)
        records = []
        with patch.object(params, 'taskno', 2):
            metrics, _ = masked_evaluation('trained_masked_ppo', agent, profiles, tasks,
                basis, 1, 123, lambda task: MainLoop.get_task_outcome_info(context, task),
                task_records=records, arrival_rate=2., failure_rate_scale=1.)
            torch.manual_seed(123)
            _, original = evaluate_episode(
                lambda state, mask: agent.select_action(state, 0., action_mask=mask),
                2., 1, profiles, tasks, basis, failure_rate_scale=1.)
            sml_records = []
            sml_metrics, _ = masked_evaluation('safe_min_latency', None, profiles, tasks,
                basis, 1, 123, lambda task: MainLoop.get_task_outcome_info(context, task),
                task_records=sml_records, arrival_rate=2., failure_rate_scale=1.)
        self.assertEqual(metrics['ppo_decision_count'], 1)
        self.assertEqual(metrics['fallback_count'], 1)
        self.assertTrue(records[0]['ppo_decision'])
        self.assertEqual(records[0]['ppo_action'], records[0]['action_index'])
        self.assertEqual(records[0]['action_mask'], records[0]['safe_mask'])
        self.assertFalse(records[1]['ppo_decision'])
        self.assertIsNone(records[1]['ppo_action'])
        self.assertIsNone(records[1]['action_mask'])
        for row, task in zip(records, original):
            self.assertIsNone(row['ppo_transition_index'])
            self.assertIsNone(row['ppo_delta_t'])
            self.assertEqual(row['action_index'], task.action_index)
            self.assertEqual(row['task_completion_time'], task.task_completion_time)
            self.assertEqual(row['task_qos'], MainLoop.get_task_outcome_info(context, task)['task_qos'])
            self.assertEqual(row['pair_reliability'], task.pair_reliability)
        self.assertEqual(sml_metrics['ppo_decision_count'], 0)
        self.assertTrue(all(row['ppo_action'] is None for row in sml_records))
        same_tree(snapshot, frozen_snapshot(agent))
        self.assertFalse(agent.states)


if __name__ == '__main__':
    unittest.main()
