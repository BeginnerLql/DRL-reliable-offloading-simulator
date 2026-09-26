import copy
import inspect
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from diagnostics.q_credit_audit import (
    build_reward_event_attribution,
    calculate_gae_returns,
    decompose_task_transition_rewards,
    reconstruct_bellman_target,
    supervised_fixed_target_fit,
)
from diagnostics.run_q_credit_audit import _state_dict_sha256
from agents.action_conditioned_q_critic import smdp_discount, masked_policy_expectation

ROOT = Path(__file__).resolve().parents[1]


def assignments_fixture():
    return pd.DataFrame([
        {"episode": 1, "task_id": 1, "Primary_Start": 0.0, "Primary_End": 3.0, "Backup_End": 4.0,
         "Task_Reward": 8.0, "Base_Reward": 10.0, "Reliability_Penalty": 2.0,
         "action_index": 4, "server_j": 1, "server_k": 2},
        {"episode": 1, "task_id": 2, "Primary_Start": 1.0, "Primary_End": 2.0, "Backup_End": 3.0,
         "Task_Reward": 7.0, "Base_Reward": 8.0, "Reliability_Penalty": 1.0,
         "action_index": 5, "server_j": 1, "server_k": 3},
        {"episode": 1, "task_id": 3, "Primary_Start": 4.0, "Primary_End": 5.0, "Backup_End": 6.0,
         "Task_Reward": 6.0, "Base_Reward": 7.0, "Reliability_Penalty": 1.0,
         "action_index": 6, "server_j": 2, "server_k": 3},
    ])


class QCreditAuditTests(unittest.TestCase):
    def test_01_reward_event_task_id_and_components_are_exact(self):
        frame = assignments_fixture()
        events = build_reward_event_attribution(frame, assignment_time_map={(1, 1): 4.0})
        task1 = events[events.source_task_id == 1]
        self.assertEqual(set(task1.reward_component_type), {"base_reward", "reliability_penalty"})
        self.assertAlmostEqual(task1.reward_component_value.sum(), 8.0)
        self.assertEqual(task1.reward_assignment_time.unique().tolist(), [4.0])

    def test_02_attribution_does_not_change_reward_values(self):
        frame = assignments_fixture()
        before = frame.copy(deep=True)
        build_reward_event_attribution(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_03_attribution_does_not_change_event_timing_metadata(self):
        frame = assignments_fixture()
        original = frame[["episode", "task_id", "Primary_Start", "Primary_End", "Backup_End"]].copy(deep=True)
        events = build_reward_event_attribution(frame)
        pd.testing.assert_frame_equal(frame[["episode", "task_id", "Primary_Start", "Primary_End", "Backup_End"]], original)
        first = events[events.source_task_id == 1]
        self.assertTrue((first.reward_event_time == 3.0).all())
        self.assertTrue((first.source_task_decision_time == 0.0).all())

    def test_04_action_selection_is_not_mutated(self):
        frame = assignments_fixture()
        expected = frame[["episode", "task_id", "action_index"]].copy()
        build_reward_event_attribution(frame)
        pd.testing.assert_frame_equal(frame[["episode", "task_id", "action_index"]], expected)

    def test_05_decomposed_reward_sums_to_original_transition_reward(self):
        frame = assignments_fixture()
        events = build_reward_event_attribution(frame)
        transitions = pd.DataFrame({"episode": [1, 1, 1], "task_id": [1, 2, 3], "reward_term": [8.0, 7.0, 6.0]})
        result = decompose_task_transition_rewards(transitions, events)
        np.testing.assert_allclose(result.r_current_task, transitions.reward_term)
        np.testing.assert_allclose(result.r_previous_tasks, 0.0)
        np.testing.assert_allclose(result.r_current_task + result.r_previous_tasks + result.r_other, transitions.reward_term)

    def test_06_terminal_bellman_target_has_no_bootstrap(self):
        self.assertEqual(float(reconstruct_bellman_target(4.5, .9, 100., True)), 4.5)

    def test_07_nonterminal_bellman_target_reconstructs(self):
        self.assertAlmostEqual(float(reconstruct_bellman_target(2., .81, 10., False)), 10.1)

    def test_08_smdp_discount_matches_gamma_power_delta(self):
        got = smdp_discount(.9, [0., .25, 2.])
        torch.testing.assert_close(got, torch.tensor([1., .9 ** .25, .81]))

    def test_09_frozen_policy_expectation_respects_effective_mask(self):
        got = masked_policy_expectation([[3., 9., 100.]], [[.25, .25, .5]], [[True, True, False]])
        self.assertAlmostEqual(got.item(), 6.0)

    def test_10_target_snapshot_hash_is_stable_and_sensitive(self):
        model = nn.Linear(2, 2)
        snapshot = copy.deepcopy(model.state_dict())
        digest = _state_dict_sha256(snapshot)
        self.assertEqual(digest, _state_dict_sha256(copy.deepcopy(snapshot)))
        with torch.no_grad():
            snapshot["weight"][0, 0] += 1
        self.assertNotEqual(digest, _state_dict_sha256(snapshot))

    def test_11_counterfactual_diagnostic_is_not_a_training_target_source(self):
        path = ROOT / "diagnostics/run_q_credit_counterfactual.py"
        source = path.read_text(encoding="utf-8")
        self.assertNotIn("train_rollout(", source)
        self.assertIn("counterfactual_targets_used_for_training", source)
        self.assertIn("False", source)

    def test_12_fixed_target_supervised_fit_is_reproducible_and_nonmutating(self):
        torch.manual_seed(5)
        model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
        original = copy.deepcopy(model.state_dict())
        x = np.random.default_rng(4).normal(size=(64, 4)).astype(np.float32)
        actions = np.arange(64) % 3
        targets = np.linspace(-2, 5, 64, dtype=np.float32)
        _, a = supervised_fixed_target_fit(model, x, actions, targets, steps=12, batch_size=16, seed=99, checkpoints=(0, 1, 4, 12))
        _, b = supervised_fixed_target_fit(model, x, actions, targets, steps=12, batch_size=16, seed=99, checkpoints=(0, 1, 4, 12))
        self.assertEqual(a, b)
        for key in original:
            torch.testing.assert_close(model.state_dict()[key], original[key], rtol=0, atol=0)
        self.assertLess(a[-1]["huber_loss"], a[0]["huber_loss"])

    def test_13_smoke_replay_regression_artifacts_pass(self):
        path = ROOT / "diagnostics/results/q_credit_audit/smoke_audit_status.json"
        if not path.exists():
            self.skipTest("300-episode Q credit audit smoke has not been run")
        status = json.loads(path.read_text())
        self.assertTrue(status["training_actions_keyed_exact"])
        self.assertTrue(status["training_reward_exact"])
        self.assertTrue(status["training_delay_exact"])
        self.assertTrue(status["q_online_and_target_weights_exact"])
        self.assertTrue(status["task_reward_delay_rsr_mask_and_queue_trajectory_exact"])

    def test_14_nan_inf_guards(self):
        with self.assertRaises(ValueError):
            reconstruct_bellman_target(float("nan"), .9, 1., False)
        with self.assertRaises(ValueError):
            build_reward_event_attribution(assignments_fixture().assign(Task_Reward=np.inf))

    def test_15_ordered_variable_discount_gae_stays_finite(self):
        returns, deltas = calculate_gae_returns([1., 2.], [0., 0.], [2., 0.], [.9, .8], [False, True], .95)
        np.testing.assert_allclose(deltas, [2.8, 2.])
        np.testing.assert_allclose(returns, [4.51, 2.])


if __name__ == "__main__":
    unittest.main()
