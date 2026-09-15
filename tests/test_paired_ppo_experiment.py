import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agents.ppo_agent import PPOAgent
from tools.paired_ppo_experiment import (
    FrozenEvaluationPPOAgent,
    aggregate_statistics,
    bootstrap_mean_ci,
    build_paired_differences,
    generate_seed_plan,
    summarize_metric,
    scoped_environment_seeds,
)
from core.main_loop import MainLoop
from config.params import params


class TestSeedPlan(unittest.TestCase):
    def test_reproducible_and_paired(self):
        first = generate_seed_plan(31001, 3)
        second = generate_seed_plan(31001, 3)
        different = generate_seed_plan(31002, 3)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first.equals(different))
        for column in first.columns[1:]:
            self.assertEqual(first[column].nunique(), len(first))
        self.assertTrue((first.Train_Arrival_Seed != first.Eval_Arrival_Seed).all())
        self.assertTrue((first.Train_Spatial_Seed != first.Eval_Spatial_Seed).all())

    def test_flat_and_pair_share_every_seed_by_trial(self):
        plan = generate_seed_plan(31001, 2)
        for _, row in plan.iterrows():
            flat = row.to_dict()
            pair = row.to_dict()
            self.assertEqual(flat, pair)

    def test_common_random_streams_and_scoped_restore(self):
        old_arrival, old_spatial = params.TASK_ARRIVAL_SEED, params.SPATIAL_RISK_SEED
        with scoped_environment_seeds(12345, 67890):
            loop_a = MainLoop(object(), 1, 1, 35, 28)
            loop_b = MainLoop(object(), 1, 1, 35, 28)
            arrivals_a = [loop_a._sample_interarrival_time() for _ in range(4)]
            arrivals_b = [loop_b._sample_interarrival_time() for _ in range(4)]
            spatial_a = loop_a.spatial_risk_rng.normal(size=5)
            spatial_b = loop_b.spatial_risk_rng.normal(size=5)
            np.testing.assert_array_equal(arrivals_a, arrivals_b)
            np.testing.assert_array_equal(spatial_a, spatial_b)
        self.assertEqual(params.TASK_ARRIVAL_SEED, old_arrival)
        self.assertEqual(params.SPATIAL_RISK_SEED, old_spatial)


def _snapshot(agent):
    return {
        name: {key: value.detach().clone() for key, value in module.state_dict().items()}
        for name, module in (("policy_net", agent.policy_net), ("policy_old", agent.policy_old), ("value_net", agent.value_net))
    }


class TestFrozenEvaluationAgent(unittest.TestCase):
    def test_greedy_action_and_no_update(self):
        agent = PPOAgent(35, 28, [64, 32], min_rollout=1)
        with torch.no_grad():
            agent.policy_old.output_layer.bias.fill_(-1.0)
            agent.policy_old.output_layer.bias[7] = 3.0
            agent.policy_net.load_state_dict(agent.policy_old.state_dict())
        frozen = FrozenEvaluationPPOAgent(agent)
        state = np.zeros(35, dtype=np.float32)
        self.assertEqual(frozen.select_action(state, 0.0), 7)
        self.assertGreaterEqual(frozen.select_action(state, 0.0), 0)
        self.assertLess(frozen.select_action(state, 0.0), 28)
        before = _snapshot(frozen)
        frozen.store_transition(state, 7, None, state, 1.0, done=False, task_id=1)
        frozen.assign_task_reward(1, 2.0)
        frozen.store_transition(state, 7, None, state, 1.0, done=True, task_id=2)
        frozen.assign_task_reward(2, 2.0)
        frozen.train_step()
        self.assertEqual(frozen.states, [])
        after = _snapshot(frozen)
        for name in before:
            for key in before[name]:
                self.assertTrue(torch.equal(before[name][key], after[name][key]))


class TestAggregation(unittest.TestCase):
    def test_synthetic_summary_and_bootstrap(self):
        stats = summarize_metric([1, 2, 3])
        self.assertEqual(stats["N"], 3)
        self.assertAlmostEqual(stats["Mean"], 2.0)
        self.assertAlmostEqual(stats["Std"], 1.0)
        self.assertAlmostEqual(stats["Median"], 2.0)
        low, high = bootstrap_mean_ci([1, 2, 3], samples=200, seed=2043)
        self.assertLessEqual(low, 2.0)
        self.assertGreaterEqual(high, 2.0)

        delta = aggregate_statistics([1, 2, 3], bootstrap_samples=200, paired_delta=True)
        self.assertAlmostEqual(delta["Mean"], 2.0)
        self.assertAlmostEqual(delta["Fraction_Delta_Positive"], 1.0)
        self.assertAlmostEqual(delta["Fraction_Delta_Negative"], 0.0)

    def test_paired_difference_is_pair_minus_flat(self):
        columns = [
            "Trial_ID", "Actor_Mode", "Overall_Reliability_Satisfaction_Rate",
            "Overall_Mean_Task_Delay", "Overall_Mean_Reward", "Overall_Mean_Selected_Rho",
            "Status", "Failure_Message",
        ]
        rows = [
            [0, "flat", 0.5, 10.0, 1.0, 0.4, "success", ""],
            [0, "pair_scoring", 0.7, 8.0, 2.0, 0.2, "success", ""],
        ]
        run_level = pd.DataFrame(rows, columns=columns)
        tier_rows = []
        for mode, rsr, delay, reward, rho in (("flat", .5, 10, 1, .4), ("pair_scoring", .7, 8, 2, .2)):
            for req in (0.9, 0.99, 0.999, 0.9999):
                tier_rows.append({"Trial_ID": 0, "Actor_Mode": mode, "Reliability_Requirement": req,
                                  "Count": 1, "Reliability_Satisfaction_Rate": rsr,
                                  "Mean_Task_Delay": delay, "Mean_Reward": reward,
                                  "Mean_Selected_Rho": rho, "Status": "success", "Failure_Message": ""})
        paired = build_paired_differences(run_level, pd.DataFrame(tier_rows))
        overall = paired[paired.Reliability_Requirement.isna()].iloc[0]
        self.assertAlmostEqual(overall.Delta_Reliability_Satisfaction_Rate, .2)
        self.assertAlmostEqual(overall.Delta_Mean_Task_Delay, -2.0)
        self.assertAlmostEqual(overall.Delta_Mean_Reward, 1.0)
        self.assertAlmostEqual(overall.Delta_Mean_Selected_Rho, -.2)
        self.assertEqual(len(paired), 5)


class TestDataHashes(unittest.TestCase):
    def test_formal_task_and_server_files_exist(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("data/task_parameters.xlsx", "data/server_info.xlsx"):
            path = root / name
            self.assertTrue(path.exists(), name)
            self.assertEqual(len(hashlib.sha256(path.read_bytes()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
