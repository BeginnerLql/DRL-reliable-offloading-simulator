"""Focused checks for frozen reliability-mask selection and real queue tracking."""
import unittest
from types import SimpleNamespace

import numpy as np
import simpy
import torch

from core.server import Server
from core.task import Task
from diagnostics.evaluate_reliability_masked_policy import (
    TrackingEnvironmentState, production_pair_reliability,
    select_from_probabilities, weighted_quantile,
)


class TestReliabilityMaskedPolicy(unittest.TestCase):
    def test_mask_renormalization_and_empty_fallback(self):
        probabilities = np.array([.6, .3, .1])
        reliability = np.array([.9, .995, .999])
        action, masked, safe, empty = select_from_probabilities(
            "masked_greedy", probabilities, reliability, .99)
        self.assertEqual(action, 1)
        np.testing.assert_array_equal(safe, [False, True, True])
        np.testing.assert_allclose(masked, [0, .75, .25])
        self.assertFalse(empty)
        torch.manual_seed(3)
        sampled, masked_sample, _, empty = select_from_probabilities(
            "masked_sample", probabilities, reliability, .99)
        self.assertIn(sampled, (1, 2))
        np.testing.assert_allclose(masked_sample, masked)
        self.assertFalse(empty)
        fallback, weights, safe, empty = select_from_probabilities(
            "masked_sample", probabilities, reliability, .9999)
        self.assertTrue(empty)
        self.assertEqual(fallback, 2)
        self.assertFalse(safe.any())
        np.testing.assert_array_equal(weights, np.zeros(3))
        self.assertEqual(select_from_probabilities("greedy", probabilities, reliability, .99)[0], 0)

    def test_reliability_vector_matches_production_task_method(self):
        env = simpy.Environment()
        state = TrackingEnvironmentState()
        for sid, rate in ((1, .02), (2, .03), (3, .04)):
            state.add_server_and_init_environment(Server(env, "Edge", sid, 10.0,
                rate, -37.8, 144.9))
        state.set_episode_effective_failure_rates([1, 2, 3], [.02, .03, .04])
        task = object.__new__(Task)
        task.env_state, task.computation_demand, task.reliability_requirement = state, 50.0, .99
        values, failures = production_pair_reliability(task, state, [(1, 2), (1, 3), (2, 3)])
        task.initialize_reliability_evaluation(state.get_server_by_id(1), state.get_server_by_id(2))
        self.assertAlmostEqual(values[0], task.execution_reliability)
        self.assertAlmostEqual(failures[0], task.joint_failure_probability)
        np.testing.assert_allclose(values + failures, 1.0, atol=1e-15)

    def test_queue_length_is_time_weighted_actual_waiting_replicas(self):
        env = simpy.Environment()
        state = TrackingEnvironmentState()
        state.add_server_and_init_environment(Server(env, "Edge", 1, 10.0,
            .01, -37.8, 144.9))
        first = SimpleNamespace(id=1, env=env)
        second = SimpleNamespace(id=2, env=env)
        state.register_waiting_replica(1, first, "primary", 3.0)
        state.start_replica_execution(1, first, "primary", 3.0, 0.0)
        env.run(until=1.0)
        state.register_waiting_replica(1, second, "backup", 2.0)
        env.run(until=3.0)
        state.complete_replica_execution(1, first, "primary")
        state.start_replica_execution(1, second, "backup", 2.0, 3.0)
        env.run(until=5.0)
        state.complete_replica_execution(1, second, "backup")
        state.finalize()
        self.assertAlmostEqual(state._busy_time[1], 5.0)
        self.assertAlmostEqual(state._queue_area[1], 2.0)
        self.assertEqual(state._waits[1], [0.0, 2.0])
        self.assertEqual(weighted_quantile(state._segments[1], .95), 1.0)


if __name__ == "__main__":
    unittest.main()
