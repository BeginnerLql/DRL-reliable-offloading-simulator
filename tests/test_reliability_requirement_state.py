import math
import unittest
from types import SimpleNamespace

import numpy as np
import simpy

from agents.ppo_agent import PPOAgent
from config.params import params
from core.env_state import EnvironmentState
from core.server import Server


class ReliabilityRequirementStateTests(unittest.TestCase):
    def _state_and_task(self, requirement=0.9):
        env = simpy.Environment()
        state = EnvironmentState()
        uplink_rates = (18.0, 30.0, 22.0, 36.0, 16.0, 28.0, 40.0, 24.0)
        for server_id in range(1, params.serverNo + 1):
            state.add_server_and_init_environment(
                Server(
                    env, "Edge", server_id, 10.0, 0.001 * server_id,
                    -37.80 - 0.001 * server_id, 144.95 + 0.001 * server_id,
                    uplink_rates[server_id - 1],
                )
            )
        task = SimpleNamespace(
            env=env,
            task_size=50.0,
            computation_demand=50.0,
            reliability_requirement=requirement,
        )
        return state, task

    def test_num_states_is_35_for_eight_servers(self):
        self.assertEqual(params.serverNo, 8)
        self.assertEqual(params.num_states, 35)

    def test_number_of_nines_normalization_uses_formula(self):
        for requirement in (0.9, 0.99, 0.999, 0.9999):
            expected = np.clip(
                (-math.log10(1.0 - requirement) - 1.0) / 3.0,
                0.0,
                1.0,
            )
            actual = EnvironmentState.normalize_reliability_requirement(requirement)
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(EnvironmentState.normalize_reliability_requirement(0.9), 0.0)
        self.assertAlmostEqual(EnvironmentState.normalize_reliability_requirement(0.99), 1.0 / 3.0)
        self.assertAlmostEqual(EnvironmentState.normalize_reliability_requirement(0.999), 2.0 / 3.0)
        self.assertAlmostEqual(EnvironmentState.normalize_reliability_requirement(0.9999), 1.0)

    def test_invalid_reliability_requirements_are_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 0.0, 1.0, 1.1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "reliability_requirement"):
                    EnvironmentState.normalize_reliability_requirement(value)

    def test_requirement_is_last_feature_and_only_feature_that_changes(self):
        state_a, task_a = self._state_and_task(0.9)
        state_b, task_b = self._state_and_task(0.9999)
        observation_a = state_a.get_state(task_a)
        observation_b = state_b.get_state(task_b)
        self.assertEqual(len(observation_a), 35)
        self.assertTrue(np.array_equal(observation_a[:-1], observation_b[:-1]))
        self.assertNotEqual(observation_a[-1], observation_b[-1])
        self.assertAlmostEqual(observation_a[-1], 0.0)
        self.assertAlmostEqual(observation_b[-1], 1.0)


    def test_state_blocks_are_server_id_ordered_and_uplink_normalized(self):
        env = simpy.Environment()
        state = EnvironmentState()
        uplink_rates = (18.0, 30.0, 22.0, 36.0, 16.0, 28.0, 40.0, 24.0)
        insertion_order = (8, 3, 1, 7, 2, 6, 4, 5)
        task = SimpleNamespace(
            env=env,
            input_data_size_mb=1.25,
            computation_demand=27.0,
            reliability_requirement=0.99,
        )
        for server_id in insertion_order:
            server = Server(
                env, "Edge", server_id, 10.0 + server_id,
                0.003 + 0.001 * server_id,
                -37.80 - 0.001 * server_id,
                144.95 + 0.001 * server_id,
                uplink_rates[server_id - 1],
            )
            state.add_server_and_init_environment(server)
            state.servers[server_id]["waiting_replicas"] = [{
                "task": task,
                "selection": "test",
                "service_time": float(server_id),
            }]

        observation = state.get_state(task)
        self.assertEqual(observation.shape, (35,))
        sorted_ids = range(1, 9)
        expected_failure = np.array([0.003 + 0.001 * i for i in sorted_ids])
        expected_failure = EnvironmentState().normalize(
            expected_failure, params.LAMBDA_REF,
            params.LAMBDA_REF * 10 ** params.FAILURE_RATE_OMEGA,
        )
        expected_frequency = EnvironmentState().normalize(
            np.array([10.0 + i for i in sorted_ids]),
            params.FAILURE_RATE_FMIN, params.FAILURE_RATE_FMAX,
        )
        expected_backlog = np.array([i / (i + params.BACKLOG_TIME_SCALE_SEC) for i in sorted_ids])
        expected_uplink = (
            (np.array(uplink_rates) - 16.0) / 24.0
        )
        np.testing.assert_allclose(observation[0:8], expected_failure)
        np.testing.assert_allclose(observation[8:16], expected_frequency)
        np.testing.assert_allclose(observation[16:24], expected_backlog)
        np.testing.assert_allclose(observation[24:32], expected_uplink)
        self.assertAlmostEqual(observation[32], 0.5)
        self.assertAlmostEqual(observation[33], (27.0 - 5.0) / 45.0)
        self.assertAlmostEqual(observation[34], 1.0 / 3.0)
        # Server 1 and Server 8 align across every block.
        self.assertAlmostEqual(observation[24], (18.0 - 16.0) / 24.0)
        self.assertAlmostEqual(observation[31], (24.0 - 16.0) / 24.0)

    def test_uplink_range_is_observation_only_configuration(self):
        self.assertEqual(params.UPLINK_RATE_RANGE_MBPS, (16.0, 40.0))

    def test_observation_failure_features_use_nominal_server_rates(self):
        state, task = self._state_and_task(0.99)
        baseline = state.get_state(task)
        expected_nominal = 0.001
        expected_first = (expected_nominal - params.LAMBDA_REF) / (
            params.LAMBDA_REF * 10 ** params.FAILURE_RATE_OMEGA - params.LAMBDA_REF
        )
        self.assertAlmostEqual(baseline[0], expected_first)
        # Effective runtime hazards may differ, but the state must stay nominal.
        state.set_episode_effective_failure_rates(
            list(range(1, params.serverNo + 1)),
            np.arange(1, params.serverNo + 1, dtype=float),
        )
        with_spatial_hazard = state.get_state(task)
        self.assertTrue(np.array_equal(baseline, with_spatial_hazard))

    def test_ppo_dimensions_and_action_count(self):
        agent = PPOAgent(
            params.num_states,
            params.num_actions,
            params.hidden_layers_ppo,
            activation=params.af_ppo,
        )
        self.assertEqual(agent.policy_net.hidden_layers[0].in_features, 35)
        self.assertEqual(agent.value_net.hidden_layers[0].in_features, 35)
        self.assertEqual(params.num_actions, 28)


if __name__ == "__main__":
    unittest.main()
