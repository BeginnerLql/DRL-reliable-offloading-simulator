import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import simpy

from config.params import params
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.server import Server
from core.spatial_risk import map_spatial_risk_to_effective_failure_rates
from core.task import Task


class EpisodeSpatialRiskTests(unittest.TestCase):
    @staticmethod
    def _build_state(server_ids=(9, 2, 5), base_rates=None):
        if base_rates is None:
            base_rates = {9: 0.001, 2: 0.002, 5: 0.0005}
        coordinates = {
            9: (-37.814395, 144.963537),
            2: (-37.820910, 144.955155),
            5: (-37.812390, 144.971200),
        }
        env = simpy.Environment()
        state = EnvironmentState()
        for server_id in server_ids:
            latitude, longitude = coordinates[server_id]
            state.add_server_and_init_environment(
                Server(
                    env,
                    "Edge",
                    server_id,
                    10.0,
                    base_rates[server_id],
                    latitude,
                    longitude,
                )
            )
        return state

    @staticmethod
    def _build_loop(state, seed=2026):
        loop = MainLoop.__new__(MainLoop)
        loop.env_state = state
        loop.spatial_risk_rng = np.random.default_rng(seed)
        loop.this_episode = 1
        loop.episode_spatial_risk_log = []
        return loop

    @staticmethod
    def _enabled_patch(beta=0.5, seed=2026):
        return patch.multiple(
            params,
            SPATIAL_RISK_ENABLED=True,
            SPATIAL_CORRELATION_LENGTH_KM=0.5,
            SPATIAL_RISK_BETA_P=beta,
            SPATIAL_RISK_SEED=seed,
            FAILURE_RATE_SCALE=1.0,
        )

    @staticmethod
    def _task_for_state(state):
        task = Task.__new__(Task)
        task.env_state = state
        return task

    def test_disabled_spatial_risk_keeps_runtime_scale(self):
        state = self._build_state()
        server = state.get_server_by_id(9)
        original_rate = server.failure_rate
        loop = self._build_loop(state)
        with patch.multiple(
            params,
            SPATIAL_RISK_ENABLED=False,
            FAILURE_RATE_SCALE=10.0,
        ):
            loop._initialize_episode_spatial_risk()

        task = self._task_for_state(state)
        self.assertIsNone(state.spatial_risk_field)
        self.assertIsNone(state.spatial_correlation_matrix)
        self.assertEqual(task.set_failure_rate(server), 10.0 * original_rate)
        self.assertEqual(server.failure_rate, original_rate)

    def test_episode_context_uses_explicit_sorted_server_id_mapping(self):
        state = self._build_state(server_ids=(9, 2, 5))
        loop = self._build_loop(state)
        with self._enabled_patch(beta=0.5):
            loop._initialize_episode_spatial_risk()

        self.assertEqual(state.spatial_risk_server_ids, [2, 5, 9])
        self.assertEqual(set(state.effective_failure_rates), {2, 5, 9})
        base_rates = np.array([0.002, 0.0005, 0.001])
        expected = map_spatial_risk_to_effective_failure_rates(
            base_rates,
            state.spatial_risk_field,
            0.5,
        )
        for index, server_id in enumerate(state.spatial_risk_server_ids):
            self.assertAlmostEqual(
                state.effective_failure_rates[server_id], expected[index]
            )

    def test_episode_context_does_not_overwrite_base_failure_rates(self):
        state = self._build_state()
        original_rates = {
            server_id: state.get_server_by_id(server_id).failure_rate
            for server_id in state.servers
        }
        loop = self._build_loop(state)
        with self._enabled_patch(beta=0.5):
            loop._initialize_episode_spatial_risk()

        for server_id, original_rate in original_rates.items():
            server = state.get_server_by_id(server_id)
            self.assertEqual(server.failure_rate, original_rate)
            self.assertEqual(
                state.get_active_failure_rate(server_id),
                state.effective_failure_rates[server_id],
            )

    def test_active_hazard_is_fixed_within_one_episode(self):
        state = self._build_state()
        loop = self._build_loop(state)
        with self._enabled_patch(beta=0.5):
            loop._initialize_episode_spatial_risk()

        server = state.get_server_by_id(9)
        first = state.get_active_failure_rate(server.server_id)
        second = state.get_active_failure_rate(server.server_id)
        task_a = self._task_for_state(state)
        task_b = self._task_for_state(state)
        self.assertEqual(first, second)
        self.assertEqual(task_a.set_failure_rate(server), first)
        self.assertEqual(task_b.set_failure_rate(server), first)

    def test_task_reads_effective_rate_without_changing_server_rate(self):
        state = self._build_state(server_ids=(9,), base_rates={9: 0.001})
        state.set_episode_spatial_risk_context(
            [9],
            np.zeros((1, 1)),
            np.ones((1, 1)),
            np.array([0.0]),
            np.array([0.003]),
        )
        server = state.get_server_by_id(9)
        task = self._task_for_state(state)
        self.assertEqual(task.set_failure_rate(server), 0.003)
        self.assertEqual(server.failure_rate, 0.001)

    def test_reset_clears_episode_spatial_risk_context(self):
        state = self._build_state(server_ids=(9,))
        state.set_episode_spatial_risk_context(
            [9],
            np.zeros((1, 1)),
            np.ones((1, 1)),
            np.array([1.0]),
            np.array([0.003]),
        )
        state.reset()
        self.assertIsNone(state.spatial_risk_server_ids)
        self.assertIsNone(state.spatial_distance_matrix)
        self.assertIsNone(state.spatial_correlation_matrix)
        self.assertIsNone(state.spatial_risk_field)
        self.assertIsNone(state.effective_failure_rates)

        env = simpy.Environment()
        server = Server(env, "Edge", 9, 10.0, 0.001, -37.814395, 144.963537)
        state.add_server_and_init_environment(server)
        task = self._task_for_state(state)
        self.assertEqual(task.set_failure_rate(server), 0.001)

    def test_one_rng_produces_different_episode_fields(self):
        state = self._build_state()
        loop = self._build_loop(state, seed=2026)
        with self._enabled_patch(beta=0.5):
            loop._initialize_episode_spatial_risk()
            first = state.spatial_risk_field.copy()
            loop._initialize_episode_spatial_risk()
            second = state.spatial_risk_field.copy()
        self.assertFalse(np.array_equal(first, second))

    def test_same_seed_reproduces_the_episode_risk_sequence(self):
        with self._enabled_patch(beta=0.5, seed=2026):
            loop_a = MainLoop(
                SimpleNamespace(), 0, 0, params.num_states, params.num_actions
            )
            loop_b = MainLoop(
                SimpleNamespace(), 0, 0, params.num_states, params.num_actions
            )
            loop_a.env_state = self._build_state()
            loop_b.env_state = self._build_state()
            sequence_a = []
            sequence_b = []
            for _ in range(3):
                loop_a._initialize_episode_spatial_risk()
                loop_b._initialize_episode_spatial_risk()
                sequence_a.append(loop_a.env_state.spatial_risk_field.copy())
                sequence_b.append(loop_b.env_state.spatial_risk_field.copy())

        for field_a, field_b in zip(sequence_a, sequence_b):
            self.assertTrue(np.array_equal(field_a, field_b))
        self.assertFalse(np.array_equal(sequence_a[0], sequence_a[1]))

    def test_beta_zero_recovers_base_rates(self):
        state = self._build_state()
        loop = self._build_loop(state)
        with self._enabled_patch(beta=0.0):
            loop._initialize_episode_spatial_risk()
        expected = {9: 0.001, 2: 0.002, 5: 0.0005}
        for server_id, base_rate in expected.items():
            self.assertEqual(state.get_active_failure_rate(server_id), base_rate)

    def test_enabled_without_beta_raises_clear_error(self):
        state = self._build_state()
        loop = self._build_loop(state)
        with patch.multiple(
            params,
            SPATIAL_RISK_ENABLED=True,
            SPATIAL_RISK_BETA_P=None,
        ):
            with self.assertRaisesRegex(ValueError, "explicitly configured"):
                loop._initialize_episode_spatial_risk()

    def test_ppo_observation_stays_on_base_failure_rates(self):
        state = self._build_state()
        task = SimpleNamespace(
            env=simpy.Environment(),
            task_size=50.0,
            computation_demand=50.0,
            reliability_requirement=0.9,
        )
        loop = self._build_loop(state)
        with patch.object(params, "num_states", 12):
            baseline_state = state.get_state(task)
            with self._enabled_patch(beta=0.5):
                loop._initialize_episode_spatial_risk()
            spatial_state = state.get_state(task)
            self.assertEqual(len(spatial_state), params.num_states)
        self.assertTrue(np.array_equal(spatial_state, baseline_state))

    def test_enabled_episode_logs_one_row_per_server_with_context_mapping(self):
        state = self._build_state(server_ids=(9, 2, 5))
        loop = self._build_loop(state)
        with self._enabled_patch(beta=0.5):
            loop._initialize_episode_spatial_risk()

        self.assertEqual(len(loop.episode_spatial_risk_log), 3)
        for index, row in enumerate(loop.episode_spatial_risk_log):
            server_id = state.spatial_risk_server_ids[index]
            self.assertEqual(row["episode"], 1)
            self.assertEqual(row["server_id"], server_id)
            self.assertTrue(row["spatial_risk_enabled"])
            self.assertEqual(row["z_phy"], state.spatial_risk_field[index])
            self.assertEqual(
                row["base_failure_rate"],
                state.get_server_by_id(server_id).failure_rate,
            )
            self.assertEqual(
                row["effective_failure_rate"],
                state.effective_failure_rates[server_id],
            )
            self.assertAlmostEqual(
                row["hazard_multiplier"],
                row["effective_failure_rate"] / row["base_failure_rate"],
            )

    def test_enabled_beta_zero_logs_unit_hazard_multipliers(self):
        state = self._build_state()
        loop = self._build_loop(state)
        with self._enabled_patch(beta=0.0):
            loop._initialize_episode_spatial_risk()

        self.assertEqual(len(loop.episode_spatial_risk_log), 3)
        for row in loop.episode_spatial_risk_log:
            self.assertTrue(np.isclose(row["hazard_multiplier"], 1.0))
            self.assertEqual(
                row["effective_failure_rate"],
                row["base_failure_rate"],
            )

    def test_disabled_episode_does_not_add_spatial_risk_log_rows(self):
        state = self._build_state()
        loop = self._build_loop(state)
        with patch.object(params, "SPATIAL_RISK_ENABLED", False):
            loop._initialize_episode_spatial_risk()
        self.assertEqual(loop.episode_spatial_risk_log, [])

    def test_same_server_retry_uses_same_episode_hazard(self):
        state = self._build_state(server_ids=(9,))
        loop = self._build_loop(state)
        with self._enabled_patch(beta=0.5):
            loop._initialize_episode_spatial_risk()
        server = state.get_server_by_id(9)
        task = self._task_for_state(state)
        primary_rate = task.set_failure_rate(server)
        backup_retry_rate = task.set_failure_rate(server)
        self.assertEqual(primary_rate, backup_retry_rate)

    def test_spatial_on_beta_zero_uses_scaled_base_rate(self):
        state = self._build_state(server_ids=(9,))
        loop = self._build_loop(state)
        with patch.multiple(
            params,
            SPATIAL_RISK_ENABLED=True,
            SPATIAL_RISK_BETA_P=0.0,
            SPATIAL_CORRELATION_LENGTH_KM=0.5,
            FAILURE_RATE_SCALE=10.0,
        ):
            loop._initialize_episode_spatial_risk()
        self.assertAlmostEqual(state.get_active_failure_rate(9), 0.01)
        self.assertAlmostEqual(loop.episode_spatial_risk_log[0]["scaled_base_failure_rate"], 0.01)

    def test_spatial_on_beta_positive_uses_scaled_base_rate(self):
        state = self._build_state(server_ids=(9, 2, 5))
        loop = self._build_loop(state)
        with patch.multiple(
            params,
            SPATIAL_RISK_ENABLED=True,
            SPATIAL_RISK_BETA_P=0.5,
            SPATIAL_CORRELATION_LENGTH_KM=0.5,
            FAILURE_RATE_SCALE=10.0,
        ):
            loop._initialize_episode_spatial_risk()
        raw_rates = np.array([0.002, 0.0005, 0.001])
        expected = map_spatial_risk_to_effective_failure_rates(
            10.0 * raw_rates,
            loop.env_state.spatial_risk_field,
            0.5,
        )
        self.assertTrue(np.allclose(
            [state.effective_failure_rates[sid] for sid in state.spatial_risk_server_ids],
            expected,
        ))

    def test_spatial_off_does_not_create_fake_spatial_context(self):
        state = self._build_state(server_ids=(9,))
        loop = self._build_loop(state)
        with patch.multiple(params, SPATIAL_RISK_ENABLED=False, FAILURE_RATE_SCALE=10.0):
            loop._initialize_episode_failure_rates()
        self.assertAlmostEqual(state.get_active_failure_rate(9), 0.01)
        self.assertIsNone(state.spatial_risk_field)
        self.assertIsNone(state.spatial_distance_matrix)
        self.assertIsNone(state.spatial_correlation_matrix)
        self.assertEqual(loop.episode_spatial_risk_log, [])

    def test_spatial_off_keeps_ppo_state_on_raw_rates(self):
        state = self._build_state(server_ids=(9, 2, 5))
        task = SimpleNamespace(
            env=simpy.Environment(), task_size=50.0, computation_demand=50.0,
            reliability_requirement=0.9
        )
        loop = self._build_loop(state)
        with patch.object(params, "num_states", 12):
            baseline_state = state.get_state(task)
            with patch.multiple(params, SPATIAL_RISK_ENABLED=False, FAILURE_RATE_SCALE=10.0):
                loop._initialize_episode_failure_rates()
                scaled_state = state.get_state(task)
        self.assertTrue(np.array_equal(scaled_state, baseline_state))

    def test_episode_context_rejects_inconsistent_shapes_and_server_ids(self):
        state = self._build_state(server_ids=(9, 2))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            state.set_episode_spatial_risk_context(
                [9, 9],
                np.zeros((2, 2)),
                np.eye(2),
                np.zeros(2),
                np.ones(2),
            )
        with self.assertRaisesRegex(ValueError, "match"):
            state.set_episode_spatial_risk_context(
                [9],
                np.zeros((1, 1)),
                np.ones((1, 1)),
                np.zeros(1),
                np.ones(1),
            )
        with self.assertRaisesRegex(ValueError, "invalid shape"):
            state.set_episode_spatial_risk_context(
                [9, 2],
                np.zeros((2, 2)),
                np.eye(2),
                np.zeros(1),
                np.ones(2),
            )


if __name__ == "__main__":
    unittest.main()
