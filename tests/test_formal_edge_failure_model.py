import inspect
import math
import unittest

import numpy as np
import pandas as pd
import simpy

from config.params import params
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.server import Server
from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    map_spatial_risk_to_effective_failure_rates,
)
from tools.generate_server_and_task_parameters import base_failure_rate_from_frequency


class FormalEdgeFailureModelTests(unittest.TestCase):
    def test_static_server_table_is_fixed_all_edge_and_monotone(self):
        frame = pd.read_excel("data/server_info.xlsx")
        self.assertEqual(frame["Server_ID"].tolist(), list(range(1, 9)))
        self.assertEqual(frame["Server_Type"].tolist(), ["Edge"] * 8)
        frequencies = frame["Processing_Frequency"].to_numpy(dtype=float)
        expected_frequencies = np.array([10, 11, 12, 14, 15, 17, 18, 20], dtype=float)
        self.assertTrue(np.array_equal(frequencies, expected_frequencies))
        rates = frame["Base_Failure_Rate"].to_numpy(dtype=float)
        expected = np.array([base_failure_rate_from_frequency(f) for f in expected_frequencies])
        self.assertTrue(np.allclose(rates, expected, rtol=0, atol=1e-15))
        self.assertAlmostEqual(rates[-1], 0.003, places=15)
        self.assertAlmostEqual(rates[0], 0.003 * 10 ** 0.9, places=15)
        self.assertTrue(np.all(np.diff(rates) < 0.0))

    def test_spatial_mapping_has_no_log_normal_mean_correction(self):
        base = np.array([0.001, 0.003, 0.02])
        beta = 0.8
        np.testing.assert_allclose(
            map_spatial_risk_to_effective_failure_rates(base, np.zeros(3), beta),
            base,
        )
        np.testing.assert_allclose(
            map_spatial_risk_to_effective_failure_rates(base, np.ones(3), beta),
            base * math.exp(beta),
        )
        np.testing.assert_allclose(
            map_spatial_risk_to_effective_failure_rates(base, -np.ones(3), beta),
            base * math.exp(-beta),
        )
        source = inspect.getsource(map_spatial_risk_to_effective_failure_rates)
        self.assertNotIn("square(np.float64(beta))", source)
        self.assertNotIn("beta**2 / 2", source)

    def test_sigma_is_eight_by_eight_and_episode_hazard_is_shared(self):
        env = simpy.Environment()
        state = EnvironmentState()
        frame = pd.read_excel("data/server_info.xlsx")
        for row in frame.itertuples(index=False):
            state.add_server_and_init_environment(
                Server(env, "Edge", int(row.Server_ID), float(row.Processing_Frequency),
                       float(row.Base_Failure_Rate), float(row.Latitude), float(row.Longitude))
            )
        loop = MainLoop.__new__(MainLoop)
        loop.env_state = state
        loop.spatial_risk_rng = np.random.default_rng(params.SPATIAL_RISK_SEED)
        loop.this_episode = 1
        loop.episode_spatial_risk_log = []
        loop._initialize_episode_failure_rates()
        self.assertEqual(state.spatial_correlation_matrix.shape, (8, 8))
        self.assertTrue(np.allclose(state.spatial_correlation_matrix, state.spatial_correlation_matrix.T))
        self.assertTrue(np.allclose(np.diag(state.spatial_correlation_matrix), 1.0))
        first = {sid: state.get_active_failure_rate(sid) for sid in range(1, 9)}
        second = {sid: state.get_active_failure_rate(sid) for sid in range(1, 9)}
        self.assertEqual(first, second)
        self.assertEqual(len(loop.episode_spatial_risk_log), 8)

    def test_server_base_rate_is_not_overwritten_by_effective_rate(self):
        env = simpy.Environment()
        server = Server(env, "Edge", 8, 20.0, 0.003, -37.8, 144.9)
        state = EnvironmentState()
        state.add_server_and_init_environment(server)
        state.set_episode_effective_failure_rates([8], [0.006])
        self.assertEqual(server.base_failure_rate, 0.003)
        self.assertEqual(server.failure_rate, 0.003)
        self.assertEqual(state.get_active_failure_rate(8), 0.006)


if __name__ == "__main__":
    unittest.main()
