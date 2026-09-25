import random
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from diagnostics.run_long_horizon_coupling import (
    _interval_returns,
    capture_external_rng_state,
    queue_hazard_snapshot,
    restore_external_rng_state,
    snapshot_signature,
)


class LongHorizonCouplingTests(unittest.TestCase):
    @staticmethod
    def _fake_context():
        env = SimpleNamespace(now=12.5, _queue=[])
        queue = SimpleNamespace(queue=[], users=[])
        server = SimpleNamespace(queue=queue)
        task = SimpleNamespace(
            id=3, env=env, reliability_requirement=0.999,
            computation_demand=25.0, input_data_size_mb=2.0,
        )
        waiting = [{"task": SimpleNamespace(id=8), "selection": "primary", "service_time": 2.0}]
        info = {"server_object": server, "waiting_replicas": waiting, "running_replica": None}
        env_state = SimpleNamespace(
            servers={1: info},
            effective_failure_rates={1: 0.02},
            spatial_risk_field=np.asarray([0.5]),
            spatial_distance_matrix=np.asarray([[0.0]]),
            spatial_correlation_matrix=np.asarray([[1.0]]),
            get_server_backlog_time=lambda server_id, now: 2.0,
        )
        loop = SimpleNamespace(
            this_episode=1, taskCounter=3, pendingList=[], env=env,
            arrival_rng=np.random.default_rng(101),
            spatial_risk_rng=np.random.default_rng(202),
            interarrival_trace=[0.2, 0.3, 0.4],
        )
        return loop, task, env_state

    def test_external_rng_state_restores_policy_and_environment_streams(self):
        loop, _, _ = self._fake_context()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        try:
            random.seed(11)
            np.random.seed(12)
            torch.manual_seed(13)
            state = capture_external_rng_state(loop)
            first = (random.random(), float(np.random.random()), float(loop.arrival_rng.random()),
                     float(loop.spatial_risk_rng.random()), float(torch.rand(())))
            restore_external_rng_state(loop, state)
            second = (random.random(), float(np.random.random()), float(loop.arrival_rng.random()),
                      float(loop.spatial_risk_rng.random()), float(torch.rand(())))
            self.assertEqual(first, second)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)

    def test_snapshot_is_deterministic_and_detects_queue_or_hazard_changes(self):
        loop, task, env_state = self._fake_context()
        state = np.zeros(35, dtype=np.float32)
        first = queue_hazard_snapshot(loop, task, env_state, state)
        second = queue_hazard_snapshot(loop, task, env_state, state.copy())
        self.assertEqual(snapshot_signature(first), snapshot_signature(second))

        queue_before = snapshot_signature(first)
        env_state.servers[1]["waiting_replicas"][0]["service_time"] = 3.0
        queue_after = snapshot_signature(queue_hazard_snapshot(loop, task, env_state, state))
        self.assertNotEqual(queue_before, queue_after)
        env_state.servers[1]["waiting_replicas"][0]["service_time"] = 2.0

        hazard_before = snapshot_signature(queue_hazard_snapshot(loop, task, env_state, state))
        env_state.spatial_risk_field[0] = 0.75
        hazard_after = snapshot_signature(queue_hazard_snapshot(loop, task, env_state, state))
        self.assertNotEqual(hazard_before, hazard_after)

    def test_interval_returns_assign_completion_to_the_interval_it_occurs_in(self):
        decisions = [{"task_id": task_id, "decision_time": float(task_id - 1)}
                     for task_id in range(1, 202)]
        assignments = pd.DataFrame([
            {"task_id": 1, "Primary_Start": 0.0, "Task_Delay": 0.5, "Task_Reward": 1.0},
            {"task_id": 2, "Primary_Start": 1.0, "Task_Delay": 0.5, "Task_Reward": 2.0},
            {"task_id": 3, "Primary_Start": 2.0, "Task_Delay": 0.5, "Task_Reward": 3.0},
        ])
        gamma = 0.9
        result = _interval_returns(assignments, decisions, 1, 201.0, gamma)
        self.assertAlmostEqual(result[1]["q_return"], 1.0)
        self.assertAlmostEqual(result[5]["q_return"], 1.0 + gamma * 2.0 + gamma**2 * 3.0)
        self.assertEqual(result[50]["effective_horizon"], 50)


if __name__ == "__main__":
    unittest.main()
