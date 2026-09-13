import inspect
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import simpy

from config.params import params
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.server import Server
from core.task import Task
import io_utils.save_parameters_and_logs as save_logs


class RuntimeReliabilityThresholdTests(unittest.TestCase):
    @staticmethod
    def _analytic_task(requirement=0.99, demand=10.0):
        task = Task.__new__(Task)
        task.computation_demand = demand
        task.reliability_requirement = requirement
        task.env_state = SimpleNamespace(
            get_active_failure_rate=lambda server_id: {1: 0.1, 2: 0.2}[server_id]
        )
        primary = SimpleNamespace(server_id=1, processing_frequency=10.0)
        backup = SimpleNamespace(server_id=2, processing_frequency=20.0)
        return task, primary, backup

    def test_formula_and_threshold_are_computed_once(self):
        task, primary, backup = self._analytic_task(requirement=0.99)
        task.initialize_reliability_evaluation(primary, backup)
        expected_primary_p = 1.0 - math.exp(-0.1 * 1.0)
        expected_backup_p = 1.0 - math.exp(-0.2 * 0.5)
        self.assertAlmostEqual(task.primary_failure_probability, expected_primary_p)
        self.assertAlmostEqual(task.backup_failure_probability, expected_backup_p)
        self.assertAlmostEqual(
            task.joint_failure_probability,
            expected_primary_p * expected_backup_p,
        )
        self.assertAlmostEqual(task.execution_reliability, 1.0 - expected_primary_p * expected_backup_p)
        self.assertTrue(task.reliability_satisfied)

        task.reliability_requirement = 0.9999
        task.initialize_reliability_evaluation(primary, backup)
        self.assertFalse(task.reliability_satisfied)

    def test_same_server_retry_uses_p_squared(self):
        task = Task.__new__(Task)
        task.computation_demand = 10.0
        task.reliability_requirement = 0.99
        task.env_state = SimpleNamespace(get_active_failure_rate=lambda server_id: 0.1)
        server = SimpleNamespace(server_id=1, processing_frequency=10.0)
        task.initialize_reliability_evaluation(server, server)
        expected_p = 1.0 - math.exp(-0.1)
        self.assertAlmostEqual(task.joint_failure_probability, expected_p ** 2)

    def test_no_runtime_bernoulli_sampling_remains(self):
        source = inspect.getsource(Task.primary) + inspect.getsource(Task.backup)
        self.assertNotIn("random.uniform", source)
        self.assertNotIn("fault_prob", source)

    def test_z0_primary_completion_is_nominal_and_backup_standby(self):
        env = simpy.Environment()
        state = EnvironmentState()
        server = Server(env, "Edge", 1, 10.0, 0.5, -37.8, 144.9)
        state.add_server_and_init_environment(server)
        task = Task.__new__(Task)
        task.env = env
        task.env_state = state
        task.id = 1
        task.task_size = 1.0
        task.computation_demand = 10.0
        task.reliability_requirement = 0.9
        task.primaryNode = task.backupNode = None
        task.z = None
        task.primaryStarted = task.primaryFinished = task.primaryStat = None
        task.primary_service_time = None
        task.backupStarted = task.backupFinished = task.backupStat = None
        task.resolution_event = env.event()
        task.teta = None
        task.primary_effective_failure_rate = None
        task.backup_effective_failure_rate = None
        task.primary_service_time_for_reliability = None
        task.backup_service_time_for_reliability = None
        task.primary_failure_probability = None
        task.backup_failure_probability = None
        task.joint_failure_probability = None
        task.execution_reliability = None
        task.reliability_satisfied = None
        task.initialize_reliability_evaluation(server, server)
        env.process(task.execute_task(server, server, 0))
        env.run()
        self.assertEqual(task.primaryStat, "success")
        self.assertIsNone(task.backupStarted)
        self.assertTrue(task.resolution_event.triggered)

    def test_z1_runs_both_nominal_replicas(self):
        env = simpy.Environment()
        state = EnvironmentState()
        primary = Server(env, "Edge", 1, 10.0, 0.001, -37.8, 144.9)
        backup = Server(env, "Edge", 2, 10.0, 0.001, -37.81, 144.91)
        state.add_server_and_init_environment(primary)
        state.add_server_and_init_environment(backup)
        task = Task.__new__(Task)
        task.env = env
        task.env_state = state
        task.id = 1
        task.task_size = 1.0
        task.computation_demand = 10.0
        task.reliability_requirement = 0.9
        task.primaryNode = task.backupNode = None
        task.z = None
        task.primaryStarted = task.primaryFinished = task.primaryStat = None
        task.primary_service_time = None
        task.backupStarted = task.backupFinished = task.backupStat = None
        task.resolution_event = env.event()
        task.teta = None
        for name in (
            "primary_effective_failure_rate", "backup_effective_failure_rate",
            "primary_service_time_for_reliability", "backup_service_time_for_reliability",
            "primary_failure_probability", "backup_failure_probability",
            "joint_failure_probability", "execution_reliability", "reliability_satisfied",
        ):
            setattr(task, name, None)
        task.initialize_reliability_evaluation(primary, backup)
        env.process(task.execute_task(primary, backup, 1))
        env.run()
        self.assertEqual(task.primaryStat, "success")
        self.assertEqual(task.backupStat, "success")
        self.assertEqual(task.primaryStarted, task.backupStarted)
        self.assertTrue(task.resolution_event.triggered)

    def test_runtime_scale_is_logged_and_used(self):
        env = simpy.Environment()
        state = EnvironmentState()
        state.add_server_and_init_environment(
            Server(env, "Edge", 1, 10.0, 0.001, -37.8, 144.9)
        )
        loop = MainLoop.__new__(MainLoop)
        loop.env_state = state
        loop.spatial_risk_rng = np.random.default_rng(2026)
        loop.this_episode = 1
        loop.episode_spatial_risk_log = []
        with patch.multiple(
            params,
            SPATIAL_RISK_ENABLED=True,
            SPATIAL_RISK_BETA_P=0.0,
            SPATIAL_CORRELATION_LENGTH_KM=0.5,
            FAILURE_RATE_SCALE=10.0,
        ):
            loop._initialize_episode_spatial_risk()
        self.assertAlmostEqual(state.get_active_failure_rate(1), 0.01)
        row = loop.episode_spatial_risk_log[0]
        self.assertAlmostEqual(row["base_failure_rate"], 0.001)
        self.assertAlmostEqual(row["failure_rate_scale"], 10.0)
        self.assertAlmostEqual(row["scaled_base_failure_rate"], 0.01)
        self.assertAlmostEqual(row["effective_failure_rate"], 0.01)
        self.assertAlmostEqual(row["spatial_hazard_multiplier"], 1.0)

    def test_ppo_state_dimension_includes_reliability_requirement(self):
        self.assertEqual(params.num_states, 3 * params.serverNo + 3)

    def test_final_status_uses_reliability_satisfied(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            pd.DataFrame({"Server_ID": [1]}).to_excel(temp_path / "server_info.xlsx", index=False)
            pd.DataFrame({"Task_ID": [1]}).to_excel(temp_path / "task_parameters.xlsx", index=False)
            task_row = (
                1, 1, 1, 0.0, 1.0, "success", 1, None, None, None, 0,
                0.9999, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.01, 0.99, False,
            )
            with patch.object(save_logs, "DATA_DIR", str(temp_path)), patch.object(save_logs, "RESULTS_DIR", str(temp_path / "results")):
                save_logs.save_params_and_logs(params, [], [task_row])
            assignments = pd.read_excel(temp_path / "results" / "fixed_rate_results" / "ppo_results.xlsx", sheet_name="TaskAssignments")
            self.assertEqual(assignments.loc[0, "Final_status"], "failure")


if __name__ == "__main__":
    unittest.main()
