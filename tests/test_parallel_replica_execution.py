import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import pandas as pd
import simpy

from config.params import params
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.server import Server
from core.task import Task


class ParallelReplicaExecutionTests(unittest.TestCase):
    @staticmethod
    def _task(env, state, task_id, demand=10.0, payload=1.0):
        task = Task.__new__(Task)
        task.env = env
        task.env_state = state
        task.id = task_id
        task.input_data_size_mb = payload
        task.computation_demand = demand
        task.reliability_requirement = 0.9
        task.primaryNode = None
        task.backupNode = None
        task.action_index = None
        task.primaryStarted = None
        task.primaryFinished = None
        task.primaryStat = None
        task.primary_service_time = None
        task.backupStarted = None
        task.backupFinished = None
        task.backupStat = None
        task.teta = None
        task.resolution_event = env.event()
        for name in (
            "primary_effective_failure_rate",
            "backup_effective_failure_rate",
            "primary_service_time_for_reliability",
            "backup_service_time_for_reliability",
            "primary_failure_probability",
            "backup_failure_probability",
            "joint_failure_probability",
            "execution_reliability",
            "reliability_satisfied",
            "base_reward",
            "reliability_violation",
            "reliability_penalty",
        ):
            setattr(task, name, None)
        return task

    @staticmethod
    def _server(env, server_id, frequency, uplink):
        return Server(
            env, "Edge", server_id, frequency, 0.001,
            -37.8 - server_id / 1000.0,
            144.9 + server_id / 1000.0,
            uplink,
        )

    @staticmethod
    def _install_start_tracer(state, starts):
        original = state.start_replica_execution

        def traced(self, server_id, task, selection, service_time, service_start_time):
            starts.append((task.id, server_id, selection, float(service_start_time)))
            return original(server_id, task, selection, service_time, service_start_time)

        state.start_replica_execution = MethodType(traced, state)

    def test_parallel_upload_first_result_and_slow_replica_resource_occupancy(self):
        env = simpy.Environment()
        state = EnvironmentState()
        server_a = self._server(env, 1, frequency=10.0, uplink=40.0)
        server_b = self._server(env, 2, frequency=5.0, uplink=20.0)
        server_c = self._server(env, 3, frequency=20.0, uplink=40.0)
        for server in (server_a, server_b, server_c):
            state.add_server_and_init_environment(server)

        starts = []
        self._install_start_tracer(state, starts)
        task_a = self._task(env, state, 1)
        task_a.initialize_reliability_evaluation(server_a, server_b)
        task_a.action_index = 0
        upload_starts = []
        original_delay = task_a.calc_input_output_delay

        def traced_delay(self, server):
            delay = original_delay(server)
            upload_starts.append((server.server_id, float(env.now), delay[0]))
            return delay

        task_a.calc_input_output_delay = MethodType(traced_delay, task_a)
        task_b = self._task(env, state, 2)
        task_b.initialize_reliability_evaluation(server_b, server_c)
        task_b.action_index = 1

        first_result_observation = []

        def observe_first_result():
            yield task_a.resolution_event
            running_b = state.servers[2]["running_replica"]
            first_result_observation.append({
                "time": float(env.now),
                "running_task": None if running_b is None else running_b["task"].id,
                "backlog": state.get_server_backlog_time(2, env.now),
            })

        def start_second_task():
            yield env.timeout(0.5)
            yield env.process(task_b.execute_task(server_b, server_c))

        env.process(task_a.execute_task(server_a, server_b))
        env.process(observe_first_result())
        env.process(start_second_task())
        env.run()

        # Replica uploads both begin at the task decision time and use 8D/r.
        self.assertEqual(sorted((sid, t) for sid, t, _ in upload_starts), [(1, 0.0), (2, 0.0)])
        self.assertAlmostEqual(next(d for sid, _, d in upload_starts if sid == 1), 0.2)
        self.assertAlmostEqual(next(d for sid, _, d in upload_starts if sid == 2), 0.4)
        # A finishes at .2 + 1.0 = 1.2; B would finish at .4 + 2.0 = 2.4.
        self.assertAlmostEqual(task_a.primaryFinished, 1.2)
        self.assertAlmostEqual(task_a.backupFinished, 2.4)
        self.assertAlmostEqual(task_a.resolution_event.value, 1.2)
        self.assertAlmostEqual(task_a.resolution_event.value - task_a.primaryStarted, 1.2)
        self.assertEqual(first_result_observation[0]["running_task"], 1)
        self.assertAlmostEqual(first_result_observation[0]["time"], 1.2)
        self.assertAlmostEqual(first_result_observation[0]["backlog"], 3.2)

        # The second task's B replica cannot start until task 1 releases B at 2.4.
        task_2_b_start = next(
            start for task_id, server_id, _, start in starts
            if task_id == 2 and server_id == 2
        )
        self.assertAlmostEqual(task_2_b_start, 2.4)
        self.assertAlmostEqual(task_b.primaryFinished, 4.4)
        self.assertIsNone(state.servers[2]["running_replica"])
        self.assertEqual(state.servers[2]["waiting_replicas"], [])
        self.assertEqual(state.num_completed_replicas, 4)
        self.assertEqual(state.num_resolved_tasks, 0)

    def test_same_server_pair_is_rejected_at_task_execution(self):
        env = simpy.Environment()
        state = EnvironmentState()
        server = self._server(env, 1, 10.0, 20.0)
        task = self._task(env, state, 1)
        generator = task.execute_task(server, server)
        with self.assertRaisesRegex(ValueError, "distinct Edge servers"):
            next(generator)

    def test_first_result_delay_uses_minimum_finish_timestamp(self):
        env = simpy.Environment()
        state = EnvironmentState()
        task = self._task(env, state, 1)
        task.primaryStarted = 0.0
        task.primaryFinished = 1.2
        task.backupFinished = 2.4
        task.reliability_satisfied = True
        task.reliability_requirement = 0.9
        task.joint_failure_probability = 0.0
        state.add_task(task)
        loop = MainLoop.__new__(MainLoop)
        loop.env_state = state

        reward, delay = loop.calcReward(1)

        self.assertIsNotNone(reward)
        self.assertAlmostEqual(delay, 1.2)
        self.assertAlmostEqual(loop._get_task_outcome_time(task), 1.2)

    def test_task_assignment_contains_pair_metadata_and_no_formal_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_parameters.xlsx"
            pd.DataFrame({
                "Task_ID": [1],
                "Input_Data_Size_MB": [1.0],
                "Computation_Demand": [10.0],
                "Reliability_Requirement": [0.9],
            }).to_excel(path, index=False)
            task = Task(simpy.Environment(), EnvironmentState(), 1, params_file=path)
        self.assertFalse(hasattr(task, "z"))
        self.assertEqual(params.num_actions, 28)
        self.assertEqual(params.num_states, 35)


if __name__ == "__main__":
    unittest.main()
