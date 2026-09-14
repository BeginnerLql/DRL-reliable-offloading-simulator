import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from config.params import params
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.server import Server
from core.task import Task, get_upload_time


class EdgeCommunicationTests(unittest.TestCase):
    def test_upload_time_formula_and_validation(self):
        self.assertAlmostEqual(get_upload_time(1.0, 20.0), 0.4)
        self.assertAlmostEqual(get_upload_time(0.5, 40.0), 0.1)
        self.assertAlmostEqual(get_upload_time(2.0, 16.0), 1.0)
        self.assertGreater(get_upload_time(1.0, 20.0), get_upload_time(1.0, 40.0))
        with self.assertRaises(ValueError):
            get_upload_time(-0.1, 20.0)
        with self.assertRaises(ValueError):
            get_upload_time(1.0, 0.0)

    def test_real_static_server_and_task_data(self):
        servers = pd.read_excel("data/server_info.xlsx")
        self.assertEqual(servers["Uplink_Rate"].tolist(), [18, 30, 22, 36, 16, 28, 40, 24])
        self.assertTrue((servers["Uplink_Rate"] > 0).all())
        self.assertEqual(servers["Server_Type"].tolist(), ["Edge"] * 8)
        tasks = pd.read_excel("data/task_parameters.xlsx")
        self.assertEqual(list(tasks.columns), [
            "Task_ID", "Input_Data_Size_MB", "Computation_Demand",
            "Reliability_Requirement",
        ])
        self.assertTrue(tasks["Input_Data_Size_MB"].between(0.5, 2.0).all())

    def test_main_loop_loads_uplink_rates(self):
        loop = MainLoop.__new__(MainLoop)
        loop.env = simpy.Environment()
        loop.env_state = EnvironmentState()
        loop.setServers()
        self.assertEqual(len(loop.env_state.servers), 8)
        self.assertEqual(
            [loop.env_state.servers[i]["server_object"].uplink_rate_mbps for i in range(1, 9)],
            [18, 30, 22, 36, 16, 28, 40, 24],
        )

    def test_upload_completes_before_cpu_and_does_not_extend_cpu_service(self):
        env = simpy.Environment()
        state = EnvironmentState()
        server = Server(env, "Edge", 1, 10.0, 0.001, -37.8, 144.9, 20.0)
        state.add_server_and_init_environment(server)
        events = []
        state.register_waiting_replica = lambda *args: events.append(("queued", env.now))
        state.start_replica_execution = lambda *args: events.append(("cpu_start", env.now))
        state.complete_replica_execution = lambda *args: events.append(("cpu_end", env.now))
        state.complete_task = lambda *args: None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_parameters.xlsx"
            pd.DataFrame({
                "Task_ID": [1],
                "Input_Data_Size_MB": [1.0],
                "Computation_Demand": [10.0],
                "Reliability_Requirement": [0.9],
            }).to_excel(path, index=False)
            task = Task(env, state, 1, params_file=path)
            task.initialize_reliability_evaluation(server, server)
            env.process(task.execute_task(server, server, 0))
            env.run()

        self.assertAlmostEqual(events[0][1], 0.4)
        self.assertAlmostEqual(events[1][1], 0.4)
        self.assertAlmostEqual(events[2][1], 1.4)
        self.assertAlmostEqual(task.primaryFinished, 1.4)
        self.assertAlmostEqual(task.primary_service_time, 1.0)
        self.assertAlmostEqual(task.primary_service_time_for_reliability, 1.0)

    def test_reliability_exposure_does_not_use_input_size_or_uplink(self):
        env = simpy.Environment()
        state = EnvironmentState()
        server = Server(env, "Edge", 1, 10.0, 0.1, -37.8, 144.9, 16.0)
        state.add_server_and_init_environment(server)
        task = Task.__new__(Task)
        task.env = env
        task.env_state = state
        task.computation_demand = 10.0
        task.reliability_requirement = 0.9
        task.input_data_size_mb = 0.5
        task.initialize_reliability_evaluation(server, server)
        first = (task.primary_failure_probability, task.execution_reliability)
        task.input_data_size_mb = 2.0
        server.uplink_rate_mbps = 40.0
        task.initialize_reliability_evaluation(server, server)
        second = (task.primary_failure_probability, task.execution_reliability)
        self.assertEqual(first, second)

    def test_dimensions_are_unchanged(self):
        self.assertEqual(params.num_states, 27)
        self.assertEqual(params.num_actions, 92)


if __name__ == "__main__":
    unittest.main()
