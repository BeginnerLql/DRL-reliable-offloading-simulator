import unittest

import simpy

from core.env_state import EnvironmentState
from core.server import Server
from core.task import Task


class TaskLifecycleTests(unittest.TestCase):
    @staticmethod
    def _server(env, server_id, frequency=10.0, uplink=40.0):
        return Server(
            env,
            "Edge",
            server_id,
            frequency,
            0.001,
            -37.8 - server_id / 1000.0,
            144.9 + server_id / 1000.0,
            uplink,
        )

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
        task.all_replicas_done_event = env.event()
        task.resolved = False
        task.first_result_time = None
        task.first_finished_server_id = None
        task.all_replicas_finished = False
        task.all_replicas_finish_time = None
        task.resolution_bookkeeping_done = False
        task._lifecycle_cleanup_done = False
        task._replica_completion_records = set()
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

    def test_resolution_precedes_all_replica_completion_and_cleanup(self):
        env = simpy.Environment()
        state = EnvironmentState()
        server_a = self._server(env, 1, frequency=10.0, uplink=40.0)
        server_b = self._server(env, 2, frequency=5.0, uplink=20.0)
        state.add_server_and_init_environment(server_a)
        state.add_server_and_init_environment(server_b)
        task = self._task(env, state, 1)
        task.initialize_reliability_evaluation(server_a, server_b)
        state.add_task(task)

        observations = []

        def observe_lifecycle():
            yield task.resolution_event
            observations.append((
                float(env.now),
                state.num_completed_replicas,
                state.num_resolved_tasks,
                task.resolved,
                task.first_result_time,
                task.first_finished_server_id,
                task.all_replicas_finished,
                task.id in state.tasks,
            ))
            state.record_task_resolution(task)
            task.try_finalize_lifecycle()
            observations.append(("bookkeeping", task.id in state.tasks))
            yield task.all_replicas_done_event
            observations.append((
                "all_done",
                float(env.now),
                state.num_completed_replicas,
                task.all_replicas_finished,
                task.all_replicas_finish_time,
                task.id in state.tasks,
            ))

        env.process(task.execute_task(server_a, server_b))
        env.process(observe_lifecycle())
        env.run()

        self.assertAlmostEqual(task.primaryFinished, 1.2)
        self.assertAlmostEqual(task.backupFinished, 2.4)
        self.assertEqual(observations[0][:3], (1.2, 1, 0))
        self.assertTrue(observations[0][3])
        self.assertAlmostEqual(observations[0][4], 1.2)
        self.assertEqual(observations[0][5], 1)
        self.assertFalse(observations[0][6])
        self.assertTrue(observations[0][7])
        self.assertEqual(observations[1], ("bookkeeping", True))
        self.assertEqual(observations[2][:4], ("all_done", 2.4, 2, True))
        self.assertAlmostEqual(observations[2][4], 2.4)
        self.assertFalse(observations[2][5])
        self.assertEqual(state.num_resolved_tasks, 1)
        self.assertEqual(state.num_completed_tasks, 1)
        self.assertEqual(state.num_completed_replicas, 2)
        self.assertEqual(len(state.replica_completion_log), 2)
        self.assertEqual(
            {(row["task_id"], row["server_id"], row["replica_label"])
             for row in state.replica_completion_log},
            {(1, 1, "primary"), (1, 2, "backup")},
        )
        self.assertEqual(state.servers[1]["waiting_replicas"], [])
        self.assertEqual(state.servers[2]["waiting_replicas"], [])
        self.assertIsNone(state.servers[1]["running_replica"])
        self.assertIsNone(state.servers[2]["running_replica"])

    def test_all_replicas_can_finish_before_resolution_bookkeeping(self):
        env = simpy.Environment()
        state = EnvironmentState()
        server_a = self._server(env, 1)
        server_b = self._server(env, 2)
        state.add_server_and_init_environment(server_a)
        state.add_server_and_init_environment(server_b)
        task = self._task(env, state, 1)
        task.initialize_reliability_evaluation(server_a, server_b)
        state.add_task(task)

        env.process(task.execute_task(server_a, server_b))
        env.run()

        self.assertTrue(task.all_replicas_finished)
        self.assertFalse(task.resolution_bookkeeping_done)
        self.assertIn(task.id, state.tasks)
        self.assertEqual(state.num_completed_replicas, 2)
        self.assertEqual(state.num_resolved_tasks, 0)

        state.record_task_resolution(task)
        self.assertTrue(task.resolution_bookkeeping_done)
        self.assertTrue(task.try_finalize_lifecycle())
        self.assertNotIn(task.id, state.tasks)

    def test_three_tasks_have_one_resolution_and_two_replica_records_each(self):
        env = simpy.Environment()
        state = EnvironmentState()
        servers = [self._server(env, index, frequency=10.0, uplink=40.0)
                   for index in (1, 2, 3)]
        for server in servers:
            state.add_server_and_init_environment(server)
        pairs = [(servers[0], servers[1]), (servers[1], servers[2]), (servers[0], servers[2])]
        tasks = []
        for task_id, (primary, backup) in enumerate(pairs, start=1):
            task = self._task(env, state, task_id)
            task.initialize_reliability_evaluation(primary, backup)
            state.add_task(task)
            tasks.append(task)

            def record_resolution(task=task):
                yield task.resolution_event
                state.record_task_resolution(task)
                task.try_finalize_lifecycle()

            env.process(record_resolution())
            env.process(task.execute_task(primary, backup))

        env.run()

        self.assertEqual(state.num_resolved_tasks, 3)
        self.assertEqual(state.num_completed_tasks, 3)
        self.assertEqual(state.num_completed_replicas, 6)
        self.assertEqual(len(state.replica_completion_log), 6)
        self.assertEqual(set(state.tasks), set())
        self.assertTrue(all(task.all_replicas_finished for task in tasks))
        self.assertEqual(
            {(row["task_id"], row["replica_label"])
             for row in state.replica_completion_log},
            {(task_id, label) for task_id in (1, 2, 3)
             for label in ("primary", "backup")},
        )
        for server in servers:
            self.assertIsNone(state.servers[server.server_id]["running_replica"])
            self.assertEqual(state.servers[server.server_id]["waiting_replicas"], [])


if __name__ == "__main__":
    unittest.main()
