import unittest
from types import SimpleNamespace

import simpy

from config.params import params
from core.main_loop import MainLoop
from core.task import Task


def _task_with_event(env, z, primary_stat=None, backup_stat=None,
                     primary_finished=None, backup_finished=None):
    task = Task.__new__(Task)
    task.env = env
    task.resolution_event = env.event()
    task.z = z
    task.primaryStat = primary_stat
    task.backupStat = backup_stat
    task.primaryFinished = primary_finished
    task.backupFinished = backup_finished
    return task


class TaskResolutionEventTests(unittest.TestCase):
    def test_resolution_conditions(self):
        cases = [
            (0, "success", None, 3.0, None, True),
            (0, "failure", None, 3.0, None, False),
            (0, "failure", "success", 3.0, 7.0, True),
            (0, "failure", "failure", 3.0, 8.0, True),
            (1, "success", None, 5.0, None, True),
            (1, None, "success", None, 6.0, True),
            (1, "failure", None, 4.0, None, False),
            (1, "failure", "failure", 4.0, 7.0, True),
        ]
        for z, primary_stat, backup_stat, primary_finished, backup_finished, expected in cases:
            with self.subTest(z=z, primary=primary_stat, backup=backup_stat):
                env = simpy.Environment()
                task = _task_with_event(
                    env, z, primary_stat, backup_stat,
                    primary_finished, backup_finished,
                )
                task._signal_resolution_if_ready()
                self.assertEqual(task.resolution_event.triggered, expected)

    def test_resolution_event_is_one_shot_and_carries_timestamp(self):
        env = simpy.Environment()
        task = _task_with_event(env, 1, "success", None, 5.0, None)
        env._now = 5.0

        task._signal_resolution_if_ready()
        self.assertTrue(task.resolution_event.triggered)
        self.assertEqual(task.resolution_event.value, 5.0)

        task._signal_resolution_if_ready()
        self.assertEqual(task.resolution_event.value, 5.0)

    def test_drain_waits_for_resolution_events(self):
        env = simpy.Environment()
        task_a = _task_with_event(env, 0)
        task_b = _task_with_event(env, 0)
        tasks = {1: task_a, 2: task_b}

        class Registry:
            def get_task_by_id(self, task_id):
                return tasks.get(task_id)

            def remove_task(self, task_id):
                tasks.pop(task_id, None)

        loop = MainLoop.__new__(MainLoop)
        loop.model_name = "ppo"
        loop.env = env
        loop.env_state = Registry()
        loop.pendingList = [1, 2]
        loop.ppo_interval_reward = 0.0
        loop.ppo_last_resolved_outcome_time = None
        wake_times = []

        def calc_reward(task_id):
            task = tasks[task_id]
            return (1.0, 1.0) if task.resolution_event.triggered else (None, None)

        loop.calcReward = calc_reward
        loop._get_task_outcome_time = lambda task: task.resolution_event.value

        def finalize(task_id, reward, delay):
            wake_times.append(env.now)
            loop.pendingList.remove(task_id)
            loop.env_state.remove_task(task_id)

        loop._finalize_resolved_task = finalize

        def resolve_later(task, timestamp):
            yield env.timeout(timestamp)
            task.primaryStat = "success"
            task.primaryFinished = timestamp
            task._signal_resolution_if_ready()

        env.process(resolve_later(task_a, 2.0))
        env.process(resolve_later(task_b, 5.0))
        env.process(loop._drain_pending_tasks())
        env.run()

        self.assertEqual(wake_times, [2.0, 5.0])
        self.assertEqual(loop.pendingList, [])
        self.assertAlmostEqual(loop.ppo_interval_reward, 2.0)


if __name__ == "__main__":
    unittest.main()
