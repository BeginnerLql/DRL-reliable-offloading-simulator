import unittest
from types import SimpleNamespace

import simpy

from config.params import params
from core.main_loop import MainLoop
from core.task import Task


def _task_with_event(env, primary_stat=None, backup_stat=None,
                     primary_finished=None, backup_finished=None):
    task = Task.__new__(Task)
    task.env = env
    task.resolution_event = env.event()
    task.primaryStat = primary_stat
    task.backupStat = backup_stat
    task.primaryFinished = primary_finished
    task.backupFinished = backup_finished
    return task


class TaskResolutionEventTests(unittest.TestCase):
    def test_resolution_fires_when_either_replica_finishes(self):
        cases = [
            (3.0, None, True),
            (None, 6.0, True),
            (3.0, 6.0, True),
            (None, None, False),
        ]
        for primary_finished, backup_finished, expected in cases:
            with self.subTest(primary=primary_finished, backup=backup_finished):
                env = simpy.Environment()
                task = _task_with_event(
                    env, primary_finished=primary_finished,
                    backup_finished=backup_finished,
                )
                task._signal_resolution_if_ready()
                self.assertEqual(task.resolution_event.triggered, expected)

    def test_resolution_event_is_one_shot_and_carries_timestamp(self):
        env = simpy.Environment()
        task = _task_with_event(env, "success", None, 5.0, None)
        env._now = 5.0

        task._signal_resolution_if_ready()
        self.assertTrue(task.resolution_event.triggered)
        self.assertEqual(task.resolution_event.value, 5.0)

        task._signal_resolution_if_ready()
        self.assertEqual(task.resolution_event.value, 5.0)

    def test_drain_waits_for_resolution_events(self):
        env = simpy.Environment()
        task_a = _task_with_event(env)
        task_b = _task_with_event(env)
        tasks = {1: task_a, 2: task_b}

        class Registry:
            def get_task_by_id(self, task_id):
                return tasks.get(task_id)

            def remove_task(self, task_id):
                tasks.pop(task_id, None)

        loop = MainLoop.__new__(MainLoop)
        loop.model_name = "ppo"
        assigned_rewards = []
        loop.model = SimpleNamespace(
            assign_task_reward=lambda task_id, reward: assigned_rewards.append((task_id, reward))
        )
        loop.env = env
        loop.env_state = Registry()
        loop.pendingList = [1, 2]
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
        self.assertEqual(assigned_rewards, [(1, 1.0), (2, 1.0)])


if __name__ == "__main__":
    unittest.main()
