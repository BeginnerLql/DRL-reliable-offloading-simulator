import unittest
from types import SimpleNamespace

from config.params import params
from core.main_loop import MainLoop


class _TaskRegistry:
    def __init__(self, tasks):
        self.tasks = {task.id: task for task in tasks}

    def get_task_by_id(self, task_id):
        return self.tasks.get(task_id)

    def remove_task(self, task_id):
        self.tasks.pop(task_id, None)


def _task(task_id, z, primary_stat, backup_stat, primary_finished, backup_finished):
    node = SimpleNamespace(server_id=1)
    return SimpleNamespace(
        id=task_id,
        z=z,
        primaryStat=primary_stat,
        backupStat=backup_stat,
        primaryFinished=primary_finished,
        backupFinished=backup_finished,
        primaryStarted=primary_finished - 1.0 if primary_finished is not None else None,
        backupStarted=backup_finished - 1.0 if backup_finished is not None else None,
        primaryNode=node,
        backupNode=node,
    )


class PPOOutcomeTimestampTests(unittest.TestCase):
    def setUp(self):
        self.assigned_rewards = []
        self.loop = MainLoop(
            model=SimpleNamespace(
                assign_task_reward=lambda task_id, reward: self.assigned_rewards.append((task_id, reward))
            ),
            total_episodes=0,
            maxtaskno=0,
            num_states=params.num_states,
            num_actions=params.num_actions,
        )

    def test_parallel_outcome_timestamps(self):
        cases = [
            ("success", "success", 5.0, 8.0, 5.0),
            ("failure", "success", 4.0, 6.0, 6.0),
            ("failure", "failure", 4.0, 7.0, 7.0),
            ("success", "failure", 5.0, 8.0, 5.0),
            ("success", None, 5.0, None, 5.0),
            (None, "success", None, 8.0, 8.0),
        ]
        for primary_stat, backup_stat, primary_finished, backup_finished, expected in cases:
            task = _task(
                1, 1, primary_stat, backup_stat,
                primary_finished, backup_finished,
            )
            self.assertEqual(self.loop._get_task_outcome_time(task), expected)

    def test_sequential_outcome_timestamps(self):
        cases = [
            ("success", None, 3.0, None, 3.0),
            ("failure", "success", 3.0, 7.0, 7.0),
            ("failure", "failure", 3.0, 8.0, 8.0),
        ]
        for primary_stat, backup_stat, primary_finished, backup_finished, expected in cases:
            task = _task(
                1, 0, primary_stat, backup_stat,
                primary_finished, backup_finished,
            )
            self.assertEqual(self.loop._get_task_outcome_time(task), expected)

    def test_terminal_time_uses_latest_outcome_not_poll_wakeup(self):
        self.loop.ppo_last_decision_time = 100.0
        self.loop.ppo_last_resolved_outcome_time = 101.2
        poll_wakeup_time = 120.0

        terminal_time = self.loop._get_ppo_terminal_time()
        terminal_delta = terminal_time - self.loop.ppo_last_decision_time

        self.assertAlmostEqual(terminal_time, 101.2)
        self.assertAlmostEqual(terminal_delta, 1.2)
        self.assertNotAlmostEqual(terminal_delta, poll_wakeup_time - 100.0)
        self.assertAlmostEqual(0.9 ** terminal_delta, 0.9 ** 1.2)

    def test_resolved_outcomes_conserve_interval_reward_and_latest_time(self):
        tasks = [
            _task(1, 0, "success", None, 100.4, None),
            _task(2, 0, "success", None, 100.8, None),
            _task(3, 0, "success", None, 101.2, None),
        ]
        self.loop.env_state = _TaskRegistry(tasks)
        self.loop.pendingList = [1, 2, 3]
        self.loop.ppo_last_resolved_outcome_time = None
        self.loop.calcReward = lambda task_id: (2.0, 1.0)

        self.loop._collect_resolved_task_outcomes()

        self.assertEqual(self.assigned_rewards, [(1, 2.0), (2, 2.0), (3, 2.0)])
        self.assertAlmostEqual(self.loop.ppo_last_resolved_outcome_time, 101.2)
        self.assertEqual(self.loop.pendingList, [])
        self.assertEqual(self.loop.env_state.tasks, {})


if __name__ == "__main__":
    unittest.main()
