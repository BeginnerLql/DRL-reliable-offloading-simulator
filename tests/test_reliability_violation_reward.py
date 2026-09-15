import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config.params import params
from core.main_loop import MainLoop


class ReliabilityViolationRewardTests(unittest.TestCase):
    @staticmethod
    def _loop_for(task):
        loop = MainLoop.__new__(MainLoop)
        loop.env_state = SimpleNamespace(get_task_by_id=lambda task_id: task)
        return loop

    @staticmethod
    def _task(requirement, joint_failure_probability, satisfied, delay=1.0):
        return SimpleNamespace(
            primaryStarted=0.0,
            primaryFinished=delay,
            backupFinished=None,
            reliability_requirement=requirement,
            joint_failure_probability=joint_failure_probability,
            reliability_satisfied=satisfied,
        )

    def test_failure_budget_violation_values(self):
        self.assertEqual(
            MainLoop._calculate_reliability_violation(
                SimpleNamespace(reliability_requirement=0.99, joint_failure_probability=0.005)
            ),
            0.0,
        )
        self.assertAlmostEqual(
            MainLoop._calculate_reliability_violation(
                SimpleNamespace(reliability_requirement=0.999, joint_failure_probability=0.005)
            ),
            math.log10(5.0),
        )
        self.assertAlmostEqual(
            MainLoop._calculate_reliability_violation(
                SimpleNamespace(reliability_requirement=0.9999, joint_failure_probability=0.005)
            ),
            math.log10(50.0),
        )
        self.assertAlmostEqual(
            MainLoop._calculate_reliability_violation(
                SimpleNamespace(reliability_requirement=0.99, joint_failure_probability=0.1)
            ),
            1.0,
        )
        self.assertAlmostEqual(
            MainLoop._calculate_reliability_violation(
                SimpleNamespace(reliability_requirement=0.99, joint_failure_probability=1.0)
            ),
            2.0,
        )
        self.assertEqual(
            MainLoop._calculate_reliability_violation(
                SimpleNamespace(reliability_requirement=0.9999, joint_failure_probability=0.0)
            ),
            0.0,
        )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            MainLoop._calculate_reliability_violation(
                SimpleNamespace(reliability_requirement=1.0, joint_failure_probability=0.0)
            )
        with self.assertRaises(ValueError):
            MainLoop._calculate_reliability_violation(
                SimpleNamespace(reliability_requirement=0.99, joint_failure_probability=1.1)
            )

    def test_success_reward_is_unchanged(self):
        task = self._task(0.99, 0.005, True)
        loop = self._loop_for(task)
        old_base_reward = math.log(1.0 - math.exp(-1.0)) / math.log(0.995)
        with patch.object(params, "RELIABILITY_VIOLATION_WEIGHT", 10.0):
            reward, delay = loop.calcReward(1)
        self.assertAlmostEqual(reward, old_base_reward)
        self.assertEqual(delay, 1.0)
        self.assertAlmostEqual(task.base_reward, old_base_reward)
        self.assertEqual(task.reliability_violation, 0.0)
        self.assertEqual(task.reliability_penalty, 0.0)

    def test_alpha_zero_reproduces_old_failure_reward(self):
        task = self._task(0.999, 0.005, False)
        loop = self._loop_for(task)
        with patch.object(params, "RELIABILITY_VIOLATION_WEIGHT", 0.0):
            reward, _ = loop.calcReward(1)
        self.assertEqual(reward, -3.0)
        self.assertAlmostEqual(task.base_reward, -3.0)
        self.assertGreater(task.reliability_violation, 0.0)
        self.assertEqual(task.reliability_penalty, 0.0)

    def test_failure_reward_includes_penalty_and_severity_is_monotonic(self):
        task_5x = self._task(0.999, 0.005, False)
        task_50x = self._task(0.9999, 0.005, False)
        loop_5x = self._loop_for(task_5x)
        loop_50x = self._loop_for(task_50x)
        with patch.object(params, "RELIABILITY_VIOLATION_WEIGHT", 10.0):
            reward_5x, _ = loop_5x.calcReward(1)
            reward_50x, _ = loop_50x.calcReward(1)
        self.assertAlmostEqual(reward_5x, -3.0 - 10.0 * math.log10(5.0))
        self.assertAlmostEqual(reward_50x, -3.0 - 10.0 * math.log10(50.0))
        self.assertLess(reward_50x, reward_5x)
        self.assertAlmostEqual(
            task_5x.reliability_penalty,
            10.0 * task_5x.reliability_violation,
        )

    def test_state_action_and_default_weight_are_unchanged_or_configured(self):
        self.assertEqual(params.num_states, 4 * params.serverNo + 3)
        self.assertEqual(params.num_actions, 28)
        self.assertEqual(params.RELIABILITY_VIOLATION_WEIGHT, 10.0)


if __name__ == "__main__":
    unittest.main()
