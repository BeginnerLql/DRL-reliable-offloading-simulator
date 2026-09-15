import unittest

import numpy as np
import pandas as pd

from tools.ppo_convergence_diagnostic import (
    action_concentration,
    advance_training_loop,
    mean_policy_entropy_by_requirement,
    parse_checkpoints,
    selected_rho_summary,
    server_usage_entropy,
)


class TestCheckpointParsing(unittest.TestCase):
    def test_sorted_and_deduplicated(self):
        self.assertEqual(parse_checkpoints("20,0,10,10,0"), [0, 10, 20])
        self.assertEqual(parse_checkpoints([4, 2, 4, 0]), [0, 2, 4])

    def test_negative_and_empty_rejected(self):
        with self.assertRaises(ValueError):
            parse_checkpoints("0,-1,2")
        with self.assertRaises(ValueError):
            parse_checkpoints("")


class _FakeLoop:
    def __init__(self):
        self.this_episode = 0
        self.total_episodes = 0
        self.calls = []

    def EP(self):
        self.calls.append(self.total_episodes)
        self.this_episode = self.total_episodes


class TestContinuation(unittest.TestCase):
    def test_one_loop_advances_without_reset(self):
        loop = _FakeLoop()
        self.assertEqual(advance_training_loop(loop, [0, 2, 4]), [0, 2, 4])
        self.assertEqual(loop.calls, [0, 2, 4])
        self.assertEqual(loop.this_episode, 4)


class TestBehaviorSummaries(unittest.TestCase):
    def test_entropy_and_normalization(self):
        rows = []
        for requirement in (0.9, 0.99, 0.999, 0.9999):
            rows.extend({"Reliability_Requirement": requirement, "Policy_Entropy": np.log(28.0)} for _ in range(2))
        frame = mean_policy_entropy_by_requirement(pd.DataFrame(rows))
        self.assertEqual(len(frame), 4)
        self.assertTrue(np.allclose(frame["Normalized_Policy_Entropy"], 1.0))

    def test_action_concentration_and_server_usage(self):
        frame = pd.DataFrame({"action_index": [1, 1, 2, 3], "server_j": [1, 1, 2, 3], "server_k": [2, 3, 3, 4]})
        concentration = action_concentration(frame)
        self.assertEqual(concentration["Unique_Actions_Selected"], 3)
        self.assertAlmostEqual(concentration["Top1_Action_Fraction"], .5)
        usage = server_usage_entropy(frame, 8)
        self.assertGreater(usage["Server_Usage_Entropy"], 0.0)
        self.assertLessEqual(usage["Normalized_Server_Usage_Entropy"], 1.0)
        self.assertAlmostEqual(sum(usage[f"Server_{i}_Usage_Fraction"] for i in range(1, 9)), 1.0)

    def test_selected_rho_quantiles(self):
        frame = pd.DataFrame({"action_index": [0, 1, 2, 3]})
        summary = selected_rho_summary(frame, [0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(summary["Selected_Rho_Mean"], .25)
        self.assertAlmostEqual(summary["Selected_Rho_Median"], .25)
        self.assertAlmostEqual(summary["Selected_Rho_Min"], .1)
        self.assertAlmostEqual(summary["Selected_Rho_Max"], .4)


if __name__ == "__main__":
    unittest.main()
