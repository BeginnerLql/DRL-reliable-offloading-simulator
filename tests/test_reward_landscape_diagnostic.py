"""Focused tie and regret classification tests for the offline audit."""
import math
import unittest

import pandas as pd

from diagnostics.reward_landscape_diagnostic import group_landscape


class RewardLandscapeTest(unittest.TestCase):
    def test_numerical_tie_and_second_distinct_reward(self):
        frame = pd.DataFrame({"action_index": [0, 1, 2],
                              "reward_total": [10.0, 10.0 - 5e-12, 9.99]})
        result, optimal, regrets = group_landscape(frame)
        self.assertEqual(optimal, {0, 1})
        self.assertEqual(result["exact_optimal_set_size"], 2)
        self.assertAlmostEqual(result["second_distinct_gap"], .01)
        self.assertEqual(result["near_size_1e-06"], 2)
        self.assertEqual(result["near_size_0.01"], 3)
        self.assertGreater(regrets[2], .009)

    def test_all_actions_tied_has_no_second_distinct_reward(self):
        frame = pd.DataFrame({"action_index": [0, 1, 2],
                              "reward_total": [3.0, 3.0, 3.0]})
        result, optimal, _ = group_landscape(frame)
        self.assertEqual(optimal, {0, 1, 2})
        self.assertTrue(result["all_actions_tied"])
        self.assertTrue(math.isnan(result["second_distinct_gap"]))


if __name__ == "__main__":
    unittest.main()
