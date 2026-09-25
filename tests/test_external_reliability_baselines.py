import unittest
from types import SimpleNamespace

import numpy as np
import simpy

from agents.masked_pair_ppo_agent import effective_action_mask
from diagnostics.external_reliability_baselines import (
    deterministic_policy_rng,
    estimate_pair_completion_latencies,
    max_reliability_choice,
    safe_min_latency_choice,
    safe_random_choice,
    validate_evaluation_count,
)


class ExternalReliabilityBaselineTests(unittest.TestCase):
    def test_max_reliability_always_selects_a_maximum_and_randomizes_ties(self):
        values = np.asarray([0.8, 0.95, 0.7, 0.95])
        rng = np.random.default_rng(2026)
        choices = [max_reliability_choice(values, rng)[0] for _ in range(100)]
        self.assertTrue(set(choices).issubset({1, 3}))
        self.assertEqual(set(choices), {1, 3})
        self.assertEqual(max_reliability_choice(values, np.random.default_rng(17))[0],
                         max_reliability_choice(values, np.random.default_rng(17))[0])

    def test_safe_min_latency_uses_only_production_safe_mask(self):
        reliability = np.asarray([0.89, 0.91, 0.97, 0.93])
        requirement = 0.92
        latency = np.asarray([0.1, 3.0, 5.0, 2.0])
        expected_safe, expected_effective, empty, best = effective_action_mask(
            reliability, requirement
        )
        action, safe, effective, actual_empty, actual_best, _ = safe_min_latency_choice(
            reliability, requirement, latency, np.random.default_rng(3)
        )
        self.assertTrue(np.array_equal(safe, expected_safe))
        self.assertTrue(np.array_equal(effective, expected_effective))
        self.assertFalse(empty)
        self.assertEqual(actual_empty, empty)
        self.assertEqual(actual_best, best)
        self.assertIn(action, {1, 2, 3})
        self.assertEqual(action, 3)

    def test_safe_set_single_action_produces_that_action(self):
        action, safe, effective, empty, _, _ = safe_min_latency_choice(
            [0.9, 0.95, 0.8], 0.94, [0.1, 3.0, 0.01], np.random.default_rng(4)
        )
        self.assertEqual(action, 1)
        self.assertEqual(int(safe.sum()), 1)
        self.assertEqual(int(effective.sum()), 1)
        self.assertFalse(empty)

    def test_empty_safe_set_fallback_max_reliability_then_min_latency(self):
        reliability = np.asarray([0.90, 0.95, 0.95, 0.91])
        latency = np.asarray([1.0, 6.0, 2.0, 0.1])
        action, safe, effective, empty, best, _ = safe_min_latency_choice(
            reliability, 0.99, latency, np.random.default_rng(8)
        )
        self.assertTrue(empty)
        self.assertEqual(best, 0.95)
        self.assertFalse(safe.any())
        self.assertTrue(np.array_equal(effective, [False, True, True, False]))
        self.assertEqual(action, 2)

    def test_empty_fallback_latency_ties_are_reproducible_and_not_first_biased(self):
        reliability = [0.9, 0.95, 0.95]
        latency = [1.0, 2.0, 2.0]
        first = [safe_min_latency_choice(reliability, 0.99, latency,
                                         np.random.default_rng(seed))[0]
                 for seed in range(100)]
        self.assertEqual(set(first), {1, 2})
        left = safe_min_latency_choice(reliability, 0.99, latency,
                                       np.random.default_rng(120))[0]
        right = safe_min_latency_choice(reliability, 0.99, latency,
                                        np.random.default_rng(120))[0]
        self.assertEqual(left, right)

    def test_safe_random_uses_exact_safe_or_max_reliability_fallback_set(self):
        reliability = [0.90, 0.95, 0.94]
        action, safe, effective, empty, best = safe_random_choice(
            reliability, 0.94, np.random.default_rng(2)
        )
        expected_safe, expected_effective, expected_empty, expected_best = effective_action_mask(
            reliability, 0.94
        )
        self.assertTrue(np.array_equal(safe, expected_safe))
        self.assertTrue(np.array_equal(effective, expected_effective))
        self.assertEqual((empty, best), (expected_empty, expected_best))
        self.assertIn(action, {1, 2})

    def test_decision_latency_estimator_uses_only_current_observables(self):
        env = simpy.Environment()
        task = SimpleNamespace(
            env=env, computation_demand=30.0, input_data_size_mb=10.0,
            reliability_requirement=0.99,
        )
        servers = {
            1: SimpleNamespace(server_id=1, processing_frequency=10.0, uplink_rate_mbps=20.0),
            2: SimpleNamespace(server_id=2, processing_frequency=15.0, uplink_rate_mbps=20.0),
        }

        class ObservableState:
            def get_server_by_id(self, sid):
                return servers[sid]

            def get_server_backlog_time(self, sid, current_time):
                return {1: 5.0, 2: 2.0}[sid]

        # The task intentionally has no Task_Delay/outcome attributes.
        estimated, per_server = estimate_pair_completion_latencies(
            task, ObservableState(), [(1, 2)]
        )
        np.testing.assert_allclose(per_server[1], 8.0)
        np.testing.assert_allclose(per_server[2], 6.0)
        np.testing.assert_allclose(estimated, [6.0])

    def test_policy_rng_is_deterministic_and_separate_by_policy_tag(self):
        trial = {"Trial_ID": 2, "Eval_Arrival_Seed": 17, "Eval_Spatial_Seed": 23}
        a = deterministic_policy_rng(trial, 123).integers(0, 2**31, size=20)
        b = deterministic_policy_rng(trial, 123).integers(0, 2**31, size=20)
        c = deterministic_policy_rng(trial, 124).integers(0, 2**31, size=20)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_formal_evaluation_count_and_nonfinite_validation(self):
        rows = 20 * 200
        frame = __import__("pandas").DataFrame({
            "episode": np.repeat(np.arange(1, 21), 200),
            "task_id": np.tile(np.arange(1, 201), 20),
            "numeric": np.zeros(rows),
        })
        validate_evaluation_count(frame)
        frame.loc[0, "numeric"] = np.inf
        with self.assertRaisesRegex(RuntimeError, "Inf in numeric"):
            validate_evaluation_count(frame)
        with self.assertRaisesRegex(RuntimeError, "Expected 4000"):
            validate_evaluation_count(frame.iloc[:-1])

    def test_nonfinite_values_are_rejected_before_selection(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            safe_min_latency_choice([0.9, np.nan], 0.8, [1.0, 2.0], np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "finite"):
            max_reliability_choice([0.9, np.inf], np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
