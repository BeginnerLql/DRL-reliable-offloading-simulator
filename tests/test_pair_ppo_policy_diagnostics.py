import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from agents.ppo_agent import PPOAgent, PPOPairScoringPolicyNetwork
from tools.pair_policy_diagnostics import (
    DiagnosticPPOAgent,
    RELIABILITY_REQUIREMENT_TIERS,
    expected_pair_correlation,
    get_action_probabilities,
    jensen_shannon_divergence,
    low_rho_quartile_mass,
    make_requirement_counterfactual_state,
    normalized_reliability_feature,
    policy_entropy,
    requirement_sensitivity_rows,
    build_rho_counterfactual_vectors,
    rho_counterfactual_probabilities,
    sample_probe_indices,
    total_variation_distance,
)


class TestPolicyDiagnosticMetrics(unittest.TestCase):
    def test_probability_extraction_is_finite_28_and_normalized(self):
        policy = torch.nn.Linear(35, 28)
        probabilities = get_action_probabilities(policy, np.zeros(35))
        self.assertEqual(probabilities.shape, (28,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

    def test_distribution_metrics(self):
        p = np.array([0.0, 0.25, 0.75])
        q = np.array([0.0, 0.5, 0.5])
        self.assertAlmostEqual(total_variation_distance(p, p), 0.0)
        self.assertAlmostEqual(jensen_shannon_divergence(p, p), 0.0)
        self.assertAlmostEqual(total_variation_distance(p, q), 0.25)
        self.assertGreaterEqual(jensen_shannon_divergence(p, q), 0.0)

    def test_expected_rho_low_quartile_and_entropy(self):
        probabilities = np.zeros(28)
        probabilities[:7] = 1.0 / 7.0
        rho = np.arange(28, dtype=float) / 27.0
        self.assertAlmostEqual(expected_pair_correlation(probabilities, rho), rho[:7].mean())
        self.assertAlmostEqual(low_rho_quartile_mass(probabilities, rho), 1.0)
        self.assertAlmostEqual(policy_entropy(probabilities), np.log(7.0))

    def test_requirement_counterfactual_changes_only_last_state_feature(self):
        state = np.arange(35, dtype=np.float32)
        changed = make_requirement_counterfactual_state(state, 0.9999)
        np.testing.assert_array_equal(changed[:-1], state[:-1])
        self.assertAlmostEqual(changed[-1], 1.0)
        np.testing.assert_allclose(
            [normalized_reliability_feature(r) for r in RELIABILITY_REQUIREMENT_TIERS],
            [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        )

    def test_probe_sampling_is_deterministic(self):
        np.testing.assert_array_equal(sample_probe_indices(10, 5), [0, 2, 4, 6, 9])
        np.testing.assert_array_equal(sample_probe_indices(3, 20), [0, 1, 2])
        np.testing.assert_array_equal(sample_probe_indices(0, 5), [])


class TestPairDiagnosticCounterfactuals(unittest.TestCase):
    def setUp(self):
        self.rho = np.linspace(0.01, 0.99, 28)
        self.agent = PPOAgent(
            35,
            28,
            [64, 32],
            actor_mode="pair_scoring",
            num_servers=8,
            pair_correlations=self.rho,
        )
        self.state = np.zeros(35, dtype=np.float32)

    def test_zero_and_shuffled_do_not_mutate_original(self):
        original = self.agent.policy_old.pair_correlations.detach().cpu().numpy().copy()
        first = rho_counterfactual_probabilities(
            self.agent.policy_old, self.state, self.rho, seed=2041
        )
        second = rho_counterfactual_probabilities(
            self.agent.policy_old, self.state, self.rho, seed=2041
        )
        np.testing.assert_array_equal(first["shuffled"], second["shuffled"])
        np.testing.assert_array_equal(
            np.sort(self.rho), np.sort(np.linspace(0.01, 0.99, 28))
        )
        vectors = build_rho_counterfactual_vectors(self.rho, seed=2041)
        np.testing.assert_allclose(np.sort(vectors["shuffled"]), np.sort(self.rho))
        np.testing.assert_array_equal(
            self.agent.policy_old.pair_correlations.detach().cpu().numpy(), original
        )
        self.assertEqual(first["true"].shape, (28,))
        self.assertEqual(first["zero"].shape, (28,))

    def test_shuffled_rho_preserves_multiset_and_is_reproducible(self):
        first = rho_counterfactual_probabilities(
            self.agent.policy_old, self.state, self.rho, seed=2041
        )
        second = rho_counterfactual_probabilities(
            self.agent.policy_old, self.state, self.rho, seed=2041
        )
        np.testing.assert_array_equal(first["shuffled"], second["shuffled"])
        vectors = build_rho_counterfactual_vectors(self.rho, seed=2041)
        np.testing.assert_allclose(np.sort(vectors["shuffled"]), np.sort(self.rho))

    def test_flat_rho_counterfactual_is_rejected(self):
        flat = PPOAgent(35, 28, [64, 32], actor_mode="flat")
        with self.assertRaises(ValueError):
            rho_counterfactual_probabilities(flat.policy_old, self.state, self.rho)

    def test_deterministic_rho_channel_changes_logits(self):
        class FeatureScorer(torch.nn.Module):
            def __init__(self, feature_index):
                super().__init__()
                self.feature_index = feature_index
            def forward(self, features):
                return features[:, self.feature_index:self.feature_index + 1]
        self.agent.policy_old.scorer = FeatureScorer(8)
        true = get_action_probabilities(self.agent.policy_old, self.state)
        zero = rho_counterfactual_probabilities(
            self.agent.policy_old, self.state, self.rho
        )["zero"]
        self.assertFalse(np.allclose(true, zero))

    def test_deterministic_requirement_channel_changes_logits(self):
        class FeatureScorer(torch.nn.Module):
            def forward(self, features):
                return features[:, 11:12] * features[:, 0:1]
        self.agent.policy_old.scorer = FeatureScorer()
        requirement_state = self.state.copy()
        requirement_state[:8] = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        low = get_action_probabilities(
            self.agent.policy_old, make_requirement_counterfactual_state(requirement_state, 0.9)
        )
        high = get_action_probabilities(
            self.agent.policy_old, make_requirement_counterfactual_state(requirement_state, 0.9999)
        )
        self.assertFalse(np.allclose(low, high))

    def test_requirement_rows_are_four_tiers_per_probe(self):
        rows = requirement_sensitivity_rows(
            self.agent.policy_old,
            [{"state": self.state, "task_id": 1}],
            self.rho,
            list(__import__("itertools").combinations(range(1, 9), 2)),
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows["Reliability_Requirement"].tolist(), list(RELIABILITY_REQUIREMENT_TIERS))


class TestDiagnosticAgentCapture(unittest.TestCase):
    def test_capture_only_adds_archive_and_preserves_transition_count(self):
        agent = DiagnosticPPOAgent(
            35,
            28,
            [64, 32],
            actor_mode="flat",
            min_rollout=99,
        )
        state = np.zeros(35, dtype=np.float32)
        for task_id in range(2):
            agent.store_transition(
                state,
                task_id,
                1.0,
                state,
                1.0,
                done=task_id == 1,
                task_id=task_id,
            )
        self.assertEqual(len(agent.observation_archive), 2)
        self.assertEqual(len(agent.states), 2)
        np.testing.assert_array_equal(agent.observation_archive[0]["state"], state)


if __name__ == "__main__":
    unittest.main()
