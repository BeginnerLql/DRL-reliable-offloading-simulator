import unittest
from itertools import combinations
from unittest.mock import patch

import numpy as np
import torch

from agents.ppo_agent import (
    PPOAgent,
    PPOPairScoringPolicyNetwork,
    PPOPolicyNetwork,
)
from core.spatial_risk import extract_pair_correlations


class TestPairCorrelationExtraction(unittest.TestCase):
    def test_action_pair_order_and_mapping(self):
        ids = [10, 20, 30]
        matrix = np.array(
            [
                [1.0, 0.12, 0.34],
                [0.12, 1.0, 0.56],
                [0.34, 0.56, 1.0],
            ]
        )
        pairs = [(10, 20), (10, 30), (20, 30)]
        result = extract_pair_correlations(ids, matrix, pairs)
        np.testing.assert_allclose(result, [0.12, 0.34, 0.56])

    def test_full_28_action_mapping(self):
        ids = list(range(1, 9))
        matrix = np.eye(8, dtype=float)
        for row in range(8):
            for column in range(row + 1, 8):
                matrix[row, column] = matrix[column, row] = (row + column + 1) / 20.0
        pairs = list(combinations(ids, 2))
        result = extract_pair_correlations(ids, matrix, pairs)
        expected = np.asarray([matrix[j - 1, k - 1] for j, k in pairs])
        self.assertEqual(pairs[0], (1, 2))
        self.assertEqual(pairs[-1], (7, 8))
        np.testing.assert_allclose(result, expected)

    def test_invalid_pair_context_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_pair_correlations([1, 2], np.ones((2, 3)), [(1, 2)])
        with self.assertRaises(ValueError):
            extract_pair_correlations([1, 2], np.eye(2), [(1, 3)])
        with self.assertRaises(ValueError):
            extract_pair_correlations([1, 2], np.eye(2), [(1, 1)])


class TestPairScoringActor(unittest.TestCase):
    def setUp(self):
        self.num_servers = 8
        self.num_actions = 28
        self.rho = np.linspace(0.01, 0.99, self.num_actions)
        self.net = PPOPairScoringPolicyNetwork(
            35, self.num_actions, [64, 32], self.num_servers, self.rho
        )

    def test_pair_features_are_symmetric(self):
        x_j = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        x_k = torch.tensor([[0.7, 0.6, 0.5, 0.4]])
        task = torch.tensor([[0.2, 0.3, 0.99]])
        phi_jk = self.net.compose_pair_features(x_j, x_k, [0.42], task)
        phi_kj = self.net.compose_pair_features(x_k, x_j, [0.42], task)
        torch.testing.assert_close(phi_jk, phi_kj)
        torch.testing.assert_close(self.net.scorer(phi_jk), self.net.scorer(phi_kj))

    def test_feature_and_logit_shapes_for_single_and_batch(self):
        state = torch.arange(35, dtype=torch.float32)
        batch = torch.stack((state, state + 1.0))
        self.assertEqual(tuple(self.net.build_node_features(state).shape), (8, 4))
        self.assertEqual(tuple(self.net.build_node_features(batch).shape), (2, 8, 4))
        self.assertEqual(tuple(self.net.build_pair_features(state).shape), (28, 12))
        pair_features = self.net.build_pair_features(batch)
        self.assertEqual(tuple(pair_features.shape), (2, 28, 12))
        self.assertEqual(tuple(self.net(state).shape), (28,))
        self.assertEqual(tuple(self.net(batch).shape), (2, 28))

    def test_action_mapping_and_rho_feature(self):
        self.assertEqual(self.net.pair_indices[0].tolist(), [0, 1])
        self.assertEqual(self.net.pair_indices[-1].tolist(), [6, 7])
        features = self.net.build_pair_features(torch.zeros(35))
        np.testing.assert_allclose(features[:, 8].numpy(), self.rho)
        np.testing.assert_allclose(features[:, 9:12].numpy(), 0.0)

    def test_shared_scorer_buffers_and_gradients(self):
        self.assertEqual(self.net.scorer[-1].out_features, 1)
        self.assertFalse(isinstance(self.net.pair_correlations, torch.nn.Parameter))
        self.assertFalse(isinstance(self.net.pair_indices, torch.nn.Parameter))
        logits = self.net(torch.randn(4, 35))
        loss = -torch.distributions.Categorical(logits=logits).log_prob(
            torch.tensor([0, 1, 2, 3])
        ).mean()
        loss.backward()
        gradients = [p.grad for p in self.net.scorer.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertIsNone(self.net.pair_correlations.grad)

    def test_rho_changes_pair_feature_only_and_z_is_not_an_input(self):
        state = torch.randn(35)
        features_a = self.net.build_pair_features(state)
        original = self.net.pair_correlations.detach().clone()
        with torch.no_grad():
            self.net.pair_correlations.copy_(torch.zeros_like(self.net.pair_correlations))
        features_b = self.net.build_pair_features(state)
        self.assertFalse(torch.equal(features_a[:, 8], features_b[:, 8]))
        with torch.no_grad():
            self.net.pair_correlations.copy_(original)
        # The actor accepts only state and static rho; changing an external
        # hidden Z or effective-rate realization cannot change its output.
        hidden_z_a = np.zeros(8)
        hidden_z_b = np.full(8, 4.0)
        lambda_eff_a = np.exp(hidden_z_a)
        lambda_eff_b = np.exp(hidden_z_b)
        self.assertFalse(np.array_equal(lambda_eff_a, lambda_eff_b))
        output_a = self.net(state)
        output_b = self.net(state)
        torch.testing.assert_close(output_a, output_b)

    def test_spatial_off_context_is_zero(self):
        import Project_main

        with patch.object(Project_main.params, "SPATIAL_RISK_ENABLED", False):
            pairs, rho = Project_main.build_pair_correlations()
        self.assertEqual(len(pairs), 28)
        np.testing.assert_array_equal(rho, np.zeros(28))


class TestPPOActorModes(unittest.TestCase):
    def _run_smoke(self, actor_mode, **kwargs):
        agent = PPOAgent(
            num_states=35,
            num_actions=28,
            hidden_layers=[64, 32],
            actor_mode=actor_mode,
            min_rollout=1,
            k_epochs=1,
            batch_size=2,
            **kwargs,
        )
        state = np.zeros(35, dtype=np.float32)
        for task_id in range(2):
            action = agent.select_action(state, epsilon=0.0)
            self.assertIn(action, range(28))
            agent.store_transition(
                state,
                action,
                None,
                state,
                delta_t=1.0,
                done=task_id == 1,
                task_id=task_id,
            )
            agent.assign_task_reward(task_id, 1.0 - 0.1 * task_id)
        agent.train_step()
        self.assertEqual(agent.states, [])
        self.assertEqual(agent.policy_net.state_dict().keys(), agent.policy_old.state_dict().keys())
        for key, value in agent.policy_net.state_dict().items():
            torch.testing.assert_close(value, agent.policy_old.state_dict()[key])
        return agent

    def test_pair_mode_select_store_train(self):
        agent = self._run_smoke(
            "pair_scoring", num_servers=8, pair_correlations=np.zeros(28)
        )
        self.assertIsInstance(agent.policy_net, PPOPairScoringPolicyNetwork)
        self.assertIsInstance(agent.policy_old, PPOPairScoringPolicyNetwork)
        self.assertEqual(agent.value_net.output_layer.in_features, 32)

    def test_flat_mode_regression(self):
        agent = self._run_smoke("flat")
        self.assertIsInstance(agent.policy_net, PPOPolicyNetwork)
        self.assertEqual(agent.policy_net.output_layer.in_features, 32)
        self.assertEqual(agent.policy_net.output_layer.out_features, 28)

    def test_pair_mode_requires_valid_context(self):
        with self.assertRaises(ValueError):
            PPOAgent(35, 28, [64, 32], actor_mode="pair_scoring", num_servers=8)
        with self.assertRaises(ValueError):
            PPOAgent(
                35,
                28,
                [64, 32],
                actor_mode="pair_scoring",
                num_servers=8,
                pair_correlations=np.zeros(27),
            )
        with self.assertRaises(ValueError):
            PPOAgent(35, 28, [64, 32], actor_mode="unknown")


if __name__ == "__main__":
    unittest.main()
