import unittest
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import torch

from agents.ddpg_agent import ddpgModel
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from config.params import params
from core.main_loop import MainLoop


class DualReplicaActionSpaceTests(unittest.TestCase):
    def _loop(self):
        return MainLoop(
            model=SimpleNamespace(),
            total_episodes=0,
            maxtaskno=0,
            num_states=params.num_states,
            num_actions=params.num_actions,
        )

    def test_action_space_is_exactly_all_28_unordered_distinct_pairs(self):
        loop = self._loop()
        expected = list(combinations(range(1, 9), 2))

        self.assertEqual(params.num_states, 27)
        self.assertEqual(params.num_actions, 28)
        self.assertEqual(len(loop.action_pairs), 28)
        self.assertEqual(loop.action_pairs, expected)
        self.assertEqual(loop.action_pairs[0], (1, 2))
        self.assertEqual(loop.action_pairs[-1], (7, 8))
        self.assertEqual(len(set(loop.action_pairs)), 28)
        self.assertTrue(all(server_j < server_k for server_j, server_k in loop.action_pairs))
        self.assertFalse(any(server_j == server_k for server_j, server_k in loop.action_pairs))
        self.assertFalse(any((server_k, server_j) in loop.action_pairs for server_j, server_k in loop.action_pairs))

    def test_every_action_index_decodes_to_its_distinct_server_pair(self):
        loop = self._loop()
        servers = {server_id: object() for server_id in range(1, 9)}
        loop.env_state = SimpleNamespace(get_server_by_id=servers.get)

        for action_index, (server_j, server_k) in enumerate(loop.action_pairs):
            with self.subTest(action_index=action_index):
                decoded_j, decoded_k = loop.extract_parameters_from_index(action_index)
                self.assertIs(decoded_j, servers[server_j])
                self.assertIs(decoded_k, servers[server_k])

        with self.assertRaisesRegex(IndexError, "out of range"):
            loop.extract_parameters_from_index(-1)
        with self.assertRaisesRegex(IndexError, "out of range"):
            loop.extract_parameters_from_index(28)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            loop.extract_parameters_from_index(1.5)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            loop.extract_parameters_from_index(True)

    def test_ddpg_scores_select_one_of_the_shared_pairs(self):
        loop = self._loop()
        scores = np.zeros(28)
        scores[17] = 1.0
        self.assertEqual(loop.action_index_from_scores(scores), 17)
        self.assertEqual(loop.action_index_from_scores(scores.tolist()), 17)
        with self.assertRaisesRegex(ValueError, "length 28"):
            loop.action_index_from_scores(np.zeros(92))

    def test_pair_set_is_not_filtered_by_reliability_requirement(self):
        loop = self._loop()
        all_pairs = tuple(loop.action_pairs)
        for requirement in (0.9, 0.99, 0.999, 0.9999):
            with self.subTest(requirement=requirement):
                # Requirement changes do not enter pair generation or index decoding.
                self.assertEqual(tuple(loop.action_pairs), all_pairs)
                self.assertEqual(len(loop.action_pairs), 28)

    def test_ppo_transition_keeps_pair_action_index_and_origin_reward(self):
        agent = PPOAgent(
            params.num_states, params.num_actions, params.hidden_layers_ppo,
            activation=params.af_ppo, min_rollout=1, k_epochs=0,
        )
        state = np.zeros(params.num_states, dtype=np.float32)
        agent.store_transition(state, 27, None, state, delta_t=0.5, done=False, task_id=11)
        agent.assign_task_reward(11, 3.25)
        self.assertEqual(agent.actions, [27])
        self.assertEqual(agent.rewards, [3.25])
        self.assertEqual(agent.task_ids, [11])

    def test_dqn_and_ddpg_outputs_use_the_same_28_action_head(self):
        state = np.zeros(params.num_states, dtype=np.float32)
        dqn = DQNAgent(
            params.num_states, params.num_actions, [8], device="cpu",
            batch_size=1,
        )
        self.assertEqual(tuple(dqn.policy_net(torch.tensor(state).unsqueeze(0)).shape), (1, 28))
        ddpg = ddpgModel(
            params.num_states, params.num_actions, 0.25, 1e-3, 3e-4,
            0.9, 0.005, "softmax", buffer_capacity=2, batch_size=1,
        )
        self.assertEqual(tuple(ddpg.policy(state).shape), (28,))

    def test_ppo_actor_outputs_28_logits_and_critic_outputs_one_value(self):
        agent = PPOAgent(
            params.num_states,
            params.num_actions,
            params.hidden_layers_ppo,
            activation=params.af_ppo,
        )
        state = np.zeros(params.num_states, dtype=np.float32)
        self.assertEqual(agent.policy_net(agent._to_tensor(state).unsqueeze(0)).shape, (1, 28))
        self.assertEqual(agent.value_net(agent._to_tensor(state).unsqueeze(0)).shape, (1,))


if __name__ == "__main__":
    unittest.main()
