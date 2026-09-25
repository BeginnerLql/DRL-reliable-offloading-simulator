import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Categorical

from agents.action_conditioned_q_critic import ActionValueCritic, expected_sarsa_targets, masked_policy_expectation, selected_action_values, smdp_discount
from agents.action_value_masked_ppo_agent import ActionValueMaskedPairPPOAgent
from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent, effective_action_mask, masked_logits
from config.action_value_critic import ActionValueCriticConfig


class ActionValueCriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rho = np.linspace(0.01, 0.8, 28)
        cls.agent_kwargs = dict(num_states=35, num_actions=28, hidden_layers=[16, 8], device="cpu", gamma=.9,
            actor_lr=3e-4, critic_lr=1e-3, clip_eps=.2, k_epochs=2, batch_size=8, entropy_coef=.01,
            reward_scale=1., gae_lambda=.95, value_loss_coef=.5, max_grad_norm=.5, activation="tanh",
            min_rollout=1, minibatch_seed=77, actor_mode="pair_scoring", num_servers=8, pair_correlations=cls.rho)

    def critic(self, seed=42):
        return ActionValueCritic(35, 28, [16, 8], 8, self.rho, device="cpu",
            config=ActionValueCriticConfig(initialization_seed=seed))

    def test_01_q_output_shape(self):
        self.assertEqual(tuple(self.critic().online(torch.zeros(4, 35)).shape), (4, 28))

    def test_02_selected_action_gather(self):
        q = torch.arange(84.).reshape(3, 28)
        torch.testing.assert_close(selected_action_values(q, [0, 17, 27]), torch.tensor([0., 45., 83.]))

    def test_03_masked_policy_expectation(self):
        got = masked_policy_expectation([[1., 4., 10.]], [[.2, .3, .5]], [[False, True, True]])
        self.assertAlmostEqual(got.item(), 7.75)

    def test_04_empty_safe_fallback_mask(self):
        safe, effective, empty, best = effective_action_mask([.8, .9, .9], .99)
        self.assertTrue(empty); self.assertEqual(best, .9)
        np.testing.assert_array_equal(safe, [False, False, False])
        np.testing.assert_array_equal(effective, [False, True, True])
        got = masked_policy_expectation([[2., 4., 8.]], [[.1, .6, .3]], [effective])
        self.assertAlmostEqual(got.item(), 5.3333333, places=5)

    def test_05_terminal_target_equals_reward(self):
        got = expected_sarsa_targets([3.25], [.81], [True], [[10., 20.]], [[.5, .5]], [[True, True]])
        torch.testing.assert_close(got, torch.tensor([3.25]))

    def test_06_discount_reuses_gamma_power_delta_t(self):
        torch.testing.assert_close(smdp_discount(.9, [0., .5, 2.]), torch.tensor([1., .9**.5, .81]))

    def test_07_saved_old_policy_controls_expectation(self):
        target = expected_sarsa_targets([0.], [1.], [False], [[2., 20.]], [[1., 0.]], [[True, True]])
        self.assertEqual(target.item(), 2.)

    def test_08_soft_target_update(self):
        q = self.critic()
        with torch.no_grad():
            for p in q.target.parameters(): p.zero_()
            for p in q.online.parameters(): p.fill_(2.)
        q.soft_update_target()
        for p in q.target.parameters(): torch.testing.assert_close(p, torch.full_like(p, 2*q.config.target_tau))

    def test_09_q_gradient_does_not_reach_actor(self):
        agent = ActionValueMaskedPairPPOAgent(**self.agent_kwargs)
        selected_action_values(agent.q_critic.online(torch.randn(2, 35)), [1, 3]).sum().backward()
        self.assertTrue(all(p.grad is None for p in agent.policy_net.parameters()))
        self.assertTrue(any(p.grad is not None for p in agent.q_critic.online.parameters()))

    def test_10_legacy_and_augmented_masked_actions_match(self):
        torch.manual_seed(120); legacy = ReliabilityMaskedPairPPOAgent(**self.agent_kwargs)
        torch.manual_seed(120); sidecar = ActionValueMaskedPairPPOAgent(**self.agent_kwargs)
        for key, val in legacy.policy_old.state_dict().items(): torch.testing.assert_close(val, sidecar.policy_old.state_dict()[key], rtol=0, atol=0)
        state = np.linspace(-.2, .8, 35, dtype=np.float32); mask = np.ones(28, bool)
        context = {"task_id": 1, "state": state, "effective_mask": mask, "requirement": .9, "episode": 1,
                   "reliabilities": np.ones(28), "safe_mask": mask, "safe_set_empty": False,
                   "best_achievable_reliability": 1.}
        legacy.current_decision = dict(context); sidecar.current_decision = dict(context)
        torch.manual_seed(988); a = legacy.select_action(state)
        torch.manual_seed(988); b = sidecar.select_action(state)
        self.assertEqual(a, b)
        with torch.no_grad():
            self.assertEqual(int(legacy.policy_old(torch.tensor(state)).argmax()), int(sidecar.policy_old(torch.tensor(state)).argmax()))

    def test_11_training_receives_rollouts_only_not_counterfactual_files(self):
        q = self.critic(); sig = inspect.signature(q.train_rollout)
        self.assertFalse(any("counterfactual" in name or name.startswith("q_h") for name in sig.parameters))
        with patch("pandas.read_csv", side_effect=AssertionError("offline validation CSV must not be loaded for training")):
            metrics, _ = q.train_rollout(states=np.zeros((2,35),np.float32), actions=[0,1], rewards=[1.,2.],
                next_states=np.zeros((2,35),np.float32), delta_times=[1.,1.], dones=[False,True],
                effective_masks=np.ones((2,28),bool), old_policy_probabilities=np.full((2,28),1/28,np.float32),
                next_policy_probabilities=np.full((2,28),1/28,np.float32), next_effective_masks=np.ones((2,28),bool))
        self.assertTrue(np.isfinite(metrics["q_loss"]))

    def test_12_checkpoint_round_trip(self):
        q = self.critic(2026); state = torch.randn(3,35); before = q.online(state).detach().clone()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"q.pt"; q.save(path); restored=self.critic(4); restored.load(path)
        torch.testing.assert_close(before, restored.online(state), rtol=0, atol=0)

    def test_13_nan_inf_rejected(self):
        with self.assertRaises(ValueError): masked_policy_expectation([[float("nan")]], [[1.]], [[True]])
        with self.assertRaises(ValueError): smdp_discount(.9, [float("inf")])

    def test_14_seeded_initialization_reproducible(self):
        a,b=self.critic(2026),self.critic(2026); state=torch.randn(4,35)
        torch.testing.assert_close(a.online(state),b.online(state),rtol=0,atol=0)

    def test_actor_policy_ratio_is_one_before_actor_update(self):
        agent=ActionValueMaskedPairPPOAgent(**self.agent_kwargs); state=torch.randn(2,35); mask=torch.ones(2,28,dtype=torch.bool)
        old=Categorical(logits=masked_logits(agent.policy_old(state),mask)).probs
        new=Categorical(logits=masked_logits(agent.policy_net(state),mask)).probs
        torch.testing.assert_close(old,new,rtol=0,atol=0); torch.testing.assert_close(old[:,0]/new[:,0],torch.ones(2),atol=1e-6,rtol=0)


if __name__ == "__main__": unittest.main()
