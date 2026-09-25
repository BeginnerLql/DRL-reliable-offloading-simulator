"""Reliability-masked PPO with an independent, behavior-neutral Q sidecar."""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from agents.action_conditioned_q_critic import ActionValueCritic
from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent, masked_logits
from config.action_value_critic import ActionValueCriticConfig


class ActionValueMaskedPairPPOAgent(ReliabilityMaskedPairPPOAgent):
    """Existing Masked PPO plus an auxiliary Expected-SARSA Q learner.

    The Q learner is trained only from simulator rollouts. Its parameters do
    not participate in Actor or V(s) losses, action selection, rewards, masks,
    PPO ratios, or entropy calculations.
    """

    agent_name = "action_value_reliability_masked_pair_ppo"

    def __init__(self, *args, q_config=None, **kwargs):
        actor_hidden_layers = kwargs.get("hidden_layers", args[2] if len(args) > 2 else None)
        actor_activation = kwargs.get("activation", "tanh")
        super().__init__(*args, **kwargs)
        if self.actor_mode != "pair_scoring":
            raise ValueError("ActionValueMaskedPairPPOAgent requires pair_scoring mode")
        self.q_config = q_config or ActionValueCriticConfig()
        self.q_critic = ActionValueCritic(
            input_dim=self.policy_old.input_dim,
            num_actions=self.num_actions,
            hidden_layers=actor_hidden_layers,
            num_servers=self.num_servers,
            pair_correlations=self.pair_correlations,
            activation=actor_activation,
            device=self.device,
            config=self.q_config,
        )
        self.pending_old_policy_probabilities = {}
        self.rollout_old_policy_probabilities = []
        self.q_training_metrics = []
        self.q_calibration_samples = []
        self.q_pair_visits = np.zeros(self.num_actions, dtype=np.int64)
        self.q_pair_sum = np.zeros(self.num_actions, dtype=np.float64)
        self.q_pair_abs_td_sum = np.zeros(self.num_actions, dtype=np.float64)

    def select_action(self, state, epsilon=0.0, use_softmax=False, temperature=1.5):
        context = self.current_decision
        if context is None:
            raise RuntimeError("Action-value agent requires a prepared masked decision")
        with torch.no_grad():
            logits = self.policy_old(self._to_tensor(state).unsqueeze(0)).squeeze(0)
            distribution = Categorical(logits=masked_logits(logits, context["effective_mask"]))
            frozen_probabilities = distribution.probs.detach().cpu().numpy().astype(np.float32)
        action = super().select_action(state, epsilon, use_softmax, temperature)
        self.pending_old_policy_probabilities[int(context["task_id"])] = frozen_probabilities
        return action

    def store_transition(self, s, a, r, s_next, delta_t, done=False, task_id=None):
        probabilities = self.pending_old_policy_probabilities.pop(task_id, None)
        if probabilities is None:
            raise RuntimeError("Missing frozen masked policy distribution for Q rollout")
        super().store_transition(s, a, r, s_next, delta_t, done=done, task_id=task_id)
        if probabilities.shape != (self.num_actions,) or not np.isfinite(probabilities).all():
            raise RuntimeError("Invalid frozen masked policy distribution")
        self.rollout_old_policy_probabilities.append(probabilities)

    def train_step(self):
        n = len(self.states)
        if n == 0:
            return super().train_step()
        # Snapshot rollout data before the inherited PPO train_step clears it
        # and changes policy_old. Saved probabilities freeze pi_old exactly.
        if len(self.rollout_old_policy_probabilities) != n or len(self.effective_masks) != n:
            raise RuntimeError("Q rollout has incomplete old-policy/mask data")
        snapshot = {
            "states": np.asarray(self.states, dtype=np.float32).copy(),
            "actions": np.asarray(self.actions, dtype=np.int64).copy(),
            "rewards": np.asarray(self.rewards, dtype=np.float32).copy(),
            "next_states": np.asarray(self.next_states, dtype=np.float32).copy(),
            "delta_times": np.asarray(self.delta_times, dtype=np.float32).copy(),
            "dones": np.asarray(self.dones, dtype=bool).copy(),
            "masks": np.asarray(self.effective_masks, dtype=bool).copy(),
            "old_probs": np.asarray(self.rollout_old_policy_probabilities, dtype=np.float32).copy(),
        }
        if any(reward is None for reward in self.rewards):
            raise RuntimeError("Q rollout contains unresolved rewards")
        super().train_step()
        if self.frozen:
            return
        # A next arrival's policy is the frozen policy distribution saved on
        # that next transition. Terminal rows are masked out by `done`.
        next_probs = np.empty_like(snapshot["old_probs"])
        next_masks = np.empty_like(snapshot["masks"])
        if n > 1:
            next_probs[:-1] = snapshot["old_probs"][1:]
            next_masks[:-1] = snapshot["masks"][1:]
        next_probs[-1] = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        next_masks[-1] = True
        if n > 1 and not snapshot["dones"][:-1].any():
            if not np.allclose(snapshot["next_states"][:-1], snapshot["states"][1:], rtol=0, atol=1e-6):
                raise RuntimeError("Arrival-ordered SMDP next states do not match the next rollout states")
        metrics, samples = self.q_critic.train_rollout(
            states=snapshot["states"], actions=snapshot["actions"],
            rewards=snapshot["rewards"], next_states=snapshot["next_states"],
            delta_times=snapshot["delta_times"], dones=snapshot["dones"],
            effective_masks=snapshot["masks"], old_policy_probabilities=snapshot["old_probs"],
            next_policy_probabilities=next_probs, next_effective_masks=next_masks,
            reward_scale=self.reward_scale, gamma=self.gamma, episode=self._current_episode,
        )
        self.q_training_metrics.append(metrics)
        calibration, visit = samples
        self.q_calibration_samples.append(calibration)
        for index, selected_q, abs_td in zip(visit["actions"], visit["selected_q"], visit["abs_td_error"]):
            self.q_pair_visits[int(index)] += 1
            self.q_pair_sum[int(index)] += float(selected_q)
            self.q_pair_abs_td_sum[int(index)] += float(abs_td)

    def clear_rollout(self):
        super().clear_rollout()
        if hasattr(self, "pending_old_policy_probabilities"):
            self.pending_old_policy_probabilities.clear()
            self.rollout_old_policy_probabilities.clear()

    def q_pair_summary(self):
        rows = []
        for action in range(self.num_actions):
            visits = int(self.q_pair_visits[action])
            rows.append({
                "action_index": action,
                "pair": str(self.pairs[action]),
                "visit_count": visits,
                "mean_selected_q": float(self.q_pair_sum[action] / visits) if visits else 0.0,
                "mean_absolute_td_error": float(self.q_pair_abs_td_sum[action] / visits) if visits else 0.0,
            })
        return rows
