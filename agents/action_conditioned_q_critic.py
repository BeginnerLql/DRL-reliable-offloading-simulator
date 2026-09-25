"""Action-conditioned Q side learner for the reliability-masked pair policy.

The Q scorer uses the same deterministic 12-dimensional state/pair feature
construction as the pair Actor, with fully independent trainable parameters.
It is deliberately not connected to PPO's policy or value losses.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math

import torch
from torch import nn
from torch.distributions import Categorical

from agents.ppo_agent import PPOPairScoringPolicyNetwork
from config.action_value_critic import ActionValueCriticConfig


def selected_action_values(all_action_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if all_action_values.ndim != 2:
        raise ValueError("all_action_values must have shape [batch, actions]")
    actions = torch.as_tensor(actions, dtype=torch.long, device=all_action_values.device).reshape(-1)
    if actions.shape[0] != all_action_values.shape[0]:
        raise ValueError("actions batch size must match all_action_values")
    if (actions < 0).any() or (actions >= all_action_values.shape[1]).any():
        raise ValueError("action index is out of range")
    return all_action_values.gather(1, actions.unsqueeze(1)).squeeze(1)


def masked_policy_expectation(q_values, policy_probabilities, effective_mask):
    """Return E_pi[Q] over a nonempty effective action support.

    Probabilities are expected to be the frozen masked behavior policy saved
    when the next-state decision was collected. A mask application and
    renormalization is retained defensively for exact fallback/effective-set
    consistency.
    """
    q_values = torch.as_tensor(q_values)
    probs = torch.as_tensor(policy_probabilities, dtype=q_values.dtype, device=q_values.device)
    mask = torch.as_tensor(effective_mask, dtype=torch.bool, device=q_values.device)
    if q_values.shape != probs.shape or q_values.shape != mask.shape or q_values.ndim != 2:
        raise ValueError("Q, policy probability, and effective mask shapes must match [B, A]")
    if not torch.isfinite(q_values).all() or not torch.isfinite(probs).all():
        raise ValueError("Q values and policy probabilities must be finite")
    if (probs < 0).any() or not mask.any(dim=1).all():
        raise ValueError("Probabilities must be nonnegative and each mask nonempty")
    supported = probs * mask.to(probs.dtype)
    normalizer = supported.sum(dim=1, keepdim=True)
    if (normalizer <= 0).any():
        raise ValueError("Frozen policy has zero mass on the effective action set")
    normalized = supported / normalizer
    return (normalized * q_values).sum(dim=1)


def smdp_discount(gamma, delta_times, *, device=None, dtype=torch.float32):
    delta_times = torch.as_tensor(delta_times, dtype=dtype, device=device)
    if not torch.isfinite(delta_times).all() or (delta_times < 0).any():
        raise ValueError("SMDP delta_times must be finite and nonnegative")
    result = torch.pow(torch.full_like(delta_times, float(gamma)), delta_times)
    if not torch.isfinite(result).all():
        raise ValueError("SMDP discounts are non-finite")
    return result


def expected_sarsa_targets(
    rewards,
    discounts,
    dones,
    next_q_values,
    next_policy_probabilities,
    next_effective_masks,
):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    discounts = torch.as_tensor(discounts, dtype=rewards.dtype, device=rewards.device)
    dones = torch.as_tensor(dones, dtype=torch.bool, device=rewards.device)
    next_q_values = torch.as_tensor(next_q_values, dtype=rewards.dtype, device=rewards.device)
    next_policy_probabilities = torch.as_tensor(
        next_policy_probabilities, dtype=rewards.dtype, device=rewards.device
    )
    next_effective_masks = torch.as_tensor(next_effective_masks, dtype=torch.bool, device=rewards.device)
    batch = rewards.shape[0]
    if rewards.ndim != 1 or discounts.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("rewards, discounts, and dones must have shape [batch]")
    if next_q_values.shape != next_policy_probabilities.shape or next_q_values.shape != next_effective_masks.shape:
        raise ValueError("next Q, policy probabilities, and masks must have matching [B, A] shapes")
    if next_q_values.shape[0] != batch:
        raise ValueError("next-state batch size must match rewards")
    continuation = masked_policy_expectation(
        next_q_values, next_policy_probabilities, next_effective_masks
    )
    targets = rewards + discounts * (~dones).to(rewards.dtype) * continuation
    if not torch.isfinite(targets).all():
        raise ValueError("Expected-SARSA Bellman targets are non-finite")
    return targets


class ActionConditionedQNetwork(PPOPairScoringPolicyNetwork):
    """Shared pair scorer Q(s,a), returning a value for every legal pair."""

    def forward(self, state):
        values = super().forward(state)
        return values


class ActionValueCritic:
    """Online Q network, frozen target network, and its independent optimizer."""

    def __init__(
        self,
        input_dim,
        num_actions,
        hidden_layers,
        num_servers,
        pair_correlations,
        *,
        activation="tanh",
        device="cpu",
        config: ActionValueCriticConfig | None = None,
    ):
        self.device = torch.device(device)
        self.config = config or ActionValueCriticConfig()
        # Preserve the caller's RNG stream so merely adding Q leaves PPO's
        # subsequent sampled actions bit-for-bit unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.initialization_seed)
            self.online = ActionConditionedQNetwork(
                input_dim, num_actions, hidden_layers, num_servers,
                pair_correlations, activation=activation,
            ).to(self.device)
        self.target = deepcopy(self.online).to(self.device)
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.update_count = 0

    @torch.no_grad()
    def soft_update_target(self):
        tau = float(self.config.target_tau)
        for target_parameter, online_parameter in zip(self.target.parameters(), self.online.parameters()):
            target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)
        for target_buffer, online_buffer in zip(self.target.buffers(), self.online.buffers()):
            target_buffer.copy_(online_buffer)

    def train_rollout(
        self,
        *,
        states,
        actions,
        rewards,
        next_states,
        delta_times,
        dones,
        effective_masks,
        old_policy_probabilities,
        next_policy_probabilities,
        next_effective_masks,
        reward_scale=1.0,
        gamma=0.9,
        episode=None,
    ):
        tensors = {
            "states": torch.as_tensor(states, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(actions, dtype=torch.long, device=self.device),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=self.device) * float(reward_scale),
            "next_states": torch.as_tensor(next_states, dtype=torch.float32, device=self.device),
            "delta_times": torch.as_tensor(delta_times, dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(dones, dtype=torch.bool, device=self.device),
            "effective_masks": torch.as_tensor(effective_masks, dtype=torch.bool, device=self.device),
            "old_policy_probabilities": torch.as_tensor(old_policy_probabilities, dtype=torch.float32, device=self.device),
            "next_policy_probabilities": torch.as_tensor(next_policy_probabilities, dtype=torch.float32, device=self.device),
            "next_effective_masks": torch.as_tensor(next_effective_masks, dtype=torch.bool, device=self.device),
        }
        n = tensors["states"].shape[0]
        if n == 0:
            return None, None
        for key, value in tensors.items():
            if value.shape[0] != n:
                raise ValueError(f"{key} batch size does not match states")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{key} contains NaN/Inf")
        if not tensors["effective_masks"].any(dim=1).all():
            raise ValueError("Every current effective mask must be nonempty")
        if not tensors["effective_masks"].gather(1, tensors["actions"].unsqueeze(1)).all():
            raise ValueError("Selected action is outside its effective mask")

        discounts = smdp_discount(gamma, tensors["delta_times"], device=self.device)
        with torch.no_grad():
            next_q = self.target(tensors["next_states"])
            targets = expected_sarsa_targets(
                tensors["rewards"], discounts, tensors["dones"], next_q,
                tensors["next_policy_probabilities"], tensors["next_effective_masks"],
            )
            before = self.online(tensors["states"])
            predictions = selected_action_values(before, tensors["actions"])
            td_errors = targets - predictions
        if not torch.isfinite(predictions).all() or not torch.isfinite(td_errors).all():
            raise FloatingPointError("Q predictions or TD errors became non-finite")

        losses = []
        for update_idx in range(int(self.config.updates_per_rollout)):
            self.optimizer.zero_grad(set_to_none=True)
            all_values = self.online(tensors["states"])
            selected = selected_action_values(all_values, tensors["actions"])
            loss = nn.functional.smooth_l1_loss(selected, targets.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError("Q Huber loss became non-finite")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError("Q gradient norm became non-finite")
            self.optimizer.step()
            self.update_count += 1
            if self.update_count % int(self.config.target_update_interval) == 0:
                self.soft_update_target()
            losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            post_values = self.online(tensors["states"])
        flat_q = post_values.detach().cpu().numpy()
        flat_target_q = next_q.detach().cpu().numpy()
        td_np = td_errors.detach().cpu().numpy()
        metrics = {
            "episode": episode,
            "transitions": n,
            "q_updates": int(self.config.updates_per_rollout),
            "q_loss": float(np_mean(losses)),
            "q_target_mean": float(targets.mean().item()),
            "q_target_std": float(targets.std(unbiased=False).item()),
            "q_pred_mean": float(predictions.mean().item()),
            "q_pred_std": float(predictions.std(unbiased=False).item()),
            "td_error_mean": float(td_errors.mean().item()),
            "td_error_std": float(td_errors.std(unbiased=False).item()),
            "td_error_abs_p95": float(torch.quantile(td_errors.abs(), 0.95).item()),
            "q_min": float(post_values.min().item()),
            "q_max": float(post_values.max().item()),
            "target_q_min": float(next_q.min().item()),
            "target_q_max": float(next_q.max().item()),
            "target_q_mean": float(next_q.mean().item()),
            "target_q_std": float(next_q.std(unbiased=False).item()),
        }
        numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
        if not all(math.isfinite(float(v)) for v in numeric):
            raise FloatingPointError("Q training metrics contain NaN/Inf")
        if max(abs(metrics["q_min"]), abs(metrics["q_max"]), abs(metrics["target_q_min"]), abs(metrics["target_q_max"])) > 1e6:
            raise FloatingPointError("Q values exceeded the 1e6 explosion guard")
        calibration = {
            "predicted_q": predictions.detach().cpu().numpy(),
            "bellman_target": targets.detach().cpu().numpy(),
        }
        visit = {
            "actions": tensors["actions"].detach().cpu().numpy(),
            "selected_q": predictions.detach().cpu().numpy(),
            "abs_td_error": np_abs(td_np),
        }
        return metrics, (calibration, visit)

    def save(self, path):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "config": asdict(self.config),
            "update_count": self.update_count,
        }, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["target"])
        self.update_count = int(payload.get("update_count", 0))


def np_mean(values):
    return float(sum(values) / len(values))


def np_abs(values):
    import numpy as np
    return np.abs(values)
