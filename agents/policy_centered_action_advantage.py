"""Read-only policy-centered action-advantage side learner.

This module is deliberately separate from PPO's Actor, V critic, optimizers,
and action selection. It uses the same pair-scoring feature builder, learns
only the actually selected action's PPO-used advantage target, and centers
scores over the decision-time effective support of the frozen masked policy.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agents.ppo_agent import PPOPairScoringPolicyNetwork


def policy_centered_advantage(raw_scores, policy_probabilities, effective_mask):
    """Center scores by the saved masked-policy expectation on effective actions.

    Unsafe scores are returned as zero so they cannot leak into downstream
    ranking or diagnostics. Probability mass is defensively restricted to the
    effective mask and renormalized, matching the saved effective support.
    """
    scores = torch.as_tensor(raw_scores)
    probs = torch.as_tensor(policy_probabilities, dtype=scores.dtype, device=scores.device)
    mask = torch.as_tensor(effective_mask, dtype=torch.bool, device=scores.device)
    if scores.ndim != 2 or probs.shape != scores.shape or mask.shape != scores.shape:
        raise ValueError("scores, probabilities and effective_mask must share [B, A] shape")
    if not torch.isfinite(scores).all() or not torch.isfinite(probs).all():
        raise ValueError("scores and policy probabilities must be finite")
    if (probs < 0).any() or not mask.any(dim=1).all():
        raise ValueError("probabilities must be nonnegative and effective masks nonempty")
    supported = probs * mask.to(probs.dtype)
    mass = supported.sum(dim=1, keepdim=True)
    if (mass <= 0).any():
        raise ValueError("Frozen policy has no probability mass on effective actions")
    normalized = supported / mass
    baseline = (normalized * scores).sum(dim=1, keepdim=True)
    return torch.where(mask, scores - baseline, torch.zeros_like(scores))


class PolicyCenteredActionAdvantage(nn.Module):
    """Independent 28-output pair scorer trained on selected-action PPO targets."""

    def __init__(
        self,
        input_dim,
        num_actions,
        hidden_layers,
        num_servers,
        pair_correlations,
        *,
        activation="tanh",
        initialization_seed=7042026,
        learning_rate=1e-3,
        batch_size=128,
        epochs_per_rollout=2,
    ):
        super().__init__()
        self.device = torch.device("cpu")
        # Creating this network cannot consume or perturb PPO's global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.network = PPOPairScoringPolicyNetwork(
                input_dim, num_actions, hidden_layers, num_servers,
                pair_correlations, activation=activation,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=float(learning_rate))
        self.batch_size = int(batch_size)
        self.epochs_per_rollout = int(epochs_per_rollout)
        if self.batch_size <= 0 or self.epochs_per_rollout <= 0:
            raise ValueError("batch_size and epochs_per_rollout must be positive")
        self.private_generator = torch.Generator(device="cpu")
        self.private_generator.manual_seed(int(initialization_seed) + 1)
        self.pairs = list(combinations(range(1, int(num_servers) + 1), 2))
        self.update_count = 0

    def forward(self, states):
        return self.network(states)

    def centered(self, states, policy_probabilities, effective_mask):
        raw = self.network(states)
        return raw, policy_centered_advantage(raw, policy_probabilities, effective_mask)

    def loss_for_selected(self, states, actions, targets, policy_probabilities, effective_mask):
        raw, centered = self.centered(states, policy_probabilities, effective_mask)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=raw.device).reshape(-1)
        target_tensor = torch.as_tensor(targets, dtype=raw.dtype, device=raw.device).reshape(-1)
        mask = torch.as_tensor(effective_mask, dtype=torch.bool, device=raw.device)
        if action_tensor.shape != (raw.shape[0],) or target_tensor.shape != (raw.shape[0],):
            raise ValueError("actions and targets must have shape [B]")
        if (action_tensor < 0).any() or (action_tensor >= raw.shape[1]).any():
            raise ValueError("selected action index is out of range")
        if not mask.gather(1, action_tensor.unsqueeze(1)).all():
            raise ValueError("selected action is outside its effective mask")
        prediction = centered.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        return F.smooth_l1_loss(prediction, target_tensor.detach()), prediction, raw, centered

    @torch.no_grad()
    def rollout_diagnostics(self, payload):
        states = torch.tensor(np.array(payload["states"], copy=True), dtype=torch.float32)
        masks = torch.tensor(np.array(payload["effective_masks"], copy=True), dtype=torch.bool)
        probabilities = torch.tensor(np.array(payload["old_policy_probabilities"], copy=True), dtype=torch.float32)
        actions = torch.tensor(np.array(payload["actions"], copy=True), dtype=torch.long)
        targets = np.asarray(payload["actor_used_advantages"], dtype=float)
        raw, centered = self.centered(states, probabilities, masks)
        selected = centered.gather(1, actions.unsqueeze(1)).squeeze(1)
        raw_np, centered_np = raw.numpy(), centered.numpy()
        mask_np = masks.numpy()
        policy_np = probabilities.numpy()
        residual = np.abs(np.sum(policy_np * centered_np, axis=1))
        rows = []
        target_std = float(np.std(targets))
        for index in range(len(actions)):
            selected_values = raw_np[index, mask_np[index]]
            centered_values = centered_np[index, mask_np[index]]
            rows.append({
                "episode": int(payload["episode"]),
                "task_id": int(payload["task_ids"][index]),
                "action_index": int(actions[index]),
                "actor_used_advantage_target": float(targets[index]),
                "selected_centered_advantage_prediction": float(selected[index]),
                "effective_set_size": int(mask_np[index].sum()),
                "raw_effective_mean": float(np.mean(selected_values)),
                "raw_effective_std": float(np.std(selected_values)),
                "centered_effective_mean": float(np.mean(centered_values)),
                "centered_effective_std": float(np.std(centered_values)),
                "centered_effective_spread": float(np.max(centered_values) - np.min(centered_values)),
                "relative_action_spread": float(np.std(centered_values) / (target_std + 1e-8)),
                "centering_residual": float(residual[index]),
            })
        return rows, {
            "mean_absolute_residual": float(np.mean(residual)),
            "max_absolute_residual": float(np.max(residual)),
            "mean_effective_set_std": float(np.mean([r["centered_effective_std"] for r in rows])),
            "median_effective_set_spread": float(np.median([r["centered_effective_spread"] for r in rows])),
            "collapse_rate": float(np.mean([r["centered_effective_spread"] < 1e-6 for r in rows])),
        }

    def train_rollout(self, payload):
        """Fit selected actions to actor-used targets after PPO has fully updated."""
        states = torch.tensor(np.array(payload["states"], copy=True), dtype=torch.float32)
        actions = torch.tensor(np.array(payload["actions"], copy=True), dtype=torch.long)
        targets = torch.tensor(np.array(payload["actor_used_advantages"], copy=True), dtype=torch.float32)
        probs = torch.tensor(np.array(payload["old_policy_probabilities"], copy=True), dtype=torch.float32)
        masks = torch.tensor(np.array(payload["effective_masks"], copy=True), dtype=torch.bool)
        n = len(actions)
        losses = []
        grad_norms = []
        for _ in range(self.epochs_per_rollout):
            permutation = torch.randperm(n, generator=self.private_generator)
            for start in range(0, n, self.batch_size):
                batch = permutation[start:start + self.batch_size]
                self.optimizer.zero_grad(set_to_none=True)
                loss, _, _, _ = self.loss_for_selected(
                    states[batch], actions[batch], targets[batch], probs[batch], masks[batch]
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Side learner Huber loss became non-finite")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError("Side learner gradient became non-finite")
                self.optimizer.step()
                losses.append(float(loss.detach()))
                grad_norms.append(float(torch.as_tensor(grad_norm).detach()))
                self.update_count += 1
        return {
            "episode": int(payload["episode"]), "transitions": n,
            "optimizer_steps": len(losses), "loss_mean": float(np.mean(losses)),
            "loss_last": float(losses[-1]), "gradient_norm_mean": float(np.mean(grad_norms)),
            "target_std": float(targets.std(unbiased=False)),
            "update_count": int(self.update_count),
        }

    def wiring_diagnostics(self, states):
        """Check pair feature diversity, ordering, mapping, and feature gradients."""
        if isinstance(states, torch.Tensor):
            state = states.detach().to(dtype=torch.float32).clone()
        else:
            state = torch.tensor(np.array(states, dtype=np.float32, copy=True))
        state = state.reshape(-1, self.network.input_dim)
        if len(state) == 0:
            raise ValueError("at least one state is required")
        features = self.network.build_pair_features(state)
        outputs = self.network(state)
        permutation = torch.arange(features.shape[1] - 1, -1, -1)
        permuted_outputs = self.network.scorer(features[:, permutation].reshape(-1, features.shape[-1]))
        permuted_outputs = permuted_outputs.reshape(len(state), -1)
        permutation_error = float(torch.max(torch.abs(permuted_outputs - outputs[:, permutation])).item())
        feature_input = features.detach().clone().requires_grad_(True)
        score_sum = self.network.scorer(feature_input.reshape(-1, feature_input.shape[-1])).square().sum()
        gradients = torch.autograd.grad(score_sum, feature_input)[0]
        pair_indices = self.network.pair_indices.detach().cpu().numpy()
        mapped_pairs = [(int(a) + 1, int(b) + 1) for a, b in pair_indices]
        return {
            "pair_feature_std_mean": float(features.std(dim=1, unbiased=False).mean().item()),
            "initial_or_current_output_spread_mean": float((outputs.max(dim=1).values - outputs.min(dim=1).values).mean().item()),
            "permutation_max_abs_error": permutation_error,
            "permutation_pass": bool(permutation_error <= 1e-7),
            "pair_mapping_matches_combinations": mapped_pairs == self.pairs,
            "selected_pair_mapping_sample": str(self.pairs[int(torch.argmax(outputs[0]).item())]),
            "pair_feature_gradient_norm": float(gradients.norm().item()),
            "pair_feature_gradient_nonzero": bool(gradients.norm().item() > 0.0),
        }
