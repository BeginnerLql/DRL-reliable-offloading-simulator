"""Independent reliability-masked Pair PPO agent.

The Actor and Critic architectures, SMDP GAE, loss, optimizers and reward are
unchanged from PPOAgent. The only learning difference is that the same saved
per-decision effective action mask is used for sampling, old/new log-probability,
and entropy. Legacy PPOAgent remains a separate baseline.
"""
from __future__ import annotations

from itertools import combinations
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agents.ppo_agent import PPOAgent
from core.task import Task


def masked_logits(logits, effective_mask):
    mask = torch.as_tensor(effective_mask, dtype=torch.bool, device=logits.device)
    if logits.shape != mask.shape or not torch.all(mask.any(dim=-1)):
        raise ValueError("Each logit vector requires a nonempty matching effective mask")
    if not torch.isfinite(logits).all():
        raise ValueError("Actor logits must be finite before masking")
    return logits.masked_fill(~mask, -torch.inf)


def production_reliability_vector(task, env_state, pairs):
    """Evaluate candidates via Task's exact production feasibility method."""
    probe = object.__new__(Task)
    probe.env_state = env_state
    probe.computation_demand = task.computation_demand
    probe.reliability_requirement = task.reliability_requirement
    values = np.empty(len(pairs), dtype=float)
    for index, (server_j, server_k) in enumerate(pairs):
        probe.initialize_reliability_evaluation(
            env_state.get_server_by_id(server_j), env_state.get_server_by_id(server_k)
        )
        values[index] = probe.execution_reliability
    if not np.isfinite(values).all():
        raise RuntimeError("Production pair reliability must be finite")
    return values


def effective_action_mask(reliabilities, requirement):
    reliabilities = np.asarray(reliabilities, dtype=float)
    safe = reliabilities >= float(requirement)  # same as Task.reliability_satisfied
    if safe.any():
        return safe, safe.copy(), False, float(reliabilities.max())
    best = float(reliabilities.max())
    # Equal-best pairs all remain in the fallback distribution; no index bias.
    fallback = np.isclose(reliabilities, best, rtol=0.0, atol=1e-12)
    if not fallback.any():
        raise RuntimeError("Max-reliability fallback is empty")
    return safe, fallback, True, best


class ReliabilityMaskedPairPPOAgent(PPOAgent):
    """Pair PPO with matched masked sampling and PPO updates.

    ``deployment_mode='stochastic'`` is the formal policy. ``'greedy'`` is an
    explicit ablation only; the training selector always samples.
    """

    agent_name = "reliability_masked_pair_ppo"

    def __init__(self, *args, deployment_mode="stochastic", frozen=False, **kwargs):
        if kwargs.get("actor_mode") != "pair_scoring":
            raise ValueError("ReliabilityMaskedPairPPOAgent requires pair_scoring Actor")
        super().__init__(*args, **kwargs)
        if deployment_mode not in ("stochastic", "greedy"):
            raise ValueError("deployment_mode must be stochastic or greedy")
        self.deployment_mode = deployment_mode
        self.frozen = bool(frozen)
        self.pairs = list(combinations(range(1, self.num_servers + 1), 2))
        self.effective_masks = []
        self.pending_decisions = {}
        self.current_decision = None
        self.selection_archive = []
        self.update_diagnostics = []
        self._current_episode = None
        # Optional read-only diagnostic tap. Normal PPO runs leave this unset.
        self.advantage_target_observer = None
        self._pending_old_policy_probabilities = {}
        self.rollout_old_policy_probabilities = []

    def prepare_action(self, task, env_state, episode, state):
        if self.current_decision is not None:
            raise RuntimeError("Previous decision was not selected")
        values = production_reliability_vector(task, env_state, self.pairs)
        safe, effective, empty, best = effective_action_mask(values, task.reliability_requirement)
        self.current_decision = {
            "task_id": int(task.id), "episode": int(episode),
            "state": np.asarray(state, dtype=np.float32).copy(),
            "reliabilities": values, "safe_mask": safe,
            "effective_mask": effective, "safe_set_empty": empty,
            "best_achievable_reliability": best,
            "requirement": float(task.reliability_requirement),
        }
        self._current_episode = int(episode)

    def select_action(self, state, epsilon=0.0, use_softmax=False, temperature=1.5):
        context = self.current_decision
        if context is None or not np.array_equal(context["state"], np.asarray(state, dtype=np.float32)):
            raise RuntimeError("Masked selection requires matching prepared task context")
        self.current_decision = None
        with torch.no_grad():
            logits = self.policy_old(self._to_tensor(state).unsqueeze(0)).squeeze(0)
            dist = Categorical(logits=masked_logits(logits, context["effective_mask"]))
            action = int((dist.sample() if not self.frozen or self.deployment_mode == "stochastic"
                          else torch.argmax(dist.logits)).item())
            action_tensor = torch.tensor(action, dtype=torch.long, device=self.device)
            old_log_prob = float(dist.log_prob(action_tensor).item())
            entropy = float(dist.entropy().item())
            old_policy_probabilities = (
                dist.probs.detach().cpu().numpy().astype(np.float32, copy=True)
                if self.advantage_target_observer is not None else None
            )
        task_id = context["task_id"]
        if task_id in self.pending_decisions:
            raise RuntimeError("Duplicate pending task decision")
        self.pending_decisions[task_id] = (context["effective_mask"].copy(), old_log_prob, action, context["state"])
        if old_policy_probabilities is not None:
            self._pending_old_policy_probabilities[task_id] = old_policy_probabilities
        selected_reliability = float(context["reliabilities"][action])
        record = {
            "episode": context["episode"], "task_id": task_id,
            "action_index": action, "selected_pair": str(self.pairs[action]),
            "R_req": context["requirement"],
            "safe_set_size": int(context["safe_mask"].sum()),
            "safe_set_empty": bool(context["safe_set_empty"]),
            "effective_mask": context["effective_mask"].astype(int).tolist(),
            "safe_mask": context["safe_mask"].astype(int).tolist(),
            "best_achievable_reliability": context["best_achievable_reliability"],
            "reliability_deficit": max(context["requirement"] - context["best_achievable_reliability"], 0.0),
            "masked_entropy": entropy, "selected_pair_reliability": selected_reliability,
            "selected_rho": float(self.pair_correlations[action]),
            "old_log_probability": old_log_prob,
            "selected_action_safe": bool(context["safe_mask"][action]),
        }
        self.selection_archive.append(record)
        return action

    def store_transition(self, s, a, r, s_next, delta_t, done=False, task_id=None):
        if task_id not in self.pending_decisions:
            raise RuntimeError("No prepared effective mask for rollout task")
        mask, log_prob, selected, decision_state = self.pending_decisions.pop(task_id)
        if int(a) != selected or not np.array_equal(np.asarray(s, dtype=np.float32), decision_state):
            raise RuntimeError("Rollout action/state does not match its saved mask")
        try:
            super().store_transition(s, a, r, s_next, delta_t, done=done, task_id=task_id)
        except Exception:
            self.pending_decisions[task_id] = (mask, log_prob, selected, decision_state)
            raise
        self.old_log_probs[-1] = log_prob  # overwrite legacy unmasked bookkeeping
        self.effective_masks.append(mask)
        if self.advantage_target_observer is not None:
            try:
                old_probabilities = self._pending_old_policy_probabilities.pop(task_id)
            except KeyError as exc:
                raise RuntimeError("Missing decision-time masked policy distribution") from exc
            self.rollout_old_policy_probabilities.append(old_probabilities)

    def clear_rollout(self):
        super().clear_rollout()
        self.effective_masks.clear()
        self.pending_decisions.clear()
        self.current_decision = None
        self._pending_old_policy_probabilities.clear()
        self.rollout_old_policy_probabilities.clear()

    def train_step(self):
        n = len(self.states)
        if n == 0:
            if self.pending_task_rewards or self.pending_decisions:
                raise RuntimeError("Masked PPO has unresolved rollout data")
            return
        if not (n == len(self.actions) == len(self.rewards) == len(self.next_states)
                == len(self.dones) == len(self.old_log_probs) == len(self.delta_times)
                == len(self.task_ids) == len(self.effective_masks)):
            raise RuntimeError("Masked PPO rollout buffers have inconsistent lengths")
        if self.pending_decisions or self.pending_task_rewards or any(x is None for x in self.rewards):
            raise RuntimeError("Masked PPO rollout has unresolved task outcomes")
        if not all(self.task_id_to_transition_index.get(task_id) == i for i, task_id in enumerate(self.task_ids)):
            raise RuntimeError("Masked PPO task-to-mask mapping is inconsistent")
        if not all(self.task_ids[i] < self.task_ids[i+1] for i in range(n-1)) or not any(self.dones):
            raise RuntimeError("Masked PPO rollout order or terminal marker is invalid")
        if self.frozen or n < self.min_rollout:
            self.clear_rollout()
            return
        states = self._to_tensor(self.states)
        next_states = self._to_tensor(self.next_states)
        actions = torch.as_tensor(self.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device) * float(self.reward_scale)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=self.device)
        delta_times = torch.as_tensor(self.delta_times, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(self.old_log_probs, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack(self.effective_masks), dtype=torch.bool, device=self.device)
        if (not torch.isfinite(states).all() or not torch.isfinite(next_states).all()
                or not torch.isfinite(rewards).all() or not torch.isfinite(delta_times).all()
                or (delta_times < 0).any() or not masks.any(dim=1).all()
                or not masks.gather(1, actions.unsqueeze(1)).all()):
            raise RuntimeError("Masked PPO rollout contains invalid state, interval, mask or action")
        values = self.value_net(states)
        with torch.no_grad():
            next_values = self.value_net(next_states)
            gamma_k = torch.pow(torch.full_like(delta_times, float(self.gamma)), delta_times)
            nonterminal = 1.0 - dones
            deltas = rewards + gamma_k * next_values * nonterminal - values.detach()
            advantages = torch.zeros_like(rewards)
            gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            for k in reversed(range(n)):
                gae = deltas[k] + gamma_k[k] * self.gae_lambda * nonterminal[k] * gae
                advantages[k] = gae
            returns = advantages + values.detach()
            if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
                raise RuntimeError("Masked PPO GAE is non-finite")
            raw_advantages = advantages.detach().clone()
            mean = advantages.mean()
            std = advantages.std(unbiased=False)
            advantages = (advantages - mean) if std.item() < 1e-8 else (advantages - mean) / (std + 1e-8)
        if self.advantage_target_observer is not None:
            if len(self.rollout_old_policy_probabilities) != n:
                raise RuntimeError("Decision-time masked policy distributions do not match rollout")
            # Export immutable CPU copies only; PPO tensors and update inputs are untouched.
            payload = {
                "episode": int(self._current_episode or 0),
                "task_ids": np.asarray(self.task_ids, dtype=np.int64).copy(),
                "states": np.asarray(self.states, dtype=np.float32).copy(),
                "actions": np.asarray(self.actions, dtype=np.int64).copy(),
                "effective_masks": np.asarray(self.effective_masks, dtype=np.bool_).copy(),
                "old_policy_probabilities": np.asarray(
                    self.rollout_old_policy_probabilities, dtype=np.float32
                ).copy(),
                "raw_gae_advantages": raw_advantages.cpu().numpy().astype(np.float32, copy=True),
                "actor_used_advantages": advantages.detach().cpu().numpy().astype(np.float32, copy=True),
                "value_predictions": values.detach().cpu().numpy().astype(np.float32, copy=True),
                "gae_return_targets": returns.detach().cpu().numpy().astype(np.float32, copy=True),
            }
            for value in payload.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
            self.advantage_target_observer(payload)
        batch_size = min(int(self.batch_size), n)
        first_ratio_max_error = None
        min_ratio, max_ratio = math.inf, -math.inf
        entropy_sum = 0.0
        minibatches = 0
        for epoch in range(int(self.k_epochs)):
            indices = self._shuffled_indices(n)
            for start in range(0, n, batch_size):
                batch = indices[start:start+batch_size]
                logits = self.policy_net(states[batch])
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Masked PPO Actor logits became non-finite")
                dist = Categorical(logits=masked_logits(logits, masks[batch]))
                new_log_probs = dist.log_prob(actions[batch])
                entropy = dist.entropy().mean()
                log_ratio = torch.clamp(new_log_probs - old_log_probs[batch], -20.0, 20.0)
                ratios = torch.exp(log_ratio)
                if first_ratio_max_error is None:
                    first_ratio_max_error = float(torch.max(torch.abs(ratios-1.0)).item())
                min_ratio = min(min_ratio, float(ratios.min().item()))
                max_ratio = max(max_ratio, float(ratios.max().item()))
                entropy_sum += float(entropy.item())
                minibatches += 1
                surrogate_1 = ratios * advantages[batch]
                surrogate_2 = torch.clamp(ratios, 1.0-self.clip_eps, 1.0+self.clip_eps) * advantages[batch]
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean() - self.entropy_coef * entropy
                new_values = self.value_net(states[batch])
                value_loss = self.value_loss_coef * nn.MSELoss()(new_values, returns[batch].detach())
                total_loss = policy_loss + value_loss
                if not torch.isfinite(total_loss):
                    raise RuntimeError("Masked PPO loss became non-finite")
                self.optimizer_policy.zero_grad(set_to_none=True)
                self.optimizer_value.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()), self.max_grad_norm
                )
                self.optimizer_policy.step()
                self.optimizer_value.step()
        self.policy_old.load_state_dict(self.policy_net.state_dict())
        self.update_diagnostics.append({
            "episode": self._current_episode, "transitions": n,
            "first_minibatch_ratio_max_abs_error": first_ratio_max_error,
            "ratio_min": min_ratio, "ratio_max": max_ratio,
            "mean_masked_entropy": entropy_sum/minibatches,
        })
        self.clear_rollout()
