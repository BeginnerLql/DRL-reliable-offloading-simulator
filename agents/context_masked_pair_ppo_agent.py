"""Versioned fixes; legacy masked Pair PPO remains reproducible.

Event-interval credit assigns every resolved reward to its actual time interval,
including rewards of tasks decided earlier. It discounts within the interval
once, then uses existing elapsed-time PPO GAE for subsequent intervals.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent


def event_interval_rewards(decision_times, terminal_time, outcomes, gamma):
    """Aggregate (task_id, reward, resolution_time); final endpoint is included.

    At an arrival boundary, outcomes belong to the interval starting there.
    Equal-time decisions create empty zero-duration intervals. Delayed reward
    bookkeeping does not change the event's timestamp or ownership.
    """
    times = np.asarray(decision_times, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all()
            or (np.diff(times) < 0).any() or not math.isfinite(terminal_time)
            or terminal_time < times[-1] or not 0 < gamma <= 1):
        raise ValueError("Invalid decision times, terminal time or discount")
    rewards = np.zeros(len(times), dtype=float)
    seen = set()
    for task_id, reward, timestamp in outcomes:
        if task_id in seen:
            raise ValueError("Duplicate resolved task reward")
        seen.add(task_id)
        if (not math.isfinite(reward) or not math.isfinite(timestamp)
                or timestamp < times[0] or timestamp > terminal_time):
            raise ValueError("Outcome must be finite and inside the episode")
        index = int(np.searchsorted(times, timestamp, side="right") - 1)
        rewards[index] += gamma ** (timestamp - times[index]) * reward
    return rewards


class ContextMaskedPairPPOAgent(ReliabilityMaskedPairPPOAgent):
    agent_name = "context_masked_pair_ppo_v2"

    def __init__(self, *args, credit_mode="event_interval",
                 gradient_clipping="independent", **kwargs):
        kwargs.setdefault("actor_mode", "pair_context")
        if credit_mode not in {"event_interval", "origin_task"}:
            raise ValueError("credit_mode must be event_interval or origin_task")
        if gradient_clipping not in {"independent", "joint"}:
            raise ValueError("gradient_clipping must be independent or joint")
        super().__init__(*args, **kwargs)
        self.credit_mode = credit_mode
        self.gradient_clipping = gradient_clipping
        self.decision_times = {}
        self.outcome_records = {}
        self.credit_diagnostics = []
        self.gradient_diagnostics = []

    def prepare_action(self, task, env_state, episode, state):
        if task.id in self.decision_times:
            raise RuntimeError("Duplicate task decision time")
        super().prepare_action(task, env_state, episode, state)
        self.decision_times[task.id] = float(task.env.now)

    def record_task_outcome(self, task_id, reward, timestamp):
        if task_id in self.outcome_records or task_id not in self.decision_times:
            raise RuntimeError("Duplicate or unknown task outcome")
        if (not math.isfinite(timestamp) or not math.isfinite(reward)
                or timestamp < self.decision_times[task_id]):
            raise ValueError("Invalid task outcome")
        # Preserve origin-task lifecycle bookkeeping and pending-reward checks.
        super().assign_task_reward(task_id, reward)
        self.outcome_records[task_id] = (task_id, float(reward), float(timestamp))

    def train_step(self):
        if self.states:
            if (set(self.outcome_records) != set(self.task_ids)
                    or set(self.decision_times) != set(self.task_ids)
                    or any(r is None for r in self.rewards)):
                raise RuntimeError("Incomplete episode outcome/time records")
            times = np.array([self.decision_times[i] for i in self.task_ids])
            if not np.allclose(np.diff(times), np.asarray(self.delta_times[:-1]),
                               rtol=0, atol=1e-10):
                raise RuntimeError("Decision times disagree with rollout intervals")
            if any(self.dones[:-1]) or not self.dones[-1]:
                raise RuntimeError("Expected one terminal at end of episode")
            terminal = float(times[-1] + self.delta_times[-1])
            outcomes = list(self.outcome_records.values())
            interval = event_interval_rewards(times, terminal, outcomes, self.gamma)
            expected = sum(self.gamma ** (t-times[0]) * r for _, r, t in outcomes)
            actual = float(np.dot(self.gamma ** (times-times[0]), interval))
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10):
                raise RuntimeError("Event-interval return conservation failed")
            self.credit_diagnostics.append({
                "episode": self._current_episode, "credit_mode": self.credit_mode,
                "outcomes": len(outcomes), "event_return": expected,
                "interval_return": actual, "return_residual": abs(actual-expected),
                "task_reward_sum": float(sum(self.rewards)),
            })
            if self.credit_mode == "event_interval":
                self.rewards[:] = interval.tolist()
        super().train_step()

    def _clip_gradients(self):
        if self.gradient_clipping == "joint":
            return super()._clip_gradients()
        actor_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        critic_norm = torch.nn.utils.clip_grad_norm_(
            self.value_net.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        self.gradient_diagnostics.append({
            "episode": self._current_episode,
            "actor_norm_before": float(actor_norm),
            "critic_norm_before": float(critic_norm),
        })

    def clear_rollout(self):
        super().clear_rollout()
        self.decision_times.clear()
        self.outcome_records.clear()
