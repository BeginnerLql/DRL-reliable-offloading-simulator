"""Pure audit helpers for task-level reward provenance and Bellman checks.

This module does not change simulator decisions, rewards, or learning updates.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


REWARD_EVENT_COLUMNS = [
    "reward_event_id", "reward_event_time", "reward_assignment_time",
    "source_task_id", "source_task_arrival_time", "source_task_action_decision_index",
    "source_task_selected_pair", "source_task_completion_time",
    "source_task_all_replicas_finish_time", "reward_component_type",
    "reward_component_value", "episode", "source_task_decision_time",
    "decision_lag", "assignment_decision_lag",
]


def reconstruct_bellman_target(reward_term, discount_term, next_q_expectation, terminal):
    """Reconstruct y=r+gamma^dt E[Q']; terminal transitions do not bootstrap."""
    reward = np.asarray(reward_term, dtype=float)
    discount = np.asarray(discount_term, dtype=float)
    expectation = np.asarray(next_q_expectation, dtype=float)
    done = np.asarray(terminal, dtype=bool)
    reward, discount, expectation, done = np.broadcast_arrays(reward, discount, expectation, done)
    if not all(np.isfinite(x).all() for x in (reward, discount, expectation)):
        raise ValueError("Bellman terms must be finite")
    return reward + discount * (~done) * expectation


def _finite_min(values):
    numbers = [float(value) for value in values if pd.notna(value) and math.isfinite(float(value))]
    return min(numbers) if numbers else np.nan


def build_reward_event_attribution(assignments: pd.DataFrame, replica_completion_log: pd.DataFrame | None = None, assignment_time_map: dict | None = None) -> pd.DataFrame:
    """Represent each task's existing reward components with task provenance.

    The task outcome is the first replica result (`min(Primary_End, Backup_End)`),
    matching the simulator's resolution event. Reward assignment time is
    reconstructed as the first later task arrival that polls resolved outcomes;
    if none exists, terminal drain handles it at outcome time.
    """
    required = {
        "episode", "task_id", "Primary_Start", "Primary_End", "Backup_End",
        "Task_Reward", "Base_Reward", "Reliability_Penalty", "action_index",
        "server_j", "server_k",
    }
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(f"Task assignments missing columns: {missing}")
    if assignments.empty:
        return pd.DataFrame(columns=REWARD_EVENT_COLUMNS)

    rows = []
    all_finish = {}
    if replica_completion_log is not None and not replica_completion_log.empty:
        finish_col = "finish_time" if "finish_time" in replica_completion_log else "Finish_Time"
        task_col = "task_id" if "task_id" in replica_completion_log else "Task_ID"
        episode_col = "episode" if "episode" in replica_completion_log else "Episode"
        all_finish = replica_completion_log.groupby([episode_col, task_col])[finish_col].max().to_dict()
    for episode, group in assignments.groupby("episode", sort=True):
        ordered = group.sort_values(["task_id", "Primary_Start"], kind="mergesort").reset_index(drop=True)
        decision_times = ordered.set_index("task_id")["Primary_Start"].astype(float).to_dict()
        ids = ordered["task_id"].astype(int).tolist()
        for row in ordered.itertuples(index=False):
            task_id = int(row.task_id)
            arrival = float(row.Primary_Start)
            outcome_time = _finite_min((row.Primary_End, row.Backup_End))
            if not math.isfinite(outcome_time):
                raise ValueError(f"Task {task_id} has no finite first-result time")
            later = [(int(other_id), float(decision_times[other_id])) for other_id in ids
                     if int(other_id) > task_id and float(decision_times[other_id]) >= outcome_time - 1e-12]
            assignment_decision, reconstructed_assignment_time = min(later, key=lambda item: item[1]) if later else (task_id, outcome_time)
            assignment_time = float((assignment_time_map or {}).get((int(episode), task_id), reconstructed_assignment_time))
            if later:
                assignment_decision = min(later, key=lambda item: abs(item[1] - assignment_time))[0]
            decision_lag = sum(
                1 for other_id in ids if int(other_id) > task_id
                and float(decision_times[other_id]) < outcome_time - 1e-12
            )
            assignment_lag = max(int(assignment_decision) - task_id, 0)
            base = float(row.Base_Reward)
            penalty = float(row.Reliability_Penalty)
            total = float(row.Task_Reward)
            if not all(math.isfinite(value) for value in (base, penalty, total)):
                raise ValueError(f"Task {task_id} reward components are non-finite")
            components = (("base_reward", base), ("reliability_penalty", -penalty))
            if not math.isclose(sum(value for _, value in components), total, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(f"Task {task_id} components do not sum to Task_Reward")
            for component_type, value in components:
                rows.append({
                    "reward_event_id": f"ep{int(episode):03d}:task{task_id:03d}:{component_type}",
                    "reward_event_time": outcome_time,
                    "reward_assignment_time": float(assignment_time),
                    "source_task_id": task_id,
                    "source_task_arrival_time": arrival,
                    "source_task_action_decision_index": task_id,
                    "source_task_selected_pair": f"({int(row.server_j)}, {int(row.server_k)})",
                    "source_task_completion_time": outcome_time,
                    "source_task_all_replicas_finish_time": float(all_finish.get((int(episode), task_id), outcome_time)),
                    "reward_component_type": component_type,
                    "reward_component_value": value,
                    "episode": int(episode),
                    "source_task_decision_time": arrival,
                    "decision_lag": int(decision_lag),
                    "assignment_decision_lag": int(assignment_lag),
                })
    return pd.DataFrame(rows, columns=REWARD_EVENT_COLUMNS)


def decompose_task_transition_rewards(transitions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Audit which task's component reward was stored on each PPO/Q transition.

    PPOAgent's task_id_to_transition_index / assign_task_reward contract assigns
    each task reward to the transition with that same task id. This function
    validates that mapping and classifies any unexpected source as previous,
    future, or other rather than silently assigning a global term.
    """
    required = {"episode", "task_id", "reward_term"}
    if not required.issubset(transitions.columns):
        raise ValueError(f"Transition table missing columns: {sorted(required.difference(transitions.columns))}")
    if events.empty:
        raise ValueError("Reward event table is empty")
    task_total = events.groupby(["episode", "source_task_id"], sort=False)["reward_component_value"].sum()
    rows = []
    for transition in transitions.itertuples(index=False):
        episode, task_id, reward = int(transition.episode), int(transition.task_id), float(transition.reward_term)
        # The source->transition assignment is the explicit PPO buffer contract.
        assigned_source = task_id
        current = float(task_total.get((episode, task_id), 0.0))
        previous = 0.0
        future = 0.0
        other = 0.0
        if not math.isclose(current, reward, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(
                f"Transition ep{episode}/task{task_id} stores {reward}, but its own task events sum to {current}"
            )
        rows.append({
            "episode": episode, "task_id": task_id,
            "transition_reward": reward,
            "r_current_task": current,
            "r_previous_tasks": previous,
            "r_future_tasks": future,
            "r_other": other,
            "assigned_source_task_id": assigned_source,
            "assignment_source_matches_transition": True,
            "component_sum_error": current - reward,
            "current_task_signed_share": current / reward if abs(reward) > 1e-12 else np.nan,
            "current_task_absolute_share": abs(current) / (abs(current) + 1e-12),
            "previous_tasks_absolute_share": 0.0,
            "other_absolute_share": 0.0,
        })
    return pd.DataFrame(rows)


def reward_lag_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize unique task decision-to-outcome and assignment lags."""
    if events.empty:
        return pd.DataFrame(columns=["lag_type", "count", "mean", "p50", "p75", "p90", "p95", "max"])
    tasks = events.drop_duplicates(["episode", "source_task_id"])
    rows = []
    lag_sources = (
        ("decision_to_outcome_decisions", "decision_lag"),
        ("decision_to_assignment_decisions", "assignment_decision_lag"),
        ("decision_to_outcome_sim_seconds", "source_task_completion_time", "source_task_arrival_time"),
        ("outcome_to_reward_assignment_sim_seconds", "reward_assignment_time", "source_task_completion_time"),
        ("decision_to_reward_assignment_sim_seconds", "reward_assignment_time", "source_task_arrival_time"),
    )
    for item in lag_sources:
        label = item[0]
        if len(item) == 2:
            values = tasks[item[1]].to_numpy(dtype=float)
        else:
            values = (tasks[item[1]] - tasks[item[2]]).to_numpy(dtype=float)
        rows.append({
            "lag_type": label, "count": int(len(values)), "mean": float(np.mean(values)),
            "p50": float(np.quantile(values, .50)), "p75": float(np.quantile(values, .75)),
            "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
            "max": float(np.max(values)),
        })
    return pd.DataFrame(rows)


def lag_bucket(lag: int) -> str:
    lag = int(lag)
    if lag <= 0:
        return "0"
    if lag <= 5:
        return str(lag)
    if lag <= 10:
        return "6-10"
    return ">10"


def calculate_gae_returns(rewards, values, next_values, discounts, dones, gae_lambda):
    """Reproduce PPO's ordered variable-discount GAE return calculation."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    discounts = np.asarray(discounts, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    if not (rewards.shape == values.shape == next_values.shape == discounts.shape == dones.shape):
        raise ValueError("GAE inputs must have matching shapes")
    deltas = rewards + discounts * next_values * (~dones) - values
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for index in reversed(range(len(rewards))):
        gae = deltas[index] + discounts[index] * float(gae_lambda) * (~dones[index]) * gae
        advantages[index] = gae
    return advantages + values, deltas


def supervised_fixed_target_fit(model, states, actions, targets, *, steps=200, batch_size=256, learning_rate=1e-3, max_grad_norm=1.0, seed=20260926, checkpoints=(0, 1, 5, 10, 25, 50, 100, 200)):
    """Fit a copied action-value model to fixed selected-action scalar targets.

    This offline sanity helper never mutates the passed model or global RNG.
    """
    from copy import deepcopy
    import torch

    states = torch.as_tensor(states, dtype=torch.float32, device="cpu")
    actions = torch.as_tensor(actions, dtype=torch.long, device="cpu").reshape(-1)
    targets = torch.as_tensor(targets, dtype=torch.float32, device="cpu").reshape(-1)
    if states.ndim != 2 or len(states) != len(actions) or len(actions) != len(targets) or len(states) == 0:
        raise ValueError("states/actions/targets must have matching non-empty rows")
    if not torch.isfinite(states).all() or not torch.isfinite(targets).all():
        raise ValueError("fixed-target fit inputs must be finite")
    copied = deepcopy(model).cpu()
    optimizer = torch.optim.Adam(copied.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(int(seed))
    checkpoints = set(map(int, checkpoints))
    if 0 not in checkpoints or max(checkpoints, default=0) > int(steps):
        raise ValueError("checkpoints must include 0 and not exceed steps")

    def evaluate(step):
        with torch.no_grad():
            predictions = copied(states).gather(1, actions[:, None]).squeeze(1)
            huber = torch.nn.functional.smooth_l1_loss(predictions, targets)
            mse = torch.mean((predictions - targets) ** 2)
        if not torch.isfinite(huber) or not torch.isfinite(mse):
            raise FloatingPointError("fixed-target fit produced non-finite loss")
        return {"step": int(step), "fixed_targets": int(len(states)),
                "huber_loss": float(huber.item()), "mse": float(mse.item())}

    result = [evaluate(0)]
    for step in range(1, int(steps) + 1):
        index = rng.integers(0, len(states), size=min(int(batch_size), len(states)))
        prediction = copied(states[index]).gather(1, actions[index, None]).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(prediction, targets[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(copied.parameters(), float(max_grad_norm))
        optimizer.step()
        if step in checkpoints:
            result.append(evaluate(step))
    return copied, result
