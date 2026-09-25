"""Pure policies and decision-time estimators for external reliability baselines.

This module contains no learning code. Feasibility always comes from the
production Task reliability method via ``production_reliability_vector`` and
``effective_action_mask`` in ``agents.masked_pair_ppo_agent``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from agents.masked_pair_ppo_agent import effective_action_mask, production_reliability_vector
from core.task import get_upload_time


def estimate_replica_completion_latencies(task, env_state, server_ids: Iterable[int]):
    """Estimate first-result completion times using only decision-time state.

    The estimate for replica n is

        max(B_n(t), U_{i,n}) + C_i/f_n,

    where B is the currently observable CPU backlog, U is input upload time,
    and C/f is nominal CPU service time. Existing queued work is assumed to
    drain while this task uploads; no future arrivals, realized outcomes, or
    candidate-action simulations are used. Download time is zero in the
    production task model. Returned values are measured from the task's
    decision/arrival time.
    """
    demand = float(task.computation_demand)
    if not np.isfinite(demand) or demand < 0.0:
        raise ValueError("computation_demand must be finite and non-negative")
    task_size = float(task.input_data_size_mb)
    if not np.isfinite(task_size) or task_size < 0.0:
        raise ValueError("input_data_size_mb must be finite and non-negative")
    now = float(task.env.now)
    estimates = {}
    for server_id in server_ids:
        server = env_state.get_server_by_id(int(server_id))
        if server is None:
            raise ValueError(f"unknown server_id {server_id}")
        backlog = float(env_state.get_server_backlog_time(int(server_id), now))
        frequency = float(server.processing_frequency)
        if not np.isfinite(backlog) or backlog < 0.0:
            raise ValueError("server backlog must be finite and non-negative")
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("processing_frequency must be finite and positive")
        upload = get_upload_time(task_size, server.uplink_rate_mbps)
        service = demand / frequency
        estimates[int(server_id)] = float(max(backlog, upload) + service)
    if not np.isfinite(list(estimates.values())).all():
        raise RuntimeError("decision-time latency estimates must be finite")
    return estimates


def estimate_pair_completion_latencies(task, env_state, pairs):
    """Estimate duplicated-task resolution as the first replica result."""
    server_ids = sorted({int(server_id) for pair in pairs for server_id in pair})
    per_server = estimate_replica_completion_latencies(task, env_state, server_ids)
    pair_estimates = np.asarray(
        [min(per_server[int(j)], per_server[int(k)]) for j, k in pairs], dtype=float
    )
    if not np.isfinite(pair_estimates).all():
        raise RuntimeError("pair latency estimates must be finite")
    return pair_estimates, per_server


def uniform_masked_choice(mask, rng: np.random.Generator) -> int:
    """Sample uniformly among true entries; never silently use first-index ties."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or not mask.any():
        raise ValueError("choice mask must be a nonempty one-dimensional mask")
    candidates = np.flatnonzero(mask)
    return int(rng.choice(candidates))


def max_reliability_choice(reliabilities, rng: np.random.Generator, atol=1e-12):
    values = np.asarray(reliabilities, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("reliabilities must be a nonempty finite vector")
    best = float(values.max())
    mask = np.isclose(values, best, rtol=0.0, atol=atol)
    return uniform_masked_choice(mask, rng), mask


def safe_min_latency_choice(reliabilities, requirement, latencies, rng: np.random.Generator):
    """Choose min estimated latency over safe pairs, with shared PPO fallback."""
    values = np.asarray(reliabilities, dtype=float)
    delays = np.asarray(latencies, dtype=float)
    if values.ndim != 1 or delays.shape != values.shape:
        raise ValueError("reliabilities and latencies must be matching vectors")
    if not np.isfinite(values).all() or not np.isfinite(delays).all():
        raise ValueError("reliabilities and latencies must be finite")
    safe_mask, effective_mask, empty, best = effective_action_mask(values, requirement)
    candidate_mask = effective_mask
    best_latency = float(delays[candidate_mask].min())
    min_latency_mask = candidate_mask & np.isclose(delays, best_latency, rtol=0.0, atol=1e-12)
    action = uniform_masked_choice(min_latency_mask, rng)
    return action, safe_mask, effective_mask, bool(empty), float(best), min_latency_mask


def safe_random_choice(reliabilities, requirement, rng: np.random.Generator):
    """Sample uniformly from the shared safe mask/fallback effective mask."""
    safe_mask, effective_mask, empty, best = effective_action_mask(reliabilities, requirement)
    action = uniform_masked_choice(effective_mask, rng)
    return action, safe_mask, effective_mask, bool(empty), float(best)


def deterministic_policy_rng(trial, tag: int) -> np.random.Generator:
    """Make a policy-only RNG stream independent of environment generators."""
    seed_words = [
        int(trial["Trial_ID"]),
        int(trial["Eval_Arrival_Seed"]),
        int(trial["Eval_Spatial_Seed"]),
        int(tag),
    ]
    return np.random.default_rng(np.random.SeedSequence(seed_words))


def validate_evaluation_count(assignments, episodes=20, tasks_per_episode=200):
    expected = int(episodes) * int(tasks_per_episode)
    if len(assignments) != expected:
        raise RuntimeError(f"Expected {expected} task assignments, got {len(assignments)}")
    counts = assignments.groupby("episode").size()
    if len(counts) != int(episodes) or not (counts == int(tasks_per_episode)).all():
        raise RuntimeError("Evaluation episode/task counts do not match the formal plan")
    numeric_columns = assignments.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        values = assignments[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise RuntimeError(f"Evaluation assignments contain Inf in {column}")
        # Z is deliberately unset; one replica may still be running when the
        # first-result task record is emitted, so its end/status can be absent.
        optional_at_resolution = {"Z", "Primary_End", "Primary_Status", "Backup_End", "Backup_Status"}
        if column not in optional_at_resolution and np.isnan(values).any():
            raise RuntimeError(f"Evaluation assignments contain NaN in {column}")
