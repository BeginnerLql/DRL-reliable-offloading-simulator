"""Causal long-horizon diagnostics for the formal Masked Pair PPO setting.

Counterfactual branches replay the deterministic prefix from episode start
instead of deepcopying live SimPy generators (which Python cannot clone). Each
branch is accepted only when its pre-intervention observable/runtime snapshot,
exogenous RNG digests, task arrivals, hazard realization, and action prefix
exactly match the reference replay. Only the target action is overridden; the
same frozen stochastic Masked PPO continues afterward.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import core.main_loop as main_loop_module
from agents.masked_pair_ppo_agent import (
    ReliabilityMaskedPairPPOAgent, effective_action_mask, masked_logits,
)
from config.params import params
from diagnostics.evaluate_reliability_masked_policy import (
    ArrivalTraceLoop, TrackingEnvironmentState,
)
from diagnostics.external_reliability_baselines import estimate_pair_completion_latencies
from diagnostics.run_masked_pair_ppo_10seed import OUT as PPO_OUT, action_seed, formal_spec
from Project_main import build_pair_correlations
from tools.pair_policy_diagnostics import TASK_ASSIGNMENT_COLUMNS
from tools.paired_ppo_experiment import _agent_kwargs, scoped_environment_seeds, sha256_file

OUT = ROOT / "diagnostics/results/long_horizon_coupling"
HORIZONS = (1, 5, 10, 20, 50)
STATE_COUNT = 1000


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def stable_digest(value):
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rng_digest():
    np_state = np.random.get_state()
    return {
        "python": stable_digest(random.getstate()),
        "numpy_global": stable_digest((np_state[0], np_state[1], np_state[2], np_state[3], np_state[4])),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
    }


def capture_external_rng_state(loop):
    """Capture streams used by arrivals, spatial risk, and policy sampling."""
    return {
        "arrival": copy.deepcopy(loop.arrival_rng.bit_generator.state),
        "spatial": copy.deepcopy(loop.spatial_risk_rng.bit_generator.state),
        "python": random.getstate(),
        "numpy_global": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
    }


def restore_external_rng_state(loop, state):
    """Restore RNG state for deterministic replay tests and isolated branches."""
    loop.arrival_rng.bit_generator.state = copy.deepcopy(state["arrival"])
    loop.spatial_risk_rng.bit_generator.state = copy.deepcopy(state["spatial"])
    random.setstate(state["python"])
    np.random.set_state(copy.deepcopy(state["numpy_global"]))
    torch.set_rng_state(state["torch_cpu"].clone())


def _request_signature(request):
    return {
        "priority": _jsonable(getattr(request, "priority", None)),
        "time": _jsonable(getattr(request, "time", None)),
        "key": _jsonable(getattr(request, "key", None)),
        "preempt": bool(getattr(request, "preempt", False)),
    }


def queue_hazard_snapshot(loop, task, env_state, state):
    """Serialize the complete observable CPU queue/hazard context at arrival."""
    servers = []
    for sid in sorted(env_state.servers):
        info = env_state.servers[sid]
        server = info["server_object"]
        waiting = [
            {
                "task_id": int(item["task"].id),
                "selection": str(item["selection"]),
                "service_time": float(item["service_time"]),
            }
            for item in info["waiting_replicas"]
        ]
        running = info["running_replica"]
        running_value = None if running is None else {
            "task_id": int(running["task"].id),
            "selection": str(running["selection"]),
            "service_time": float(running["service_time"]),
            "service_start_time": float(running["service_start_time"]),
        }
        servers.append({
            "server_id": int(sid),
            "backlog_time": float(env_state.get_server_backlog_time(int(sid), task.env.now)),
            "waiting": waiting,
            "running": running_value,
            "resource_queue": [_request_signature(req) for req in server.queue.queue],
            "resource_users": [_request_signature(req) for req in server.queue.users],
        })
    pending = []
    for task_id in list(loop.pendingList):
        item = env_state.get_task_by_id(int(task_id))
        pending.append({
            "task_id": int(task_id),
            "primary_started": _jsonable(item.primaryStarted),
            "primary_finished": _jsonable(item.primaryFinished),
            "backup_started": _jsonable(item.backupStarted),
            "backup_finished": _jsonable(item.backupFinished),
            "resolved": bool(item.resolved),
            "all_replicas_finished": bool(item.all_replicas_finished),
        })
    calendar = []
    for when, priority, event_id, event in loop.env._queue:
        calendar.append({
            "time": float(when), "priority": int(priority), "event_id": int(event_id),
            "event_type": type(event).__name__,
            "callbacks": None if event.callbacks is None else len(event.callbacks),
            "triggered": bool(event.triggered),
            "processed": bool(event.callbacks is None),
        })
    effective = env_state.effective_failure_rates
    hazards = {
        "effective_failure_rates": {str(k): float(v) for k, v in sorted(effective.items())},
        "spatial_risk_field": None if env_state.spatial_risk_field is None
            else np.asarray(env_state.spatial_risk_field, dtype=float).tolist(),
        "distance_matrix": None if env_state.spatial_distance_matrix is None
            else np.asarray(env_state.spatial_distance_matrix, dtype=float).tolist(),
        "correlation_matrix": None if env_state.spatial_correlation_matrix is None
            else np.asarray(env_state.spatial_correlation_matrix, dtype=float).tolist(),
    }
    return {
        "episode": int(loop.this_episode), "task_id": int(task.id),
        "simulation_time": float(task.env.now),
        "state": np.asarray(state, dtype=np.float32).tolist(),
        "task_requirement": float(task.reliability_requirement),
        "task_computation_demand": float(task.computation_demand),
        "task_input_size_mb": float(task.input_data_size_mb),
        "task_counter": int(loop.taskCounter), "pending_task_ids": list(map(int, loop.pendingList)),
        "servers": servers, "pending_tasks": pending, "hazards": hazards,
        "simpy_calendar": sorted(calendar, key=lambda x: (x["time"], x["priority"], x["event_id"])),
        "arrival_rng_state": copy.deepcopy(loop.arrival_rng.bit_generator.state),
        "spatial_rng_state": copy.deepcopy(loop.spatial_risk_rng.bit_generator.state),
        "rng_digest": _rng_digest(),
        "arrival_prefix": [float(x) for x in loop.interarrival_trace],
    }


def snapshot_signature(snapshot):
    return stable_digest(snapshot)


def queue_metrics(env_state, now):
    lengths, backlogs = [], []
    for sid in sorted(env_state.servers):
        info = env_state.servers[sid]
        length = len(info["waiting_replicas"]) + int(info["running_replica"] is not None)
        lengths.append(length)
        backlogs.append(float(env_state.get_server_backlog_time(int(sid), now)))
    return {
        "total_queue_length": int(sum(lengths)),
        "maximum_server_queue_length": int(max(lengths, default=0)),
        "total_backlog_seconds": float(sum(backlogs)),
        "maximum_server_backlog_seconds": float(max(backlogs, default=0.0)),
    }


class CouplingEnvironmentState(TrackingEnvironmentState):
    """Instrument time-weighted CPU utilization and task-level wait events."""

    active_agent = None
    instances = []

    def __init__(self):
        super().__init__()
        self.busy_segments = {}
        self.wait_events = []

    def add_server_and_init_environment(self, server_object):
        super().add_server_and_init_environment(server_object)
        sid = int(server_object.server_id)
        self.busy_segments[sid] = []

    def _advance(self, server_id, now):
        sid = int(server_id)
        start = float(self._last[sid])
        info = self.servers[sid]
        busy = int(info["running_replica"] is not None)
        queued = len(info["waiting_replicas"])
        super()._advance(sid, now)
        end = float(now)
        if end > start:
            self.busy_segments[sid].append((start, end, busy, queued))

    def start_replica_execution(self, server_id, task, selection, service_time, service_start_time):
        key = (int(server_id), int(task.id), selection)
        enqueued = self._enqueued.get(key)
        super().start_replica_execution(server_id, task, selection, service_time, service_start_time)
        if enqueued is not None:
            self.wait_events.append({
                "server_id": int(server_id), "task_id": int(task.id),
                "selection": str(selection), "wait_seconds": max(float(service_start_time) - enqueued, 0.0),
                "service_start_time": float(service_start_time),
            })

    def mean_utilization(self, start, end):
        duration = max(float(end) - float(start), 0.0)
        if duration <= 0.0:
            return 0.0
        per_server = []
        for sid, segments in self.busy_segments.items():
            busy = sum(max(min(b, end) - max(a, start), 0.0) * is_busy
                       for a, b, is_busy, _ in segments)
            per_server.append(busy / duration)
        return float(np.mean(per_server)) if per_server else 0.0


class CouplingArrivalLoop(ArrivalTraceLoop):
    pass


class CounterfactualMaskedAgent(ReliabilityMaskedPairPPOAgent):
    """Frozen stochastic Masked PPO with one-action intervention support."""

    def __init__(self, *args, pairs, pair_rho, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairs = list(pairs)
        self.pair_rho = np.asarray(pair_rho, dtype=float)
        self.loop = None
        self.capture_all_snapshots = False
        self.forced_task_id = None
        self.forced_action = None
        self.target_snapshot = None
        self.snapshot_records = []
        self.action_history = []
        self.queue_history = {}

    def select_action(self, state, epsilon=0.0, use_softmax=False, temperature=1.5):
        context = self.current_decision
        if context is None or not np.array_equal(context["state"], np.asarray(state, dtype=np.float32)):
            raise RuntimeError("Counterfactual decision context mismatch")
        task, env_state, episode = self.current_context
        task_id = int(context["task_id"])
        snap = queue_hazard_snapshot(self.loop, task, env_state, state)
        snapshot_hash = snapshot_signature(snap)
        reliability = np.asarray(context["reliabilities"], dtype=float).copy()
        safe = np.asarray(context["safe_mask"], dtype=bool).copy()
        effective = np.asarray(context["effective_mask"], dtype=bool).copy()
        latency, per_server = estimate_pair_completion_latencies(task, env_state, self.pairs)
        with torch.no_grad():
            logits = self.policy_old(self._to_tensor(state).unsqueeze(0)).squeeze(0)
            masked = masked_logits(logits, effective)
            distribution = torch.distributions.Categorical(logits=masked)
            probabilities = distribution.probs.cpu().numpy().astype(float)
            torch_rng_before_sample = _rng_digest()["torch_cpu"]
            sampled_action = int(distribution.sample().item())
            action = sampled_action
            if task_id == self.forced_task_id:
                action = int(self.forced_action)
            if not effective[action]:
                raise RuntimeError("Intervention selected outside the effective action set")
            log_probability = float(distribution.log_prob(torch.tensor(action, device=self.device)).item())
            entropy = float(distribution.entropy().item())
            value = float(self.value_net(self._to_tensor(state).unsqueeze(0)).squeeze().item())
        if task_id == self.forced_task_id:
            self.target_snapshot = {"payload": snap, "digest": snapshot_hash}
        if self.capture_all_snapshots:
            self.snapshot_records.append({
                "task_id": task_id, "snapshot": snap, "snapshot_digest": snapshot_hash,
                "state": np.asarray(state, dtype=np.float32).copy(),
                "requirement": float(task.reliability_requirement),
                "reliabilities": reliability, "safe_mask": safe, "effective_mask": effective,
                "estimated_latencies": latency.copy(), "per_server_latency": dict(per_server),
                "probabilities": probabilities, "sampled_action": sampled_action,
                "state_value": value, "masked_entropy": entropy,
            })
        if self.forced_task_id is not None and task_id >= int(self.forced_task_id):
            self.queue_history[task_id] = queue_metrics(env_state, task.env.now)
        if task_id in self.pending_decisions:
            raise RuntimeError("Duplicate counterfactual pending decision")
        self.current_decision = None
        self.pending_decisions[task_id] = (
            effective.copy(), log_probability, int(action), np.asarray(state, dtype=np.float32).copy()
        )
        selected_j, selected_k = self.pairs[action]
        record = {
            "episode": int(episode), "task_id": task_id, "decision_time": float(task.env.now),
            "action_index": int(action), "selected_pair": str(self.pairs[action]),
            "safe_set_size": int(safe.sum()), "safe_set_empty": not bool(safe.any()),
            "selected_action_safe": bool(safe[action]),
            "selected_pair_reliability": float(reliability[action]),
            "selected_rho": float(self.pair_correlations[action]),
            "masked_entropy": entropy, "masked_probabilities": probabilities.tolist(),
            "sampled_policy_action_before_intervention": sampled_action,
            "torch_rng_before_sample": torch_rng_before_sample,
            "snapshot_digest": snapshot_hash,
            "estimated_latency": float(latency[action]),
            "server_j": int(selected_j), "server_k": int(selected_k),
        }
        self.selection_archive.append(record)
        self.action_history.append(record)
        return int(action)


def _candidate_reward_proxy(latency, reliability, requirement):
    """Reuse MainLoop.calcReward on decision-time reliability and latency proxy."""
    probe = type("RewardProbe", (), {})()
    probe.resolution_bookkeeping_done = False
    probe.primaryStarted = 0.0
    probe.primaryFinished = float(latency)
    probe.backupFinished = None
    probe.reliability_satisfied = bool(reliability >= requirement)
    probe.reliability_requirement = float(requirement)
    probe.joint_failure_probability = float(1.0 - reliability)
    fake_state = type("StateProbe", (), {"get_task_by_id": lambda self, task_id: probe})()
    fake_loop = type("LoopProbe", (), {
        "env_state": fake_state,
        "_calculate_reliability_violation": staticmethod(main_loop_module.MainLoop._calculate_reliability_violation),
    })()
    reward, _ = main_loop_module.MainLoop.calcReward(fake_loop, 1)
    return {
        "proxy_reward": float(reward),
        "proxy_base_reward": float(probe.base_reward),
        "proxy_reliability_penalty": float(probe.reliability_penalty),
        "known_reliability_satisfied": bool(probe.reliability_satisfied),
        "known_reliability_violation_log10": float(probe.reliability_violation),
        "known_reliability_penalty": float(probe.reliability_penalty),
        "latency_component_is_estimated": True,
    }


def _set_cached_excel_reads():
    original = pd.read_excel
    paths = [ROOT / "data/task_parameters.xlsx", ROOT / "data/server_info.xlsx"]
    cache = {p.resolve(): original(p) for p in paths}

    def cached(path, *args, **kwargs):
        try:
            resolved = Path(path).resolve()
        except TypeError:
            resolved = None
        if resolved in cache:
            return cache[resolved]
        return original(path, *args, **kwargs)
    return original, cached


def _make_agent(trial_id, trial, pairs, rho):
    checkpoint = PPO_OUT / "runs" / f"trial_{trial_id:03d}" / "masked"
    kwargs = _agent_kwargs("pair_scoring", rho, int(trial["PPO_Minibatch_Seed"]))
    agent = CounterfactualMaskedAgent(**kwargs, pairs=pairs, pair_rho=rho,
                                      frozen=True, deployment_mode="stochastic")
    actor = torch.load(checkpoint / "actor.pt", map_location="cpu", weights_only=True)
    critic = torch.load(checkpoint / "critic.pt", map_location="cpu", weights_only=True)
    agent.policy_net.load_state_dict(actor)
    agent.policy_old.load_state_dict(actor)
    agent.value_net.load_state_dict(critic)
    agent.policy_old.eval()
    agent.value_net.eval()
    agent.clear_rollout()
    actor_hash = sha256_file(checkpoint / "actor.pt")
    critic_hash = sha256_file(checkpoint / "critic.pt")
    return agent, actor_hash, critic_hash


def _run_episode(agent, trial, max_task, *, capture_all=False, target_task=None, forced_action=None):
    agent.clear_rollout()
    agent.selection_archive = []
    agent.action_history = []
    agent.snapshot_records = []
    agent.queue_history = {}
    agent.target_snapshot = None
    agent.capture_all_snapshots = bool(capture_all)
    agent.forced_task_id = target_task
    agent.forced_action = forced_action
    CouplingEnvironmentState.instances = []
    CouplingEnvironmentState.active_agent = agent
    original_read, cached_read = _set_cached_excel_reads()
    try:
        with patch.object(main_loop_module, "EnvironmentState", CouplingEnvironmentState):
            with patch.object(pd, "read_excel", side_effect=cached_read):
                with scoped_environment_seeds(int(trial["Eval_Arrival_Seed"]), int(trial["Eval_Spatial_Seed"])):
                    loop = CouplingArrivalLoop(agent, 1, int(max_task), params.num_states, params.num_actions)
                    agent.loop = loop
                    torch.manual_seed(action_seed(trial))
                    with redirect_stdout(io.StringIO()):
                        loop.EP()
    finally:
        pd.read_excel = original_read
    env_state = CouplingEnvironmentState.instances[0]
    return loop, env_state


def _frame(loop):
    frame = pd.DataFrame(loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS)
    if len(frame):
        frame = frame.sort_values("task_id").reset_index(drop=True)
    return frame


def _load_reference_episode(trial_id):
    path = PPO_OUT / "runs" / f"trial_{trial_id:03d}" / "masked_stochastic"
    decisions = pd.read_csv(path / "evaluation_decisions.csv")
    assignments = pd.read_csv(path / "evaluation_task_assignments.csv")
    decisions = decisions[decisions.episode == 1].sort_values("task_id").reset_index(drop=True)
    assignments = assignments[assignments.episode == 1].sort_values("task_id").reset_index(drop=True)
    return decisions, assignments


def _stratified_sample(snapshots, count, trial_id):
    rows = []
    for source in snapshots:
        snap = source["snapshot"]
        backlogs = [x["backlog_time"] for x in snap["servers"]]
        rows.append({**source, "max_backlog_seconds": float(max(backlogs, default=0.0)),
                     "total_backlog_seconds": float(sum(backlogs))})
    if count >= len(rows):
        selected = rows
    else:
        values = np.asarray([row["max_backlog_seconds"] for row in rows], dtype=float)
        cuts = np.quantile(values, [1/3, 2/3])
        for row in rows:
            value = row["max_backlog_seconds"]
            row["load_tertile"] = "low" if value <= cuts[0] else ("medium" if value <= cuts[1] else "high")
        strata = {}
        for row in rows:
            key = (float(row["requirement"]), row["load_tertile"])
            strata.setdefault(key, []).append(row)
        raw = {key: count * len(group) / len(rows) for key, group in strata.items()}
        allocation = {key: int(math.floor(value)) for key, value in raw.items()}
        left = count - sum(allocation.values())
        for key in sorted(raw, key=lambda k: (-(raw[k] - allocation[k]), str(k))):
            if left <= 0:
                break
            allocation[key] += 1
            left -= 1
        rng = np.random.default_rng(np.random.SeedSequence([20260925, int(trial_id), count]))
        selected = []
        for key in sorted(strata, key=str):
            group = strata[key]
            n = min(allocation[key], len(group))
            if n:
                indexes = rng.choice(len(group), size=n, replace=False)
                selected.extend(group[int(i)] for i in indexes)
        if len(selected) < count:
            used = {row["task_id"] for row in selected}
            remaining = [row for row in rows if row["task_id"] not in used]
            indexes = rng.choice(len(remaining), size=count-len(selected), replace=False)
            selected.extend(remaining[int(i)] for i in indexes)
    selected = sorted(selected, key=lambda x: int(x["task_id"]))
    for row in selected:
        row["state_id"] = f"trial_{trial_id:03d}_ep1_task{int(row['task_id']):03d}"
        row.setdefault("load_tertile", "all")
    return selected


def _decision_proxy(snapshot, pairs, rho):
    candidates = []
    for index, (j, k) in enumerate(pairs):
        rel = float(snapshot["reliabilities"][index])
        if not snapshot["effective_mask"][index]:
            continue
        latency = float(snapshot["estimated_latencies"][index])
        reward_parts = _candidate_reward_proxy(latency, rel, float(snapshot["requirement"]))
        queues = {int(item["server_id"]): float(item["backlog_time"])
                  for item in snapshot["snapshot"]["servers"]}
        candidates.append({
            "state_id": snapshot["state_id"], "trial_id": int(snapshot["trial_id"]),
            "episode": 1, "task_id": int(snapshot["task_id"]),
            "action_index": int(index), "server_j": int(j), "server_k": int(k),
            "pair": str((int(j), int(k))), "requirement": float(snapshot["requirement"]),
            "load_tertile": snapshot["load_tertile"],
            "safe_set_size": int(snapshot["safe_mask"].sum()),
            "safe_set_empty": not bool(snapshot["safe_mask"].any()),
            "effective_set_size": int(snapshot["effective_mask"].sum()),
            "estimated_latency": latency, "reliability": rel,
            "rho": float(rho[index]), "queue_j_seconds": queues[int(j)],
            "queue_k_seconds": queues[int(k)],
            "total_backlog_seconds": float(snapshot["total_backlog_seconds"]),
            "maximum_backlog_seconds": float(snapshot["max_backlog_seconds"]),
            "actor_probability": float(snapshot["probabilities"][index]),
            "sampled_ppo_action": int(snapshot["sampled_action"]),
            "ppo_high_probability_action": int(snapshot["high_probability_action"]),
            "critic_value": float(snapshot["state_value"]),
            **reward_parts,
        })
    if not candidates:
        raise RuntimeError("Effective action set unexpectedly empty")
    return candidates


def _interval_returns(assignments, decisions, target_task, terminal_time, gamma):
    decision_times = {int(row["task_id"]): float(row["decision_time"]) for row in decisions}
    if target_task not in decision_times:
        raise RuntimeError("Target decision timestamp missing from counterfactual branch")
    outcomes = []
    for row in assignments.itertuples(index=False):
        if pd.isna(row.Task_Reward) or pd.isna(row.Primary_Start) or pd.isna(row.Task_Delay):
            continue
        outcomes.append((float(row.Primary_Start) + float(row.Task_Delay), float(row.Task_Reward), int(row.task_id)))
    outcomes.sort()
    results = {}
    available_transitions = 201 - int(target_task)
    for horizon in HORIZONS:
        effective_horizon = min(int(horizon), available_transitions)
        total = 0.0
        starts = []
        ends = []
        for step in range(effective_horizon):
            current_id = int(target_task) + step
            start = decision_times[current_id]
            next_id = current_id + 1
            end = decision_times[next_id] if next_id in decision_times else float(terminal_time)
            include_terminal = next_id not in decision_times
            interval_reward = sum(
                reward for time, reward, _task_id in outcomes
                if time >= start - 1e-12 and (time <= end + 1e-12 if include_terminal else time < end - 1e-12)
            )
            total += (float(gamma) ** step) * interval_reward
            starts.append(start)
            ends.append(end)
        results[horizon] = {
            "q_return": float(total), "effective_horizon": int(effective_horizon),
            "start_time": starts[0] if starts else float("nan"),
            "end_time": ends[-1] if ends else float("nan"),
        }
    return results


def _window_metrics(assignments, state_history, wait_events, busy_segments,
                    target_task, effective_horizon, terminal_time, target_time):
    end_task = min(200, int(target_task) + max(int(effective_horizon) - 1, 0))
    rows = assignments[(assignments.task_id >= int(target_task)) & (assignments.task_id <= end_task)]
    delays = pd.to_numeric(rows.Task_Delay, errors="coerce").dropna().to_numpy(dtype=float)
    rewards = pd.to_numeric(rows.Task_Reward, errors="coerce").dropna().to_numpy(dtype=float)
    if int(target_task) + int(effective_horizon) <= 200:
        boundary_task = int(target_task) + int(effective_horizon)
        end_time = float(state_history[boundary_task]["decision_time"])
        queue = state_history[boundary_task]
    else:
        end_time = float(terminal_time)
        queue = queue_metrics_from_zero()  # the episode has drained all CPU work
    relevant_waits = [float(row["wait_seconds"]) for row in wait_events
                      if row["service_start_time"] >= target_time and row["service_start_time"] < end_time]
    utilization = mean_utilization(busy_segments, target_time, end_time)
    return {
        "end_time": end_time,
        "total_queue_length": float(queue["total_queue_length"]),
        "maximum_server_queue_length": float(queue["maximum_server_queue_length"]),
        "total_backlog_seconds": float(queue["total_backlog_seconds"]),
        "maximum_server_backlog_seconds": float(queue["maximum_server_backlog_seconds"]),
        "mean_waiting_time_seconds": float(np.mean(relevant_waits)) if relevant_waits else 0.0,
        "mean_future_task_latency_seconds": float(np.mean(delays)) if len(delays) else float("nan"),
        "mean_future_task_reward": float(np.mean(rewards)) if len(rewards) else float("nan"),
        "mean_server_utilization": float(utilization),
    }


def queue_metrics_from_zero():
    return {"total_queue_length": 0, "maximum_server_queue_length": 0,
            "total_backlog_seconds": 0.0, "maximum_server_backlog_seconds": 0.0}


def mean_utilization(segments, start, end):
    duration = max(float(end) - float(start), 0.0)
    if duration <= 0:
        return 0.0
    server_values = []
    for values in segments.values():
        busy = sum(max(min(stop, end) - max(begin, start), 0.0) * is_busy
                   for begin, stop, is_busy, _queued in values)
        server_values.append(busy / duration)
    return float(np.mean(server_values)) if server_values else 0.0


def _run_branch(agent, trial, target_snapshot, forced_action, baseline_actions,
                baseline_arrivals, baseline_hazards, target_task):
    max_task = min(200, int(target_task) + max(HORIZONS))
    loop, env_state = _run_episode(agent, trial, max_task, capture_all=False,
                                   target_task=int(target_task), forced_action=int(forced_action))
    if agent.target_snapshot is None:
        raise RuntimeError("Counterfactual branch did not capture its target state")
    if agent.target_snapshot["digest"] != target_snapshot["snapshot_digest"]:
        raise RuntimeError("Counterfactual replay did not restore exact state/queue/hazard/RNG snapshot")
    before = [x["action_index"] for x in agent.action_history if int(x["task_id"]) < int(target_task)]
    expected = [int(baseline_actions[i]) for i in range(1, int(target_task))]
    if before != expected:
        raise RuntimeError("Action prefix changed before the intervention")
    actual_arrivals = np.asarray(loop.interarrival_trace[:int(target_task)], dtype=float)
    expected_arrivals = np.asarray(baseline_arrivals[:int(target_task)], dtype=float)
    if not np.array_equal(actual_arrivals, expected_arrivals):
        raise RuntimeError("External arrival stream changed before the intervention")
    hazard_payload = agent.target_snapshot["payload"]["hazards"]
    if stable_digest(hazard_payload) != stable_digest(baseline_hazards):
        raise RuntimeError("Episode spatial-hazard realization changed across replay")
    decision_rows = [x for x in agent.action_history if int(x["task_id"]) >= int(target_task)]
    for row in decision_rows:
        if row["task_id"] > int(target_task):
            ref_rng = baseline_actions.get("torch_rng_by_task", {}).get(row["task_id"])
            # RNG advances once per decision regardless of probabilities; its
            # task-index stream position is checked separately by the tests.
            if ref_rng is not None and row.get("torch_rng_before_sample") not in (None, ref_rng):
                raise RuntimeError("Downstream policy random stream shifted after intervention")
    assignments = _frame(loop)
    qvalues = _interval_returns(assignments, agent.action_history, int(target_task),
                               float(loop.env.now), float(params.gamma_ppo))
    decision_times = {int(row["task_id"]): float(row["decision_time"])
                      for row in agent.action_history}
    target_time = decision_times[int(target_task)]
    # queue_history captures the exact pre-action queue at the target arrival;
    # retain it rather than replacing it with the branch's terminal queue.
    history = dict(agent.queue_history)
    if int(target_task) not in history:
        raise RuntimeError("Target queue snapshot missing from intervention replay")
    history[int(target_task)]["decision_time"] = target_time
    for row in agent.action_history:
        tid = int(row["task_id"])
        if tid in history:
            history[tid]["decision_time"] = float(row["decision_time"])
    history[201] = queue_metrics(env_state, loop.env.now)
    history[201]["decision_time"] = float(loop.env.now)
    future = {}
    for horizon in HORIZONS:
        effective = qvalues[horizon]["effective_horizon"]
        future[horizon] = _window_metrics(assignments, history, env_state.wait_events,
                                          env_state.busy_segments, int(target_task),
                                          effective, float(loop.env.now), target_time)
    return {
        "action_index": int(forced_action), "qvalues": qvalues, "history": history,
        "future_metrics": future,
        "assignments": assignments,
        "terminal_time": float(loop.env.now),
        "target_time": target_time,
        "target_snapshot_digest": agent.target_snapshot["digest"],
        "branch_action_trace": [int(x["action_index"]) for x in agent.action_history],
        "arrival_trace": list(map(float, loop.interarrival_trace)),
        "hazard_digest": stable_digest(hazard_payload),
        "actor_digest_after": stable_digest({k: v.detach().cpu().numpy() for k, v in agent.policy_old.state_dict().items()}),
    }


def _baseline_trial(trial_id, trial, agent, pairs, rho):
    loop, _ = _run_episode(agent, trial, 200, capture_all=True)
    assignments = _frame(loop)
    if len(assignments) != 200 or len(agent.snapshot_records) != 200:
        raise RuntimeError("Baseline replay must collect exactly 200 decision states")
    saved_decisions, saved_assignments = _load_reference_episode(trial_id)
    observed_actions = np.asarray([x["action_index"] for x in agent.action_history], dtype=int)
    if len(saved_decisions) != 200 or not np.array_equal(observed_actions, saved_decisions.action_index.to_numpy(dtype=int)):
        raise RuntimeError(f"Formal Masked PPO replay actions differ for seed {trial_id}")
    for name in ("Task_Delay", "Task_Reward", "Execution_Reliability"):
        if not np.allclose(assignments[name], saved_assignments[name], rtol=0, atol=1e-10):
            raise RuntimeError(f"Formal Masked PPO replay outcomes differ in {name} for seed {trial_id}")
    baselines = {int(row["task_id"]): int(row["action_index"]) for row in agent.action_history}
    baselines["torch_rng_by_task"] = {
        int(row["task_id"]): row.get("torch_rng_before_sample") for row in agent.action_history
    }
    hazard = {
        "effective_failure_rates": {str(k): float(v) for k, v in sorted(loop.env_state.effective_failure_rates.items())},
        "spatial_risk_field": np.asarray(loop.env_state.spatial_risk_field, dtype=float).tolist(),
        "distance_matrix": np.asarray(loop.env_state.spatial_distance_matrix, dtype=float).tolist(),
        "correlation_matrix": np.asarray(loop.env_state.spatial_correlation_matrix, dtype=float).tolist(),
    }
    snapshots = []
    for record in agent.snapshot_records:
        raw = record["snapshot"]
        record = dict(record)
        record["trial_id"] = int(trial_id)
        record["state_id"] = f"trial_{trial_id:03d}_ep1_task{int(record['task_id']):03d}"
        record["high_probability_action"] = int(np.argmax(record["probabilities"]))
        record["safe_set_size"] = int(np.asarray(record["safe_mask"], dtype=bool).sum())
        record["safe_set_empty"] = not bool(record["safe_mask"].any())
        record["effective_set_size"] = int(np.asarray(record["effective_mask"], dtype=bool).sum())
        record["max_backlog_seconds"] = float(max(x["backlog_time"] for x in raw["servers"]))
        record["total_backlog_seconds"] = float(sum(x["backlog_time"] for x in raw["servers"]))
        record["load_tertile"] = "all"
        snapshots.append(record)
    return snapshots, baselines, list(loop.interarrival_trace), hazard


def _copy_candidate_row(template, action, candidate_result):
    row = dict(template)
    row["action_index"] = int(action)
    for horizon in HORIZONS:
        row[f"q_h{horizon}"] = float(candidate_result["qvalues"][horizon]["q_return"])
        row[f"effective_h_h{horizon}"] = int(candidate_result["qvalues"][horizon]["effective_horizon"])
        metrics = candidate_result["future_metrics"][horizon]
        row[f"future_latency_h{horizon}"] = metrics["mean_future_task_latency_seconds"]
        row[f"future_reward_h{horizon}"] = metrics["mean_future_task_reward"]
        row[f"future_queue_h{horizon}"] = metrics["total_queue_length"]
        row[f"future_backlog_h{horizon}"] = metrics["total_backlog_seconds"]
        row[f"future_wait_h{horizon}"] = metrics["mean_waiting_time_seconds"]
        row[f"future_utilization_h{horizon}"] = metrics["mean_server_utilization"]
    return row


def _trial_worker(trial_id, per_trial_states, output_dir):
    torch.set_num_threads(1)
    meta, seed_plan = formal_spec()
    trial = seed_plan.iloc[int(trial_id)].to_dict()
    pairs, rho = build_pair_correlations()
    pairs, rho = list(pairs), np.asarray(rho, dtype=float)
    agent, actor_hash, critic_hash = _make_agent(trial_id, trial, pairs, rho)
    snapshots, base_actions, arrival_trace, hazards = _baseline_trial(trial_id, trial, agent, pairs, rho)
    selected = _stratified_sample(snapshots, int(per_trial_states), int(trial_id))
    state_rows, candidate_rows, effect_rows, half_rows, disagreement_rows, replay_rows = [], [], [], [], [], []
    for snapshot in selected:
        state_id = snapshot["state_id"]
        template = {k: snapshot[k] for k in (
            "state_id", "trial_id", "task_id", "requirement", "load_tertile",
            "safe_set_size", "safe_set_empty", "effective_mask", "safe_mask",
            "estimated_latencies", "reliabilities", "probabilities", "sampled_action",
            "high_probability_action", "state_value", "state", "snapshot_digest",
            "max_backlog_seconds", "total_backlog_seconds",
        )}
        safe_mask = np.asarray(snapshot["safe_mask"], dtype=bool)
        effective_mask = np.asarray(snapshot["effective_mask"], dtype=bool)
        candidate_mask = safe_mask if safe_mask.any() else effective_mask
        ml_action = int(np.flatnonzero(candidate_mask)[np.argmin(
            np.asarray(snapshot["estimated_latencies"])[candidate_mask])])
        ordered_by_latency = sorted(np.flatnonzero(candidate_mask),
                                   key=lambda i: (float(snapshot["estimated_latencies"][i]), int(i)))
        alternate = int(ordered_by_latency[1]) if len(ordered_by_latency) > 1 else ml_action
        ppo_action = int(snapshot["high_probability_action"])
        sampled_action = int(snapshot["sampled_action"])
        watched = {ml_action, alternate, ppo_action, sampled_action}
        per_action = {}
        candidate_base = _decision_proxy(snapshot, pairs, rho)
        templates = {int(row["action_index"]): row for row in candidate_base}
        baseline_branch = None
        for action in np.flatnonzero(candidate_mask):
            action = int(action)
            branch = _run_branch(agent, trial, snapshot, action, base_actions,
                                 arrival_trace, hazards, int(snapshot["task_id"]))
            per_action[action] = branch
            candidate_rows.append(_copy_candidate_row(templates[action], action, branch))
            if action == ml_action:
                baseline_branch = branch
            replay_rows.append({
                "state_id": state_id, "trial_id": int(trial_id), "task_id": int(snapshot["task_id"]),
                "action_index": action, "snapshot_match": True, "prefix_match": True,
                "arrival_stream_match": True, "hazard_match": True,
                "safe_mask_size": int(safe_mask.sum()), "effective_set_size": int(effective_mask.sum()),
            })
        if baseline_branch is None:
            raise RuntimeError("Safe Min-Latency action was not evaluated")
        # State summary and actor/critic features.
        state_vector = np.asarray(snapshot["state"], dtype=float)
        selected_policy_actions = [("PPO sampled", sampled_action), ("PPO high probability", ppo_action)]
        for label, action in selected_policy_actions:
            branch = per_action.get(action)
            if branch is None:
                # A numerical mask mismatch is fatal; the sampled/greedy masked
                # action must always be in the effective set.
                raise RuntimeError(f"{label} action is not in the evaluated effective mask")
            ml_template = templates[ml_action]
            ppo_template = templates[action]
            disagreement = {
                "state_id": state_id, "trial_id": int(trial_id), "task_id": int(snapshot["task_id"]),
                "requirement": float(snapshot["requirement"]), "load_tertile": snapshot["load_tertile"],
                "safe_set_size": int(safe_mask.sum()), "action_source": label,
                "myopic_action": ml_action, "ppo_action": int(action),
                "same_action": bool(action == ml_action),
                "myopic_estimated_latency": float(ml_template["estimated_latency"]),
                "ppo_estimated_latency": float(ppo_template["estimated_latency"]),
                "immediate_latency_gap": float(ppo_template["estimated_latency"] - ml_template["estimated_latency"]),
                "reliability_gap": float(ppo_template["reliability"] - ml_template["reliability"]),
                "rho_gap": float(ppo_template["rho"] - ml_template["rho"]),
            }
            for horizon in HORIZONS:
                qml = float(baseline_branch["qvalues"][horizon]["q_return"])
                qppo = float(branch["qvalues"][horizon]["q_return"])
                disagreement[f"q_ml_h{horizon}"] = qml
                disagreement[f"q_ppo_h{horizon}"] = qppo
                disagreement[f"q_gap_ppo_minus_ml_h{horizon}"] = qppo - qml
                disagreement[f"justified_h{horizon}"] = bool(
                    ppo_template["estimated_latency"] > ml_template["estimated_latency"] + 1e-9
                    and qppo > qml + 1e-9
                )
            disagreement_rows.append(disagreement)
        state_rows.append({
            "state_id": state_id, "trial_id": int(trial_id), "task_id": int(snapshot["task_id"]),
            "requirement": float(snapshot["requirement"]), "load_tertile": snapshot["load_tertile"],
            "safe_set_size": int(safe_mask.sum()), "safe_set_empty": not bool(safe_mask.any()),
            "effective_set_size": int(effective_mask.sum()),
            "total_backlog_seconds": float(snapshot["total_backlog_seconds"]),
            "maximum_backlog_seconds": float(snapshot["max_backlog_seconds"]),
            "estimated_min_latency": float(np.min(np.asarray(snapshot["estimated_latencies"])[candidate_mask])),
            "myopic_action": ml_action, "ppo_sampled_action": sampled_action,
            "ppo_high_probability_action": ppo_action, "critic_value": float(snapshot["state_value"]),
            "snapshot_digest": snapshot["snapshot_digest"],
            **{f"state_feature_{i}": float(v) for i, v in enumerate(state_vector)},
        })
        # Future queue/reward/latency effects for the myopic choice, runner-up,
        # and PPO-probability choices; all are read from paired action branches.
        for label, action in (("Safe Min-Latency", ml_action),
                              ("Next-best immediate", alternate),
                              ("PPO high probability", ppo_action),
                              ("PPO sampled", sampled_action)):
            branch = per_action[action]
            for horizon in HORIZONS:
                eff = branch["qvalues"][horizon]["effective_horizon"]
                left, right = baseline_branch["future_metrics"][horizon], branch["future_metrics"][horizon]
                effect_rows.append({
                    "state_id": state_id, "trial_id": int(trial_id), "task_id": int(snapshot["task_id"]),
                    "requirement": float(snapshot["requirement"]), "load_tertile": snapshot["load_tertile"],
                    "safe_set_size": int(safe_mask.sum()), "choice": label, "action_index": int(action),
                    "horizon": int(horizon), "effective_horizon": int(eff),
                    **{f"delta_{key}": (float(right[key]) - float(left[key])
                                        if np.isfinite(right[key]) and np.isfinite(left[key]) else float("nan"))
                       for key in (
                           "total_queue_length", "maximum_server_queue_length",
                           "total_backlog_seconds", "maximum_server_backlog_seconds",
                           "mean_waiting_time_seconds", "mean_future_task_latency_seconds",
                           "mean_future_task_reward", "mean_server_utilization")},
                })
        # Queue-effect persistence against the Safe Min-Latency branch.
        base_hist = baseline_branch["history"]
        for action, branch in per_action.items():
            if action == ml_action:
                continue
            deltas = []
            for step in range(1, max(HORIZONS) + 1):
                tid = int(snapshot["task_id"]) + step
                h_alt = branch["history"].get(tid)
                h_ml = base_hist.get(tid)
                if h_alt is None or h_ml is None:
                    continue
                deltas.append((step, float(h_alt["total_backlog_seconds"] - h_ml["total_backlog_seconds"])))
            peak_window = [(h, abs(d)) for h, d in deltas if h <= 5]
            initial = max((v for _, v in peak_window), default=0.0)
            peak_step = next((h for h, v in peak_window if v == initial), None)
            half = {0.50: None, 0.25: None, 0.10: None}
            if initial > 1e-9 and peak_step is not None:
                for threshold in half:
                    for idx, (h, delta) in enumerate(deltas):
                        if h < peak_step:
                            continue
                        window = [abs(x[1]) for x in deltas[idx:idx+3]]
                        if len(window) == 3 and max(window) <= threshold * initial:
                            half[threshold] = h
                            break
            half_rows.append({
                "state_id": state_id, "trial_id": int(trial_id), "task_id": int(snapshot["task_id"]),
                "requirement": float(snapshot["requirement"]), "load_tertile": snapshot["load_tertile"],
                "safe_set_size": int(safe_mask.sum()), "reference_action": ml_action,
                "alternative_action": int(action),
                "initial_peak_queue_difference_seconds": float(initial), "peak_step": peak_step,
                "half_life_50_steps": half[0.50], "half_life_25_steps": half[0.25],
                "half_life_10_steps": half[0.10], "right_censored": bool(initial > 1e-9 and all(v is None for v in half.values())),
            })
    trial_dir = Path(output_dir) / "runs" / f"trial_{int(trial_id):03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("states.csv", state_rows), ("candidate_returns.csv", candidate_rows),
                       ("action_future_effects.csv", effect_rows), ("action_effect_half_life.csv", half_rows),
                       ("ppo_disagreement.csv", disagreement_rows), ("replay_checks.csv", replay_rows)):
        pd.DataFrame(rows).to_csv(trial_dir / name, index=False)
    manifest = {
        "trial_id": int(trial_id), "num_states": len(state_rows),
        "counterfactual_actions": len(candidate_rows), "formal_eval_replay_actions_match": True,
        "actor_sha256": actor_hash, "critic_sha256": critic_hash,
        "prefix_snapshot_external_stream_replay_checks": len(replay_rows),
        "branch_intervention_count": len(candidate_rows),
        "horizons": list(HORIZONS), "gamma": float(params.gamma_ppo),
        "downstream_policy": "frozen_masked_pair_ppo_stochastic",
        "episode": 1,
    }
    (trial_dir / "replay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"trial {trial_id}: {len(state_rows)} sampled states, {len(candidate_rows)} action branches", flush=True)
    return str(trial_dir)


def _concat_trial_files(trial_dirs, name):
    frames = [pd.read_csv(Path(directory) / name) for directory in trial_dirs]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) <= 1e-12 or np.std(b[mask]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _group_cv_r2(X, y, groups, folds=5):
    X, y, groups = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(groups)
    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y, groups = X[finite], y[finite], groups[finite]
    unique = np.unique(groups)
    if len(unique) < 2 or len(y) < 10:
        return float("nan")
    n_splits = min(int(folds), len(unique))
    # Deterministic greedy group partitioning keeps all rows from one state
    # together while balancing fold sizes, without an optional sklearn runtime.
    group_indexes = {group: np.flatnonzero(groups == group) for group in unique}
    ordered_groups = sorted(unique, key=lambda group: (-len(group_indexes[group]), str(group)))
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for group in ordered_groups:
        fold = min(range(n_splits), key=lambda index: (fold_sizes[index], index))
        fold_groups[fold].append(group)
        fold_sizes[fold] += len(group_indexes[group])
    prediction = np.full(len(y), np.nan)
    for fold in range(n_splits):
        test = np.concatenate([group_indexes[group] for group in fold_groups[fold]])
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        mean = X[train].mean(axis=0)
        scale = X[train].std(axis=0)
        scale[scale <= 1e-12] = 1.0
        train_x = np.column_stack([np.ones(len(train)), (X[train] - mean) / scale])
        test_x = np.column_stack([np.ones(len(test)), (X[test] - mean) / scale])
        beta, *_ = np.linalg.lstsq(train_x, y[train], rcond=None)
        prediction[test] = test_x @ beta
    ss_total = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - prediction) ** 2) / ss_total) if ss_total > 1e-12 else float("nan")


def _regret_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p75": float("nan"),
                "p90": float("nan"), "p95": float("nan"), "max": float("nan"),
                "p_le_1e_3": float("nan"), "p_le_0_01": float("nan"),
                "p_le_0_1": float("nan"), "p_le_1": float("nan")}
    return {
        "n": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p75": float(np.quantile(values, .75)), "p90": float(np.quantile(values, .90)),
        "p95": float(np.quantile(values, .95)), "max": float(np.max(values)),
        "p_le_1e_3": float(np.mean(values <= 1e-3)), "p_le_0_01": float(np.mean(values <= .01)),
        "p_le_0_1": float(np.mean(values <= .1)), "p_le_1": float(np.mean(values <= 1.0)),
    }


def summarize_queue_memory():
    telemetry_path = ROOT / "diagnostics/results/external_baselines/baseline_decision_telemetry.csv.gz"
    frame = pd.read_csv(telemetry_path)
    frame = frame[frame.policy == "Masked PPO stochastic"].sort_values(
        ["trial_id", "episode", "task_id"]
    )
    rows = []
    for sid in range(1, 9):
        col = f"backlog_server_{sid}"
        lag_values = {lag: [] for lag in (1, 2, 5, 10, 20)}
        future_latency = {lag: [] for lag in (1, 5, 10, 20)}
        for _, group in frame.groupby(["trial_id", "episode"], sort=False):
            backlog = group[col].to_numpy(dtype=float)
            latency = group.Task_Delay.to_numpy(dtype=float)
            for lag in lag_values:
                if len(backlog) > lag:
                    a, b = backlog[:-lag], backlog[lag:]
                    corr = _safe_corr(a, b)
                    if np.isfinite(corr): lag_values[lag].append(corr)
            for lag in future_latency:
                if len(backlog) > lag:
                    future_latency[lag].extend(zip(backlog[:-lag], latency[lag:]))
        row = {"server_id": sid, "measure": "decision_backlog_seconds"}
        for lag, values in lag_values.items():
            row[f"lag_{lag}_autocorrelation"] = float(np.mean(values)) if values else float("nan")
            row[f"lag_{lag}_episodes_n"] = int(len(values))
        for lag, values in future_latency.items():
            row[f"backlog_vs_latency_lag_{lag}_correlation"] = _safe_corr(
                [x[0] for x in values], [x[1] for x in values]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def safe_min_externality():
    frame = pd.read_csv(ROOT / "diagnostics/results/external_baselines/baseline_decision_telemetry.csv.gz")
    frame = frame[frame.policy == "Safe Min-Latency"].sort_values(["trial_id", "episode", "task_id"])
    output = []
    for k in (5, 10, 20):
        recurrences, deltas, latencies, backlog_deltas = [], [], [], []
        for _, group in frame.groupby(["trial_id", "episode"], sort=False):
            records = list(group.to_dict("records"))
            for i, row in enumerate(records):
                future = records[i+1:i+1+k]
                if not future: continue
                endpoints = {int(row["server_j"]), int(row["server_k"])}
                counts = sum(int(r["server_j"]) in endpoints for r in future) + sum(int(r["server_k"]) in endpoints for r in future)
                j, kk = int(row["server_j"]), int(row["server_k"])
                later = records[min(i+k, len(records)-1)]
                before = .5 * (float(row[f"backlog_server_{j}"]) + float(row[f"backlog_server_{kk}"]))
                after = .5 * (float(later[f"backlog_server_{j}"]) + float(later[f"backlog_server_{kk}"]))
                recurrences.append(counts); deltas.append(after-before)
                latencies.append(float(row["estimated_latency"]))
                backlog_deltas.append(float(row["next_decision_selected_pair_backlog_delta"]))
        output.append({
            "future_window_tasks": k, "n": len(deltas),
            "mean_endpoint_reselections": float(np.mean(recurrences)) if recurrences else float("nan"),
            "corr_immediate_estimated_latency_vs_endpoint_reselections": _safe_corr(latencies, recurrences),
            "corr_immediate_estimated_latency_vs_backlog_change": _safe_corr(latencies, deltas),
            "mean_endpoint_backlog_change_seconds": float(np.mean(deltas)) if deltas else float("nan"),
            "mean_one_step_selected_backlog_change_seconds": float(np.mean(backlog_deltas)) if backlog_deltas else float("nan"),
        })
    return pd.DataFrame(output)


def aggregate(output_dir, trial_dirs):
    states = _concat_trial_files(trial_dirs, "states.csv")
    candidates = _concat_trial_files(trial_dirs, "candidate_returns.csv")
    effects = _concat_trial_files(trial_dirs, "action_future_effects.csv")
    half = _concat_trial_files(trial_dirs, "action_effect_half_life.csv")
    disagreement = _concat_trial_files(trial_dirs, "ppo_disagreement.csv")
    replay = _concat_trial_files(trial_dirs, "replay_checks.csv")
    horizons = list(HORIZONS)
    per_state = []
    for (state_id, trial, task_id), group in candidates.groupby(["state_id", "trial_id", "task_id"], sort=False):
        first = group.iloc[0]
        state = states[states.state_id == state_id].iloc[0]
        effective = group
        ml_action = int(state.myopic_action)
        ppo_action = int(state.ppo_sampled_action)
        row = {
            "state_id": state_id, "trial_id": int(trial), "task_id": int(task_id),
            "requirement": float(state.requirement), "load_tertile": state.load_tertile,
            "safe_set_size": int(state.safe_set_size), "safe_set_empty": bool(state.safe_set_empty),
            "maximum_backlog_seconds": float(state.maximum_backlog_seconds),
            "total_backlog_seconds": float(state.total_backlog_seconds),
            "num_counterfactual_actions": int(len(group)), "myopic_action": ml_action,
            "ppo_sampled_action": ppo_action,
        }
        for h in horizons:
            qcol = f"q_h{h}"
            best = float(group[qcol].max())
            tied = group[np.isclose(group[qcol].to_numpy(dtype=float), best, rtol=0, atol=1e-9)]
            ml = group[group.action_index == ml_action].iloc[0]
            ppo = group[group.action_index == ppo_action].iloc[0]
            row[f"oracle_q_h{h}"] = best
            row[f"myopic_q_h{h}"] = float(ml[qcol])
            row[f"ppo_q_h{h}"] = float(ppo[qcol])
            ppo_high = group[group.action_index == int(state.ppo_high_probability_action)].iloc[0]
            row[f"ppo_high_probability_q_h{h}"] = float(ppo_high[qcol])
            row[f"regret_h{h}"] = max(best - float(ml[qcol]), 0.0)
            row[f"agreement_h{h}"] = bool(ml_action in set(tied.action_index.astype(int)))
            row[f"optimal_set_size_h{h}"] = int(len(tied))
            row[f"effective_h_h{h}"] = int(group[f"effective_h_h{h}"].iloc[0])
            row[f"oracle_future_latency_h{h}"] = float(tied[f"future_latency_h{h}"].mean())
            row[f"myopic_future_latency_h{h}"] = float(ml[f"future_latency_h{h}"])
            row[f"oracle_future_reward_h{h}"] = float(tied[f"future_reward_h{h}"].mean())
            row[f"myopic_future_reward_h{h}"] = float(ml[f"future_reward_h{h}"])
            row[f"ppo_future_latency_h{h}"] = float(ppo[f"future_latency_h{h}"])
            row[f"ppo_future_reward_h{h}"] = float(ppo[f"future_reward_h{h}"])
            row[f"ppo_future_queue_h{h}"] = float(ppo[f"future_queue_h{h}"])
            row[f"ppo_future_backlog_h{h}"] = float(ppo[f"future_backlog_h{h}"])
        per_state.append(row)
    state_results = pd.DataFrame(per_state)
    state_results.to_csv(output_dir / "counterfactual_state_results.csv", index=False)
    state_results["load_level"] = state_results.load_tertile
    state_results["safe_set_bin"] = pd.cut(
        state_results.safe_set_size, bins=[-1, 0, 5, 15, 28],
        labels=["empty", "1-5", "6-15", "16-28"], include_lowest=True,
    ).astype(str)
    agreement_rows, regret_rows, load_rows, req_rows, safe_rows = [], [], [], [], []
    for h in horizons:
        for label, group_col, groups in (
            ("overall", None, [("all", state_results)]),
            ("load", "load_level", list(state_results.groupby("load_level", observed=True))),
            ("requirement", "requirement", list(state_results.groupby("requirement"))),
            ("safe_set_size", "safe_set_bin", list(state_results.groupby("safe_set_bin", observed=True))),
        ):
            for group_name, subset in groups:
                if label == "overall" or len(subset):
                    vals = subset[f"regret_h{h}"].to_numpy(dtype=float)
                    stat = _regret_stats(vals)
                    ag = float(subset[f"agreement_h{h}"].mean()) if len(subset) else float("nan")
                    summary = {"horizon": h, "stratification": label, "group": str(group_name),
                               "n": int(len(subset)), "myopic_oracle_agreement_rate": ag,
                               "mean_optimal_set_size": float(subset[f"optimal_set_size_h{h}"].mean()) if len(subset) else float("nan"),
                               "mean_safe_set_size": float(subset.safe_set_size.mean()) if len(subset) else float("nan"),
                               "safe_set_empty_rate": float(subset.safe_set_empty.mean()) if len(subset) else float("nan"),
                               "ppo_sampled_mean_advantage": float((subset[f"ppo_q_h{h}"] - subset[f"myopic_q_h{h}"]).mean()) if len(subset) else float("nan"),
                               "ppo_sampled_positive_advantage_rate": float(((subset[f"ppo_q_h{h}"] - subset[f"myopic_q_h{h}"]) > 1e-9).mean()) if len(subset) else float("nan"),
                               "ppo_high_probability_mean_advantage": float((subset[f"ppo_high_probability_q_h{h}"] - subset[f"myopic_q_h{h}"]).mean()) if len(subset) else float("nan"),
                               "ppo_high_probability_positive_advantage_rate": float(((subset[f"ppo_high_probability_q_h{h}"] - subset[f"myopic_q_h{h}"]) > 1e-9).mean()) if len(subset) else float("nan"),
                               **stat}
                    agreement_rows.append(summary)
                    regret_rows.append(summary.copy())
                    if label == "load": load_rows.append(summary.copy())
                    elif label == "requirement": req_rows.append(summary.copy())
                    elif label == "safe_set_size": safe_rows.append(summary.copy())
    agreement = pd.DataFrame(agreement_rows)
    regret = pd.DataFrame(regret_rows)
    agreement.to_csv(output_dir / "myopic_oracle_agreement.csv", index=False)
    regret.to_csv(output_dir / "myopic_long_term_regret.csv", index=False)
    pd.DataFrame(load_rows).to_csv(output_dir / "load_stratified_results.csv", index=False)
    pd.DataFrame(req_rows).to_csv(output_dir / "requirement_stratified_results.csv", index=False)
    pd.DataFrame(safe_rows).to_csv(output_dir / "safe_set_size_analysis.csv", index=False)
    ceiling_rows = []
    for h in horizons:
        ceiling_rows.append({
            "horizon": h, "n_states": int(len(state_results)),
            "mean_q_return_oracle_improvement_over_myopic": float(state_results[f"regret_h{h}"].mean()),
            "p95_q_return_oracle_improvement_over_myopic": float(state_results[f"regret_h{h}"].quantile(.95)),
            "mean_task_reward_delta_for_q_oracle_action_set": float((state_results[f"oracle_future_reward_h{h}"] - state_results[f"myopic_future_reward_h{h}"]).mean()),
            "mean_task_latency_delta_for_q_oracle_action_set_seconds": float((state_results[f"oracle_future_latency_h{h}"] - state_results[f"myopic_future_latency_h{h}"]).mean()),
        })
    pd.DataFrame(ceiling_rows).to_csv(output_dir / "oracle_improvement_ceiling.csv", index=False)
    effects.to_csv(output_dir / "action_future_queue_effect.csv", index=False)
    half.to_csv(output_dir / "action_effect_half_life.csv", index=False)
    disagreement.to_csv(output_dir / "ppo_vs_myopic_disagreement.csv", index=False)
    replay.to_csv(output_dir / "replay_checks.csv", index=False)

    justification_rows = []
    for source in sorted(disagreement.action_source.unique()):
        part = disagreement[disagreement.action_source == source]
        for h in horizons:
            eligible = part[(part.immediate_latency_gap > 1e-9) & (part[f"q_ml_h{h}"].notna())]
            justified = eligible[f"q_gap_ppo_minus_ml_h{h}"] > 1e-9
            justification_rows.append({
                "action_source": source, "horizon": h, "eligible_deviations": int(len(eligible)),
                "justified_deviation_count": int(justified.sum()),
                "long_term_justified_deviation_rate": float(justified.mean()) if len(eligible) else float("nan"),
            })
    justification = pd.DataFrame(justification_rows)
    justification.to_csv(output_dir / "ppo_long_term_justification.csv", index=False)
    stratum_rows = []
    for source in sorted(disagreement.action_source.unique()):
        part = disagreement[disagreement.action_source == source]
        for horizon in horizons:
            for stratification, column in (("load", "load_tertile"), ("requirement", "requirement")):
                for group_name, group in part.groupby(column, sort=True):
                    eligible = group[(group.immediate_latency_gap > 1e-9) & group[f"q_ml_h{horizon}"].notna()]
                    advantage = eligible[f"q_gap_ppo_minus_ml_h{horizon}"]
                    stratum_rows.append({
                        "action_source": source, "horizon": horizon, "stratification": stratification,
                        "group": str(group_name), "n_states": int(len(group)),
                        "eligible_deviations": int(len(eligible)),
                        "justified_deviation_count": int((advantage > 1e-9).sum()),
                        "long_term_justified_deviation_rate": float((advantage > 1e-9).mean()) if len(eligible) else float("nan"),
                        "mean_ppo_minus_myopic_q": float(advantage.mean()) if len(eligible) else float("nan"),
                    })
    pd.DataFrame(stratum_rows).to_csv(output_dir / "ppo_justification_by_stratum.csv", index=False)

    critic_rows = []
    # Cross-validated predictive value of current observable/myopic state features.
    feature_columns = [f"state_feature_{i}" for i in range(states.filter(regex=r"^state_feature_\d+$").shape[1])]
    base_features = states[["total_backlog_seconds", "maximum_backlog_seconds", "estimated_min_latency",
                            "safe_set_size", "requirement", *feature_columns]].to_numpy(dtype=float)
    critic_r2 = _group_cv_r2(base_features, states.critic_value.to_numpy(dtype=float), states.state_id.to_numpy())
    critic_rows.append({"analysis": "critic_value_from_current_observable_state", "horizon": 0,
                        "n": int(len(states)), "correlation": _safe_corr(states.critic_value, states.estimated_min_latency),
                        "cross_validated_r2": critic_r2})
    state_index = states.set_index("state_id")
    for h in (5, 10, 20):
        for measure, column in (
            ("ppo_branch_return", f"ppo_q_h{h}"),
            ("ppo_future_latency", f"ppo_future_latency_h{h}"),
            ("ppo_future_queue_length", f"ppo_future_queue_h{h}"),
            ("ppo_future_backlog", f"ppo_future_backlog_h{h}"),
        ):
            critic_rows.append({"analysis": f"critic_value_vs_{measure}", "horizon": h,
                                "n": int(len(states)), "correlation": _safe_corr(states.critic_value, state_results[column]),
                                "cross_validated_r2": float("nan")})
    pd.DataFrame(critic_rows).to_csv(output_dir / "critic_long_term_analysis.csv", index=False)

    qrows = []
    pair_features = ["estimated_latency", "safe_set_size", "queue_j_seconds", "queue_k_seconds",
                     "total_backlog_seconds", "maximum_backlog_seconds", "reliability", "rho",
                     "requirement", "actor_probability"]
    for h in (5, 10, 20):
        r2 = _group_cv_r2(candidates[pair_features].to_numpy(dtype=float),
                          candidates[f"q_h{h}"].to_numpy(dtype=float),
                          candidates.state_id.to_numpy())
        qrows.append({"model": "grouped_ordinary_least_squares", "horizon": h,
                      "n_candidate_actions": int(len(candidates)), "grouped_5fold_r2": r2})
    pd.DataFrame(qrows).to_csv(output_dir / "q_predictability.csv", index=False)

    memory = summarize_queue_memory()
    memory.to_csv(output_dir / "queue_autocorrelation.csv", index=False)
    ext = safe_min_externality()
    ext.to_csv(output_dir / "safe_min_latency_externality.csv", index=False)
    _make_figures(output_dir, agreement, regret, memory, justification, state_results, qrows, effects)
    _write_report(output_dir, states, state_results, agreement, regret, effects,
                  half, justification, memory, ext, pd.DataFrame(critic_rows), pd.DataFrame(qrows),
                  replay, candidates)
    return state_results


def _make_figures(output, agreement, regret, memory, justification, state_results, qrows, effects):
    overall_a = agreement[agreement.stratification == "overall"].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(overall_a.horizon, overall_a.myopic_oracle_agreement_rate, marker="o")
    ax.set(xlabel="H-step horizon", ylabel="Myopic action in H-step optimal set", ylim=(0, 1)); fig.tight_layout(); fig.savefig(output/"agreement_vs_horizon.png"); plt.close(fig)
    overall_r = regret[regret.stratification == "overall"].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(overall_r.horizon, overall_r["mean"], marker="o", label="mean")
    ax.plot(overall_r.horizon, overall_r["median"], marker="s", label="median"); ax.set(xlabel="H-step horizon", ylabel="Safe Min-Latency regret"); ax.legend(); fig.tight_layout(); fig.savefig(output/"myopic_regret_vs_horizon.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    lags = [1,2,5,10,20]
    vals = [memory[f"lag_{x}_autocorrelation"].mean() for x in lags]
    ax.plot(lags, vals, marker="o"); ax.set(xlabel="Task-arrival lag", ylabel="Mean backlog autocorrelation", ylim=(-1,1)); fig.tight_layout(); fig.savefig(output/"queue_memory.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for choice in ("Next-best immediate", "PPO high probability", "PPO sampled"):
        part = effects[effects.choice == choice]
        if len(part):
            grouped = part.groupby("horizon").delta_total_backlog_seconds.apply(lambda x: float(np.nanmean(np.abs(x))))
            ax.plot(grouped.index, grouped.values, marker="o", label=choice)
    ax.set(xlabel="H-step horizon", ylabel="Mean absolute future backlog difference (s)")
    ax.legend(); fig.tight_layout(); fig.savefig(output/"action_effect_decay.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for source, group in justification.groupby("action_source"):
        ax.plot(group.horizon, group.long_term_justified_deviation_rate, marker="o", label=source)
    ax.set(xlabel="H-step horizon", ylabel="Long-term justified deviation rate", ylim=(0,1)); ax.legend(); fig.tight_layout(); fig.savefig(output/"ppo_justified_deviation_vs_horizon.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for req, group in state_results.groupby("requirement"):
        ax.plot(HORIZONS, [group[f"regret_h{h}"].mean() for h in HORIZONS], marker="o", label=str(req))
    ax.set(xlabel="H-step horizon", ylabel="Mean Safe Min-Latency regret"); ax.legend(title="R_req"); fig.tight_layout(); fig.savefig(output/"regret_by_requirement.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for load, group in state_results.groupby("load_tertile"):
        ax.plot(HORIZONS, [group[f"regret_h{h}"].mean() for h in HORIZONS], marker="o", label=load)
    ax.set(xlabel="H-step horizon", ylabel="Mean Safe Min-Latency regret"); ax.legend(title="load"); fig.tight_layout(); fig.savefig(output/"regret_by_load_level.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    grouped = state_results.groupby("safe_set_size")
    means = grouped[[f"regret_h{h}" for h in HORIZONS]].mean()
    if len(means):
        ax.plot(means.index, means["regret_h20"], marker="o")
    ax.set(xlabel="Safe-set size", ylabel="Mean H=20 regret"); fig.tight_layout(); fig.savefig(output/"regret_vs_safe_set_size.png"); plt.close(fig)
    if qrows:
        fig, ax = plt.subplots(figsize=(6,4)); qframe = pd.DataFrame(qrows)
        ax.plot(qframe.horizon, qframe.grouped_5fold_r2, marker="o"); ax.set(xlabel="H-step horizon", ylabel="Grouped CV R²"); fig.tight_layout(); fig.savefig(output/"q_predictability_vs_horizon.png"); plt.close(fig)
    h = pd.read_csv(output / "action_effect_half_life.csv")
    fig, ax = plt.subplots(figsize=(6,4))
    for col, label in (("half_life_50_steps","50%"),("half_life_25_steps","25%"),("half_life_10_steps","10%")):
        x = pd.to_numeric(h[col], errors="coerce").dropna()
        if len(x): ax.hist(x, bins=np.arange(0.5, 51.5, 1), alpha=.4, label=label)
    ax.set(xlabel="Subsequent task arrivals to decay threshold", ylabel="Action comparisons"); ax.legend(); fig.tight_layout(); fig.savefig(output/"action_effect_decay_histogram.png"); plt.close(fig)


def _write_report(output, states, state_results, agreement, regret, effects, half,
                  justification, memory, ext, critic, qpred, replay, candidates):
    overall_a = agreement[agreement.stratification == "overall"].set_index("horizon")
    overall_r = regret[regret.stratification == "overall"].set_index("horizon")
    sampled = justification[justification.action_source == "PPO sampled"].set_index("horizon")
    half_stats = half[["half_life_50_steps", "half_life_25_steps", "half_life_10_steps"]].apply(pd.to_numeric, errors="coerce")
    half_med = half_stats.median().to_dict() if len(half_stats) else {}
    high_load = state_results[state_results.load_tertile == "high"]
    ppo_dev = {}
    overall_ppo_gap20 = float(agreement[(agreement.stratification == "overall") & (agreement.horizon == 20)].ppo_sampled_mean_advantage.iloc[0])
    queue_memory_summary = {lag: float(memory[f"lag_{lag}_autocorrelation"].mean()) for lag in (1, 2, 5, 10, 20)}
    ppo_future_effect20 = effects[(effects.choice == "PPO sampled") & (effects.horizon == 20)]
    mean_abs_backlog_delta20 = float(np.nanmean(np.abs(ppo_future_effect20.delta_total_backlog_seconds)))
    mean_abs_latency_delta20 = float(np.nanmean(np.abs(ppo_future_effect20.delta_mean_future_task_latency_seconds)))
    for h in HORIZONS:
        sub = effects[(effects.choice == "PPO sampled") & (effects.horizon == h)]
        ppo_dev[h] = float(sampled.loc[h, "long_term_justified_deviation_rate"]) if h in sampled.index else float("nan")
    # Evidence-based, preregistered descriptive label: weak when regret is
    # negligible and agreement remains high; C only with a replicated PPO gain
    # in a stratum at 20/50 steps; otherwise B.
    h20 = overall_r.loc[20]
    agreement20 = float(overall_a.loc[20, "myopic_oracle_agreement_rate"])
    relative_regret = float(h20["mean"] / max(abs(float(state_results.myopic_q_h20.mean())), 1e-9))
    ppo_regimes = []
    for group_name, group in state_results.groupby(["load_tertile", "requirement"]):
        if len(group) < 20: continue
        for h in (20,50):
            group = group.copy()
            group["ppo_minus_myopic"] = group[f"ppo_q_h{h}"] - group[f"myopic_q_h{h}"]
            seed_means = group.groupby("trial_id").ppo_minus_myopic.mean()
            positive_seed_rate = float((seed_means > 0).mean()) if len(seed_means) else 0.0
            gap = group["ppo_minus_myopic"]
            if (float((gap > 1e-9).mean()) >= .65 and float(gap.mean()) > 0
                    and len(seed_means) >= 7 and positive_seed_rate >= .70):
                ppo_regimes.append((group_name, h, float(gap.mean()),
                                    float((gap > 1e-9).mean()), positive_seed_rate, int(len(seed_means))))
    if agreement20 >= .85 and relative_regret <= .01 and (not ppo_regimes):
        conclusion = "A. LONG-TERM COUPLING IS WEAK"
        final_line = "在本次正式配置与抽样状态中，Safe Min-Latency 的 H-step regret 很小且动作一致率高；一步安全最小时延动作通常接近长期最优，当前正式问题不足以证明必须使用 RL。当前论文主方法更适合聚焦 correlation-aware reliability-constrained offloading，PPO 可作为扩展方法。"
    elif ppo_regimes:
        conclusion = "C. PPO CAPTURES LONG-TERM VALUE IN SPECIFIC REGIMES"
        final_line = "总体结果之外，至少一个负载/需求分层在 H=20/50 下出现可重复的 PPO return 优势；应将该区域作为后续方法动机，而不要只看 pooled average。"
    else:
        conclusion = "B. LONG-TERM COUPLING EXISTS BUT PPO DOES NOT EXPLOIT IT"
        final_line = "H-step action branches show measurable long-term differences, but the sampled PPO deviations do not establish a repeatable advantage over the myopic heuristic."
    lines = [
        "# Long-Horizon Coupling Diagnosis", "",
        "## Design and validity", "",
        f"Evaluated {len(states)} sampled decision states from the first evaluation episode of each of the 10 formal seed rows; all states come from the current formal 20×200 evaluation schedule. For each sampled state, every exact production safe action was branched (or the shared max-reliability fallback set when the safe set was empty). Each branch used the same frozen Masked PPO stochastic policy after one forced action, the same arrival/spatial-risk seeds, and the same Torch action RNG seed.",
        "", "A live SimPy environment cannot be deep-copied because its event queue contains Python generator objects (`TypeError: cannot pickle 'generator' object`). The diagnostic therefore uses deterministic prefix replay from episode start as snapshot/restore: before each intervention it requires an exact digest match for the 35-D observation, CPU waiting/running/resource queues, pending tasks, SimPy event calendar, effective hazards, spatial field/correlation/distance, and RNG states; it also requires identical pre-intervention actions and arrivals. A failed check aborts the run. The original evaluation action/reward/delay/reliability replay is checked against the saved formal files.",
        "", f"Horizons are arrival-decision intervals H={', '.join(map(str,HORIZONS))}; interval rewards are discounted by the formal PPO γ={params.gamma_ppo} per decision interval. Outcomes resolved during each interval are assigned to that interval, including outcomes of tasks admitted before the target snapshot. Episode-terminal-shortened horizons record their effective length. Counterfactual Q is a diagnostic finite-horizon upper bound under the fixed downstream PPO policy, not a deployable policy.",
        "", "## Main results", "",
        "| H | Effective H mean | Myopic-oracle agreement | Mean regret | Median | P90 | P95 | P(regret≤0.01) | P(regret≤0.1) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in HORIZONS:
        r, a = overall_r.loc[h], overall_a.loc[h]
        lines.append(f"| {h} | {state_results[f'effective_h_h{h}'].mean():.1f} | {a.myopic_oracle_agreement_rate:.3f} | {r['mean']:.4f} | {r['median']:.4f} | {r['p90']:.4f} | {r['p95']:.4f} | {r['p_le_0_01']:.3f} | {r['p_le_0_1']:.3f} |")
    agreement_text = ", ".join(
        f"{float(overall_a.loc[h, 'myopic_oracle_agreement_rate']):.1%}"
        for h in HORIZONS
    )
    lines += ["", "### Required questions", "",
        f"**Q1 — How long does one pair action affect queues/latency?** The median threshold crossing of action-induced total-backlog differences is 50%: {half_med.get('half_life_50_steps', float('nan')):.1f} arrivals; 25%: {half_med.get('half_life_25_steps', float('nan')):.1f}; 10%: {half_med.get('half_life_10_steps', float('nan')):.1f}. The half-life is measured against the Safe Min-Latency branch, with early peak difference (first five arrivals) as the normalization; {float(half.right_censored.mean()):.1%} did not cross any threshold by H=50. Queue-memory autocorrelation falls from lag-1 {queue_memory_summary[1]:.3f} to lag-5 {queue_memory_summary[5]:.3f} and lag-20 {queue_memory_summary[20]:.3f}. At H=20, PPO-sampled branches have mean absolute backlog difference {mean_abs_backlog_delta20:.3f}s and mean absolute task-latency difference {mean_abs_latency_delta20:.3f}s from Safe Min-Latency.",
        f"**Q2 — Agreement?** At H=1/5/10/20/50, the myopic action belongs to the H-step oracle optimal set in {agreement_text} of sampled states, respectively; empty-safe-set cases use the maximum-reliability fallback candidate set. Exact oracle ties are handled as a set with 1e-9 reward tolerance.",
        f"**Q3 — Myopic regret?** At H=20, mean regret is {overall_r.loc[20, 'mean']:.4f}, P95 is {overall_r.loc[20, 'p95']:.4f}, and {overall_r.loc[20, 'p_le_0_01']:.1%}/{overall_r.loc[20, 'p_le_0_1']:.1%} of states are within 0.01/0.1 return units. At H=50, mean regret is {overall_r.loc[50, 'mean']:.4f}.",
        f"**Q4 — Are PPO deviations justified?** For the sampled stochastic action, among states where PPO's decision-time latency estimate exceeds the myopic minimum, the fraction with larger Q_H is {', '.join(f'H={h}: {ppo_dev[h]:.1%}' for h in HORIZONS)}. At H=20 the mean sampled-PPO Q gap is {overall_ppo_gap20:.4f} versus Safe Min-Latency. The high-probability/greedy actor action is separately included in `ppo_long_term_justification.csv`.",
        f"**Q5 — Critic long-term information?** A deterministic grouped 5-fold linear model predicts Critic V(s) from current observable state/backlog and minimum safe latency with R²={critic.iloc[0].cross_validated_r2:.3f}. Correlations between V(s), sampled-PPO branch return, and future queue/backlog/latency appear in `critic_long_term_analysis.csv`; these are associations, not proof of what the network represents.",
        f"**Q6 — Where could RL matter?** High-load H=20 mean regret is {high_load.regret_h20.mean():.4f} over {len(high_load)} states. Requirement and safe-set strata are reported separately in `load_stratified_results.csv`, `requirement_stratified_results.csv`, and `safe_set_size_analysis.csv`; PPO deviation rates by load/requirement are in `ppo_justification_by_stratum.csv`. Regime-specific PPO candidate return advantages meeting the 7-of-10-seed consistency criterion: {ppo_regimes if ppo_regimes else 'none found'}.",
        f"**Q7 — Is coupling strong enough to support RL?** {conclusion}. {final_line}",
        "", "## PPO disagreement and externalities", "",
        "`ppo_vs_myopic_disagreement.csv` compares Masked PPO sampled/high-probability actions against the Safe Min-Latency action on identical snapshot states, including estimated latency, reliability, rho, and Q_H differences. `action_future_queue_effect.csv` gives paired future queue length/backlog, utilization, wait, latency, and task-reward changes for Safe Min-Latency, its nearest-latency competitor, PPO high-probability, and PPO sampled choices. `oracle_improvement_ceiling.csv` reports the finite-horizon Q_H headroom, plus latency/reward outcomes of the Q_H-optimal action set (these outcome deltas are descriptive and may be negative because the objective is total interval reward). `safe_min_latency_externality.csv` evaluates endpoint reselection and selected-pair backlog change over the next 5/10/20 decisions using the complete formal evaluation traces.",
        "", "## Queue memory and Q predictability", "",
        "Queue memory is measured as per-server service-backlog (seconds) autocorrelation at 1/2/5/10/20 task-arrival lags across all 20 formal episodes and 10 seeds. Future backlog/latency correlations are in `queue_autocorrelation.csv`. Linear grouped 5-fold models predict candidate Q_H from decision-time latency, safe-set size, selected-server queues, reliability, rho, and task requirement; R² by H is in `q_predictability.csv`.",
        "", "## Files", "",
        "The requested CSVs and figures are in this directory. `counterfactual_state_results.csv` and `replay_checks.csv` provide state/action-level audit data. `replay_manifest.json` files record per-seed sample counts and checkpoint hashes. Oracle reward and latency ceilings compare the best-Q_H action branches with the myopic branch for tasks arriving inside each finite horizon.",
        "", "## Limits", "",
        "Only 1000 states are counterfactually branched, stratified across the 10 formal seed rows, requirements, and within-seed max-backlog tertiles. This provides causal action comparisons for the current fixed downstream policy; it is not an optimal continuation-policy oracle, and its action-value ceiling is consequently conditional on the downstream PPO. States come from episode 1 per formal seed to keep external RNG snapshot restoration exact and bounded. Results should not be generalized beyond this simulator configuration.",
    ]
    lines += ["", "### H=20 strata (paired state comparisons)", "",
              "| Stratum | Group | n | Mean safe-set size | Agreement | Mean regret | P90 regret | P95 regret | PPO sampled mean Q gap | PPO positive-gap rate |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in ("load", "requirement", "safe_set_size"):
        subset = agreement[(agreement.stratification == label) & (agreement.horizon == 20)]
        for _, row in subset.iterrows():
            lines.append(
                f"| {label} | {row['group']} | {int(row['n'])} | {row['mean_safe_set_size']:.2f} | "
                f"{row['myopic_oracle_agreement_rate']:.3f} | {row['mean']:.4f} | {row['p90']:.4f} | "
                f"{row['p95']:.4f} | {row['ppo_sampled_mean_advantage']:.4f} | "
                f"{row['ppo_sampled_positive_advantage_rate']:.3f} |"
            )
    lines += ["", "The H=20 PPO advantage columns are sampled-action Q_H minus Safe Min-Latency Q_H on the same snapshot, and the positive-gap rate is descriptive. Empty-safe-set states use the maximum-reliability fallback action set."]
    (output / "LONG_HORIZON_COUPLING_DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir=OUT, states=STATE_COUNT, workers=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, seed_plan = formal_spec()
    nstates = int(states)
    if nstates < 1000 and workers != 0:
        # CLI smoke mode is explicitly separated below; standard runs must be formal-scale.
        raise ValueError("Formal run requires at least 1000 counterfactual states")
    per_trial = [nstates // len(seed_plan)] * len(seed_plan)
    for i in range(nstates % len(seed_plan)):
        per_trial[i] += 1
    payloads = [(int(i), int(per_trial[i]), str(output_dir)) for i in range(len(seed_plan)) if per_trial[i] > 0]
    if workers <= 1:
        trial_dirs = [_trial_worker(*payload) for payload in payloads]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            trial_dirs = list(pool.map(_trial_worker, *zip(*payloads)))
    aggregate(output_dir, trial_dirs)
    metadata = {
        "source_formal_commit": meta["git_commit"],
        "formal_reference_metadata_sha256": sha256_file(PPO_OUT / "formal_reference_metadata.json"),
        "server_info_sha256": meta["server_info_sha256"],
        "task_parameters_sha256": meta["task_parameters_sha256"],
        "diagnostic_script_sha256": sha256_file(Path(__file__)),
        "formal_trial_count": int(len(seed_plan)),
        "sampled_state_count": int(nstates), "states_per_trial": per_trial,
        "evaluation_stream_seeds": seed_plan[["Trial_ID", "Eval_Arrival_Seed", "Eval_Spatial_Seed"]].to_dict(orient="records"),
        "horizons": list(HORIZONS), "gamma": float(params.gamma_ppo),
        "max_reliability_counterfactual_actions": "all safe actions; max-reliability fallback actions only for empty safe sets",
        "downstream_policy": "frozen stochastic Masked Pair PPO",
        "simulator_reward_policy_or_environment_modified": False,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def run_smoke(output_dir=OUT / "smoke", states=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta, seed_plan = formal_spec()
    trial = seed_plan.iloc[0].to_dict()
    pairs, rho = build_pair_correlations()
    pairs, rho = list(pairs), np.asarray(rho, dtype=float)
    agent, *_ = _make_agent(0, trial, pairs, rho)
    snapshots, actions, arrivals, hazards = _baseline_trial(0, trial, agent, pairs, rho)
    selected = _stratified_sample(snapshots, int(states), 0)
    checks = []
    for snapshot in selected:
        mask = np.asarray(snapshot["safe_mask"], dtype=bool)
        effective = np.asarray(snapshot["effective_mask"], dtype=bool)
        candidate = np.flatnonzero(mask if mask.any() else effective)
        action = int(candidate[0])
        first = _run_branch(agent, trial, snapshot, action, actions, arrivals, hazards, snapshot["task_id"])
        second = _run_branch(agent, trial, snapshot, action, actions, arrivals, hazards, snapshot["task_id"])
        if first["qvalues"] != second["qvalues"] or first["branch_action_trace"] != second["branch_action_trace"]:
            raise RuntimeError("Repeated identical counterfactual branch was not deterministic")
        checks.append({"state_id": snapshot["state_id"], "snapshot_match": True,
                       "repeated_branch_match": True, "external_stream_match": True,
                       "action_prefix_match": True})
    pd.DataFrame(checks).to_csv(output_dir / "smoke_replay_checks.csv", index=False)
    return pd.DataFrame(checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--states", type=int, default=STATE_COUNT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        print(run_smoke(args.output_dir / "smoke", states=min(args.states, 10)).to_string(index=False))
    else:
        print(run(args.output_dir, states=args.states, workers=args.workers))


if __name__ == "__main__":
    main()
