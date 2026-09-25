"""Run/analyze paired non-learning reliability baselines on the formal seeds.

This evaluation-only tool reuses the ten formal environment seeds and frozen
Masked Pair PPO checkpoints. It does not modify simulator or learning code.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import sys
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
    ReliabilityMaskedPairPPOAgent, effective_action_mask,
    production_reliability_vector,
)
from config.params import params
from core.main_loop import MainLoop
from diagnostics.evaluate_reliability_masked_policy import (
    ArrivalTraceLoop, TrackingEnvironmentState, summarize_server_load,
)
from diagnostics.external_reliability_baselines import (
    deterministic_policy_rng, estimate_pair_completion_latencies,
    max_reliability_choice, safe_min_latency_choice, safe_random_choice,
    validate_evaluation_count,
)
from diagnostics.run_masked_pair_ppo import trace
from diagnostics.run_masked_pair_ppo_10seed import OUT as PPO_OUT, action_seed, formal_spec
from tools.pair_policy_diagnostics import TASK_ASSIGNMENT_COLUMNS
from tools.paired_ppo_experiment import (
    _agent_kwargs, _state_dict_snapshot, assert_state_dict_unchanged,
    scoped_environment_seeds, sha256_file,
)
from Project_main import build_pair_correlations

OUT = ROOT / "diagnostics/results/external_baselines"
METHODS = (
    "Max-Reliability", "Safe Min-Latency", "Safe Random",
    "Masked PPO stochastic", "Pair PPO greedy", "Pair PPO stochastic",
    "Masked PPO greedy",
)
BASELINE_TAGS = {
    "Max-Reliability": 0x4D415852,
    "Safe Min-Latency": 0x4D494E4C,
    "Safe Random": 0x53414645,
}
COMPARE_METRICS = (
    ("mean_reward", "higher"), ("mean_latency", "lower"),
    ("p95_latency", "lower"), ("overall_rsr", "higher"),
    ("highest_rsr", "higher"), ("pair_selection_hhi", "lower"),
    ("maximum_mean_queue_length", "lower"),
)


def _finite(frame, label):
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{label} contains NaN or Inf")


class EvaluationOnlyHeuristicAgent:
    """Minimal MainLoop SMDP adapter: no actor, critic, optimizer, or learning."""

    def __init__(self, policy, pairs, pair_rho, rng):
        self.policy = str(policy)
        self.pairs = list(pairs)
        self.pair_rho = np.asarray(pair_rho, dtype=float)
        self.rng = rng
        self.current_context = None
        self.selection_archive = []
        self._transitions = {}
        self._rewards = {}

    def prepare_action(self, task, env_state, episode, state):
        return None

    def select_action(self, state, epsilon=0.0):
        if self.current_context is None:
            raise RuntimeError("No current task context")
        task, env_state, episode = self.current_context
        reliability = production_reliability_vector(task, env_state, self.pairs)
        latency, per_server = estimate_pair_completion_latencies(task, env_state, self.pairs)
        safe, effective, empty, best = effective_action_mask(
            reliability, task.reliability_requirement
        )
        max_rel = np.isclose(reliability, best, rtol=0.0, atol=1e-12)
        if self.policy == "Max-Reliability":
            action, max_rel = max_reliability_choice(reliability, self.rng)
        elif self.policy == "Safe Min-Latency":
            action, safe, effective, empty, best, _ = safe_min_latency_choice(
                reliability, task.reliability_requirement, latency, self.rng
            )
        elif self.policy == "Safe Random":
            action, safe, effective, empty, best = safe_random_choice(
                reliability, task.reliability_requirement, self.rng
            )
        else:
            raise ValueError(f"Unknown heuristic: {self.policy}")
        self.selection_archive.append(_decision_record(
            self.policy, task, env_state, episode, self.pairs, reliability, latency,
            per_server, safe, effective, empty, best, max_rel, action, self.pair_rho,
        ))
        return int(action)

    def store_transition(self, s, a, r, s_next, delta_t, done=False, task_id=None):
        task_id = int(task_id)
        if task_id in self._transitions:
            raise RuntimeError("Duplicate heuristic transition")
        self._transitions[task_id] = int(a)

    def assign_task_reward(self, task_id, reward):
        task_id = int(task_id)
        if task_id in self._rewards:
            raise RuntimeError("Duplicate heuristic task reward")
        self._rewards[task_id] = float(reward)

    def train_step(self):
        expected = set(range(1, 201))
        if set(self._transitions) != expected or set(self._rewards) != expected:
            raise RuntimeError("Heuristic evaluation did not resolve all 200 episode tasks")
        self._transitions.clear()
        self._rewards.clear()


def _decision_record(policy, task, env_state, episode, pairs, reliability, latency,
                     per_server, safe, effective, empty, best, max_rel, action, rho):
    j, k = pairs[int(action)]
    record = {
        "policy": policy, "episode": int(episode), "task_id": int(task.id),
        "decision_time": float(task.env.now), "action_index": int(action),
        "selected_pair": str((int(j), int(k))), "server_j": int(j), "server_k": int(k),
        "R_req": float(task.reliability_requirement),
        "safe_set_size": int(np.asarray(safe, dtype=bool).sum()),
        "effective_set_size": int(np.asarray(effective, dtype=bool).sum()),
        "safe_set_empty": bool(empty),
        "max_reliability_tie_count": int(np.asarray(max_rel, dtype=bool).sum()),
        "best_achievable_reliability": float(best),
        "reliability_deficit": max(float(task.reliability_requirement) - float(best), 0.0),
        "selected_pair_reliability": float(reliability[action]),
        "selected_rho": float(rho[action]),
        "estimated_latency": float(latency[action]),
        "estimated_copy_latency_j": float(per_server[int(j)]),
        "estimated_copy_latency_k": float(per_server[int(k)]),
        "selected_action_safe": bool(safe[action]),
        "safe_mask": json.dumps(np.asarray(safe, dtype=int).tolist()),
        "effective_mask": json.dumps(np.asarray(effective, dtype=int).tolist()),
    }
    for sid in sorted(env_state.servers):
        record[f"backlog_server_{sid}"] = float(
            env_state.get_server_backlog_time(int(sid), task.env.now)
        )
    record["decision_backlog_mean_selected"] = float(np.mean([
        record[f"backlog_server_{int(j)}"], record[f"backlog_server_{int(k)}"]
    ]))
    return record


class ObservableMaskedPPOAgent(ReliabilityMaskedPairPPOAgent):
    """Frozen Masked PPO with read-only queue/latency decision telemetry."""

    def __init__(self, *args, rho, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_rho = np.asarray(rho, dtype=float)
        self.telemetry_archive = []

    def select_action(self, state, epsilon=0.0, use_softmax=False, temperature=1.5):
        context = self.current_decision
        if context is None or self.current_context is None:
            raise RuntimeError("Masked telemetry requires a prepared decision context")
        task, env_state, episode = self.current_context
        reliability = np.asarray(context["reliabilities"], dtype=float)
        latency, per_server = estimate_pair_completion_latencies(task, env_state, self.pairs)
        safe = np.asarray(context["safe_mask"], dtype=bool)
        effective = np.asarray(context["effective_mask"], dtype=bool)
        empty = bool(context["safe_set_empty"])
        best = float(context["best_achievable_reliability"])
        max_rel = np.isclose(reliability, best, rtol=0.0, atol=1e-12)
        action = super().select_action(state, epsilon, use_softmax, temperature)
        self.telemetry_archive.append(_decision_record(
            "Masked PPO stochastic", task, env_state, episode, self.pairs,
            reliability, latency, per_server, safe, effective, empty, best,
            max_rel, action, self.pair_rho,
        ))
        return action


def _add_outcomes_and_queue_deltas(decisions, assignments):
    decision = decisions.sort_values(["episode", "task_id"]).reset_index(drop=True).copy()
    fields = [
        "episode", "task_id", "Task_Delay", "Task_Reward",
        "Execution_Reliability", "Reliability_Satisfied", "Reliability_Requirement",
    ]
    result = decision.merge(
        assignments[fields], on=["episode", "task_id"], validate="one_to_one"
    )
    if result.Task_Delay.isna().any() or result.Task_Reward.isna().any():
        raise RuntimeError("Decision telemetry has no matching realized task outcomes")
    servers = sorted(int(c.rsplit("_", 1)[1]) for c in result if c.startswith("backlog_server_"))
    has_next = result.groupby("episode").task_id.transform("max") > result.task_id
    result["next_decision_exists"] = has_next.astype(bool)
    for sid in servers:
        col = f"backlog_server_{sid}"
        nxt = f"next_decision_backlog_server_{sid}"
        result[nxt] = result.groupby("episode", sort=False)[col].shift(-1).fillna(0.0)
        result[f"next_decision_backlog_delta_server_{sid}"] = result[nxt] - result[col]
    next_means = np.zeros(len(result), dtype=float)
    next_deltas = np.zeros(len(result), dtype=float)
    prior_j = np.zeros(len(result), dtype=int)
    prior_k = np.zeros(len(result), dtype=int)
    row_position = {idx: pos for pos, idx in enumerate(result.index)}
    for _, group in result.groupby("episode", sort=False):
        counts = {sid: 0 for sid in servers}
        for idx, row in group.iterrows():
            pos, j, k = row_position[idx], int(row.server_j), int(row.server_k)
            prior_j[pos], prior_k[pos] = counts[j], counts[k]
            if bool(row.next_decision_exists):
                next_mean = np.mean([row[f"next_decision_backlog_server_{j}"],
                                     row[f"next_decision_backlog_server_{k}"]])
                current_mean = np.mean([row[f"backlog_server_{j}"], row[f"backlog_server_{k}"]])
                next_means[pos] = next_mean
                next_deltas[pos] = next_mean - current_mean
            counts[j] += 1
            counts[k] += 1
    result["prior_endpoint_selections_server_j"] = prior_j
    result["prior_endpoint_selections_server_k"] = prior_k
    result["next_decision_selected_pair_mean_backlog"] = next_means
    result["next_decision_selected_pair_backlog_delta"] = next_deltas
    # The full masks are already checked against the saved formal PPO replay;
    # omitting their repeated JSON vectors keeps the persisted telemetry compact.
    result = result.drop(columns=["safe_mask", "effective_mask"], errors="ignore")
    _finite(result.drop(columns=["selected_pair"], errors="ignore"), "decision telemetry")
    return result


def _metrics(assignments, decisions, load, trial_id, policy):
    frame = assignments.sort_values(["episode", "task_id"]).reset_index(drop=True)
    select = decisions.sort_values(["episode", "task_id"]).reset_index(drop=True)
    validate_evaluation_count(frame)
    if len(select) != len(frame) or not np.array_equal(select.action_index, frame.action_index):
        raise RuntimeError(f"{policy}: assignment and decision actions differ")
    req = frame.Reliability_Requirement.astype(float).to_numpy()
    rel = frame.Execution_Reliability.astype(float).to_numpy()
    satisfied = frame.Reliability_Satisfied.astype(bool).to_numpy()
    feasible = ~select.safe_set_empty.astype(bool).to_numpy()
    chosen_deficit = np.maximum(req - rel, 0.0)
    best_deficit = np.maximum(req - select.best_achievable_reliability.astype(float).to_numpy(), 0.0)
    avoidable = feasible & ~satisfied
    unavoidable = ~feasible & ~satisfied
    pair_counts = frame.action_index.astype(int).value_counts().reindex(
        range(params.num_actions), fill_value=0
    ).to_numpy(dtype=float)
    p = pair_counts / pair_counts.sum()
    nonzero = p[p > 0]
    server_counts = np.bincount(
        np.r_[frame.Primary.astype(int).to_numpy(), frame.Backup.astype(int).to_numpy()] - 1,
        minlength=params.serverNo,
    )
    server_share = server_counts / (2.0 * len(frame))
    load = load.sort_values("server_id")
    high = np.isclose(req, 0.9999, rtol=0.0, atol=1e-12)
    row = {
        "trial_id": int(trial_id), "policy": policy, "tasks": len(frame),
        "mean_reward": float(frame.Task_Reward.mean()),
        "mean_latency": float(frame.Task_Delay.mean()),
        "p50_latency": float(frame.Task_Delay.quantile(.5)),
        "p90_latency": float(frame.Task_Delay.quantile(.9)),
        "p95_latency": float(frame.Task_Delay.quantile(.95)),
        "overall_rsr": float(satisfied.mean()), "highest_rsr": float(satisfied[high].mean()),
        "feasibility_rate": float(feasible.mean()),
        "conditional_rsr": float(satisfied[feasible].mean()) if feasible.any() else 0.0,
        "conditional_rsr_defined": bool(feasible.any()),
        "avoidable_violation_count": int(avoidable.sum()),
        "avoidable_violation_rate": float(avoidable.sum() / feasible.sum()) if feasible.any() else 0.0,
        "unavoidable_violation_count": int(unavoidable.sum()),
        "unavoidable_violation_rate": float(unavoidable.mean()),
        "mean_selected_reliability_deficit": float(chosen_deficit.mean()),
        "p95_selected_reliability_deficit": float(np.quantile(chosen_deficit, .95)),
        "mean_best_achievable_deficit": float(best_deficit.mean()),
        "p95_best_achievable_deficit": float(np.quantile(best_deficit, .95)),
        "mean_safe_set_size": float(select.safe_set_size.mean()),
        "pair_selection_hhi": float(np.square(p).sum()),
        "pair_selection_entropy": float(-(nonzero * np.log(nonzero)).sum()),
        "top1_pair_frequency": float(p.max()),
        "unique_selected_pairs": int(np.count_nonzero(pair_counts)),
        "pair_78_frequency": float((frame.action_index == params.num_actions - 1).mean()),
        "maximum_server_selection_share": float(server_share.max()),
        "maximum_server_utilization": float(load.utilization.max()),
        "maximum_mean_queue_length": float(load.mean_queue_length.max()),
        "maximum_p95_queue_length": float(load.p95_queue_length.max()),
        "mean_server_waiting_time": float(load.mean_waiting_time.mean()),
    }
    if row["tasks"] != 4000:
        raise RuntimeError(f"{policy}: formal evaluation requires exactly 4000 tasks")
    if policy in ("Safe Min-Latency", "Safe Random", "Masked PPO stochastic") and row["avoidable_violation_count"]:
        raise RuntimeError(f"{policy} produced an avoidable reliability violation")
    _finite(pd.DataFrame([row]).drop(columns=["policy"]), f"{policy} seed summary")
    return row


def _requirement_rows(assignments, decisions, trial_id, policy):
    fields = ["episode", "task_id", "safe_set_size", "safe_set_empty", "best_achievable_reliability"]
    df = assignments.merge(decisions[fields], on=["episode", "task_id"], validate="one_to_one")
    result = []
    for req, frame in df.groupby("Reliability_Requirement", sort=True):
        sat = frame.Reliability_Satisfied.astype(bool).to_numpy()
        feasible = ~frame.safe_set_empty.astype(bool).to_numpy()
        chosen_deficit = np.maximum(
            frame.Reliability_Requirement.astype(float).to_numpy()
            - frame.Execution_Reliability.astype(float).to_numpy(), 0.0
        )
        best_deficit = np.maximum(
            frame.Reliability_Requirement.astype(float).to_numpy()
            - frame.best_achievable_reliability.astype(float).to_numpy(), 0.0
        )
        counts = frame.action_index.astype(int).value_counts().to_numpy(dtype=float)
        p = counts / len(frame)
        result.append({
            "trial_id": int(trial_id), "policy": policy, "R_req": float(req), "tasks": len(frame),
            "mean_reward": float(frame.Task_Reward.mean()), "mean_latency": float(frame.Task_Delay.mean()),
            "p50_latency": float(frame.Task_Delay.quantile(.5)), "p90_latency": float(frame.Task_Delay.quantile(.9)),
            "p95_latency": float(frame.Task_Delay.quantile(.95)), "overall_rsr": float(sat.mean()),
            "feasibility_rate": float(feasible.mean()),
            "conditional_rsr": float(sat[feasible].mean()) if feasible.any() else 0.0,
            "avoidable_violation_rate": float((feasible & ~sat).sum() / feasible.sum()) if feasible.any() else 0.0,
            "unavoidable_violation_rate": float((~feasible & ~sat).mean()),
            "mean_selected_reliability_deficit": float(chosen_deficit.mean()),
            "p95_selected_reliability_deficit": float(np.quantile(chosen_deficit, .95)),
            "mean_best_achievable_deficit": float(best_deficit.mean()),
            "mean_safe_set_size": float(frame.safe_set_size.mean()),
            "pair_selection_hhi": float(np.square(p).sum()),
        })
    return result


def _policy_slug(policy):
    return {
        "Max-Reliability": "max_reliability",
        "Safe Min-Latency": "safe_min_latency",
        "Safe Random": "safe_random",
        "Masked PPO stochastic": "masked_stochastic_replay",
        "Pair PPO greedy": "pair_greedy_reference",
        "Pair PPO stochastic": "pair_stochastic_reference",
        "Masked PPO greedy": "masked_greedy_reference",
    }[policy]


def _save_new_policy(policy, trial, agent, output_dir, reference_trace=None, replay_check=False):
    tid = int(trial["Trial_ID"])
    run_dir = output_dir / "runs" / f"trial_{tid:03d}" / _policy_slug(policy)
    run_dir.mkdir(parents=True, exist_ok=True)
    TrackingEnvironmentState.instances = []
    TrackingEnvironmentState.active_agent = agent
    before = None
    if policy == "Masked PPO stochastic":
        torch.manual_seed(action_seed(trial))
        before = _state_dict_snapshot(agent)
    with patch.object(main_loop_module, "EnvironmentState", TrackingEnvironmentState):
        with scoped_environment_seeds(int(trial["Eval_Arrival_Seed"]), int(trial["Eval_Spatial_Seed"])):
            with open(os.devnull, "w") as null, redirect_stdout(null):
                loop = ArrivalTraceLoop(agent, 20, 200, params.num_states, params.num_actions)
                loop.EP()
    if before is not None:
        assert_state_dict_unchanged(before, agent)
    for state in TrackingEnvironmentState.instances:
        state.finalize()
    assignments = pd.DataFrame(loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS)
    assignments = assignments.sort_values(["episode", "task_id"]).reset_index(drop=True)
    validate_evaluation_count(assignments)
    archived = agent.telemetry_archive if policy == "Masked PPO stochastic" else agent.selection_archive
    decisions = pd.DataFrame(archived).sort_values(["episode", "task_id"]).reset_index(drop=True)
    if len(decisions) != 4000 or not np.array_equal(decisions.action_index, assignments.action_index):
        raise RuntimeError(f"{policy}: 4000 decisions do not match assignments")
    actual_trace = trace(loop)
    if reference_trace is not None and (
        not np.array_equal(reference_trace[0], actual_trace[0]) or not reference_trace[1].equals(actual_trace[1])
    ):
        raise RuntimeError(f"{policy}: exogenous arrival/spatial-risk streams differ")
    if replay_check:
        reference = PPO_OUT / "runs" / f"trial_{tid:03d}" / "masked_stochastic"
        old_assignments = pd.read_csv(reference / "evaluation_task_assignments.csv")
        old_decisions = pd.read_csv(reference / "evaluation_decisions.csv")
        for col in ("episode", "task_id", "action_index"):
            if not np.array_equal(assignments[col], old_assignments[col]):
                raise RuntimeError(f"Masked PPO replay mismatch in {col} for trial {tid}")
        for col in ("Task_Reward", "Task_Delay", "Execution_Reliability"):
            if not np.allclose(assignments[col], old_assignments[col], rtol=0, atol=1e-12):
                raise RuntimeError(f"Masked PPO replay mismatch in {col} for trial {tid}")
        for current, old in zip(decisions.safe_mask, old_decisions.safe_mask):
            if not np.array_equal(json.loads(current), json.loads(old)):
                raise RuntimeError(f"Masked PPO safe-mask replay mismatch for trial {tid}")
        (run_dir / "replay_manifest.json").write_text(json.dumps({
            "trial_id": tid, "matched_reference_actions": True, "matched_safe_masks": True,
            "matched_task_outcomes": True,
        }, indent=2))
    telemetry = _add_outcomes_and_queue_deltas(decisions, assignments)
    load = summarize_server_load(TrackingEnvironmentState.instances, assignments, policy)
    load.insert(0, "trial_id", tid)
    _finite(load.drop(columns=["policy"]), f"{policy} server load")
    return assignments, decisions, telemetry, load, actual_trace, run_dir


def _make_masked_agent(trial_id, trial, pairs, rho):
    directory = PPO_OUT / "runs" / f"trial_{trial_id:03d}"
    completed = json.loads((directory / "completed.json").read_text())
    manifest = next(r for r in completed["runs"] if r["policy"] == "masked_stochastic")
    actor_path, critic_path = directory / "masked" / "actor.pt", directory / "masked" / "critic.pt"
    if sha256_file(actor_path) != manifest["actor_sha256"] or sha256_file(critic_path) != manifest["critic_sha256"]:
        raise RuntimeError(f"Trial {trial_id}: masked checkpoint hash mismatch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(trial["Torch_Init_Seed"]))
        agent = ObservableMaskedPPOAgent(
            **_agent_kwargs("pair_scoring", rho, int(trial["PPO_Minibatch_Seed"])),
            deployment_mode="stochastic", frozen=True, rho=rho,
        )
    actor = torch.load(actor_path, map_location="cpu", weights_only=True)
    critic = torch.load(critic_path, map_location="cpu", weights_only=True)
    agent.policy_net.load_state_dict(actor)
    agent.policy_old.load_state_dict(actor)
    agent.value_net.load_state_dict(critic)
    agent.clear_rollout()
    if agent.pairs != list(pairs):
        raise RuntimeError("Masked checkpoint action order differs from current pair enumeration")
    return agent, manifest


def _load_reference(trial_id, policy):
    if policy == "Pair PPO stochastic":
        folder = PPO_OUT / "pair_stochastic" / f"trial_{trial_id:03d}"
    else:
        child = {"Masked PPO stochastic": "masked_stochastic",
                 "Masked PPO greedy": "masked_greedy", "Pair PPO greedy": "pair"}[policy]
        folder = PPO_OUT / "runs" / f"trial_{trial_id:03d}" / child
    assignments = pd.read_csv(folder / "evaluation_task_assignments.csv")
    decisions = pd.read_csv(folder / "evaluation_decisions.csv")
    load = pd.read_csv(folder / "server_load.csv")
    assignments = assignments.sort_values(["episode", "task_id"]).reset_index(drop=True)
    decisions = decisions.sort_values(["episode", "task_id"]).reset_index(drop=True)
    validate_evaluation_count(assignments)
    if "best_achievable_reliability" not in decisions:
        # All existing formal evaluation logs contain this field, this branch
        # keeps an explicit error if a future artifact schema drops it.
        raise RuntimeError(f"Reference {policy} is missing best-achievable reliability")
    load["trial_id"], load["policy"] = int(trial_id), policy
    return assignments, decisions, load


def _evaluate_one_trial(payload):
    tid, output_dir = int(payload[0]), Path(payload[1])
    torch.set_num_threads(1)
    _, plan = formal_spec()
    trial = plan.iloc[tid].to_dict()
    pairs, rho = build_pair_correlations()
    pairs, rho = list(pairs), np.asarray(rho, dtype=float)
    if pairs != MainLoop.generate_combinations() or len(pairs) != params.num_actions:
        raise RuntimeError("Current production legal action order does not match the formal experiment")
    expected_trace = None
    seed_rows, tier_rows, pair_rows, load_frames = [], [], [], []
    telemetry_frames, tie_frames = [], []
    for policy in ("Max-Reliability", "Safe Min-Latency", "Safe Random", "Masked PPO stochastic"):
        if policy == "Masked PPO stochastic":
            agent, checkpoint_manifest = _make_masked_agent(tid, trial, pairs, rho)
            check_replay = True
        else:
            agent = EvaluationOnlyHeuristicAgent(
                policy, pairs, rho, deterministic_policy_rng(trial, BASELINE_TAGS[policy])
            )
            checkpoint_manifest, check_replay = None, False
        assignments, decisions, telemetry, load, current_trace, run_dir = _save_new_policy(
            policy, trial, agent, output_dir, expected_trace, check_replay
        )
        if expected_trace is None:
            expected_trace = current_trace
        if policy == "Masked PPO stochastic":
            (run_dir / "checkpoint_replay.json").write_text(json.dumps({
                "trial_id": tid,
                "actor_sha256": checkpoint_manifest["actor_sha256"],
                "critic_sha256": checkpoint_manifest["critic_sha256"],
                "checkpoint_unchanged": True, "replay_match": True,
            }, indent=2))
        seed_rows.append(_metrics(assignments, decisions, load, tid, policy))
        tier_rows.extend(_requirement_rows(assignments, decisions, tid, policy))
        if policy == "Max-Reliability":
            ties = decisions[["episode", "task_id", "max_reliability_tie_count",
                             "best_achievable_reliability", "action_index", "selected_pair"]].copy()
            ties.insert(0, "trial_id", tid)
            ties.rename(columns={"max_reliability_tie_count": "number_of_max_reliability_ties"}, inplace=True)
            tie_frames.append(ties)
        load_frames.append(load)
        for idx, pair in enumerate(pairs):
            chosen = assignments.action_index.astype(int) == idx
            pair_rows.append({
                "trial_id": tid, "policy": policy, "action_index": idx,
                "pair": str(pair), "selection_count": int(chosen.sum()),
                "selection_frequency": float(chosen.mean()),
            })
        if policy in ("Safe Min-Latency", "Masked PPO stochastic"):
            telemetry = telemetry.copy()
            telemetry.insert(0, "trial_id", tid)
            telemetry_frames.append(telemetry)
    for policy in ("Pair PPO greedy", "Pair PPO stochastic", "Masked PPO greedy"):
        assignments, decisions, load = _load_reference(tid, policy)
        seed_rows.append(_metrics(assignments, decisions, load, tid, policy))
        tier_rows.extend(_requirement_rows(assignments, decisions, tid, policy))
        load_frames.append(load)
        for idx, pair in enumerate(pairs):
            chosen = assignments.action_index.astype(int) == idx
            pair_rows.append({
                "trial_id": tid, "policy": policy, "action_index": idx,
                "pair": str(pair), "selection_count": int(chosen.sum()),
                "selection_frequency": float(chosen.mean()),
            })
    return {
        "seeds": seed_rows, "tiers": tier_rows, "pairs": pair_rows,
        "loads": load_frames, "telemetry": telemetry_frames, "ties": tie_frames,
    }


def run_evaluations(output_dir=OUT, trial_ids=None, workers=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formal_spec()
    trial_ids = list(range(10)) if trial_ids is None else [int(x) for x in trial_ids]
    if sorted(set(trial_ids)) != trial_ids or any(tid < 0 or tid >= 10 for tid in trial_ids):
        raise ValueError("trial_ids must be unique values in the formal 0..9 seed plan")
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    payloads = [(tid, str(output_dir)) for tid in trial_ids]
    results = []
    with ProcessPoolExecutor(max_workers=min(int(workers), len(payloads)),
                             mp_context=get_context("spawn")) as executor:
        for tid, result in zip(trial_ids, executor.map(_evaluate_one_trial, payloads)):
            print(f"Completed paired baseline trial {tid}/9", flush=True)
            results.append(result)
    seed_rows = [row for result in results for row in result["seeds"]]
    tier_rows = [row for result in results for row in result["tiers"]]
    pair_rows = [row for result in results for row in result["pairs"]]
    load_frames = [frame for result in results for frame in result["loads"]]
    telemetry_frames = [frame for result in results for frame in result["telemetry"]]
    tie_frames = [frame for result in results for frame in result["ties"]]
    seeds = pd.DataFrame(seed_rows).sort_values(["policy", "trial_id"]).reset_index(drop=True)
    tiers = pd.DataFrame(tier_rows).sort_values(["policy", "trial_id", "R_req"]).reset_index(drop=True)
    pairs_df = pd.DataFrame(pair_rows).sort_values(["policy", "trial_id", "action_index"]).reset_index(drop=True)
    loads = pd.concat(load_frames, ignore_index=True).sort_values(
        ["policy", "trial_id", "server_id"]
    ).reset_index(drop=True)
    max_ties = pd.concat(tie_frames, ignore_index=True).sort_values(
        ["trial_id", "episode", "task_id"]
    ).reset_index(drop=True)
    tie_summary = max_ties.groupby("trial_id", as_index=False).agg(
        mean_number_of_max_reliability_ties=("number_of_max_reliability_ties", "mean"),
        multiway_tie_rate=("number_of_max_reliability_ties", lambda x: float((x > 1).mean())),
    )
    if len(seeds) != 7 * len(trial_ids):
        raise RuntimeError("Missing method/seed summaries")
    if trial_ids == list(range(10)) and (len(seeds) != 70 or set(seeds.policy) != set(METHODS)):
        raise RuntimeError("Formal evaluation requires ten seed results for all seven methods")
    for frame, label in ((seeds, "seed results"), (tiers, "requirement results"),
                         (pairs_df, "pair distribution"), (loads, "server load"),
                         (max_ties, "max-reliability tie diagnostics")):
        _finite(frame.drop(columns=["policy", "pair", "selected_pair"], errors="ignore"), label)
    seeds.to_csv(output_dir / "baseline_seed_results.csv", index=False)
    tiers.to_csv(output_dir / "baseline_requirement_results.csv", index=False)
    pairs_df.to_csv(output_dir / "baseline_pair_distribution.csv", index=False)
    loads.to_csv(output_dir / "baseline_server_load.csv", index=False)
    max_ties.to_csv(output_dir / "baseline_max_reliability_ties.csv", index=False)
    tie_summary.to_csv(output_dir / "baseline_max_reliability_tie_summary.csv", index=False)
    telemetry = pd.concat(telemetry_frames, ignore_index=True).sort_values(
        ["policy", "trial_id", "episode", "task_id"]
    ).reset_index(drop=True)
    _finite(telemetry.drop(columns=["policy", "selected_pair"], errors="ignore"), "saved decision telemetry")
    telemetry.to_csv(output_dir / "baseline_decision_telemetry.csv.gz", index=False, compression="gzip")
    return seeds, tiers, pairs_df, loads

def bootstrap_comparisons(seeds, output_dir, n_bootstrap=20000, seed=2043):
    rng = np.random.default_rng(seed)
    comparison_rows, bootstrap_rows = [], []
    for baseline in ("Safe Min-Latency", "Max-Reliability", "Safe Random"):
        ppo = seeds[seeds.policy.eq("Masked PPO stochastic")].set_index("trial_id").sort_index()
        base = seeds[seeds.policy.eq(baseline)].set_index("trial_id").sort_index()
        deltas = {}
        for trial_id in ppo.index:
            record = {"trial_id": int(trial_id), "comparison": f"Masked PPO stochastic vs {baseline}"}
            for metric, direction in COMPARE_METRICS:
                delta = float(ppo.loc[trial_id, metric] - base.loc[trial_id, metric])
                deltas.setdefault(metric, []).append(delta)
                record[f"ppo_{metric}"] = float(ppo.loc[trial_id, metric])
                record[f"baseline_{metric}"] = float(base.loc[trial_id, metric])
                record[f"delta_{metric}"] = delta
                record[f"ppo_win_{metric}"] = bool(delta > 1e-12) if direction == "higher" else bool(delta < -1e-12)
            comparison_rows.append(record)
        for metric, direction in COMPARE_METRICS:
            values = np.asarray(deltas[metric], dtype=float)
            if len(values) != 10 or not np.isfinite(values).all():
                raise RuntimeError("Paired bootstrap requires ten finite seed-level deltas")
            idx = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
            means = values[idx].mean(axis=1)
            wins = int(np.sum(values > 1e-12) if direction == "higher" else np.sum(values < -1e-12))
            losses = int(np.sum(values < -1e-12) if direction == "higher" else np.sum(values > 1e-12))
            bootstrap_rows.append({
                "comparison": f"Masked PPO stochastic vs {baseline}", "metric": metric,
                "direction": direction, "n_seeds": 10, "bootstrap_samples": int(n_bootstrap),
                "mean_delta_ppo_minus_baseline": float(values.mean()),
                "ci95_low": float(np.quantile(means, .025)),
                "ci95_high": float(np.quantile(means, .975)),
                "ppo_wins": wins, "baseline_wins": losses, "ties": 10 - wins - losses,
                "bootstrap_seed": int(seed),
            })
    paired, intervals = pd.DataFrame(comparison_rows), pd.DataFrame(bootstrap_rows)
    paired.to_csv(output_dir / "baseline_paired_comparisons.csv", index=False)
    intervals.to_csv(output_dir / "baseline_bootstrap_ci.csv", index=False)
    return paired, intervals


def write_audit(output_dir):
    text = f"""# Baseline Information Audit

The ten formal environment seed rows and saved 300-episode Masked PPO
checkpoints are reused. Within each trial, all new policies use the same task
arrival and spatial-risk streams. The policies' internal queues may diverge.
Tie-breaking uses separate deterministic NumPy RNGs, independent of the
environment generators.

| Policy | Decision-time features | Reliability information |
|---|---|---|
| Masked PPO stochastic | The {params.num_states}-value observation: per-server nominal/base failure rate, processing frequency, normalized CPU backlog time, uplink rate; task input size, computation demand, and requirement. | Production reliability vector and the same safe/effective mask used during training, evaluated from episode-effective server rates; no direct observation of the spatial field. |
| Max-Reliability | Current task computation demand and episode-effective failure inputs passed through the production Task reliability initializer. | All legal pair reliabilities, without threshold filtering for its choice. |
| Safe Min-Latency | Task input size/computation demand plus current per-server backlog, uplink rate, and processing frequency. Those server/task quantities correspond to the PPO observation (backlog is normalized reversibly using the configured scale). | Same production reliability vector and effective-action-mask helper as Masked PPO. |
| Safe Random | No latency preference; it samples uniformly from the shared safe mask or, if empty, shared maximum-reliability fallback mask. | Same production reliability vector and mask/fallback as Masked PPO. |

## Decision-time latency estimate

For server n at decision time t, the estimator uses B_n(t) from
EnvironmentState.get_server_backlog_time, upload U_i,n = 8 * input MB /
uplink Mbps, and service S_i,n = computation demand / processing frequency.
It computes D_hat_i,n = max(B_n(t), U_i,n) + S_i,n. Existing CPU work drains
while the task uploads. The two replicas upload and execute in parallel, and
the production task resolves at the first replica result, so the estimated
pair latency is min(D_hat_i,j, D_hat_i,k). The production model assumes zero
download time and does not Bernoulli-sample execution failures; reliability is
evaluated analytically. The estimate never reads realized Task_Delay or future
arrivals/queue events, and never steps candidate actions. It is a no-new-arrivals
estimate, so it may be optimistic if future arrivals or already-uploading
replicas join the queue before this task.

## Checkpoint/replay checks

Actor and Critic files are verified against their trial manifest and not
written back. Frozen stochastic Masked PPO replay must match the original 4000
actions, safe masks, task reward, delay, and selected reliability per trial.
"""
    (output_dir / "baseline_information_audit.md").write_text(text, encoding="utf-8")


def save_figures(seeds, output_dir):
    labels = list(METHODS)
    specs = [
        ("mean_reward", "Mean task reward", "baseline_reward.png"),
        ("mean_latency", "Mean latency (s)", "baseline_mean_latency.png"),
        ("p95_latency", "P95 latency (s)", "baseline_p95_latency.png"),
        ("overall_rsr", "Overall RSR", "baseline_overall_rsr.png"),
        ("highest_rsr", "Highest-tier RSR", "baseline_highest_rsr.png"),
        ("pair_selection_hhi", "Pair-selection HHI", "baseline_pair_hhi.png"),
        ("maximum_mean_queue_length", "Maximum mean queue", "baseline_max_queue.png"),
    ]
    for metric, title, file_name in specs:
        grouped = seeds.groupby("policy")[metric]
        mean, sd = grouped.mean().reindex(labels), grouped.std(ddof=1).reindex(labels)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(np.arange(len(labels)), mean, yerr=sd, capsize=3)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        ax.set_ylabel(title)
        ax.set_title("Ten environment seeds: mean ± SD")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(output_dir / file_name, dpi=160)
        plt.close(fig)
    ppo = seeds[seeds.policy.eq("Masked PPO stochastic")].set_index("trial_id")
    for baseline, prefix in (("Safe Min-Latency", "ppo_vs_safe_min_latency"),
                             ("Safe Random", "ppo_vs_safe_random")):
        base = seeds[seeds.policy.eq(baseline)].set_index("trial_id")
        metrics = (("mean_reward", "Mean reward"), ("mean_latency", "Mean latency (s)"),
                   ("maximum_mean_queue_length", "Maximum mean queue"))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (metric, title) in zip(axes, metrics):
            x, y = base.loc[ppo.index, metric].to_numpy(), ppo[metric].to_numpy()
            low, high = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([low, high], [low, high], color="gray", linestyle="--", linewidth=1)
            ax.scatter(x, y, s=30)
            ax.set(xlabel=baseline, ylabel="Masked PPO stochastic", title=title)
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}.png", dpi=160)
        plt.close(fig)
    base = seeds[seeds.policy.eq("Safe Min-Latency")].set_index("trial_id")
    for metric, title, suffix in (("mean_reward", "Mean reward", "reward"),
                                  ("mean_latency", "Mean latency (s)", "latency"),
                                  ("maximum_mean_queue_length", "Maximum mean queue", "queue")):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        x, y = base.loc[ppo.index, metric].to_numpy(), ppo[metric].to_numpy()
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
        ax.scatter(x, y, s=34)
        ax.set(xlabel="Safe Min-Latency", ylabel="Masked PPO stochastic", title=title)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"ppo_vs_safe_min_latency_{suffix}.png", dpi=160)
        plt.close(fig)


def _safe_corr(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def latency_mechanism_analysis(output_dir):
    """Summarize myopic estimates, realized delay, and subsequent queues."""
    path = Path(output_dir) / "baseline_decision_telemetry.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "trial_id", "policy", "episode", "task_id", "estimated_latency", "Task_Delay",
        "decision_backlog_mean_selected", "next_decision_selected_pair_mean_backlog",
        "next_decision_selected_pair_backlog_delta", "next_decision_exists",
        "prior_endpoint_selections_server_j", "prior_endpoint_selections_server_k",
        "selected_pair",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Decision telemetry misses columns: {sorted(missing)}")
    rows, quartile_rows = [], []
    for (policy, trial_id), group in frame.groupby(["policy", "trial_id"], sort=True):
        g = group.sort_values(["episode", "task_id"]).copy()
        residual = g.Task_Delay.to_numpy(dtype=float) - g.estimated_latency.to_numpy(dtype=float)
        prior_reuse = (g.prior_endpoint_selections_server_j.astype(float)
                       + g.prior_endpoint_selections_server_k.astype(float)).to_numpy()
        valid_next = g.next_decision_exists.astype(bool).to_numpy()
        same_prev = g.groupby("episode").selected_pair.transform(lambda x: x.eq(x.shift(1))).to_numpy(dtype=bool)
        rows.append({
            "trial_id": int(trial_id), "policy": policy, "tasks": len(g),
            "mean_estimated_latency": float(g.estimated_latency.mean()),
            "mean_realized_latency": float(g.Task_Delay.mean()),
            "mean_realized_minus_estimated": float(residual.mean()),
            "p95_absolute_estimate_error": float(np.quantile(np.abs(residual), .95)),
            "estimated_vs_realized_correlation": _safe_corr(g.estimated_latency, g.Task_Delay),
            "mean_selected_pair_decision_backlog": float(g.decision_backlog_mean_selected.mean()),
            "mean_next_arrival_selected_pair_backlog": float(g.loc[valid_next, "next_decision_selected_pair_mean_backlog"].mean()),
            "mean_next_arrival_backlog_delta": float(g.loc[valid_next, "next_decision_selected_pair_backlog_delta"].mean()),
            "next_arrival_backlog_increase_rate": float((g.loc[valid_next, "next_decision_selected_pair_backlog_delta"] > 1e-12).mean()),
            "mean_prior_endpoint_reuses": float(prior_reuse.mean()),
            "reuse_vs_realized_estimate_error_correlation": _safe_corr(prior_reuse, residual),
            "same_pair_as_previous_decision_rate": float(same_prev.mean()),
        })
        g["task_order_quartile"] = pd.cut(
            g.task_id.astype(int), bins=[0, 50, 100, 150, 200],
            labels=[1, 2, 3, 4], include_lowest=True,
        ).astype(int)
        for quartile, part in g.groupby("task_order_quartile", sort=True):
            valid = part.next_decision_exists.astype(bool)
            quartile_rows.append({
                "trial_id": int(trial_id), "policy": policy,
                "task_order_quartile": int(quartile), "tasks": len(part),
                "mean_estimated_latency": float(part.estimated_latency.mean()),
                "mean_realized_latency": float(part.Task_Delay.mean()),
                "mean_realized_minus_estimated": float((part.Task_Delay-part.estimated_latency).mean()),
                "mean_decision_backlog_selected": float(part.decision_backlog_mean_selected.mean()),
                "mean_next_arrival_backlog_delta": float(part.loc[valid, "next_decision_selected_pair_backlog_delta"].mean()),
                "mean_prior_endpoint_reuses": float((part.prior_endpoint_selections_server_j + part.prior_endpoint_selections_server_k).mean()),
            })
    summary = pd.DataFrame(rows)
    by_quartile = pd.DataFrame(quartile_rows)
    _finite(summary.drop(columns="policy"), "latency mechanism summary")
    _finite(by_quartile.drop(columns="policy"), "latency by task quartile")
    summary.to_csv(Path(output_dir) / "baseline_latency_mechanism.csv", index=False)
    by_quartile.to_csv(Path(output_dir) / "baseline_latency_by_task_quartile.csv", index=False)
    # Compact diagnostic figures: estimate calibration and queue behavior over
    # decision order. Sampling only affects visualization, not reported stats.
    fig, ax = plt.subplots(figsize=(6, 5))
    for policy, part in frame.groupby("policy"):
        sampled = part.sample(n=min(2500, len(part)), random_state=2043)
        ax.scatter(sampled.estimated_latency, sampled.Task_Delay, s=5, alpha=.15, label=policy)
    upper = float(max(frame.estimated_latency.max(), frame.Task_Delay.max()))
    ax.plot([0, upper], [0, upper], linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="Decision-time estimated latency (s)", ylabel="Realized task delay (s)",
           title="Myopic estimate vs realized first-result delay")
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "myopic_latency_estimate_vs_realized.png", dpi=160)
    plt.close(fig)
    agg = by_quartile.groupby(["policy", "task_order_quartile"], as_index=False).agg(
        mean_decision_backlog=("mean_decision_backlog_selected", "mean"),
        mean_realized_latency=("mean_realized_latency", "mean"),
        mean_next_backlog_delta=("mean_next_arrival_backlog_delta", "mean"),
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy, part in agg.groupby("policy"):
        ax.plot(part.task_order_quartile, part.mean_decision_backlog, marker="o", label=policy)
    ax.set(xlabel="Within-episode task-order quartile", ylabel="Selected-pair current backlog (s)",
           title="Queue state over each episode")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "queue_backlog_by_task_quartile.png", dpi=160)
    plt.close(fig)
    return summary, by_quartile


def write_report(seeds, tiers, ci, mechanism, quartiles, output_dir):
    def ms(policy, metric):
        values = seeds.loc[seeds.policy.eq(policy), metric].astype(float)
        return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
    table = []
    for policy in METHODS:
        table.append(
            f"| {policy} | {ms(policy, 'mean_reward')} | {ms(policy, 'mean_latency')} | "
            f"{ms(policy, 'p95_latency')} | {ms(policy, 'overall_rsr')} | "
            f"{ms(policy, 'highest_rsr')} | {ms(policy, 'pair_selection_hhi')} | "
            f"{ms(policy, 'maximum_mean_queue_length')} |"
        )
    ci_table = []
    for row in ci.itertuples(index=False):
        ci_table.append(
            f"| {row.comparison} | {row.metric} | {row.mean_delta_ppo_minus_baseline:.5f} "
            f"[{row.ci95_low:.5f}, {row.ci95_high:.5f}] | "
            f"{row.ppo_wins}/{row.baseline_wins}/{row.ties} |"
        )
    ppo = seeds[seeds.policy.eq("Masked PPO stochastic")]
    safe = seeds[seeds.policy.eq("Safe Min-Latency")]
    maxr = seeds[seeds.policy.eq("Max-Reliability")]
    random = seeds[seeds.policy.eq("Safe Random")]
    high_safe = tiers[(tiers.policy.eq("Safe Min-Latency")) & np.isclose(tiers.R_req, .9999)]
    high_ppo = tiers[(tiers.policy.eq("Masked PPO stochastic")) & np.isclose(tiers.R_req, .9999)]
    primary = ci[ci.comparison.eq("Masked PPO stochastic vs Safe Min-Latency")].set_index("metric")
    reward_ci = primary.loc["mean_reward"]
    mean_latency_ci = primary.loc["mean_latency"]
    p95_ci = primary.loc["p95_latency"]
    queue_ci = primary.loc["maximum_mean_queue_length"]
    strict_ppo_advantage = (
        (reward_ci.ci95_low > 0 and queue_ci.ci95_high < 0)
        or (reward_ci.ci95_low > 0 and p95_ci.ci95_high < 0)
    ) and mean_latency_ci.ci95_high < 0.10
    all_practical_deltas_small = (
        abs(reward_ci.mean_delta_ppo_minus_baseline) < 1.0
        and abs(mean_latency_ci.mean_delta_ppo_minus_baseline) < 0.10
        and abs(p95_ci.mean_delta_ppo_minus_baseline) < 0.20
        and abs(queue_ci.mean_delta_ppo_minus_baseline) < 0.10
    )
    if strict_ppo_advantage:
        answer = "Yes, in this tested setting: paired intervals show a PPO advantage on reward and at least one long-run queue/tail metric without a practically large mean-latency penalty."
    elif all_practical_deltas_small and all(
        primary.loc[m].ci95_low <= 0 <= primary.loc[m].ci95_high
        for m in ("mean_reward", "mean_latency", "p95_latency", "maximum_mean_queue_length")
    ):
        answer = "No evidence here that PPO is necessary over Safe Min-Latency: the main paired effects are practically small and their 95% intervals include zero."
    else:
        answer = "Evidence is mixed: some paired metrics favor one method, but the predeclared reward/queue/tail criteria do not establish general PPO necessity over Safe Min-Latency."
    def mech_mean(policy, metric):
        values = mechanism.loc[mechanism.policy.eq(policy), metric].astype(float)
        return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
    mechanism_table = []
    for method in ("Safe Min-Latency", "Masked PPO stochastic"):
        mechanism_table.append(
            f"| {method} | {mech_mean(method, 'mean_estimated_latency')} | "
            f"{mech_mean(method, 'mean_realized_latency')} | "
            f"{mech_mean(method, 'mean_realized_minus_estimated')} | "
            f"{mech_mean(method, 'estimated_vs_realized_correlation')} | "
            f"{mech_mean(method, 'mean_next_arrival_backlog_delta')} | "
            f"{mech_mean(method, 'same_pair_as_previous_decision_rate')} |"
        )
    qmeans = quartiles.groupby(["policy", "task_order_quartile"], as_index=False).mean(numeric_only=True)
    q4_safe = qmeans[(qmeans.policy.eq("Safe Min-Latency")) & (qmeans.task_order_quartile == 4)].iloc[0]
    q1_safe = qmeans[(qmeans.policy.eq("Safe Min-Latency")) & (qmeans.task_order_quartile == 1)].iloc[0]
    safe_backlog_change = float(q4_safe.mean_decision_backlog_selected - q1_safe.mean_decision_backlog_selected)
    accumulation = ("increased" if safe_backlog_change > 1e-6 else
                    "decreased" if safe_backlog_change < -1e-6 else "was approximately flat")
    text = f"""# External Reliability Baseline Report

## 1. Baseline Definitions

* **Max-Reliability:** enumerate every legal pair and use the production reliability initializer; sample uniformly among maximum-reliability ties.
* **Reliability-Constrained Min-Latency:** select minimum decision-time estimated first-result latency among the exact production safe set. If it is empty, use the exact Masked PPO maximum-reliability fallback set, then minimize estimated latency within it.
* **Reliability-Constrained Random:** sample uniformly from the exact safe set, or uniformly from the shared maximum-reliability fallback when empty.
* PPO reference policies are the stored formal Pair greedy/stochastic and Masked greedy/stochastic deployments. The Masked stochastic trajectory was replayed only for telemetry and had to match its original recorded actions, masks, and outcomes.

Every method uses 10 paired environment seeds × 20 evaluation episodes × 200 tasks. Exogenous arrivals and spatial-risk streams match per seed. Policy-dependent server queues are expected to differ.

## 2. Information Fairness

See [baseline_information_audit.md](baseline_information_audit.md) for the exact observation variables and estimate equation. Safe-set inputs use the same production feasibility functions as Masked PPO. The heuristic does not receive realized delay or simulate candidate actions.

## 3. Main Results

Mean ± sample SD across the 10 environment seeds; tasks are not treated as independent samples.

| Method | Reward | Mean latency (s) | P95 latency (s) | Overall RSR | Highest-tier RSR | Pair HHI | Maximum mean queue |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## 4. PPO vs Max-Reliability

Max-Reliability achieves mean overall/highest-tier RSR of {maxr.overall_rsr.mean():.4f}/{maxr.highest_rsr.mean():.4f}. The paired table tests whether reliability-first choices pay for that reliability with reward, latency, queue, or pair concentration. Tie counts are present in the per-decision logs.

## 5. PPO vs Reliability-Constrained Min-Latency

Mean overall RSR is {ppo.overall_rsr.mean():.4f} for Masked PPO stochastic and {safe.overall_rsr.mean():.4f} for Safe Min-Latency. At requirement 0.9999, mean feasibility is {high_safe.feasibility_rate.mean():.4f} for Safe Min-Latency and {high_ppo.feasibility_rate.mean():.4f} for PPO; corresponding RSR is {high_safe.overall_rsr.mean():.4f}/{high_ppo.overall_rsr.mean():.4f}. See paired reward/latency/queue comparisons before judging whether the myopic heuristic is sufficient.

## 6. PPO vs Safe Random

Safe Random's mean overall RSR is {random.overall_rsr.mean():.4f}; its comparison isolates the value of actor preference beyond the shared reliability mask.

## 7. Queue and Load Analysis

Per-server utilization, time-weighted mean/P95 queues, and waiting time are in baseline_server_load.csv. Decision telemetry records all current server backlog values and the next task-arrival backlog for the selected servers. Per-seed concentration metrics include pair HHI, maximum server selection share, maximum utilization, and maximum queues.

## 8. Reliability Analysis

Feasibility rate is P(safe set nonempty). Conditional RSR is success among feasible decisions. Avoidable violation means the safe set was nonempty but a selected pair failed the production reliability threshold. Unavoidable violation means the safe set was empty and the chosen fallback could not meet the requirement. The constrained methods are checked for zero avoidable violations. Selected-action shortfall and best-achievable shortfall are both recorded separately.

## 9. Myopic vs Long-Term Behavior

The compressed baseline_decision_telemetry.csv.gz joins each selected-action estimate to realized Task_Delay, current backlog, and selected-server backlog at the next task arrival. Across seeds, Safe Min-Latency had estimated/realized latency {mech_mean('Safe Min-Latency', 'mean_estimated_latency')} / {mech_mean('Safe Min-Latency', 'mean_realized_latency')} s, mean realized-minus-estimated error {mech_mean('Safe Min-Latency', 'mean_realized_minus_estimated')} s, estimate/realized correlation {mech_mean('Safe Min-Latency', 'estimated_vs_realized_correlation')}, and mean next-arrival selected-pair backlog change {mech_mean('Safe Min-Latency', 'mean_next_arrival_backlog_delta')} s. Its selected-pair backlog {accumulation} by {safe_backlog_change:.4f} s from the first to fourth within-episode task quartile. Same-pair consecutive-decision frequency was {mech_mean('Safe Min-Latency', 'same_pair_as_previous_decision_rate')}. These measurements describe whether the myopic choice accumulates congestion; they do not feed future queue outcomes back into earlier choices.

| Policy | Estimated latency | Realized latency | Realized − estimated | Estimate/realized corr. | Next-arrival backlog Δ | Same pair consecutively |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(mechanism_table)}

## 10. Seed Stability

The intervals below use 20,000 bootstrap resamples of paired seed-level deltas (n=10), seed 2043. Delta is Masked PPO minus baseline. For reward/RSR, positive favors PPO; for latency/HHI/queue, negative favors PPO. W/L/T counts are seed-level.

| Comparison | Metric | Mean delta [95% CI] | PPO win/loss/tie |
|---|---|---:|---:|
{chr(10).join(ci_table)}

## 11. Conclusion

**Q1.** Max-Reliability is reliability-first by construction; its actual reward, latency, queue, and RSR results are shown above and compared by paired bootstrap.

**Q2.** Safe Min-Latency reaches mean RSR {safe.overall_rsr.mean():.4f}; whether it can replace PPO depends on paired reward, mean/tail latency, and queue results, not RSR alone.

**Q3.** The same safe action rule is used by PPO and Safe Min-Latency. Their paired queue/reward/tail-latency deltas test whether the learned stochastic preference adds long-term allocation value.

**Q4.** Safe Random tests whether a mask alone plus random sampling is enough; its measured paired gaps are reported above.

**Does PPO remain necessary given a reliability-safe action set?** {answer} This is a ten-seed experimental conclusion, not a universal claim. If intervals overlap zero, report the evidence as inconclusive/competitive rather than asserting a PPO advantage.
"""
    (output_dir / "EXTERNAL_BASELINE_REPORT.md").write_text(text, encoding="utf-8")


def analyze(output_dir=OUT):
    output_dir = Path(output_dir)
    seeds = pd.read_csv(output_dir / "baseline_seed_results.csv")
    tiers = pd.read_csv(output_dir / "baseline_requirement_results.csv")
    if len(seeds) != 70 or set(seeds.policy) != set(METHODS):
        raise RuntimeError("Analysis expects 10 seeds for each of seven methods")
    _, ci = bootstrap_comparisons(seeds, output_dir, n_bootstrap=20000, seed=2043)
    write_audit(output_dir)
    save_figures(seeds, output_dir)
    mechanism, quartiles = latency_mechanism_analysis(output_dir)
    write_report(seeds, tiers, ci, mechanism, quartiles, output_dir)
    return seeds, tiers, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--trial-ids", type=int, nargs="*", default=None)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    full_run = args.trial_ids is None or sorted(args.trial_ids) == list(range(10))
    if not args.analyze_only:
        run_evaluations(args.output_dir, args.trial_ids, args.workers)
    if full_run or args.analyze_only:
        analyze(args.output_dir)


if __name__ == "__main__":
    main()
