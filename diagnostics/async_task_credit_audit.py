"""Read-only full-terminal audit of asynchronous task credit and externality.

This diagnostic reuses the frozen 1,000-state manifest and the exact replay
helpers from the value/GAE identifiability audit. It forces each effective
action once and then continues the frozen stochastic Masked PPO policy. No
production simulator, policy, critic, reward, or training code is changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

from agents.action_conditioned_q_critic import ActionConditionedQNetwork
from agents.policy_centered_action_advantage import (
    PolicyCenteredActionAdvantage,
    policy_centered_advantage,
)
from config.params import params
from diagnostics.run_masked_pair_ppo_10seed import formal_spec
from diagnostics.value_gae_identifiability import (
    CHECKPOINT,
    OUT as VALUE_OUT,
    make_agent,
    model_hash,
    replay_checks,
    run_episode,
)
from Project_main import build_pair_correlations
from tools.pair_policy_diagnostics import TASK_ASSIGNMENT_COLUMNS
from tools.paired_ppo_experiment import sha256_file


OUT = ROOT / "diagnostics/results/async_task_credit_audit"
MANIFEST_PATH = VALUE_OUT / "matched_state_manifest.csv"
DATASET_DIR = VALUE_OUT / "dataset"
SEED_PLAN_PATH = VALUE_OUT / "episode_seed_plan.csv"
PREVIOUS_Q_PATH = VALUE_OUT / "matched_counterfactual_qh.csv"
PREVIOUS_H20_PATH = ROOT / "diagnostics/results/value_gae_identifiability/matched_counterfactual_qh.csv"
SIDE_ADVANTAGE_PATH = ROOT / "diagnostics/results/advantage_side_learner/policy_centered_advantage.pt"
OLD_Q_PATH = ROOT / "diagnostics/results/q_credit_audit/training_run/runs/trial_000/masked/q_critic.pt"
GAMMA = float(params.gamma_ppo)
TIE_ATOL = 1e-9  # same tie tolerance used by the prior long-horizon oracle audit
CONFLICT_REGRET_TOL = 0.01  # prior H20 coupled-state threshold; fixed before this audit
DISTANCE_BANDS = ((1, 1, "next_1"), (2, 5, "next_2_5"), (6, 10, "next_6_10"),
                  (11, 20, "next_11_20"), (21, 10**9, "next_gt20"))
EVENT_LAG_BANDS = ((0.0, 1.0, "lag_0_1s"), (1.0, 2.0, "lag_1_2s"),
                   (2.0, 5.0, "lag_2_5s"), (5.0, math.inf, "lag_gt5s"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_provenance(root_decision_index: int, task_decision_index: int,
                        completion_time: float, root_decision_time: float) -> str:
    """Classify task reward provenance using decision order and resolution time."""
    root_decision_index, task_decision_index = int(root_decision_index), int(task_decision_index)
    if task_decision_index == root_decision_index:
        return "current_root"
    if task_decision_index > root_decision_index:
        return "future_decision"
    if float(completion_time) > float(root_decision_time) + 1e-12:
        return "preexisting_pending"
    return "preexisting_resolved"


def full_terminal_components(records, root_decision_index: int, root_decision_time: float,
                             gamma: float = GAMMA) -> dict[str, float]:
    """Compute task-credit and event-time own/future returns to natural terminal.

    Records are task-level rewards with exact originating decision indices.
    Pre-existing pending rewards are intentionally excluded from both returns.
    """
    rows = [dict(row) for row in records]
    root = int(root_decision_index)
    current = [row for row in rows if int(row["decision_index"]) == root]
    if len(current) != 1:
        raise ValueError("Each branch must contain exactly one root-task reward")
    own = current[0]
    own_dec = float(own["reward"])
    own_evt = float(gamma) ** max(float(own["completion_time"]) - float(root_decision_time), 0.0) * own_dec
    future_dec = 0.0
    future_evt = 0.0
    for row in rows:
        index = int(row["decision_index"])
        if index <= root:
            continue
        reward = float(row["reward"])
        future_dec += float(gamma) ** max(float(row["decision_time"]) - float(root_decision_time), 0.0) * reward
        future_evt += float(gamma) ** max(float(row["completion_time"]) - float(root_decision_time), 0.0) * reward
    return {
        "own_undiscounted": own_dec,
        "own_decision": own_dec,
        "own_event": own_evt,
        "future_decision": future_dec,
        "future_event": future_evt,
        "total_decision": own_dec + future_dec,
        "total_event": own_evt + future_evt,
    }


def centered_scores(scores, probabilities):
    values = np.asarray(scores, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if values.shape != probs.shape or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores and policy probabilities must be finite matching vectors")
    if (probs < 0).any() or not np.isfinite(probs).all() or probs.sum() <= 0:
        raise ValueError("Policy probabilities must have positive finite mass")
    p = probs / probs.sum()
    centered = values - float(np.dot(p, values))
    return centered, float(abs(np.dot(p, centered)))


def optimal_set(values, atol: float = TIE_ATOL):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Optimal-set values must be a nonempty finite vector")
    return np.flatnonzero(np.isclose(values, values.max(), rtol=0.0, atol=float(atol)))


def tie_aware_top_set(values, k: int, atol: float = TIE_ATOL):
    values = np.asarray(values, dtype=np.float64)
    if k <= 0 or not len(values):
        return np.asarray([], dtype=int)
    k = min(int(k), len(values))
    cutoff = float(np.partition(values, len(values) - k)[len(values) - k])
    return np.flatnonzero(values >= cutoff - float(atol))


def regret(values, action_position):
    values = np.asarray(values, dtype=np.float64)
    return float(values.max() - values[int(action_position)])


def decision_event_alignment(decision_values, event_values, atol: float = TIE_ATOL):
    dec, evt = np.asarray(decision_values, float), np.asarray(event_values, float)
    if dec.shape != evt.shape or dec.ndim != 1 or not np.isfinite(dec).all() or not np.isfinite(evt).all():
        raise ValueError("Decision/event values must be matching finite vectors")
    dec_adv = dec - dec.mean()
    evt_adv = evt - evt.mean()
    rho = float(spearmanr(dec, evt).statistic) if np.std(dec) > 1e-12 and np.std(evt) > 1e-12 else float("nan")
    tau = float(kendalltau(dec, evt).statistic) if np.std(dec) > 1e-12 and np.std(evt) > 1e-12 else float("nan")
    dec_opt, evt_opt = optimal_set(dec, atol), optimal_set(evt, atol)
    union = set(dec_opt) | set(evt_opt)
    return {
        "spearman": rho,
        "kendall": tau,
        "sign_agreement": float(np.mean(np.sign(dec_adv) == np.sign(evt_adv))),
        "optimal_set_jaccard": float(len(set(dec_opt) & set(evt_opt)) / len(union)) if union else 1.0,
        "optimal_set_exact_agreement": bool(np.array_equal(dec_opt, evt_opt)),
        "mean_regret_dec_ranking_on_event": float(np.max(evt) - np.max(evt[dec_opt])),
        "mean_regret_event_ranking_on_dec": float(np.max(dec) - np.max(dec[evt_opt])),
    }


def compare_original_branch(reference, branch):
    """Return replay mismatch counts and max absolute reward/transition error."""
    fields = ("states", "next_states", "actions", "rewards", "delta_t", "done")
    mismatches = {}
    max_abs = 0.0
    for field in fields:
        a, b = np.asarray(reference[field]), np.asarray(branch[field])
        if a.shape != b.shape:
            mismatches[field] = 1
            continue
        if a.dtype.kind in "f" or b.dtype.kind in "f":
            diff = np.abs(a.astype(float) - b.astype(float))
            max_abs = max(max_abs, float(np.nanmax(diff, initial=0.0)))
            mismatches[field] = int(not np.array_equal(a, b))
        else:
            mismatches[field] = int(not np.array_equal(a, b))
    return mismatches, max_abs


def common_randomness_mismatches(reference, branch, baseline_arrivals):
    """Compare fixed arrival and Torch action draw streams for a whole branch."""
    return {
        "arrival_stream_mismatch": int(not np.array_equal(branch["arrival_trace"], baseline_arrivals)),
        "torch_stream_mismatch": int(not np.array_equal(branch["rng_trace"], reference["rng_trace"])),
    }


def assignment_records(assignments: pd.DataFrame, decision_times: np.ndarray):
    """Validate reward outcome times and return task-level outcome records."""
    frame = assignments.copy()
    if frame.empty or frame["task_id"].duplicated().any():
        raise RuntimeError("Each resolved task must have exactly one assignment/reward row")
    frame["task_id"] = pd.to_numeric(frame["task_id"], errors="raise").astype(int)
    frame = frame.sort_values("task_id").reset_index(drop=True)
    if len(frame) != len(decision_times) or not np.array_equal(frame.task_id.to_numpy(), np.arange(1, len(decision_times) + 1)):
        raise RuntimeError("Task reward provenance is incomplete or out of decision order")
    records = []
    completion_ranks = {}
    finish_times = []
    for row in frame.itertuples(index=False):
        started = float(row.Primary_Start)
        delay = float(row.Task_Delay)
        outcome = started + delay
        replica_finishes = [float(value) for value in (row.Primary_End, row.Backup_End) if pd.notna(value)]
        if not replica_finishes:
            raise RuntimeError(f"Task {row.task_id} has no replica completion time")
        production_outcome = min(replica_finishes)
        if not math.isclose(outcome, production_outcome, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(f"Task {row.task_id} reward resolution timestamp disagrees with production delay")
        finish_times.append((outcome, int(row.task_id)))
    for rank, (_, task_id) in enumerate(sorted(finish_times), start=1):
        completion_ranks[task_id] = rank
    for row in frame.itertuples(index=False):
        task_id = int(row.task_id)
        records.append({
            "decision_index": task_id,
            "task_id": task_id,
            "decision_time": float(decision_times[task_id - 1]),
            "primary_start": float(row.Primary_Start),
            "completion_time": float(row.Primary_Start) + float(row.Task_Delay),
            "action_index": int(row.action_index),
            "server_j": int(row.server_j),
            "server_k": int(row.server_k),
            "reward": float(row.Task_Reward),
            "delay": float(row.Task_Delay),
            "reliability": float(row.Execution_Reliability),
            "reliability_requirement": float(row.Reliability_Requirement),
            "reliability_satisfied": bool(row.Reliability_Satisfied),
            "completion_order": int(completion_ranks[task_id]),
            "primary_status": str(row.Primary_Status),
            "backup_status": str(row.Backup_Status),
        })
    return records


def select_relevant_provenance(records, root_id, root_time):
    selected = []
    for row in records:
        role = classify_provenance(root_id, row["decision_index"], row["completion_time"], root_time)
        if role == "preexisting_resolved":
            continue
        selected.append({**row, "task_role": role,
                         "future_component_included": role == "future_decision"})
    return selected


def band_contributions(records, root_id, root_time, gamma=GAMMA):
    result = {}
    for lo, hi, label in DISTANCE_BANDS:
        dec = evt = 0.0
        for row in records:
            gap = int(row["decision_index"]) - int(root_id)
            if gap < lo or gap > hi:
                continue
            dec += float(gamma) ** max(float(row["decision_time"]) - root_time, 0.0) * float(row["reward"])
            evt += float(gamma) ** max(float(row["completion_time"]) - root_time, 0.0) * float(row["reward"])
        result[f"future_dec_{label}"] = dec
        result[f"future_evt_{label}"] = evt
    for lo, hi, label in EVENT_LAG_BANDS:
        evt = 0.0
        for row in records:
            if int(row["decision_index"]) <= int(root_id):
                continue
            lag = float(row["completion_time"]) - root_time
            if lo <= lag < hi:
                evt += float(gamma) ** max(lag, 0.0) * float(row["reward"])
        result[f"future_evt_{label}"] = evt
    return result


def digest_protected_inputs():
    protected = [
        MANIFEST_PATH, SEED_PLAN_PATH, VALUE_OUT / "offline_gae.npz",
        VALUE_OUT / "matched_counterfactual_qh.csv", VALUE_OUT / "matched_state_replay_checks.csv",
        VALUE_OUT / "frozen_policy_integrity.json", SIDE_ADVANTAGE_PATH,
        CHECKPOINT / "actor.pt", CHECKPOINT / "critic.pt",
        ROOT / "data/task_parameters.xlsx", ROOT / "data/server_info.xlsx",
    ]
    protected.extend(sorted(DATASET_DIR.glob("episode_*.npz")))
    return {str(path.relative_to(ROOT)): file_digest(path) for path in protected if path.exists()}


def _episode_worker(episode: int, output_dir: str):
    """Run all fixed matched-state/effective-action branches for one episode."""
    torch.set_num_threads(1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"episode_{episode:03d}_branch_summary.csv.gz"
    provenance_path = output / f"episode_{episode:03d}_provenance.csv.gz"
    checks_path = output / f"episode_{episode:03d}_checks.csv"
    if summary_path.exists() and provenance_path.exists() and checks_path.exists():
        return episode, "cached"

    manifest = pd.read_csv(MANIFEST_PATH)
    selected = manifest[manifest.episode.astype(int) == int(episode)].sort_values("task_id")
    if selected.empty:
        raise RuntimeError(f"Matched-state manifest has no states in episode {episode}")
    seeds = pd.read_csv(SEED_PLAN_PATH).set_index("episode").loc[int(episode)].to_dict()
    with np.load(DATASET_DIR / f"episode_{episode:03d}.npz") as archive:
        ref = {key: archive[key] for key in archive.files}
    agent, pair_rho = make_agent()
    actor_hash_before, critic_hash_before = model_hash(agent.policy_old), model_hash(agent.value_net)

    # Natural frozen-policy replay verifies the already-published matched episode.
    baseline, baseline_loop = run_episode(agent, seeds, details=True,
                                          capture_ids=selected.task_id.astype(int).tolist())
    natural_keys = ["states", "next_states", "actions", "probabilities", "effective_mask",
                    "backlog", "task_parameters", "rewards", "delta_t", "done", "rng_trace",
                    "time", "arrival_trace", "effective_rates", "reliabilities"]
    for key in natural_keys:
        if not np.array_equal(baseline[key], ref[key]):
            raise RuntimeError(f"Natural frozen replay mismatch: episode={episode} key={key}")
    reference_snapshots = copy.deepcopy(agent.snapshots)
    base_assignments = pd.DataFrame(baseline_loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS)
    base_records = assignment_records(base_assignments, ref["time"])
    base_by_id = {row["task_id"]: row for row in base_records}

    rows, provenance, checks = [], [], []
    for state_row in selected.to_dict("records"):
        task_id = int(state_row["task_id"])
        root_pos = task_id - 1
        target_snapshot = reference_snapshots[task_id]
        target_actions = np.flatnonzero(ref["effective_mask"][root_pos])
        if len(target_actions) != int(state_row["effective_set_size"]):
            raise RuntimeError(f"Candidate count mismatch in {state_row['state_id']}")
        probabilities = np.asarray(ref["probabilities"][root_pos, target_actions], dtype=np.float64)
        probabilities /= probabilities.sum()
        root_time = float(ref["time"][root_pos])
        original_action = int(ref["actions"][root_pos])
        original_branch_residual = np.nan

        for action in target_actions:
            branch, branch_loop = run_episode(agent, seeds, details=False,
                                              capture_ids=[task_id], forced=(task_id, int(action)),
                                              cache=ref)
            snapshot = agent.snapshots[task_id]
            replay = replay_checks(snapshot, ref, task_id)
            replay["action_prefix_mismatch"] = int(not np.array_equal(branch["actions"][:root_pos], ref["actions"][:root_pos]))
            replay["future_arrival_mismatch"] = int(not np.array_equal(branch["arrival_trace"], ref["arrival_trace"]))
            replay["future_torch_rng_mismatch"] = int(not np.array_equal(branch["rng_trace"], ref["rng_trace"]))
            replay["target_action_mismatch"] = int(int(branch["actions"][root_pos]) != int(action))
            replay["target_effective_rates_mismatch"] = int(not np.array_equal(snapshot["effective_rates"], ref["effective_rates"][root_pos]))
            replay["target_reliabilities_mismatch"] = int(not np.array_equal(snapshot["reliabilities"], ref["reliabilities"][root_pos]))
            replay["downstream_state_count"] = max(len(branch["actions"]) - task_id, 0)
            replay.update(episode=episode, task_id=task_id, state_id=state_row["state_id"], action_index=int(action))
            if any(int(value) for key, value in replay.items() if key.endswith("mismatch")):
                raise RuntimeError(f"Matched branch replay mismatch: {replay}")
            assignments = pd.DataFrame(branch_loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS)
            records = assignment_records(assignments, ref["time"])
            component = full_terminal_components(records, task_id, root_time, GAMMA)
            relevant = select_relevant_provenance(records, task_id, root_time)
            # All older tasks resolved after the root decision are logged but have exactly zero return weight.
            pending_leakage = 0.0
            duplicate_count = int(assignments.task_id.duplicated().sum())
            for item in relevant:
                task_decision_index = int(item["decision_index"])
                future_dec_contribution = 0.0
                future_evt_contribution = 0.0
                if task_decision_index > task_id:
                    future_dec_contribution = float(GAMMA) ** max(float(item["decision_time"]) - root_time, 0.0) * float(item["reward"])
                    future_evt_contribution = float(GAMMA) ** max(float(item["completion_time"]) - root_time, 0.0) * float(item["reward"])
                if item["task_role"] == "preexisting_pending" and (future_dec_contribution != 0.0 or future_evt_contribution != 0.0):
                    pending_leakage += abs(future_dec_contribution) + abs(future_evt_contribution)
                provenance.append({
                    "episode": episode, "state_id": state_row["state_id"],
                    "root_task_id": task_id, "root_decision_index": task_id,
                    "root_decision_time": root_time, "forced_action_index": int(action),
                    "forced_server_j": int(agent.pairs[int(action)][0]),
                    "forced_server_k": int(agent.pairs[int(action)][1]),
                    "task_id": int(item["task_id"]), "decision_index": task_decision_index,
                    "decision_time": float(item["decision_time"]),
                    "completion_time": float(item["completion_time"]),
                    "completion_order": int(item["completion_order"]),
                    "task_role": item["task_role"],
                    "future_component_included": bool(item["future_component_included"]),
                    "action_index": int(item["action_index"]),
                    "server_j": int(item["server_j"]), "server_k": int(item["server_k"]),
                    "pair_rho": float(pair_rho[int(item["action_index"])]),
                    "task_reward": float(item["reward"]), "task_delay": float(item["delay"]),
                    "execution_reliability": float(item["reliability"]),
                    "reliability_requirement": float(item["reliability_requirement"]),
                    "reliability_satisfied": bool(item["reliability_satisfied"]),
                    "primary_start": float(item["primary_start"]),
                    "decision_return_contribution": future_dec_contribution,
                    "event_return_contribution": future_evt_contribution,
                    "preexisting_pending_return_contribution": 0.0,
                })
            band_fields = band_contributions(records, task_id, root_time, GAMMA)
            policy_probability = float(ref["probabilities"][root_pos, int(action)])
            summary = {
                **state_row,
                "root_decision_index": task_id, "root_decision_time": root_time,
                "action_index": int(action), "pair_j": int(agent.pairs[int(action)][0]),
                "pair_k": int(agent.pairs[int(action)][1]),
                "pair_rho": float(pair_rho[int(action)]),
                "policy_probability": policy_probability,
                "selected_original_action": original_action,
                "root_completion_time": float(records[task_id - 1]["completion_time"]),
                "root_reward": float(records[task_id - 1]["reward"]),
                "root_delay": float(records[task_id - 1]["delay"]),
                "root_reliability": float(records[task_id - 1]["reliability"]),
                "root_reliability_satisfied": bool(records[task_id - 1]["reliability_satisfied"]),
                "preexisting_pending_task_count": int(sum(x["task_role"] == "preexisting_pending" for x in relevant)),
                "preexisting_pending_leakage": pending_leakage,
                "duplicate_task_reward_count": duplicate_count,
                "relevant_provenance_records": len(relevant),
                **component, **band_fields,
                "counterfactual_training_used": False,
            }
            rows.append(summary)

            if int(action) == original_action:
                mismatch, max_abs = compare_original_branch(ref, branch)
                original_branch_residual = max_abs
                assignment_comparison = compare_assignment_outcomes(base_by_id, records)
                decision_residual = abs(component["total_decision"] - float(ref["mc_return"][root_pos]))
                replay.update({f"original_branch_{key}_mismatch": value for key, value in mismatch.items()})
                replay.update(assignment_comparison)
                replay["original_branch_max_transition_abs_diff"] = max_abs
                replay["original_branch_decision_return_abs_diff"] = decision_residual
                replay["original_branch_event_return_abs_diff"] = compare_event_returns(base_records, records, task_id, root_time, GAMMA)
                if max_abs > 1e-8 or decision_residual > 1e-8 or assignment_comparison["original_task_outcome_mismatch_count"]:
                    raise RuntimeError(f"Original-action full-terminal replay failed: {replay}")
            replay["original_branch_return_abs_diff"] = original_branch_residual if int(action) == original_action else np.nan
            replay["preexisting_pending_leakage"] = pending_leakage
            replay["duplicate_task_reward_count"] = duplicate_count
            replay["own_future_decomposition_abs_residual"] = abs(component["total_event"] - component["own_event"] - component["future_event"])
            if replay["own_future_decomposition_abs_residual"] > 1e-10 or pending_leakage != 0.0 or duplicate_count:
                raise RuntimeError(f"Provenance integrity failure: {replay}")
            checks.append(replay)

    if actor_hash_before != model_hash(agent.policy_old) or critic_hash_before != model_hash(agent.value_net):
        raise RuntimeError("Frozen Actor/Critic parameters changed during replay")
    pd.DataFrame(rows).to_csv(summary_path, index=False, compression="gzip")
    pd.DataFrame(provenance).to_csv(provenance_path, index=False, compression="gzip")
    pd.DataFrame(checks).to_csv(checks_path, index=False)
    return episode, {"candidate_branches": len(rows), "provenance_records": len(provenance)}


def compare_assignment_outcomes(baseline_by_id, branch_records):
    mismatches = 0
    max_abs = 0.0
    numeric = ("reward", "delay", "reliability", "completion_time")
    for row in branch_records:
        ref = baseline_by_id[int(row["task_id"])]
        for field in numeric:
            difference = abs(float(row[field]) - float(ref[field]))
            max_abs = max(max_abs, difference)
            if difference > 1e-10:
                mismatches += 1
        for field in ("action_index", "reliability_satisfied", "completion_order"):
            if row[field] != ref[field]:
                mismatches += 1
    return {"original_task_outcome_mismatch_count": int(mismatches),
            "original_task_outcome_max_abs_diff": float(max_abs)}


def compare_event_returns(baseline_records, branch_records, root_id, root_time, gamma):
    base = full_terminal_components(baseline_records, root_id, root_time, gamma)["total_event"]
    branch = full_terminal_components(branch_records, root_id, root_time, gamma)["total_event"]
    return abs(float(base) - float(branch))


def _read_episode_products(episodes):
    output = OUT / "branches"
    summaries, provenance, checks = [], [], []
    for episode in episodes:
        summaries.append(pd.read_csv(output / f"episode_{episode:03d}_branch_summary.csv.gz"))
        provenance.append(pd.read_csv(output / f"episode_{episode:03d}_provenance.csv.gz"))
        checks.append(pd.read_csv(output / f"episode_{episode:03d}_checks.csv"))
    return (pd.concat(summaries, ignore_index=True), pd.concat(provenance, ignore_index=True),
            pd.concat(checks, ignore_index=True))


def run_branches(workers=2, episodes=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists() or not PREVIOUS_Q_PATH.exists():
        raise FileNotFoundError("The frozen 1,000-state manifest and previous counterfactual audit are required")
    manifest = pd.read_csv(MANIFEST_PATH)
    if len(manifest) != 1000 or manifest.state_id.nunique() != 1000:
        raise RuntimeError("Frozen manifest must contain exactly 1,000 unique matched states")
    selected_episodes = sorted(manifest.episode.astype(int).unique().tolist())
    if episodes is not None:
        selected_episodes = sorted(set(map(int, episodes)))
        if not set(selected_episodes).issubset(set(manifest.episode.astype(int))):
            raise ValueError("Requested episode outside the frozen matched-state manifest")
    output = OUT / "branches"
    output.mkdir(parents=True, exist_ok=True)
    if int(workers) == 1:
        for i, episode in enumerate(selected_episodes, start=1):
            print("async branches", _episode_worker(episode, str(output)), f"{i}/{len(selected_episodes)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(_episode_worker, episode, str(output)): episode for episode in selected_episodes}
            for i, future in enumerate(as_completed(futures), start=1):
                print("async branches", future.result(), f"{i}/{len(futures)}", flush=True)
    if episodes is not None and set(selected_episodes) != set(manifest.episode.astype(int).unique()):
        return {"partial_episodes": selected_episodes}
    summary, provenance, checks = _read_episode_products(selected_episodes)
    if summary.state_id.nunique() != 1000:
        raise RuntimeError("Full audit must include all 1,000 frozen matched states")
    expected_candidates = 0
    for _, frame in summary.groupby("state_id"):
        first = frame.iloc[0]
        expected_candidates += int(first.effective_set_size)
        if len(frame) != int(first.effective_set_size) or frame.action_index.duplicated().any():
            raise RuntimeError(f"Missing or duplicate candidate actions for state {first.state_id}")
    if len(summary) != expected_candidates:
        raise RuntimeError("Candidate action branch total is inconsistent")
    summary.to_csv(OUT / "decision_vs_event_return.csv", index=False)
    summary.to_csv(OUT / "own_future_total_components.csv.gz", index=False, compression="gzip")
    compact_provenance = provenance.drop(columns=["root_decision_index", "forced_server_j", "forced_server_k",
                                                  "future_component_included", "primary_start"], errors="ignore")
    compact_provenance.to_csv(
        OUT / "branch_reward_provenance.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 9}, float_format="%.10g",
    )
    checks.to_csv(OUT / "task_credit_integrity_checks.csv", index=False)
    return {"matched_states": int(summary.state_id.nunique()),
            "candidate_branches": int(len(summary)),
            "branch_task_records": int(len(provenance)),
            "checks": checks}



def _make_plots(conflict, distance, gae, safe, ppo, components):
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(conflict.own_event_spread, conflict.future_event_spread, s=10, alpha=.45)
    ax.set(xlabel="Own-task event-time spread", ylabel="Future event-time spread",
           title="Own reward vs future externality")
    save(fig, "own_vs_future_spread.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ratio = conflict.future_own_spread_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(ratio.clip(upper=20), bins=35, color="#4c78a8", alpha=.85)
    ax.set(xlabel="Future / own spread ratio (clipped at 20)", ylabel="Matched states",
           title="Relative future externality scale")
    save(fig, "future_own_spread_ratio.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    rates = [float(conflict.own_total_exact_optimal_set_agreement.mean()),
             float((conflict.own_total_optimal_jaccard).mean()), float(conflict.conflict_state.mean())]
    ax.bar(["Exact-set agreement", "Mean Jaccard", "Conflict rate"], rates, color=["#59a14f", "#f28e2b", "#e15759"])
    ax.set_ylim(0, 1); ax.set_ylabel("Fraction / similarity")
    ax.set_title("Own-optimal vs total-optimal sets")
    save(fig, "own_total_optimal_conflict.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    parts = safe[(safe.semantics == "event") & (safe.component == "total")]
    ax.hist(parts.loc[parts.conflict_state, "regret"], bins=30, alpha=.8, label="SafeMin, conflict")
    ax.set(xlabel="Total event-time regret", ylabel="States", title="Safe Min-Latency regret on conflict states")
    save(fig, "safemin_total_regret_conflicts.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    parts = ppo[(ppo.semantics == "event") & (ppo.component == "total")]
    ax.hist(parts.loc[parts.conflict_state, "regret"], bins=30, alpha=.8, color="#e15759")
    ax.set(xlabel="Total event-time regret", ylabel="States", title="PPO action regret on conflict states")
    save(fig, "ppo_total_regret_conflicts.png")

    summary_rows = []
    for (gae_kind, component), part in gae[(gae.semantics == "event")].groupby(["gae_kind", "component"]):
        summary_rows.append({"gae_kind": gae_kind, "component": component,
                             "spearman": _safe_correlation(part.gae_value, part.selected_advantage, rank=True),
                             "pearson": _safe_correlation(part.gae_value, part.selected_advantage)})
    gsum = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(gsum):
        piv = gsum.pivot(index="component", columns="gae_kind", values="spearman").reindex(["own", "future", "total"])
        piv.plot(kind="bar", ax=ax, color=["#4c78a8", "#f28e2b"])
    ax.set(ylabel="Spearman correlation", xlabel="Centered event-time component", title="PPO GAE alignment")
    ax.axhline(0, color="black", lw=.7); ax.legend(title="GAE")
    save(fig, "gae_component_alignment.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(conflict.total_decision_spread, conflict.total_event_spread, s=9, alpha=.4)
    ax.set(xlabel="Decision-time total spread", ylabel="Event-time total spread", title="Decision vs event return landscape")
    save(fig, "decision_vs_event_total.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["low", "medium", "high"]
    arrays = [conflict.loc[conflict.load.astype(str) == label, "future_event_spread"].to_numpy() for label in order]
    ax.boxplot(arrays, labels=order, showfliers=False)
    ax.set(xlabel="Frozen audit load stratum", ylabel="Future event-time spread", title="Externality by load")
    save(fig, "future_spread_by_load.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    dist = distance[distance.band_type.eq("decision_distance")]
    order = [name for _, _, name in DISTANCE_BANDS]
    grouped = dist.groupby("band").future_event_spread.mean().reindex(order)
    ax.plot(order, grouped.to_numpy(), marker="o")
    ax.set(xlabel="Future decision distance", ylabel="Mean action spread", title="Future externality by task distance")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "externality_accumulation_by_task_distance.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in (("coupled", "#e15759"), ("easy", "#4c78a8")):
        values = conflict.loc[conflict.coupling.eq(label), "future_event_spread"]
        ax.hist(values, bins=25, alpha=.55, label=f"{label} (n={len(values)})", color=color)
    ax.set(xlabel="Future event-time spread", ylabel="States", title="Coupled vs easy states")
    ax.legend()
    save(fig, "coupled_easy_externality_spread.png")

    top = conflict.sort_values(["future_event_spread", "state_id"], ascending=[False, True]).head(5)
    top.to_csv(OUT / "representative_case_studies.csv", index=False)
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=False)
    for ax, item in zip(axes, top.itertuples(index=False)):
        local = components[components.state_id.eq(item.state_id)].sort_values("action_index")
        x = local.action_index.to_numpy()
        ax.plot(x, local.own_event_advantage, marker=".", label="Own event")
        ax.plot(x, local.future_event_advantage, marker=".", label="Future event")
        ax.plot(x, local.total_event_advantage, marker=".", label="Total event")
        safe_action = int(safe.loc[(safe.state_id == item.state_id) & (safe.semantics == "event") & (safe.component == "total"), "action_index"].iloc[0])
        ppo_action = int(ppo.loc[(ppo.state_id == item.state_id) & (ppo.semantics == "event") & (ppo.component == "total"), "action_index"].iloc[0])
        ax.axvline(safe_action, color="black", linestyle=":", label="SafeMin action")
        ax.axvline(ppo_action, color="purple", linestyle="--", label="PPO action")
        ax.set_title(f"{item.state_id}: Future spread={item.future_event_spread:.4g}; conflict={item.conflict_state}")
        ax.set_ylabel("Policy-centered value")
    axes[-1].set_xlabel("Effective action index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(.5, .995))
    save(fig, "representative_top5_action_components.png")


def _md_table(frame, columns=None, max_rows=30):
    if frame.empty:
        return "(no rows)"
    shown = frame if columns is None else frame[columns]
    return shown.head(max_rows).to_markdown(index=False, floatfmt=".4f")


def _write_report(conflict, externality, gae_summary, ranking_summary,
                  strata_summary, integrity, q_status):
    event_spreads = conflict.future_event_spread
    own_spreads = conflict.own_event_spread
    conflict_rows = conflict[conflict.conflict_state]
    safe_total = pd.read_csv(OUT / "safe_min_latency_component_alignment.csv")
    ppo_total = pd.read_csv(OUT / "ppo_component_alignment.csv")
    safe_total = safe_total[(safe_total.semantics == "event") & (safe_total.component == "total")]
    ppo_total = ppo_total[(ppo_total.semantics == "event") & (ppo_total.component == "total")]
    future_gae = gae_summary[(gae_summary.semantics == "event") & (gae_summary.component == "future")]
    own_gae = gae_summary[(gae_summary.semantics == "event") & (gae_summary.component == "own")]
    total_gae = gae_summary[(gae_summary.semantics == "event") & (gae_summary.component == "total")]
    dec_evt = conflict[["spearman", "kendall", "sign_agreement", "optimal_set_jaccard", "optimal_set_exact_agreement"]]
    q20 = pd.read_csv(PREVIOUS_Q_PATH)
    hist = q20.groupby("state_id").first()
    if len(hist) != 1000:
        raise RuntimeError("Prior H20 matched-state audit did not contain all 1,000 states")
    q20_best = q20.groupby("state_id").q_h20.max()
    q20_ml = q20.drop_duplicates("state_id").set_index("state_id")
    q20_regret = (q20_best - q20_ml.q_h20)
    prior_coupled_count = int(sum(coupling_value > CONFLICT_REGRET_TOL for coupling_value in (
        q20_best.loc[state_id] - float(q20.loc[(q20.state_id == state_id) & (q20.action_index.astype(int) == int(q20_ml.loc[state_id].myopic_action)), "q_h20"].iloc[0])
        for state_id in q20_best.index)))
    high_load = conflict[conflict.load.astype(str).eq("high")]
    high_req = conflict[np.isclose(conflict.requirement, .9999)]
    safe_615 = conflict[conflict.safe_set_bin.eq("6-15")]
    distance = pd.read_csv(OUT / "externality_distance_decomposition.csv")
    distance = distance[distance.band_type.eq("decision_distance")].groupby("band").future_event_contribution.agg("mean").reindex([n for _, _, n in DISTANCE_BANDS])

    # A transparent classification rule: use effect sizes, set conflicts, baseline regret,
    # and GAE correlations jointly; no threshold is tuned after inspecting one method.
    mean_future, median_future = float(event_spreads.mean()), float(event_spreads.median())
    disjoint_rate = float(conflict.own_total_disjoint.mean())
    safe_conflict_regret = float(safe_total.loc[safe_total.conflict_state, "regret"].mean()) if conflict_rows.size else 0.0
    future_spearman = float(future_gae.loc[future_gae.gae_kind.eq("raw"), "spearman"].iloc[0])
    own_spearman = float(own_gae.loc[own_gae.gae_kind.eq("raw"), "spearman"].iloc[0])
    dec_event_rank = float(dec_evt.spearman.median())
    dec_event_jaccard = float(dec_evt.optimal_set_jaccard.mean())
    if mean_future < 1e-3 and disjoint_rate < .05 and safe_conflict_regret < .01:
        primary, secondary = "A", "None"
    elif dec_event_rank < .8 or dec_event_jaccard < .8:
        primary, secondary = "C", "B" if mean_future >= 1e-3 and future_spearman < own_spearman else "E"
    elif mean_future >= 1e-3 and disjoint_rate >= .05 and safe_conflict_regret >= .01 and future_spearman < own_spearman:
        primary, secondary = "B", "None"
    elif future_spearman >= .2:
        primary, secondary = "D", "None"
    else:
        primary, secondary = "E", "B" if mean_future >= 1e-3 else "A"
    worthwhile = bool(mean_future >= 1e-3 and disjoint_rate >= .05 and safe_conflict_regret >= .01)
    recommend_externality = bool(worthwhile and future_spearman < own_spearman)
    recommend_temporal = bool(primary == "C" or (secondary == "C" and dec_event_rank < .8))
    if primary == "A":
        explanation = "Observed future spread, optimal-set conflict, and SafeMin conflict regret are all small under the predeclared comparison tolerances."
    elif primary == "B":
        explanation = "Future event-time spread and own-vs-total action conflict are material, SafeMin gives up total return on conflict states, while raw PPO GAE aligns less with future than own advantage."
    elif primary == "C":
        explanation = "The same state/action rankings change materially under event-time versus decision-index discounting, so resolution-time semantics are implicated."
    elif primary == "D":
        explanation = "Raw PPO GAE has meaningful alignment with the future component, so target credit is not the primary explanation."
    else:
        explanation = "The evidence is mixed across externality size, optimal-set conflict, baseline regret, and GAE alignment; no single cause dominates."

    req_summary = conflict.groupby("requirement").agg(
        states=("state_id", "nunique"), future_spread_mean=("future_event_spread", "mean"),
        conflict_rate=("conflict_state", "mean"), own_total_jaccard=("own_total_optimal_jaccard", "mean"),
    ).reset_index()
    safe_summary = conflict.groupby("safe_set_bin").agg(
        states=("state_id", "nunique"), future_spread_mean=("future_event_spread", "mean"),
        conflict_rate=("conflict_state", "mean"),
    ).reset_index()
    load_summary = conflict.groupby("load").agg(
        states=("state_id", "nunique"), future_spread_mean=("future_event_spread", "mean"),
        conflict_rate=("conflict_state", "mean"),
    ).reset_index()

    lines = [
        "# Asynchronous Task Credit and Future Externality Audit", "",
        f"Frozen source commit: `{json.loads((VALUE_OUT / 'frozen_policy_integrity.json').read_text()).get('source_commit', 'as recorded in prior audit')}`.",
        f"Matched-state source: `{MANIFEST_PATH.relative_to(ROOT)}`; all {integrity['matched_states']} preselected states; {integrity['candidate_branches']} forced effective-action branches.",
        f"Policy: frozen Masked Pair PPO stochastic continuation. No training or production behavior changed. Gamma={GAMMA:g}; rewards remain in task reward units.", "",
        "## Time definitions and decomposition", "",
        "Decision time `t_i` is the task arrival/action time from the frozen rollout. Task reward resolution time `tau_i` is the production outcome timestamp `Primary_Start + Task_Delay`; the replay validates it against `min(Primary_End, Backup_End)` (first completed replica). Decision-index return is `r_i + sum_{j>i} gamma^(t_j-t_i) r_j`; event-time return is `sum_{j>=i} gamma^(tau_j-t_i) r_j`. Own/future are split by task decision index, not completion order.",
        "Previously decided tasks `j<i` that are still pending at `t_i` are preserved as `preexisting_pending` provenance records, but have exactly zero contribution to the root action’s future return. Only `j>i` enters the future externality.", "",
        "## Replay and provenance integrity", "",
        f"- State/mask/probability/backlog/task-parameter mismatch counts: {integrity['state_mismatch_count']}/{integrity['mask_mismatch_count']}/{integrity['probability_mismatch_count']}/{integrity['backlog_mismatch_count']}/{integrity['task_parameter_mismatch_count']}. Effective hazards and task reliabilities at root: {integrity['target_spatial_hazard_mismatch_count']}/{integrity['target_reliability_mismatch_count']} mismatches.",
        f"- Candidate branches: {integrity['candidate_branches']}; task-level provenance records: {integrity['branch_task_records']}; duplicate reward records: {integrity['duplicate_reward_count']}; pending-task leakage records: {integrity['preexisting_pending_leakage_count']}.",
        f"- Arrival/Torch RNG stream mismatches: {integrity['arrival_stream_mismatch_count']}/{integrity['torch_stream_mismatch_count']}. Current simulator has no independent per-replica Bernoulli failure draw; environmental randomness comes from the fixed arrival/spatial streams and policy Torch sampling.",
        f"- Original-action branch max transition abs diff: {integrity['original_transition_max_abs_diff']:.3g}; decision return diff: {integrity['original_decision_return_max_abs_diff']:.3g}; event return diff: {integrity['original_event_return_max_abs_diff']:.3g}; complete assignment outcome mismatches: {integrity['original_assignment_mismatch_count']}.",
        f"- Maximum `Q_event - O_event - F_event` residual: {integrity['own_future_decomposition_max_abs_residual']:.3g}; policy-centering residual mean/max: {integrity['centering_mean_abs_residual']:.3g}/{integrity['centering_max_abs_residual']:.3g}.",
        f"- Actor/critic checkpoint SHA-256: `{integrity['actor_critic_hashes']['actor_file_sha256']}` / `{integrity['actor_critic_hashes']['critic_file_sha256']}`. Frozen model files were only read.", "",
        "## Full-terminal reward landscape", "",
        f"Event-time own spread mean/median/P90/P95: {own_spreads.mean():.4f}/{own_spreads.median():.4f}/{own_spreads.quantile(.90):.4f}/{own_spreads.quantile(.95):.4f}.",
        f"Event-time future spread mean/median/P90/P95: {event_spreads.mean():.4f}/{event_spreads.median():.4f}/{event_spreads.quantile(.90):.4f}/{event_spreads.quantile(.95):.4f}.",
        f"Future/own spread ratio median/P75/P90: {conflict.future_own_spread_ratio.median():.4f}/{conflict.future_own_spread_ratio.quantile(.75):.4f}/{conflict.future_own_spread_ratio.quantile(.90):.4f}.",
        f"Own/total exact optimal-set agreement: {conflict.own_total_exact_optimal_set_agreement.mean():.1%}; mean Jaccard overlap: {conflict.own_total_optimal_jaccard.mean():.3f}; disjoint rate: {conflict.own_total_disjoint.mean():.1%}; conflict rate (disjoint or own-optimal total regret > {CONFLICT_REGRET_TOL:g}): {conflict.conflict_state.mean():.1%}.",
        f"SafeMin event-time total mean regret: {safe_total.regret.mean():.4f}; on conflict states: {safe_total.loc[safe_total.conflict_state, 'regret'].mean():.4f}. PPO sampled action total mean regret: {ppo_total.regret.mean():.4f}; on conflict states: {ppo_total.loc[ppo_total.conflict_state, 'regret'].mean():.4f}.", "",
        "## GAE alignment", "",
        _md_table(gae_summary, ["gae_kind", "semantics", "component", "pearson", "spearman", "sign_agreement", "sign_agreement_including_zero"]), "",
        "GAE correlations are across the original sampled action at each matched state, against within-state policy-centered component advantages. Sign agreement omits target values with absolute magnitude at most `1e-8`; the zero-inclusive sensitivity is also shown.", "",
        "## Decision-time versus event-time return", "",
        f"Within-state total ranking Spearman median: {conflict.spearman.median():.4f}; Kendall median: {conflict.kendall.median():.4f}; centered sign agreement mean: {conflict.sign_agreement.mean():.1%}; optimal-set Jaccard mean: {conflict.optimal_set_jaccard.mean():.3f}; exact-set agreement: {conflict.optimal_set_exact_agreement.mean():.1%}.", "",
        "## Frozen H20 coupled/easy and predefined strata", "",
        "Coupled is reused unchanged from the previous H20 audit: max H20 decision-index return minus the SafeMin H20 return exceeds 0.01. Load, requirement, and safe-set bins are the frozen matched manifest’s original labels.",
        _md_table(strata_summary, ["stratum", "count", "gae_kind", "mean_future_spread", "conflict_rate", "gae_future_spearman", "gae_future_sign", "safemin_total_mean_regret", "ppo_total_mean_regret"]), "",
        "By requirement:", _md_table(req_summary), "", "By load:", _md_table(load_summary), "", "By effective safe-set bin:", _md_table(safe_summary), "",
        "## Future reward by downstream distance", "",
        "Decision-distance groups are descriptive propagation distances, not algorithm horizons. Event-lag bins are also descriptive only.",
        _md_table(distance.rename("mean_event_discounted_contribution").reset_index(), ["band", "mean_event_discounted_contribution"]), "",
        "## Action-ranking comparison", "",
        f"Centered Advantage and old Q are read-only scorers. Old Q status: {q_status}.",
        _md_table(ranking_summary), "",
        "## Conclusion and go/no-go", "",
        f"Primary conclusion = **{primary}**; secondary = **{secondary}**. {explanation}",
        f"Predeclared evidence: mean future spread={mean_future:.4f}, median={median_future:.4f}, own/total disjoint={disjoint_rate:.1%}, conflict-state SafeMin regret={safe_conflict_regret:.4f}, raw GAE-own/future Spearman={own_spearman:.4f}/{future_spearman:.4f}, median decision/event rank={dec_event_rank:.4f}, mean decision/event optimal-set Jaccard={dec_event_jaccard:.3f}.",
        f"Long-term externality worth further RL study: **{'YES' if worthwhile else 'NO'}**. Externality-Aware/Counterfactual Advantage design recommended: **{'YES' if recommend_externality else 'NO'}**. Change current PPO GAE temporal semantics recommended: **{'YES' if recommend_temporal else 'NO'}**.",
        "No algorithm change is implemented or proposed here beyond the evidence-based go/no-go decision. Fixed H=5/10/20/50 values remain in the old immutable audit for historical comparison; this audit’s primary returns run to the natural 200-task episode terminal.", "",
        "## Figures", "",
        "- `own_vs_future_spread.png`", "- `future_own_spread_ratio.png`", "- `own_total_optimal_conflict.png`",
        "- `safemin_total_regret_conflicts.png`", "- `ppo_total_regret_conflicts.png`", "- `gae_component_alignment.png`",
        "- `decision_vs_event_total.png`", "- `future_spread_by_load.png`", "- `externality_accumulation_by_task_distance.png`",
        "- `coupled_easy_externality_spread.png`", "- `representative_top5_action_components.png`", "",
    ]
    report = "\n".join(lines)
    report = report.replace("\n+", "\n")
    (OUT / "ASYNC_TASK_CREDIT_AUDIT.md").write_text(report)
    (OUT / "async_credit_decision.json").write_text(json.dumps({
        "primary": primary, "secondary": secondary,
        "future_externality_worth_learning": worthwhile,
        "recommend_externality_aware_counterfactual_advantage": recommend_externality,
        "recommend_change_current_gae_temporal_semantics": recommend_temporal,
        "evidence": {
            "mean_future_event_spread": mean_future,
            "median_future_event_spread": median_future,
            "own_total_disjoint_rate": disjoint_rate,
            "conflict_rate": float(conflict.conflict_state.mean()),
            "safemin_conflict_total_regret": safe_conflict_regret,
            "raw_gae_own_spearman": own_spearman,
            "raw_gae_future_spearman": future_spearman,
            "decision_event_total_spearman_median": dec_event_rank,
            "decision_event_optimal_set_jaccard_mean": dec_event_jaccard,
            "prior_h20_coupled_states": prior_coupled_count,
            "q_h20_safemin_regret_mean": float(q20_regret.mean()),
        },
        "classification_rule": "predefined evidence thresholds in ASYNC_TASK_CREDIT_AUDIT.md",
        "frozen_side_model_old_q_status": q_status,
    }, indent=2, sort_keys=True))


def run(workers=2, episodes=None):
    protected_before = digest_protected_inputs()
    branch_status = run_branches(workers=workers, episodes=episodes)
    if episodes is not None:
        return branch_status
    integrity = analyze_results()
    protected_after = digest_protected_inputs()
    if protected_before != protected_after:
        changed = [key for key in protected_before if protected_before.get(key) != protected_after.get(key)]
        raise RuntimeError(f"Protected previous audit/checkpoint inputs changed: {changed}")
    manifest = {
        "scope": "read-only asynchronous task credit and future externality diagnosis",
        "frozen_source_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "matched_states": 1000,
        "discount_gamma": GAMMA,
        "tie_atol": TIE_ATOL,
        "coupled_threshold": CONFLICT_REGRET_TOL,
        "terminal": "natural completion/drain of all 200 tasks and replicas",
        "completion_resolution_time": "Primary_Start + Task_Delay, verified equal to min(Primary_End, Backup_End)",
        "common_random_numbers": ["fixed episode arrival seed", "fixed spatial seed and episode hazard realization", "fixed Torch action stream; one draw consumed at forced action", "fixed task parameter rows", "simulator has no per-replica Bernoulli failure draws"],
        "actor_checkpoint_sha256": protected_before.get(str((CHECKPOINT / "actor.pt").relative_to(ROOT))),
        "critic_checkpoint_sha256": protected_before.get(str((CHECKPOINT / "critic.pt").relative_to(ROOT))),
        "protected_input_sha256": protected_before,
        "branch_count": int(integrity["candidate_branches"]),
        "task_provenance_record_count": int(integrity["branch_task_records"]),
        "preexisting_task_policy": "retain pending prior tasks in provenance, exclude them from root own/future returns",
        "production_modified": False,
        "training_run": False,
        "integrity": integrity,
    }
    (OUT / "reward_provenance_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return integrity




def _safe_correlation(x, y, rank=False):
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic if rank else np.corrcoef(a, b)[0, 1])


def _summary(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": np.nan, "median": np.nan, "p75": np.nan, "p90": np.nan, "p95": np.nan}
    return {"n": int(len(array)), "mean": float(np.mean(array)), "median": float(np.median(array)),
            "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95))}


def load_ranking_models():
    pairs, rho = build_pair_correlations()
    side = PolicyCenteredActionAdvantage(
        params.num_states, params.num_actions, params.hidden_layers_ppo,
        params.serverNo, rho, activation=params.af_ppo,
    )
    side.network.load_state_dict(torch.load(SIDE_ADVANTAGE_PATH, map_location="cpu", weights_only=True))
    side.eval()
    side_hash = model_hash(side.network)
    q_model, q_hash, q_status = None, None, "checkpoint unavailable"
    if OLD_Q_PATH.exists():
        try:
            q_model = ActionConditionedQNetwork(
                params.num_states, params.num_actions, params.hidden_layers_ppo,
                params.serverNo, rho, activation="tanh",
            )
            payload = torch.load(OLD_Q_PATH, map_location="cpu", weights_only=True)
            q_model.load_state_dict(payload.get("target", payload.get("online", payload)))
            q_model.eval()
            q_hash = model_hash(q_model)
            q_status = "scored using the existing target-network checkpoint"
        except Exception as exc:
            q_model, q_hash = None, None
            q_status = f"N/A: incompatible checkpoint ({type(exc).__name__}: {exc})"
    return side, side_hash, q_model, q_hash, q_status, pairs


def _method_action_scores(ref, index, actions, side, q_model):
    state = torch.as_tensor(np.array(ref["states"][index:index + 1], copy=True), dtype=torch.float32)
    probs = np.asarray(ref["probabilities"][index], dtype=float)
    mask = np.asarray(ref["effective_mask"][index], dtype=bool)
    latency = np.asarray(ref["latencies"][index], dtype=float)
    with torch.no_grad():
        side_raw = side.network(state).cpu().numpy()[0].astype(float)
        side_centered = policy_centered_advantage(
            torch.as_tensor(side_raw[None, :], dtype=torch.float32),
            torch.as_tensor(probs[None, :], dtype=torch.float32),
            torch.as_tensor(mask[None, :], dtype=torch.bool),
        ).numpy()[0]
        q_scores = q_model(state).cpu().numpy()[0].astype(float) if q_model is not None else None
    result = {"SafeMin": -latency[actions], "PPOActor": probs[actions],
              "CenteredAdvantage": side_centered[actions]}
    if q_scores is not None:
        result["OldQ"] = q_scores[actions]
    return result


def _ranking_rows(state_id, actions, methods, references):
    output = []
    for ref_name, ref in references.items():
        ref = np.asarray(ref, float)
        ref_opt = set(optimal_set(ref))
        ref_top3 = set(tie_aware_top_set(ref, 3))
        ref_top5 = set(tie_aware_top_set(ref, 5))
        for method, scores in methods.items():
            scores = np.asarray(scores, float)
            method_best = set(optimal_set(scores))
            best_position = min(method_best)
            top3 = set(tie_aware_top_set(scores, 3))
            top5 = set(tie_aware_top_set(scores, 5))
            output.append({
                "state_id": state_id, "method": method, "reference": ref_name,
                "candidate_count": len(actions),
                "spearman": _safe_correlation(scores, ref, rank=True),
                "pearson": _safe_correlation(scores, ref),
                "top1_hit": bool(bool(method_best & ref_opt)),
                "top3_optimal_overlap": len(top3 & ref_opt) / max(1, len(top3)),
                "top5_optimal_overlap": len(top5 & ref_opt) / max(1, len(top5)),
                "top3_reference_overlap": len(top3 & ref_top3) / max(1, len(top3)),
                "top5_reference_overlap": len(top5 & ref_top5) / max(1, len(top5)),
                "best_action_index": int(actions[best_position]),
                "reference_regret": float(ref.max() - ref[best_position]),
                "reference_optimal_set_size": len(ref_opt),
            })
    return output


def analyze_results():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    all_episodes = sorted(manifest.episode.astype(int).unique())
    branches, provenance, checks = _read_episode_products(all_episodes)
    prior_q = pd.read_csv(PREVIOUS_Q_PATH)
    if len(manifest) != 1000 or manifest.state_id.nunique() != 1000 or branches.state_id.nunique() != 1000:
        raise RuntimeError("The exact frozen 1,000-state matched set is required")
    if branches.duplicated(["state_id", "action_index"]).any():
        raise RuntimeError("Duplicate candidate branch")
    if provenance.duplicated(["episode", "state_id", "root_task_id", "forced_action_index", "task_id"]).any():
        raise RuntimeError("Duplicate task reward provenance")

    episode_data = {}
    for episode in all_episodes:
        with np.load(DATASET_DIR / f"episode_{episode:03d}.npz") as archive:
            episode_data[episode] = {key: archive[key] for key in archive.files}
    side, side_hash, q_model, q_hash, q_status, _pairs = load_ranking_models()

    action_rows, conflict_rows, method_rows, gae_rows, ranking_rows, distance_rows = [], [], [], [], [], []
    centering = []
    for state_id, frame in branches.groupby("state_id", sort=True):
        frame = frame.sort_values("action_index").reset_index(drop=True)
        first = frame.iloc[0]
        ep, tid = int(first.episode), int(first.task_id)
        idx = tid - 1  # per-episode NPZ row; manifest dataset_index is global across episodes
        ref = episode_data[ep]
        actions = frame.action_index.to_numpy(int)
        if set(actions) != set(np.flatnonzero(ref["effective_mask"][idx])):
            raise RuntimeError(f"Effective candidate set mismatch for {state_id}")
        prob = frame.policy_probability.to_numpy(float)
        prob /= prob.sum()
        vectors = {
            "own_decision": frame.own_decision.to_numpy(float),
            "future_decision": frame.future_decision.to_numpy(float),
            "total_decision": frame.total_decision.to_numpy(float),
            "own_event": frame.own_event.to_numpy(float),
            "future_event": frame.future_event.to_numpy(float),
            "total_event": frame.total_event.to_numpy(float),
        }
        centered = {}
        for name, values in vectors.items():
            centered[name], residual = centered_scores(values, prob)
            centering.append(residual)

        context = {key: first[key] for key in ("episode", "task_id", "load", "requirement", "safe_set_bin",
                                                 "safe_set_size", "effective_set_size", "max_backlog") if key in first.index}
        for position, action in enumerate(actions):
            action_rows.append({
                "state_id": state_id, **context, "action_index": int(action),
                "policy_probability": float(prob[position]),
                **{f"{name}_advantage": float(values[position]) for name, values in centered.items()},
                "selected_original_action": int(first.selected_original_action),
            })

        optimal = {name: set(actions[optimal_set(values)]) for name, values in centered.items()}
        own_evt_set, total_evt_set = optimal["own_event"], optimal["total_event"]
        inter, union = own_evt_set & total_evt_set, own_evt_set | total_evt_set
        own_total_regret = max(centered["total_event"]) - max(centered["total_event"][np.isin(actions, list(own_evt_set))])
        conflict = (not inter) or own_total_regret > CONFLICT_REGRET_TOL
        decision_evt = decision_event_alignment(vectors["total_decision"], vectors["total_event"])
        spreads = {name: float(np.ptp(values)) for name, values in centered.items()}
        state = {
            "state_id": state_id, **context,
            "own_event_spread": spreads["own_event"], "future_event_spread": spreads["future_event"],
            "total_event_spread": spreads["total_event"],
            "own_decision_spread": spreads["own_decision"], "future_decision_spread": spreads["future_decision"],
            "total_decision_spread": spreads["total_decision"],
            "future_own_spread_ratio": spreads["future_event"] / max(spreads["own_event"], 1e-12),
            "own_optimal_set_size": len(own_evt_set), "total_optimal_set_size": len(total_evt_set),
            "own_total_optimal_overlap": len(inter), "own_total_optimal_jaccard": len(inter) / len(union),
            "own_total_exact_optimal_set_agreement": own_evt_set == total_evt_set,
            "own_total_disjoint": not bool(inter), "own_optimal_total_regret": float(own_total_regret),
            "conflict_state": bool(conflict), **decision_evt,
        }
        conflict_rows.append(state)

        prior_state = prior_q[prior_q.state_id.eq(state_id)]
        if prior_state.empty:
            raise RuntimeError(f"Previous H20 branch data missing {state_id}")
        safe_action = int(prior_state.myopic_action.iloc[0])
        original_action = int(first.selected_original_action)
        posmap = {int(action): pos for pos, action in enumerate(actions)}
        if safe_action not in posmap or original_action not in posmap:
            raise RuntimeError(f"SafeMin or frozen PPO action outside effective set for {state_id}")
        for method, selected_action in (("SafeMin", safe_action), ("PPO", original_action)):
            position = posmap[selected_action]
            dest = method_rows
            for semantics in ("decision", "event"):
                for component in ("own", "future", "total"):
                    name = f"{component}_{semantics}"
                    value = vectors[name]
                    opt = set(actions[optimal_set(value)])
                    top3 = set(actions[tie_aware_top_set(value, 3)])
                    top5 = set(actions[tie_aware_top_set(value, 5)])
                    dest.append({"state_id": state_id, **context, "method": method,
                                 "action_index": selected_action, "semantics": semantics,
                                 "component": component, "optimal_set_hit": selected_action in opt,
                                 "top3_hit": selected_action in top3, "top5_hit": selected_action in top5,
                                 "regret": float(value.max() - value[position]),
                                 "optimal_set_size": len(opt), "conflict_state": bool(conflict)})

        for kind, gae_value in (("raw", float(ref["original_gae"][idx])),
                                ("actor_normalized", float(ref["original_normalized_gae"][idx]))):
            for semantics in ("decision", "event"):
                for component in ("own", "future", "total"):
                    name = f"{component}_{semantics}"
                    position = posmap[original_action]
                    gae_rows.append({"state_id": state_id, **context,
                                     "gae_kind": kind, "gae_value": gae_value,
                                     "semantics": semantics, "component": component,
                                     "selected_action": original_action,
                                     "selected_advantage": float(centered[name][position]),
                                     "conflict_state": bool(conflict)})

        methods = _method_action_scores(ref, idx, actions, side, q_model)
        references = {"OwnEvent": centered["own_event"], "FutureEvent": centered["future_event"],
                      "TotalEvent": centered["total_event"]}
        ranking_rows.extend(_ranking_rows(state_id, actions, methods, references))

        for _, _, band in DISTANCE_BANDS:
            dec = frame[f"future_dec_{band}"].to_numpy(float)
            evt = frame[f"future_evt_{band}"].to_numpy(float)
            dec_adv, dec_res = centered_scores(dec, prob)
            evt_adv, evt_res = centered_scores(evt, prob)
            centering.extend([dec_res, evt_res])
            for position, action in enumerate(actions):
                distance_rows.append({"state_id": state_id, **context, "band_type": "decision_distance",
                                      "band": band, "action_index": int(action),
                                      "future_decision_contribution": dec[position],
                                      "future_event_contribution": evt[position],
                                      "future_decision_advantage": dec_adv[position],
                                      "future_event_advantage": evt_adv[position],
                                      "future_decision_spread": float(np.ptp(dec_adv)),
                                      "future_event_spread": float(np.ptp(evt_adv))})
        for _, _, band in EVENT_LAG_BANDS:
            dec = np.zeros(len(frame))
            evt = frame[f"future_evt_{band}"].to_numpy(float)
            dec_adv, dec_res = centered_scores(dec, prob)
            evt_adv, evt_res = centered_scores(evt, prob)
            centering.extend([dec_res, evt_res])
            for position, action in enumerate(actions):
                distance_rows.append({"state_id": state_id, **context, "band_type": "event_lag",
                                      "band": band, "action_index": int(action),
                                      "future_decision_contribution": dec[position],
                                      "future_event_contribution": evt[position],
                                      "future_decision_advantage": dec_adv[position],
                                      "future_event_advantage": evt_adv[position],
                                      "future_decision_spread": float(np.ptp(dec_adv)),
                                      "future_event_spread": float(np.ptp(evt_adv))})

    action_frame = pd.DataFrame(action_rows)
    conflict = pd.DataFrame(conflict_rows)
    methods = pd.DataFrame(method_rows)
    gae = pd.DataFrame(gae_rows)
    ranking = pd.DataFrame(ranking_rows)
    distance = pd.DataFrame(distance_rows)
    action_frame.to_csv(OUT / "centered_advantage_components.csv.gz", index=False, compression="gzip")
    conflict.to_csv(OUT / "own_total_action_conflict.csv", index=False)
    methods[methods.method.eq("SafeMin")].to_csv(OUT / "safe_min_latency_component_alignment.csv", index=False)
    methods[methods.method.eq("PPO")].to_csv(OUT / "ppo_component_alignment.csv", index=False)
    gae.to_csv(OUT / "gae_component_alignment.csv", index=False)
    ranking.to_csv(OUT / "action_ranking_component_comparison.csv", index=False)
    distance.to_csv(OUT / "externality_distance_decomposition.csv", index=False)

    externality_rows = []
    for semantics in ("decision", "event"):
        for component in ("own", "future", "total"):
            externality_rows.append({"semantics": semantics, "component": component,
                                     **_summary(conflict[f"{component}_{semantics}_spread"])})
    externality_rows.extend([
        {"semantics": "event", "component": "future_over_own_spread_ratio", **_summary(conflict.future_own_spread_ratio)},
        {"semantics": "event", "component": "conflict_rate", **_summary(conflict.conflict_state.astype(float))},
    ])
    sign_rows = []
    action_components = pd.DataFrame(action_rows)
    for state_id, group in action_components.groupby("state_id"):
        future = group.future_event_advantage.to_numpy(float)
        nonzero = future[np.abs(future) > 1e-8]
        signs = np.sign(nonzero)
        sign_rows.append({"state_id": state_id, "positive_fraction": float(np.mean(signs > 0)) if len(signs) else np.nan,
                          "negative_fraction": float(np.mean(signs < 0)) if len(signs) else np.nan,
                          "near_zero_fraction": float(np.mean(np.abs(future) <= 1e-8)),
                          "within_state_sign_diversity": int(len(np.unique(signs)))})
    sign_frame = pd.DataFrame(sign_rows)
    for metric in ("positive_fraction", "negative_fraction", "near_zero_fraction", "within_state_sign_diversity"):
        externality_rows.append({"semantics": "event", "component": metric, **_summary(sign_frame[metric])})
    sign_frame.to_csv(OUT / "externality_sign_diagnostics.csv", index=False)
    externality = pd.DataFrame(externality_rows)
    externality.to_csv(OUT / "externality_spread_summary.csv", index=False)

    # Reuse the previous H20 coupled definition exactly: oracle H20 regret of SafeMin > 0.01.
    coupled = {}
    for state_id, group in prior_q.groupby("state_id"):
        best = float(group.q_h20.max())
        ml_action = int(group.myopic_action.iloc[0])
        myopic = float(group.loc[group.action_index.astype(int).eq(ml_action), "q_h20"].iloc[0])
        coupled[state_id] = best - myopic > CONFLICT_REGRET_TOL
    conflict["coupling"] = conflict.state_id.map(lambda sid: "coupled" if coupled[sid] else "easy")
    gae["coupling"] = gae.state_id.map(lambda sid: "coupled" if coupled[sid] else "easy")
    for filename, group_col in (("component_alignment_by_coupling.csv", "coupling"),
                                ("component_alignment_by_load.csv", "load"),
                                ("component_alignment_by_requirement.csv", "requirement"),
                                ("component_alignment_by_safe_set.csv", "safe_set_bin")):
        rows = []
        for value, group in gae.groupby(group_col, dropna=False, sort=True):
            for (kind, semantics, component), part in group.groupby(["gae_kind", "semantics", "component"]):
                x, y = part.gae_value.to_numpy(float), part.selected_advantage.to_numpy(float)
                nonzero = np.abs(y) > 1e-8
                rows.append({"group": group_col, "group_value": value, "gae_kind": kind,
                             "semantics": semantics, "component": component, "n": len(part),
                             "pearson": _safe_correlation(x, y), "spearman": _safe_correlation(x, y, True),
                             "sign_agreement": float(np.mean(np.sign(x[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else np.nan,
                             "sign_agreement_including_zero": float(np.mean(np.sign(x) == np.sign(np.where(nonzero, y, 0.0))))})
        pd.DataFrame(rows).to_csv(OUT / filename, index=False)

    gae_summary_rows = []
    for (kind, semantics, component), part in gae.groupby(["gae_kind", "semantics", "component"]):
        x, y = part.gae_value.to_numpy(float), part.selected_advantage.to_numpy(float)
        nonzero = np.abs(y) > 1e-8
        gae_summary_rows.append({"gae_kind": kind, "semantics": semantics, "component": component,
                                 "n": len(part), "pearson": _safe_correlation(x, y),
                                 "spearman": _safe_correlation(x, y, True),
                                 "sign_agreement": float(np.mean(np.sign(x[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else np.nan,
                                 "sign_agreement_including_zero": float(np.mean(np.sign(x) == np.sign(np.where(nonzero, y, 0.0))))})
    gae_summary = pd.DataFrame(gae_summary_rows)
    gae_summary.to_csv(OUT / "gae_component_alignment_summary.csv", index=False)
    rank_summary = ranking.groupby(["reference", "method"]).agg(
        state_count=("state_id", "nunique"), mean_spearman=("spearman", "mean"),
        median_spearman=("spearman", "median"), top1_hit_rate=("top1_hit", "mean"),
        top3_optimal_overlap=("top3_optimal_overlap", "mean"), top5_optimal_overlap=("top5_optimal_overlap", "mean"),
        mean_regret=("reference_regret", "mean"), median_regret=("reference_regret", "median"),
        p90_regret=("reference_regret", lambda value: float(np.percentile(value, 90))),
    ).reset_index()
    rank_summary.to_csv(OUT / "action_ranking_component_summary.csv", index=False)

    q20_best = prior_q.groupby("state_id").q_h20.max()
    q20_first = prior_q.drop_duplicates("state_id").set_index("state_id")
    prior_coupled_count = sum(
        q20_best[sid] - float(prior_q.loc[(prior_q.state_id == sid) &
                                          (prior_q.action_index.astype(int) == int(q20_first.loc[sid].myopic_action)), "q_h20"].iloc[0]) > CONFLICT_REGRET_TOL
        for sid in q20_best.index
    )
    safe_total = methods[(methods.method == "SafeMin") & (methods.semantics == "event") & (methods.component == "total")]
    ppo_total = methods[(methods.method == "PPO") & (methods.semantics == "event") & (methods.component == "total")]
    if float(np.max(centering, initial=0.0)) > 1e-10 or float(checks.own_future_decomposition_abs_residual.max()) > 1e-10:
        raise RuntimeError("A decomposition or policy-centering integrity condition failed")
    if int((checks.preexisting_pending_leakage != 0).sum()) or int(checks.duplicate_task_reward_count.sum()):
        raise RuntimeError("Pending-task leakage or duplicate task reward was found")
    if model_hash(side.network) != side_hash or (q_model is not None and model_hash(q_model) != q_hash):
        raise RuntimeError("Read-only side-model parameters changed")

    _make_plots(conflict, distance, gae, safe_total, ppo_total, action_frame)
    integrity = _replay_summary(checks)
    integrity.update({
        "matched_states": int(conflict.state_id.nunique()), "candidate_branches": int(len(branches)),
        "branch_task_records": int(len(provenance)),
        "duplicate_reward_count": int(provenance.duplicated(["episode", "state_id", "root_task_id", "forced_action_index", "task_id"]).sum()),
        "preexisting_pending_leakage_count": int((checks.preexisting_pending_leakage != 0).sum()),
        "own_future_decomposition_max_abs_residual": float(checks.own_future_decomposition_abs_residual.max()),
        "centering_mean_abs_residual": float(np.mean(centering)),
        "centering_max_abs_residual": float(np.max(centering, initial=0.0)),
        "actor_file_sha256": file_digest(CHECKPOINT / "actor.pt"),
        "critic_file_sha256": file_digest(CHECKPOINT / "critic.pt"),
        "centered_advantage_model_hash": side_hash, "old_q_model_hash": q_hash,
        "old_q_status": q_status, "prior_h20_coupled_state_count": int(prior_coupled_count),
    })
    (OUT / "integrity_summary.json").write_text(json.dumps(integrity, indent=2, sort_keys=True))
    _write_report(conflict, externality, gae_summary, rank_summary,
                  _strata_summary(conflict, gae, methods), integrity, q_status)
    return integrity


def _replay_summary(checks):
    mismatch_cols = [key for key in checks.columns if key.endswith("mismatch")]
    original = checks[checks.original_branch_return_abs_diff.notna()]
    return {
        **{key + "_count": int(checks[key].fillna(0).sum()) for key in mismatch_cols},
        "original_action_branch_count": int(len(original)),
        "original_decision_return_max_abs_diff": float(original.original_branch_decision_return_abs_diff.max()),
        "original_event_return_max_abs_diff": float(original.original_branch_event_return_abs_diff.max()),
        "original_transition_max_abs_diff": float(original.original_branch_max_transition_abs_diff.max()),
        "original_assignment_mismatch_count": int(original.original_task_outcome_mismatch_count.sum()),
    }


def _strata_summary(conflict, gae, method_frame):
    rows = []
    group_specs = [
        ("coupling", "coupled", conflict.coupling.eq("coupled")),
        ("coupling", "easy", conflict.coupling.eq("easy")),
        ("load", "high", conflict.load.astype(str).eq("high")),
        ("requirement", 0.9999, np.isclose(conflict.requirement, 0.9999)),
        ("safe_set_bin", "6-15", conflict.safe_set_bin.eq("6-15")),
    ]
    for group_name, label, mask in group_specs:
        states = set(conflict.loc[mask, "state_id"])
        c = conflict[conflict.state_id.isin(states)]
        g = gae[(gae.state_id.isin(states)) & (gae.semantics.eq("event")) & (gae.component.eq("future"))]
        m = method_frame[(method_frame.state_id.isin(states)) & (method_frame.semantics.eq("event")) & (method_frame.component.eq("total"))]
        for kind, part in g.groupby("gae_kind"):
            nonzero = np.abs(part.selected_advantage.to_numpy(float)) > 1e-8
            rows.append({
                "stratum": label, "count": int(len(c)), "gae_kind": kind,
                "mean_future_spread": float(c.future_event_spread.mean()) if len(c) else np.nan,
                "conflict_rate": float(c.conflict_state.mean()) if len(c) else np.nan,
                "gae_future_spearman": _safe_correlation(part.gae_value, part.selected_advantage, True),
                "gae_future_sign": float(np.mean(np.sign(part.gae_value.to_numpy(float)[nonzero]) == np.sign(part.selected_advantage.to_numpy(float)[nonzero]))) if nonzero.any() else np.nan,
                "safemin_total_mean_regret": float(m.loc[m.method.eq("SafeMin"), "regret"].mean()),
                "ppo_total_mean_regret": float(m.loc[m.method.eq("PPO"), "regret"].mean()),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "component_alignment_strata_summary.csv", index=False)
    return table


def _make_plots(conflict, distance, gae, safe_total, ppo_total, components):
    def save(fig, name):
        fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(conflict.own_event_spread, conflict.future_event_spread, s=9, alpha=.4)
    ax.set(xlabel="Own-task event-time spread", ylabel="Future event-time spread", title="Own reward vs future externality"); save(fig, "own_vs_future_spread.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(conflict.future_own_spread_ratio.replace([np.inf, -np.inf], np.nan).dropna().clip(upper=20), bins=35)
    ax.set(xlabel="Future / own spread (clipped at 20)", ylabel="States", title="Relative future externality"); save(fig, "future_own_spread_ratio.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Exact set", "Jaccard", "Conflict"], [conflict.own_total_exact_optimal_set_agreement.mean(), conflict.own_total_optimal_jaccard.mean(), conflict.conflict_state.mean()])
    ax.set_ylim(0, 1); ax.set_ylabel("Rate / similarity"); ax.set_title("Own vs total optimal sets"); save(fig, "own_total_optimal_conflict.png")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(safe_total.loc[safe_total.conflict_state, "regret"], bins=30)
    ax.set(xlabel="Total event-time regret", ylabel="States", title="SafeMin regret on conflict states"); save(fig, "safemin_total_regret_conflicts.png")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(ppo_total.loc[ppo_total.conflict_state, "regret"], bins=30, color="#e15759")
    ax.set(xlabel="Total event-time regret", ylabel="States", title="PPO regret on conflict states"); save(fig, "ppo_total_regret_conflicts.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    event = gae[gae.semantics.eq("event")]
    labels, values = [], []
    for kind in ("raw", "actor_normalized"):
        for component in ("own", "future", "total"):
            part = event[(event.gae_kind == kind) & (event.component == component)]
            labels.append(f"{kind}\n{component}")
            values.append(_safe_correlation(part.gae_value, part.selected_advantage, True))
    ax.bar(labels, values); ax.axhline(0, color="black", lw=.7)
    ax.set(ylabel="Spearman", title="GAE alignment with event-time components"); save(fig, "gae_component_alignment.png")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(conflict.total_decision_spread, conflict.total_event_spread, s=8, alpha=.4)
    ax.set(xlabel="Decision-time total spread", ylabel="Event-time total spread", title="Decision vs event-time landscape"); save(fig, "decision_vs_event_total.png")
    fig, ax = plt.subplots(figsize=(6, 4)); labels = ["low", "medium", "high"]
    ax.boxplot([conflict.loc[conflict.load.astype(str).eq(label), "future_event_spread"] for label in labels], tick_labels=labels, showfliers=False)
    ax.set(xlabel="Frozen load group", ylabel="Future event-time spread", title="Externality by load"); save(fig, "future_spread_by_load.png")
    fig, ax = plt.subplots(figsize=(8, 4)); dist = distance[distance.band_type.eq("decision_distance")]
    order = [name for _, _, name in DISTANCE_BANDS]; y = dist.groupby("band").future_event_spread.mean().reindex(order)
    ax.plot(order, y, marker="o"); ax.tick_params(axis="x", rotation=20)
    ax.set(xlabel="Downstream task distance", ylabel="Mean within-state spread", title="Externality accumulation by task distance"); save(fig, "externality_accumulation_by_task_distance.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in (("coupled", "#e15759"), ("easy", "#4c78a8")):
        sample = conflict.loc[conflict.coupling.eq(label), "future_event_spread"]
        ax.hist(sample, bins=25, alpha=.55, label=f"{label} n={len(sample)}", color=color)
    ax.legend(); ax.set(xlabel="Future event-time spread", ylabel="States", title="Coupled vs easy"); save(fig, "coupled_easy_externality_spread.png")
    top = conflict.sort_values(["future_event_spread", "state_id"], ascending=[False, True]).head(5)
    top.to_csv(OUT / "representative_case_studies.csv", index=False)
    fig, axes = plt.subplots(5, 1, figsize=(11, 14))
    for ax, row in zip(axes, top.itertuples(index=False)):
        f = components[components.state_id.eq(row.state_id)].sort_values("action_index")
        ax.plot(f.action_index, f.own_event_advantage, marker=".", label="Own")
        ax.plot(f.action_index, f.future_event_advantage, marker=".", label="Future")
        ax.plot(f.action_index, f.total_event_advantage, marker=".", label="Total")
        for column, color, label in (("own_event_advantage", "#4c78a8", "Own-opt"),
                                     ("future_event_advantage", "#f28e2b", "Future-opt"),
                                     ("total_event_advantage", "#59a14f", "Total-opt")):
            opt = f[np.isclose(f[column], f[column].max(), rtol=0.0, atol=TIE_ATOL)]
            ax.scatter(opt.action_index, opt[column], marker="*", s=70, color=color, label=label)
        ax.set_title(f"{row.state_id}; future spread={row.future_event_spread:.4f}; conflict={row.conflict_state}")
        ax.set_ylabel("Centered value")
    axes[-1].set_xlabel("Effective action index"); axes[0].legend(ncol=3)
    save(fig, "representative_top5_action_components.png")


def _md_table(frame, columns=None):
    show = frame if columns is None else frame[columns]
    if show.empty:
        return "(empty)"
    def cell(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [str(column) for column in show.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |"
                 for row in show.itertuples(index=False, name=None))
    return "\n".join(lines)


def _write_report(conflict, externality, gae_summary, ranking_summary, strata_summary, integrity, q_status):
    event_gae = gae_summary[gae_summary.semantics.eq("event")]
    raw_own = float(event_gae.loc[(event_gae.gae_kind == "raw") & (event_gae.component == "own"), "spearman"].iloc[0])
    raw_future = float(event_gae.loc[(event_gae.gae_kind == "raw") & (event_gae.component == "future"), "spearman"].iloc[0])
    raw_total = float(event_gae.loc[(event_gae.gae_kind == "raw") & (event_gae.component == "total"), "spearman"].iloc[0])
    safe = pd.read_csv(OUT / "safe_min_latency_component_alignment.csv")
    ppo = pd.read_csv(OUT / "ppo_component_alignment.csv")
    safe_total = safe[(safe.semantics == "event") & (safe.component == "total")]
    ppo_total = ppo[(ppo.semantics == "event") & (ppo.component == "total")]
    conflict_rows = conflict[conflict.conflict_state]
    mean_future, median_future = float(conflict.future_event_spread.mean()), float(conflict.future_event_spread.median())
    disjoint_rate = float(conflict.own_total_disjoint.mean())
    safe_conflict_regret = float(safe_total.loc[safe_total.conflict_state, "regret"].mean()) if len(conflict_rows) else 0.0
    dec_evt_rank = float(conflict.spearman.median())
    dec_evt_jaccard = float(conflict.optimal_set_jaccard.mean())
    if mean_future < 1e-3 and disjoint_rate < .05 and safe_conflict_regret < .01:
        primary, secondary = "A", "None"
        why = "Future spread, optimal-set conflict, and SafeMin conflict regret are all small under the fixed tie/regret tolerances."
    elif dec_evt_rank < .8 or dec_evt_jaccard < .8:
        primary, secondary = "C", "B" if mean_future >= 1e-3 and raw_future < raw_own else "E"
        why = "Decision-index and event-time return rankings/optimal sets differ materially."
    elif mean_future >= 1e-3 and disjoint_rate >= .05 and safe_conflict_regret >= .01 and raw_future < raw_own:
        primary, secondary = "B", "None"
        why = "Future spread, own/total conflicts, and SafeMin conflict regret are present, while raw PPO GAE aligns less with future than own advantage."
    elif raw_future >= .2:
        primary, secondary = "D", "None"
        why = "Raw PPO GAE has material alignment with the future component."
    else:
        primary, secondary = "E", "B" if mean_future >= 1e-3 else "A"
        why = "Evidence is mixed across externality size, conflict, baseline regret, and GAE alignment."
    worthwhile = bool(mean_future >= 1e-3 and disjoint_rate >= .05 and safe_conflict_regret >= .01)
    rec_externality = bool(worthwhile and raw_future < raw_own)
    rec_temporal = bool(primary == "C")

    req = conflict.groupby("requirement").agg(states=("state_id", "nunique"), future_spread_mean=("future_event_spread", "mean"), conflict_rate=("conflict_state", "mean"), own_total_jaccard=("own_total_optimal_jaccard", "mean")).reset_index()
    load = conflict.groupby("load").agg(states=("state_id", "nunique"), future_spread_mean=("future_event_spread", "mean"), conflict_rate=("conflict_state", "mean")).reset_index()
    safe = conflict.groupby("safe_set_bin").agg(states=("state_id", "nunique"), future_spread_mean=("future_event_spread", "mean"), conflict_rate=("conflict_state", "mean")).reset_index()
    distance = pd.read_csv(OUT / "externality_distance_decomposition.csv")
    distance = distance[distance.band_type.eq("decision_distance")].groupby("band").future_event_contribution.mean().reindex([name for _, _, name in DISTANCE_BANDS])
    lines = [
        "# Asynchronous Task Credit and Future Externality Audit", "",
        f"Matched source: `{MANIFEST_PATH.relative_to(ROOT)}`; exactly {integrity['matched_states']} frozen states and {integrity['candidate_branches']} candidate-action branches.",
        f"Frozen Masked Pair PPO stochastic continuation; gamma={GAMMA:g}; no retraining and no production code changed.", "",
        "## Timing and reward semantics", "",
        "Decision time `t_i` is the task arrival/action time in the frozen rollout. The actual reward resolution time `tau_i` is production's `Primary_Start + Task_Delay`, validated against the earlier actual replica completion `min(Primary_End, Backup_End)`. Decision-index returns use the originating task order: `r_i + sum_{j>i} gamma^(t_j-t_i) r_j`. Event-time returns use actual resolution times: `sum_{j>=i} gamma^(tau_j-t_i) r_j`. Own/future membership depends only on decision index.",
        "Prior tasks `j<i` still pending at `t_i` are logged as pre-existing provenance and excluded from the root return. Only `j>i` contributes to future externality.", "",
        "## Integrity", "",
        f"Matched states: {integrity['matched_states']}; candidate branches: {integrity['candidate_branches']}; task reward provenance rows: {integrity['branch_task_records']}; duplicate rewards: {integrity['duplicate_reward_count']}; pre-existing leakage: {integrity['preexisting_pending_leakage_count']}.",
        f"State/mask/probability/backlog/task/hazard/reliability mismatch counts: {integrity['state_mismatch_count']}/{integrity['mask_mismatch_count']}/{integrity['probability_mismatch_count']}/{integrity['backlog_mismatch_count']}/{integrity['task_mismatch_count']}/{integrity['target_effective_rates_mismatch_count']}/{integrity['target_reliabilities_mismatch_count']}.",
        f"Arrival/Torch stream mismatch counts: {integrity['future_arrival_mismatch_count']}/{integrity['future_torch_rng_mismatch_count']}. There are no separate runtime per-replica Bernoulli draws in the current simulator; environmental randomness is the fixed arrival/spatial streams plus stochastic policy actions.",
        f"Original-action full-episode return max error (decision/event): {integrity['original_decision_return_max_abs_diff']:.3g}/{integrity['original_event_return_max_abs_diff']:.3g}; original outcome row mismatches: {integrity['original_assignment_mismatch_count']}; Qevent=Oevent+Fevent max residual: {integrity['own_future_decomposition_max_abs_residual']:.3g}.",
        f"Policy-centering weighted residual mean/max: {integrity['centering_mean_abs_residual']:.3g}/{integrity['centering_max_abs_residual']:.3g}. Actor/Critic checkpoint SHA-256: `{integrity['actor_file_sha256']}` / `{integrity['critic_file_sha256']}`.", "",
        "## Event-time externality size", "",
        f"Own spread mean/median/P90/P95: {conflict.own_event_spread.mean():.4f}/{conflict.own_event_spread.median():.4f}/{conflict.own_event_spread.quantile(.9):.4f}/{conflict.own_event_spread.quantile(.95):.4f}.",
        f"Future spread mean/median/P90/P95: {mean_future:.4f}/{median_future:.4f}/{conflict.future_event_spread.quantile(.9):.4f}/{conflict.future_event_spread.quantile(.95):.4f}.",
        f"Future/own spread ratio median/P75/P90: {conflict.future_own_spread_ratio.median():.4f}/{conflict.future_own_spread_ratio.quantile(.75):.4f}/{conflict.future_own_spread_ratio.quantile(.9):.4f}.",
        f"Own/total exact set agreement {conflict.own_total_exact_optimal_set_agreement.mean():.1%}; mean optimal-set Jaccard {conflict.own_total_optimal_jaccard.mean():.3f}; disjoint rate {disjoint_rate:.1%}; conflict rate (disjoint or own-optimal total regret > {CONFLICT_REGRET_TOL}) {conflict.conflict_state.mean():.1%}.",
        f"SafeMin event total regret mean/all and conflict: {safe_total.regret.mean():.4f}/{safe_total.loc[safe_total.conflict_state, 'regret'].mean():.4f}; PPO sampled action: {ppo_total.regret.mean():.4f}/{ppo_total.loc[ppo_total.conflict_state, 'regret'].mean():.4f}.", "",
        "## GAE and time-semantic comparison", "",
        _md_table(gae_summary, ["gae_kind", "semantics", "component", "pearson", "spearman", "sign_agreement", "sign_agreement_including_zero"]), "",
        f"For raw GAE, event-time own/future/total Spearman={raw_own:.4f}/{raw_future:.4f}/{raw_total:.4f}.",
        f"Decision-vs-event total median Spearman/Kendall={conflict.spearman.median():.4f}/{conflict.kendall.median():.4f}; sign agreement={conflict.sign_agreement.mean():.1%}; mean optimal-set Jaccard={conflict.optimal_set_jaccard.mean():.3f}; exact optimal-set agreement={conflict.optimal_set_exact_agreement.mean():.1%}.", "",
        "## Frozen H20 coupled/easy and fixed strata", "",
        "Coupled is unchanged from the previous H20 audit: the best H20 decision-index return exceeds SafeMin's H20 return by more than 0.01. Load, requirement, and safe-set bins reuse the prior frozen manifest.",
        _md_table(strata_summary), "", "Requirement:", _md_table(req), "", "Load:", _md_table(load), "", "Effective safe-set:", _md_table(safe), "",
        "## Future reward by downstream distance", "",
        "These groups are descriptive propagation distances, not learned horizon parameters. Event-time bins are descriptive only.",
        _md_table(distance.rename("mean_event_discounted_contribution").reset_index()), "",
        "## Action-ranking comparison", "",
        f"Old-Q scoring status: {q_status}. Centered Advantage and Q checkpoints were read only.",
        _md_table(ranking_summary), "",
        "## Conclusion and go/no-go", "",
        f"Primary = **{primary}**, secondary = **{secondary}**. {why}",
        f"Evidence: mean future spread={mean_future:.4f}; own/total disjoint={disjoint_rate:.1%}; conflict SafeMin regret={safe_conflict_regret:.4f}; raw GAE own/future rho={raw_own:.4f}/{raw_future:.4f}; median decision/event rho={dec_evt_rank:.4f}; optimal-set Jaccard={dec_evt_jaccard:.3f}.",
        f"Worthwhile long-term externality: **{'YES' if worthwhile else 'NO'}**. Recommend Externality-Aware/Counterfactual Advantage: **{'YES' if rec_externality else 'NO'}**. Recommend changing current PPO temporal GAE semantics: **{'YES' if rec_temporal else 'NO'}**.", "",
        "## Figures", "",
        "- Own vs future spread: `own_vs_future_spread.png`", "- Future/own ratio: `future_own_spread_ratio.png`", "- Optimal-set conflict: `own_total_optimal_conflict.png`", "- SafeMin/PPO regret: `safemin_total_regret_conflicts.png`, `ppo_total_regret_conflicts.png`", "- GAE alignment: `gae_component_alignment.png`", "- Decision/event landscape: `decision_vs_event_total.png`", "- Load: `future_spread_by_load.png`", "- Task-distance accumulation: `externality_accumulation_by_task_distance.png`", "- Coupled/easy: `coupled_easy_externality_spread.png`", "- Top-5 cases: `representative_top5_action_components.png`", "",
    ]
    (OUT / "ASYNC_TASK_CREDIT_AUDIT.md").write_text("\n".join(map(str, lines)))
    decision = {
        "primary": primary, "secondary": secondary,
        "future_externality_worth_learning": worthwhile,
        "recommend_externality_aware_counterfactual_advantage": rec_externality,
        "recommend_change_current_gae_temporal_semantics": rec_temporal,
        "evidence": {"mean_future_event_spread": mean_future, "own_total_disjoint_rate": disjoint_rate,
                     "conflict_rate": float(conflict.conflict_state.mean()), "safemin_conflict_regret": safe_conflict_regret,
                     "raw_gae_own_spearman": raw_own, "raw_gae_future_spearman": raw_future,
                     "decision_event_total_spearman_median": dec_evt_rank,
                     "decision_event_optimal_set_jaccard_mean": dec_evt_jaccard},
        "frozen_old_q_status": q_status,
    }
    (OUT / "async_credit_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))


def run(workers=2, episodes=None):
    before = digest_protected_inputs()
    branch_status = run_branches(workers, episodes)
    if episodes is not None:
        return branch_status
    integrity = analyze_results()
    after = digest_protected_inputs()
    if before != after:
        changed = [path for path in before if before.get(path) != after.get(path)]
        raise RuntimeError(f"Protected prior artifacts changed: {changed}")
    manifest = {
        "scope": "read-only asynchronous task credit/future externality audit",
        "frozen_manifest": str(MANIFEST_PATH.relative_to(ROOT)), "matched_states": 1000,
        "gamma": GAMMA, "tie_atol": TIE_ATOL, "conflict_regret_tolerance": CONFLICT_REGRET_TOL,
        "terminal": "all 200 tasks resolved and all replicas drained naturally",
        "reward_resolution_timestamp": "Primary_Start + Task_Delay, checked against min(Primary_End, Backup_End)",
        "common_random_numbers": ["episode-specific fixed arrival seed", "fixed episode spatial-risk seed", "same task rows", "same Torch action-draw stream; forced action consumes its usual draw"],
        "actor_sha256": before.get(str((CHECKPOINT / "actor.pt").relative_to(ROOT))),
        "critic_sha256": before.get(str((CHECKPOINT / "critic.pt").relative_to(ROOT))),
        "protected_prior_artifact_sha256": before,
        "branch_count": int(integrity["candidate_branches"]),
        "reward_provenance_count": int(integrity["branch_task_records"]),
        "preexisting_task_rule": "pending tasks with decision_index < root are recorded but excluded from root future return",
        "production_modified": False, "training_run": False, "integrity": integrity,
    }
    (OUT / "reward_provenance_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return integrity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--episodes", nargs="*", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze_results() if args.analyze_only else run(args.workers, args.episodes)
    print(json.dumps(result, indent=2, default=str), flush=True)
