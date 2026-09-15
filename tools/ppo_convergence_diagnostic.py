"""Checkpoint based PPO convergence diagnostics.

The diagnostic deliberately lives outside the simulator.  It trains one Flat
and one Pair-Scoring agent per trial on a continuous MainLoop trajectory and
evaluates frozen copies on the same held-out random streams at each checkpoint.
No formal environment or PPO implementation is modified by this module.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from config.params import params
from core.main_loop import MainLoop
from Project_main import build_pair_correlations
from tools.pair_policy_diagnostics import (
    RELIABILITY_REQUIREMENT_TIERS,
    TASK_ASSIGNMENT_COLUMNS,
    performance_summary,
    requirement_sensitivity_rows,
    rho_counterfactual_rows,
    sample_probe_indices,
    summarize_behavior,
    policy_entropy,
)
from tools.paired_ppo_experiment import (
    BOOTSTRAP_SEED,
    DEFAULT_MASTER_SEED,
    SEED_COLUMNS,
    FrozenEvaluationPPOAgent,
    _agent_kwargs,
    _as_json_value,
    _parameter_count,
    _state_dict_snapshot,
    assert_state_dict_unchanged,
    generate_seed_plan,
    scoped_environment_seeds,
    sha256_file,
)
from tools.pair_policy_diagnostics import DiagnosticPPOAgent


DEFAULT_CHECKPOINTS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def parse_checkpoints(value: str | Iterable[int] = DEFAULT_CHECKPOINTS) -> list[int]:
    """Parse, validate, sort, and de-duplicate checkpoint episode numbers."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("checkpoints must not be empty")
        try:
            values = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError("checkpoints must be comma-separated integers") from exc
    else:
        try:
            values = [int(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoints must be integer values") from exc
    if any(item < 0 for item in values):
        raise ValueError("checkpoints cannot be negative")
    values = sorted(set(values))
    if not values:
        raise ValueError("checkpoints must not be empty")
    return values


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _frame_from_assignments(rows: Sequence[Sequence]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=TASK_ASSIGNMENT_COLUMNS)
    for col in ("Reliability_Requirement", "Task_Reward", "Task_Delay", "action_index", "server_j", "server_k"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def mean_policy_entropy_by_requirement(requirement_df: pd.DataFrame) -> pd.DataFrame:
    """Return mean and normalized policy entropy for each requirement tier."""
    max_entropy = float(math.log(params.num_actions))
    rows = []
    for requirement in RELIABILITY_REQUIREMENT_TIERS:
        group = requirement_df[np.isclose(requirement_df["Reliability_Requirement"], requirement)]
        value = float(group["Policy_Entropy"].mean()) if not group.empty else np.nan
        rows.append({
            "Reliability_Requirement": float(requirement),
            "Mean_Policy_Entropy": value,
            "Normalized_Policy_Entropy": value / max_entropy if np.isfinite(value) else np.nan,
        })
    return pd.DataFrame(rows)


def action_concentration(task_rows: pd.DataFrame, num_actions: int = 28) -> dict:
    actions = pd.to_numeric(task_rows.get("action_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)
    if actions.empty:
        return {"Unique_Actions_Selected": 0, "Top1_Action_Fraction": np.nan,
                "Top3_Action_Fraction": np.nan, "Top5_Action_Fraction": np.nan}
    counts = actions.value_counts().to_numpy(dtype=float)
    total = float(len(actions))
    return {"Unique_Actions_Selected": int(actions.nunique()),
            "Top1_Action_Fraction": float(counts[:1].sum() / total),
            "Top3_Action_Fraction": float(counts[:3].sum() / total),
            "Top5_Action_Fraction": float(counts[:5].sum() / total)}


def server_usage_entropy(task_rows: pd.DataFrame, num_servers: int = 8) -> dict:
    counts = np.zeros(int(num_servers), dtype=float)
    for column in ("server_j", "server_k"):
        if column not in task_rows:
            continue
        for value in pd.to_numeric(task_rows[column], errors="coerce").dropna():
            index = int(value) - 1
            if 0 <= index < num_servers:
                counts[index] += 1
    total = counts.sum()
    if total <= 0:
        result = {"Server_Usage_Entropy": np.nan, "Normalized_Server_Usage_Entropy": np.nan}
        result.update({f"Server_{index + 1}_Usage_Fraction": np.nan for index in range(num_servers)})
        return result
    probabilities = counts / total
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    result = {"Server_Usage_Entropy": entropy,
              "Normalized_Server_Usage_Entropy": entropy / math.log(num_servers)}
    result.update({f"Server_{index + 1}_Usage_Fraction": float(probabilities[index])
                   for index in range(num_servers)})
    return result


def selected_rho_summary(task_rows: pd.DataFrame, true_rho: Sequence[float], requirement=None) -> dict:
    actions = pd.to_numeric(task_rows.get("action_index", pd.Series(dtype=float)), errors="coerce")
    if requirement is not None and "Reliability_Requirement" in task_rows:
        mask = np.isclose(pd.to_numeric(task_rows["Reliability_Requirement"], errors="coerce"), requirement)
        actions = actions.loc[mask]
    values = np.asarray(true_rho, dtype=float)
    rho = np.asarray([values[int(a)] for a in actions.dropna() if 0 <= int(a) < len(values)], dtype=float)
    if rho.size == 0:
        return {"Selected_Rho_Mean": np.nan, "Selected_Rho_Std": np.nan,
                "Selected_Rho_Median": np.nan, "Selected_Rho_Q25": np.nan,
                "Selected_Rho_Q75": np.nan, "Selected_Rho_Min": np.nan,
                "Selected_Rho_Max": np.nan}
    return {"Selected_Rho_Mean": float(rho.mean()),
            "Selected_Rho_Std": float(rho.std(ddof=1)) if rho.size > 1 else 0.0,
            "Selected_Rho_Median": float(np.median(rho)),
            "Selected_Rho_Q25": float(np.quantile(rho, .25)),
            "Selected_Rho_Q75": float(np.quantile(rho, .75)),
            "Selected_Rho_Min": float(rho.min()),
            "Selected_Rho_Max": float(rho.max())}


def _write_json(path: Path, payload: Mapping):
    path.write_text(json.dumps(_as_json_value(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _git_commit(project_root: Path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return None


def _evaluate_checkpoint(actor_mode, trained_agent, trial, checkpoint, eval_episodes, tasks,
                         true_rho, action_pairs, num_probes):
    """Evaluate a frozen clone without changing parameters or Torch RNG."""
    frozen = FrozenEvaluationPPOAgent.from_trained(trained_agent)
    before_trained = _state_dict_snapshot(trained_agent)
    before_frozen = _state_dict_snapshot(frozen)
    before_optimizer = {name: copy.deepcopy(getattr(trained_agent, name).state_dict())
                       for name in ("optimizer_policy", "optimizer_value")}
    rollout_fields = ("states", "actions", "rewards", "next_states", "dones", "old_log_probs",
                      "delta_times", "task_ids", "task_id_to_transition_index", "pending_task_rewards")
    before_rollout = {name: copy.deepcopy(getattr(trained_agent, name)) for name in rollout_fields}
    torch_state = torch.get_rng_state()
    try:
        with scoped_environment_seeds(trial["Eval_Arrival_Seed"], trial["Eval_Spatial_Seed"]):
            eval_loop = MainLoop(frozen, int(eval_episodes), int(tasks), params.num_states, params.num_actions)
            eval_loop.EP()
    finally:
        torch.set_rng_state(torch_state)
    assert_state_dict_unchanged(before_trained, trained_agent)
    assert_state_dict_unchanged(before_frozen, frozen)
    optimizer_after = {name: getattr(trained_agent, name).state_dict()
                       for name in ("optimizer_policy", "optimizer_value")}
    if not _nested_equal(before_optimizer, optimizer_after):
        raise RuntimeError("frozen evaluation changed the training optimizer state")
    if not _nested_equal(before_rollout, {name: getattr(trained_agent, name) for name in rollout_fields}):
        raise RuntimeError("frozen evaluation changed the training rollout state")
    if not torch.equal(torch_state, torch.get_rng_state()):
        raise RuntimeError("checkpoint evaluation changed the training Torch RNG state")
    expected = int(eval_episodes) * int(tasks)
    if len(eval_loop.task_Assignments_info) != expected:
        raise RuntimeError(f"evaluation produced {len(eval_loop.task_Assignments_info)} rows; expected {expected}")
    task_frame = _frame_from_assignments(eval_loop.task_Assignments_info)
    overall, tiers = performance_summary(eval_loop.task_Assignments_info, true_rho)
    probe_indices = sample_probe_indices(len(frozen.observation_archive), num_probes)
    probes = [frozen.observation_archive[i] for i in probe_indices]
    if not probes:
        raise RuntimeError("evaluation produced no probe states")
    req_df = requirement_sensitivity_rows(frozen.policy_old, probes, true_rho, action_pairs)
    rho_df = rho_counterfactual_rows(frozen.policy_old, probes, true_rho, action_pairs) if actor_mode == "pair_scoring" else None
    behavior = summarize_behavior(actor_mode, req_df, rho_df)
    entropy = mean_policy_entropy_by_requirement(req_df)
    concentration = action_concentration(task_frame)
    usage = server_usage_entropy(task_frame, params.serverNo)
    rho_summary = selected_rho_summary(task_frame, true_rho)
    rho_by_requirement = {
        float(requirement): selected_rho_summary(task_frame, true_rho, requirement)
        for requirement in RELIABILITY_REQUIREMENT_TIERS
    }
    return {
        "task_frame": task_frame,
        "overall": overall,
        "tiers": tiers,
        "requirement": req_df,
        "rho": rho_df,
        "behavior": behavior,
        "entropy": entropy,
        "concentration": concentration,
        "usage": usage,
        "rho_summary": rho_summary,
        "rho_by_requirement": rho_by_requirement,
        "frozen": frozen,
        "evaluation_integrity": {
            "training_state_unchanged": True,
            "frozen_state_unchanged": True,
            "optimizer_unchanged": True,
            "rollout_unchanged": True,
            "torch_rng_restored": True,
        },
    }


def _nested_equal(left, right):
    """Compare nested optimizer/rollout structures containing tensors and arrays."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right, equal_nan=True)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    try:
        result = left == right
        return bool(result) if not isinstance(result, (np.ndarray, torch.Tensor)) else bool(result.all())
    except (TypeError, ValueError, RuntimeError):
        return False


def _train_agent(actor_mode, trial, pair_correlations):
    torch.manual_seed(int(trial["Torch_Init_Seed"]))
    agent = DiagnosticPPOAgent(**_agent_kwargs(actor_mode, pair_correlations, trial["PPO_Minibatch_Seed"]))
    torch.manual_seed(int(trial["Train_Action_Seed"]))
    return agent


def advance_training_loop(train_loop, checkpoints):
    """Advance one existing MainLoop through sorted checkpoints without reset."""
    completed = []
    for checkpoint in parse_checkpoints(checkpoints):
        train_loop.total_episodes = int(checkpoint)
        train_loop.EP()
        if train_loop.this_episode != int(checkpoint):
            raise RuntimeError(
                f"training continuation reached episode {train_loop.this_episode}, expected {checkpoint}"
            )
        completed.append(int(train_loop.this_episode))
    return completed


def _training_curve(loop):
    rows = []
    for i, entry in enumerate(loop.log_data):
        episode, rolling_reward, episode_reward, rolling_delay = entry
        delay = loop.ep_delay_list[i] if i < len(loop.ep_delay_list) else np.nan
        rows.append({"Episode": int(episode), "Episode_Reward": float(episode_reward),
                     "Rolling_Avg_Reward": float(rolling_reward), "Episode_Delay": float(delay),
                     "Rolling_Avg_Delay": float(rolling_delay)})
    return pd.DataFrame(rows)


def _checkpoint_rows(trial_id, actor_mode, checkpoint, result):
    overall = result["overall"]
    row = {"Trial_ID": trial_id, "Actor_Mode": actor_mode, "Checkpoint_Episode": checkpoint,
           "Reliability_Satisfaction_Rate": overall.get("Reliability_Satisfaction_Rate", np.nan),
           "Mean_Task_Delay": overall.get("Mean_Task_Delay", np.nan),
           "Mean_Reward": overall.get("Mean_Reward", np.nan),
           "Mean_Selected_Rho": overall.get("Mean_Selected_Rho", np.nan)}
    return row


def _behavior_row(trial_id, actor_mode, checkpoint, result):
    b = result["behavior"]
    row = {"Trial_ID": trial_id, "Actor_Mode": actor_mode, "Checkpoint_Episode": checkpoint}
    keys = ["Mean_Delta_Expected_Rho_HighMinusLow", "Mean_TV_R09999_vs_R09", "Mean_JS_R09999_vs_R09",
            "Argmax_Change_Rate_R09999_vs_R09", "Mean_TV_True_vs_ZeroRho", "Mean_JS_True_vs_ZeroRho",
            "Argmax_Change_Rate_True_vs_ZeroRho", "Mean_TV_True_vs_ShuffledRho",
            "Mean_JS_True_vs_ShuffledRho", "Argmax_Change_Rate_True_vs_ShuffledRho"]
    for label in ("R09", "R099", "R0999", "R09999"):
        keys.extend((f"Mean_Expected_Rho_{label}", f"Mean_Low_Rho_Mass_{label}"))
    for key in keys:
        value = b.get(key, np.nan)
        row[key] = np.nan if isinstance(value, str) else value
    return row


def _write_analysis_report(output_dir: Path, result: Mapping, checkpoints: list[int]):
    overall = result["checkpoint_overall"]
    tiers = result["checkpoint_by_requirement"]
    behavior = result["checkpoint_behavior"]
    entropy = result["checkpoint_entropy"]
    concentration = result["checkpoint_action_concentration"]
    rho = result["checkpoint_selected_rho"]
    convergence = result["aggregate_convergence"]
    shown = [value for value in (0, 20, 40, 60, 80, 100) if value in checkpoints]

    def mode_table(mode):
        rows=[]
        for checkpoint in shown:
            o=overall[(overall.Actor_Mode == mode) & (overall.Checkpoint_Episode == checkpoint)]
            tr=tiers[(tiers.Actor_Mode == mode) & (tiers.Checkpoint_Episode == checkpoint)]
            er=entropy[(entropy.Actor_Mode == mode) & (entropy.Checkpoint_Episode == checkpoint)]
            br=behavior[(behavior.Actor_Mode == mode) & (behavior.Checkpoint_Episode == checkpoint)]
            rr=rho[(rho.Actor_Mode == mode) & (rho.Checkpoint_Episode == checkpoint)]
            if o.empty: continue
            def tier_value(frame, req, col):
                value=frame[np.isclose(frame.Reliability_Requirement,req)][col]
                return float(value.mean()) if len(value) else np.nan
            row={"Checkpoint":checkpoint,"Overall RSR":float(o.Reliability_Satisfaction_Rate.mean()),
                 "R=.999 RSR":tier_value(tr,.999,"Reliability_Satisfaction_Rate"),
                 "R=.9999 RSR":tier_value(tr,.9999,"Reliability_Satisfaction_Rate"),
                 "Selected rho":float(o.Mean_Selected_Rho.mean()),
                 "Entropy(.9999)":tier_value(er,.9999,"Mean_Policy_Entropy"),
                 "Delta Expected rho high-low":float(br.Mean_Delta_Expected_Rho_HighMinusLow.mean())}
            if mode == "pair_scoring":
                row["TV true-zero"]=float(br.Mean_TV_True_vs_ZeroRho.mean())
                row["JS true-zero"]=float(br.Mean_JS_True_vs_ZeroRho.mean())
            rows.append(row)
        return pd.DataFrame(rows)

    pair_table=mode_table("pair_scoring"); flat_table=mode_table("flat")
    lines=["# PPO Convergence-Length Diagnostic","","## 1. Configuration","",
           f"- Trials: {result['num_trials']}",f"- Checkpoints: {checkpoints}",
           f"- Training tasks per episode: {result['tasks']}",f"- Evaluation episodes per checkpoint: {result['eval_episodes']}",
           f"- Evaluation tasks per episode: {result['tasks']}",f"- Probe states: {result['num_probes']}",
           f"- Master seed: {result['metadata']['master_seed']}",
           f"- Max action entropy: {result['metadata']['max_action_entropy']:.12f}","",
           "## 2. Integrity checks","",
           "Each trial/actor used one agent and one MainLoop. Training episode counters were asserted at each checkpoint; held-out arrival and spatial seeds are reused at every checkpoint and shared by Flat/Pair within trial. Evaluation uses a frozen greedy clone. We checked actor/critic state, both optimizer states, rollout buffers, and Torch RNG before/after evaluation.","",
           "## 3. Runtime","",
           f"- Wall-clock seconds: {result['metadata']['elapsed_seconds']:.2f}",
           f"- Successful Trial × Actor trajectories: {result['metadata']['successful_runs']}",
           f"- Failures: {result['metadata'].get('failed_runs',0)}","",
           "## 4. Training trajectory","",
           f"- Maximum completed episode: {max(checkpoints)}; each training curve should contain episodes 1…{max(checkpoints)} exactly once.","",
           "## 5. Overall evaluation vs checkpoint","",
           "Pair-Scoring PPO:","",pair_table.to_string(index=False) if not pair_table.empty else "No requested checkpoints present.","",
           "Flat PPO:","",flat_table.to_string(index=False) if not flat_table.empty else "No requested checkpoints present.","",
           "## 6. Reliability-tier RSR vs checkpoint","",
           tiers.groupby(["Actor_Mode","Checkpoint_Episode","Reliability_Requirement"],as_index=False).Reliability_Satisfaction_Rate.mean().to_string(index=False),"",
           "## 7. Policy entropy vs checkpoint","",
           entropy.groupby(["Actor_Mode","Checkpoint_Episode","Reliability_Requirement"],as_index=False)[["Mean_Policy_Entropy","Normalized_Policy_Entropy"]].mean().to_string(index=False),"",
           "## 8. rho counterfactual sensitivity vs checkpoint","",
           behavior[[c for c in behavior.columns if c in ("Actor_Mode","Checkpoint_Episode","Mean_TV_True_vs_ZeroRho","Mean_JS_True_vs_ZeroRho","Mean_TV_True_vs_ShuffledRho","Mean_JS_True_vs_ShuffledRho")]].groupby(["Actor_Mode","Checkpoint_Episode"],as_index=False).mean(numeric_only=True).to_string(index=False),"",
           "## 9. Reliability-requirement sensitivity vs checkpoint","",
           behavior[[c for c in behavior.columns if c in ("Actor_Mode","Checkpoint_Episode","Mean_Expected_Rho_R09","Mean_Expected_Rho_R099","Mean_Expected_Rho_R0999","Mean_Expected_Rho_R09999","Mean_Delta_Expected_Rho_HighMinusLow","Mean_Low_Rho_Mass_R09","Mean_Low_Rho_Mass_R09999")]].groupby(["Actor_Mode","Checkpoint_Episode"],as_index=False).mean(numeric_only=True).to_string(index=False),"",
           "## 10. Selected-rho evolution","",
           rho.groupby(["Actor_Mode","Checkpoint_Episode","Reliability_Requirement"],as_index=False).Selected_Rho_Mean.mean().to_string(index=False),"",
           "## 11. Action concentration","",
           concentration.groupby(["Actor_Mode","Checkpoint_Episode"],as_index=False)[["Unique_Actions_Selected","Top1_Action_Fraction","Top3_Action_Fraction","Top5_Action_Fraction"]].mean().to_string(index=False),"",
           "## 12. Server usage","",
           concentration.groupby(["Actor_Mode","Checkpoint_Episode"],as_index=False)[[c for c in concentration.columns if c.startswith("Server_")]].mean(numeric_only=True).to_string(index=False),"",
           "## 13. Trial-to-trial variability","",
           convergence[[c for c in convergence.columns if c in ("Actor_Mode","Checkpoint_Episode","Mean_Overall_RSR","Std_Overall_RSR","Mean_R0999_RSR","Std_R0999_RSR","Mean_R09999_RSR","Std_R09999_RSR","Mean_Selected_Rho","Std_Selected_Rho")]].to_string(index=False),"",
           "## 14. Episode 80–100 late-change analysis","",
           convergence[[c for c in convergence.columns if c in ("Actor_Mode","Metric","Early_Change_40_minus_0","Late_Change_100_minus_80","Absolute_Late_Change")]].to_string(index=False) if "Metric" in convergence.columns else "Late changes are recorded in aggregate_convergence.csv.","",
           "## 15. Training-length assessment","",
           "Assessment is based on the two paired trials and is diagnostic rather than a statistical claim. Inspect entropy, rho sensitivity, high-tier RSR, reliability-tier rho coupling, and the early/late changes above. The classification below is a qualitative interpretation of this run, not a rule embedded in code.","",
           "## 16. Recommendation for next step","",
           "The next step should follow the observed rho sensitivity and late-change trends. If behavior has stabilized while rho sensitivity and reliability-tier coupling remain negligible, investigate reward/credit assignment or actor expressiveness before simply extending training. This two-trial run is not a final 10-seed result.",""]
    (output_dir / "CONVERGENCE_ANALYSIS.md").write_text("\n".join(lines),encoding="utf-8")


def run_convergence_diagnostic(num_trials=2, checkpoints=DEFAULT_CHECKPOINTS, eval_episodes=3, tasks=200,
                               num_probes=50, master_seed=DEFAULT_MASTER_SEED, bootstrap_samples=500,
                               output_dir="diagnostics/ppo_convergence/checkpoint_100ep_2trial", project_root=None):
    """Run paired continuous-training convergence diagnostics and write outputs."""
    root = Path(project_root or Path(__file__).resolve().parents[1])
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = parse_checkpoints(checkpoints)
    if params.num_states != 35 or params.num_actions != 28:
        raise ValueError("convergence diagnostic expects formal 35-state/28-action setup")
    plan = generate_seed_plan(master_seed, num_trials)
    plan.to_csv(output / "seed_plan.csv", index=False)
    action_pairs, true_rho = build_pair_correlations()
    true_rho = np.asarray(true_rho, dtype=float)
    all_overall, all_tier, all_behavior, all_entropy, all_concentration, all_rho, curves = [], [], [], [], [], [], []
    start = time.perf_counter()
    for _, trial in plan.iterrows():
        trial_id = int(trial.Trial_ID)
        for actor_mode in ("flat", "pair_scoring"):
            run_dir = output / "runs" / f"trial_{trial_id:03d}" / actor_mode
            run_dir.mkdir(parents=True, exist_ok=True)
            agent = _train_agent(actor_mode, trial, true_rho)
            with scoped_environment_seeds(trial["Train_Arrival_Seed"], trial["Train_Spatial_Seed"]):
                train_loop = MainLoop(agent, 0, int(tasks), params.num_states, params.num_actions)
                for checkpoint in checkpoints:
                    advance_training_loop(train_loop, [checkpoint])
                    ep_dir = run_dir / "checkpoints" / f"ep_{checkpoint:03d}"
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    result = _evaluate_checkpoint(actor_mode, agent, trial, checkpoint, eval_episodes, tasks, true_rho, action_pairs, num_probes)
                    result["requirement"].to_csv(ep_dir / "requirement_sensitivity.csv", index=False)
                    if result["rho"] is not None:
                        result["rho"].to_csv(ep_dir / "rho_counterfactual.csv", index=False)
                    pd.DataFrame([result["overall"]]).to_csv(ep_dir / "evaluation_summary.csv", index=False)
                    result["tiers"].to_csv(ep_dir / "evaluation_by_requirement.csv", index=False)
                    result["task_frame"].to_csv(ep_dir / "evaluation_task_assignments.csv", index=False)
                    summary_payload = {"Trial_ID": trial_id, "Actor_Mode": actor_mode, "Checkpoint_Episode": checkpoint,
                                      "training_episodes_completed": train_loop.this_episode,
                                      "continuation_assertion": train_loop.this_episode == int(checkpoint),
                                      **result["evaluation_integrity"]}
                    _write_json(ep_dir / "checkpoint_summary.json", summary_payload)
                    pd.DataFrame([summary_payload]).to_csv(ep_dir / "checkpoint_summary.csv", index=False)
                    all_overall.append(_checkpoint_rows(trial_id, actor_mode, checkpoint, result))
                    tiers = result["tiers"].copy(); tiers.insert(0, "Checkpoint_Episode", checkpoint); tiers.insert(0, "Actor_Mode", actor_mode); tiers.insert(0, "Trial_ID", trial_id); all_tier.extend(tiers.to_dict("records"))
                    all_behavior.append(_behavior_row(trial_id, actor_mode, checkpoint, result))
                    entropy = result["entropy"].copy(); entropy.insert(0, "Checkpoint_Episode", checkpoint); entropy.insert(0, "Actor_Mode", actor_mode); entropy.insert(0, "Trial_ID", trial_id); all_entropy.extend(entropy.to_dict("records"))
                    concentration = {"Trial_ID": trial_id, "Actor_Mode": actor_mode, "Checkpoint_Episode": checkpoint, **result["concentration"], **result["usage"]}; all_concentration.append(concentration)
                    all_rho.append({"Trial_ID": trial_id, "Actor_Mode": actor_mode, "Checkpoint_Episode": checkpoint,
                                    "Reliability_Requirement": np.nan, **result["rho_summary"]})
                    for requirement, summary in result["rho_by_requirement"].items():
                        all_rho.append({"Trial_ID": trial_id, "Actor_Mode": actor_mode, "Checkpoint_Episode": checkpoint,
                                        "Reliability_Requirement": requirement, **summary})
                curve = _training_curve(train_loop); curve.insert(0, "Actor_Mode", actor_mode); curve.insert(0, "Trial_ID", trial_id); curves.append(curve); curve.to_csv(run_dir / "training_curve.csv", index=False)
    overall_df = pd.DataFrame(all_overall); tier_df = pd.DataFrame(all_tier); behavior_df = pd.DataFrame(all_behavior); entropy_df = pd.DataFrame(all_entropy); concentration_df = pd.DataFrame(all_concentration); rho_df = pd.DataFrame(all_rho)
    aggregate_dir = output / "aggregate"; aggregate_dir.mkdir(exist_ok=True)
    overall_df.to_csv(aggregate_dir / "checkpoint_overall.csv", index=False); tier_df.to_csv(aggregate_dir / "checkpoint_by_requirement.csv", index=False); behavior_df.to_csv(aggregate_dir / "checkpoint_behavior.csv", index=False); entropy_df.to_csv(aggregate_dir / "checkpoint_entropy.csv", index=False); concentration_df.to_csv(aggregate_dir / "checkpoint_action_concentration.csv", index=False); rho_df.to_csv(aggregate_dir / "checkpoint_selected_rho.csv", index=False)
    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(); curves_df.to_csv(aggregate_dir / "aggregate_training_curve.csv", index=False)
    convergence_rows = []
    fields = {"Overall_RSR": "Reliability_Satisfaction_Rate", "Delay": "Mean_Task_Delay", "Reward": "Mean_Reward", "Selected_Rho": "Mean_Selected_Rho"}
    for (mode, checkpoint), group in overall_df.groupby(["Actor_Mode", "Checkpoint_Episode"]):
        row = {"Actor_Mode": mode, "Checkpoint_Episode": checkpoint}
        for label, field in fields.items(): row[f"Mean_{label}"] = group[field].mean(); row[f"Std_{label}"] = group[field].std(ddof=1)
        tier_high = tier_df[(tier_df.Actor_Mode == mode) & (tier_df.Checkpoint_Episode == checkpoint)]
        for req, label in ((.999, "R0999_RSR"), (.9999, "R09999_RSR")):
            vals = tier_high[np.isclose(tier_high.Reliability_Requirement, req)]["Reliability_Satisfaction_Rate"]; row[f"Mean_{label}"] = vals.mean(); row[f"Std_{label}"] = vals.std(ddof=1)
        ent = entropy_df[(entropy_df.Actor_Mode == mode) & (entropy_df.Checkpoint_Episode == checkpoint) & np.isclose(entropy_df.Reliability_Requirement, .9999)]["Mean_Policy_Entropy"]; row["Mean_Entropy_R09999"] = ent.mean(); row["Std_Entropy_R09999"] = ent.std(ddof=1)
        bh = behavior_df[(behavior_df.Actor_Mode == mode) & (behavior_df.Checkpoint_Episode == checkpoint)]
        for column in ("Mean_Delta_Expected_Rho_HighMinusLow", "Mean_TV_True_vs_ZeroRho", "Mean_JS_True_vs_ZeroRho", "Mean_TV_True_vs_ShuffledRho", "Mean_JS_True_vs_ShuffledRho"):
            row[column] = pd.to_numeric(bh[column], errors="coerce").mean() if column in bh else np.nan
        convergence_rows.append(row)
    convergence_df = pd.DataFrame(convergence_rows)
    for mode in convergence_df.Actor_Mode.unique() if not convergence_df.empty else []:
        mode_rows = convergence_df[convergence_df.Actor_Mode == mode].set_index("Checkpoint_Episode")
        for metric in ("Overall_RSR", "Delay", "Reward", "Selected_Rho", "R0999_RSR", "R09999_RSR", "Entropy_R09999", "DeltaExpectedRho_HighMinusLow", "TV_True_vs_ZeroRho", "JS_True_vs_ZeroRho", "TV_True_vs_ShuffledRho", "JS_True_vs_ShuffledRho"):
            field = f"Mean_{metric}"
            if field not in convergence_df: continue
            early = float(mode_rows[field].get(40, np.nan) - mode_rows[field].get(0, np.nan))
            late = float(mode_rows[field].get(100, np.nan) - mode_rows[field].get(80, np.nan))
            convergence_df.loc[convergence_df.Actor_Mode == mode, f"Early_Change_{metric}_40_minus_0"] = early
            convergence_df.loc[convergence_df.Actor_Mode == mode, f"Late_Change_{metric}_100_minus_80"] = late
            convergence_df.loc[convergence_df.Actor_Mode == mode, f"Absolute_Late_Change_{metric}"] = abs(late)
    convergence_df.to_csv(aggregate_dir / "aggregate_convergence.csv", index=False)
    metadata = {"git_commit": _git_commit(root), "timestamp": pd.Timestamp.utcnow().isoformat(), "master_seed": int(master_seed), "num_trials": int(num_trials), "checkpoints": checkpoints, "eval_episodes": int(eval_episodes), "tasks_per_episode": int(tasks), "num_probes": int(num_probes), "bootstrap_samples": int(bootstrap_samples), "num_states": int(params.num_states), "num_actions": int(params.num_actions), "max_action_entropy": float(math.log(params.num_actions)), "seed_columns": list(SEED_COLUMNS), "server_info_sha256": sha256_file(root / "data/server_info.xlsx"), "task_parameters_sha256": sha256_file(root / "data/task_parameters.xlsx"), "successful_runs": int(len(plan) * 2), "failed_runs": 0, "elapsed_seconds": time.perf_counter() - start}
    _write_json(output / "metadata.json", metadata)
    result = {"output_dir": str(output), "seed_plan": plan, "num_trials": int(num_trials), "eval_episodes": int(eval_episodes), "tasks": int(tasks), "num_probes": int(num_probes), "checkpoint_overall": overall_df, "checkpoint_by_requirement": tier_df, "checkpoint_behavior": behavior_df, "checkpoint_entropy": entropy_df, "checkpoint_action_concentration": concentration_df, "checkpoint_selected_rho": rho_df, "aggregate_convergence": convergence_df, "metadata": metadata}
    _write_analysis_report(output, result, checkpoints)
    return result
