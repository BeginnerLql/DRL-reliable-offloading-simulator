"""Paired multi-seed Flat-vs-Pair PPO experiment utilities.

This module deliberately lives outside the simulator implementation.  It
controls random streams, runs a PPO train/evaluation pair, and aggregates
trial-level results without changing the formal environment or agents.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from config.params import params
from core.main_loop import MainLoop
from Project_main import build_pair_correlations
from tools.pair_policy_diagnostics import (
    DiagnosticPPOAgent,
    RELIABILITY_REQUIREMENT_TIERS,
    TASK_ASSIGNMENT_COLUMNS,
    performance_summary,
    requirement_sensitivity_rows,
    rho_counterfactual_rows,
    sample_probe_indices,
    summarize_behavior,
)


SEED_COLUMNS = (
    "Torch_Init_Seed",
    "Train_Action_Seed",
    "Train_Arrival_Seed",
    "Train_Spatial_Seed",
    "PPO_Minibatch_Seed",
    "Eval_Arrival_Seed",
    "Eval_Spatial_Seed",
)
BOOTSTRAP_SEED = 2043
DEFAULT_MASTER_SEED = 31001


def _as_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    if pd.isna(value) if isinstance(value, (float, np.floating)) else False:
        return None
    return value


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: os.PathLike | str) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_seed_plan(master_seed: int = DEFAULT_MASTER_SEED, num_trials: int = 10) -> pd.DataFrame:
    """Generate a reproducible, paired seed plan using SeedSequence."""
    if isinstance(num_trials, bool) or int(num_trials) <= 0:
        raise ValueError("num_trials must be a positive integer")
    root = np.random.SeedSequence(int(master_seed))
    rows = []
    for trial_id, child in enumerate(root.spawn(int(num_trials))):
        values = child.generate_state(len(SEED_COLUMNS), dtype=np.uint32)
        row = {"Trial_ID": trial_id}
        row.update({name: int(value) for name, value in zip(SEED_COLUMNS, values)})
        if row["Train_Arrival_Seed"] == row["Eval_Arrival_Seed"]:
            raise RuntimeError("SeedSequence produced identical train/eval arrival seeds")
        if row["Train_Spatial_Seed"] == row["Eval_Spatial_Seed"]:
            raise RuntimeError("SeedSequence produced identical train/eval spatial seeds")
        rows.append(row)
    return pd.DataFrame(rows, columns=["Trial_ID", *SEED_COLUMNS])


# Alias with a descriptive name for callers that prefer plan terminology.
build_seed_plan = generate_seed_plan


@contextmanager
def scoped_environment_seeds(arrival_seed: int, spatial_seed: int):
    """Temporarily override MainLoop's class-level environment seed inputs."""
    old_arrival = params.TASK_ARRIVAL_SEED
    old_spatial = params.SPATIAL_RISK_SEED
    params.TASK_ARRIVAL_SEED = int(arrival_seed)
    params.SPATIAL_RISK_SEED = int(spatial_seed)
    try:
        yield
    finally:
        params.TASK_ARRIVAL_SEED = old_arrival
        params.SPATIAL_RISK_SEED = old_spatial


def _parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def _state_dict_snapshot(agent) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in (
            ("policy_net", agent.policy_net),
            ("policy_old", agent.policy_old),
            ("value_net", agent.value_net),
        )
    }


def assert_state_dict_unchanged(before, agent) -> None:
    after = _state_dict_snapshot(agent)
    for module_name, tensors in before.items():
        for key, old_value in tensors.items():
            new_value = after[module_name][key]
            if not torch.equal(old_value, new_value):
                raise RuntimeError(f"frozen evaluation changed {module_name}.{key}")


class FrozenEvaluationPPOAgent(DiagnosticPPOAgent):
    """A diagnostic PPO agent that greedily evaluates without optimizer steps."""

    def __init__(self, trained_agent):
        # Deep-copying the complete agent preserves the exact actor/critic
        # architecture and weights, including the pair scorer's rho buffer.
        self.__dict__ = copy.deepcopy(trained_agent.__dict__)
        self.observation_archive = []
        self.clear_rollout()
        self._frozen_evaluation = True

    @classmethod
    def from_trained(cls, trained_agent):
        return cls(trained_agent)

    def select_action(self, state, epsilon=0.0, use_softmax=False, temperature=1.5):
        state_tensor = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_old(state_tensor)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Frozen evaluation encountered non-finite policy logits")
        return int(torch.argmax(logits, dim=-1).item())

    def train_step(self):
        """Validate evaluation bookkeeping and clear it; never optimize."""
        count = len(self.states)
        if count == 0:
            if self.pending_task_rewards:
                self.clear_rollout()
                raise RuntimeError("evaluation rollout has pending rewards without transitions")
            return
        lengths = (
            len(self.actions), len(self.rewards), len(self.next_states),
            len(self.dones), len(self.old_log_probs), len(self.delta_times),
            len(self.task_ids),
        )
        if any(length != count for length in lengths):
            self.clear_rollout()
            raise RuntimeError("evaluation rollout buffers have inconsistent lengths")
        if self.pending_task_rewards or any(reward is None for reward in self.rewards):
            self.clear_rollout()
            raise RuntimeError("evaluation rollout has unresolved rewards")
        if not any(self.dones):
            self.clear_rollout()
            raise RuntimeError("evaluation rollout is missing its terminal transition")
        self.clear_rollout()


def _agent_kwargs(actor_mode: str, pair_correlations: np.ndarray, minibatch_seed: int):
    kwargs = {
        "num_states": params.num_states,
        "num_actions": params.num_actions,
        "hidden_layers": params.hidden_layers_ppo,
        "device": "cpu",
        "gamma": params.gamma_ppo,
        "actor_lr": params.actor_lr_ppo,
        "critic_lr": params.critic_lr_ppo,
        "clip_eps": params.clip_eps_ppo,
        "k_epochs": params.k_epochs_ppo,
        "batch_size": params.batch_size_ppo,
        "entropy_coef": params.entropy_coef_ppo,
        "reward_scale": params.reward_scale_ppo,
        "gae_lambda": params.gae_lambda_ppo,
        "value_loss_coef": params.value_loss_coef_ppo,
        "max_grad_norm": params.max_grad_norm_ppo,
        "activation": params.af_ppo,
        "minibatch_seed": int(minibatch_seed),
        "actor_mode": actor_mode,
    }
    if actor_mode == "pair_scoring":
        kwargs.update(num_servers=params.serverNo, pair_correlations=pair_correlations)
    return kwargs


def build_diagnostic_agent(actor_mode: str, pair_correlations: np.ndarray, minibatch_seed: int):
    actor_mode = str(actor_mode).strip().lower()
    if actor_mode not in {"flat", "pair_scoring"}:
        raise ValueError("actor_mode must be flat or pair_scoring")
    return DiagnosticPPOAgent(**_agent_kwargs(actor_mode, pair_correlations, minibatch_seed))


def _training_curve(loop: MainLoop) -> pd.DataFrame:
    rows = []
    for index, entry in enumerate(loop.log_data):
        episode, rolling_reward, episode_reward, rolling_delay = entry
        episode_delay = loop.ep_delay_list[index] if index < len(loop.ep_delay_list) else np.nan
        rows.append({
            "Episode": int(episode),
            "Episode_Reward": float(episode_reward),
            "Rolling_Avg_Reward": float(rolling_reward),
            "Episode_Delay": float(episode_delay),
            "Rolling_Avg_Delay": float(rolling_delay),
        })
    return pd.DataFrame(rows, columns=[
        "Episode", "Episode_Reward", "Rolling_Avg_Reward", "Episode_Delay", "Rolling_Avg_Delay"
    ])


def _run_training(actor_mode, trial, train_episodes, tasks, pair_correlations):
    torch.manual_seed(int(trial["Torch_Init_Seed"]))
    agent = build_diagnostic_agent(actor_mode, pair_correlations, trial["PPO_Minibatch_Seed"])
    # Separate the network initialization stream from action sampling.
    torch.manual_seed(int(trial["Train_Action_Seed"]))
    with scoped_environment_seeds(trial["Train_Arrival_Seed"], trial["Train_Spatial_Seed"]):
        loop = MainLoop(agent, int(train_episodes), int(tasks), params.num_states, params.num_actions)
        loop.EP()
    return agent, loop, _training_curve(loop)


def _run_evaluation(actor_mode, trained_agent, trial, eval_episodes, tasks, pair_correlations):
    frozen = FrozenEvaluationPPOAgent.from_trained(trained_agent)
    before_trained = _state_dict_snapshot(trained_agent)
    before_frozen = _state_dict_snapshot(frozen)
    with scoped_environment_seeds(trial["Eval_Arrival_Seed"], trial["Eval_Spatial_Seed"]):
        loop = MainLoop(frozen, int(eval_episodes), int(tasks), params.num_states, params.num_actions)
        loop.EP()
    assert_state_dict_unchanged(before_trained, trained_agent)
    assert_state_dict_unchanged(before_frozen, frozen)
    expected_rows = int(eval_episodes) * int(tasks)
    if len(loop.task_Assignments_info) != expected_rows:
        raise RuntimeError(
            f"evaluation produced {len(loop.task_Assignments_info)} task rows; expected {expected_rows}"
        )
    return frozen, loop


def _write_json(path: Path, payload: Mapping):
    path.write_text(json.dumps(_as_json_value(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _metric_value(row: Mapping, key: str):
    value = row.get(key, np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _run_level_record(trial_id, actor_mode, overall, behavior, status="success", failure_message=""):
    fields = {
        "Trial_ID": int(trial_id),
        "Actor_Mode": actor_mode,
        "Overall_Reliability_Satisfaction_Rate": _metric_value(overall, "Reliability_Satisfaction_Rate"),
        "Overall_Mean_Task_Delay": _metric_value(overall, "Mean_Task_Delay"),
        "Overall_Mean_Reward": _metric_value(overall, "Mean_Reward"),
        "Overall_Mean_Selected_Rho": _metric_value(overall, "Mean_Selected_Rho"),
        "Mean_Delta_Expected_Rho_HighMinusLow": _metric_value(behavior, "Mean_Delta_Expected_Rho_HighMinusLow"),
        "Mean_TV_R09999_vs_R09": _metric_value(behavior, "Mean_TV_R09999_vs_R09"),
        "Mean_JS_R09999_vs_R09": _metric_value(behavior, "Mean_JS_R09999_vs_R09"),
        "Mean_TV_True_vs_ZeroRho": _metric_value(behavior, "Mean_TV_True_vs_ZeroRho"),
        "Mean_JS_True_vs_ZeroRho": _metric_value(behavior, "Mean_JS_True_vs_ZeroRho"),
        "Mean_TV_True_vs_ShuffledRho": _metric_value(behavior, "Mean_TV_True_vs_ShuffledRho"),
        "Mean_JS_True_vs_ShuffledRho": _metric_value(behavior, "Mean_JS_True_vs_ShuffledRho"),
        "Status": status,
        "Failure_Message": failure_message,
    }
    return fields


def bootstrap_mean_ci(values: Iterable[float], samples: int = 10000, seed: int = BOOTSTRAP_SEED):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    if isinstance(samples, bool) or int(samples) <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_metric(values: Iterable[float], bootstrap_samples: int = 10000,
                     bootstrap_seed: int = BOOTSTRAP_SEED, bootstrap: bool = False):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"N": 0, "Mean": np.nan, "Std": np.nan, "Median": np.nan,
                "Min": np.nan, "Max": np.nan, "SEM": np.nan,
                "Mean_95CI_Low": np.nan, "Mean_95CI_High": np.nan,
                "d_z": np.nan, "Fraction_Delta_Positive": np.nan,
                "Fraction_Delta_Negative": np.nan}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n else np.nan
    if bootstrap:
        low, high = bootstrap_mean_ci(values, bootstrap_samples, bootstrap_seed)
    else:
        low, high = mean - 1.96 * sem, mean + 1.96 * sem
    return {
        "N": n,
        "Mean": mean,
        "Std": std,
        "Median": float(np.median(values)),
        "Min": float(values.min()),
        "Max": float(values.max()),
        "SEM": sem,
        "Mean_95CI_Low": float(low),
        "Mean_95CI_High": float(high),
        "d_z": np.nan,
        "Fraction_Delta_Positive": np.nan,
        "Fraction_Delta_Negative": np.nan,
    }


def aggregate_statistics(values: Iterable[float], bootstrap_samples: int = 10000,
                         bootstrap_seed: int = BOOTSTRAP_SEED, paired_delta: bool = False):
    values = list(values)
    result = summarize_metric(values, bootstrap_samples, bootstrap_seed, bootstrap=paired_delta)
    if paired_delta:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        result["d_z"] = float(finite.mean() / finite.std(ddof=1)) if len(finite) > 1 and finite.std(ddof=1) > 1e-12 else np.nan
        result["Fraction_Delta_Positive"] = float(np.mean(finite > 0.0)) if len(finite) else np.nan
        result["Fraction_Delta_Negative"] = float(np.mean(finite < 0.0)) if len(finite) else np.nan
    return result


def build_paired_differences(run_level: pd.DataFrame, tier_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trial_id in sorted(pd.to_numeric(run_level["Trial_ID"], errors="coerce").dropna().astype(int).unique()):
        flat = run_level[(run_level.Trial_ID == trial_id) & (run_level.Actor_Mode == "flat")]
        pair = run_level[(run_level.Trial_ID == trial_id) & (run_level.Actor_Mode == "pair_scoring")]
        flat = flat.iloc[0] if not flat.empty else None
        pair = pair.iloc[0] if not pair.empty else None
        specs = [
            ("overall", None, "Overall_Reliability_Satisfaction_Rate", "Overall_Mean_Task_Delay", "Overall_Mean_Reward", "Overall_Mean_Selected_Rho"),
        ]
        for req in RELIABILITY_REQUIREMENT_TIERS:
            specs.append(("tier", req, "Reliability_Satisfaction_Rate", "Mean_Task_Delay", "Mean_Reward", "Mean_Selected_Rho"))
        for level, requirement, *metrics in specs:
            if level == "overall":
                flat_values = [flat.get(metric, np.nan) if flat is not None else np.nan for metric in metrics]
                pair_values = [pair.get(metric, np.nan) if pair is not None else np.nan for metric in metrics]
            else:
                f = tier_level[(tier_level.Trial_ID == trial_id) & (tier_level.Actor_Mode == "flat") & np.isclose(tier_level.Reliability_Requirement, requirement)]
                p = tier_level[(tier_level.Trial_ID == trial_id) & (tier_level.Actor_Mode == "pair_scoring") & np.isclose(tier_level.Reliability_Requirement, requirement)]
                f = f.iloc[0] if not f.empty else None
                p = p.iloc[0] if not p.empty else None
                flat_values = [f.get(metric, np.nan) if f is not None else np.nan for metric in metrics]
                pair_values = [p.get(metric, np.nan) if p is not None else np.nan for metric in metrics]
            row = {"Trial_ID": int(trial_id), "Reliability_Requirement": requirement}
            for label, pv, fv in zip(("Reliability_Satisfaction_Rate", "Mean_Task_Delay", "Mean_Reward", "Mean_Selected_Rho"), pair_values, flat_values):
                try:
                    row[f"Delta_{label}"] = float(pv) - float(fv) if np.isfinite(float(pv)) and np.isfinite(float(fv)) else np.nan
                except (TypeError, ValueError):
                    row[f"Delta_{label}"] = np.nan
            row["Status"] = "success" if all(np.isfinite(x) for x in row.values() if isinstance(x, (float, np.floating))) else "incomplete"
            row["Failure_Message"] = "" if row["Status"] == "success" else "missing flat or pair result"
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_table(values_frame: pd.DataFrame, group_columns, metric_columns,
                     bootstrap_samples, bootstrap_seed, paired=False):
    rows = []
    if values_frame.empty:
        return pd.DataFrame()
    if not group_columns:
        groups = [((), values_frame)]
    else:
        groups = values_frame.groupby(group_columns, dropna=False)
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        for metric in metric_columns:
            stats = aggregate_statistics(group[metric].tolist(), bootstrap_samples, bootstrap_seed, paired_delta=paired)
            row = dict(zip(group_columns, keys))
            row["Metric"] = metric
            row.update(stats)
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_outputs(run_level: pd.DataFrame, tier_level: pd.DataFrame,
                      paired_differences: pd.DataFrame, training_curves: pd.DataFrame,
                      bootstrap_samples: int = 10000, bootstrap_seed: int = BOOTSTRAP_SEED):
    overall_metrics = [
        "Overall_Reliability_Satisfaction_Rate", "Overall_Mean_Task_Delay",
        "Overall_Mean_Reward", "Overall_Mean_Selected_Rho",
    ]
    overall = _aggregate_table(run_level[run_level.Status.eq("success")], ["Actor_Mode"], overall_metrics, bootstrap_samples, bootstrap_seed)
    if not paired_differences.empty:
        overall_delta = paired_differences[paired_differences.Reliability_Requirement.isna() if paired_differences.Reliability_Requirement.dtype != object else paired_differences.Reliability_Requirement.eq("overall")]
        if not overall_delta.empty:
            delta_cols = [column for column in overall_delta.columns if column.startswith("Delta_")]
            delta_rows = _aggregate_table(overall_delta, [], delta_cols, bootstrap_samples, bootstrap_seed, paired=True)
            if not delta_rows.empty:
                delta_rows.insert(0, "Actor_Mode", "pair_minus_flat")
                overall = pd.concat([overall, delta_rows], ignore_index=True)
    tier_metrics = ["Reliability_Satisfaction_Rate", "Mean_Task_Delay", "Mean_Reward", "Mean_Selected_Rho"]
    by_requirement = _aggregate_table(tier_level[tier_level.Status.eq("success")] if "Status" in tier_level else tier_level, ["Actor_Mode", "Reliability_Requirement"], tier_metrics, bootstrap_samples, bootstrap_seed)
    behavior_metrics = [column for column in run_level.columns if column.startswith("Mean_") and column not in overall_metrics]
    behavior = _aggregate_table(run_level[run_level.Status.eq("success")], ["Actor_Mode"], behavior_metrics, bootstrap_samples, bootstrap_seed)
    if training_curves.empty:
        training = pd.DataFrame()
    else:
        training = training_curves.groupby(["Actor_Mode", "Episode"], as_index=False).agg(
            Mean_Episodic_Reward=("Episode_Reward", "mean"), Std_Episodic_Reward=("Episode_Reward", "std"),
            Mean_Rolling_Reward=("Rolling_Avg_Reward", "mean"), Std_Rolling_Reward=("Rolling_Avg_Reward", "std"),
            Mean_Delay=("Episode_Delay", "mean"), Std_Delay=("Episode_Delay", "std"),
        )
    return overall, by_requirement, behavior, training


def _metadata_base(project_root: Path, master_seed, num_trials, train_episodes, eval_episodes, tasks, num_probes, bootstrap_samples):
    return {
        "git_commit": _git_commit(project_root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "master_seed": int(master_seed), "num_trials": int(num_trials),
        "train_episodes": int(train_episodes), "eval_episodes": int(eval_episodes),
        "tasks_per_episode": int(tasks), "num_probes": int(num_probes),
        "bootstrap_samples": int(bootstrap_samples), "bootstrap_seed": BOOTSTRAP_SEED,
        "num_states": int(params.num_states), "num_actions": int(params.num_actions),
        "lambda_ref": float(params.LAMBDA_REF), "omega": float(params.FAILURE_RATE_OMEGA),
        "beta": float(params.SPATIAL_RISK_BETA_P), "ell": float(params.SPATIAL_CORRELATION_LENGTH_KM),
        "task_arrival_rate": float(params.TASK_ARRIVAL_RATE),
        "ppo_hidden_layers": list(params.hidden_layers_ppo), "ppo_actor_lr": float(params.actor_lr_ppo),
        "ppo_critic_lr": float(params.critic_lr_ppo), "ppo_gamma": float(params.gamma_ppo),
        "ppo_gae_lambda": float(params.gae_lambda_ppo), "ppo_clip_eps": float(params.clip_eps_ppo),
        "ppo_epochs": int(params.k_epochs_ppo), "ppo_batch_size": int(params.batch_size_ppo),
        "ppo_entropy_coef": float(params.entropy_coef_ppo), "evaluation_policy": "greedy",
        "task_parameters_sha256": sha256_file(project_root / "data" / "task_parameters.xlsx"),
        "server_info_sha256": sha256_file(project_root / "data" / "server_info.xlsx"),
    }


def run_experiment(num_trials=10, train_episodes=100, eval_episodes=20, tasks=200,
                   num_probes=100, master_seed=DEFAULT_MASTER_SEED,
                   bootstrap_samples=10000, output_dir="diagnostics/paired_ppo_experiment/formal_10seed",
                   project_root=None):
    """Run paired train/freeze/evaluate experiments and write CSV/JSON outputs."""
    project_root = Path(project_root or Path(__file__).resolve().parents[1])
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if params.num_states != 35 or params.num_actions != 28:
        raise ValueError("paired experiment expects the formal 35-state/28-action setup")
    seed_plan = generate_seed_plan(master_seed, num_trials)
    seed_plan.to_csv(output_dir / "seed_plan.csv", index=False)
    action_pairs, true_rho = build_pair_correlations()
    true_rho = np.asarray(true_rho, dtype=float)
    experiment_metadata = _metadata_base(project_root, master_seed, num_trials, train_episodes, eval_episodes, tasks, num_probes, bootstrap_samples)
    experiment_metadata.update({"actor_modes": ["flat", "pair_scoring"], "true_rho_min": float(true_rho.min()), "true_rho_max": float(true_rho.max()), "true_rho_mean": float(true_rho.mean()), "true_rho_median": float(np.median(true_rho))})
    run_records, tier_records, curve_frames = [], [], []
    for _, seed_row in seed_plan.iterrows():
        trial_id = int(seed_row.Trial_ID)
        trial_dir = output_dir / "runs" / f"trial_{trial_id:03d}"
        for actor_mode in ("flat", "pair_scoring"):
            run_dir = trial_dir / actor_mode
            run_dir.mkdir(parents=True, exist_ok=True)
            status = "success"
            failure_message = ""
            training_completed = False
            evaluation_completed = False
            frozen_check_passed = False
            overall, behavior = {}, {}
            tier = pd.DataFrame()
            training_curve = pd.DataFrame()
            try:
                trained_agent, train_loop, training_curve = _run_training(actor_mode, seed_row, train_episodes, tasks, true_rho)
                training_curve.to_csv(run_dir / "training_curve.csv", index=False)
                training_completed = True
                curve = training_curve.copy(); curve.insert(0, "Actor_Mode", actor_mode); curve.insert(0, "Trial_ID", trial_id); curve_frames.append(curve)
                frozen, eval_loop = _run_evaluation(actor_mode, trained_agent, seed_row, eval_episodes, tasks, true_rho)
                evaluation_completed = True
                frozen_check_passed = True
                overall, tier = performance_summary(eval_loop.task_Assignments_info, true_rho)
                probe_indices = sample_probe_indices(len(frozen.observation_archive), num_probes)
                probes = [frozen.observation_archive[index] for index in probe_indices]
                if not probes:
                    raise RuntimeError("evaluation produced no probe states")
                requirement_df = requirement_sensitivity_rows(frozen.policy_old, probes, true_rho, action_pairs)
                rho_df = rho_counterfactual_rows(frozen.policy_old, probes, true_rho, action_pairs) if actor_mode == "pair_scoring" else None
                behavior = summarize_behavior(actor_mode, requirement_df, rho_df)
                requirement_df.to_csv(run_dir / "requirement_sensitivity.csv", index=False)
                if rho_df is not None:
                    rho_df.to_csv(run_dir / "rho_counterfactual.csv", index=False)
                pd.DataFrame([overall]).to_csv(run_dir / "evaluation_summary.csv", index=False)
                tier.to_csv(run_dir / "evaluation_by_requirement.csv", index=False)
                evaluation_rows = pd.DataFrame(eval_loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS)
                evaluation_rows.to_csv(run_dir / "evaluation_task_assignments.csv", index=False)
                tier = tier.copy(); tier.insert(0, "Actor_Mode", actor_mode); tier.insert(0, "Trial_ID", trial_id); tier["Status"] = "success"; tier["Failure_Message"] = ""
            except Exception as exc:  # preserve failed runs in aggregate tables
                status = "failed"
                failure_message = f"{type(exc).__name__}: {exc}"
                if not (run_dir / "training_curve.csv").exists():
                    training_curve.to_csv(run_dir / "training_curve.csv", index=False)
                tier = pd.DataFrame([{"Trial_ID": trial_id, "Actor_Mode": actor_mode, "Reliability_Requirement": requirement, "Count": 0, "Reliability_Satisfaction_Rate": np.nan, "Mean_Task_Delay": np.nan, "Mean_Reward": np.nan, "Mean_Selected_Rho": np.nan, "Status": "failed", "Failure_Message": failure_message} for requirement in RELIABILITY_REQUIREMENT_TIERS])
            record = _run_level_record(trial_id, actor_mode, overall, behavior, status, failure_message)
            record.update({"Actor_Parameter_Count": np.nan, "Critic_Parameter_Count": np.nan})
            if status == "success":
                record["Actor_Parameter_Count"] = _parameter_count(trained_agent.policy_net)
                record["Critic_Parameter_Count"] = _parameter_count(trained_agent.value_net)
            run_records.append(record)
            tier_records.append(tier)
            run_metadata = {
                "Trial_ID": trial_id, "Actor_Mode": actor_mode, **{column: int(seed_row[column]) for column in SEED_COLUMNS},
                "train_episodes": int(train_episodes), "eval_episodes": int(eval_episodes), "tasks_per_episode": int(tasks),
                "rho_min": float(true_rho.min()), "rho_max": float(true_rho.max()), "rho_mean": float(true_rho.mean()), "rho_median": float(np.median(true_rho)),
                "training_completed": training_completed, "evaluation_completed": evaluation_completed, "frozen_parameter_check_passed": frozen_check_passed,
                "actor_parameter_count": run_records[-1]["Actor_Parameter_Count"], "critic_parameter_count": run_records[-1]["Critic_Parameter_Count"],
                "status": status, "failure_message": failure_message,
            }
            _write_json(run_dir / "metadata.json", run_metadata)
    run_level = pd.DataFrame(run_records)
    tier_level = pd.concat(tier_records, ignore_index=True) if tier_records else pd.DataFrame()
    paired = build_paired_differences(run_level, tier_level)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    aggregate_dir = output_dir / "aggregate"; aggregate_dir.mkdir(exist_ok=True)
    run_level.to_csv(aggregate_dir / "run_level_results.csv", index=False)
    tier_level.to_csv(aggregate_dir / "tier_level_results.csv", index=False)
    paired.to_csv(aggregate_dir / "paired_differences.csv", index=False)
    overall_agg, tier_agg, behavior_agg, curve_agg = aggregate_outputs(run_level, tier_level, paired, curves, bootstrap_samples)
    overall_agg.to_csv(aggregate_dir / "aggregate_overall.csv", index=False)
    tier_agg.to_csv(aggregate_dir / "aggregate_by_requirement.csv", index=False)
    behavior_agg.to_csv(aggregate_dir / "aggregate_behavior.csv", index=False)
    curve_agg.to_csv(aggregate_dir / "aggregate_training_curve.csv", index=False)
    experiment_metadata.update({"successful_runs": int((run_level.Status == "success").sum()), "failed_runs": int((run_level.Status == "failed").sum()), "actor_parameter_counts": {mode: run_level.loc[(run_level.Actor_Mode == mode) & run_level.Status.eq("success"), "Actor_Parameter_Count"].dropna().unique().tolist() for mode in ("flat", "pair_scoring")}, "critic_parameter_counts": run_level.loc[run_level.Status.eq("success"), "Critic_Parameter_Count"].dropna().unique().tolist()})
    _write_json(output_dir / "metadata.json", experiment_metadata)
    return {"output_dir": str(output_dir), "seed_plan": seed_plan, "run_level": run_level, "tier_level": tier_level, "paired_differences": paired, "successful_runs": int((run_level.Status == "success").sum()), "failed_runs": int((run_level.Status == "failed").sum())}
