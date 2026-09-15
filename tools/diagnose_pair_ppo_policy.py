"""Run short, reproducible behavior diagnostics for Pair or Flat PPO.

This tool runs MainLoop directly and writes only to the requested diagnostics
output directory. It never calls Project_main.run_simulation(), so the formal
results workbook is not overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.params import params
from core.main_loop import MainLoop
from Project_main import build_pair_correlations
from tools.pair_policy_diagnostics import (
    DiagnosticPPOAgent,
    PAIR_DIAGNOSTIC_TORCH_SEED,
    RELIABILITY_REQUIREMENT_TIERS,
    action_pairs_for_server_count,
    performance_summary,
    requirement_sensitivity_rows,
    rho_counterfactual_rows,
    sample_probe_indices,
    summarize_behavior,
)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _make_pair_context(actor_mode):
    # Both modes are evaluated against the same formal true rho vector. Flat
    # PPO simply does not receive it as an actor input.
    action_pairs, pair_correlations = build_pair_correlations()
    return action_pairs, np.asarray(pair_correlations, dtype=float)


def _build_agent(actor_mode, pair_correlations):
    torch.manual_seed(PAIR_DIAGNOSTIC_TORCH_SEED)
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
        "minibatch_seed": params.PPO_MINIBATCH_SEED,
        "actor_mode": actor_mode,
    }
    if actor_mode == "pair_scoring":
        kwargs.update(num_servers=params.serverNo, pair_correlations=pair_correlations)
    return DiagnosticPPOAgent(**kwargs)


def _metadata(actor_mode, episodes, tasks, num_probes, rho):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "actor_mode": actor_mode,
        "episodes": episodes,
        "tasks": tasks,
        "num_probes": num_probes,
        "num_states": params.num_states,
        "num_actions": params.num_actions,
        "spatial_enabled": bool(params.SPATIAL_RISK_ENABLED),
        "ell": float(params.SPATIAL_CORRELATION_LENGTH_KM),
        "beta": float(params.SPATIAL_RISK_BETA_P),
        "lambda_ref": float(params.LAMBDA_REF),
        "torch_diagnostic_seed": PAIR_DIAGNOSTIC_TORCH_SEED,
        "arrival_seed": int(params.TASK_ARRIVAL_SEED),
        "spatial_seed": int(params.SPATIAL_RISK_SEED),
        "ppo_minibatch_seed": int(params.PPO_MINIBATCH_SEED),
        "rho_min": None if rho is None else float(np.min(rho)),
        "rho_max": None if rho is None else float(np.max(rho)),
        "rho_mean": None if rho is None else float(np.mean(rho)),
        "rho_median": None if rho is None else float(np.median(rho)),
    }


def run_diagnostic(actor_mode, episodes, tasks, num_probes, output_dir):
    if actor_mode not in {"flat", "pair_scoring"}:
        raise ValueError("actor_mode must be flat or pair_scoring")
    if episodes <= 0 or tasks <= 0 or num_probes <= 0:
        raise ValueError("episodes, tasks, and num_probes must be positive")
    if params.num_states != 35 or params.num_actions != 28:
        raise ValueError("diagnostic expects the formal 35-state/28-action PPO setup")

    action_pairs, true_rho = _make_pair_context(actor_mode)
    agent = _build_agent(actor_mode, true_rho)
    loop = MainLoop(agent, episodes, tasks, params.num_states, params.num_actions)
    loop.EP()
    probe_indices = sample_probe_indices(len(agent.observation_archive), num_probes)
    probes = [agent.observation_archive[index] for index in probe_indices]
    if not probes:
        raise RuntimeError("MainLoop produced no captured PPO observations")

    # The model samples with policy_old. This is intentionally also the policy
    # used for behavior diagnostics; train_step synchronizes both policies.
    behavior_policy = agent.policy_old
    if actor_mode == "pair_scoring":
        requirement_df = requirement_sensitivity_rows(
            behavior_policy, probes, true_rho, action_pairs
        )
        rho_df = rho_counterfactual_rows(
            behavior_policy, probes, true_rho, action_pairs
        )
    else:
        # Flat actor has no rho input, so requirement sensitivity still uses
        # its policy but rho counterfactual is explicitly N/A.
        requirement_df = requirement_sensitivity_rows(
            behavior_policy, probes, true_rho, action_pairs
        )
        rho_df = None
    summary = summarize_behavior(actor_mode, requirement_df, rho_df)
    overall_metrics, tier_metrics = performance_summary(
        loop.task_Assignments_info,
        true_rho,
    )
    summary.update({f"Overall_{key}": value for key, value in overall_metrics.items()})

    os.makedirs(output_dir, exist_ok=True)
    prefix = "pair" if actor_mode == "pair_scoring" else "flat"
    requirement_path = os.path.join(
        output_dir, f"{prefix}_requirement_sensitivity.csv"
    )
    summary_path = os.path.join(output_dir, f"{prefix}_summary.csv")
    metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")
    performance_path = os.path.join(output_dir, f"{prefix}_performance_by_requirement.csv")
    requirement_df.to_csv(requirement_path, index=False)
    if rho_df is not None:
        rho_path = os.path.join(output_dir, f"{prefix}_rho_counterfactual.csv")
        rho_df.to_csv(rho_path, index=False)
    else:
        rho_path = None
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    tier_metrics.to_csv(performance_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(_metadata(actor_mode, episodes, tasks, len(probes), true_rho), handle, indent=2)

    print(f"actor mode: {actor_mode}")
    print(f"captured observations: {len(agent.observation_archive)}")
    print(f"probe count: {len(probes)}")
    print("mean expected true rho by tier:")
    for requirement in RELIABILITY_REQUIREMENT_TIERS:
        subset = requirement_df[
            np.isclose(requirement_df["Reliability_Requirement"], requirement)
        ]
        print(f"  {requirement}: {subset['Expected_True_Rho'].mean():.8f}")
    if rho_df is None:
        print("rho counterfactual: N/A for flat actor")
    else:
        zero = rho_df[rho_df["Rho_Scenario"].eq("zero")]
        print(
            "rho counterfactual true-vs-zero: "
            f"mean TV={zero['TV_From_True'].mean():.8f}, "
            f"mean JS={zero['JS_From_True'].mean():.8f}, "
            f"argmax change rate={zero['Argmax_Changed_From_True'].astype(bool).mean():.8f}"
        )
    print(f"requirement output: {requirement_path}")
    print(f"rho output: {rho_path or 'N/A'}")
    print(f"summary output: {summary_path}")
    print(f"metadata output: {metadata_path}")
    return {
        "requirement_path": requirement_path,
        "rho_path": rho_path,
        "summary_path": summary_path,
        "metadata_path": metadata_path,
        "performance_path": performance_path,
        "probe_count": len(probes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-mode", choices=("flat", "pair_scoring"), default="pair_scoring")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--num-probes", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "diagnostics", "pair_policy_behavior"),
    )
    args = parser.parse_args()
    run_diagnostic(args.actor_mode, args.episodes, args.tasks, args.num_probes, args.output_dir)


if __name__ == "__main__":
    main()
