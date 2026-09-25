"""Train/evaluate the action-conditioned Q sidecar independently of legacy PPO.

Training consumes only online Masked PPO rollouts. Existing long-horizon
counterfactual CSVs are read only by analyze_action_value_critic.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agents.action_value_masked_ppo_agent import ActionValueMaskedPairPPOAgent
from config.action_value_critic import ActionValueCriticConfig
from config.params import params
from Project_main import build_pair_correlations
from diagnostics.run_masked_pair_ppo_10seed import (
    OUT as PPO_OUT, action_seed, evaluate as evaluate_masked, formal_spec, train as train_masked,
)
from tools.paired_ppo_experiment import _agent_kwargs, scoped_environment_seeds, sha256_file

OUT = ROOT / "diagnostics/results/action_value_critic"


def _assert_numeric_finite(frame, label):
    values = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if values.size and not np.isfinite(values).all():
        raise RuntimeError(f"{label} contains NaN/Inf")


def _assert_model_equal(current, reference_path, label):
    expected = torch.load(reference_path, map_location="cpu", weights_only=True)
    current_state = current.state_dict()
    if current_state.keys() != expected.keys():
        raise RuntimeError(f"{label} checkpoint keys differ from legacy Masked PPO")
    for key in current_state:
        if not torch.equal(current_state[key].detach().cpu(), expected[key].detach().cpu()):
            raise RuntimeError(f"Q sidecar changed legacy {label} parameter {key}")


def compare_legacy_regression(agent, curve, metrics, run_dir, trial_id):
    """Compare same-seed outputs against existing formal Masked PPO artifacts."""
    baseline = PPO_OUT / "runs" / f"trial_{trial_id:03d}" / "masked"
    if not baseline.exists():
        raise RuntimeError(f"Missing formal baseline artifacts for trial {trial_id}: {baseline}")
    _assert_model_equal(agent.policy_old, baseline / "actor.pt", "Actor")
    _assert_model_equal(agent.value_net, baseline / "critic.pt", "V critic")
    reference_curve = pd.read_csv(baseline / "training_curve.csv")
    for column in ("episode_reward", "episode_total_delay"):
        ref_col = column if column in reference_curve else {
            "episode_reward": "Episode_Reward", "episode_total_delay": "Episode_Total_Delay"
        }[column]
        if ref_col not in reference_curve or not np.allclose(curve[column].to_numpy(), reference_curve[ref_col].to_numpy(dtype=float), rtol=0.0, atol=1e-9):
            raise RuntimeError(f"Q sidecar changed legacy PPO training curve {column}")
    ref_metrics = pd.read_csv(baseline / "training_episode_metrics.csv")
    comparable = [c for c in metrics.columns if c in ref_metrics.columns and c != "episode"]
    for col in comparable:
        a, b = metrics[col].to_numpy(dtype=float), ref_metrics[col].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=0.0, atol=1e-10, equal_nan=True):
            raise RuntimeError(f"Q sidecar changed legacy training metric {col}")
    # Existing action-by-episode summaries do not preserve the original 60k
    # action sequence; the seeded selector regression is covered unit-by-unit.
    return {
        "actor_weights_exact": True,
        "value_weights_exact": True,
        "training_reward_and_delay_exact": True,
        "training_episode_metrics_exact": True,
        "per_decision_training_action_sequence_reference_available": False,
    }


def calibration_table(agent):
    predicted = np.concatenate([x["predicted_q"] for x in agent.q_calibration_samples])
    target = np.concatenate([x["bellman_target"] for x in agent.q_calibration_samples])
    if len(predicted) == 0 or not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise RuntimeError("Q calibration data is empty or non-finite")
    frame = pd.DataFrame({"predicted_q": predicted, "bellman_target": target})
    frame["predicted_decile"] = pd.qcut(frame.predicted_q, 10, labels=False, duplicates="drop")
    table = frame.groupby("predicted_decile", observed=True).agg(
        sample_count=("predicted_q", "size"), mean_predicted_q=("predicted_q", "mean"),
        mean_bellman_target=("bellman_target", "mean"),
        target_std=("bellman_target", "std"),
    ).reset_index()
    table["calibration_error"] = table.mean_predicted_q - table.mean_bellman_target
    return table


def _plot_training(metrics, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(metrics.episode, metrics.q_loss, color="#3569a8")
    ax.set(xlabel="Episode", ylabel="Huber loss", title="Action-value critic training loss")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "q_loss_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(metrics.episode, metrics.td_error_mean, label="Mean TD error")
    ax.plot(metrics.episode, metrics.td_error_abs_p95, label="P95 absolute TD error")
    ax.set(xlabel="Episode", ylabel="TD error", title="Action-value critic TD error")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "q_td_error.png", dpi=160); plt.close(fig)


def _write_trial_outputs(agent, run_dir):
    metrics = pd.DataFrame(agent.q_training_metrics)
    if len(metrics) != 300:
        raise RuntimeError(f"Expected 300 Q updates, got {len(metrics)}")
    _assert_numeric_finite(metrics, "Q training metrics")
    metrics.to_csv(run_dir / "q_training_metrics.csv", index=False)
    pd.DataFrame(agent.q_pair_summary()).to_csv(run_dir / "q_pair_visitation.csv", index=False)
    calibration = calibration_table(agent)
    calibration.to_csv(run_dir / "q_calibration.csv", index=False)
    # Keep the actual action trace for audit/regression. It is generated online,
    # not from held-out counterfactual data.
    pd.DataFrame(agent.selection_archive).to_csv(run_dir / "q_training_action_trace.csv", index=False)
    agent.q_critic.save(run_dir / "q_critic.pt")
    _plot_training(metrics, run_dir)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(calibration.mean_predicted_q, calibration.mean_bellman_target, "o-", label="Observed bins")
    lo = min(calibration.mean_predicted_q.min(), calibration.mean_bellman_target.min())
    hi = max(calibration.mean_predicted_q.max(), calibration.mean_bellman_target.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="Ideal")
    ax.set(xlabel="Mean predicted selected-action Q", ylabel="Mean Bellman target", title="TD calibration")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(run_dir / "q_calibration.png", dpi=160); plt.close(fig)
    return metrics, calibration


def run_trial(trial_id, *, smoke=False):
    torch.set_num_threads(1)
    meta, plan = formal_spec()
    trial_id = int(trial_id)
    trial = plan.iloc[trial_id].to_dict()
    run_dir = OUT / "runs" / f"trial_{trial_id:03d}" / "masked"
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "completed.json"
    if done.exists():
        print(f"Q trial {trial_id} already complete; skipping", flush=True)
        return

    pairs, rho = build_pair_correlations()
    pairs, rho = list(pairs), np.asarray(rho, dtype=float)
    torch.manual_seed(int(trial["Torch_Init_Seed"]))
    kwargs = _agent_kwargs("pair_scoring", rho, int(trial["PPO_Minibatch_Seed"]))
    q_config = ActionValueCriticConfig()
    agent = ActionValueMaskedPairPPOAgent(**kwargs, q_config=q_config)
    train_loop, curve = train_masked(agent, "masked", trial, run_dir, meta)
    decisions = pd.DataFrame(agent.selection_archive)
    if len(decisions) != 60_000:
        raise RuntimeError(f"Expected 60,000 online decisions, got {len(decisions)}")
    episode_metrics = pd.read_csv(run_dir / "training_episode_metrics.csv")
    regression = compare_legacy_regression(agent, curve, episode_metrics, run_dir, trial_id)
    q_metrics, calibration = _write_trial_outputs(agent, run_dir)

    eval_outputs = {}
    for label, mode in (("masked_stochastic", "stochastic"), ("masked_greedy", "greedy")):
        dest = OUT / "runs" / f"trial_{trial_id:03d}" / label
        dest.mkdir(parents=True, exist_ok=True)
        loop, assignments, decisions_eval, load = evaluate_masked(agent, label, trial, dest, pairs, rho, mode)
        ref_dir = PPO_OUT / "runs" / f"trial_{trial_id:03d}" / label
        ref_assignments = pd.read_csv(ref_dir / "evaluation_task_assignments.csv")
        ref_decisions = pd.read_csv(ref_dir / "evaluation_decisions.csv")
        if not np.array_equal(assignments.action_index, ref_assignments.action_index):
            raise RuntimeError(f"{label} actions differ from formal Masked PPO")
        if not np.array_equal(decisions_eval.action_index, ref_decisions.action_index):
            raise RuntimeError(f"{label} deployment sampling differs from formal Masked PPO")
        if not np.allclose(assignments.Task_Reward.to_numpy(), ref_assignments.Task_Reward.to_numpy(), rtol=0.0, atol=1e-10):
            raise RuntimeError(f"{label} rewards differ from formal Masked PPO")
        eval_outputs[label] = {
            "actions_exact": True, "rewards_exact": True,
            "task_count": int(len(assignments)),
            "mean_reward": float(assignments.Task_Reward.mean()),
            "mean_latency": float(assignments.Task_Delay.mean()),
        }

    run_meta = {
        "trial_id": trial_id, "seed_plan": trial, "smoke_stage": bool(smoke),
        "q_config": q_config.__dict__, "ppo_formal_reference_commit": meta["git_commit"],
        "training_samples": int(len(decisions)), "training_q_updates": int(agent.q_critic.update_count),
        "counterfactual_files_used_for_training": False,
        "training_policy_distributions_saved_at_decision_time": True,
        "regression": regression, "evaluation_regression": eval_outputs,
        "q_stability": {
            "all_metrics_finite": bool(np.isfinite(q_metrics.select_dtypes(include=[np.number]).to_numpy()).all()),
            "max_abs_q": float(max(q_metrics.q_min.abs().max(), q_metrics.q_max.abs().max(),
                                    q_metrics.target_q_min.abs().max(), q_metrics.target_q_max.abs().max())),
            "minimum_pair_visits": int(min(agent.q_pair_visits)),
            "unvisited_pairs": int((agent.q_pair_visits == 0).sum()),
            "selected_action_q_std_last_episode": float(q_metrics.q_pred_std.iloc[-1]),
        },
        "evaluation": eval_outputs,
    }
    (run_dir / "completed.json").write_text(json.dumps(run_meta, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "algorithm": agent.agent_name, "training_episodes": 300, "tasks_per_episode": 200,
        "ppo_actor_critic_unchanged": True, "q_config": q_config.__dict__,
        "discount": "gamma**delta_t", "target": "expected-SARSA under saved rollout pi_old",
        "loss": "Huber selected-action only", "target_update": "soft update after configured interval",
        "counterfactual_training": False,
    }, indent=2))
    return run_meta


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Run the first formal seed as a smoke test")
    group.add_argument("--trial-id", type=int, choices=range(10), help="Run one formal seed")
    group.add_argument("--all-trials", action="store_true", help="Run all ten formal seeds after smoke approval")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        result = run_trial(0, smoke=True)
        (OUT / "smoke_status.json").write_text(json.dumps(result, indent=2))
    elif args.trial_id is not None:
        run_trial(args.trial_id)
    else:
        smoke_status = OUT / "smoke_status.json"
        if not smoke_status.exists():
            raise SystemExit("Run and review --smoke successfully before --all-trials")
        status = json.loads(smoke_status.read_text())
        if status.get("regression", {}).get("actor_weights_exact") is not True:
            raise SystemExit("Smoke regression did not preserve the legacy Actor")
        for trial_id in range(10):
            run_trial(trial_id)


if __name__ == "__main__":
    main()
