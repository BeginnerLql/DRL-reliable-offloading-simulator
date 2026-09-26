"""One-seed audit of a read-only policy-centered action-advantage learner."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import torch

from agents.action_conditioned_q_critic import ActionValueCritic
from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent
from agents.policy_centered_action_advantage import (
    PolicyCenteredActionAdvantage, policy_centered_advantage,
)
from agents.ppo_agent import PPOPairScoringPolicyNetwork
from config.action_value_critic import ActionValueCriticConfig
from config.params import params
from diagnostics.analyze_action_value_critic import _effective_mask, _optimal_set
from diagnostics.evaluate_reliability_masked_policy import ArrivalTraceLoop
from diagnostics.run_masked_pair_ppo_10seed import OUT as PPO_OUT, formal_spec, train as train_masked
from Project_main import build_pair_correlations
from tools.pair_policy_diagnostics import TASK_ASSIGNMENT_COLUMNS
from tools.paired_ppo_experiment import _agent_kwargs, scoped_environment_seeds, sha256_file

OUT = ROOT / "diagnostics/results/advantage_side_learner"
LONG = ROOT / "diagnostics/results/long_horizon_coupling"
OLD_Q = ROOT / "diagnostics/results/action_value_critic"
Q_AUDIT = ROOT / "diagnostics/results/q_credit_audit"
HORIZONS = (5, 10, 20, 50)
SMOKE_EPISODES = 8
FIXED_TARGET_COUNT = 4096
FIXED_TARGET_STEPS = 2000
FIXED_TARGET_BATCH = 256


def _digest_state_dict(module):
    digest = hashlib.sha256()
    state_dict = module.state_dict() if hasattr(module, "state_dict") else module
    for key, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _max_abs_difference(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _safe_corr(left, right, method="pearson"):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    value = (spearmanr(left, right).statistic if method == "spearman" else
             kendalltau(left, right).statistic if method == "kendall" else
             np.corrcoef(left, right)[0, 1])
    return float(value) if np.isfinite(value) else float("nan")


def _new_side_learner(rho, initialization_seed=7042026):
    return PolicyCenteredActionAdvantage(
        params.num_states, params.num_actions, params.hidden_layers_ppo,
        params.serverNo, rho, activation=params.af_ppo,
        initialization_seed=initialization_seed,
    )


class SideLearnerObserver:
    """Collect immutable PPO targets and train only after each PPO update."""
    def __init__(self, learner):
        self.learner = learner
        self.pending_payload = None
        self.payloads, self.target_rows, self.prediction_rows = [], [], []
        self.update_rows, self.centering_rows, self.wiring_rows = [], [], []

    def observe(self, payload):
        if self.pending_payload is not None:
            raise RuntimeError("Previous PPO target payload was not consumed")
        fields = (
            "task_ids", "states", "actions", "effective_masks",
            "old_policy_probabilities", "raw_gae_advantages",
            "actor_used_advantages", "value_predictions", "gae_return_targets",
        )
        n = len(payload["actions"])
        if any(len(payload[key]) != n for key in fields):
            raise RuntimeError("PPO observer payload fields have inconsistent lengths")
        masks = np.asarray(payload["effective_masks"], dtype=bool)
        actions = np.asarray(payload["actions"], dtype=int)
        probs = np.asarray(payload["old_policy_probabilities"], dtype=float)
        if not masks.any(axis=1).all() or not masks[np.arange(n), actions].all():
            raise RuntimeError("PPO target snapshot has an empty mask or unsafe action")
        if np.max(np.abs(probs[~masks]), initial=0.0) > 1e-7:
            raise RuntimeError("Saved policy probability leaked outside effective mask")
        if not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=2e-6):
            raise RuntimeError("Decision-time masked policy probabilities do not sum to one")
        if any(value.flags.writeable for value in payload.values() if isinstance(value, np.ndarray)):
            raise RuntimeError("Observer payload must consist of immutable copies")
        self.pending_payload = payload

    def consume_after_ppo_update(self):
        payload = self.pending_payload
        if payload is None:
            raise RuntimeError("PPO update completed without an observer payload")
        self.pending_payload = None
        episode = int(payload["episode"])
        raw = np.asarray(payload["raw_gae_advantages"], dtype=float)
        actor_used = np.asarray(payload["actor_used_advantages"], dtype=float)
        values = np.asarray(payload["value_predictions"], dtype=float)
        returns = np.asarray(payload["gae_return_targets"], dtype=float)
        actions = np.asarray(payload["actions"], dtype=int)
        masks = np.asarray(payload["effective_masks"], dtype=bool)
        if not self.wiring_rows:
            self.wiring_rows.append({
                "checkpoint": "initial_before_side_update",
                **self.learner.wiring_diagnostics(payload["states"][:min(32, len(actions))]),
            })
        predictions, centering = self.learner.rollout_diagnostics(payload)
        if centering["max_absolute_residual"] > 2e-6:
            raise RuntimeError("Policy-centering residual exceeds float32 tolerance")
        self.prediction_rows.extend(predictions)
        self.centering_rows.append({"episode": episode, **centering})
        for i, action in enumerate(actions):
            self.target_rows.append({
                "episode": episode, "task_id": int(payload["task_ids"][i]),
                "action_index": int(action), "raw_gae_advantage": float(raw[i]),
                "actor_used_advantage": float(actor_used[i]),
                "value_prediction": float(values[i]), "gae_return_target": float(returns[i]),
                "effective_set_size": int(masks[i].sum()),
            })
        self.payloads.append(payload)
        update = self.learner.train_rollout(payload)
        self.update_rows.append(update)


def attach_side_learner(agent, learner, observer):
    """Add a detached target tap and an after-PPO-update sidecar callback."""
    agent.advantage_target_observer = observer.observe
    original_train_step = agent.train_step

    def train_step_with_sidecar():
        result = original_train_step()
        if observer.pending_payload is not None:
            observer.consume_after_ppo_update()
        return result

    agent.train_step = train_step_with_sidecar


def _run_smoke_once(trial, rho, *, instrumented, log_path):
    torch.manual_seed(int(trial["Torch_Init_Seed"]))
    agent = ReliabilityMaskedPairPPOAgent(**_agent_kwargs(
        "pair_scoring", np.asarray(rho, dtype=float), int(trial["PPO_Minibatch_Seed"])
    ))
    observer = None
    if instrumented:
        learner = _new_side_learner(rho)
        observer = SideLearnerObserver(learner)
        attach_side_learner(agent, learner, observer)
    torch.manual_seed(int(trial["Train_Action_Seed"]))
    with scoped_environment_seeds(int(trial["Train_Arrival_Seed"]), int(trial["Train_Spatial_Seed"])):
        with log_path.open("w") as log, redirect_stdout(log):
            loop = ArrivalTraceLoop(agent, SMOKE_EPISODES, 200, params.num_states, params.num_actions)
            loop.EP()
    return agent, loop, observer


def run_regression_smoke(plan, rho):
    dest = OUT / "regression_smoke"
    dest.mkdir(parents=True, exist_ok=True)
    trial = plan.iloc[0].to_dict()
    base_agent, base_loop, _ = _run_smoke_once(
        trial, rho, instrumented=False, log_path=dest / "baseline.log"
    )
    side_agent, side_loop, observer = _run_smoke_once(
        trial, rho, instrumented=True, log_path=dest / "side_learner.log"
    )
    base_decisions = pd.DataFrame(base_agent.selection_archive).sort_values(["episode", "task_id"])
    side_decisions = pd.DataFrame(side_agent.selection_archive).sort_values(["episode", "task_id"])
    base_assign = pd.DataFrame(base_loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS).sort_values(["episode", "task_id"])
    side_assign = pd.DataFrame(side_loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS).sort_values(["episode", "task_id"])
    result = {
        "episodes": SMOKE_EPISODES, "tasks_per_episode": 200,
        "action_count": len(base_decisions),
        "action_mismatch": int(np.count_nonzero(base_decisions.action_index.to_numpy() != side_decisions.action_index.to_numpy())),
        "reward_max_abs_diff": _max_abs_difference(base_assign.Task_Reward, side_assign.Task_Reward),
        "delay_max_abs_diff": _max_abs_difference(base_assign.Task_Delay, side_assign.Task_Delay),
        "effective_mask_mismatch": int(sum(a != b for a, b in zip(base_decisions.effective_mask, side_decisions.effective_mask))),
        "actor_hash_match": _digest_state_dict(base_agent.policy_old) == _digest_state_dict(side_agent.policy_old),
        "critic_hash_match": _digest_state_dict(base_agent.value_net) == _digest_state_dict(side_agent.value_net),
        "side_updates": len(observer.update_rows),
        "side_targets_captured": int(sum(len(p["actions"]) for p in observer.payloads)),
    }
    result["passed"] = bool(
        result["action_count"] == SMOKE_EPISODES * 200
        and result["action_mismatch"] == 0
        and result["reward_max_abs_diff"] == 0
        and result["delay_max_abs_diff"] == 0
        and result["effective_mask_mismatch"] == 0
        and result["actor_hash_match"] and result["critic_hash_match"]
        and result["side_updates"] == SMOKE_EPISODES
    )
    (OUT / "regression_smoke.json").write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(f"Regression smoke failed: {result}")
    return result


def _target_phase_summary(targets):
    rows = []
    for phase, start, end in (("early", 1, 100), ("middle", 101, 200), ("late", 201, 300)):
        frame = targets[targets.episode.between(start, end)]
        raw = frame.raw_gae_advantage.to_numpy(float)
        actor = frame.actor_used_advantage.to_numpy(float)
        row = {"phase": phase, "episodes": f"{start}-{end}", "transition_count": len(frame)}
        for prefix, values in (("raw", raw), ("actor", actor)):
            row.update({f"{prefix}_{k}": v for k, v in {
                "mean": np.mean(values), "std": np.std(values), "median": np.median(values),
                "p05": np.quantile(values, .05), "p25": np.quantile(values, .25),
                "p75": np.quantile(values, .75), "p95": np.quantile(values, .95),
                "min": np.min(values), "max": np.max(values),
            }.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _write_training_csvs(observer):
    targets = pd.DataFrame(observer.target_rows)
    predictions = pd.DataFrame(observer.prediction_rows)
    updates = pd.DataFrame(observer.update_rows)
    centering = pd.DataFrame(observer.centering_rows)
    if len(targets) != 60_000 or len(predictions) != 60_000 or len(updates) != 300:
        raise RuntimeError("Training diagnostics must contain 60,000 transitions and 300 updates")
    targets.to_csv(OUT / "training_advantage_targets.csv", index=False)
    predictions.to_csv(OUT / "training_advantage_predictions.csv", index=False)
    updates.to_csv(OUT / "advantage_update_diagnostics.csv", index=False)
    centering.to_csv(OUT / "centering_checks.csv", index=False)
    if float(centering.max_absolute_residual.max()) > 2e-6:
        raise RuntimeError("Policy-centering residual failed numeric tolerance")
    phases = _target_phase_summary(targets)
    phases.to_csv(OUT / "advantage_target_phase_statistics.csv", index=False)
    return targets, predictions, updates, centering, phases


def _write_v_observation(targets):
    rows = []
    for episode, group in targets.groupby("episode", sort=True):
        value = group.value_prediction.to_numpy(float)
        target = group.gae_return_target.to_numpy(float)
        variance = float(np.var(target))
        rows.append({
            "episode": int(episode), "transition_count": len(group),
            "value_mean": float(value.mean()), "gae_return_target_mean": float(target.mean()),
            "value_target_mae": float(np.mean(np.abs(value-target))),
            "value_target_correlation": _safe_corr(value, target),
            "explained_variance": float(1-np.var(value-target)/variance) if variance > 0 else np.nan,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "v_critic_observation_only.csv", index=False)
    return frame


def _compare_full_run(agent, loop, curve, observer, trial_id):
    baseline = PPO_OUT / "runs" / f"trial_{trial_id:03d}" / "masked"
    actor_ref = torch.load(baseline / "actor.pt", map_location="cpu", weights_only=True)
    critic_ref = torch.load(baseline / "critic.pt", map_location="cpu", weights_only=True)
    actor_match = all(torch.equal(agent.policy_old.state_dict()[k].cpu(), actor_ref[k].cpu()) for k in actor_ref)
    critic_match = all(torch.equal(agent.value_net.state_dict()[k].cpu(), critic_ref[k].cpu()) for k in critic_ref)
    reference_curve = pd.read_csv(baseline / "training_curve.csv")
    reward_curve_diff = _max_abs_difference(curve.episode_reward, reference_curve.episode_reward)
    delay_curve_diff = _max_abs_difference(curve.episode_total_delay, reference_curve.episode_total_delay)
    ref_decisions = pd.read_csv(Q_AUDIT / "training_decisions.csv.gz").sort_values(["episode", "task_id"]).reset_index(drop=True)
    ref_assignments = pd.read_csv(Q_AUDIT / "training_task_assignments.csv.gz").sort_values(["episode", "task_id"]).reset_index(drop=True)
    decisions = pd.DataFrame(agent.selection_archive).sort_values(["episode", "task_id"]).reset_index(drop=True)
    assignments = pd.DataFrame(loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS).sort_values(["episode", "task_id"]).reset_index(drop=True)
    def parse_mask(value):
        parsed = json.loads(value) if isinstance(value, str) else value
        return np.asarray(parsed, dtype=bool)

    mask_mismatch = sum(
        not np.array_equal(parse_mask(current), parse_mask(reference))
        for current, reference in zip(decisions.effective_mask, ref_decisions.effective_mask)
    )
    reward_diff = _max_abs_difference(assignments.Task_Reward, ref_assignments.Task_Reward)
    delay_diff = _max_abs_difference(assignments.Task_Delay, ref_assignments.Task_Delay)
    result = {
        "trial_id": int(trial_id), "transition_count": len(decisions),
        "action_mismatch": int(np.count_nonzero(decisions.action_index.to_numpy() != ref_decisions.action_index.to_numpy())),
        "effective_mask_mismatch": int(mask_mismatch),
        "reward_max_abs_diff": reward_diff,
        "delay_max_abs_diff": delay_diff,
        "episode_reward_curve_max_abs_diff": reward_curve_diff,
        "episode_delay_curve_max_abs_diff": delay_curve_diff,
        "actor_state_dict_match": bool(actor_match), "critic_state_dict_match": bool(critic_match),
        "actor_state_dict_digest": _digest_state_dict(agent.policy_old),
        "critic_state_dict_digest": _digest_state_dict(agent.value_net),
        "baseline_actor_state_dict_digest": _digest_state_dict(actor_ref),
        "baseline_critic_state_dict_digest": _digest_state_dict(critic_ref),
        "captured_ppo_updates": len(observer.payloads),
    }
    result["actor_hash_match"] = result["actor_state_dict_match"] and result["actor_state_dict_digest"] == result["baseline_actor_state_dict_digest"]
    result["critic_hash_match"] = result["critic_state_dict_match"] and result["critic_state_dict_digest"] == result["baseline_critic_state_dict_digest"]
    result["passed"] = bool(
        result["transition_count"] == 60_000 and result["action_mismatch"] == 0
        and result["effective_mask_mismatch"] == 0 and reward_diff <= 1e-10
        and delay_diff <= 1e-10 and reward_curve_diff <= 1e-9 and delay_curve_diff <= 1e-9
        and actor_match and critic_match and len(observer.payloads) == 300
    )
    (OUT / "regression_full_run.json").write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(f"Full PPO regression mismatch: {result}")
    return result


def run_training(meta, plan, rho, smoke):
    if not smoke.get("passed"):
        raise RuntimeError("Regression smoke must pass before the 300-episode run")
    if (OUT / "experiment_status.json").exists():
        raise RuntimeError("Diagnostic run is already complete")
    trial = plan.iloc[0].to_dict()
    run_dir = OUT / "training_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(trial["Torch_Init_Seed"]))
    agent = ReliabilityMaskedPairPPOAgent(**_agent_kwargs(
        "pair_scoring", np.asarray(rho, dtype=float), int(trial["PPO_Minibatch_Seed"])
    ))
    learner = _new_side_learner(rho)
    observer = SideLearnerObserver(learner)
    attach_side_learner(agent, learner, observer)
    loop, curve = train_masked(agent, "masked", trial, run_dir, meta)
    if len(loop.task_Assignments_info) != 60_000 or len(agent.selection_archive) != 60_000:
        raise RuntimeError("Run did not complete exactly 300 × 200 tasks")
    regression = _compare_full_run(agent, loop, curve, observer, 0)
    torch.save(learner.network.state_dict(), OUT / "policy_centered_advantage.pt")
    payloads = observer.payloads
    np.savez_compressed(OUT / "training_sidecar_observations.npz",
        episode=np.concatenate([np.full(len(p["actions"]), int(p["episode"]), dtype=np.int16) for p in payloads]),
        task_ids=np.concatenate([p["task_ids"] for p in payloads]),
        states=np.concatenate([p["states"] for p in payloads]),
        actions=np.concatenate([p["actions"] for p in payloads]),
        effective_masks=np.concatenate([p["effective_masks"] for p in payloads]),
        old_policy_probabilities=np.concatenate([p["old_policy_probabilities"] for p in payloads]),
        raw_gae_advantages=np.concatenate([p["raw_gae_advantages"] for p in payloads]),
        actor_used_advantages=np.concatenate([p["actor_used_advantages"] for p in payloads]),
        value_predictions=np.concatenate([p["value_predictions"] for p in payloads]),
        gae_return_targets=np.concatenate([p["gae_return_targets"] for p in payloads]))
    targets, predictions, updates, centering, phases = _write_training_csvs(observer)
    v_rows = _write_v_observation(targets)
    initial = pd.DataFrame(observer.wiring_rows)
    wiring_states = observer.payloads[0]["states"][:32]
    wiring = pd.concat([initial, pd.DataFrame([{
        "checkpoint": "final_after_300_episodes", **learner.wiring_diagnostics(wiring_states)
    }])], ignore_index=True)
    wiring.to_csv(OUT / "pair_feature_wiring_checks.csv", index=False)
    if not wiring.permutation_pass.all() or not wiring.pair_mapping_matches_combinations.all() or not wiring.pair_feature_gradient_nonzero.all():
        raise RuntimeError("Pair-feature wiring sanity check failed")
    return {
        "meta": meta, "trial": trial, "agent": agent, "learner": learner,
        "observer": observer, "regression": regression, "targets": targets,
        "predictions": predictions, "updates": updates, "centering": centering,
        "phases": phases, "v_rows": v_rows, "wiring": wiring,
    }


def fit_fixed_targets(payloads, rho):
    """Fit a fresh network on a deterministic 4,096-transition frozen sample."""
    states = np.concatenate([p["states"] for p in payloads])
    actions = np.concatenate([p["actions"] for p in payloads])
    targets = np.concatenate([p["actor_used_advantages"] for p in payloads])
    masks = np.concatenate([p["effective_masks"] for p in payloads])
    probs = np.concatenate([p["old_policy_probabilities"] for p in payloads])
    if len(states) != 60_000:
        raise RuntimeError(f"Expected 60,000 transitions, got {len(states)}")
    sample_idx = np.random.default_rng(20260926).choice(len(states), FIXED_TARGET_COUNT, replace=False)
    x = torch.as_tensor(states[sample_idx], dtype=torch.float32)
    a = torch.as_tensor(actions[sample_idx], dtype=torch.long)
    y = torch.as_tensor(targets[sample_idx], dtype=torch.float32)
    m = torch.as_tensor(masks[sample_idx], dtype=torch.bool)
    p = torch.as_tensor(probs[sample_idx], dtype=torch.float32)
    learner = _new_side_learner(rho, initialization_seed=39052026)

    def evaluate(step):
        predictions = []
        learner.network.eval()
        with torch.no_grad():
            for start in range(0, FIXED_TARGET_COUNT, 256):
                sl = slice(start, start + 256)
                _, selected, _, _ = learner.loss_for_selected(x[sl], a[sl], y[sl], p[sl], m[sl])
                predictions.append(selected.numpy())
        prediction, target = np.concatenate(predictions), y.numpy()
        error = prediction - target
        huber = np.where(np.abs(error) < 1, .5 * error ** 2, np.abs(error) - .5)
        return {"step": int(step), "huber_loss": float(huber.mean()),
                "correlation": _safe_corr(prediction, target),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2)))}

    rows = [evaluate(0)]
    learner.network.train()
    for step in range(1, FIXED_TARGET_STEPS + 1):
        indices = torch.randint(FIXED_TARGET_COUNT, (FIXED_TARGET_BATCH,), generator=learner.private_generator)
        learner.optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = learner.loss_for_selected(x[indices], a[indices], y[indices], p[indices], m[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.network.parameters(), 1.0)
        learner.optimizer.step()
        if step in {100, 500, 1000, 2000}:
            rows.append(evaluate(step))
            learner.network.train()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fixed_target_supervised_fit.csv", index=False)
    return frame


def _rank_metrics(scores, q_targets, actions, optimal_set):
    score, target = np.asarray(scores, dtype=float)[actions], np.asarray(q_targets, dtype=float)[actions]
    ranked = actions[np.argsort(-score, kind="stable")]
    optimal = set(map(int, optimal_set))
    action = int(ranked[0])
    return {
        "spearman": _safe_corr(score, target, "spearman"),
        "kendall": _safe_corr(score, target, "kendall"),
        "top1_hit": action in optimal,
        "top3_hit": any(int(v) in optimal for v in ranked[:min(3, len(ranked))]),
        "top5_hit": any(int(v) in optimal for v in ranked[:min(5, len(ranked))]),
        "mean_regret": float(np.max(target) - q_targets[action]),
        "predicted_action": action,
    }


def _load_heldout_models(rho):
    actors = {}
    for trial_id in range(10):
        checkpoint = PPO_OUT / "runs" / f"trial_{trial_id:03d}" / "masked" / "actor.pt"
        actor = PPOPairScoringPolicyNetwork(
            params.num_states, params.num_actions, params.hidden_layers_ppo,
            params.serverNo, rho, activation=params.af_ppo,
        ).cpu().eval()
        actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        actors[trial_id] = actor
    q = ActionValueCritic(
        params.num_states, params.num_actions, params.hidden_layers_ppo,
        params.serverNo, rho, activation=params.af_ppo, device="cpu",
        config=ActionValueCriticConfig(),
    )
    q.load(OLD_Q / "runs/trial_000/masked/q_critic.pt")
    q.online.eval()
    return actors, q.online


def evaluate_heldout(learner, rho):
    """Score the existing 1,000 held-out states; never trains on their labels."""
    actors, old_q = _load_heldout_models(rho)
    metadata = pd.read_csv(LONG / "counterfactual_state_results.csv")
    rows, state_count = [], 0
    learner.network.eval()
    with torch.no_grad():
        for trial_id in range(10):
            trial_dir = LONG / "runs" / f"trial_{trial_id:03d}"
            states = pd.read_csv(trial_dir / "states.csv")
            candidates = pd.read_csv(trial_dir / "candidate_returns.csv")
            state_meta = metadata[metadata.trial_id == trial_id].set_index("state_id")
            if states.state_id.nunique() != 100:
                raise RuntimeError(f"Trial {trial_id} must contain 100 held-out states")
            for state_row in states.itertuples(index=False):
                state_id = str(state_row.state_id)
                group = candidates[candidates.state_id == state_id].sort_values("action_index").reset_index(drop=True)
                candidate_actions = group.action_index.to_numpy(dtype=int)
                if (len(group) == 0 or len(np.unique(candidate_actions)) != len(group)
                        or (candidate_actions < 0).any() or (candidate_actions >= params.num_actions).any()):
                    raise RuntimeError(f"Invalid candidate action mapping for {state_id}")
                local_mask = _effective_mask(group)
                if int(local_mask.sum()) != int(group.effective_set_size.iloc[0]):
                    raise RuntimeError(f"Effective action mask mismatch for {state_id}")
                actions = candidate_actions[local_mask]
                local_probabilities = group.actor_probability.to_numpy(dtype=np.float32)
                if (np.max(np.abs(local_probabilities[~local_mask]), initial=0.0) > 1e-7
                        or not np.isclose(local_probabilities.sum(), 1.0, atol=2e-6)):
                    raise RuntimeError(f"Saved masked behavior policy does not match effective set for {state_id}")
                full_mask = np.zeros(params.num_actions, dtype=bool)
                full_mask[actions] = True
                probabilities = np.zeros(params.num_actions, dtype=np.float32)
                probabilities[candidate_actions] = local_probabilities
                state = np.asarray([getattr(state_row, f"state_feature_{i}") for i in range(params.num_states)], dtype=np.float32)
                tensor = torch.as_tensor(state).unsqueeze(0)
                raw = learner.network(tensor).squeeze(0)
                centered = policy_centered_advantage(
                    raw.unsqueeze(0), torch.as_tensor(probabilities).unsqueeze(0),
                    torch.as_tensor(full_mask).unsqueeze(0),
                ).squeeze(0).cpu().numpy()
                actor_scores = actors[trial_id](tensor).squeeze(0).cpu().numpy()
                q_scores = old_q(tensor).squeeze(0).cpu().numpy()
                if not np.isfinite(centered).all() or not np.isfinite(actor_scores).all() or not np.isfinite(q_scores).all():
                    raise FloatingPointError(f"Non-finite held-out score for {state_id}")
                state_info = state_meta.loc[state_id]
                coupling = "coupled" if float(state_info.regret_h20) > .01 else "easy"
                safe_n = int(group.safe_set_size.iloc[0])
                safe_bin = "empty" if safe_n == 0 else ("1-5" if safe_n <= 5 else ("6-15" if safe_n <= 15 else "16-28"))
                common = {
                    "state_id": state_id, "trial_id": trial_id,
                    "requirement": float(group.requirement.iloc[0]),
                    "load_tertile": str(group.load_tertile.iloc[0]),
                    "safe_set_size": safe_n, "safe_set_bin": safe_bin,
                    "effective_set_size": int(full_mask.sum()), "coupling_group": coupling,
                    "h20_myopic_regret": float(state_info.regret_h20),
                }
                for horizon in HORIZONS:
                    local_targets = group[f"q_h{horizon}"].to_numpy(dtype=float)
                    if not np.isfinite(local_targets).all():
                        raise RuntimeError(f"Invalid Q_H{horizon} labels for {state_id}")
                    q_targets = np.full(params.num_actions, np.nan, dtype=float)
                    q_targets[candidate_actions] = local_targets
                    optimum = actions[_optimal_set(q_targets[actions])]
                    for method, scores in (("Centered Advantage", centered), ("PPO Actor", actor_scores), ("Old Q critic", q_scores)):
                        rows.append({**common, "horizon": horizon, "method": method,
                                     "optimal_set_size": int(len(optimum)),
                                     **_rank_metrics(scores, q_targets, actions, optimum)})
                state_count += 1
    if state_count != 1000:
        raise RuntimeError(f"Expected exactly 1,000 held-out states, got {state_count}")
    return pd.DataFrame(rows)


def _aggregate_alignment(frame, groups):
    return frame.groupby(groups, observed=True, dropna=False).agg(
        state_count=("state_id", "nunique"), spearman=("spearman", "mean"),
        kendall=("kendall", "mean"), top1_rate=("top1_hit", "mean"),
        top3_rate=("top3_hit", "mean"), top5_rate=("top5_hit", "mean"),
        mean_regret=("mean_regret", "mean"), median_regret=("mean_regret", "median"),
    ).reset_index()


def _write_heldout_tables(alignment):
    alignment.to_csv(OUT / "heldout_advantage_alignment.csv", index=False)
    tables = {
        "heldout_advantage_by_coupling.csv": ["horizon", "coupling_group", "method"],
        "heldout_advantage_by_load.csv": ["horizon", "load_tertile", "method"],
        "heldout_advantage_by_requirement.csv": ["horizon", "requirement", "method"],
        "heldout_advantage_by_safe_set_size.csv": ["horizon", "safe_set_bin", "method"],
    }
    for name, keys in tables.items():
        _aggregate_alignment(alignment, keys).to_csv(OUT / name, index=False)
    return _aggregate_alignment(alignment, ["horizon", "method"])


def _selected_qh_status():
    reason = (
        "Q_H labels belong to frozen held-out evaluation replay states; captured GAE targets belong to training rollouts "
        "with separate arrival/spatial RNG streams. No shared transition mapping is saved, so a same-state comparison "
        "cannot be established."
    )
    pd.DataFrame([{
        "status": "not_alignable_from_existing_artifacts",
        "training_transitions": 60_000, "heldout_states": 1_000,
        "matched_state_transition_count": 0, "reason": reason,
        "counterfactual_qh_used_for_training": False,
    }]).to_csv(OUT / "advantage_vs_qh_selected_actions.csv", index=False)
    return reason


def _plot_outputs(predictions, updates, alignment, fixed):
    sample = predictions.sample(min(10000, len(predictions)), random_state=20260926)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(sample.actor_used_advantage_target, sample.selected_centered_advantage_prediction, s=7, alpha=.25)
    ax.set(xlabel="PPO actor-used advantage target", ylabel="Centered side-learner prediction", title="Online selected-action targets and predictions")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / "target_vs_predicted_advantage.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(updates.episode, updates.loss_mean, color="#3569a8")
    ax.set(xlabel="Episode", ylabel="Mean Huber loss", title="Online side-learner loss")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / "advantage_training_loss.png", dpi=160); plt.close(fig)

    by_ep = predictions.groupby("episode").agg(
        centered_std=("centered_effective_std", "mean"),
        centered_spread=("centered_effective_spread", "median"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(by_ep.episode, by_ep.centered_std, label="Effective-set std")
    ax.plot(by_ep.episode, by_ep.centered_spread, label="Median spread")
    ax.set(xlabel="Episode", ylabel="Centered advantage", title="Effective-set action differentiation")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / "centered_advantage_spread.png", dpi=160); plt.close(fig)

    h20 = alignment[alignment.horizon == 20]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    h20.groupby("method", observed=True).spearman.mean().reindex(
        ["Centered Advantage", "PPO Actor", "Old Q critic"]
    ).plot(kind="bar", ax=ax, color=["#49a078", "#e0a342", "#6989b8"])
    ax.axhline(0, color="gray", linewidth=.8)
    ax.set(xlabel="Method", ylabel="Mean Spearman", title="Held-out H=20 ranking")
    ax.tick_params(axis="x", rotation=15); fig.tight_layout(); fig.savefig(OUT / "h20_spearman_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    h20.groupby(["coupling_group", "method"], observed=True).spearman.mean().unstack().reindex(
        ["easy", "coupled"]
    ).plot(kind="bar", ax=ax)
    ax.set(xlabel="State subset", ylabel="Mean Spearman", title="Easy and coupled-state ranking")
    ax.tick_params(axis="x", rotation=0); fig.tight_layout(); fig.savefig(OUT / "coupled_vs_easy_ranking.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, group in h20.groupby("method", observed=True):
        ax.hist(group.mean_regret, bins=40, alpha=.45, label=method)
    ax.set(xlabel="H=20 argmax regret", ylabel="Held-out states", title="Held-out action regret")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "argmax_regret_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(fixed.step, fixed.huber_loss, "o-", color="#6e4c9b")
    ax.set(xlabel="Optimization step", ylabel="Frozen-sample Huber loss", title="Fixed-target supervised fit")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / "fixed_target_supervised_fit.png", dpi=160); plt.close(fig)


def _write_report(result, regression_smoke):
    targets, predictions = result["targets"], result["predictions"]
    centering, fixed, alignment = result["centering"], result["fixed"], result["alignment"]
    phases, v_rows, wiring = result["phases"], result["v_rows"], result["wiring"]
    summaries = result["summaries"]
    h20 = alignment[alignment.horizon == 20]
    high_h = alignment[alignment.horizon >= 10]
    coupled = high_h[high_h.coupling_group == "coupled"]
    easy = high_h[high_h.coupling_group == "easy"]
    h10plus = _aggregate_alignment(high_h, ["method"])
    h20c = _aggregate_alignment(coupled[coupled.horizon == 20], ["method"]).set_index("method")
    h10plus_idx = h10plus.set_index("method")
    spread = predictions.centered_effective_spread.to_numpy(float)
    multi = predictions.effective_set_size.to_numpy(int) >= 2
    collapse = float(np.mean(spread[multi] < 1e-6)) if multi.any() else 1.0
    median_spread = float(np.median(spread[multi])) if multi.any() else 0.0
    fit0, fit1000, fit2000 = [fixed[fixed.step == s].iloc[0] for s in (0, 1000, 2000)]
    fit_good = bool(fit2000.huber_loss <= .75 * fit0.huber_loss and fit2000.correlation > .3)
    spread_good = bool(median_spread > 1e-6 and collapse < .25)
    adv_coupled = float(h20c.loc["Centered Advantage", "spearman"])
    q_coupled = float(h20c.loc["Old Q critic", "spearman"])
    coupled_enough = int(h20c.loc["Centered Advantage", "state_count"]) >= 50
    overall_better = float(h10plus_idx.loc["Centered Advantage", "spearman"]) > float(h10plus_idx.loc["Old Q critic", "spearman"])
    top3_or_regret = (
        float(h10plus_idx.loc["Centered Advantage", "top3_rate"]) > float(h10plus_idx.loc["Old Q critic", "top3_rate"])
        or float(h10plus_idx.loc["Centered Advantage", "mean_regret"]) < float(h10plus_idx.loc["Old Q critic", "mean_regret"])
    )
    if not fit_good and not overall_better:
        classification = "C. BOTH TARGET AND REPRESENTATION ARE LIMITED"
    elif not fit_good or not spread_good:
        classification = "A. ADVANTAGE REPRESENTATION STILL FAILS"
    elif not (overall_better and adv_coupled > q_coupled and coupled_enough and top3_or_regret):
        classification = "B. PPO ADVANTAGE TARGET IS THE MAIN BOTTLENECK"
    else:
        classification = "D. POLICY-CENTERED ADVANTAGE IS LEARNED RELIABLY"
    go = bool(classification.startswith("D.") and regression_smoke["passed"]
              and result["regression_full"]["passed"] and wiring.permutation_pass.all()
              and wiring.pair_mapping_matches_combinations.all()
              and wiring.pair_feature_gradient_nonzero.all())
    phase_lines = []
    for row in phases.itertuples(index=False):
        phase_lines.append(
            f"| {row.phase} ({row.episodes}) | {row.raw_mean:.5g} | {row.raw_std:.5g} | "
            f"{row.actor_mean:.5g} | {row.actor_std:.5g} | {row.actor_median:.5g} | "
            f"{row.actor_p05:.5g} | {row.actor_p95:.5g} |"
        )
    hlines = []
    for row in summaries.sort_values(["horizon", "method"]).itertuples(index=False):
        hlines.append(
            f"| {row.horizon} | {row.method} | {row.spearman:.4f} | {row.kendall:.4f} | "
            f"{row.top1_rate:.3f} | {row.top3_rate:.3f} | {row.top5_rate:.3f} | {row.mean_regret:.5g} |"
        )
    subgroup_lines = []
    for label, frame in (("Coupled", coupled), ("Easy", easy)):
        summary = _aggregate_alignment(frame, ["method"])
        for row in summary.itertuples(index=False):
            subgroup_lines.append(
                f"| {label} | {row.method} | {row.spearman:.4f} | {row.top1_rate:.3f} | "
                f"{row.top3_rate:.3f} | {row.top5_rate:.3f} | {row.mean_regret:.5g} | {row.state_count} |"
            )
    regression = result["regression_full"]
    wiring_ok = bool(wiring.permutation_pass.all() and wiring.pair_mapping_matches_combinations.all()
                     and wiring.pair_feature_gradient_nonzero.all())
    report = [
        "# Policy-Centered Action-Advantage Side Learner Diagnosis", "",
        f"Formal PPO reference metadata commit: {result['meta']['git_commit']}; source checkout HEAD: {result['meta']['source_checkout_head']}. One seed (trial 0) × 300 episodes × 200 tasks = 60,000 transitions. This is descriptive one-seed evidence, not a formal multi-seed conclusion.", "",
        "## Scope and regression", "",
        "The side learner has separate pair-scoring weights and optimizer. It did not select actions or enter PPO losses. An optional observer copied immutable rollout data at the exact production location after raw GAE and actor-used normalization; side updates occurred only after the original PPO episode update. PPO Actor, V critic, losses, hyperparameters, reward, simulator, and masks were not changed.", "",
        f"Eight-episode regression: action mismatch {regression_smoke['action_mismatch']}; reward max difference {regression_smoke['reward_max_abs_diff']:.3g}; delay max difference {regression_smoke['delay_max_abs_diff']:.3g}; effective-mask mismatch {regression_smoke['effective_mask_mismatch']}; Actor/V hashes match {regression_smoke['actor_hash_match']}/{regression_smoke['critic_hash_match']}.",
        f"Full 300-episode regression against the existing formal trial-0 and Q-credit trace passed: {regression['passed']}; action/mask mismatches {regression['action_mismatch']}/{regression['effective_mask_mismatch']}; reward/delay max differences {regression['reward_max_abs_diff']:.3g}/{regression['delay_max_abs_diff']:.3g}; Actor/V hash match {regression['actor_hash_match']}/{regression['critic_hash_match']}.", "",
        "## PPO advantage targets", "",
        "Raw GAE is reported for diagnosis. The side learner target is the exact normalized advantage tensor consumed by the PPO Actor loss. Phase statistics pool transitions descriptively; the 60,000 transitions are not independent seeds.", "",
        "| Phase | Episodes | Raw GAE mean | Raw GAE std | Actor-used mean | Actor-used std | Median | P05 | P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *phase_lines, "",
        f"Policy-centering residual: mean {centering.mean_absolute_residual.mean():.3g}; max {centering.max_absolute_residual.max():.3g}. For states with at least two effective actions, median centered spread {median_spread:.5g}, collapse rate (<1e-6) {collapse:.2%}. Single-action support is excluded from the collapse statistic because zero spread is mathematically forced there.", "",
        f"Fixed frozen sample (n={FIXED_TARGET_COUNT}): Huber loss {fit0.huber_loss:.5g} at initialization, {fit1000.huber_loss:.5g} at 1,000 steps, {fit2000.huber_loss:.5g} at 2,000; final selected-action correlation {fit2000.correlation:.4f}, MAE {fit2000.mae:.5g}, RMSE {fit2000.rmse:.5g}.",
        f"Pair-feature wiring checks all pass: {wiring_ok}. Different pair features are fed to the shared scorer, pair permutation is equivariant, action indices map to the existing combinations order, and feature gradients are nonzero.", "",
        "## Held-out ranking against existing Q_H labels", "",
        "The established 1,000 held-out states and Q_H labels are evaluation-only. Methods share the effective action set and Q_H labels. Tie-aware optimal sets use the existing old-Q diagnostic tolerance max(1e-8, |Q_best| × 1e-10). Coupled uses regret_h20 > 0.01. Load tertiles, four R_req tiers, and safe-set bins 1–5 / 6–15 / 16–28 / empty reuse historical definitions.", "",
        "| H | Method | Spearman | Kendall | Top1 | Top3 | Top5 | Mean regret |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        *hlines, "",
        "### Coupled and easy states, H ≥ 10", "",
        "| Subset | Method | Spearman | Top1 | Top3 | Top5 | Mean regret | States |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *subgroup_lines, "",
        "Full subgroup detail is in heldout_advantage_by_load.csv, heldout_advantage_by_requirement.csv, and heldout_advantage_by_safe_set_size.csv.", "",
        "## GAE versus selected-action Q_H", "",
        result["selected_qh_status"], "No counterfactual values were used for training.", "",
        "## V critic observation only", "",
        f"Across episodes, mean V-to-GAE-return MAE {v_rows.value_target_mae.mean():.5g}, correlation {v_rows.value_target_correlation.mean():.4f}, explained variance {v_rows.explained_variance.mean():.4f}. No value parameters or updates were changed.", "",
        "## Required answers and decision", "",
        "- Q1: PPO trajectory unchanged in smoke and full trial-0 regression; action, reward, delay, masks, Actor, and V match.",
        f"- Q2: actor-used advantage standard deviations early/middle/late: {phases.actor_std.tolist()}.",
        f"- Q3: fixed-target fit {'converged under the pre-set rule' if fit_good else 'did not clearly converge under the pre-set rule'}.",
        f"- Q4: centering is numerically correct; maximum residual {centering.max_absolute_residual.max():.3g}.",
        f"- Q5: pair-specific wiring passed: {wiring_ok}.",
        f"- Q6: effective-set collapse rate among multi-action states: {collapse:.2%}; median centered spread {median_spread:.5g}.",
        f"- Q7/Q8: see ranking table. H≥10 overall Spearman Advantage/Actor/Old-Q = {h10plus_idx.loc['Centered Advantage','spearman']:.4f}/{h10plus_idx.loc['PPO Actor','spearman']:.4f}/{h10plus_idx.loc['Old Q critic','spearman']:.4f}.",
        f"- Q9: H20 coupled Spearman Advantage/Old-Q = {adv_coupled:.4f}/{q_coupled:.4f} over {int(h20c.loc['Centered Advantage','state_count'])} states.",
        "- Q10/Q11: the load and safe-set files show all strata; 6–15 is reported explicitly.",
        f"- Q12: {result['selected_qh_status']}",
        f"- Q13 root-cause category: {classification}.",
        "", f"Actor integration: {'YES' if go else 'NO'}. Only category D meets the stated gate; this one-seed diagnostic does not establish multi-seed significance.", "",
        "## Output files", "",
        "All experiment artifacts are under diagnostics/results/advantage_side_learner. Q_H is evaluation-only; the formal 10-seed experiment, old Q audit, external baselines, and simulator results were not modified.",
    ]
    (OUT / "ADVANTAGE_SIDE_LEARNER_DIAGNOSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return classification, go


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--regression-smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    group.add_argument("--resume-analysis", action="store_true",
                       help="Finish analysis from the saved 300-episode learner and observations")
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    meta, plan = formal_spec()
    meta["source_checkout_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    pairs, rho = build_pair_correlations()
    pairs, rho = list(pairs), np.asarray(rho, dtype=float)
    if len(pairs) != params.num_actions or len(pairs) != 28:
        raise RuntimeError("This audit expects the formal 28-action pair policy")
    if args.regression_smoke:
        result = run_regression_smoke(plan, rho)
        print(json.dumps(result, indent=2))
        return
    smoke_path = OUT / "regression_smoke.json"
    if not smoke_path.exists():
        raise RuntimeError("Run --regression-smoke and review it before --full")
    smoke = json.loads(smoke_path.read_text())
    if args.resume_analysis:
        model_path = OUT / "policy_centered_advantage.pt"
        observations_path = OUT / "training_sidecar_observations.npz"
        if not model_path.exists() or not observations_path.exists():
            raise RuntimeError("Saved model and sidecar observations are required for --resume-analysis")
        learner = _new_side_learner(rho)
        learner.network.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        learner.network.eval()
        with np.load(observations_path) as saved:
            payloads = []
            for episode in range(1, 301):
                selected = saved["episode"] == episode
                payloads.append({
                    "episode": episode,
                    "task_ids": saved["task_ids"][selected],
                    "states": saved["states"][selected],
                    "actions": saved["actions"][selected],
                    "effective_masks": saved["effective_masks"][selected],
                    "old_policy_probabilities": saved["old_policy_probabilities"][selected],
                    "raw_gae_advantages": saved["raw_gae_advantages"][selected],
                    "actor_used_advantages": saved["actor_used_advantages"][selected],
                    "value_predictions": saved["value_predictions"][selected],
                    "gae_return_targets": saved["gae_return_targets"][selected],
                })
        if sum(len(payload["actions"]) for payload in payloads) != 60_000:
            raise RuntimeError("Saved sidecar does not contain exactly 60,000 transitions")
        trial = plan.iloc[0].to_dict()
        training = {
            "meta": meta, "trial": trial, "learner": learner,
            "regression": json.loads((OUT / "regression_full_run.json").read_text()),
            "targets": pd.read_csv(OUT / "training_advantage_targets.csv"),
            "predictions": pd.read_csv(OUT / "training_advantage_predictions.csv"),
            "updates": pd.read_csv(OUT / "advantage_update_diagnostics.csv"),
            "centering": pd.read_csv(OUT / "centering_checks.csv"),
            "phases": pd.read_csv(OUT / "advantage_target_phase_statistics.csv"),
            "v_rows": pd.read_csv(OUT / "v_critic_observation_only.csv"),
            "wiring": pd.read_csv(OUT / "pair_feature_wiring_checks.csv"),
        }
    else:
        training = run_training(meta, plan, rho, smoke)
        payloads = training["observer"].payloads
    if not training["regression"]["passed"]:
        raise RuntimeError("Full PPO regression is required before analysis")
    fixed = fit_fixed_targets(payloads, rho)
    alignment = evaluate_heldout(training["learner"], rho)
    summaries = _write_heldout_tables(alignment)
    selected_qh_status = _selected_qh_status()
    _plot_outputs(training["predictions"], training["updates"], alignment, fixed)
    result = {**training, "regression_full": training["regression"],
              "fixed": fixed, "alignment": alignment,
              "summaries": summaries, "selected_qh_status": selected_qh_status}
    classification, go = _write_report(result, smoke)
    torch.save(training["learner"].network.state_dict(), OUT / "policy_centered_advantage.pt")
    status = {
        "formal_reference_commit": meta["git_commit"],
        "source_checkout_head": meta["source_checkout_head"],
        "trial_id": 0,
        "seed_plan": {key: int(value) for key, value in training["trial"].items()},
        "training_episodes": 300, "tasks_per_episode": 200,
        "training_transitions": 60_000, "heldout_state_count": 1000,
        "heldout_horizons": list(HORIZONS),
        "counterfactual_qh_used_for_training": False,
        "fixed_horizon_as_algorithm_parameter": False,
        "actor_v_loss_reward_mask_simulator_changed": False,
        "regression_smoke_passed": bool(smoke["passed"]),
        "full_regression_passed": bool(training["regression"]["passed"]),
        "centering_residual_max": float(training["centering"].max_absolute_residual.max()),
        "classification": classification,
        "actor_integration_go": bool(go),
        "legacy_formal_and_q_diagnostics_modified": False,
        "input_sha256": {
            "server_info": sha256_file(ROOT / "data/server_info.xlsx"),
            "task_parameters": sha256_file(ROOT / "data/task_parameters.xlsx"),
            "heldout_state_results": sha256_file(LONG / "counterfactual_state_results.csv"),
        },
    }
    (OUT / "experiment_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
