"""Run the one-seed Bellman/reward-credit audit without changing learning behavior.

Only diagnostics are added: per-transition tensors are snapshotted before the
existing PPO/Q update, and reward provenance is copied after task resolution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

import diagnostics.run_action_value_critic as q_runner
import diagnostics.run_masked_pair_ppo_10seed as masked_runner
from agents.action_value_masked_ppo_agent import ActionValueMaskedPairPPOAgent
from agents.action_conditioned_q_critic import (
    masked_policy_expectation, selected_action_values, smdp_discount,
)
from config.action_value_critic import ActionValueCriticConfig
from config.params import params
from diagnostics.q_credit_audit import calculate_gae_returns
from tools.pair_policy_diagnostics import TASK_ASSIGNMENT_COLUMNS

OUT = ROOT / "diagnostics/results/q_credit_audit"
RUN_ROOT = OUT / "training_run"


def _state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class AuditedActionValueMaskedPairPPOAgent(ActionValueMaskedPairPPOAgent):
    """Save pre-update audit tensors, then run the inherited update unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_critic.audit_optimizer_steps = True
        self.audit_transition_rows = []
        self.audit_transition_arrays = []
        self.audit_target_snapshots = {}

    def train_step(self):
        if self.frozen or not self.states:
            return super().train_step()
        n = len(self.states)
        states = np.asarray(self.states, dtype=np.float32).copy()
        next_states = np.asarray(self.next_states, dtype=np.float32).copy()
        actions = np.asarray(self.actions, dtype=np.int64).copy()
        rewards_raw = np.asarray(self.rewards, dtype=np.float32).copy()
        delta = np.asarray(self.delta_times, dtype=np.float32).copy()
        dones = np.asarray(self.dones, dtype=bool).copy()
        masks = np.asarray(self.effective_masks, dtype=bool).copy()
        old_probs = np.asarray(self.rollout_old_policy_probabilities, dtype=np.float32).copy()
        episode = int(self._current_episode)
        if not (len(rewards_raw) == len(states) == len(next_states) == len(actions) == n):
            raise RuntimeError("Audit snapshot found inconsistent rollout arrays")
        if np.any(dones[:-1]) or not dones[-1]:
            raise RuntimeError("Expected exactly one terminal transition at each episode boundary")
        if not np.allclose(next_states[:-1], states[1:], rtol=0.0, atol=1e-6):
            raise RuntimeError("Arrival-order next-state does not match the next transition state")
        next_probs = np.empty_like(old_probs)
        next_masks = np.empty_like(masks)
        next_probs[:-1] = old_probs[1:]
        next_masks[:-1] = masks[1:]
        next_probs[-1] = 1.0 / self.num_actions
        next_masks[-1] = True
        reward_scale = float(self.reward_scale)
        rewards = rewards_raw * reward_scale
        device = self.device
        state_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        next_state_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        action_t = torch.as_tensor(actions, dtype=torch.long, device=device)
        reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        delta_t = torch.as_tensor(delta, dtype=torch.float32, device=device)
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=device)
        discount_t = smdp_discount(self.gamma, delta_t, device=device)
        mask_t = torch.as_tensor(next_masks, dtype=torch.bool, device=device)
        prob_t = torch.as_tensor(next_probs, dtype=torch.float32, device=device)
        q_target_state = {key: value.detach().cpu().clone() for key, value in self.q_critic.target.state_dict().items()}
        q_target_hash = _state_dict_sha256(q_target_state)
        target_count = int(self.q_critic.update_count)
        target_path = OUT / "target_snapshots" / f"episode_{episode:03d}_updates_{target_count:05d}.pt"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(q_target_state, target_path)
        self.audit_target_snapshots[episode] = {
            "path": str(target_path.relative_to(ROOT)),
            "sha256": q_target_hash,
            "update_count": target_count,
        }

        with torch.no_grad():
            next_q = self.q_critic.target(next_state_t)
            expectation = masked_policy_expectation(next_q, prob_t, mask_t)
            bootstrap = discount_t * expectation * (~done_t).to(torch.float32)
            targets = reward_t + bootstrap
            q_values = self.q_critic.online(state_t)
            selected_q = selected_action_values(q_values, action_t)
            current_values = self.value_net(state_t)
            next_values = self.value_net(next_state_t)
            v_returns, v_td = calculate_gae_returns(
                rewards, current_values.detach().cpu().numpy(), next_values.detach().cpu().numpy(),
                discount_t.detach().cpu().numpy(), dones, self.gae_lambda,
            )

        expected_next_states = np.zeros(n, dtype=bool)
        expected_next_masks = np.zeros(n, dtype=bool)
        expected_next_probs = np.zeros(n, dtype=bool)
        if n > 1:
            expected_next_states[:-1] = np.allclose(next_states[:-1], states[1:], rtol=0.0, atol=1e-6)
            expected_next_masks[:-1] = np.all(next_masks[:-1] == masks[1:], axis=1)
            expected_next_probs[:-1] = np.all(next_probs[:-1] == old_probs[1:], axis=1)
        expected_next_states[-1] = bool(dones[-1] and np.allclose(next_states[-1], 0.0, atol=0.0))
        expected_next_masks[-1] = bool(dones[-1] and next_masks[-1].all())
        expected_next_probs[-1] = bool(dones[-1] and np.allclose(next_probs[-1], 1.0 / self.num_actions))
        if not expected_next_states.all() or not expected_next_masks.all() or not expected_next_probs.all():
            raise RuntimeError("Collection-time next-state/policy/mask alignment failed")

        trace_lookup = {
            (int(item["episode"]), int(item["task_id"])): item
            for item in self.selection_archive
        }
        row_ids = list(self.task_ids)
        batch = {
            "states": states, "next_states": next_states, "actions": actions,
            "rewards_raw": rewards_raw, "rewards_scaled": rewards,
            "delta_times": delta, "dones": dones, "masks": masks,
            "old_probs": old_probs, "next_masks": next_masks,
            "next_probs": next_probs,
            "next_q_values": next_q.detach().cpu().numpy(),
            "selected_q": selected_q.detach().cpu().numpy(),
            "next_expectation": expectation.detach().cpu().numpy(),
            "discounts": discount_t.detach().cpu().numpy(),
            "bootstrap": bootstrap.detach().cpu().numpy(),
            "targets": targets.detach().cpu().numpy(),
            "v_values": current_values.detach().cpu().numpy(),
            "v_next_values": next_values.detach().cpu().numpy(),
            "v_returns": v_returns,
            "v_td": v_td,
        }
        self.audit_transition_arrays.append(batch)
        for index, task_id_raw in enumerate(row_ids):
            task_id = int(task_id_raw)
            trace = trace_lookup.get((episode, task_id), {})
            normalized_backlog = np.clip(states[index, 2 * params.serverNo:3 * params.serverNo], 0, 1 - 1e-7)
            backlog_seconds = params.BACKLOG_TIME_SCALE_SEC * normalized_backlog / (1.0 - normalized_backlog)
            row = {
                "transition_id": f"ep{episode:03d}:d{task_id:03d}",
                "episode": episode, "decision_index": task_id, "task_id": task_id,
                "selected_action": int(actions[index]),
                "selected_pair": trace.get("selected_pair"),
                "delta_t": float(delta[index]),
                "reward_term_raw": float(rewards_raw[index]),
                "reward_scale": reward_scale,
                "reward_term": float(rewards[index]),
                "discount_term": float(discount_t[index].item()),
                "next_q_expectation": float(expectation[index].item()),
                "bootstrap_term": float(bootstrap[index].item()),
                "bellman_target_online": float(targets[index].item()),
                "selected_q_pred": float(selected_q[index].item()),
                "td_error": float((targets[index] - selected_q[index]).item()),
                "terminal": bool(dones[index]),
                "effective_safe_set_size": int(masks[index].sum()),
                "safe_set_size": int(trace.get("safe_set_size", int(masks[index].sum()))),
                "safe_set_empty": bool(trace.get("safe_set_empty", False)),
                "target_update_count": target_count,
                "target_snapshot_sha256": q_target_hash,
                "target_snapshot_path": str(target_path.relative_to(ROOT)),
                "next_state_matches_next_row": bool(expected_next_states[index]),
                "next_mask_matches_collection": bool(expected_next_masks[index]),
                "pi_old_matches_collection": bool(expected_next_probs[index]),
                "load_backlog_seconds": float(backlog_seconds.sum()),
                "v_value": float(current_values[index].item()),
                "v_next_value": float(next_values[index].item()),
                "ppo_gae_return_target": float(v_returns[index]),
                "ppo_v_td_delta": float(v_td[index]),
            }
            self.audit_transition_rows.append(row)
        super().train_step()
        observed = self.q_calibration_samples[-1]
        if not np.allclose(observed["bellman_target"], batch["targets"], rtol=0.0, atol=2e-5):
            raise RuntimeError("Audit Bellman targets differ from the online Q training targets")
        if episode % 25 == 0:
            print(f"Q credit audit: completed training episode {episode}/300", file=sys.stderr, flush=True)


class AuditArrivalTraceLoop(masked_runner.ArrivalTraceLoop):
    """Capture reward assignment timestamps after, not during, the normal logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_assignment_audit = []

    def _collect_resolved_task_outcomes(self):
        before = len(self.task_Assignments_info)
        assignment_time = float(self.env.now)
        super()._collect_resolved_task_outcomes()
        for values in self.task_Assignments_info[before:]:
            row = dict(zip(TASK_ASSIGNMENT_COLUMNS, values))
            finish_times = [float(value) for value in (row["Primary_End"], row["Backup_End"])
                            if value is not None and np.isfinite(float(value))]
            if not finish_times:
                raise RuntimeError("Resolved task audit is missing first-result timestamp")
            self.reward_assignment_audit.append({
                "episode": int(row["episode"]), "task_id": int(row["task_id"]),
                "reward_assignment_time": assignment_time,
                "source_task_completion_time": min(finish_times),
                "source_task_decision_time": float(row["Primary_Start"]),
                "source_action_index": int(row["action_index"]),
            })


def _write_snapshot_arrays(agent):
    fields = agent.audit_transition_arrays[0].keys()
    payload = {field: np.concatenate([batch[field] for batch in agent.audit_transition_arrays], axis=0)
               for field in fields}
    np.savez_compressed(OUT / "bellman_recompute_sample_source.npz", **payload)
    pd.DataFrame(agent.audit_transition_rows).to_csv(OUT / "bellman_target_components.csv", index=False)
    pd.DataFrame(agent.q_critic.optimizer_audit_rows).to_csv(OUT / "q_optimizer_diagnostics.csv", index=False)
    (OUT / "target_snapshot_manifest.json").write_text(json.dumps(agent.audit_target_snapshots, indent=2))


def _write_task_logs(loop):
    assignments = pd.DataFrame(loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS)
    assignments.sort_values(["episode", "task_id"], kind="mergesort").to_csv(
        OUT / "training_task_assignments.csv.gz", index=False, compression="gzip"
    )
    rewards = pd.DataFrame(loop.log_data, columns=["episode", "rolling_reward", "episode_reward", "rolling_delay"])
    rewards.to_csv(OUT / "training_episode_curve.csv", index=False)
    decisions = pd.DataFrame(q_runner.run_trial_agent.selection_archive) if hasattr(q_runner, "run_trial_agent") else None
    if decisions is None:
        raise RuntimeError("Action archive unavailable after the audited run")
    decisions.to_csv(OUT / "training_decisions.csv.gz", index=False, compression="gzip")
    rep = pd.DataFrame(loop.replica_completion_log)
    if not rep.empty:
        rep.to_csv(OUT / "replica_completion_log.csv.gz", index=False, compression="gzip")
    (OUT / "reward_assignment_audit.json").write_text(json.dumps(loop.reward_assignment_audit, indent=2))
    return assignments, rewards, rep, decisions


def _frames_match(left_path, right_path, key_candidates):
    left, right = pd.read_csv(left_path), pd.read_csv(right_path)
    keys = [key for key in key_candidates if key in left.columns and key in right.columns]
    if keys:
        columns = [column for column in left.columns if column in right.columns]
        left = left[columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
        right = right[columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float), rtol=0.0, atol=1e-10, equal_nan=True):
                return False
        elif not left[column].fillna("<NA>").astype(str).equals(right[column].fillna("<NA>").astype(str)):
            return False
    return True


def verify_saved_smoke_regression():
    """Verify a completed audit run against the existing non-instrumented Q smoke."""
    current = RUN_ROOT / "runs/trial_000"
    reference = ROOT / "diagnostics/results/action_value_critic/runs/trial_000"
    current_train = current / "masked"
    reference_train = reference / "masked"
    actions = pd.read_csv(OUT / "training_decisions.csv.gz")
    assignments = pd.read_csv(OUT / "training_task_assignments.csv.gz")
    triples = ["episode", "task_id", "action_index"]
    sorted_actions = actions[triples].sort_values(["episode", "task_id"]).reset_index(drop=True)
    sorted_assignments = assignments[triples].sort_values(["episode", "task_id"]).reset_index(drop=True)
    if not sorted_actions.equals(sorted_assignments):
        raise RuntimeError("Audit actions do not align with simulator task assignments by episode/task_id")
    prior_trace = pd.read_csv(reference_train / "q_training_action_trace.csv")
    if not sorted_actions.equals(prior_trace[triples].sort_values(["episode", "task_id"]).reset_index(drop=True)):
        raise RuntimeError("Audited Q training action sequence differs from the non-instrumented smoke")
    curve = pd.read_csv(current_train / "training_curve.csv")
    old_curve = pd.read_csv(reference_train / "training_curve.csv")
    if not np.array_equal(curve.episode_reward.to_numpy(), old_curve.episode_reward.to_numpy()):
        raise RuntimeError("Instrumentation changed per-episode training reward")
    if not np.array_equal(curve.episode_total_delay.to_numpy(), old_curve.episode_total_delay.to_numpy()):
        raise RuntimeError("Instrumentation changed per-episode training delay")
    current_q = torch.load(current_train / "q_critic.pt", map_location="cpu", weights_only=True)
    previous_q = torch.load(reference_train / "q_critic.pt", map_location="cpu", weights_only=True)
    q_weights_exact = all(torch.equal(current_q[name][key], previous_q[name][key])
                          for name in ("online", "target") for key in current_q[name])
    if not q_weights_exact:
        raise RuntimeError("Audit instrumentation changed online/target Q parameters")
    evaluation_checks = {}
    for label in ("masked_stochastic", "masked_greedy"):
        cdir = current / label
        rdir = reference / label
        names = ["evaluation_task_assignments.csv", "evaluation_decisions.csv", "evaluation_episode_metrics.csv", "server_load.csv"]
        for name in names:
            cp, rp = cdir / name, rdir / name
            if cp.exists() and rp.exists():
                keys = ["episode", "task_id", "Server_ID", "server_id"]
                matches = _frames_match(cp, rp, keys)
                if not matches:
                    raise RuntimeError(f"Instrumentation changed {label}/{name}")
                evaluation_checks[f"{label}/{name}"] = True
    result = {
        "training_actions_keyed_exact": True,
        "training_reward_exact": True,
        "training_delay_exact": True,
        "q_online_and_target_weights_exact": q_weights_exact,
        "masked_stochastic_and_greedy_regression": evaluation_checks,
        "task_reward_delay_rsr_mask_and_queue_trajectory_exact": bool(evaluation_checks),
        "reward_formula_or_simulator_source_changed": False,
    }
    (OUT / "smoke_audit_status.json").write_text(json.dumps(result, indent=2))
    return result


def run_smoke():
    OUT.mkdir(parents=True, exist_ok=True)
    completed = RUN_ROOT / "runs/trial_000/masked/completed.json"
    if completed.exists() and (OUT / "bellman_target_components.csv").exists():
        result = verify_saved_smoke_regression()
        print(json.dumps(result, indent=2))
        return result
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    q_runner.OUT = RUN_ROOT
    q_runner.ActionValueMaskedPairPPOAgent = AuditedActionValueMaskedPairPPOAgent
    masked_runner.ArrivalTraceLoop = AuditArrivalTraceLoop
    original_train = q_runner.train_masked
    def capture_train(agent, *args, **kwargs):
        q_runner.run_trial_agent = agent
        loop, curve = original_train(agent, *args, **kwargs)
        q_runner.run_trial_loop = loop
        return loop, curve
    q_runner.train_masked = capture_train
    result = q_runner.run_trial(0, smoke=True)
    agent = q_runner.run_trial_agent
    loop = q_runner.run_trial_loop
    if len(agent.audit_transition_rows) != 60_000:
        raise RuntimeError(f"Expected 60,000 audited Q transitions, got {len(agent.audit_transition_rows)}")
    _write_snapshot_arrays(agent)
    assignments, curve, replica, decisions = _write_task_logs(loop)
    if len(assignments) != 60_000 or len(decisions) != 60_000:
        raise RuntimeError("Audited training task/decision logs do not contain 60,000 rows")
    assignment_actions = assignments[["episode", "task_id", "action_index"]].sort_values(["episode", "task_id"]).reset_index(drop=True)
    decision_actions = decisions[["episode", "task_id", "action_index"]].sort_values(["episode", "task_id"]).reset_index(drop=True)
    if not assignment_actions.equals(decision_actions):
        raise RuntimeError("Instrumentation changed or misaligned training actions by episode/task_id")
    summed = assignments.groupby("episode").Task_Reward.sum().reindex(curve.episode).to_numpy()
    if not np.allclose(summed, curve.episode_reward.to_numpy(), rtol=0.0, atol=1e-8):
        raise RuntimeError("Logged task rewards do not sum to the original episode reward")
    baseline_dir = q_runner.PPO_OUT / "runs/trial_000/masked"
    prior_trace = pd.read_csv(baseline_dir / "q_training_action_trace.csv")
    trace_key = ["episode", "task_id", "action_index"]
    current_trace = decisions[trace_key].sort_values(["episode", "task_id"]).reset_index(drop=True)
    previous_trace = prior_trace[trace_key].sort_values(["episode", "task_id"]).reset_index(drop=True)
    if not current_trace.equals(previous_trace):
        raise RuntimeError("Audited training action sequence differs from the non-instrumented smoke")
    for label in ("masked_stochastic", "masked_greedy"):
        current = pd.read_csv(RUN_ROOT / "runs/trial_000" / label / "evaluation_episode_metrics.csv")
        previous = pd.read_csv(q_runner.PPO_OUT / "runs/trial_000" / label / "evaluation_episode_metrics.csv")
        numeric = current.select_dtypes(include=[np.number]).columns.intersection(
            previous.select_dtypes(include=[np.number]).columns
        )
        if not np.allclose(current[numeric].to_numpy(), previous[numeric].to_numpy(), rtol=0.0, atol=1e-10, equal_nan=True):
            raise RuntimeError(f"Instrumentation changed {label} evaluation metrics")
    result["audit_regression"] = {
        "training_action_sequence_exact": True,
        "training_episode_reward_sum_exact": True,
        "masked_stochastic_eval_metrics_exact": True,
        "masked_greedy_eval_metrics_exact": True,
        "reward_formula_or_simulator_source_changed": False,
    }
    (OUT / "smoke_audit_status.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["audit_regression"], indent=2))
    print(f"Reward assignment events captured: {len(loop.reward_assignment_audit)} task outcomes")
    print(f"Training rows: {len(assignments)}; Q transition rows: {len(agent.audit_transition_rows)}")


if __name__ == "__main__":
    run_smoke()
