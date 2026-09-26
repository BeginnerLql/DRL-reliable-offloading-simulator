"""Replay the existing 1,000 held-out states for interval reward/bootstrap audits.

This uses the original long-horizon coupling branch runner and reuses its exact
safe/effective masks, task reward implementation, and simulator transitions.
No branches enter training.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

import diagnostics.run_long_horizon_coupling as cf
import diagnostics.run_action_value_critic as q_runner
from agents.action_value_masked_ppo_agent import ActionValueMaskedPairPPOAgent
from config.action_value_critic import ActionValueCriticConfig
from config.params import params
from tools.paired_ppo_experiment import _agent_kwargs
from Project_main import build_pair_correlations
from tools.paired_ppo_experiment import _agent_kwargs

OUT = ROOT / "diagnostics/results/q_credit_audit"
AUDIT_ROWS = []
BOOTSTRAP_ROWS = []


def _pair(task_frame, task_id):
    row = task_frame.loc[task_frame.task_id.astype(int) == int(task_id)]
    if row.empty:
        return None
    return f"({int(row.iloc[0].server_j)}, {int(row.iloc[0].server_k)})"


def run(states_per_trial=100, trial_ids=range(10), output_dir=None):
    global AUDIT_ROWS, BOOTSTRAP_ROWS
    AUDIT_ROWS, BOOTSTRAP_ROWS = [], []
    cf.HORIZONS = (1,)
    audit_agent_cls = cf.CounterfactualMaskedAgent
    original_select = audit_agent_cls.select_action
    original_branch = cf._run_branch
    original_copy = cf._copy_candidate_row
    original_sampler = cf._stratified_sample
    q_rows_by_task = {}

    def traced_select(agent, state, *args, **kwargs):
        context = agent.current_decision
        task_id = int(context["task_id"]) if context is not None else -1
        effective = np.asarray(context["effective_mask"], dtype=bool).copy() if context else None
        state_copy = np.asarray(state, dtype=np.float32).copy()
        action = original_select(agent, state, *args, **kwargs)
        if agent.forced_task_id is not None and task_id == int(agent.forced_task_id) + 1:
            decision = dict(agent.action_history[-1])
            agent._q_audit_next_decision = {
                "task_id": task_id,
                "decision_time": float(decision["decision_time"]),
                "state": state_copy,
                "effective_mask": effective,
                "probabilities": np.asarray(decision["masked_probabilities"], dtype=np.float64),
                "snapshot_digest": decision["snapshot_digest"],
            }
        return action

    def traced_branch(agent, *args, **kwargs):
        agent._q_audit_next_decision = None
        result = original_branch(agent, *args, **kwargs)
        result["q_audit_next_decision"] = agent._q_audit_next_decision
        return result

    def capture_copy(template, action, branch):
        row = original_copy(template, action, branch)
        assignments = branch["assignments"]
        task_id = int(template["task_id"])
        target_time = float(branch["target_time"])
        next_decision = branch.get("q_audit_next_decision")
        interval_end = float(next_decision["decision_time"]) if next_decision else float(branch["terminal_time"])
        if next_decision is None and task_id < 200:
            raise RuntimeError(f"No next arrival captured for nonterminal task {task_id}")
        if next_decision is not None and int(next_decision["task_id"]) != task_id + 1:
            raise RuntimeError("Counterfactual next decision is not task_id + 1")
        event_frame = assignments.copy()
        event_frame["outcome_time"] = pd.to_numeric(event_frame.Primary_Start, errors="coerce") + pd.to_numeric(event_frame.Task_Delay, errors="coerce")
        event_frame = event_frame[np.isfinite(event_frame.outcome_time) & np.isfinite(pd.to_numeric(event_frame.Task_Reward, errors="coerce"))]
        terminal_inclusive = next_decision is None
        interval_predicate = event_frame.outcome_time <= interval_end + 1e-12 if terminal_inclusive else event_frame.outcome_time < interval_end - 1e-12
        events = event_frame[(event_frame.outcome_time >= target_time - 1e-12) & interval_predicate]
        current_mask = events.task_id.astype(int) == task_id
        previous_mask = events.task_id.astype(int) < task_id
        current = float(events.loc[current_mask, "Task_Reward"].sum())
        previous = float(events.loc[previous_mask, "Task_Reward"].sum())
        other = 0.0
        total = float(events.Task_Reward.sum())
        if not np.isclose(current + previous + other, total, rtol=0.0, atol=1e-8):
            raise RuntimeError("Counterfactual interval task reward decomposition does not sum")
        if not np.isclose(total, float(branch["qvalues"][1]["q_return"]), rtol=0.0, atol=1e-8):
            raise RuntimeError("Recomputed interval reward differs from existing Q_H=1")
        event_base = float(events.Base_Reward.sum()) if "Base_Reward" in events else np.nan
        event_penalty = float(events.Reliability_Penalty.sum()) if "Reliability_Penalty" in events else np.nan
        discount = 1.0
        expectation = 0.0
        bootstrap = 0.0
        next_action_spread = np.nan
        next_action_q_std = np.nan
        next_mask_size = 0
        next_digest = None
        if next_decision is not None:
            next_state = torch.as_tensor(next_decision["state"], dtype=torch.float32).reshape(1, -1)
            next_mask = np.asarray(next_decision["effective_mask"], dtype=bool)
            next_prob = np.asarray(next_decision["probabilities"], dtype=np.float64)
            if not next_mask.any() or next_prob.shape != next_mask.shape or next_prob[next_mask].sum() <= 0:
                raise RuntimeError("Invalid saved next-state policy/mask")
            with torch.no_grad():
                q_next = q_runner_audit_critic.target(next_state).cpu().numpy()[0].astype(np.float64)
            supported = next_prob * next_mask
            supported /= supported.sum()
            expectation = float(np.dot(supported, q_next))
            next_action_spread = float(np.ptp(q_next[next_mask]))
            next_action_q_std = float(np.sqrt(np.average((q_next[next_mask] - np.average(q_next[next_mask], weights=supported[next_mask])) ** 2, weights=supported[next_mask])))
            next_mask_size = int(next_mask.sum())
            discount = float(params.gamma_ppo) ** max(interval_end - target_time, 0.0)
            bootstrap = discount * expectation
            next_digest = next_decision["snapshot_digest"]
        full = total + bootstrap
        q_rows_by_task.setdefault(str(template["state_id"]), []).append(full)
        out = {
            "state_id": str(template["state_id"]), "trial_id": int(template["trial_id"]),
            "episode": int(template["episode"]), "task_id": task_id,
            "action_index": int(action), "pair": row.get("pair"),
            "requirement": float(template["requirement"]),
            "target_decision_time": target_time, "next_decision_time": interval_end,
            "interval_duration_seconds": interval_end - target_time,
            "interval_reward_total": total, "r_current_task": current,
            "r_previous_tasks": previous, "r_other": other,
            "base_reward_component": event_base,
            "reliability_penalty_component_signed": -event_penalty,
            "reward_sum_matches_existing_q_h1": True,
            "next_effective_mask_size": next_mask_size,
            "next_action_q_spread": next_action_spread,
            "next_action_q_weighted_std": next_action_q_std,
            "discount_term": discount, "next_q_expectation": expectation,
            "bootstrap_term": bootstrap, "y_full": full,
            "y_current": current + bootstrap,
            "y_no_history": (total - previous) + bootstrap,
            "y_centered_placeholder": np.nan,
            "q_h1_existing": float(branch["qvalues"][1]["q_return"]),
            "q_h20_reference": np.nan, "q_h50_reference": np.nan,
            "next_state_snapshot_digest": next_digest,
            "counterfactual_training_used": False,
        }
        AUDIT_ROWS.append(out)
        if next_decision is not None:
            BOOTSTRAP_ROWS.append({
                "state_id": out["state_id"], "trial_id": out["trial_id"],
                "task_id": task_id, "action_index": int(action), "pair": row.get("pair"),
                "next_task_id": int(next_decision["task_id"]),
                "interval_duration_seconds": interval_end - target_time,
                "next_effective_mask_size": next_mask_size,
                "next_q_expectation": expectation,
                "next_action_q_spread": next_action_spread,
                "next_action_q_weighted_std": next_action_q_std,
                "discount_term": discount, "bootstrap_term": bootstrap,
            })
        return row

    audit_agent_cls.select_action = traced_select
    cf._run_branch = traced_branch
    cf._copy_candidate_row = capture_copy
    try:
        # The 10 trial files contain exactly 100 held-out states each. Re-running
        # with the original deterministic stratified sampler verifies/reuses them.
        scratch = Path(tempfile.mkdtemp(prefix="q_credit_cf_"))
        trial_ids = list(trial_ids)
        if int(states_per_trial) != 100:
            def sampler(snapshots, count, trial_id):
                saved = pd.read_csv(cf.OUT / "runs" / f"trial_{int(trial_id):03d}" / "states.csv")
                chosen_ids = set(saved.state_id.iloc[:int(count)])
                selected = [item for item in snapshots if item["state_id"] in chosen_ids]
                if len(selected) != int(count):
                    raise RuntimeError("Pilot state IDs cannot be restored from existing held-out sweep")
                return selected
            cf._stratified_sample = sampler
        for trial_id in trial_ids:
            before = len(AUDIT_ROWS)
            cf._trial_worker(trial_id, int(states_per_trial), scratch)
            rows = AUDIT_ROWS[before:]
            unique = {r["state_id"] for r in rows}
            print(f"held-out trial {trial_id + 1}/10: {len(unique)} states, {len(rows)} action branches", flush=True)
        shutil.rmtree(scratch, ignore_errors=True)
    finally:
        cf._stratified_sample = original_sampler
        audit_agent_cls.select_action = original_select
        cf._run_branch = original_branch
        cf._copy_candidate_row = original_copy

    out = pd.DataFrame(AUDIT_ROWS)
    output_dir = Path(output_dir) if output_dir is not None else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_dirs = [cf.OUT / "runs" / f"trial_{trial:03d}" for trial in trial_ids]
    reference = pd.concat([pd.read_csv(path / "candidate_returns.csv") for path in ref_dirs], ignore_index=True)
    ref = reference[["state_id", "action_index", "q_h20", "q_h50"]]
    out = out.drop(columns=["q_h20_reference", "q_h50_reference"]).merge(ref, on=["state_id", "action_index"], how="left", validate="one_to_one")
    if out.q_h20.isna().any() or out.q_h50.isna().any():
        raise RuntimeError("Replayed held-out state/action does not align with existing H=20/50 branch data")
    expected_states = int(states_per_trial) * len(trial_ids)
    if out.state_id.nunique() != expected_states:
        raise RuntimeError(f"Expected {expected_states} reused held-out states, got {out.state_id.nunique()}")
    reference = reference[reference.state_id.isin(out.state_id.unique())]
    expected = reference[["state_id", "action_index"]].drop_duplicates()
    observed = out[["state_id", "action_index"]].drop_duplicates()
    if len(observed) != len(expected):
        raise RuntimeError(f"Action candidate count differs from prior held-out sweep: {len(observed)} vs {len(expected)}")
    out.to_csv(output_dir / "counterfactual_reward_decomposition.csv", index=False)
    boot = pd.DataFrame(BOOTSTRAP_ROWS)
    boot.to_csv(output_dir / "bootstrap_action_spread.csv", index=False)
    (output_dir / "counterfactual_replay_status.json").write_text(json.dumps({
        "held_out_state_count": int(out.state_id.nunique()),
        "state_action_branch_count": int(len(out)),
        "existing_q_h1_replay_match_count": int(out.reward_sum_matches_existing_q_h1.sum()),
        "q_h20_q_h50_rows_matched": int(out.q_h20.notna().sum()),
        "counterfactual_targets_used_for_training": False,
    }, indent=2))
    print(f"Counterfactual branches written: {len(out)}; states: {out.state_id.nunique()}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-states", type=int, default=0, help="Run one trial with this many existing states as a replay validation")
    parser.add_argument("--trial-id", type=int, choices=range(10), help="Run one 100-state trial")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    from agents.action_value_masked_ppo_agent import ActionValueMaskedPairPPOAgent
    from Project_main import build_pair_correlations
    from diagnostics.run_action_value_critic import formal_spec
    from agents.action_conditioned_q_critic import ActionValueCritic
    from config.action_value_critic import ActionValueCriticConfig

    meta, plan = formal_spec()
    trial = plan.iloc[0].to_dict()
    pairs, rho = build_pair_correlations()
    kwargs = _agent_kwargs("pair_scoring", np.asarray(rho, dtype=float), int(trial["PPO_Minibatch_Seed"]))
    torch.manual_seed(int(trial["Torch_Init_Seed"]))
    bootstrap_agent = ActionValueMaskedPairPPOAgent(**kwargs, q_config=ActionValueCriticConfig())
    bootstrap_agent.q_critic.load(OUT / "training_run/runs/trial_000/masked/q_critic.pt")
    q_runner_audit_critic = bootstrap_agent.q_critic
    if args.pilot_states:
        run(args.pilot_states, [0], args.output_dir)
    elif args.trial_id is not None:
        run(100, [args.trial_id], args.output_dir)
    else:
        run(100, range(10), args.output_dir)
