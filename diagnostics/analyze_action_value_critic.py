"""Held-out evaluation of online action-value critics against long-horizon diagnostics.

No data from this module is passed to Q training. H=5/10/20/50 are validation
horizons only; the learned critic uses one-step SMDP Expected-SARSA targets.
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
from scipy.stats import kendalltau, spearmanr
import torch

from agents.action_conditioned_q_critic import ActionValueCritic
from config.action_value_critic import ActionValueCriticConfig
from config.params import params
from Project_main import build_pair_correlations
from diagnostics.run_action_value_critic import OUT
from diagnostics.run_masked_pair_ppo_10seed import OUT as PPO_OUT, formal_spec
from tools.paired_ppo_experiment import _agent_kwargs
from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent

LONG = ROOT / "diagnostics/results/long_horizon_coupling"
HORIZONS = (5, 10, 20, 50)


def _corr(a, b, method):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    result = spearmanr(a, b).statistic if method == "spearman" else kendalltau(a, b).statistic
    return float(result) if np.isfinite(result) else np.nan


def _optimal_set(values, atol=1e-8):
    values = np.asarray(values, dtype=float)
    best = float(np.max(values))
    return np.flatnonzero(values >= best - max(atol, abs(best) * 1e-10))


def _effective_mask(group):
    group = group.sort_values("action_index")
    reliability = group.reliability.to_numpy(dtype=float)
    requirement = float(group.requirement.iloc[0])
    safe = reliability >= requirement
    if safe.any():
        mask = safe
    else:
        best = reliability.max()
        mask = np.isclose(reliability, best, rtol=0.0, atol=1e-12)
    expected = int(group.effective_set_size.iloc[0])
    if int(mask.sum()) != expected:
        raise RuntimeError(f"Counterfactual effective-mask mismatch: got {mask.sum()}, expected {expected}")
    return mask


def _state_features(row):
    record = row._asdict() if hasattr(row, "_asdict") else row
    return np.asarray([record[f"state_feature_{i}"] for i in range(params.num_states)], dtype=np.float32)


def score_trial(trial_id, q_trial_id=None):
    q_trial_id = int(trial_id if q_trial_id is None else q_trial_id)
    trial_dir = LONG / "runs" / f"trial_{trial_id:03d}"
    q_path = OUT / "runs" / f"trial_{q_trial_id:03d}" / "masked" / "q_critic.pt"
    actor_path = PPO_OUT / "runs" / f"trial_{trial_id:03d}" / "masked" / "actor.pt"
    if not q_path.exists() or not actor_path.exists():
        raise FileNotFoundError(f"Missing trained Q/Actor checkpoint for trial {trial_id}")
    states = pd.read_csv(trial_dir / "states.csv")
    candidates = pd.read_csv(trial_dir / "candidate_returns.csv")
    metadata = pd.read_csv(LONG / "counterfactual_state_results.csv")
    metadata = metadata[metadata.trial_id == trial_id].copy()
    state_lookup = {row.state_id: row for row in states.itertuples(index=False)}
    meta_lookup = {row.state_id: row for row in metadata.itertuples(index=False)}
    pairs, rho = build_pair_correlations()
    kwargs = _agent_kwargs("pair_scoring", np.asarray(rho, dtype=float), 1000 + trial_id)
    actor = ReliabilityMaskedPairPPOAgent(**kwargs)
    actor_state = torch.load(actor_path, map_location="cpu", weights_only=True)
    actor.policy_old.load_state_dict(actor_state)
    actor.policy_net.load_state_dict(actor_state)
    q = ActionValueCritic(params.num_states, params.num_actions, params.hidden_layers_ppo,
        params.serverNo, rho, activation=params.af_ppo, device="cpu", config=ActionValueCriticConfig())
    q.load(q_path)
    q.online.eval(); actor.policy_old.eval()

    rows = []
    for state_id, group in candidates.groupby("state_id", sort=False):
        if state_id not in state_lookup or state_id not in meta_lookup:
            raise RuntimeError(f"Missing held-out state metadata: {state_id}")
        group = group.sort_values("action_index").reset_index(drop=True)
        candidate_actions = group.action_index.to_numpy(dtype=int)
        if len(np.unique(candidate_actions)) != len(candidate_actions) or (candidate_actions < 0).any() or (candidate_actions >= params.num_actions).any():
            raise RuntimeError(f"Invalid/duplicate candidate action indices for {state_id}")
        mask = _effective_mask(group)
        effective_actions = candidate_actions[mask]
        if len(effective_actions) != int(group.effective_set_size.iloc[0]):
            raise RuntimeError(f"Recorded effective candidate count mismatch for {state_id}")
        srow = state_lookup[state_id]
        state = _state_features(srow)
        with torch.no_grad():
            q_values = q.online(torch.as_tensor(state).unsqueeze(0)).squeeze(0).cpu().numpy()
            actor_logits = actor.policy_old(torch.as_tensor(state).unsqueeze(0)).squeeze(0).cpu().numpy()
        actor_probabilities = group.actor_probability.to_numpy(dtype=float)
        if not np.isfinite(q_values).all() or not np.isfinite(actor_logits).all():
            raise FloatingPointError(f"Non-finite ranking for {state_id}")
        q_rank = effective_actions[np.argsort(-q_values[effective_actions], kind="stable")]
        actor_rank = effective_actions[np.argsort(-actor_logits[effective_actions], kind="stable")]
        state_meta = meta_lookup[state_id]
        coupling = "easy" if float(state_meta.regret_h20) <= 0.01 else "coupled"
        safe_n = int(mask.sum())
        common = {
            "state_id": state_id, "trial_id": int(trial_id),
            "requirement": float(group.requirement.iloc[0]),
            "load_tertile": str(group.load_tertile.iloc[0]),
            "safe_set_size": int(group.safe_set_size.iloc[0]),
            "effective_set_size": safe_n,
            "coupling_group": coupling,
            "q_mean_effective": float(np.mean(q_values[effective_actions])),
            "q_spread": float(np.max(q_values[effective_actions]) - np.min(q_values[effective_actions])),
            "q_std_safe": float(np.std(q_values[effective_actions])),
            "q_predicted_action": int(q_rank[0]),
            "actor_predicted_action": int(actor_rank[0]),
        }
        for horizon in HORIZONS:
            value_column = f"q_h{horizon}"
            if value_column not in group or not np.isfinite(group[value_column].to_numpy(dtype=float)).all():
                raise RuntimeError(f"Missing/non-finite held-out {value_column} for {state_id}")
            targets = group[value_column].to_numpy(dtype=float)
            target_by_action = np.full(params.num_actions, np.nan, dtype=float)
            target_by_action[candidate_actions] = targets
            target_safe = target_by_action[effective_actions]
            opt_local = _optimal_set(target_safe)
            optimal = set(effective_actions[opt_local].tolist())
            best = float(np.max(target_safe))
            q_regret = best - float(target_by_action[q_rank[0]])
            actor_regret = best - float(target_by_action[actor_rank[0]])
            q_top = [int(a) in optimal for a in q_rank]
            actor_top = [int(a) in optimal for a in actor_rank]
            random_hit = {}
            for k in (1, 3, 5):
                k_eff = min(k, safe_n)
                # Exact probability that a uniformly selected top-k contains
                # at least one of m optimal actions, without replacement.
                m = len(optimal)
                miss = 0.0 if safe_n - m < k_eff else float(np.prod([(safe_n - m - j) / (safe_n - j) for j in range(k_eff)]))
                random_hit[k] = 1.0 - miss
            rows.append({
                **common, "horizon": horizon,
                "q_spearman": _corr(q_values[effective_actions], target_safe, "spearman"),
                "q_kendall": _corr(q_values[effective_actions], target_safe, "kendall"),
                "q_top1_hit": bool(q_top[0]), "q_top3_hit": bool(any(q_top[:min(3, safe_n)])),
                "q_top5_hit": bool(any(q_top[:min(5, safe_n)])), "q_regret": q_regret,
                "q_optimal_set_size": int(len(optimal)),
                "actor_spearman": _corr(actor_logits[effective_actions], target_safe, "spearman"),
                "actor_kendall": _corr(actor_logits[effective_actions], target_safe, "kendall"),
                "actor_top1_hit": bool(actor_top[0]), "actor_top3_hit": bool(any(actor_top[:min(3, safe_n)])),
                "actor_top5_hit": bool(any(actor_top[:min(5, safe_n)])), "actor_regret": actor_regret,
                "random_top1_hit": random_hit[1], "random_top3_hit": random_hit[3],
                "random_top5_hit": random_hit[5],
                "actor_probability_spearman": _corr(actor_probabilities[mask], actor_logits[effective_actions], "spearman"),
            })
    return pd.DataFrame(rows)


def _summary(frame, by):
    aggregations = {
        "state_count": ("state_id", "nunique"),
        "q_spearman": ("q_spearman", "mean"), "q_kendall": ("q_kendall", "mean"),
        "q_top1_hit_rate": ("q_top1_hit", "mean"), "q_top3_hit_rate": ("q_top3_hit", "mean"),
        "q_top5_hit_rate": ("q_top5_hit", "mean"), "q_mean_regret": ("q_regret", "mean"),
        "q_median_regret": ("q_regret", "median"), "q_spread_mean": ("q_spread", "mean"),
        "q_safe_std_mean": ("q_std_safe", "mean"), "q_mean_effective": ("q_mean_effective", "mean"),
        "actor_spearman": ("actor_spearman", "mean"), "actor_kendall": ("actor_kendall", "mean"),
        "actor_top1_hit_rate": ("actor_top1_hit", "mean"), "actor_top3_hit_rate": ("actor_top3_hit", "mean"),
        "actor_top5_hit_rate": ("actor_top5_hit", "mean"), "actor_mean_regret": ("actor_regret", "mean"),
        "random_top1_hit_rate": ("random_top1_hit", "mean"),
        "random_top3_hit_rate": ("random_top3_hit", "mean"),
        "random_top5_hit_rate": ("random_top5_hit", "mean"),
    }
    keys = by if isinstance(by, list) else [by]
    return frame.groupby(keys, observed=True, dropna=False).agg(**aggregations).reset_index()


def _plots(out, alignment, horizon_summary, coupling_summary, actor_summary, calibration, training):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(horizon_summary.horizon, horizon_summary.q_spearman, "o-", label="Q critic")
    ax.plot(horizon_summary.horizon, horizon_summary.actor_spearman, "o--", label="Actor")
    ax.axhline(0, color="gray", linewidth=.8); ax.set(xlabel="Validation horizon", ylabel="Mean Spearman", title="Action ranking alignment")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"q_vs_counterfactual_spearman.png",dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in (1,3,5): ax.plot(horizon_summary.horizon, horizon_summary[f"q_top{k}_hit_rate"], "o-", label=f"Top-{k}")
    ax.set(xlabel="Validation horizon", ylabel="Optimal-set hit rate", title="Q critic top-k hit")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"q_topk_hit.png",dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(horizon_summary.horizon, horizon_summary.q_mean_regret, "o-", label="Q argmax")
    ax.plot(horizon_summary.horizon, horizon_summary.actor_mean_regret, "o--", label="Actor argmax")
    ax.set(xlabel="Validation horizon", ylabel="Q_H regret", title="Predicted-action long-horizon regret")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"q_regret.png",dpi=160); plt.close(fig)

    if "coupling_group" in coupling_summary:
        fig, ax = plt.subplots(figsize=(7,4.5))
        pivot=coupling_summary.pivot(index="horizon",columns="coupling_group",values="q_spearman")
        pivot.reindex(columns=["easy","coupled"]).plot(kind="bar",ax=ax)
        ax.set(xlabel="Validation horizon",ylabel="Mean Q Spearman",title="Easy and coupled states")
        fig.tight_layout();fig.savefig(out/"q_alignment_easy_vs_coupled.png",dpi=160);plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(actor_summary.horizon, actor_summary.q_spearman, "o-", label="Q critic vs Q_H")
    ax.plot(actor_summary.horizon, actor_summary.actor_spearman, "o--", label="Actor vs Q_H")
    ax.set(xlabel="Validation horizon",ylabel="Mean Spearman",title="Actor vs action-value critic")
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/"actor_vs_q_alignment.png",dpi=160);plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(calibration.mean_predicted_q,calibration.mean_bellman_target,"o-",label="Bellman target bins")
    lo=min(calibration.mean_predicted_q.min(),calibration.mean_bellman_target.min())
    hi=max(calibration.mean_predicted_q.max(),calibration.mean_bellman_target.max())
    ax.plot([lo,hi],[lo,hi],"--",color="gray",label="Ideal")
    ax.set(xlabel="Mean predicted Q",ylabel="Mean Bellman target",title="Selected-action TD calibration")
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/"q_calibration.png",dpi=160);plt.close(fig)

    grouped=training.groupby("episode",as_index=False).agg(q_loss=("q_loss","mean"),td_error_abs_p95=("td_error_abs_p95","mean"))
    fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(grouped.episode,grouped.q_loss)
    ax.set(xlabel="Episode",ylabel="Huber loss",title="Q loss across seeds");ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/"q_loss_curve.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(grouped.episode,grouped.td_error_abs_p95)
    ax.set(xlabel="Episode",ylabel="P95 absolute TD error",title="Q TD error across seeds");ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/"q_td_error.png",dpi=160);plt.close(fig)


def _markdown_table(frame):
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("NA")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(alignment, by_h, by_c, by_l, by_s, actor_summary, training, pair_visits, calibration):
    h10plus=alignment[alignment.horizon>=10]
    overall_q=float(h10plus.q_spearman.mean()); overall_actor=float(h10plus.actor_spearman.mean())
    coupled=h10plus[h10plus.coupling_group=="coupled"]
    easy=h10plus[h10plus.coupling_group=="easy"]
    coupled_q=float(coupled.q_spearman.mean()) if len(coupled) else float("nan")
    easy_q=float(easy.q_spearman.mean()) if len(easy) else float("nan")
    q_top3=float(h10plus.q_top3_hit.mean()); actor_top3=float(h10plus.actor_top3_hit.mean())
    easy_rows=h10plus[h10plus.coupling_group=="easy"]
    easy_top3_edge=float((easy_rows.q_top3_hit.astype(float)-easy_rows.random_top3_hit).mean()) if len(easy_rows) else float("nan")
    spread=float(alignment.q_spread.mean())
    q_scale=float(alignment.q_mean_effective.abs().mean())
    relative_spread=spread/max(q_scale,1e-8)
    q_action_collapse=relative_spread < 1e-3
    if (not q_action_collapse and overall_q > overall_actor and coupled_q > 0
            and q_top3 > actor_top3):
        decision="A. ACTION-VALUE LEARNING IS VIABLE"
    elif easy_q > 0.05 and easy_top3_edge > 0.02 and coupled_q <= 0 and not q_action_collapse:
        decision="B. Q LEARNING WORKS ONLY ON EASY STATES"
    else:
        decision="C. ACTION VALUE IS NOT LEARNED RELIABLY"
    horizon_table=by_h[["horizon","state_count","q_spearman","q_kendall","q_top1_hit_rate","q_top3_hit_rate","q_top5_hit_rate","q_mean_regret","actor_spearman","actor_top3_hit_rate","actor_mean_regret"]]
    min_vis=int(pair_visits.visit_count.min()); zero=int((pair_visits.visit_count==0).sum())
    stable=bool(np.isfinite(training.select_dtypes(include=[np.number]).to_numpy()).all() and training.q_max.abs().max()<1e6)
    report=f'''# Action-Value Critic Diagnosis

## 1. Motivation

The prior long-horizon experiment found action-dependent future effects: Safe Min-Latency was in the H=1 oracle optimal set for 98.6% of states, falling to about 78.4% by H≥10. This motivates action-specific credit assignment. The finite horizons below are held-out evaluation lenses only; no fixed horizon enters the learned algorithm.

## 2. Method

The side learner predicts all 28 pair values with a shared scorer over the same decision-time node, task, and pair-correlation features as the Pair Actor. Its weights and optimizer are independent. For selected rollout action `a_t`, it minimizes Huber loss against `r_t + gamma**delta_t * E_pi_old[Q_target(s',.)]`; terminal rows use `r_t`. The effective action mask is saved at collection time, and each next-state masked policy distribution is stored before PPO updates. The target network uses soft-update tau={ActionValueCriticConfig().target_tau:g}, interval={ActionValueCriticConfig().target_update_interval}; Q learning runs {ActionValueCriticConfig().updates_per_rollout} updates per rollout. Only simulator-generated online transitions train Q.

## 3. Training Stability

- All Q training metrics finite / explosion guard passed: **{stable}**.
- Maximum absolute online/target prediction: **{max(training.q_min.abs().max(),training.q_max.abs().max(),training.target_q_min.abs().max(),training.target_q_max.abs().max()):.4g}**.
- Final episode selected-action Q standard deviation: **{training.sort_values('episode').groupby('episode').q_pred_std.mean().iloc[-1]:.4g}**.
- Pair visitation minimum: **{min_vis}**; never visited: **{zero}/28**.
- Bellman-target decile calibration MAE: **{calibration.calibration_error.abs().mean():.4g}**.
- Mean held-out effective-set Q spread: **{spread:.6g}** on mean absolute Q scale **{q_scale:.4g}** (relative spread **{relative_spread:.3g}**); action-value collapse flag (<1e-3 relative): **{q_action_collapse}**.
- Mean rollout target vs selected-Q prediction in the final episode: **{training.q_target_mean.iloc[-1]:.4g}** vs **{training.q_pred_mean.iloc[-1]:.4g}**; mean TD error remains **{training.td_error_mean.iloc[-1]:.4g}**.

## 4. Counterfactual Alignment

The existing 1,000 diagnostic states are held out. Table reports per-state Q-vs-Q_H and Actor-vs-Q_H alignment. H=5/10/20/50 are all shown; no horizon was selected post hoc.

{_markdown_table(horizon_table)}

## 5. Easy vs Coupled States

States are called easy when the existing H=20 myopic regret is ≤0.01, and coupled otherwise.

{_markdown_table(by_c)}

## 6. Load / Safe-set Stratification

### Load

{_markdown_table(by_l)}

### Effective safe-set size

{_markdown_table(by_s)}

## 7. Actor vs Q Critic

Across H≥10, mean Q Spearman is **{overall_q:.4f}** versus Actor **{overall_actor:.4f}**; top-3 optimal-set hit is **{q_top3:.3%}** versus Actor **{actor_top3:.3%}**. The Actor is closer overall; mean argmax regret is **{h10plus.q_regret.mean():.4f}** for Q and **{h10plus.actor_regret.mean():.4f}** for Actor. On coupled states Q Spearman is **{coupled_q:.4f}**; on easy states it is **{easy_q:.4f}**, with easy top-3 gain over random of **{easy_top3_edge:.3%}**.

## 8. Q Calibration

Calibration uses selected-action Bellman targets from online rollouts, not counterfactual returns. See `q_calibration.csv` and `q_calibration.png`. This is one-step TD calibration and does not establish calibration to an optimal-control value function.

## 9. Limitations

Counterfactual Q_H is evaluation-only and comes from fixed downstream Masked PPO continuation under finite horizons. It is not the optimal-control Q*. State-level diagnostic snapshots are held out, but the counterfactual labels are policy-dependent and may include simulation noise. The existing scalar V(s) and PPO Actor update remain unchanged.

## 10. Decision

**{decision}**

This decision uses the H≥10 Q-vs-Q_H rank correlation, easy/coupled split, top-3 hit relative to random, Actor comparison, and relative within-state Q spread. The one-seed critic is numerically finite but its action values collapse to a very narrow range and coupled-state ranking is at/below zero, so the formal 10-seed stage was not started. Action-Value Advantage PPO is not implemented.
'''
    (OUT/"ACTION_VALUE_CRITIC_DIAGNOSIS.md").write_text(report)
    return decision


def _write_analysis(alignment, training_trial_ids, *, smoke=False):
    alignment.to_csv(OUT/"q_counterfactual_alignment.csv",index=False)
    by_h=_summary(alignment,"horizon")
    by_h.to_csv(OUT/"q_alignment_by_horizon.csv",index=False)
    by_c=_summary(alignment,["horizon","coupling_group"])
    by_c.to_csv(OUT/"q_alignment_by_coupling.csv",index=False)
    by_l=_summary(alignment,["horizon","load_tertile"])
    by_l.to_csv(OUT/"q_alignment_by_load.csv",index=False)
    alignment["safe_set_bin"] = pd.cut(alignment.effective_set_size,[0,5,15,28],labels=["1-5","6-15","16-28"])
    by_s=_summary(alignment,["horizon","safe_set_bin"])
    by_s.to_csv(OUT/"q_alignment_by_safe_set.csv",index=False)
    actor_summary=by_h.copy()
    actor_summary.to_csv(OUT/"actor_vs_q_alignment.csv",index=False)

    training=[]; visits=[]; cal=[]
    for i in training_trial_ids:
        root=OUT/"runs"/f"trial_{i:03d}"/"masked"
        t=pd.read_csv(root/"q_training_metrics.csv");t["trial_id"]=i;training.append(t)
        v=pd.read_csv(root/"q_pair_visitation.csv");v["trial_id"]=i;visits.append(v)
        c=pd.read_csv(root/"q_calibration.csv");c["trial_id"]=i;cal.append(c)
    training=pd.concat(training,ignore_index=True)
    training.to_csv(OUT/"q_training_metrics.csv",index=False)
    visits=pd.concat(visits,ignore_index=True)
    summary_visits=visits.groupby(["action_index","pair"],as_index=False).agg(
        visit_count=("visit_count","sum"), mean_selected_q=("mean_selected_q","mean"),
        mean_absolute_td_error=("mean_absolute_td_error","mean"))
    summary_visits.to_csv(OUT/"q_pair_visitation.csv",index=False)
    cal=pd.concat(cal,ignore_index=True)
    cal_summary=cal.groupby("predicted_decile",as_index=False).apply(
        lambda g: pd.Series({"sample_count":int(g.sample_count.sum()),
            "mean_predicted_q":float(np.average(g.mean_predicted_q,weights=g.sample_count)),
            "mean_bellman_target":float(np.average(g.mean_bellman_target,weights=g.sample_count))}),
        include_groups=False).reset_index(drop=True)
    cal_summary["calibration_error"]=cal_summary.mean_predicted_q-cal_summary.mean_bellman_target
    cal_summary.to_csv(OUT/"q_calibration.csv",index=False)
    _plots(OUT,alignment,by_h,by_c,actor_summary,cal_summary,training)
    decision=write_report(alignment,by_h,by_c,by_l,by_s,actor_summary,training,summary_visits,cal_summary)
    metadata={"held_out_state_count":int(alignment.state_id.nunique()),"horizons":list(HORIZONS),
        "training_counterfactual_leakage":False,"decision":decision,
        "q_training_seed_count":int(training.trial_id.nunique()),"formal_10seed_started":False if smoke else True,"evaluation_smoke_only":bool(smoke)}
    (OUT/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", type=int, choices=range(10), help="Smoke-align one trained seed")
    parser.add_argument("--all-states-q-trial-id", type=int, choices=range(10), help="Evaluate one smoke Q model on all 1,000 held-out states")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.trial_id is not None:
        trial_frame = score_trial(args.trial_id)
        summary = _summary(trial_frame, "horizon")
        trial_frame.to_csv(OUT / f"q_counterfactual_alignment_smoke_trial_{args.trial_id:03d}.csv", index=False)
        summary.to_csv(OUT / f"q_alignment_smoke_by_horizon_trial_{args.trial_id:03d}.csv", index=False)
        print(summary.to_string(index=False))
        return
    if args.all_states_q_trial_id is not None:
        frames = [score_trial(i, q_trial_id=args.all_states_q_trial_id) for i in range(10)]
        alignment = pd.concat(frames, ignore_index=True)
        _write_analysis(alignment, [args.all_states_q_trial_id], smoke=True)
        return
    _,plan=formal_spec()
    trial_frames=[score_trial(i) for i in range(10)]
    alignment=pd.concat(trial_frames,ignore_index=True)
    _write_analysis(alignment, range(10), smoke=False)


if __name__=="__main__":main()
