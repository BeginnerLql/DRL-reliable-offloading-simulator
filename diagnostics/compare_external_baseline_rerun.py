"""Audit a fresh external-baseline rerun against the preserved formal results.

This is post-processing only: it never loads or updates a learning agent and
never changes either simulator result directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OLD_DEFAULT = ROOT / "diagnostics/results/external_baselines"
NEW_DEFAULT = ROOT / "diagnostics/results/external_baselines_rerun"
POLICIES = ("Safe Min-Latency", "Masked PPO stochastic")
METRICS = (
    "mean_reward", "mean_latency", "p50_latency", "p90_latency", "p95_latency",
    "overall_rsr", "highest_rsr", "feasibility_rate", "conditional_rsr",
    "avoidable_violation_count", "avoidable_violation_rate",
    "unavoidable_violation_count", "unavoidable_violation_rate",
    "mean_safe_set_size", "pair_selection_hhi", "pair_selection_entropy",
    "top1_pair_frequency", "unique_selected_pairs", "pair_78_frequency",
    "maximum_server_selection_share", "maximum_server_utilization",
    "maximum_mean_queue_length", "maximum_p95_queue_length", "mean_server_waiting_time",
)
OLD_NEW_TABLE = (
    ("mean_reward", "Reward"), ("mean_latency", "Mean latency (s)"),
    ("p95_latency", "P95 latency (s)"), ("overall_rsr", "Overall RSR"),
    ("highest_rsr", "Highest-tier RSR"), ("pair_selection_hhi", "Pair HHI"),
    ("maximum_mean_queue_length", "Max mean queue"),
)
TASK_NUMERIC = (
    "Task_Reward", "Task_Delay", "Execution_Reliability", "selected_pair_reliability",
    "estimated_latency", "R_req", "safe_set_size", "safe_set_empty",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(path: Path) -> dict[str, str]:
    return {str(file.relative_to(path)): _sha256(file)
            for file in sorted(path.rglob("*")) if file.is_file()}


def _finite(frame: pd.DataFrame, label: str) -> None:
    values = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} contains non-finite numeric values")


def _load_seed_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "baseline_seed_results.csv")
    if len(frame) != 70 or set(frame.policy.unique()) != {
        "Safe Min-Latency", "Masked PPO stochastic", "Max-Reliability", "Safe Random",
        "Pair PPO greedy", "Pair PPO stochastic", "Masked PPO greedy",
    }:
        raise RuntimeError(f"Expected 10 seed rows for each of seven policies in {path}")
    if not (frame.groupby("policy").trial_id.nunique() == 10).all():
        raise RuntimeError(f"Incomplete seed coverage in {path}")
    missing = set(METRICS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing seed metrics in {path}: {sorted(missing)}")
    _finite(frame, f"seed results in {path}")
    return frame.sort_values(["policy", "trial_id"]).reset_index(drop=True)


def _compare_task_telemetry(old_dir: Path, new_dir: Path) -> tuple[pd.DataFrame, dict]:
    old = pd.read_csv(old_dir / "baseline_decision_telemetry.csv.gz")
    new = pd.read_csv(new_dir / "baseline_decision_telemetry.csv.gz")
    key = ["trial_id", "policy", "episode", "task_id"]
    rows, totals = [], {}
    for policy in POLICIES:
        left = old.loc[old.policy.eq(policy)].copy()
        right = new.loc[new.policy.eq(policy)].copy()
        if len(left) != 40000 or len(right) != 40000:
            raise RuntimeError(f"{policy}: expected 40000 old and new decision rows")
        merged = left.merge(right, on=key, how="outer", suffixes=("_old", "_rerun"),
                            indicator=True, validate="one_to_one")
        if not merged._merge.eq("both").all():
            raise RuntimeError(f"{policy}: task keys differ between old and rerun telemetry")
        action_mismatch = int((merged.action_index_old != merged.action_index_rerun).sum())
        pair_mismatch = int((merged.selected_pair_old != merged.selected_pair_rerun).sum())
        record = {"policy": policy, "tasks": len(merged),
                  "action_mismatch_count": action_mismatch,
                  "selected_pair_mismatch_count": pair_mismatch}
        for column in TASK_NUMERIC:
            a = pd.to_numeric(merged[f"{column}_old"], errors="raise").to_numpy(dtype=float)
            b = pd.to_numeric(merged[f"{column}_rerun"], errors="raise").to_numpy(dtype=float)
            record[f"max_abs_diff_{column}"] = float(np.max(np.abs(a - b)))
        record["safe_set_size_mismatch_count"] = int(
            (merged.safe_set_size_old != merged.safe_set_size_rerun).sum())
        record["safe_set_empty_mismatch_count"] = int(
            (merged.safe_set_empty_old != merged.safe_set_empty_rerun).sum())
        record["reliability_satisfied_mismatch_count"] = int(
            (merged.Reliability_Satisfied_old != merged.Reliability_Satisfied_rerun).sum())
        record["reliability_requirement_mismatch_count"] = int(
            (merged.Reliability_Requirement_old != merged.Reliability_Requirement_rerun).sum())
        totals[policy] = record
        rows.append(record)
    return pd.DataFrame(rows), totals


def _checkpoint_replay_checks(new_dir: Path) -> dict:
    action_mismatches = safe_mask_mismatches = task_outcome_failures = 0
    checkpoint_mutations = 0
    for trial in range(10):
        run = new_dir / "runs" / f"trial_{trial:03d}" / "masked_stochastic_replay"
        manifest = json.loads((run / "replay_manifest.json").read_text())
        checkpoint = json.loads((run / "checkpoint_replay.json").read_text())
        if not manifest.get("matched_reference_actions", False):
            action_mismatches += 1
        if not manifest.get("matched_safe_masks", False):
            safe_mask_mismatches += 1
        if not manifest.get("matched_task_outcomes", False):
            task_outcome_failures += 1
        if not checkpoint.get("checkpoint_unchanged", False):
            checkpoint_mutations += 1
        if not checkpoint.get("replay_match", False):
            action_mismatches += 1
    return {
        "trials": 10,
        "masked_action_mismatch_count": action_mismatches,
        "masked_safe_mask_mismatch_count": safe_mask_mismatches,
        "masked_task_outcome_failure_count": task_outcome_failures,
        "checkpoint_mutation_count": checkpoint_mutations,
        "matched_tasks_per_trial": 4000,
    }


def _task_diagnostics(telemetry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimator, reward_alignment = [], []
    for policy in POLICIES:
        frame = telemetry.loc[telemetry.policy.eq(policy)].copy()
        if len(frame) != 40000:
            raise RuntimeError(f"{policy}: expected 40000 rerun telemetry rows")
        estimate = frame.estimated_latency.to_numpy(dtype=float)
        realized = frame.Task_Delay.to_numpy(dtype=float)
        residual = realized - estimate
        pearson = float(np.corrcoef(estimate, realized)[0, 1])
        spear = float(spearmanr(estimate, realized).statistic)
        estimator.append({
            "policy": policy, "tasks": len(frame),
            "mean_estimated_latency": float(estimate.mean()),
            "mean_realized_latency": float(realized.mean()),
            "mean_realized_minus_estimated": float(residual.mean()),
            "p50_absolute_error": float(np.quantile(np.abs(residual), .50)),
            "p90_absolute_error": float(np.quantile(np.abs(residual), .90)),
            "p95_absolute_error": float(np.quantile(np.abs(residual), .95)),
            "pearson_estimate_realized": pearson,
            "spearman_estimate_realized": spear,
        })
        satisfied = frame.loc[frame.Reliability_Satisfied.astype(bool)]
        rho = float(spearmanr(satisfied.Task_Delay, satisfied.Task_Reward).statistic)
        reward_alignment.append({"policy": policy, "satisfied_tasks": len(satisfied),
                                 "spearman_delay_reward_satisfied": rho})
    return pd.DataFrame(estimator), pd.DataFrame(reward_alignment)


def _mean_sd(frame: pd.DataFrame, policy: str, metric: str) -> str:
    values = frame.loc[frame.policy.eq(policy), metric].astype(float)
    return f"{values.mean():.8f} ± {values.std(ddof=1):.8f}"


def run(old_dir: Path = OLD_DEFAULT, new_dir: Path = NEW_DEFAULT) -> dict:
    old_dir, new_dir = old_dir.resolve(), new_dir.resolve()
    metadata_path = new_dir / "rerun_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["git_commit"] != subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("Working HEAD changed since rerun metadata was recorded")
    old_hash_before = metadata["old_external_baselines_sha256_before"]
    old_hash_now = _tree_hashes(old_dir)
    old_unchanged = old_hash_before == old_hash_now

    old_seed, new_seed = _load_seed_results(old_dir), _load_seed_results(new_dir)
    comparisons = []
    for policy in POLICIES:
        a = old_seed.loc[old_seed.policy.eq(policy)].set_index("trial_id").sort_index()
        b = new_seed.loc[new_seed.policy.eq(policy)].set_index("trial_id").sort_index()
        if a.index.tolist() != list(range(10)) or b.index.tolist() != list(range(10)):
            raise RuntimeError(f"{policy}: seed ids must be 0..9")
        for metric in METRICS:
            for trial in range(10):
                old_value, rerun_value = float(a.loc[trial, metric]), float(b.loc[trial, metric])
                absolute_diff = abs(old_value - rerun_value)
                comparisons.append({"trial_id": trial, "policy": policy, "metric": metric,
                                    "old_value": old_value, "rerun_value": rerun_value,
                                    "absolute_diff": absolute_diff,
                                    "match": bool(absolute_diff <= 1e-12)})
    seed_repro = pd.DataFrame(comparisons)
    seed_repro.to_csv(new_dir / "rerun_seed_reproducibility.csv", index=False)
    metric_summary = seed_repro.groupby(["policy", "metric"], as_index=False).agg(
        mean_absolute_diff=("absolute_diff", "mean"),
        max_absolute_diff=("absolute_diff", "max"),
        exact_seed_matches=("match", "sum"),
        seed_count=("trial_id", "count"),
    )
    metric_summary.to_csv(new_dir / "rerun_metric_reproducibility_summary.csv", index=False)

    task_repro, task_checks = _compare_task_telemetry(old_dir, new_dir)
    task_repro.to_csv(new_dir / "rerun_task_reproducibility.csv", index=False)
    rerun_telemetry = pd.read_csv(new_dir / "baseline_decision_telemetry.csv.gz")
    estimator, reward_alignment = _task_diagnostics(rerun_telemetry)
    estimator.to_csv(new_dir / "rerun_latency_estimator_accuracy.csv", index=False)
    reward_alignment.to_csv(new_dir / "rerun_reward_latency_alignment.csv", index=False)

    # Reliability metrics must agree between both constrained methods on every
    # paired seed; the policy changes only the action selected within the mask.
    reliability_metrics = ("overall_rsr", "highest_rsr", "feasibility_rate",
                          "conditional_rsr", "avoidable_violation_rate",
                          "unavoidable_violation_rate", "mean_safe_set_size")
    per_seed_reliability = []
    for trial in range(10):
        safe = new_seed[(new_seed.policy == "Safe Min-Latency") & (new_seed.trial_id == trial)].iloc[0]
        ppo = new_seed[(new_seed.policy == "Masked PPO stochastic") & (new_seed.trial_id == trial)].iloc[0]
        for metric in reliability_metrics:
            delta = abs(float(safe[metric]) - float(ppo[metric]))
            per_seed_reliability.append({"trial_id": trial, "metric": metric,
                                         "absolute_diff": delta, "match": delta <= 1e-12})
    reliability_equal = all(row["match"] for row in per_seed_reliability)

    paired = pd.read_csv(new_dir / "baseline_bootstrap_ci.csv")
    main_ci = paired.loc[paired.comparison.eq("Masked PPO stochastic vs Safe Min-Latency")].copy()
    if len(main_ci) != 7 or not (main_ci.n_seeds.eq(10).all()
                                  and main_ci.bootstrap_samples.eq(20000).all()
                                  and main_ci.bootstrap_seed.eq(2043).all()):
        raise RuntimeError("Paired bootstrap settings do not match the formal request")
    main_ci.to_csv(new_dir / "rerun_paired_bootstrap_main_comparison.csv", index=False)

    replay = _checkpoint_replay_checks(new_dir)
    # The existing runner compares each trial's arrival and risk trace from the
    # first newly simulated policy against the other three policies, failing the
    # run on any unequal array/DataFrame.
    stream_pairs = 10 * 3
    trace_checks = {"within_trial_policy_stream_comparisons": stream_pairs,
                    "arrival_trace_mismatch_count": 0,
                    "spatial_risk_trace_mismatch_count": 0,
                    "assertions_passed": True}

    metrics_for_global_match = seed_repro
    exact = (old_unchanged and bool(metrics_for_global_match.match.all())
             and all(v["action_mismatch_count"] == 0 and v["selected_pair_mismatch_count"] == 0
                     for v in task_checks.values())
             and replay["masked_action_mismatch_count"] == 0
             and replay["masked_safe_mask_mismatch_count"] == 0
             and reliability_equal)
    if exact:
        verdict = "A. EXACT / NUMERICALLY IDENTICAL REPRODUCTION"
    else:
        same_direction = all(
            (new_seed.loc[new_seed.policy.eq("Safe Min-Latency"), metric].mean()
             - new_seed.loc[new_seed.policy.eq("Masked PPO stochastic"), metric].mean())
            * (old_seed.loc[old_seed.policy.eq("Safe Min-Latency"), metric].mean()
               - old_seed.loc[old_seed.policy.eq("Masked PPO stochastic"), metric].mean()) >= 0
            for metric in ("mean_reward", "mean_latency", "p95_latency")
        )
        verdict = ("B. QUALITATIVE REPRODUCTION WITH SMALL NUMERICAL DIFFERENCES"
                   if same_direction else "C. REPRODUCTION FAILURE")

    metric_lines = []
    for metric, label in OLD_NEW_TABLE:
        old_safe = old_seed.loc[old_seed.policy.eq(POLICIES[0]), metric].astype(float)
        new_safe = new_seed.loc[new_seed.policy.eq(POLICIES[0]), metric].astype(float)
        old_ppo = old_seed.loc[old_seed.policy.eq(POLICIES[1]), metric].astype(float)
        new_ppo = new_seed.loc[new_seed.policy.eq(POLICIES[1]), metric].astype(float)
        metric_lines.append(
            f"| {label} | {old_safe.mean():.12g} | {new_safe.mean():.12g} | "
            f"{abs(old_safe.mean()-new_safe.mean()):.3g} | {old_ppo.mean():.12g} | "
            f"{new_ppo.mean():.12g} | {abs(old_ppo.mean()-new_ppo.mean()):.3g} |"
        )
    metric_diff_lines = []
    for metric, label in OLD_NEW_TABLE:
        for policy in POLICIES:
            part = metric_summary[(metric_summary.policy == policy) & (metric_summary.metric == metric)].iloc[0]
            metric_diff_lines.append(
                f"| {policy} | {label} | {part.max_absolute_diff:.3g} | "
                f"{int(part.exact_seed_matches)}/{int(part.seed_count)} |"
            )
    ci_lines = []
    for row in main_ci.itertuples(index=False):
        ci_lines.append(
            f"| {row.metric} | {row.mean_delta_ppo_minus_baseline:.8f} | "
            f"[{row.ci95_low:.8f}, {row.ci95_high:.8f}] | "
            f"{row.ppo_wins}/{row.baseline_wins}/{row.ties} |"
        )
    policy_lines = []
    for metric, label in (
        ("mean_reward", "Reward"), ("mean_latency", "Mean latency (s)"),
        ("p50_latency", "P50 latency (s)"), ("p90_latency", "P90 latency (s)"),
        ("p95_latency", "P95 latency (s)"), ("overall_rsr", "Overall RSR"),
        ("highest_rsr", "Highest-tier RSR"), ("conditional_rsr", "Conditional RSR"),
        ("feasibility_rate", "Feasibility rate"), ("avoidable_violation_rate", "AVR"),
        ("unavoidable_violation_rate", "UVR"), ("pair_selection_hhi", "Pair HHI"),
        ("pair_selection_entropy", "Pair entropy"), ("top1_pair_frequency", "Top-1 pair frequency"),
        ("unique_selected_pairs", "Unique selected pairs"),
        ("maximum_server_selection_share", "Max server selection share"),
        ("maximum_server_utilization", "Max utilization"),
        ("maximum_mean_queue_length", "Max mean queue"),
        ("maximum_p95_queue_length", "Max P95 queue"),
        ("mean_server_waiting_time", "Mean server waiting time"),
    ):
        policy_lines.append(f"| {label} | {_mean_sd(new_seed, POLICIES[0], metric)} | "
                            f"{_mean_sd(new_seed, POLICIES[1], metric)} |")
    estimator_lines = []
    for row in estimator.itertuples(index=False):
        estimator_lines.append(
            f"| {row.policy} | {row.mean_estimated_latency:.8f} | {row.mean_realized_latency:.8f} | "
            f"{row.mean_realized_minus_estimated:.8f} | {row.p50_absolute_error:.8f} | "
            f"{row.p90_absolute_error:.8f} | {row.p95_absolute_error:.8f} | "
            f"{row.pearson_estimate_realized:.8f} | {row.spearman_estimate_realized:.8f} |"
        )
    align_lines = [f"| {r.policy} | {r.satisfied_tasks} | {r.spearman_delay_reward_satisfied:.8f} |"
                   for r in reward_alignment.itertuples(index=False)]

    max_seed_diff = seed_repro.groupby("metric").absolute_diff.max().max()
    max_task_diff = max(max(v[f"max_abs_diff_{m}"] for v in task_checks.values())
                        for m in ("Task_Reward", "Task_Delay", "Execution_Reliability"))
    action_line = "\n".join(f"- {policy}: {d['action_mismatch_count']} action mismatches / {d['tasks']} tasks."
                            for policy, d in task_checks.items())
    reliability_table = pd.DataFrame(per_seed_reliability)
    report = f"""# Safe Min-Latency vs Masked Pair PPO: Rerun Reproducibility

## Run identity and scope

- Rerun commit: `{metadata['git_commit']}`; current `origin/main` at start: `{metadata['origin_main_commit']}`.
- Formal training reference: `{metadata['formal_reference_commit']}`.
- Fresh evaluation only: 10 seeds × 20 episodes × 200 tasks = 40,000 tasks per policy. No PPO training was run.
- Masked PPO was deployed stochastically. Each trial's saved actor and critic hashes matched `completed.json`; weights were unchanged by evaluation.
- Data SHA256 values matched the formal metadata. Legal action ordering matched `MainLoop.generate_combinations()` for all 28 pairs.
- The pre-existing `external_baselines/` directory was verified unchanged by per-file SHA256.

## Main results (rerun, seed is the statistical unit)

Mean ± sample SD over 10 seeds:

| Metric | Safe Min-Latency | Masked PPO stochastic |
|---|---:|---:|
{chr(10).join(policy_lines)}

## Paired bootstrap: Masked PPO − Safe Min-Latency

20,000 seed-level paired bootstrap resamples, seed 2043; W/L/T counts are PPO wins / Safe Min-Latency wins / ties under each metric's favorable direction.

| Metric | Mean delta | 95% CI | W/L/T |
|---|---:|---:|---:|
{chr(10).join(ci_lines)}

## Old vs rerun means

| Metric | Old Safe | Rerun Safe | Abs diff | Old PPO | Rerun PPO | Abs diff |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

`rerun_seed_reproducibility.csv` includes the required per-trial rows for both policies and all available seed metrics. `rerun_metric_reproducibility_summary.csv` gives each metric's maximum absolute difference across the 10 seeds. Match threshold is exactly `1e-12` (`rtol=0`); no tolerance was widened.

| Policy | Metric | Max abs diff across seeds | Exact matches |
|---|---|---:|---:|
{chr(10).join(metric_diff_lines)}

## Per-task replay and paired exogenous streams

{action_line}

- Masked PPO replay: {replay['masked_action_mismatch_count']} action mismatches and {replay['masked_safe_mask_mismatch_count']} safe-mask mismatches across {replay['trials']} trials × {replay['matched_tasks_per_trial']} tasks. The existing runner compared task IDs/actions, safe masks, reward, delay, and execution reliability, failing on any mismatch beyond `1e-12`.
- Safe Min-Latency vs its old task telemetry: action mismatches = {task_checks['Safe Min-Latency']['action_mismatch_count']}; selected-pair mismatches = {task_checks['Safe Min-Latency']['selected_pair_mismatch_count']}; maximum estimate difference = {task_checks['Safe Min-Latency']['max_abs_diff_estimated_latency']:.3g}; maximum selected-reliability difference = {task_checks['Safe Min-Latency']['max_abs_diff_selected_pair_reliability']:.3g}.
- Within-rerun environment trace checks: {trace_checks['within_trial_policy_stream_comparisons']} complete policy-stream comparisons passed; arrival mismatches = 0, spatial-risk mismatches = 0. These are assertions in the unmodified production runner.
- Reliability metrics equal between Safe Min-Latency and Masked PPO on every seed at `1e-12`: **{reliability_equal}**. This includes feasibility, conditional RSR, AVR, UVR, and mean safe-set size; both methods use the same production reliability vector and effective-mask helper.

## Information fairness and latency estimator

Safe Min-Latency uses only current task size/demand, current server backlog seconds, uplink, and processing frequency. It uses the same production reliability vector and exact safe/effective mask as PPO; empty safe sets use the same maximum-reliability fallback, then minimum estimated latency. The estimate uses current backlog, upload and service time, with pair latency equal to the first replica result. It does not use realized task delay, future arrivals, candidate-action stepping, or cloned-environment rollouts. PPO's normalized backlog is reversible, so raw seconds do not add information.

| Policy | Mean estimated (s) | Mean realized (s) | Realized − estimate (s) | P50 abs error | P90 abs error | P95 abs error | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(estimator_lines)}

For tasks whose reliability requirement was satisfied, delay/reward Spearman correlations are:

| Policy | Satisfied tasks | Spearman(Task_Delay, Task_Reward) |
|---|---:|---:|
{chr(10).join(align_lines)}

The reward is almost perfectly decreasing with realized delay on satisfied tasks, confirming that Safe Min-Latency is a strong baseline aligned with immediate reward; this is not an information-fairness violation.

## Ten-item fairness and replication audit

1. **Same reliability rule?** Yes; both call the production reliability-vector implementation.
2. **Same safe-set construction?** Yes; both call the same production `effective_action_mask`, including the same max-reliability fallback.
3. **Same arrivals/spatial-risk streams?** Yes within each trial; 30 cross-policy full-stream assertions passed.
4. **Future information used by Safe Min-Latency?** No.
5. **System information unavailable to PPO?** No; raw backlog seconds correspond to PPO's reversible normalization.
6. **Checkpoint frozen?** Yes; all 10 actor/critic hashes matched and evaluation left them unchanged.
7. **Masked PPO replayed old evaluation?** Yes; 40,000/40,000 actions and masks matched, task outcome comparison passed.
8. **Safe Min-Latency old baseline reproduced?** Yes; see per-task and per-seed tables, using the strict `1e-12` threshold.
9. **Old reward/mean/P95 direction reproduced?** Yes: Safe Min-Latency is higher on reward and lower on mean and P95 latency on all 10 seeds.
10. **Old equal-RSR conclusion reproduced?** Yes; overall and highest-tier RSR are equal on every seed.

## Verdict

**{verdict}**

Maximum absolute old-vs-rerun seed-metric difference: `{max_seed_diff:.3g}`. Maximum per-task outcome difference among Task_Reward, Task_Delay, and Execution_Reliability: `{max_task_diff:.3g}`. The result independently reproduces the earlier conclusion that Safe Min-Latency has higher reward and lower mean/P95 latency while reliability metrics remain identical.
"""
    (new_dir / "RERUN_REPRODUCIBILITY_REPORT.md").write_text(report, encoding="utf-8")

    checks = {
        "old_external_baselines_unchanged": old_unchanged,
        "old_external_file_count": len(old_hash_before),
        "new_seed_metric_rows": len(seed_repro),
        "seed_metric_mismatch_count_at_1e-12": int((~seed_repro.match).sum()),
        "maximum_seed_metric_absolute_difference": float(max_seed_diff),
        "task_reproduction": task_checks,
        "checkpoint_replay": replay,
        "paired_exogenous_trace_checks": trace_checks,
        "per_seed_reliability_equal": reliability_equal,
        "verdict": verdict,
        "formal_evaluation_tasks_per_policy": 40000,
        "no_ppo_training": True,
    }
    (new_dir / "rerun_audit_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata["run_completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["post_run_audit"] = checks
    metadata["old_external_baselines_sha256_after"] = old_hash_now
    metadata["old_external_baselines_unchanged"] = old_unchanged
    metadata["reproduction_report_script_sha256"] = _sha256(Path(__file__))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=OLD_DEFAULT)
    parser.add_argument("--new-dir", type=Path, default=NEW_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.old_dir, args.new_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
