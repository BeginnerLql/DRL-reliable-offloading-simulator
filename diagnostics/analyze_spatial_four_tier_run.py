"""Reproduce Step 13 spatial fields and summarize the one full PPO run."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.params import params
from config.paths import DATA_DIR, RESULTS_DIR
from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    map_spatial_risk_to_effective_failure_rates,
    sample_spatial_risk_field,
)

OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "results" / "spatial_four_tier_run"
RESULT_WORKBOOK = Path(RESULTS_DIR) / "fixed_rate_results" / "ppo_results.xlsx"


def reconstruct_run():
    servers = pd.read_excel(Path(DATA_DIR) / "server_info.xlsx", sheet_name="Servers")
    tasks = pd.read_excel(Path(DATA_DIR) / "task_parameters.xlsx")
    servers = servers.sort_values("Server_ID").reset_index(drop=True)
    tasks = tasks.sort_values("Task_ID").reset_index(drop=True)
    levels = params.TASK_RELIABILITY_REQUIREMENT_LEVELS
    assert servers["Server_ID"].tolist() == list(range(1, params.NUM_SERVERS + 1))
    assert tasks["Task_ID"].tolist() == list(range(1, params.taskno + 1))
    assert all((tasks["Reliability_Requirement"] == level).sum() == 50 for level in levels)

    server_objects = [
        SimpleNamespace(
            server_id=int(row.Server_ID),
            latitude=float(row.Latitude),
            longitude=float(row.Longitude),
        )
        for row in servers.itertuples(index=False)
    ]
    server_ids, distances = build_distance_matrix(server_objects)
    assert server_ids == servers["Server_ID"].tolist()
    correlation = build_spatial_correlation_matrix(
        distances, params.SPATIAL_CORRELATION_LENGTH_KM
    )
    base_rates = servers["Base_Failure_Rate"].to_numpy(dtype=float)
    frequencies = servers["Processing_Frequency"].to_numpy(dtype=float)
    demands = tasks["Computation_Demand"].to_numpy(dtype=float)
    requirements = tasks["Reliability_Requirement"].to_numpy(dtype=float)
    pair_indices = np.asarray(list(combinations(range(params.NUM_SERVERS), 2)))
    assert len(pair_indices) == params.num_actions

    fields = np.empty((params.total_episodes, params.NUM_SERVERS))
    effective_rates = np.empty_like(fields)
    safe_counts = np.empty((params.total_episodes, params.taskno), dtype=np.int16)
    spatial_rng = np.random.default_rng(params.MASTER_SEED)
    for episode_index in range(params.total_episodes):
        field = sample_spatial_risk_field(correlation, rng=spatial_rng)
        effective = map_spatial_risk_to_effective_failure_rates(
            base_rates, field, params.SPATIAL_RISK_BETA_P
        )
        replica_reliability = np.exp(
            -demands[:, None] * effective[None, :] / frequencies[None, :]
        )
        pair_reliability = 1.0 - (
            1.0 - replica_reliability[:, pair_indices[:, 0]]
        ) * (1.0 - replica_reliability[:, pair_indices[:, 1]])
        fields[episode_index] = field
        effective_rates[episode_index] = effective
        safe_counts[episode_index] = (pair_reliability >= requirements[:, None]).sum(axis=1)

    return servers, tasks, distances, correlation, fields, effective_rates, safe_counts


def feasibility_table(tasks, safe_counts):
    requirements = tasks["Reliability_Requirement"].to_numpy(dtype=float)
    rows = []
    for label, mask in [("Overall", np.ones(len(tasks), dtype=bool))] + [
        (f"{level:g}", requirements == level)
        for level in params.TASK_RELIABILITY_REQUIREMENT_LEVELS
    ]:
        values = safe_counts[:, mask].ravel()
        total = len(values)
        zero = int((values == 0).sum())
        partial = int(((values > 0) & (values < params.num_actions)).sum())
        all_safe = int((values == params.num_actions).sum())
        rows.append({
            "Tier": label,
            "Samples": total,
            "Zero_Safe": zero,
            "Zero_Safe_Rate": zero / total,
            "Partial_Safe": partial,
            "Partial_Safe_Rate": partial / total,
            "All_28_Safe": all_safe,
            "All_28_Safe_Rate": all_safe / total,
            "Safe_Count_Min": int(values.min()),
            "Safe_Count_Median": float(np.median(values)),
            "Safe_Count_Mean": float(np.mean(values)),
            "Safe_Count_Max": int(values.max()),
            "Feasibility_Ceiling": 1.0 - zero / total,
        })
    return pd.DataFrame(rows)


def print_preflight(distances, correlation, fields, effective_rates, tasks, safe_counts):
    off_diagonal = correlation[np.triu_indices(params.NUM_SERVERS, 1)]
    base = pd.read_excel(Path(DATA_DIR) / "server_info.xlsx", sheet_name="Servers")
    ratio = effective_rates / base.sort_values("Server_ID")["Base_Failure_Rate"].to_numpy()
    print("Distance matrix (km), Server_ID 1..8:")
    print(pd.DataFrame(distances).round(6).to_string(index=False, header=False))
    print("Spatial correlation matrix, Server_ID 1..8:")
    print(pd.DataFrame(correlation).round(6).to_string(index=False, header=False))
    print("Off-diagonal correlation min/median/max:",
          *[f"{x:.6g}" for x in (off_diagonal.min(), np.median(off_diagonal), off_diagonal.max())])
    print("lambda_eff / base_lambda min/median/mean/P5/P95/max:",
          *[f"{x:.6g}" for x in (ratio.min(), np.median(ratio), ratio.mean(),
                                   np.percentile(ratio, 5), np.percentile(ratio, 95), ratio.max())])
    print("Feasibility across 100 episodes x 200 tasks:")
    print(feasibility_table(tasks, safe_counts).to_string(index=False))
    print("Spatial latent fields reconstructed:", fields.shape)


def fmt(value):
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def summarize_results(servers, tasks, distances, correlation, fields, effective_rates, safe_counts):
    logs = pd.read_excel(RESULT_WORKBOOK, sheet_name="Logs").sort_values("Episode")
    outcomes = pd.read_excel(RESULT_WORKBOOK, sheet_name="TaskResults")
    assert len(logs) == params.total_episodes
    assert logs["Episode"].tolist() == list(range(1, params.total_episodes + 1))
    assert len(outcomes) == params.total_episodes * params.taskno
    assert len(outcomes.columns) == 20
    assert not outcomes.duplicated(["episode", "task_id"]).any()
    assert outcomes.groupby("episode").size().eq(params.taskno).all()

    episode = outcomes["episode"].to_numpy(dtype=int) - 1
    task_id = outcomes["task_id"].to_numpy(dtype=int) - 1
    server_a = outcomes["server_A_id"].to_numpy(dtype=int) - 1
    server_b = outcomes["server_B_id"].to_numpy(dtype=int) - 1
    requirements = tasks["Reliability_Requirement"].to_numpy(dtype=float)
    demands = tasks["Computation_Demand"].to_numpy(dtype=float)
    frequencies = servers["Processing_Frequency"].to_numpy(dtype=float)
    raw_req = outcomes["reliability_requirement"].to_numpy(dtype=float)
    assert np.array_equal(raw_req, requirements[task_id])
    safe_for_outcome = safe_counts[episode, task_id]
    replica_a = np.exp(-effective_rates[episode, server_a] * demands[task_id] / frequencies[server_a])
    replica_b = np.exp(-effective_rates[episode, server_b] * demands[task_id] / frequencies[server_b])
    pair_reliability = 1.0 - (1.0 - replica_a) * (1.0 - replica_b)
    for saved_column, predicted in (
        ("replica_A_reliability", replica_a),
        ("replica_B_reliability", replica_b),
        ("pair_reliability", pair_reliability),
    ):
        assert np.allclose(outcomes[saved_column].to_numpy(dtype=float), predicted, rtol=1e-12, atol=1e-12)
    satisfied = outcomes["requirement_satisfied"].to_numpy(dtype=bool)
    assert np.array_equal(satisfied, pair_reliability >= raw_req)
    assert np.all(safe_for_outcome[satisfied] > 0)

    outcomes = outcomes.assign(
        safe_pair_count=safe_for_outcome,
        margin=outcomes["pair_reliability"] - raw_req,
        pair=[f"({min(a,b)},{max(a,b)})" for a, b in zip(server_a + 1, server_b + 1)],
    )
    feasibility = feasibility_table(tasks, safe_counts)
    tier_rows = []
    margin_rows = []
    latency_rows = []
    pair_rows = []
    for tier, frame in [("Overall", outcomes)] + [
        (f"{level:g}", outcomes.loc[outcomes["reliability_requirement"] == level])
        for level in params.TASK_RELIABILITY_REQUIREMENT_LEVELS
    ]:
        n = len(frame)
        feasible = frame["safe_pair_count"].gt(0)
        met = frame["requirement_satisfied"].astype(bool)
        tier_rows.append({
            "Tier": tier,
            "Outcomes": n,
            "Satisfied": int(met.sum()),
            "RSR": float(met.mean()),
            "Conditional_RSR": float(met[feasible].mean()) if feasible.any() else float("nan"),
            "Feasibility_Ceiling": float(feasible.mean()),
        })
        margin_rows.append({
            "Tier": tier,
            "Mean": float(frame["margin"].mean()),
            "Median": float(frame["margin"].median()),
            "P5": float(frame["margin"].quantile(0.05)),
            "Min": float(frame["margin"].min()),
        })
        latency_rows.append({
            "Tier": tier,
            "Mean_s": float(frame["task_latency"].mean()),
            "Median_s": float(frame["task_latency"].median()),
            "P95_s": float(frame["task_latency"].quantile(0.95)),
        })
        counts = frame["pair"].value_counts()
        pair_rows.append({"Tier": tier, "Most_Selected_Pair": counts.index[0],
                          "Selections": int(counts.iloc[0]), "Share": float(counts.iloc[0] / n)})

    all_pairs = [f"({a},{b})" for a, b in combinations(range(1, params.NUM_SERVERS + 1), 2)]
    pair_counts = outcomes["pair"].value_counts().reindex(all_pairs, fill_value=0)
    pair_summary = pd.DataFrame({
        "Pair": pair_counts.index,
        "Selections": pair_counts.to_numpy(),
        "Share": pair_counts.to_numpy() / len(outcomes),
    }).sort_values(["Selections", "Pair"], ascending=[False, True])
    pair_hhi = float(np.square(pair_summary["Share"]).sum())

    ratio = effective_rates / servers["Base_Failure_Rate"].to_numpy(dtype=float)[None, :]
    episode_summary = pd.DataFrame({
        "Episode": np.arange(1, params.total_episodes + 1),
        "Mean_Lambda_Ratio": ratio.mean(axis=1),
        "Zero_Safe_Rate": (safe_counts == 0).mean(axis=1),
        "Partial_Safe_Rate": ((safe_counts > 0) & (safe_counts < params.num_actions)).mean(axis=1),
        "RSR": outcomes.groupby("episode")["requirement_satisfied"].mean().reindex(range(1, params.total_episodes + 1)).to_numpy(),
        "task_Avg_Delay": logs["task_Avg_Delay"].to_numpy(dtype=float),
    })
    for index in range(params.NUM_SERVERS):
        episode_summary[f"Z_{index + 1}"] = fields[:, index]
        episode_summary[f"Lambda_Eff_{index + 1}"] = effective_rates[:, index]

    feasibility_episode_tier_rows = []
    for episode_index in range(params.total_episodes):
        for level in params.TASK_RELIABILITY_REQUIREMENT_LEVELS:
            counts = safe_counts[episode_index, requirements == level]
            feasibility_episode_tier_rows.append({
                "Episode": episode_index + 1,
                "Tier": level,
                "Samples": len(counts),
                "Zero_Safe": int((counts == 0).sum()),
                "Partial_Safe": int(((counts > 0) & (counts < params.num_actions)).sum()),
                "All_28_Safe": int((counts == params.num_actions).sum()),
                "Mean_Safe_Pairs": float(counts.mean()),
            })

    correlations = []
    for metric in ("Zero_Safe_Rate", "Partial_Safe_Rate", "RSR", "task_Avg_Delay"):
        correlations.append({
            "Severity_vs": metric,
            "Pearson": episode_summary["Mean_Lambda_Ratio"].corr(episode_summary[metric], method="pearson"),
            "Spearman": episode_summary["Mean_Lambda_Ratio"].corr(episode_summary[metric], method="spearman"),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    episode_summary.to_csv(OUTPUT_DIR / "spatial_episode_summary.csv", index=False)
    pd.DataFrame(feasibility_episode_tier_rows).to_csv(
        OUTPUT_DIR / "feasibility_by_episode_tier.csv", index=False
    )
    pair_summary.to_csv(OUTPUT_DIR / "pair_selection_summary.csv", index=False)

    off_diagonal = correlation[np.triu_indices(params.NUM_SERVERS, 1)]
    report = f"""# Step 13: four-tier reliability and spatial PPO diagnostic

## Configuration and provenance

- Run: `python Project_main.py`, {params.total_episodes} episodes × {params.taskno} tasks, unconstrained PPO, {params.NUM_SERVERS} servers, {params.num_states}-dimensional observation.
- Raw reliability requirements: `{params.TASK_RELIABILITY_REQUIREMENT_LEVELS}`; exactly 50 tasks per tier in the frozen task profile and 5,000 outcomes per tier.
- Observation reliability encoding: `0.9 → 0`, `0.99 → 1/3`, `0.999 → 2/3`, `0.9999 → 1`. Feasibility, RSR, and margin use raw requirements.
- Spatial correlation length: {params.SPATIAL_CORRELATION_LENGTH_KM} km. **SPATIAL_RISK_BETA_P = 0.5 is a temporary exploratory value.** It is not a literature-calibrated final parameter.
- One latent physical risk field is sampled per episode using `MASTER_SEED = {params.MASTER_SEED}`. It modulates effective failure intensities; pair reliability retains the independent-replica formula. The spatial fields are reproducible; PPO policy and workload RNG are unchanged by this step.
- TaskResults retains its 20-column schema. Chosen-pair reliability snapshots agree with the independently reconstructed episode fields for all {len(outcomes):,} outcomes.

## Spatial topology

Off-diagonal correlation (min / median / max): **{fmt(off_diagonal.min())} / {fmt(np.median(off_diagonal))} / {fmt(off_diagonal.max())}**.

Distance matrix in km, rows and columns Server_ID 1..8:

```
{pd.DataFrame(distances).round(6).to_string(index=False, header=False)}
```

Spatial correlation matrix, rows and columns Server_ID 1..8:

```
{pd.DataFrame(correlation).round(6).to_string(index=False, header=False)}
```

Across all episode × server values, `lambda_eff/base_lambda` (min / median / mean / P5 / P95 / max): **{fmt(ratio.min())} / {fmt(np.median(ratio))} / {fmt(ratio.mean())} / {fmt(np.percentile(ratio, 5))} / {fmt(np.percentile(ratio, 95))} / {fmt(ratio.max())}**. Effective failure intensity (min / median / max, s⁻¹): **{fmt(effective_rates.min())} / {fmt(np.median(effective_rates))} / {fmt(effective_rates.max())}**.

## Episode-specific feasibility

The audit evaluates all {params.total_episodes * params.taskno * params.num_actions:,} task–episode–pair combinations using each episode's effective failure rates. `Partial_Safe` means 1–27 safe actions. The feasibility ceiling is the fraction with at least one safe action.

{markdown_table(feasibility)}

## Observed reliability and latency

RSR is the fraction of chosen actions satisfying the raw requirement. Conditional RSR excludes instances with no safe action.

{markdown_table(pd.DataFrame(tier_rows))}

Task latency in seconds:

{markdown_table(pd.DataFrame(latency_rows))}

Chosen-pair reliability margin (`pair_reliability - raw_requirement`):

{markdown_table(pd.DataFrame(margin_rows))}

## Reward and action concentration

Episode Reward (mean / last / best / worst): **{fmt(logs['Episode Reward'].mean())} / {fmt(logs['Episode Reward'].iloc[-1])} / {fmt(logs['Episode Reward'].max())} / {fmt(logs['Episode Reward'].min())}**. Reward remains `-task_latency`; the Logs sheet retains `task_Avg_Delay`.

Pair HHI: **{fmt(pair_hhi)}**. Most selected pair: **{pair_rows[0]['Most_Selected_Pair']}**, {pair_rows[0]['Selections']} selections ({fmt(pair_rows[0]['Share'])} share). Top five pairs:

{markdown_table(pair_summary.head(5))}

Most selected pair by reliability tier:

{markdown_table(pd.DataFrame(pair_rows[1:]))}

## Spatial severity and episode outcomes

Correlations use 100 episode-level observations and are descriptive. Severity is the episode mean `lambda_eff/base_lambda`.

{markdown_table(pd.DataFrame(correlations))}

Supporting CSVs contain the 100 reconstructed spatial fields and effective rates, episode/tier feasibility, and all 28 pair selection counts.

## Figures

![Feasibility and RSR by tier](STEP13_FEASIBILITY_RSR.png)

![PPO performance and diagnostic relationships](STEP13_PERFORMANCE.png)

![Spatial latent-risk correlation](STEP13_SPATIAL_CORRELATION.png)
"""
    (OUTPUT_DIR / "STEP13_REPORT.md").write_text(report, encoding="utf-8")
    print("Saved", OUTPUT_DIR / "STEP13_REPORT.md")
    print("Overall RSR:", fmt(tier_rows[0]["RSR"]))
    print("Overall conditional RSR:", fmt(tier_rows[0]["Conditional_RSR"]))
    print("Pair HHI:", fmt(pair_hhi))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Audit feasibility before the full run")
    arguments = parser.parse_args()
    servers, tasks, distances, correlation, fields, effective_rates, safe_counts = reconstruct_run()
    if arguments.preflight:
        print_preflight(distances, correlation, fields, effective_rates, tasks, safe_counts)
    else:
        summarize_results(servers, tasks, distances, correlation, fields, effective_rates, safe_counts)


if __name__ == "__main__":
    main()
