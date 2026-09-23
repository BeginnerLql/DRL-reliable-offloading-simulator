"""Tie-aware reward-landscape audit of the existing 1000-state Oracle sweep.

Reads the persisted 112,000 Oracle rows. Only the absent original-policy
R_req=0.9 stratum is supplemented with the validated EpisodeTrace,
Task.initialize_reliability_evaluation and MainLoop.calcReward replay helpers.
No training or production simulator logic is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diagnostics.requirement_conditioning_diagnostic import (
    FORMAL, OBSERVED_HASHES, ROOT, REQS, EpisodeTrace, episode_rates,
    file_hash, simulator_reward,
)
from Project_main import build_pair_correlations

EXACT_TOL = 1e-10
EPSILONS = (1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1)
EXPECTED_STATES = 1000
EXPECTED_PAIRS = 28


def distribution(values: pd.Series, prefix: str = "") -> dict:
    values = pd.Series(values, dtype=float)
    return {f"{prefix}mean": float(values.mean()),
            f"{prefix}median": float(values.median()),
            f"{prefix}p10": float(values.quantile(.10)),
            f"{prefix}p25": float(values.quantile(.25)),
            f"{prefix}p75": float(values.quantile(.75)),
            f"{prefix}p90": float(values.quantile(.90)),
            f"{prefix}p95": float(values.quantile(.95)),
            f"{prefix}max": float(values.max())}


def group_landscape(group: pd.DataFrame, tol: float = EXACT_TOL) -> tuple[dict, set[int], dict]:
    """Classify one fixed state/requirement without relying on action order."""
    rewards = group.reward_total.to_numpy(dtype=float)
    actions = group.action_index.to_numpy(dtype=int)
    best = float(rewards.max())
    regrets = best - rewards
    if np.any(regrets < -tol):
        raise ValueError("Negative regret beyond numerical tolerance")
    optimal = set(actions[regrets <= tol])
    distinct = regrets[regrets > tol]
    gap = float(distinct.min()) if len(distinct) else np.nan
    sizes = {epsilon: int(np.count_nonzero(regrets <= epsilon + tol)) for epsilon in EPSILONS}
    return ({"r_star": best, "exact_optimal_set_size": len(optimal),
             "second_distinct_gap": gap, "all_actions_tied": not len(distinct),
             **{f"near_size_{epsilon:g}": value for epsilon, value in sizes.items()}},
            optimal, {int(a): float(max(0.0, r)) for a, r in zip(actions, regrets)})


def load_recorded_actions() -> pd.DataFrame:
    frames = []
    for trial in range(10):
        path = FORMAL / "runs" / f"trial_{trial:03d}" / "pair_scoring" / "evaluation_task_assignments.csv"
        frame = pd.read_csv(path)
        frame["Trial_ID"] = trial
        frame["state_id"] = [f"trial{trial:02d}_episode{int(ep):02d}_task{int(tid):03d}"
                             for ep, tid in zip(frame.episode, frame.task_id)]
        frames.append(frame[["state_id", "Trial_ID", "episode", "task_id", "Reliability_Requirement",
                             "action_index", "Primary", "Backup", "Task_Reward", "Task_Delay"]])
    recorded = pd.concat(frames, ignore_index=True)
    assert len(recorded) == 40000 and recorded.state_id.is_unique
    return recorded.set_index("state_id")


def supplement_missing_low_requirement(output: Path) -> pd.DataFrame:
    """Add 200 distinct actual-policy .9 states, without rerunning training."""
    for name, expected in OBSERVED_HASHES.items():
        actual = file_hash(ROOT / "data" / name)
        if actual != expected:
            raise RuntimeError(f"Formal-run input changed: {name}: {actual} != {expected}")
    servers = pd.read_excel(ROOT / "data/server_info.xlsx").set_index("Server_ID").sort_index()
    tasks = pd.read_excel(ROOT / "data/task_parameters.xlsx").set_index("Task_ID").sort_index()
    low_ids = tasks.index[np.isclose(tasks.Reliability_Requirement, .9)].to_numpy(dtype=int)
    assert len(low_ids) == 50
    # Cover the task-ID horizon rather than repeatedly using one task.
    sampled_ids = low_ids[np.linspace(0, len(low_ids) - 1, 20).round().astype(int)]
    assert len(set(sampled_ids)) == 20
    pairs, _ = build_pair_correlations()
    pairs = list(pairs)
    assert len(pairs) == EXPECTED_PAIRS
    seed_plan = pd.read_csv(FORMAL / "seed_plan.csv").set_index("Trial_ID")
    rows = []
    for trial in range(10):
        frame = pd.read_csv(FORMAL / "runs" / f"trial_{trial:03d}" / "pair_scoring" / "evaluation_task_assignments.csv")
        rates_by_ep = episode_rates(int(seed_plan.loc[trial, "Eval_Spatial_Seed"]), 20, servers)
        for ep, ep_frame in frame.groupby("episode", sort=True):
            task_id = int(sampled_ids[(int(ep) - 1) % 20])
            trace = EpisodeTrace(ep_frame, tasks, servers)
            observed = trace.frame.loc[task_id]
            assert np.isclose(float(observed.Reliability_Requirement), .9, rtol=0, atol=1e-12)
            actual_pair = (int(observed.Primary), int(observed.Backup))
            assert pairs[int(observed.action_index)] == actual_pair
            rates = rates_by_ep[int(ep)]
            state_id = f"trial{trial:02d}_episode{int(ep):02d}_task{task_id:03d}"
            for action, pair in enumerate(pairs):
                delay = trace.candidate_delay(task_id, pair)
                reward = simulator_reward(task_id, float(tasks.loc[task_id, "Computation_Demand"]),
                                          .9, pair, delay, rates, servers)["reward_total"]
                rows.append({"state_id": state_id, "R_req": .9, "action_index": action,
                             "reward_total": reward, "source": "supplemental_actual_requirement_0.9"})
                if action == int(observed.action_index):
                    assert abs(delay - float(observed.Task_Delay)) <= 1e-8
                    assert abs(reward - float(observed.Task_Reward)) <= 1e-8
    supplement = pd.DataFrame(rows)
    assert len(supplement) == 200 * EXPECTED_PAIRS
    supplement.to_csv(output / "policy_regret_low_requirement_supplement.csv", index=False)
    return supplement


def make_plots(out: Path, exact_summary: pd.DataFrame, near: pd.DataFrame,
               gaps: pd.DataFrame, policy_detail: pd.DataFrame, overlap: pd.DataFrame):
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7, 4))
    for req, group in exact_summary.groupby("R_req"):
        ax.hist(group.exact_optimal_set_size, bins=np.arange(.5, 29.5, 1), alpha=.45, label=f"{req:g}")
    ax.set(xlabel="Number of exact-optimal pairs (of 28)", ylabel="States")
    ax.legend(); fig.savefig(out / "optimal_set_size_distribution.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for req, group in near.groupby("R_req"):
        ax.plot([str(x) for x in group.epsilon], group.mean_size, marker="o", label=f"{req:g}")
    ax.set(xlabel="Absolute reward tolerance epsilon", ylabel="Mean near-optimal set size")
    ax.legend(); fig.savefig(out / "near_optimal_set_size.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for req, group in gaps[~gaps.all_actions_tied].groupby("R_req"):
        ax.hist(group.second_distinct_gap, bins=np.logspace(-10, 3, 50), alpha=.4, label=f"{req:g}")
    ax.set_xscale("log")
    ax.set(xlabel="Best minus second distinct reward (log scale)", ylabel="States")
    ax.legend(); fig.savefig(out / "oracle_reward_gap_distribution.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for req, group in policy_detail.groupby("R_req"):
        ax.hist(group.regret, bins=np.linspace(0, max(1, policy_detail.regret.quantile(.99)), 45),
                alpha=.4, label=f"{req:g}")
    ax.set(xlabel="Matched-state policy regret", ylabel="States")
    ax.legend(); fig.savefig(out / "policy_regret_distribution.png"); plt.close(fig)

    matrix = overlap.pivot(index="R_req_A", columns="R_req_B", values="mean_jaccard").reindex(index=REQS, columns=REQS)
    fig, ax = plt.subplots(figsize=(5.5, 4.7))
    im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(4), [str(x) for x in REQS]); ax.set_yticks(range(4), [str(x) for x in REQS])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{matrix.iloc[i,j]:.2f}", ha="center", va="center", color="white")
    ax.set(xlabel="R_req B", ylabel="R_req A")
    fig.colorbar(im, ax=ax, label="Mean Jaccard similarity")
    fig.savefig(out / "oracle_set_jaccard_heatmap.png"); plt.close(fig)


def add_report(out: Path, exact_by: pd.DataFrame, near: pd.DataFrame, gaps_by: pd.DataFrame,
               policy_by: pd.DataFrame, pair78_by: pd.DataFrame, overlap: pd.DataFrame,
               ties_by: pd.DataFrame, membership: pd.DataFrame):
    def table(frame: pd.DataFrame) -> str:
        names = list(frame.columns)
        text = ["| " + " | ".join(names) + " |", "|" + "|".join(["---"]*len(names)) + "|"]
        for row in frame.itertuples(index=False, name=None):
            text.append("| " + " | ".join(f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
        return "\n".join(text)
    x = exact_by[np.isclose(exact_by.R_req, .9)].iloc[0]
    y = exact_by[np.isclose(exact_by.R_req, .9999)].iloc[0]
    low_near = near[np.isclose(near.R_req, .9) & np.isclose(near.epsilon, .01)].iloc[0]
    high_near = near[np.isclose(near.R_req, .9999) & np.isclose(near.epsilon, .01)].iloc[0]
    p78h = pair78_by[np.isclose(pair78_by.R_req, .9999)].iloc[0]
    all_overlap = overlap[np.isclose(overlap.R_req_A,.9) & np.isclose(overlap.R_req_B,.9999)].iloc[0]
    pactual = policy_by[np.isclose(policy_by.R_req,.9999)].iloc[0]
    tie_low = ties_by[np.isclose(ties_by.R_req,.9)].iloc[0]
    tie_high = ties_by[np.isclose(ties_by.R_req,.9999)].iloc[0]
    lines = ["# Reward Landscape and Oracle Regret", "",
        "本节基于原 1,000 个固定状态、每状态四档需求及 28 个合法 pair 的既有 Oracle CSV；exact tie 容差为绝对 reward 差 ≤1e-10（正式结果 reward 重放最大误差 7.11e-14），无相对容差。Near-optimal 使用绝对 reward epsilon。Oracle 是单任务即时收益，不是长期 PPO return。", "",
        f"1. **Argmax 变化主要含 tie-breaking artifact。** 之前按固定索引选单一 argmax 的 .9→.9999 变化率 80.9%，但最优集合不相交仅 7.1%；集合 Jaccard 均值 {all_overlap.mean_jaccard:.3f}、有交集比例 {all_overlap.any_overlap_rate:.1%}。", "",
        f"2. **Exact optimal set 大小。** .9 档均值 {x['mean']:.2f}/28、median {x['median']:.0f}；.9999 档均值 {y['mean']:.2f}/28、median {y['median']:.0f}。四档详表：", "", table(exact_by), "",
        f"3. **Near-optimal set。** epsilon=0.01 时，.9 档均值 {low_near.mean_size:.2f}/28、.9999 档 {high_near.mean_size:.2f}/28。注意绝对 0.01 reward 容差很小；六种 epsilon 详见 near_optimal_set_size.csv。", "",
        "4. **Best-vs-second-distinct gap。** 排除所有动作完全并列的状态后才统计 second-distinct；这些全并列状态另列计数。gap 不应与原 best-vs-second（允许并列，常为 0）混淆。", "", table(gaps_by), "",
        "5. **实际 Pair PPO regret。** 原 1,000 状态的实际任务等级没有 0.9（所抽 Task_ID 覆盖 .99:200、.999:200、.9999:600）。为得到四档，复用既有 EpisodeTrace/可靠性/奖励函数，额外重放 200 个真实 0.9 任务状态；这 200 个仅用于实际 policy regret，不混入原 1,000 状态的 Oracle 景观/集合分析。每个实际动作的重放 delay、reward 均与正式日志匹配。", "", table(policy_by), "",
        f"最高档实际选中动作的无条件 mean regret {pactual.mean_regret:.3f}、median {pactual.median_regret:.3f}；严格次优动作中的 mean regret {pactual.mean_regret_given_suboptimal:.3f}。其 exact optimal rate {pactual.exact_optimal_rate:.1%}，在 epsilon=0.01 内的比例 {pactual.near_0_01_rate:.1%}。因此需看 regret 而不能仅看‘严格次优’频率。", "",
        f"6. **(7,8) 不是稳定 near-optimal shortcut。** 最高档 exact-optimal rate {p78h.exact_optimal_rate:.1%}、epsilon=0.01 rate {p78h.near_0_01_rate:.1%}、mean regret {p78h.mean_regret:.3f}；四档详表：", "", table(pair78_by), "",
        "7. **Reward 平台来源。** 当前没有 deadline；也没有 round reward 或直接 rho/correlation/energy 分量。可靠性门槛把满足阈值后的 reliability component 固定为 0，此时只由首个完成副本的 latency 决定 reward。不同 pair 共享最快副本时 latency 完全相同，即使另一副本的 reliability/rho 不同也得到 exact tie。未满足阈值时走另一延迟分支并有连续 log10 violation penalty；该分支的 `max(-3*delay,-3)` 可在 delay<1s 时截断，但不能解释主要平台。binary threshold 本身不是一个固定值 penalty。", "",
        f"在并列最优状态中，.9 档同延迟且全满足阈值的比例 {tie_low.same_latency_all_satisfied_rate:.1%}；.9999 档 {tie_high.same_latency_all_satisfied_rate:.1%}。最优 pair 共有至少一台服务器的比例依次为 {tie_low.common_server_rate:.1%}、{tie_high.common_server_rate:.1%}。这些集合内可靠性仍有差别的比例依次为 {tie_low.reliability_varies_rate:.1%}、{tie_high.reliability_varies_rate:.1%}；rho 有差别的比例依次为 {tie_low.rho_varies_rate:.1%}、{tie_high.rho_varies_rate:.1%}。见 reward_tie_source_summary.csv / reward_tie_sources.csv。", "",
        "8. **需求间集合重叠。** 四档两两 Jaccard 见 oracle_set_overlap.csv / heatmap；(7,8) 同时属于多少档 exact / epsilon=0.01 集合见下表。", "", table(membership), "",
        "**综合判断：B 最符合数据，但有关键限定。** 高档阈值改变了不少候选 pair 的 reward，故 C 不成立；真正强制最优动作切换的原样本仅 7.1%，大量 exact tie 与需求间集合重叠，故 A 过强。B 描述的是‘某些固定 pair 可跨需求保持最优或近优’，**不能套到 (7,8)**：它的 regret 和 near-optimal 率表明当前策略的全局偏好本身仍有明显即时 reward 代价。策略与 Oracle 样本并非逐状态相同，只有实际动作 regret 采用逐状态匹配。", ""]
    report = out / "DIAGNOSIS.md"
    original = report.read_text(encoding="utf-8")
    marker = "# Reward Landscape and Oracle Regret"
    if marker in original:
        original = original[:original.index(marker)].rstrip() + "\n"
    report.write_text(original.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((output / "oracle_summary.json").read_text())
    replay_error = float(summary["replay_max_abs_reward_error"])
    assert replay_error < EXACT_TOL / 100
    original = pd.read_csv(output / "oracle_pair_results.csv")
    assert len(original) == EXPECTED_STATES * len(REQS) * EXPECTED_PAIRS
    assert original.groupby(["state_id", "R_req"]).size().eq(EXPECTED_PAIRS).all()
    assert original.state_id.nunique() == EXPECTED_STATES
    assert original.groupby("state_id").state_context_sha256.nunique().eq(1).all()
    assert original.groupby(["state_id", "action_index"]).latency.nunique().eq(1).all()
    assert original.groupby(["state_id", "action_index"]).pair_reliability.nunique().eq(1).all()
    assert np.isfinite(original.select_dtypes(include="number").to_numpy()).all()
    recorded = load_recorded_actions()
    sets = {}
    regrets = {}
    exact_rows = []
    gap_rows = []
    near_rows = []
    tie_rows = []
    for (sid, req), g in original.groupby(["state_id", "R_req"], sort=True):
        req = float(req)
        item, optimal, action_regrets = group_landscape(g)
        key = (sid, req)
        sets[key] = optimal
        regrets[key] = action_regrets
        exact_rows.append({"state_id": sid, "R_req": req, "exact_optimal_set_size": item["exact_optimal_set_size"]})
        gap_rows.append({"state_id": sid, "R_req": req, "second_distinct_gap": item["second_distinct_gap"],
                         "all_actions_tied": item["all_actions_tied"], "exact_optimal_set_size": len(optimal)})
        for epsilon in EPSILONS:
            near_rows.append({"state_id": sid, "R_req": req, "epsilon": epsilon,
                              "near_optimal_set_size": item[f"near_size_{epsilon:g}"]})
        if len(optimal) >= 2:
            tied = g[g.action_index.isin(optimal)]
            def span(column):
                return float(tied[column].max() - tied[column].min())
            all_satisfied = bool(tied.reliability_satisfied.all())
            all_violated = bool((~tied.reliability_satisfied).all())
            common_servers = set.intersection(*[
                {int(row.primary_server), int(row.backup_server)} for row in tied.itertuples()
            ])
            tie_rows.append({"state_id": sid, "R_req": req, "exact_optimal_set_size": len(optimal),
                "common_server_in_all_optimal_pairs": bool(common_servers),
                "latency_span": span("latency"), "reliability_span": span("pair_reliability"),
                "joint_failure_span": span("joint_failure_probability"), "rho_span": span("rho_jk"),
                "delay_reward_span": span("reward_delay_component"),
                "reliability_reward_span": span("reward_reliability_component"),
                "gate_span": span("reward_reliability_gate_component"),
                "violation_penalty_span": span("reward_reliability_penalty_component"),
                "correlation_component_span": span("reward_correlation_component"),
                "energy_component_span": span("reward_energy_component"),
                "all_satisfied": all_satisfied, "all_violated": all_violated,
                "same_latency_all_satisfied": all_satisfied and span("latency") <= EXACT_TOL,
                "same_latency_all_violated": all_violated and span("latency") <= EXACT_TOL,
                "reliability_varies": span("pair_reliability") > EXACT_TOL,
                "rho_varies": span("rho_jk") > EXACT_TOL,
                "any_delay_lt_1_sec": bool((tied.latency < 1).any()),
                "any_different_violation_indicator": not (all_satisfied or all_violated)})
    exact = pd.DataFrame(exact_rows)
    gaps = pd.DataFrame(gap_rows)
    near_detail = pd.DataFrame(near_rows)
    ties = pd.DataFrame(tie_rows)
    exact.to_csv(output / "optimal_set_size_per_state.csv", index=False)
    gaps.to_csv(output / "oracle_reward_gap.csv", index=False)
    near_detail.to_csv(output / "near_optimal_set_size_per_state.csv", index=False)
    ties.to_csv(output / "reward_tie_sources.csv", index=False)
    exact_by = []
    gap_by = []
    near_by = []
    ties_by = []
    for req in REQS:
        e = exact[np.isclose(exact.R_req, req)].exact_optimal_set_size
        exact_by.append({"R_req": req, "States": len(e), **distribution(e),
            "p_size_1":float((e==1).mean()), "p_size_gt_1":float((e>1).mean()),
            "p_size_ge_5":float((e>=5).mean()), "p_size_ge_10":float((e>=10).mean())})
        group = gaps[np.isclose(gaps.R_req, req)]
        nonempty = group.second_distinct_gap.dropna()
        gap_by.append({"R_req":req,"States":len(group),"all_actions_tied_count":int(group.all_actions_tied.sum()),
                       "states_with_second_distinct":len(nonempty), **distribution(nonempty)})
        for epsilon in EPSILONS:
            n = near_detail[np.isclose(near_detail.R_req,req) & np.isclose(near_detail.epsilon,epsilon)].near_optimal_set_size
            near_by.append({"R_req":req,"epsilon":epsilon,"States":len(n),
                            "mean_size":float(n.mean()),"median_size":float(n.median()),
                            "p25_size":float(n.quantile(.25)),"p75_size":float(n.quantile(.75)),
                            "p90_size":float(n.quantile(.90))})
        group = ties[np.isclose(ties.R_req,req)]
        ties_by.append({"R_req":req,"states_with_ties":len(group),
            "same_latency_all_satisfied_rate":float(group.same_latency_all_satisfied.mean()),
            "same_latency_all_violated_rate":float(group.same_latency_all_violated.mean()),
            "reliability_varies_rate":float(group.reliability_varies.mean()),
            "rho_varies_rate":float(group.rho_varies.mean()),
            "common_server_rate":float(group.common_server_in_all_optimal_pairs.mean()),
            "mixed_violation_indicator_rate":float(group.any_different_violation_indicator.mean()),
            "any_delay_lt_1_sec_rate":float(group.any_delay_lt_1_sec.mean()),
            "median_latency_span":float(group.latency_span.median()),
            "median_reliability_span":float(group.reliability_span.median()),
            "median_rho_span":float(group.rho_span.median())})
    exact_by = pd.DataFrame(exact_by); exact_by.to_csv(output / "optimal_set_size_by_requirement.csv", index=False)
    gap_by = pd.DataFrame(gap_by); gap_by.to_csv(output / "oracle_reward_gap_summary.csv", index=False)
    near_by = pd.DataFrame(near_by); near_by.to_csv(output / "near_optimal_set_size.csv", index=False)
    ties_by = pd.DataFrame(ties_by); ties_by.to_csv(output / "reward_tie_source_summary.csv", index=False)

    # Original action is available only at each task's actual requirement.
    # The original five task IDs contain no 0.9 task, so add that stratum only.
    original_actual = recorded.loc[exact.state_id.unique()]
    actual_levels = np.unique(original_actual.Reliability_Requirement.to_numpy())
    assert len(original_actual) == EXPECTED_STATES and not np.any(np.isclose(actual_levels,.9))
    supplement = supplement_missing_low_requirement(output)
    for (sid, req), g in supplement.groupby(["state_id", "R_req"]):
        item, optimal, action_regrets = group_landscape(g)
        sets[(sid, float(req))] = optimal
        regrets[(sid, float(req))] = action_regrets
    actual_ids = sorted(set(exact.state_id) | set(supplement.state_id))
    policy_rows = []
    original_lookup = original.set_index(["state_id", "R_req", "action_index"])
    original_state_ids = set(exact.state_id)
    for sid in actual_ids:
        observed = recorded.loc[sid]
        req = float(observed.Reliability_Requirement)
        action = int(observed.action_index)
        pair = (int(observed.Primary), int(observed.Backup))
        key = (sid, req)
        assert key in regrets and action in regrets[key]
        if sid in original_state_ids:
            oracle_row = original_lookup.loc[(sid, req, action)]
            assert pair == (int(oracle_row.primary_server), int(oracle_row.backup_server))
            assert abs(float(oracle_row.reward_total) - float(observed.Task_Reward)) <= 1e-8
            assert abs(float(oracle_row.latency) - float(observed.Task_Delay)) <= 1e-8
            source = "original_1000_states"
        else:
            source = "supplemental_0.9_states"
        regret = regrets[key][action]
        policy_rows.append({"state_id":sid,"R_req":req,"action_index":action,"pair":str(pair),
                            "regret":regret,"exact_optimal":regret<=EXACT_TOL,
                            "near_1e-3":regret<=1e-3+EXACT_TOL,
                            "near_1e-2":regret<=1e-2+EXACT_TOL,"source":source})
    policy_detail = pd.DataFrame(policy_rows)
    assert len(policy_detail)==1200 and policy_detail.state_id.is_unique
    policy_detail.to_csv(output / "policy_regret_per_state.csv", index=False)
    policy_by=[]
    for req, g in policy_detail.groupby("R_req"):
        suboptimal = g.loc[g.regret > EXACT_TOL, "regret"]
        row={"R_req":req,"States":len(g),"source":g.source.iloc[0],
             "mean_regret":float(g.regret.mean()),"median_regret":float(g.regret.median()),
             "strictly_suboptimal_rate":float(len(suboptimal)/len(g)),
             "mean_regret_given_suboptimal":float(suboptimal.mean()),
             "median_regret_given_suboptimal":float(suboptimal.median()),
             "p75_regret":float(g.regret.quantile(.75)),"p90_regret":float(g.regret.quantile(.9)),
             "p95_regret":float(g.regret.quantile(.95)),"max_regret":float(g.regret.max()),
             "exact_optimal_rate":float(g.exact_optimal.mean()),
             "near_1e-3_rate":float(g['near_1e-3'].mean()),
             "near_0_01_rate":float(g['near_1e-2'].mean())}
        for epsilon in EPSILONS:
            row[f"p_regret_le_{epsilon:g}"]=float((g.regret<=epsilon+EXACT_TOL).mean())
        policy_by.append(row)
    policy_by=pd.DataFrame(policy_by); policy_by.to_csv(output / "policy_regret_by_requirement.csv", index=False)

    pair78=original[original.action_index==27].copy()
    assert pair78[['primary_server','backup_server']].drop_duplicates().values.tolist()==[[7,8]]
    pair78['regret']=[regrets[(sid,float(req))][27] for sid,req in zip(pair78.state_id,pair78.R_req)]
    pair78_by=[]
    for req,g in pair78.groupby('R_req'):
        suboptimal = g.loc[g.regret > EXACT_TOL, "regret"]
        pair78_by.append({"R_req":req,"States":len(g),"exact_optimal_rate":float((g.regret<=EXACT_TOL).mean()),
            "mean_regret_given_suboptimal":float(suboptimal.mean()),
            "near_1e-3_rate":float((g.regret<=1e-3+EXACT_TOL).mean()),
            "near_0_01_rate":float((g.regret<=1e-2+EXACT_TOL).mean()),
            "mean_regret":float(g.regret.mean()),"median_regret":float(g.regret.median()),
            "p90_regret":float(g.regret.quantile(.90)),"p95_regret":float(g.regret.quantile(.95))})
    pair78_by=pd.DataFrame(pair78_by); pair78_by.to_csv(output / "pair_78_landscape_regret.csv", index=False)

    overlap_rows=[]
    original_ids=sorted(exact.state_id.unique())
    for a in REQS:
        for b in REQS:
            intersections=[]; unions=[]; jaccards=[]
            for sid in original_ids:
                left=sets[(sid,a)]; right=sets[(sid,b)]
                common=len(left&right); combined=len(left|right)
                intersections.append(common); unions.append(combined); jaccards.append(common/combined)
            overlap_rows.append({"R_req_A":a,"R_req_B":b,"States":len(original_ids),
                "mean_jaccard":float(np.mean(jaccards)),"median_jaccard":float(np.median(jaccards)),
                "any_overlap_rate":float(np.mean(np.asarray(intersections)>0)),
                "mean_intersection_size":float(np.mean(intersections)),
                "mean_union_size":float(np.mean(unions))})
    overlap=pd.DataFrame(overlap_rows); overlap.to_csv(output / "oracle_set_overlap.csv", index=False)
    membership=[]
    for sid in original_ids:
        exact_count=sum(27 in sets[(sid,req)] for req in REQS)
        near_count=sum(regrets[(sid,req)][27] <= 1e-2+EXACT_TOL for req in REQS)
        membership.append({"state_id":sid,"pair_78_exact_tier_count":exact_count,
                           "pair_78_near_0_01_tier_count":near_count})
    membership=pd.DataFrame(membership)
    membership.to_csv(output / "pair_78_multi_requirement_membership_per_state.csv", index=False)
    membership_summary=pd.DataFrame([
        {"Measure":column,"tiers":count,"states":int((membership[column]==count).sum()),
         "fraction":float((membership[column]==count).mean())}
        for column in ("pair_78_exact_tier_count","pair_78_near_0_01_tier_count") for count in range(5)])
    membership_summary.to_csv(output / "pair_78_multi_requirement_membership.csv", index=False)

    make_plots(output, exact, near_by, gaps, policy_detail, overlap)
    add_report(output, exact_by, near_by, gap_by, policy_by, pair78_by, overlap, ties_by, membership_summary)
    status={"original_oracle_states":EXPECTED_STATES,"original_oracle_rows":len(original),
            "supplemental_actual_requirement_0.9_states":200,"matched_actual_policy_states":len(policy_detail),
            "exact_reward_tolerance":EXACT_TOL,"max_original_reward_replay_error":replay_error,
            "production_code_modified":False,"training_run":False}
    (output/"reward_landscape_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    return status


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=ROOT/"diagnostics/results")
    args=parser.parse_args()
    print(json.dumps(run(args.output_dir),indent=2))
