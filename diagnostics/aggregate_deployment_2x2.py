"""Build the final deployment × reliability-mask ablation from saved runs.

This script only reads saved Pair/Masked evaluations and the new Pair
stochastic deployment outputs. The Masked stochastic-vs-greedy paired result
is reused verbatim from the existing formal report tables.
"""
from __future__ import annotations

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

from config.params import params
from diagnostics.run_masked_pair_ppo_10seed import OUT, formal_spec
from diagnostics.run_pair_stochastic_deployment import DEPLOYMENT_OUT

POLICIES = ("Pair Greedy", "Pair Stochastic", "Masked Greedy", "Masked Stochastic")
POLICY_SOURCE = {
    "pair": "Pair Greedy",
    "pair_stochastic": "Pair Stochastic",
    "masked_greedy": "Masked Greedy",
    "masked_stochastic": "Masked Stochastic",
}
TIERS = (0.9, 0.99, 0.999, 0.9999)
BOOTSTRAP_SEED = 2043
BOOTSTRAP_SAMPLES = 20_000
TIE_TOLERANCE = 1e-10
OUT_FILES = {
    "seed": OUT / "deployment_2x2_seed_results.csv",
    "tier": OUT / "deployment_2x2_requirement_results.csv",
    "server": OUT / "deployment_2x2_server_load_results.csv",
    "pairs": OUT / "deployment_2x2_pair_selection_results.csv",
    "paired": OUT / "deployment_2x2_paired_comparisons.csv",
    "main": OUT / "deployment_2x2_main_table.csv",
    "interaction": OUT / "deployment_2x2_interaction.csv",
    "report": OUT / "DEPLOYMENT_2X2_ABLATION.md",
}


def check_pair_stochastic_trial(trial):
    folder = DEPLOYMENT_OUT / f"trial_{trial:03d}"
    if not (folder / "completed.json").exists():
        raise RuntimeError(f"Pair stochastic trial {trial} has not completed")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if meta["training_performed"] or meta["reliability_mask_applied"]:
        raise RuntimeError(f"Trial {trial} did not use the required frozen unmasked policy")
    if not meta["greedy_replay_matches_formal_pair_results"]:
        raise RuntimeError(f"Trial {trial} failed Pair greedy reproduction")
    if not meta["arrival_stream_matches_greedy_replay"] or not meta["spatial_risk_stream_matches_greedy_replay"]:
        raise RuntimeError(f"Trial {trial} exogenous streams differ")
    replay = json.loads((folder / "greedy_replay_validation.json").read_text(encoding="utf-8"))
    if replay["tasks_compared"] != 4000 or replay["action_mismatch_count"] != 0:
        raise RuntimeError(f"Trial {trial} Pair greedy replay check is incomplete")
    return folder, meta


def metrics_for(assignments, decisions, load):
    a = assignments.sort_values(["episode", "task_id"]).reset_index(drop=True)
    d = decisions.sort_values(["episode", "task_id"]).reset_index(drop=True)
    if len(a) != 4000 or len(d) != 4000 or not np.array_equal(a.action_index, d.action_index):
        raise RuntimeError("Pair stochastic decision and assignment records are not paired")
    satisfied = a.Reliability_Satisfied.astype(bool).to_numpy()
    safe_set_nonempty = d.safe_set_size.to_numpy(dtype=int) > 0
    selected_safe = d.selected_action_safe.astype(bool).to_numpy()
    if not np.array_equal(satisfied, selected_safe):
        raise RuntimeError("Post-hoc safe mask does not match production task feasibility")
    n = len(a)
    counts = a.action_index.value_counts().reindex(range(params.num_actions), fill_value=0).to_numpy(dtype=float)
    frequencies = counts / n
    positive = frequencies[frequencies > 0]
    high = np.isclose(a.Reliability_Requirement.to_numpy(dtype=float), 0.9999)
    if int(high.sum()) != 1000:
        raise RuntimeError("Highest reliability tier must contain 1000 evaluation tasks")
    return {
        "tasks": n,
        "mean_reward": float(a.Task_Reward.mean()),
        "mean_latency": float(a.Task_Delay.mean()),
        "p50_latency": float(a.Task_Delay.quantile(.50)),
        "p90_latency": float(a.Task_Delay.quantile(.90)),
        "p95_latency": float(a.Task_Delay.quantile(.95)),
        "overall_rsr": float(satisfied.mean()),
        "highest_rsr": float(satisfied[high].mean()),
        "feasibility_rate": float(safe_set_nonempty.mean()),
        "conditional_rsr": float(satisfied[safe_set_nonempty].mean()) if safe_set_nonempty.any() else np.nan,
        "avoidable_violation_rate": float((safe_set_nonempty & ~satisfied).mean()),
        "unavoidable_violation_rate": float((~safe_set_nonempty & ~satisfied).mean()),
        "empty_safe_set_rate": float((~safe_set_nonempty).mean()),
        "pair_selection_hhi": float(np.square(frequencies).sum()),
        "pair_selection_entropy": float(-(positive * np.log(positive)).sum()),
        "top1_pair_frequency": float(frequencies.max()),
        "unique_selected_pairs": int(np.count_nonzero(counts)),
        "pair_78_frequency": float(frequencies[-1]),
        "mean_native_policy_entropy": float(d.native_policy_entropy.mean()),
        "mean_safe_set_size": float(d.safe_set_size.mean()),
        "maximum_server_selection_share": float(load.selection_count.max() / (2 * n)),
        "maximum_server_utilization": float(load.utilization.max()),
        "maximum_mean_queue_length": float(load.mean_queue_length.max()),
        "maximum_p95_queue_length": float(load.p95_queue_length.max()),
        "mean_server_queue_length": float(load.mean_queue_length.mean()),
        "mean_server_p95_queue_length": float(load.p95_queue_length.mean()),
        "mean_server_waiting_time": float(load.mean_waiting_time.mean()),
        "avoidable_violation_count": int((safe_set_nonempty & ~satisfied).sum()),
        "unavoidable_violation_count": int((~safe_set_nonempty & ~satisfied).sum()),
    }


def metrics_for_tier(assignments, decisions):
    rows = []
    for requirement in TIERS:
        select = np.isclose(assignments.Reliability_Requirement.to_numpy(dtype=float), requirement)
        a = assignments.loc[select].reset_index(drop=True)
        d = decisions.loc[select].reset_index(drop=True)
        if len(a) != 1000:
            raise RuntimeError(f"R_req={requirement}: expected 1000 tasks, got {len(a)}")
        satisfied = a.Reliability_Satisfied.astype(bool).to_numpy()
        feasible = d.safe_set_size.to_numpy(dtype=int) > 0
        rows.append({
            "R_req": requirement,
            "tasks": len(a),
            "overall_rsr": float(satisfied.mean()),
            "feasibility_rate": float(feasible.mean()),
            "conditional_rsr": float(satisfied[feasible].mean()) if feasible.any() else np.nan,
            "avoidable_violation_rate": float((feasible & ~satisfied).mean()),
            "unavoidable_violation_rate": float((~feasible & ~satisfied).mean()),
        })
    return rows


def compile_pair_stochastic():
    summaries, tiers, loads, actions = [], [], [], []
    for trial in range(10):
        folder, meta = check_pair_stochastic_trial(trial)
        assignments = pd.read_csv(folder / "evaluation_task_assignments.csv")
        decisions = pd.read_csv(folder / "evaluation_decisions.csv")
        load = pd.read_csv(folder / "server_load.csv")
        if len(load) != params.serverNo:
            raise RuntimeError(f"Trial {trial}: expected server-level data for every node")
        archive = np.load(folder / "evaluation_policy_probabilities.npz")["probabilities"]
        if archive.shape != (4000, params.num_actions) or not np.isfinite(archive).all():
            raise RuntimeError(f"Trial {trial}: invalid full action probability archive")
        if not np.allclose(archive.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"Trial {trial}: native policy probabilities do not normalize")
        chosen = assignments.action_index.to_numpy(dtype=int)
        chosen_prob = archive[np.arange(len(archive)), chosen]
        if not np.allclose(chosen_prob, decisions.selected_action_probability, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"Trial {trial}: sampled action probability archive mismatch")
        if len(decisions) != 4000 or not np.all(
            decisions.safe_mask.map(lambda text: len(json.loads(text)) == params.num_actions)
        ):
            raise RuntimeError(f"Trial {trial}: incomplete post-hoc decision audit")
        load["policy"] = "Pair Stochastic"
        load["trial_id"] = trial
        loads.append(load)
        summary = {
            "trial_id": trial,
            "policy": "Pair Stochastic",
            "actor_sha256": meta["checkpoint_actor_sha256"],
            "action_rng_seed": meta["action_rng_seed"],
            **metrics_for(assignments, decisions, pd.read_csv(folder / "server_load.csv")),
        }
        summaries.append(summary)
        for row in metrics_for_tier(assignments, decisions):
            tiers.append({"trial_id": trial, "policy": "Pair Stochastic", **row})
        counts = assignments.action_index.value_counts().reindex(
            range(params.num_actions), fill_value=0
        )
        for action, count in counts.items():
            actions.append({"trial_id": trial, "policy": "Pair Stochastic",
                            "action_index": int(action), "selection_count": int(count),
                            "selection_frequency": float(count / len(assignments))})
    return pd.DataFrame(summaries), pd.DataFrame(tiers), pd.concat(loads, ignore_index=True), pd.DataFrame(actions)


def paired_bootstrap(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise RuntimeError("Paired bootstrap requires ten finite seed-level deltas")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 10, size=(BOOTSTRAP_SAMPLES, 10))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


LOWER_IS_BETTER = {
    "mean_latency", "p50_latency", "p90_latency", "p95_latency",
    "pair_selection_hhi", "top1_pair_frequency", "pair_78_frequency",
    "maximum_server_selection_share", "maximum_server_utilization",
    "maximum_mean_queue_length", "maximum_p95_queue_length",
    "mean_server_queue_length", "mean_server_p95_queue_length",
    "mean_server_waiting_time", "avoidable_violation_rate", "unavoidable_violation_rate",
}


def compare_pair(seed_frame, label, left, right, metrics):
    a = seed_frame.loc[seed_frame.policy.eq(left)].sort_values("trial_id").reset_index(drop=True)
    b = seed_frame.loc[seed_frame.policy.eq(right)].sort_values("trial_id").reset_index(drop=True)
    if a.trial_id.tolist() != list(range(10)) or b.trial_id.tolist() != list(range(10)):
        raise RuntimeError(f"{label}: incomplete paired seed rows")
    rows = []
    for metric in metrics:
        delta = a[metric].to_numpy(dtype=float) - b[metric].to_numpy(dtype=float)
        low, high = paired_bootstrap(delta)
        beneficial = -delta if metric in LOWER_IS_BETTER else delta
        rows.append({
            "comparison": label, "left_policy": left, "right_policy": right,
            "metric": metric, "mean_delta": float(delta.mean()),
            "std_seed_delta": float(delta.std(ddof=1)),
            "ci95_low": low, "ci95_high": high,
            "wins": int((beneficial > TIE_TOLERANCE).sum()),
            "losses": int((beneficial < -TIE_TOLERANCE).sum()),
            "ties": int((np.abs(beneficial) <= TIE_TOLERANCE).sum()),
            "win_direction": "lower" if metric in LOWER_IS_BETTER else "higher",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "reused_existing_result": False,
        })
    return rows


def load_existing_masked_sampling_comparison():
    previous = pd.read_csv(OUT / "paired_comparisons.csv")
    ci = pd.read_csv(OUT / "bootstrap_ci.csv")
    wlt = pd.read_csv(OUT / "win_loss_tie.csv")
    existing = previous[previous.comparison.eq("stochastic_minus_greedy")]
    if len(existing) != 10:
        raise RuntimeError("Existing Masked stochastic-vs-greedy paired results are incomplete")
    output = []
    for metric in (
        "mean_reward", "mean_latency", "overall_rsr", "highest_rsr",
        "pair_selection_hhi", "top1_pair_frequency", "maximum_server_selection_share",
        "maximum_server_utilization", "maximum_mean_queue_length", "maximum_p95_queue_length",
        "mean_server_queue_length", "mean_server_p95_queue_length", "avoidable_violation_rate",
        "conditional_rsr",
    ):
        deltas = existing.sort_values("trial_id")[f"delta_{metric}"].to_numpy(dtype=float)
        old_ci = ci[(ci.comparison.eq("stochastic_minus_greedy")) & ci.metric.eq(metric)]
        old_wlt = wlt[(wlt.comparison.eq("stochastic_minus_greedy")) & wlt.metric.eq(metric)]
        if len(old_ci) != 1 or len(old_wlt) != 1:
            raise RuntimeError(f"Missing existing Masked sampling result for {metric}")
        ci_row, wlt_row = old_ci.iloc[0], old_wlt.iloc[0]
        output.append({
            "comparison": "Masked Stochastic - Masked Greedy",
            "left_policy": "Masked Stochastic", "right_policy": "Masked Greedy",
            "metric": metric, "mean_delta": float(deltas.mean()),
            "std_seed_delta": float(deltas.std(ddof=1)),
            "ci95_low": float(ci_row.ci95_low), "ci95_high": float(ci_row.ci95_high),
            "wins": int(wlt_row.positive_seeds), "losses": int(wlt_row.negative_seeds),
            "ties": int(wlt_row.ties), "win_direction": str(wlt_row.win_direction),
            "bootstrap_samples": int(ci_row.bootstrap_samples),
            "bootstrap_seed": int(ci_row.bootstrap_seed),
            "reused_existing_result": True,
        })
    # Preserve the prior formal report's exact key deltas as an integrity check.
    expected = {"mean_reward": 10.7552, "mean_latency": -.9031,
                "pair_selection_hhi": -.7764, "top1_pair_frequency": -.7922}
    for metric, reference in expected.items():
        row = next(item for item in output if item["metric"] == metric)
        if not np.isclose(row["mean_delta"], reference, atol=6e-5, rtol=0.0):
            raise RuntimeError(f"Existing Masked sampling result changed for {metric}")
    return output


def summarize_main(seed_frame):
    metrics = (
        "mean_reward", "mean_latency", "p50_latency", "p90_latency", "p95_latency",
        "overall_rsr", "highest_rsr", "feasibility_rate", "conditional_rsr",
        "avoidable_violation_rate", "unavoidable_violation_rate", "empty_safe_set_rate", "pair_selection_hhi",
        "pair_selection_entropy", "top1_pair_frequency", "unique_selected_pairs",
        "pair_78_frequency", "maximum_server_selection_share", "maximum_server_utilization",
        "maximum_mean_queue_length", "maximum_p95_queue_length", "mean_server_queue_length",
        "mean_server_p95_queue_length", "mean_server_waiting_time",
    )
    rows = []
    for policy in POLICIES:
        frame = seed_frame[seed_frame.policy.eq(policy)].sort_values("trial_id")
        if len(frame) != 10:
            raise RuntimeError(f"Main table needs ten seeds for {policy}")
        row = {"policy": policy, "n_seeds": len(frame)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        row["conditional_rsr_scope"] = "post-hoc diagnostic" if policy.startswith("Pair ") else "production mask diagnostics"
        row["avoidable_violation_scope"] = "post-hoc diagnostic" if policy.startswith("Pair ") else "production mask diagnostics"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_main(main):
    colors = {"Pair Greedy": "#4C78A8", "Pair Stochastic": "#72B7B2",
              "Masked Greedy": "#F58518", "Masked Stochastic": "#54A24B"}
    specs = [
        ("mean_reward", "Mean reward", "deployment_2x2_reward.png"),
        ("mean_latency", "Mean latency (s)", "deployment_2x2_latency.png"),
        ("overall_rsr", "Overall RSR", "deployment_2x2_rsr.png"),
        ("highest_rsr", "Highest-tier RSR", "deployment_2x2_highest_rsr.png"),
        ("pair_selection_hhi", "Pair-selection HHI", "deployment_2x2_hhi.png"),
        ("maximum_mean_queue_length", "Maximum server mean queue length", "deployment_2x2_queue.png"),
    ]
    for metric, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(8.3, 4.8))
        x = np.arange(len(POLICIES))
        means = main.set_index("policy").loc[list(POLICIES), f"{metric}_mean"].to_numpy()
        stds = main.set_index("policy").loc[list(POLICIES), f"{metric}_std"].to_numpy()
        for index, policy in enumerate(POLICIES):
            ax.errorbar(index, means[index], yerr=stds[index], fmt="o", capsize=4,
                        color=colors[policy], label=policy)
        ax.set_xticks(x, POLICIES, rotation=12)
        ax.set_ylabel(ylabel + " (seed mean ± sample std)")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=170)
        plt.close(fig)


def plot_paired_effects(seed_frame, pairwise):
    def plot_pair_metrics(policies, metrics, labels, filename, title):
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.3), squeeze=False)
        for ax, metric, label in zip(axes[0], metrics, labels):
            left = seed_frame[seed_frame.policy.eq(policies[0])].sort_values("trial_id")[metric].to_numpy()
            right = seed_frame[seed_frame.policy.eq(policies[1])].sort_values("trial_id")[metric].to_numpy()
            for i in range(10):
                ax.plot([0, 1], [left[i], right[i]], color="#777777", alpha=.45, lw=1)
            ax.scatter(np.zeros(10), left, color="#4C78A8", s=22, label=policies[0])
            ax.scatter(np.ones(10), right, color="#F58518", s=22, label=policies[1])
            ax.set_xticks([0, 1], [policies[0], policies[1]], rotation=15)
            ax.set_title(label)
            ax.grid(axis="y", alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=170)
        plt.close(fig)

    plot_pair_metrics(
        ("Pair Greedy", "Pair Stochastic"),
        ("mean_latency", "pair_selection_hhi", "top1_pair_frequency", "maximum_mean_queue_length"),
        ("Mean latency (s)", "Pair HHI", "Top-1 pair frequency", "Maximum mean queue"),
        "sampling_effect_pair_paired.png", "Sampling effect without reliability mask",
    )
    plot_pair_metrics(
        ("Pair Stochastic", "Masked Stochastic"),
        ("overall_rsr", "highest_rsr", "avoidable_violation_rate"),
        ("Overall RSR", "Highest-tier RSR", "Avoidable violation rate"),
        "mask_effect_stochastic_paired.png", "Reliability mask effect under stochastic deployment",
    )


def fmean(mean, std, metric):
    percent = metric in {
        "overall_rsr", "highest_rsr", "feasibility_rate", "conditional_rsr",
        "avoidable_violation_rate", "unavoidable_violation_rate", "empty_safe_set_rate", "top1_pair_frequency",
        "maximum_server_selection_share",
    }
    if percent:
        return f"{mean * 100:.2f}% ± {std * 100:.2f}%"
    return f"{mean:.4f} ± {std:.4f}"


def fdelta(row):
    return f"{row.mean_delta:+.4f} [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}]"


def make_report(main, pairwise, interactions, trial_results, pair_tiers):
    main_index = main.set_index("policy")
    main_metrics = [
        ("mean_reward", "Reward"), ("mean_latency", "Mean latency"),
        ("p95_latency", "P95 latency"), ("overall_rsr", "Overall RSR"),
        ("highest_rsr", "Highest-tier RSR"), ("conditional_rsr", "Conditional RSR"),
        ("avoidable_violation_rate", "AVR"), ("unavoidable_violation_rate", "UVR"),
        ("pair_selection_hhi", "Pair HHI"), ("top1_pair_frequency", "Top-1 pair"),
        ("maximum_mean_queue_length", "Max mean queue"),
    ]
    lines = [
        "# Deployment 2×2 Ablation",
        "",
        "## 1. Experimental Control",
        "",
        "本轮没有训练模型。十个 Pair checkpoint 均从原始正式实验目录加载，Actor/Critic SHA256 与完成清单核对；Pair greedy 每个 seed 都在相同 arrival/spatial seed 下重放，并逐任务复现已保存的 action、reward、latency 与可靠性。每个 seed 的 4,000 项 Pair stochastic 评估均与同 seed greedy 重放的到达间隔及完整空间风险轨迹逐项相同。",
        "",
        "Pair stochastic 从原 Pair Actor 的完整 28-action `Categorical(logits)` 分布采样；没有 reliability mask、logit 修改或温度调整。动作采样使用私有、由 Trial ID 和独立标签派生的 CPU Torch generator。Pair 的 feasibility / AVR / conditional RSR 是在其访问状态上按生产可靠性规则事后诊断，不是 Pair policy 的限制。Pair 与 Masked 使用各自正式训练得到的 checkpoint；因此 B 是两个正式 stochastic policy 版本的成对比较，不能解释成对同一个 Actor 只切换 inference mask 的单因素实验。统计单位为训练 seed（n=10），所有 ± 为 seed 间 sample std。A/B 的 CI 使用 20,000 次 seed-level paired bootstrap（seed=2043）；C 直接复用此前正式 Masked deployment ablation 的 per-seed paired deltas 与 bootstrap CI。",
        "",
        "## 2. 2×2 Main Table",
        "",
        "| Policy | " + " | ".join(label for _, label in main_metrics) + " |",
        "|---|" + "|".join(["---:"] * len(main_metrics)) + "|",
    ]
    for policy in POLICIES:
        row = main_index.loc[policy]
        values = [fmean(row[f"{metric}_mean"], row[f"{metric}_std"], metric) for metric, _ in main_metrics]
        lines.append("| " + policy + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "Pair Greedy / Pair Stochastic 的 Conditional RSR 与 AVR 为 post-hoc diagnostic。AVR/UVR 按全部任务数作分母；Conditional RSR 的分母为 safe set 非空任务。可靠性比率以百分数显示。",
        "",
        "## 3. Sampling Effect Without Mask",
        "",
        "Comparison A 定义为 Pair Stochastic − Pair Greedy。",
        "",
        "| Metric | Mean delta [95% paired CI] | Win/Loss/Tie |",
        "|---|---:|---:|",
    ]
    a = pairwise[pairwise.comparison.eq("Pair Stochastic - Pair Greedy")]
    for metric, label in [
        ("mean_reward", "Reward"), ("mean_latency", "Latency"),
        ("overall_rsr", "Overall RSR"), ("highest_rsr", "Highest-tier RSR"),
        ("pair_selection_hhi", "Pair HHI"), ("top1_pair_frequency", "Top-1 frequency"),
        ("maximum_server_selection_share", "Max server selection share"),
        ("maximum_mean_queue_length", "Maximum mean queue"),
        ("maximum_p95_queue_length", "Maximum P95 queue"),
    ]:
        r = a[a.metric.eq(metric)].iloc[0]
        lines.append(f"| {label} | {fdelta(r)} | {r.wins}/{r.losses}/{r.ties} |")
    lines += [
        "",
        "Latency, pair concentration, server selection concentration, and queue metrics are favorable when their deltas are negative; other displayed deltas are favorable when positive. W/L/T counts use that metric's favorable direction and 1e-10 tie tolerance.",
        "",
        "## 4. Mask Effect Under Stochastic Deployment",
        "",
        "Comparison B 定义为 Masked Stochastic − Pair Stochastic；两者都采用 stochastic deployment。Pair 的 reliability values are evaluated post hoc using `Task.initialize_reliability_evaluation`; Masked outputs use the same production rule during action selection.",
        "",
        "| Metric | Mean delta [95% paired CI] | Win/Loss/Tie |",
        "|---|---:|---:|",
    ]
    b = pairwise[pairwise.comparison.eq("Masked Stochastic - Pair Stochastic")]
    for metric, label in [
        ("overall_rsr", "Overall RSR"), ("highest_rsr", "Highest-tier RSR"),
        ("conditional_rsr", "Conditional RSR"),
        ("avoidable_violation_rate", "AVR"),
        ("unavoidable_violation_rate", "UVR"), ("mean_reward", "Reward"),
        ("mean_latency", "Latency"), ("pair_selection_hhi", "Pair HHI"),
        ("maximum_mean_queue_length", "Maximum mean queue"),
    ]:
        r = b[b.metric.eq(metric)].iloc[0]
        lines.append(f"| {label} | {fdelta(r)} | {r.wins}/{r.losses}/{r.ties} |")
    lines += [
        "",
        "RSR、AVR、UVR 等比例指标的 paired delta 使用 [0,1] 比例单位；例如 `+0.01` 等于增加 1 个百分点。",
        "",
        "## 5. Sampling Effect With Mask",
        "",
        "Comparison C 定义为 Masked Stochastic − Masked Greedy。数值逐项复用已有正式报告中的 bootstrap 输出，不重跑 checkpoint 或重新抽样。",
        "",
        "| Metric | Mean delta [95% paired CI] | Win/Loss/Tie |",
        "|---|---:|---:|",
    ]
    c = pairwise[pairwise.comparison.eq("Masked Stochastic - Masked Greedy")]
    for metric, label in [
        ("mean_reward", "Reward"), ("mean_latency", "Latency"),
        ("overall_rsr", "Overall RSR"), ("highest_rsr", "Highest-tier RSR"),
        ("pair_selection_hhi", "Pair HHI"), ("top1_pair_frequency", "Top-1 frequency"),
        ("maximum_server_selection_share", "Max server selection share"),
        ("maximum_mean_queue_length", "Maximum mean queue"),
        ("maximum_p95_queue_length", "Maximum P95 queue"),
        ("avoidable_violation_rate", "AVR"), ("conditional_rsr", "Conditional RSR"),
    ]:
        r = c[c.metric.eq(metric)].iloc[0]
        lines.append(f"| {label} | {fdelta(r)} | {r.wins}/{r.losses}/{r.ties} |")
    lines += ["", "## 6. Reliability Decomposition", ""]
    reliability_metrics = ["feasibility_rate", "conditional_rsr", "overall_rsr",
                           "avoidable_violation_rate", "unavoidable_violation_rate", "empty_safe_set_rate"]
    lines += ["| Policy | " + " | ".join(reliability_metrics) + " |",
              "|---|" + "|".join(["---:"] * len(reliability_metrics)) + "|"]
    for policy in POLICIES:
        row = main_index.loc[policy]
        lines.append("| " + policy + " | " + " | ".join(
            fmean(row[f"{m}_mean"], row[f"{m}_std"], m) for m in reliability_metrics
        ) + " |")
    lines += [
        "",
        "Pair metrics in this table remain post hoc. The production reliability test is `execution_reliability >= reliability_requirement`, evaluated through `Task.initialize_reliability_evaluation`; safe-set feasibility is whether any of the 28 legal pairs passes it.",
        "",
        "### Per-requirement RSR and feasibility",
        "",
        "| R_req | Pair Greedy RSR | Pair Stochastic RSR | Masked Greedy RSR | Masked Stochastic RSR | Pair Stochastic feasible | Masked Stochastic feasible |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    existing_tiers = pd.read_csv(OUT / "requirement_level_results.csv")
    pair_stoch_tiers = pair_tiers.groupby("R_req")[["overall_rsr", "feasibility_rate"]].agg(["mean", "std"])
    for req in TIERS:
        row = [f"{req:g}"]
        for p in ("pair", "pair_stochastic", "masked_greedy", "masked_stochastic"):
            if p == "pair_stochastic":
                vals = pair_stoch_tiers.loc[req, ("overall_rsr", "mean")], pair_stoch_tiers.loc[req, ("overall_rsr", "std")]
            else:
                f = existing_tiers[(existing_tiers.policy.eq(p)) & np.isclose(existing_tiers.R_req, req)]
                vals = f.overall_rsr.mean(), f.overall_rsr.std(ddof=1)
            row.append(fmean(vals[0], vals[1], "overall_rsr"))
        feasible = pair_stoch_tiers.loc[req, ("feasibility_rate", "mean")], pair_stoch_tiers.loc[req, ("feasibility_rate", "std")]
        masked_feas = existing_tiers[(existing_tiers.policy.eq("masked_stochastic")) & np.isclose(existing_tiers.R_req, req)]
        row += [fmean(feasible[0], feasible[1], "feasibility_rate"),
                fmean(masked_feas.feasibility_rate.mean(), masked_feas.feasibility_rate.std(ddof=1), "feasibility_rate")]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Pair Stochastic 是否产生 avoidable violation 的结果可直接从 AVR / Conditional RSR 观察；Masked 的 avoidable violation 为零是 action mask 保证的属性，并经现有 formal sanity checks 核验。",
        "",
        "## 7. Queue / Concentration Analysis",
        "",
        "| Policy | Pair HHI | Pair entropy | Top-1 pair | Unique pairs | (7,8) frequency | Max server selection share | Max utilization | Max mean queue | Max P95 queue | Mean server wait (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        row = main_index.loc[policy]
        cells = []
        for metric in ("pair_selection_hhi", "pair_selection_entropy", "top1_pair_frequency",
                       "unique_selected_pairs", "pair_78_frequency", "maximum_server_selection_share",
                       "maximum_server_utilization", "maximum_mean_queue_length",
                       "maximum_p95_queue_length", "mean_server_waiting_time"):
            cells.append(fmean(row[f"{metric}_mean"], row[f"{metric}_std"], metric))
        lines.append("| " + policy + " | " + " | ".join(cells) + " |")
    lines += ["", "服务器逐节点 selection count、utilization、mean/P95 queue、waiting time 见 `deployment_2x2_server_load_results.csv`。各 seed 的完整 Pair stochastic 原始任务、post-hoc mask、policy entropy/action probability 和 28 维概率向量也随本报告保存。", "",
              "## 8. Seed Stability", "",
              "| Comparison | Metric | Seed delta min…max | Beneficial W/L/T | Paired CI excludes zero |",
              "|---|---|---:|---:|---|"]
    for comparison in ("Pair Stochastic - Pair Greedy", "Masked Stochastic - Pair Stochastic", "Masked Stochastic - Masked Greedy"):
        comp = pairwise[pairwise.comparison.eq(comparison)]
        for metric in ("mean_latency", "overall_rsr", "highest_rsr", "avoidable_violation_rate", "pair_selection_hhi", "maximum_mean_queue_length"):
            hit = comp[comp.metric.eq(metric)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            diffs = comparison_deltas(trial_results, comparison, metric)
            beneficial = -diffs if metric in LOWER_IS_BETTER else diffs
            excludes = bool(r.ci95_low > 0 or r.ci95_high < 0)
            lines.append(f"| {comparison} | {metric} | [{diffs.min():+.4f}, {diffs.max():+.4f}] | {int((beneficial>TIE_TOLERANCE).sum())}/{int((beneficial<-TIE_TOLERANCE).sum())}/{int((np.abs(beneficial)<=TIE_TOLERANCE).sum())} | {'yes' if excludes else 'no'} |")
    lines += ["", "## 9. Mechanism Conclusion", ""]
    # Decision support from CIs and paired seed directions, not task pooling.
    a_latency = a[a.metric.eq("mean_latency")].iloc[0]
    a_hhi = a[a.metric.eq("pair_selection_hhi")].iloc[0]
    a_queue = a[a.metric.eq("maximum_mean_queue_length")].iloc[0]
    a_high = a[a.metric.eq("highest_rsr")].iloc[0]
    b_rsr = b[b.metric.eq("overall_rsr")].iloc[0]
    b_high = b[b.metric.eq("highest_rsr")].iloc[0]
    b_cond = b[b.metric.eq("conditional_rsr")].iloc[0]
    b_avr = b[b.metric.eq("avoidable_violation_rate")].iloc[0]
    q1 = a_latency.ci95_high < 0 and a_hhi.ci95_high < 0 and a_queue.ci95_high < 0
    q2 = b_avr.ci95_high < 0 and b_cond.ci95_low > 0
    q3 = b_rsr.ci95_low > 0 and b_high.ci95_low > 0 and b_cond.ci95_low > 0
    lines += [
        f"**Q1 — Pair PPO 由 greedy 改为 stochastic 后是否降低 concentration、queue 和 latency？** {'是，三项 paired 95% CI 均完全低于 0。' if q1 else '证据不支持同时稳定改善这三项。'} Pair Stochastic − Pair Greedy：latency {fdelta(a_latency)}；HHI {fdelta(a_hhi)}；max mean queue {fdelta(a_queue)}。",
        "",
        f"**Q2 — 没有 reliability mask 的 Pair stochastic 是否产生更多 avoidable violations？** {'是，Masked stochastic 相对 Pair stochastic 的 AVR 显著降低且 Conditional RSR 提高。' if q2 else '数据未能同时支持 AVR 降低与 Conditional RSR 提高。'} Pair Stochastic 的 AVR 为 {main_index.loc['Pair Stochastic','avoidable_violation_rate_mean']*100:.2f}% ± {main_index.loc['Pair Stochastic','avoidable_violation_rate_std']*100:.2f}%；Masked Stochastic 为 {main_index.loc['Masked Stochastic','avoidable_violation_rate_mean']*100:.2f}% ± {main_index.loc['Masked Stochastic','avoidable_violation_rate_std']*100:.2f}%。",
        "",
        f"**Q3 — stochastic 部署下 mask 是否提高 RSR / highest-RSR / Conditional RSR？** {'是，三项 paired 95% CI 均完全高于 0。' if q3 else '至少一项没有获得正向 paired CI 支持。'} Overall RSR {fdelta(b_rsr)}；highest-tier RSR {fdelta(b_high)}；Conditional RSR {fdelta(b_cond)}；AVR {fdelta(b_avr)}。",
        "",
    ]
    if q1 and q2 and q3:
        lines.append("这些结果支持机制解释：stochastic selection 主要缓解 pair/server concentration 与 queueing；reliability masking 消除可避免的可靠性违约；两者结合带来最终表现提升。")
    else:
        lines.append("完整机制结论应限制在上面的逐项统计证据范围内，未被 CI 和 seed 方向支持的作用不作稳定性结论。")
    lines += [
        "",
        "### Sampling × mask interaction（描述性）",
        "",
        "| Metric | Sampling delta without mask (Pair stochastic − Pair greedy) | Sampling delta with mask (Masked stochastic − Masked greedy) | With-mask minus no-mask delta |",
        "|---|---:|---:|---:|"]
    for metric, label in (("mean_latency", "Mean latency"), ("pair_selection_hhi", "Pair HHI"),
                          ("top1_pair_frequency", "Top-1 frequency"), ("maximum_mean_queue_length", "Max mean queue")):
        no_mask = a[a.metric.eq(metric)].iloc[0].mean_delta
        with_mask = c[c.metric.eq(metric)].iloc[0].mean_delta
        lines.append(f"| {label} | {no_mask:+.4f} | {with_mask:+.4f} | {with_mask-no_mask:+.4f} |")
    lines += ["", "Interaction 数值仅作描述，不进行 two-way ANOVA。"]
    OUT_FILES["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_deltas(seed_frame, comparison, metric):
    policy_pairs = {
        "Pair Stochastic - Pair Greedy": ("Pair Stochastic", "Pair Greedy"),
        "Masked Stochastic - Pair Stochastic": ("Masked Stochastic", "Pair Stochastic"),
        "Masked Stochastic - Masked Greedy": ("Masked Stochastic", "Masked Greedy"),
    }
    left, right = policy_pairs[comparison]
    a = seed_frame[seed_frame.policy.eq(left)].sort_values("trial_id")[metric].to_numpy()
    b = seed_frame[seed_frame.policy.eq(right)].sort_values("trial_id")[metric].to_numpy()
    return a - b


def main():
    formal_spec()
    DEPLOYMENT_OUT.mkdir(parents=True, exist_ok=True)
    new_result, new_tier, new_load, new_actions = compile_pair_stochastic()
    existing = pd.read_csv(OUT / "evaluation_seed_summary.csv")
    existing = existing[existing.policy.isin(("pair", "masked_greedy", "masked_stochastic"))].copy()
    existing["policy"] = existing.policy.map(POLICY_SOURCE)
    new_result["posthoc_reliability_diagnostic"] = True
    existing["posthoc_reliability_diagnostic"] = existing.policy.str.startswith("Pair ")
    seed_frame = pd.concat([existing, new_result], ignore_index=True, sort=False)
    seed_frame = seed_frame.sort_values(["policy", "trial_id"]).reset_index(drop=True)
    if seed_frame.groupby("policy").trial_id.apply(list).to_dict() != {p: list(range(10)) for p in POLICIES}:
        raise RuntimeError("The 2×2 table does not contain ten matched seeds per policy")

    main_table = summarize_main(seed_frame)
    existing_tier = pd.read_csv(OUT / "requirement_level_results.csv")
    existing_tier = existing_tier[existing_tier.policy.isin(("pair", "masked_greedy", "masked_stochastic"))].copy()
    existing_tier["policy"] = existing_tier.policy.map(POLICY_SOURCE)
    tier_frame = pd.concat([existing_tier, new_tier], ignore_index=True, sort=False)
    existing_load = pd.read_csv(OUT / "server_load_results.csv")
    existing_load = existing_load[existing_load.policy.isin(("pair", "masked_greedy", "masked_stochastic"))].copy()
    existing_load["policy"] = existing_load.policy.map(POLICY_SOURCE)
    load_frame = pd.concat([existing_load, new_load], ignore_index=True, sort=False)
    existing_actions = pd.read_csv(OUT / "pair_selection_results.csv")
    existing_actions = existing_actions[existing_actions.policy.isin(("pair", "masked_greedy", "masked_stochastic"))].copy()
    existing_actions["policy"] = existing_actions.policy.map(POLICY_SOURCE)
    action_frame = pd.concat([existing_actions, new_actions], ignore_index=True, sort=False)

    compare_metrics = (
        "mean_reward", "mean_latency", "p50_latency", "p90_latency", "p95_latency",
        "overall_rsr", "highest_rsr", "feasibility_rate", "conditional_rsr",
        "avoidable_violation_rate", "unavoidable_violation_rate", "empty_safe_set_rate", "pair_selection_hhi",
        "pair_selection_entropy", "top1_pair_frequency", "unique_selected_pairs",
        "pair_78_frequency", "maximum_server_selection_share", "maximum_server_utilization",
        "maximum_mean_queue_length", "maximum_p95_queue_length", "mean_server_queue_length",
        "mean_server_p95_queue_length", "mean_server_waiting_time",
    )
    comparisons = []
    comparisons += compare_pair(seed_frame, "Pair Stochastic - Pair Greedy",
                               "Pair Stochastic", "Pair Greedy", compare_metrics)
    comparisons += compare_pair(seed_frame, "Masked Stochastic - Pair Stochastic",
                               "Masked Stochastic", "Pair Stochastic", compare_metrics)
    comparisons += load_existing_masked_sampling_comparison()
    paired_frame = pd.DataFrame(comparisons)
    interaction_rows = []
    for metric in ("mean_latency", "pair_selection_hhi", "top1_pair_frequency", "maximum_mean_queue_length"):
        a_delta = comparison_deltas(seed_frame, "Pair Stochastic - Pair Greedy", metric)
        c_delta = comparison_deltas(seed_frame, "Masked Stochastic - Masked Greedy", metric)
        interaction_rows.append({
            "metric": metric,
            "sampling_delta_without_mask_mean": float(a_delta.mean()),
            "sampling_delta_without_mask_std": float(a_delta.std(ddof=1)),
            "sampling_delta_with_mask_mean": float(c_delta.mean()),
            "sampling_delta_with_mask_std": float(c_delta.std(ddof=1)),
            "with_mask_minus_without_mask_descriptive": float(c_delta.mean() - a_delta.mean()),
            "inferential_test_performed": False,
        })
    interaction = pd.DataFrame(interaction_rows)
    for name, frame in (("seed", seed_frame), ("tier", tier_frame), ("server", load_frame),
                        ("pairs", action_frame), ("paired", paired_frame),
                        ("main", main_table), ("interaction", interaction)):
        frame.to_csv(OUT_FILES[name], index=False)
    plot_main(main_table)
    plot_paired_effects(seed_frame, paired_frame)
    make_report(main_table, paired_frame, interaction, seed_frame, new_tier)
    print(main_table[["policy", "mean_reward_mean", "mean_reward_std", "mean_latency_mean",
                      "mean_latency_std", "overall_rsr_mean", "overall_rsr_std",
                      "highest_rsr_mean", "highest_rsr_std", "pair_selection_hhi_mean",
                      "maximum_mean_queue_length_mean"]].to_string(index=False), flush=True)
    print(f"report: {OUT_FILES['report']}", flush=True)


if __name__ == "__main__":
    main()
