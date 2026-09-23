"""Summarize the matched frozen-selector experiment from saved task-level results."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "diagnostics/results/reliability_masked_policy"
MODES = ("greedy", "native_sample", "masked_sample", "masked_greedy")


def markdown(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{item:.4f}" if isinstance(item, (float, np.floating)) else str(item) for item in row) + " |")
    return "\n".join(lines)


def paired_effects(paired, bootstrap_samples=5000):
    rows = []
    comparisons = (("native_sample", "greedy"), ("masked_sample", "native_sample"),
                   ("masked_sample", "greedy"), ("masked_sample", "masked_greedy"))
    for tier, subset in (("all", paired), ("0.9999", paired[np.isclose(paired.R_req, .9999)])):
        for candidate, reference in comparisons:
            for metric, column in (("reward", "reward"), ("latency", "latency"),
                                   ("RSR", "reliability_satisfied")):
                differences = subset.assign(delta=subset[f"{candidate}_{column}"].astype(float)
                    - subset[f"{reference}_{column}"].astype(float)).groupby("episode").delta.mean().to_numpy()
                rng = np.random.default_rng(2026)
                means = differences[rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))].mean(axis=1)
                rows.append({"R_req": tier, "candidate": candidate, "reference": reference,
                             "metric": metric, "mean_paired_delta": float(differences.mean()),
                             "episode_bootstrap_ci_low": float(np.quantile(means, .025)),
                             "episode_bootstrap_ci_high": float(np.quantile(means, .975)),
                             "episodes": len(differences)})
    return pd.DataFrame(rows)


def run(directory=DEFAULT):
    directory = Path(directory)
    summary = pd.read_csv(directory / "policy_summary.csv").set_index("policy").loc[list(MODES)]
    tiers = pd.read_csv(directory / "requirement_level_summary.csv")
    highest = tiers[np.isclose(tiers.R_req, .9999)].set_index("policy").loc[list(MODES)]
    load = pd.read_csv(directory / "server_load_by_policy.csv")
    paired = pd.read_csv(directory / "paired_action_selection_comparison.csv")
    safe = pd.read_csv(directory / "safe_set_statistics.csv")
    effects = paired_effects(paired)
    effects.to_csv(directory / "paired_episode_effects.csv", index=False)
    overall_columns = ["mean_reward", "total_reward", "mean_latency", "p50_latency", "p90_latency", "p95_latency",
                       "rsr", "pair_selection_hhi", "pair_selection_entropy", "unique_pair_count",
                       "pair_78_frequency", "safe_set_empty_rate"]
    overall = summary[overall_columns].reset_index()
    risk_table = summary[["selected_rho_mean", "selected_effective_failure_probability_mean",
        "selected_pair_reliability_mean", "reliability_violation_rate"]].reset_index()
    tier_table = tiers[["policy", "R_req", "rsr", "mean_latency", "mean_reward",
        "safe_set_empty_rate", "reliability_violation_rate"]]
    high = highest[["rsr", "mean_latency", "mean_reward", "pair_78_frequency",
                    "safe_set_empty_rate", "selected_unsafe_rate"]].reset_index()
    server78 = load[load.server_id.isin((7, 8))][["policy", "server_id", "selection_count",
        "utilization", "mean_queue_length", "p95_queue_length", "mean_waiting_time", "busy_time"]]
    high_native = safe[(safe.policy == "native_sample") & np.isclose(safe.Reliability_Requirement, .9999)].iloc[0]
    high_greedy = safe[(safe.policy == "greedy") & np.isclose(safe.Reliability_Requirement, .9999)].iloc[0]
    high_masked = safe[(safe.policy == "masked_sample") & np.isclose(safe.Reliability_Requirement, .9999)].iloc[0]
    assert high_native.safe_set_empty_count == high_greedy.safe_set_empty_count == high_masked.safe_set_empty_count
    assert int(high_masked.selected_unsafe_count) == int(high_masked.safe_set_empty_count)
    def effect(candidate, reference, metric, tier="all"):
        return effects[(effects.candidate == candidate) & (effects.reference == reference)
            & (effects.metric == metric) & (effects.R_req == tier)].iloc[0]
    changed_greedy = float((paired.masked_sample_action_index != paired.greedy_action_index).mean())
    changed_native = float((paired.masked_sample_action_index != paired.native_sample_action_index).mean())
    no_safe = int(high_masked.safe_set_empty_count)
    available = int(high_native.tasks - no_safe)
    native_avoidable = int(high_native.selected_unsafe_count - no_safe)
    greedy_avoidable = int(high_greedy.selected_unsafe_count - no_safe)
    mask_vs_native_rsr = effect("masked_sample", "native_sample", "RSR", "0.9999")
    mask_vs_maskgreedy_reward = effect("masked_sample", "masked_greedy", "reward")
    mask_vs_maskgreedy_latency = effect("masked_sample", "masked_greedy", "latency")
    lines = ["# Reliability-Masked Frozen Pair PPO Selection", "",
        "本实验复用已保存的 Pair PPO Actor/Critic checkpoint；四个独立仿真副本使用完全相同的 20×200 次到达间隔与 episode 空间风险场，Actor/Critic 权重在前后逐张量一致。Greedy 与上一轮正式 trial 0 的 4,000 条评估记录逐条一致；Native Sample 与上一轮 action seed 2026 的 4,000 条记录逐条一致。所有策略仅改变冻结 Actor 的动作选择规则。完整任务级日志、mask 前后 28 维概率、安全集及图均在 `diagnostics/results/reliability_masked_policy/`。", "",
        "## 四策略总体结果", "", markdown(overall), "",
        "mean_reward 为平均每任务 reward，total_reward 为 4,000 任务 reward 总和；latency 单位秒；entropy 使用自然对数（nats），HHI 为实际 pair 选择频率平方和。`safe_set_empty_rate` 是在任何选择规则之前按当次 episode 有效故障率计算，因此四策略相同。", "",
        "选择 pair 的风险属性（effective failure probability 为两个副本联合失败概率）：", "", markdown(risk_table), "",
        "各档需求结果：", "", markdown(tier_table), "",
        "## 最高可靠性需求 R_req=0.9999", "", markdown(high), "",
        f"在该档 1,000 个任务中，{no_safe} 个（{no_safe/1000:.1%}）没有任何满足要求的 pair；因此当前合法动作和风险场下，任意可靠性 mask 的满足率上限为 {available/1000:.1%}。空集时两种 masked 策略都明确回退到可靠性最高的合法 pair，并单独标记，不计作安全选择。Native Sample 有 {int(high_native.selected_unsafe_count)} 个违约，其中 {native_avoidable} 个发生在**存在安全 pair** 的情况下；Greedy 分别为 {int(high_greedy.selected_unsafe_count)} / {greedy_avoidable}。Masked Sample 的违约恰为 {no_safe} 个空集任务，最高档满足率 {highest.loc['masked_sample','rsr']:.1%}。相对 Native Sample 的同 episode 配对提升 {mask_vs_native_rsr.mean_paired_delta:.1%}（20 episode bootstrap 95% CI {mask_vs_native_rsr.episode_bootstrap_ci_low:.1%} 至 {mask_vs_native_rsr.episode_bootstrap_ci_high:.1%}）。", "",
        f"Masked Sample 相比 Native Sample 最高档平均延迟 {highest.loc['native_sample','mean_latency']:.3f}→{highest.loc['masked_sample','mean_latency']:.3f}s，平均 reward {highest.loc['native_sample','mean_reward']:.2f}→{highest.loc['masked_sample','mean_reward']:.2f}；它仍明显优于 Greedy 的 {highest.loc['greedy','mean_latency']:.3f}s 延迟，且 reward 更高。与 Masked Greedy 相比，两者最高档 RSR 同为 {highest.loc['masked_sample','rsr']:.1%}，但 Masked Sample 延迟 {highest.loc['masked_sample','mean_latency']:.3f}s vs {highest.loc['masked_greedy','mean_latency']:.3f}s、reward {highest.loc['masked_sample','mean_reward']:.2f} vs {highest.loc['masked_greedy','mean_reward']:.2f}。", "",
        "## 队列与动作集中度", "", markdown(server78), "",
        "`mean_queue_length` 和 `p95_queue_length` 是从实际等待副本集合在每次入队、获得 CPU、完成时的状态变化按 SimPy 时间积分得到；`busy_time` 是实际运行副本的 CPU 占用时长，`utilization=busy_time/episode horizon` 跨 20 回合汇总；`mean_waiting_time` 是从进入 CPU 等待队列到获得 CPU 的真实时长。两种 greedy 选择都让服务器 7/8 长时间积压，而两种 sampling 明显分散到更多节点。", "",
        f"Masked Sample 与 Greedy 的任务级动作不同率 {changed_greedy:.1%}，与 Native Sample 的不同率 {changed_native:.1%}（两次随机抽样即使都可行，也未必选同一 pair）。Masked Sample 相比 Masked Greedy 总体 reward 的 episode 配对差 {mask_vs_maskgreedy_reward.mean_paired_delta:+.2f}（95% CI {mask_vs_maskgreedy_reward.episode_bootstrap_ci_low:+.2f} 至 {mask_vs_maskgreedy_reward.episode_bootstrap_ci_high:+.2f}），latency 差 {mask_vs_maskgreedy_latency.mean_paired_delta:+.3f}s（95% CI {mask_vs_maskgreedy_latency.episode_bootstrap_ci_low:+.3f} 至 {mask_vs_maskgreedy_latency.episode_bootstrap_ci_high:+.3f}）。这说明仅加 mask 不足以解决负载集中；保留分散选择是关键。", "",
        "## 机制判断", "",
        f"- **H1 SUPPORTED**：Greedy 的 (7,8) 选择率 {summary.loc['greedy','pair_78_frequency']:.1%}、HHI {summary.loc['greedy','pair_selection_hhi']:.3f}，服务器 7 的时间加权平均队列 {server78[(server78.policy=='greedy')&(server78.server_id==7)].iloc[0].mean_queue_length:.3f}；Native Sample 对应 {summary.loc['native_sample','pair_78_frequency']:.1%}、{summary.loc['native_sample','pair_selection_hhi']:.3f}、{server78[(server78.policy=='native_sample')&(server78.server_id==7)].iloc[0].mean_queue_length:.3f}。",
        f"- **H2 SUPPORTED**：Native Sample 降低集中度与平均延迟，但最高档 RSR {highest.loc['greedy','rsr']:.1%}→{highest.loc['native_sample','rsr']:.1%}；在有可行 pair 的 {available} 个最高档任务中仍选了 {native_avoidable} 个不安全动作。",
        f"- **H3 SUPPORTED（受空集限制）**：Masked Sample 保持 28 对均被选择，HHI {summary.loc['masked_sample','pair_selection_hhi']:.3f}，平均延迟 {summary.loc['masked_sample','mean_latency']:.3f}s；最高档 RSR 恢复到可行性上限 {highest.loc['masked_sample','rsr']:.1%}。空集率 {no_safe/1000:.1%} 是剩余违约的全部来源。", "",
        "下一步**值得在独立里程碑中研究**可靠性 mask 与训练/部署选择方式的一致化；本轮证据不支持仅把 mask 加到贪心部署端，因为 Masked Greedy 仍高度集中。正式整合前还应检验高需求空集任务的处理方案及多 seed 稳定性。本轮没有重新训练，也没有修改 PPO、Actor、reward 或 simulator 生产逻辑。", "",
        "复现命令（仓库根目录）：", "", "```bash",
        "/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python diagnostics/evaluate_reliability_masked_policy.py",
        "/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python diagnostics/write_reliability_masked_report.py",
        "/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python diagnostics/validate_reliability_masked_policy.py",
        "```", "",
        "20 个 episode 是同一 checkpoint/同一外生 trace 上的配对重复；bootstrap CI 按 episode 重采样，不能当作跨训练 seed 推断。完整配对差与 CI 见 `paired_episode_effects.csv`。", ""]
    target=ROOT/"diagnostics/results/RELIABILITY_MASKED_SELECTION.md"
    target.write_text("\n".join(lines),encoding="utf-8")
    return target


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=Path,default=DEFAULT)
    args=parser.parse_args()
    print(run(args.results_dir))
