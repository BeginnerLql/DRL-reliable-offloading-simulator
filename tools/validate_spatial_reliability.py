"""Step 4: paired spatial-field statistics and conditional reliability feasibility."""

import argparse
from itertools import combinations
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import simpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.configuration import parameters
from config.paths import RESULTS_DIR, SERVER_INFO_PATH, TRAIN_TASK_PATH
from core.env_state import EnvironmentState
from core.server import Server
from core.task import Task
from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    compute_baseline_correlation_length_km,
    map_spatial_risk_to_effective_failure_rates,
    sample_spatial_risk_field,
)


def sample_fields(correlation, sample_count, seed, stream):
    """Streams (4, 0) and (4, 1) isolate experiments A/B from runtime RNGs."""
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(4, stream)))
    epsilon = rng.standard_normal((sample_count, correlation.shape[0]))
    fields = {
        "correlated": sample_spatial_risk_field(
            correlation, standard_normal=epsilon.T
        ).T,
        "independent": sample_spatial_risk_field(
            np.eye(correlation.shape[0]), standard_normal=epsilon.T
        ).T,
    }
    np.testing.assert_array_equal(fields["independent"], epsilon)
    np.testing.assert_allclose(
        fields["correlated"], epsilon @ np.linalg.cholesky(correlation).T,
        rtol=0, atol=1e-14,
    )
    return fields


def field_statistics(fields, server_frame, beta):
    rows = []
    base_rates = server_frame.Base_Failure_Rate.to_numpy()
    for field_type, risk in fields.items():
        rates = map_spatial_risk_to_effective_failure_rates(base_rates, risk, beta)
        assert (rates > 0).all()
        for column, server in enumerate(server_frame.itertuples(index=False)):
            mean = risk[:, column].mean()
            variance = risk[:, column].var(ddof=1)
            std = risk[:, column].std(ddof=1)
            rate_mean = rates[:, column].mean()
            absolute_error = abs(rate_mean - server.Base_Failure_Rate)
            rows.append({
                "field_type": field_type,
                "server_id": server.Server_ID,
                "site_id": server.Site_ID,
                "theoretical_mean": 0.0,
                "empirical_mean": mean,
                "mean_error": mean,
                "theoretical_variance": 1.0,
                "empirical_variance": variance,
                "variance_error": variance - 1.0,
                "theoretical_std": 1.0,
                "empirical_std": std,
                "std_error": std - 1.0,
                "base_failure_rate": server.Base_Failure_Rate,
                "effective_failure_rate_mean": rate_mean,
                "effective_failure_rate_std": rates[:, column].std(ddof=1),
                "effective_failure_rate_mean_absolute_error": absolute_error,
                "effective_failure_rate_mean_relative_error": absolute_error / server.Base_Failure_Rate,
            })
    return pd.DataFrame(rows)


def batch_reliability(rates, execution_times, pairs, requirements):
    replicas = np.exp(-rates[:, None, :] * execution_times[None, :, :])
    left, right = replicas[:, :, pairs[:, 0]], replicas[:, :, pairs[:, 1]]
    reliability = 1.0 - (1.0 - left) * (1.0 - right)
    counts = (reliability >= requirements[None, :, None]).sum(axis=2)
    assert ((replicas > 0) & (replicas <= 1)).all()
    assert ((reliability > 0) & (reliability <= 1)).all()
    assert (reliability + 1e-15 >= np.maximum(left, right)).all()
    np.testing.assert_array_equal(reliability, 1.0 - (1.0 - right) * (1.0 - left))
    assert ((counts >= 0) & (counts <= len(pairs))).all()
    return replicas, reliability, counts


def check_runtime_consistency(
    state, servers, tasks, server_ids, distances, correlation, risk, rates,
    pairs, replicas, reliability, counts,
):
    """Compare three smoke scenarios with existing runtime formulas, all tasks/pairs."""
    for scene in range(min(3, len(risk))):
        state.set_episode_spatial_risk_context(
            server_ids, distances, correlation, risk[scene], rates[scene]
        )
        for task_index, task in enumerate(tasks):
            actual_replicas = [state.compute_replica_reliability(task, s) for s in servers]
            np.testing.assert_allclose(actual_replicas, replicas[scene, task_index], rtol=0, atol=1e-15)
            actual_pairs = []
            for left, right in pairs:
                a, b = servers[left], servers[right]
                value = state.compute_pair_reliability(task, a, b)
                assert value == state.compute_pair_reliability(task, b, a)
                assert state.is_reliability_requirement_satisfied(task, a, b) == (
                    value >= task.reliability_requirement
                )
                actual_pairs.append(value)
            np.testing.assert_allclose(actual_pairs, reliability[scene, task_index], rtol=0, atol=1e-15)
            assert counts[scene, task_index] == sum(
                value >= task.reliability_requirement for value in actual_pairs
            )


def feasibility_statistics(counts_by_type, requirements, pair_count):
    summaries, distributions, episodes = [], [], []
    groups = [(str(level), requirements == level) for level in parameters.TASK_RELIABILITY_REQUIREMENT_LEVELS]
    groups.append(("overall", np.ones(len(requirements), dtype=bool)))
    for field_type, counts in counts_by_type.items():
        for level, mask in groups:
            selected = counts[:, mask]
            flat = selected.ravel()
            episode_means = selected.mean(axis=1)
            episode_empty = (selected == 0).mean(axis=1)
            episode_all_safe = (selected == pair_count).mean(axis=1)
            summaries.append({
                "field_type": field_type,
                "reliability_requirement": level,
                "observations": flat.size,
                "safe_count_mean": flat.mean(),
                "safe_count_std": flat.std(ddof=1),
                "safe_count_min": flat.min(),
                "safe_count_max": flat.max(),
                "empty_fraction": (flat == 0).mean(),
                "all_safe_fraction": (flat == pair_count).mean(),
                "episode_safe_count_mean": episode_means.mean(),
                "episode_safe_count_mean_std": episode_means.std(ddof=1),
                "episode_empty_fraction_mean": episode_empty.mean(),
                "episode_empty_fraction_std": episode_empty.std(ddof=1),
            })
            frequencies = np.bincount(flat, minlength=pair_count + 1)
            for safe_count, frequency in enumerate(frequencies):
                distributions.append({
                    "field_type": field_type, "reliability_requirement": level,
                    "safe_count": safe_count, "frequency": frequency,
                })
            for scene, (mean, empty, all_safe) in enumerate(
                zip(episode_means, episode_empty, episode_all_safe), start=1
            ):
                episodes.append({
                    "field_type": field_type, "scene_id": scene,
                    "reliability_requirement": level, "safe_count_mean": mean,
                    "empty_fraction": empty, "all_safe_fraction": all_safe,
                })
    return pd.DataFrame(summaries), pd.DataFrame(distributions), pd.DataFrame(episodes)


def markdown_table(frame):
    """Keep report generation independent of optional tabulate dependencies."""
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.8g}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(args, server_frame, task_frame, distances, ell, matrices, stats, summary, distribution, episodes):
    pair_indices = np.triu_indices(len(server_frame), k=1)
    errors = []
    for field_type in ["correlated", "independent"]:
        theoretical = matrices[f"theoretical_{field_type}"]
        empirical = matrices[f"empirical_{field_type}"]
        difference = (empirical - theoretical)[pair_indices]
        approx_se = (1.0 - theoretical[pair_indices] ** 2) / np.sqrt(args.field_samples - 3)
        errors.append({
            "field_type": field_type, "MAE": np.abs(difference).mean(),
            "RMSE": np.sqrt(np.mean(difference ** 2)), "MaxAE": np.abs(difference).max(),
            "max_absolute_error_over_approx_se": np.max(np.abs(difference) / approx_se),
        })
    differences = []
    for level in summary.reliability_requirement.unique():
        subset = episodes[episodes.reliability_requirement == level]
        correlated = subset[subset.field_type == "correlated"].set_index("scene_id")
        independent = subset[subset.field_type == "independent"].set_index("scene_id")
        delta = correlated[["safe_count_mean", "empty_fraction"]] - independent[["safe_count_mean", "empty_fraction"]]
        differences.append({
            "reliability_requirement": level,
            "delta_safe_count_mean": delta.safe_count_mean.mean(),
            "delta_safe_count_mean_std": delta.safe_count_mean.std(ddof=1),
            "delta_empty_fraction": delta.empty_fraction.mean(),
            "delta_empty_fraction_std": delta.empty_fraction.std(ddof=1),
        })
    fluctuation_rows = []
    for field_type, subset in stats.groupby("field_type", sort=False):
        fluctuation_rows.append({
            "field_type": field_type,
            "max_absolute_mean_error_over_se": subset.empirical_mean.abs().max() * np.sqrt(args.field_samples),
            "max_absolute_variance_error_over_se": subset.variance_error.abs().max() / np.sqrt(2.0 / (args.field_samples - 1)),
            "max_rate_mean_relative_error": subset.effective_failure_rate_mean_relative_error.max(),
            "max_rate_mean_error_over_theoretical_se": subset.effective_failure_rate_mean_relative_error.max() / np.sqrt(np.expm1(parameters.SPATIAL_RISK_BETA_P ** 2) / args.field_samples),
        })
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
    conclusions = []
    fully_safe_levels, constrained_levels = [], []
    for level in summary.reliability_requirement.unique():
        subset = summary[summary.reliability_requirement == level].set_index("field_type")
        a, b = subset.loc["correlated"], subset.loc["independent"]
        conclusions.append(
            f"- {level}：相关/独立场平均安全动作数 {a.safe_count_mean:.6f}/{b.safe_count_mean:.6f}；"
            f"全部 28 对安全比例 {a.all_safe_fraction:.2%}/{b.all_safe_fraction:.2%}；"
            f"空安全集比例 {a.empty_fraction:.2%}/{b.empty_fraction:.2%}。"
        )
        if level != "overall":
            if (subset.all_safe_fraction == 1).all():
                fully_safe_levels.append(level)
            else:
                constrained_levels.append(level)
    conclusions.extend([
        "", f"本次两种场下均全部安全的档位：{', '.join(fully_safe_levels) or '无'}；出现约束区分度的档位：{', '.join(constrained_levels) or '无'}。",
        "相关场与独立场的平均值、跨场景波动及下尾分布存在上表所示的描述性差异；结论仅适用于这些固定任务、服务器与风险样本。",
        "本次没有观察到空安全集，不代表未采样风险环境或不同参数下空安全集不可能出现；不能据此确定 fallback 策略。",
        "当前参数可作为低风险基线继续检查负载与执行行为。若后续研究重点是可靠性约束或动作掩码的效果，应先讨论参数校准：目前可行性表显示大部分观测几乎不限制动作。",
        "已经具备定义和计算安全动作集合的数据基础；动作掩码、空安全集策略和可靠性约束训练方式仍待决定。",
        "没有优化策略、排队或负载实验，因此不能据此声称 PPO 性能改善、时延改善或实际故障率改善；配对描述差异也不是统计显著性或普遍优劣结论。",
    ])
    lines = [
        "# Step 4：空间随机场统计验证与可靠性可行性分析",
        "", "## 实验设置", "",
        f"代码版本：`{branch}@{head}`。固定实验种子 `{args.seed}`；NumPy `{np.__version__}`，pandas `{pd.__version__}`。",
        f"实验 A：{args.field_samples:,} 组，SeedSequence spawn_key=(4, 0)；实验 B：{args.scenarios} 组，spawn_key=(4, 1)。",
        "两个流分别重新派生；每组实验内 correlated/independent 使用同一组 epsilon，场景编号逐一配对。",
        f"服务器源：`data/server_info.xlsx`；任务源：`data/task_parameters.xlsx`，固定 {len(task_frame)} 个训练任务，按 Task_ID 排序。",
        f"beta={parameters.SPATIAL_RISK_BETA_P}；ell_0={ell:.12f} km；d_min/median/max={distances[pair_indices].min():.12f}/{np.median(distances[pair_indices]):.12f}/{distances[pair_indices].max():.12f} km。",
        "距离矩阵按 Server_ID 排序，28 对为 (1,2),(1,3),...,(7,8) 字典序；使用现有 Haversine 与相关长度函数。",
        f"基础故障率配置范围：{parameters.SERVER_FAILURE_RATE_RANGE} s^-1；下面列出实际固定样本。",
        "CPU 为 cycles/s，上传速率 MB/s，故障率 s^-1；位置为纬度/经度角度。",
        "", markdown_table(server_frame), "",
        "任务大小 5–10 MB，计算需求 0.5e9–3e9 cycles；可靠性要求 0.9/0.99/0.999/0.9999，每档 50 个，输入不重新生成。",
        "模型：Sigma=exp(-D/ell_0)，Z_corr=L epsilon，Z_ind=epsilon；lambda_eff=lambda_0 exp(beta Z-beta^2/2)。",
        "R_single=exp(-lambda_eff C/f)，R_pair=1-(1-R_n)(1-R_m)，safe_count=sum(R_pair>=R_req)。",
        "双副本公式为给定风险场后的条件独立模型，没有附加相关系数或距离惩罚。没有实际失败事件、到达采样或 CPU 排队。",
        "", "## 实验 A：节点边际与有效故障率", "",
        "所有方差/标准差采用样本估计 ddof=1。mean_error/variance_error/std_error 是有符号误差；有效故障率均值的误差另列绝对与相对绝对误差。",
        markdown_table(stats), "", "## 理论与经验相关矩阵", "",
    ]
    for name, matrix in matrices.items():
        if name in ["server_ids", "distances_km"]:
            continue
        frame = pd.DataFrame(matrix, columns=[f"server{n}" for n in server_frame.Server_ID])
        frame.insert(0, "server_id", server_frame.Server_ID.to_numpy())
        lines.extend([f"### {name}", "", markdown_table(frame), ""])
    lines.extend([
        "误差指标仅取 28 个严格上三角元素，排除对角线及对称重复项；独立场的理论非对角值为 0。",
        "", markdown_table(pd.DataFrame(errors)), "",
        "## 有限样本波动解释", "",
        f"M={args.field_samples:,} 下，标准正态样本均值标准误约 {1/np.sqrt(args.field_samples):.6g}，方差标准误约 {np.sqrt(2/(args.field_samples-1)):.6g}。",
        f"有效故障率均值的理论相对标准误为 sqrt((exp(beta^2)-1)/M)={np.sqrt(np.expm1(parameters.SPATIAL_RISK_BETA_P**2)/args.field_samples):.6g}。",
        "相关系数标准误近似为 (1-rho^2)/sqrt(M-3)。下表将实际误差按其理论波动量级展示；这不是任意 MAE/RMSE 验收阈值，也不是显著性检验。",
        "", markdown_table(pd.DataFrame(fluctuation_rows)), "",
        "同一类型节点与服务器对的估计量相互关联，多个指标中的最大误差也不能当作单个独立估计量解释。期望关系不要求有限样本均值严格相等。",
        "", "## 实验 B：任务—场景级及 Episode 级统计", "",
        f"每种场类型 {args.scenarios} x {len(task_frame)} x 28 = {args.scenarios*len(task_frame)*28:,} 次检查，合计 {2*args.scenarios*len(task_frame)*28:,} 次。",
        "每档 observations 为 50×场景数；overall 为 200×场景数。固定任务、同一场景共享风险环境，所以不能把所有任务—场景观测当作独立样本。",
        "safe_count_std 描述任务—场景级分布；episode_safe_count_mean_std 描述每场景档内平均安全动作数的跨场景波动；episode_empty_fraction_std 同理。",
        "", markdown_table(summary), "",
        "## 0–28 安全动作频数分布", "",
        "每行对应一个风险场类型与可靠性档位；0–28 列完整保留零频数，不丢弃空安全集。",
        "", markdown_table(distribution.pivot(index=["field_type", "reliability_requirement"], columns="safe_count", values="frequency").reset_index().rename(columns=lambda c: str(c))), "",
        "## 配对描述性差异：correlated − independent", "",
        f"每个 scene_id 先作差，再汇总；std 为 {args.scenarios} 个场景差值的样本标准差，比例差值单位是比例（乘 100 后为百分点）。",
        "", markdown_table(pd.DataFrame(differences)), "",
        "两种场的节点边际分布相同，但多节点有效故障率的联合分布不同，因此安全动作分布可以不同；不能预设某一种总是更好。",
        "", "## 一致性检查与输出说明", "",
        "先 smoke 检查前三个场景的全部任务和 28 对；批量结果与 EnvironmentState 单副本/双副本/判定接口一致。",
        "检查了 ID/配对顺序、相关矩阵对称/单位对角/Cholesky、同基础样本、可靠性范围及交换对称、双副本下界、安全数量范围。",
        "分别重新派生 A/B 样本及 B 安全数量，确认相同种子数值与次序可复现；未调用 MainLoop 或改动正式仿真 RNG。",
        "仅输出 field_statistics.csv、correlation_matrices.npz、feasibility_summary.csv、safe_count_distribution.csv、episode_feasibility.csv、report.md。",
        "NPZ 中保存 server_ids/distances_km 和四个理论/经验矩阵；episode_feasibility.csv 保留逐场景分组汇总，方便重新计算配对描述差值。",
        "", "## 结论与后续建议", "",
        "节点边际、相关系数和有效故障率均值的偏差应结合上面的标准误量级解释，不据此修改样本或模型参数。",
        "",
        *conclusions,
        "未训练 PPO，未改参数，未开展负载扫描；是否先校准参数由观察到的约束区分度和研究目标决定。",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=parameters.EXPERIMENT_SEED)
    parser.add_argument("--field-samples", type=int, default=10000)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR) / "step4_spatial_reliability")
    args = parser.parse_args()
    server_frame = pd.read_excel(SERVER_INFO_PATH, sheet_name="Servers", dtype={"Site_ID": "string"}).sort_values("Server_ID").reset_index(drop=True)
    task_frame = pd.read_excel(TRAIN_TASK_PATH).sort_values("Task_ID").reset_index(drop=True)
    env, state = simpy.Environment(), EnvironmentState()
    servers = [Server(env, row.Server_ID, row.Site_ID, row.Processing_Frequency, row.Base_Failure_Rate, row.Transmission_Rate, row.Latitude, row.Longitude) for row in server_frame.itertuples(index=False)]
    for server in servers:
        state.add_server_and_init_environment(server)
    tasks = [Task(env, state, row["Task_ID"], row) for row in task_frame.to_dict("records")]
    server_ids, distances = build_distance_matrix(servers)
    ell = compute_baseline_correlation_length_km(distances)
    correlation = build_spatial_correlation_matrix(distances, ell)
    pairs = np.array(list(combinations(range(len(servers)), 2)))
    assert server_ids == sorted(server_ids) and task_frame.Task_ID.tolist() == sorted(task_frame.Task_ID.tolist())
    assert len(servers) == 8 and len(tasks) == 200 and len(pairs) == 28
    assert len({tuple(pair) for pair in pairs}) == 28 and (pairs[:, 0] < pairs[:, 1]).all()
    np.testing.assert_array_equal(correlation, correlation.T)
    np.testing.assert_array_equal(np.diag(correlation), np.ones(len(servers)))
    np.linalg.cholesky(correlation)
    base_rates = server_frame.Base_Failure_Rate.to_numpy()
    execution_times = task_frame.Computation_Demand.to_numpy()[:, None] / server_frame.Processing_Frequency.to_numpy()[None, :]
    requirements = task_frame.Reliability_Requirement.to_numpy()
    smoke_fields = sample_fields(correlation, min(3, args.scenarios), args.seed, 1)
    for field_type, risk in smoke_fields.items():
        matrix = correlation if field_type == "correlated" else np.eye(len(servers))
        rates = map_spatial_risk_to_effective_failure_rates(base_rates, risk, parameters.SPATIAL_RISK_BETA_P)
        replicas, reliability, counts = batch_reliability(rates, execution_times, pairs, requirements)
        check_runtime_consistency(state, servers, tasks, server_ids, distances, matrix, risk, rates, pairs, replicas, reliability, counts)
    print("Smoke consistency checks passed for both field types (three scenarios).")
    fields_a = sample_fields(correlation, args.field_samples, args.seed, 0)
    fields_b = sample_fields(correlation, args.scenarios, args.seed, 1)
    counts_by_type = {}
    for field_type, risk in fields_b.items():
        rates = map_spatial_risk_to_effective_failure_rates(base_rates, risk, parameters.SPATIAL_RISK_BETA_P)
        _, _, counts_by_type[field_type] = batch_reliability(rates, execution_times, pairs, requirements)
    for stream, fields, sample_count in [(0, fields_a, args.field_samples), (1, fields_b, args.scenarios)]:
        repeated = sample_fields(correlation, sample_count, args.seed, stream)
        for field_type in fields:
            np.testing.assert_array_equal(fields[field_type], repeated[field_type])
            if stream == 1:
                rates = map_spatial_risk_to_effective_failure_rates(base_rates, repeated[field_type], parameters.SPATIAL_RISK_BETA_P)
                _, _, repeated_counts = batch_reliability(rates, execution_times, pairs, requirements)
                np.testing.assert_array_equal(counts_by_type[field_type], repeated_counts)
    stats = field_statistics(fields_a, server_frame, parameters.SPATIAL_RISK_BETA_P)
    summary, distribution, episodes = feasibility_statistics(counts_by_type, requirements, len(pairs))
    matrices = {
        "server_ids": np.asarray(server_ids), "distances_km": distances,
        "theoretical_correlated": correlation,
        "empirical_correlated": np.corrcoef(fields_a["correlated"], rowvar=False),
        "theoretical_independent": np.eye(len(servers)),
        "empirical_independent": np.corrcoef(fields_a["independent"], rowvar=False),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in [("field_statistics.csv", stats), ("feasibility_summary.csv", summary), ("safe_count_distribution.csv", distribution), ("episode_feasibility.csv", episodes)]:
        frame.to_csv(args.output_dir / filename, index=False, float_format="%.17g")
    np.savez_compressed(args.output_dir / "correlation_matrices.npz", **matrices)
    write_report(args, server_frame, task_frame, distances, ell, matrices, stats, summary, distribution, episodes)
    print(f"Completed A={args.field_samples:,}, B={args.scenarios}; ell_0={ell:.12f} km; outputs: {args.output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
