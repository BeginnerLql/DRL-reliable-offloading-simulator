"""Offline GH scans of fixed profiles without changing simulator data or parameters."""

import argparse
import json
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import subprocess
import shlex
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH
from core.task import compute_task_deadlines
from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    compute_baseline_correlation_length_km,
    compute_marginal_failure_probability,
    compute_pair_reliability_gh,
)

SCALES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
THRESHOLDS = (0.9, 0.99, 0.999, 0.9999, 0.99999)
CALIBRATION_SCALES = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)
CALIBRATION_THRESHOLDS = (0.99, 0.999, 0.9999, 0.99999)
ORDERS = (8, 12, 16, 24, 32)
BETA = 0.5
TOLERANCE = 1e-12
LENGTH_RATIOS = (0., 0.25, 0.5, 1., 2., 4.)
LENGTH_THRESHOLDS = (0.999, 0.9999)
LENGTH_ORDERS = (16, 24, 32)
LENGTH_SCALE = 0.05
MIXED_CANDIDATES = {
    'A': (0.9, 0.99, 0.999, 0.9999),
    'B': (0.99, 0.999, 0.9995, 0.9999),
    'C': (0.999, 0.9993, 0.9996, 0.9999),
}
MIXED_ORDER = 32


def safe_pair_summary(correlated, independent, threshold):
    safe_corr, safe_ind = correlated >= threshold, independent >= threshold
    counts_corr, counts_ind = safe_corr.sum(axis=1), safe_ind.sum(axis=1)
    false_safe_mask = safe_ind & ~safe_corr
    false_safe = int(np.count_nonzero(false_safe_mask))
    affected_tasks = int(np.count_nonzero(false_safe_mask.any(axis=1)))
    summary = {
        "mean_safe_pairs_corr": float(counts_corr.mean()),
        "tasks_0_corr": int(np.count_nonzero(counts_corr == 0)),
        "tasks_1_corr": int(np.count_nonzero(counts_corr == 1)),
        "tasks_2plus_corr": int(np.count_nonzero(counts_corr >= 2)),
        "feasible_fraction_corr": float(np.mean(counts_corr > 0)),
        "mean_safe_pairs_ind": float(counts_ind.mean()),
        "tasks_0_ind": int(np.count_nonzero(counts_ind == 0)),
        "tasks_1_ind": int(np.count_nonzero(counts_ind == 1)),
        "tasks_2plus_ind": int(np.count_nonzero(counts_ind >= 2)),
        "feasible_fraction_ind": float(np.mean(counts_ind > 0)),
        "false_safe_pairs": false_safe,
        "false_safe_tasks": affected_tasks,
        "false_safe_task_fraction": affected_tasks / len(counts_corr),
        "false_unsafe_pairs": int(np.count_nonzero(safe_corr & ~safe_ind)),
        "false_safe_fraction_all_pairs": false_safe / safe_ind.size,
        "false_safe_fraction_ind_safe": (
            false_safe / np.count_nonzero(safe_ind) if np.any(safe_ind) else np.nan
        ),
        "fallback_fraction": float(np.mean(counts_corr == 0)),
        "empty_safe_fraction_ind": float(np.mean(counts_ind == 0)),
    }
    hist_corr = np.bincount(counts_corr, minlength=correlated.shape[1] + 1)
    hist_ind = np.bincount(counts_ind, minlength=independent.shape[1] + 1)
    distribution = [{
        "safe_pair_count": count,
        "tasks_corr": int(corr), "fraction_corr": corr / len(counts_corr),
        "tasks_ind": int(ind), "fraction_ind": ind / len(counts_ind),
    } for count, (corr, ind) in enumerate(zip(hist_corr, hist_ind))]
    return summary, distribution


def run_scan(servers, tasks, pairs, correlations, *, scales=SCALES, thresholds=THRESHOLDS):
    demands = tasks["Computation_Demand"].to_numpy()[:, None]
    frequencies = servers["Processing_Frequency"].to_numpy()
    base_rates = servers["Base_Failure_Rate"].to_numpy()
    independent_rho = build_spatial_correlation_matrix(
        np.zeros((len(servers), len(servers))), 0,
    )[pairs[:, 0], pairs[:, 1]]
    scan_rows, convergence_rows, validation_rows = [], [], []
    distribution_rows, difference_rows = [], []
    for scale in scales:
        rates = base_rates * scale
        by_order = {}
        for order in ORDERS:
            started = perf_counter()
            marginal = compute_marginal_failure_probability(
                demands, frequencies, rates, BETA, quadrature_order=order,
            )
            corr, ind = compute_pair_reliability_gh(
                demands, frequencies[pairs], rates[pairs], correlations, BETA,
                quadrature_order=order,
            )
            elapsed = perf_counter() - started
            by_order[order] = (corr, ind, marginal, elapsed)
            zero_corr, zero_ind = compute_pair_reliability_gh(
                demands, frequencies[pairs], rates[pairs], independent_rho, BETA,
                quadrature_order=order,
            )
            lower = np.maximum(1 - marginal[:, pairs[:, 0]], 1 - marginal[:, pairs[:, 1]])
            probabilities = np.concatenate([
                value.ravel() for value in (marginal, corr, ind, zero_corr, zero_ind)
            ])
            validation_rows.append({
                "failure_rate_scale": scale, "gh_order": order,
                "min_probability": float(probabilities.min()),
                "max_probability": float(probabilities.max()),
                "nonfinite_values": int(np.count_nonzero(~np.isfinite(probabilities))),
                "out_of_range_values": int(np.count_nonzero(
                    (probabilities < 0) | (probabilities > 1))),
                "lower_bound_violations": int(np.count_nonzero(corr < lower - TOLERANCE)),
                "upper_bound_violations": int(np.count_nonzero(corr > ind + TOLERANCE)),
                "max_lower_bound_excess": float(np.maximum(lower - corr, 0).max()),
                "max_upper_bound_excess": float(np.maximum(corr - ind, 0).max()),
                "ell_zero_max_abs_error": float(np.max(np.abs(zero_corr - zero_ind))),
                "independent_baseline_max_abs_error": float(np.max(np.abs(
                    ind - (1 - marginal[:, pairs[:, 0]] * marginal[:, pairs[:, 1]])))),
            })

        ref_corr, ref_ind, ref_marginal, _ = by_order[32]
        difference = ref_ind - ref_corr
        difference_rows.append({
            "failure_rate_scale": scale, "candidate_pairs": difference.size,
            "mean": float(difference.mean()), "std": float(difference.std()),
            **dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                       np.quantile(difference, (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)))),
        })
        for threshold in thresholds:
            summary, distribution = safe_pair_summary(ref_corr, ref_ind, threshold)
            metadata = {"failure_rate_scale": scale, "threshold": threshold}
            scan_rows.append({
                **metadata, **summary,
            })
            distribution_rows.extend({**metadata, **row} for row in distribution)
        previous = None
        for order, (corr, ind, marginal, elapsed) in by_order.items():
            safe = corr[..., None] >= thresholds
            reference_safe = ref_corr[..., None] >= thresholds
            convergence_rows.append({
                "failure_rate_scale": scale, "gh_order": order,
                "max_abs_corr_error_vs_32": float(np.max(np.abs(corr - ref_corr))),
                "mean_abs_corr_error_vs_32": float(np.mean(np.abs(corr - ref_corr))),
                "max_abs_ind_error_vs_32": float(np.max(np.abs(ind - ref_ind))),
                "max_abs_marginal_error_vs_32": float(np.max(np.abs(marginal - ref_marginal))),
                "max_abs_corr_change_from_previous": (
                    float(np.max(np.abs(corr - previous))) if previous is not None else np.nan
                ),
                "safe_pair_changes_vs_32": int(np.count_nonzero(safe != reference_safe)),
                "ind_safe_pair_changes_vs_32": int(np.count_nonzero(
                    (ind[..., None] >= thresholds) != (ref_ind[..., None] >= thresholds))),
                "false_safe_vs_32": int(np.count_nonzero(safe & ~reference_safe)),
                "false_unsafe_vs_32": int(np.count_nonzero(~safe & reference_safe)),
                "safe_pair_changes_from_previous": (
                    int(np.count_nonzero(safe != (previous[..., None] >= thresholds)))
                    if previous is not None else 0
                ),
                "elapsed_s": elapsed,
            })
            previous = corr
        print(f"scale={scale:g}: GH 8/12/16/24/32 completed", flush=True)
    return tuple(pd.DataFrame(rows) for rows in (
        scan_rows, convergence_rows, validation_rows, distribution_rows, difference_rows,
    ))


def check_edge_cases():
    """Deterministic analytic limits, including perfectly correlated latent risks."""
    demand, frequencies, rates = 3e9, np.array([2e9, 4e9]), np.array([0.2, 0.4])
    expected = 1 - np.prod(-np.expm1(-rates * demand / frequencies))
    for rho in (0.0, 0.5, 1.0):
        corr, ind = compute_pair_reliability_gh(demand, frequencies, rates, rho, 0)
        np.testing.assert_allclose([corr, ind], expected, rtol=0, atol=TOLERANCE)
        corr, ind = compute_pair_reliability_gh(0, frequencies, rates, rho)
        np.testing.assert_array_equal([corr, ind], [1, 1])
        corr, ind = compute_pair_reliability_gh(demand, frequencies, [0, 0.4], rho)
        np.testing.assert_array_equal([corr, ind], [1, 1])
    corr, _ = compute_pair_reliability_gh(demand, frequencies, rates, 1.0)
    marginal = compute_marginal_failure_probability(demand, frequencies[0], rates[0])
    assert 1 - marginal <= corr <= 1 - marginal**2 + TOLERANCE
    for invalid_length in (-1, np.nan, np.inf):
        try:
            build_spatial_correlation_matrix(np.zeros((2, 2)), invalid_length)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid correlation length accepted")
    for order in ORDERS:
        marginal = compute_marginal_failure_probability(1e10, 1, 1, 0, quadrature_order=order)
        corr, ind = compute_pair_reliability_gh(
            1e10, [1, 1], [1, 1], 0.5, 0, quadrature_order=order,
        )
        np.testing.assert_array_equal([marginal, corr, ind], [1, 0, 0])


def write_report(output, scan, convergence, validation, ell, hashes, tasks):
    summary = convergence.groupby("gh_order").agg(
        max_error=("max_abs_corr_error_vs_32", "max"),
        safe_changes=("safe_pair_changes_vs_32", "sum"),
        total_seconds=("elapsed_s", "sum"),
    )
    lines = [
        "# 未知空间风险场双副本可靠性扫描", "",
        f"- HEAD: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()}`",
        "- 数据：当前 server_info.xlsx（8 台）及 task_parameters.xlsx（200 个固定训练任务，按 Task_ID 排序）。",
        "- 28 个服务器对按 Server_ID 字典序排列，映射见 server_pairs.csv。",
        f"- beta={BETA}；ell0={ell:.12g} km；服务器对距离中位数={ell * np.log(2):.12g} km。",
        "- 仅 CPU 时间 C/f 暴露于故障；给定风险场后节点故障条件独立。",
        "- R_corr=1-E[q_m q_n]；R_ind=1-qbar_m*qbar_n，边际使用相同的对数正态风险模型。",
        "- GH 使用 Z=sqrt(2)x、标准正态权重 w/sqrt(pi)，二维张量积求积。",
        "- 以浮点权重总和归一化期望，避免饱和概率出现机器精度越界；不裁剪概率。",
        "- scan.csv 采用每维 32 阶；误差均以 32 阶为数值参考，并非已知真值。",
        "- 所有安全判定直接使用 R >= threshold；不向判定加入数值容差。",
        f"- 不等式验证容差 {TOLERANCE:g}；范围验证严格使用 [0,1]，未做裁剪。",
        "- 原任务阈值分布：" + str(tasks["Reliability_Requirement"].value_counts().sort_index().to_dict()) + "；探索扫描对所有任务统一应用每个候选阈值。",
        "- SHA256：" + "；".join(f"{Path(path).name} `{digest}`" for path, digest in hashes.items()),
        "", "## GH 收敛", "",
        "| 每维阶数 | 最大绝对可靠性误差（对比32阶） | 安全判定变化数 | 六组缩放总耗时/s |",
        "|---:|---:|---:|---:|",
    ]
    for order, row in summary.iterrows():
        lines.append(f"| {order} | {row.max_error:.6g} | {int(row.safe_changes)} | {row.total_seconds:.6f} |")
    lines += [
        "", "安全判定变化数覆盖 6 个缩放 × 200 任务 × 28 对 × 5 阈值。",
        "计时为单次批量边际及双副本计算（包含 GH 节点构造），不含 I/O、ell=0 及其他验证；运行时间有系统波动。",
        "gh_convergence.csv 另含相邻阶数变化、边际/独立基线误差及假安全/假不安全计数。", "",
        "## 故障率与阈值扫描", "",
        "| 缩放 | 阈值 | 平均安全对 corr / ind | 任务数 0 / 1 / ≥2（corr） | 可行任务 corr / ind | 独立误判占全部对 | 独立误判占其安全对 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scan.itertuples():
        conditional = (f"{row.false_safe_fraction_ind_safe:.2%}"
                       if np.isfinite(row.false_safe_fraction_ind_safe) else "N/A")
        lines.append(
            f"| {row.failure_rate_scale:g} | {row.threshold:g} "
            f"| {row.mean_safe_pairs_corr:.3f} / {row.mean_safe_pairs_ind:.3f} "
            f"| {row.tasks_0_corr} / {row.tasks_1_corr} / {row.tasks_2plus_corr} "
            f"| {row.feasible_fraction_corr:.1%} / {row.feasible_fraction_ind:.1%} "
            f"| {row.false_safe_fraction_all_pairs:.2%} | {conditional} |"
        )
    worst = scan.loc[scan.false_safe_fraction_all_pairs.idxmax()]
    baseline = scan[(scan.failure_rate_scale == 1) & (scan.threshold == 0.9)].iloc[0]
    feasibility_levels = ", ".join(f"{value:.0%}" for value in sorted(scan.feasible_fraction_corr.unique()))
    lines += [
        "", "全部对的误判分母为每组参数的 200×28=5600；条件误判分母为独立模型判安全的对数，分母为零时 CSV 留空。",
        "完整的两种模型任务分布和可行比例见 scan.csv。",
        f"全部对误判比例最高：缩放 {worst.failure_rate_scale:g}、阈值 {worst.threshold:g}，"
        f"{int(worst.false_safe_pairs)}/5600={worst.false_safe_fraction_all_pairs:.2%}。",
        f"当前缩放1.0、阈值0.9时平均有{baseline.mean_safe_pairs_corr:.3f}个安全对；"
        "更高阈值的可行性见上表。",
        f"本次粗网格相关模型的任务可行比例取值为 {feasibility_levels}；"
        f"出现恰好1个安全对任务的参数组合有 {int((scan.tasks_1_corr > 0).sum())} 组。",
        "这些结果仅用于后续选参，不自动确定正式参数。",
        "", "## 数值检查", "",
    ]
    for column in ("nonfinite_values", "out_of_range_values", "lower_bound_violations", "upper_bound_violations"):
        lines.append(f"- {column}: {int(validation[column].sum())}")
    for column in ("ell_zero_max_abs_error", "independent_baseline_max_abs_error"):
        lines.append(f"- {column}: {validation[column].max():.6g}")
    lines += [
        f"- 全部已检查概率范围：[{validation.min_probability.min():.12g}, {validation.max_probability.max():.12g}]。",
        f"- 独立基线跨 GH 阶数的安全集合变化数：{int(convergence.ind_safe_pair_changes_vs_32.sum())}。",
        "- beta=0 解析极限、C=0、零故障率、rho=1、饱和概率、负/非有限 ell 拒绝检查通过。",
        "- 未使用 Monte Carlo；未训练 PPO；未改动数据、正式故障率或任务阈值。",
        "", "## 下一阶段接口", "",
        "- core/env_state.py：用边际/联合 GH 函数替换已采样风险场下的可靠性查询，按固定任务和28对预计算安全集合。",
        "- core/main_loop.py：生成并传递动作 mask，处理零安全对任务，下一阶段再统一处理 Episode 风险场采样。",
        "- agents/ppo_agent.py：select_action、store_transition、train_step 使用同一 mask，保存 rollout mask，确保新旧 log-prob 和熵支持集一致。",
        "- config/configuration.py 与 config/params.py：查看扫描后再确定正式缩放、阈值、GH 阶数和零安全对策略；状态接口的未知风险含义需同步确认。",
        "- compare_ppo_greedy.py 与 io_utils/save_parameters_and_logs.py：后续比较与输出需使用同一可靠性口径并记录 mask/不可行任务。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_report(output, scan, convergence, validation, differences, ell, hashes):
    scores = scan[scan.threshold.isin((0.999, 0.9999))].groupby("failure_rate_scale").agg(
        min_false_safe_fraction=("false_safe_fraction_all_pairs", "min"),
        max_fallback=("fallback_fraction", "max"),
        min_affected_task_fraction=("false_safe_task_fraction", "min"),
    )
    shared = scores[(scores.min_false_safe_fraction > 0) & (scores.max_fallback < 1)]
    shared = shared.sort_values(["min_false_safe_fraction", "max_fallback"], ascending=[False, True])
    lines = [
        "# 可靠性阈值 0.999 / 0.9999 的共同故障率校准", "",
        f"固定8台服务器、200个任务、28个字典序组合；beta={BETA}，ell0={ell:.12g} km。",
        "直接复用 core/spatial_risk.py 的 GH 联合可靠性及公平独立基线，数学模型与数据未修改。",
        "全部任务分别应用每个候选阈值，未按原任务阈值分组。主结果使用每维32阶。", "",
        "执行命令：`/home/user/anaconda3/bin/python tools/scan_reliability.py --calibrate`。",
        "已有输出目录会拒绝覆盖；复跑可用 `--output-dir` 指定新目录。", "",
        "- 假安全：R_ind >= threshold 且 R_corr < threshold。",
        "- 全部候选对分母为每组参数200×28=5600；条件比例分母为独立模型安全对数，零分母记N/A。",
        "- 受影响任务比例：至少有一个假安全对的任务数/200。",
        "- Fallback比例：相关安全集合为空的任务数/200，Masked和No-Mask均按相关Mask触发；任务仍全部执行。",
        "- 独立模型空集比例仅为离线对照，不作为No-Mask PPO的Fallback触发条件。",
        "- 差值DeltaR=R_ind-R_corr，统计5600个任务—服务器对；差值与阈值无关，每个缩放只存一行分布。",
        "- 固定样本上的集合差异为描述性结果，不作统计显著性或总体泛化推断。",
        "- 本轮仅给共同缩放候选，不为不同阈值分别设置故障率，也不自动修改正式参数。", "",
        "## 参数扫描", "",
        "| 缩放 | 阈值 | 平均安全对 corr / ind | 假安全数量 | 占全部对 | 占独立安全对 | 受影响任务 | Fallback / 独立空集 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scan.itertuples():
        conditional = (f"{row.false_safe_fraction_ind_safe:.2%}"
                       if np.isfinite(row.false_safe_fraction_ind_safe) else "N/A")
        lines.append(
            f"| {row.failure_rate_scale:g} | {row.threshold:g} "
            f"| {row.mean_safe_pairs_corr:.3f} / {row.mean_safe_pairs_ind:.3f} "
            f"| {row.false_safe_pairs} | {row.false_safe_fraction_all_pairs:.2%} | {conditional} "
            f"| {row.false_safe_task_fraction:.1%} "
            f"| {row.fallback_fraction:.1%} / {row.empty_safe_fraction_ind:.1%} |"
        )
    lines += [
        "", "## 共同候选", "",
        "共同候选的最低条件：两个主要阈值均存在假安全对，且相关模型均有任务可行。",
        "为研究两个阈值共同的集合差异，优先比较两阈值中较弱的假安全对比例，再比较较高的Fallback比例；这是描述性选参准则，不是统计显著性检验。",
    ]
    if shared.empty:
        lines.append("本次网格没有满足最低条件的共同候选。需根据上表区分集合饱和与任务全部不可行，不能强行认定有共同设置。")
    else:
        lines += [
            f"优先建议共同缩放 **{shared.index[0]:g}**；满足最低条件的候选为 "
            + ", ".join(f"{scale:g}" for scale in shared.index) + "。",
            "",
            "| 缩放 | 两阈值中较弱的假安全对比例 | 较低的受影响任务比例 | 较高的Fallback比例 |",
            "|---:|---:|---:|---:|",
        ]
        for scale, row in shared.iterrows():
            lines.append(f"| {scale:g} | {row.min_false_safe_fraction:.3%} "
                         f"| {row.min_affected_task_fraction:.1%} | {row.max_fallback:.1%} |")
        lines.append(f"建议的同一组固定基础故障率为原8台服务器Base_Failure_Rate分别乘以{shared.index[0]:g}，"
                     "两个主要阈值共用这组值；此处仅给建议，不写回数据。")
        lines.append("满足最低条件并不代表两个阈值的差异同样明显，需结合假安全数量及受影响任务比例判断。")
        zero_fallback = shared[shared.max_fallback == 0]
        if not zero_fallback.empty:
            lines.append(f"如要求两个阈值下均无需Fallback，备选共同缩放为 **{zero_fallback.index[0]:g}**，"
                         "但较弱阈值的集合差异较小。所有候选均执行全部任务。")
    lines += [
        "", "## 可靠性差值分布", "",
        "| 缩放 | min | p05 | p25 | p50 | p75 | p95 | max | mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in differences.itertuples():
        lines.append(f"| {row.failure_rate_scale:g} | " + " | ".join(
            f"{getattr(row, name):.6g}" for name in ("min", "p05", "p25", "p50", "p75", "p95", "max", "mean")
        ) + " |")
    lines += [
        "", "## 分布文件与验证", "",
        "- scan.csv：32组参数的汇总，包括0/1/≥2安全对任务数。",
        "- safe_pair_distribution.csv：每组参数0～28个安全对的完整任务计数与比例，分别列出两个模型。",
        "- reliability_difference.csv：每个缩放的DeltaR分布，另含总体标准差（ddof=0）。",
        "- gh_convergence.csv、validation.csv：复用第一阶段五种阶数的精度、时间、范围、上下界及ell=0验证。",
        f"- 24阶对比32阶最大误差：{convergence.loc[convergence.gh_order == 24, 'max_abs_corr_error_vs_32'].max():.6g}。",
        f"- 跨阶数相关/独立安全判定变化总数：{int(convergence.safe_pair_changes_vs_32.sum())} / {int(convergence.ind_safe_pair_changes_vs_32.sum())}。",
        f"- 非有限/范围/上下界异常总数：{int(validation[['nonfinite_values', 'out_of_range_values', 'lower_bound_violations', 'upper_bound_violations']].to_numpy().sum())}。",
        f"- ell=0相关与独立基线最大差异：{validation.ell_zero_max_abs_error.max():.6g}。",
        f"- 假不安全对（相关判安全、独立判不安全）：{int(scan.false_unsafe_pairs.sum())}。",
        "- 未运行PPO或Monte Carlo，未修改基础故障率文件或任务阈值。",
        "- SHA256：" + "；".join(f"{Path(path).name} `{digest}`" for path, digest in hashes.items()),
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_length_sweep(servers, task_sets, pairs, ids, distances, ell0, output):
    """Offline GH only: keep marginals fixed and vary latent pair correlation."""
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(p) for p in (SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH)]
    paths += [Path(__file__).resolve(), PROJECT_ROOT/'core/spatial_risk.py',
              PROJECT_ROOT/'config/configuration.py', PROJECT_ROOT/'config/params.py']
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]),
        git_branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=PROJECT_ROOT, text=True).strip(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip(),
        git_status_before=subprocess.check_output(['git', 'status', '--short'], cwd=PROJECT_ROOT, text=True),
        alpha=LENGTH_SCALE, beta=BETA, gh_main_order=16, gh_check_orders=[24,32],
        ell0_km=ell0, ell_ratios=LENGTH_RATIOS, thresholds=LENGTH_THRESHOLDS,
        server_count=8, pair_count=28, task_counts={k:len(v) for k,v in task_sets.items()},
        numpy=np.__version__, pandas=pd.__version__, python=sys.version, input_sha256=hashes)
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    pd.DataFrame({'pair_index':np.arange(28), 'server_m':np.array(ids)[pairs[:,0]],
                  'server_n':np.array(ids)[pairs[:,1]], 'distance_km':distances[pairs[:,0],pairs[:,1]]}).to_csv(
        output/'server_pairs.csv', index=False, float_format='%.17g')
    expected_pairs = np.array(list(combinations(ids, 2)))
    np.testing.assert_array_equal(np.array(ids)[pairs], expected_pairs)
    prior = PROJECT_ROOT/'results/masked_ppo_confirmation_10seed/reliability_table.npz'
    if prior.exists():
        with np.load(prior) as old:
            np.testing.assert_array_equal(old['pairs'], expected_pairs)
    frequencies = servers.Processing_Frequency.to_numpy()
    rates = servers.Base_Failure_Rate.to_numpy() * LENGTH_SCALE
    summary_rows, distribution_rows, false_rows, validation_rows = [], [], [], []
    saved = dict(server_ids=np.array(ids), pairs=expected_pairs, distances_km=distances,
                 scaled_rates=rates, ell_ratios=LENGTH_RATIOS, thresholds=LENGTH_THRESHOLDS)
    for split, tasks in task_sets.items():
        assert len(tasks)==200 and tasks.Task_ID.is_unique and tasks.Task_ID.is_monotonic_increasing
        demand = tasks.Computation_Demand.to_numpy()[:,None]
        by_order = {}
        for order in LENGTH_ORDERS:
            marginal = compute_marginal_failure_probability(demand, frequencies, rates, BETA, quadrature_order=order)
            baseline = 1-marginal[:,pairs[:,0]]*marginal[:,pairs[:,1]]
            lower = np.maximum(1-marginal[:,pairs[:,0]], 1-marginal[:,pairs[:,1]])
            values = []
            for ratio in LENGTH_RATIOS:
                matrix = build_spatial_correlation_matrix(distances, ratio*ell0)
                rho = matrix[pairs[:,0],pairs[:,1]]
                start = perf_counter()
                corr, ind = compute_pair_reliability_gh(demand, frequencies[pairs], rates[pairs], rho,
                                                       BETA, quadrature_order=order)
                elapsed = perf_counter()-start
                assert corr.shape==ind.shape==(200,28)
                for array in (marginal, corr, ind):
                    assert np.isfinite(array).all() and ((array>=0)&(array<=1)).all()
                np.testing.assert_allclose(ind, baseline, rtol=0, atol=TOLERANCE)
                assert np.all(corr<=ind+TOLERANCE) and np.all(corr>=lower-TOLERANCE)
                if values:
                    np.testing.assert_array_equal(ind, values[0][1])
                    assert np.all(corr<=values[-1][0]+TOLERANCE), 'R_corr increased with ell'
                else:
                    np.testing.assert_allclose(corr, ind, rtol=0, atol=TOLERANCE)
                values.append((corr, ind, elapsed))
            by_order[order] = values
            saved[f'{split}_marginal_failure_{order}'] = marginal
            saved[f'{split}_correlated_{order}'] = np.stack([v[0] for v in values])
            saved[f'{split}_independent_{order}'] = values[0][1]
        saved[f'{split}_task_ids'] = tasks.Task_ID.to_numpy()
        for k, ratio in enumerate(LENGTH_RATIOS):
            corr, ind, _ = by_order[16][k]
            ref_corr, ref_ind, _ = by_order[32][k]
            difference = ind-corr
            stats = {f'{name}_{stat}':float(fn(array)) for name,array in (('r_corr',corr),('r_ind',ind))
                     for stat,fn in (('mean',np.mean),('min',np.min),('max',np.max))}
            stats.update(difference_mean=float(difference.mean()), difference_median=float(np.median(difference)),
                         difference_p95=float(np.quantile(difference,.95)), difference_max=float(difference.max()))
            for threshold in LENGTH_THRESHOLDS:
                tags = dict(split=split, ell_ratio=ratio, ell_km=ratio*ell0, threshold=threshold)
                summary, distribution = safe_pair_summary(corr, ind, threshold)
                cm, im = corr>=threshold, ind>=threshold
                false = im&~cm
                assert not np.any(cm&~im)
                if ratio==0:
                    np.testing.assert_array_equal(cm,im)
                if k:
                    assert not np.any(cm & ~(by_order[16][k-1][0]>=threshold))
                summary_rows.append({**tags, 'task_count':200, 'pair_count':28, 'candidate_pair_count':5600,
                    **stats, **summary, 'mask_different_task_fraction':float(np.any(cm!=im,axis=1).mean()),
                    'true_safe_with_extra_unsafe_tasks':int(np.count_nonzero(cm.any(axis=1)&false.any(axis=1))),
                    'true_empty_with_ind_safe_tasks':int(np.count_nonzero(~cm.any(axis=1)&im.any(axis=1))),
                    'rsr_ceiling_corr':float(1-summary['fallback_fraction'])})
                distribution_rows.extend({**tags, **row} for row in distribution)
                for i,j in zip(*np.nonzero(false)):
                    false_rows.append({**tags, 'task_id':int(tasks.Task_ID.iloc[i]), 'pair_index':int(j),
                        'server_m':int(expected_pairs[j,0]), 'server_n':int(expected_pairs[j,1]),
                        'r_corr':float(corr[i,j]), 'r_ind':float(ind[i,j]), 'difference':float(difference[i,j])})
                nearest = np.unravel_index(np.abs(ref_corr-threshold).argmin(), ref_corr.shape)
                for order in LENGTH_ORDERS:
                    current, current_ind, elapsed = by_order[order][k]
                    previous = by_order[order][k-1][0] if k else current
                    validation_rows.append({**tags, 'gh_order':order, 'elapsed_s':elapsed,
                        'max_abs_corr_error_vs_32':float(np.abs(current-ref_corr).max()),
                        'max_abs_ind_error_vs_32':float(np.abs(current_ind-ref_ind).max()),
                        'corr_mask_changes_vs_32':int(np.count_nonzero((current>=threshold)!=(ref_corr>=threshold))),
                        'ind_mask_changes_vs_32':int(np.count_nonzero((current_ind>=threshold)!=(ref_ind>=threshold))),
                        'ell_zero_max_abs_error':float(np.abs(by_order[order][0][0]-by_order[order][0][1]).max()),
                        'ind_change_with_ell':float(np.abs(current_ind-by_order[order][0][1]).max()),
                        'monotonicity_violations':int(np.count_nonzero(current>previous+TOLERANCE)),
                        'max_corr_increase_with_ell':max(0.,float((current-previous).max())),
                        'min_probability':float(min(current.min(),current_ind.min())),
                        'max_probability':float(max(current.max(),current_ind.max())),
                        'nearest_task_id':int(tasks.Task_ID.iloc[nearest[0]]), 'nearest_pair_index':int(nearest[1]),
                        'nearest_corr_value':float(current[nearest]),
                        'min_corr_margin_to_threshold_32':float(np.abs(ref_corr-threshold).min()),
                        'corr_pairs_within_1e_6_of_threshold':int(np.count_nonzero(np.abs(ref_corr-threshold)<=1e-6))})
            print(f'{split}: ell/ell0={ratio:g}, GH16/24/32 checked', flush=True)
    summary = pd.DataFrame(summary_rows)
    distribution = pd.DataFrame(distribution_rows)
    validation = pd.DataFrame(validation_rows)
    false_columns = ['split','ell_ratio','ell_km','threshold','task_id','pair_index','server_m','server_n','r_corr','r_ind','difference']
    false_pairs = pd.DataFrame(false_rows, columns=false_columns)
    for name, table in (('summary',summary),('safe_pair_distribution',distribution),
                        ('false_safe_pairs',false_pairs),('gh_validation',validation)):
        table.to_csv(output/f'{name}.csv', index=False, float_format='%.17g')
    np.savez_compressed(output/'reliability_tables.npz', **saved)
    assert validation.max_abs_corr_error_vs_32.max()<=TOLERANCE
    assert validation.max_abs_ind_error_vs_32.max()<=TOLERANCE
    assert not validation[['corr_mask_changes_vs_32','ind_mask_changes_vs_32','monotonicity_violations']].to_numpy().any()
    assert distribution.groupby(['split','ell_ratio','threshold']).tasks_corr.sum().eq(200).all()
    assert distribution.groupby(['split','ell_ratio','threshold']).tasks_ind.sum().eq(200).all()
    assert all(sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    forbidden = [name for name in ('torch','simpy','agents.ppo_agent','core.main_loop','core.task') if name in sys.modules]
    assert not forbidden, f'Unexpected training/simulator modules: {forbidden}'
    check_edge_cases()
    checks = dict(passed=True, ell_zero_max_abs_error=float(validation.ell_zero_max_abs_error.max()),
        ind_change_with_ell=float(validation.ind_change_with_ell.max()),
        max_corr_error_16_vs_32=float(validation.loc[validation.gh_order==16,'max_abs_corr_error_vs_32'].max()),
        max_corr_error_24_vs_32=float(validation.loc[validation.gh_order==24,'max_abs_corr_error_vs_32'].max()),
        safety_decision_changes=0, monotonicity_violations=0, mask_subset_violations=0,
        invalid_length_rejected=True, probability_bounds_passed=True, pair_mapping_preserved=True,
        forbidden_runtime_modules=forbidden, ppo_training_calls=0, simulation_task_execution_calls=0)
    (output/'validation_summary.json').write_text(json.dumps(checks,indent=2)+'\n')
    plot_length_sweep(output, summary)
    write_length_report(output, summary, validation, checks, metadata)
    print(f'Saved length sweep: {output}', flush=True)


def plot_length_sweep(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plots = (
        ('mean_safe_pairs.png','Mean safe pairs', [('mean_safe_pairs_corr','Correlated'),('mean_safe_pairs_ind','Independent')]),
        ('false_safe_fraction.png','False-safe pairs', [('false_safe_fraction_all_pairs','All candidates'),('false_safe_fraction_ind_safe','Independent-safe candidates')]),
        ('fallback_rsr_ceiling.png','Tasks', [('fallback_fraction','Correlated fallback'),('rsr_ceiling_corr','RSR ceiling')]),
    )
    for filename, ylabel, fields in plots:
        fig, axes = plt.subplots(2,2,figsize=(11,7),sharex=True)
        for row, split in enumerate(('train','evaluation')):
            for col, threshold in enumerate(LENGTH_THRESHOLDS):
                data=summary[(summary.split==split)&(summary.threshold==threshold)].sort_values('ell_ratio')
                ax=axes[row,col]
                for (field,label), style in zip(fields,('-o','--s')):
                    ax.plot(data.ell_ratio,data[field],style,label=label,markersize=4)
                ax.set(title=f'{split}, R_req={threshold}',xlabel='ell / ell0',ylabel=ylabel,xticks=LENGTH_RATIOS)
                ax.set_xticklabels([f'{ratio:g}' for ratio in LENGTH_RATIOS], fontsize=9)
                if filename!='mean_safe_pairs.png':
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(output/filename,dpi=150);plt.close(fig)


def write_length_report(output, summary, validation, checks, metadata):
    lines = ['# 第六阶段6A：空间相关长度敏感性扫描', '',
        f'分支`{metadata["git_branch"]}`，HEAD `{metadata["git_head"]}`。仅离线GH计算；无PPO、无任务仿真、无实际动作。',
        f'8台EUA服务器、28个字典序服务器对；训练集和独立评估集各200任务，分别统计。alpha={LENGTH_SCALE}、beta={BETA}，'
        f'ell0={metadata["ell0_km"]:.12g}km，median_distance={metadata["ell0_km"]*np.log(2):.12g}km。'
        '只考虑CPU执行C/f期间的故障；条件于风险场后节点故障独立。所有缩放和统一阈值只在内存生效。',
        '主结果GH每维16阶，24及32阶复核全部候选对和两个阈值，不只抽查边界。'
        '安全判定严格使用R≥阈值，1e-12容差仅用于数值检验，1e-6窗口只标记边界候选；没有裁剪概率或人为改变Mask。', '',
        '## 六档相关长度与完整安全集合结果', '',
        '每组分母200任务、5600个任务—服务器对。Corr/Ind表示相关/公平独立模型。'
        'Fallback为对应模型的空安全集合比例，表示现有卸载规则下将触发Fallback的任务比例，本阶段未执行调度器。', '',
        '| 数据集 | 阈值 | ell/ell0 | 平均安全对 Corr/Ind | Corr任务0/1/≥2 | Ind任务0/1/≥2 | 假安全对数 | 占全部对 | 占Ind安全对 | Mask不同任务 | 真安全且额外放行任务 | Fallback Corr/Ind | 真实RSR上限 |',
        '|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.split} | {r.threshold:g} | {r.ell_ratio:g} | {r.mean_safe_pairs_corr:.3f}/{r.mean_safe_pairs_ind:.3f} | '
            f'{r.tasks_0_corr}/{r.tasks_1_corr}/{r.tasks_2plus_corr} | {r.tasks_0_ind}/{r.tasks_1_ind}/{r.tasks_2plus_ind} | '
            f'{r.false_safe_pairs} | {r.false_safe_fraction_all_pairs:.3%} | {r.false_safe_fraction_ind_safe:.3%} | '
            f'{r.mask_different_task_fraction:.1%} | {r.true_safe_with_extra_unsafe_tasks} | '
            f'{r.fallback_fraction:.1%}/{r.empty_safe_fraction_ind:.1%} | {r.rsr_ceiling_corr:.1%} |')
    reliability = summary.drop_duplicates(['split','ell_ratio'])
    lines += ['', '## 联合可靠性及差值分布', '',
        '这里统计每组5600个任务—服务器对，不是实际卸载策略的动作分布；差值与阈值无关，因此每个任务集和相关长度只列一次。', '',
        '| 数据集 | ell/ell0 | R_corr均值 | R_corr范围 | R_ind均值 | R_ind范围 | Delta均值 | Delta中位数 | Delta P95 | Delta最大值 |',
        '|---|---:|---:|---|---:|---|---:|---:|---:|---:|']
    for r in reliability.itertuples():
        lines.append(f'| {r.split} | {r.ell_ratio:g} | {r.r_corr_mean:.9f} | [{r.r_corr_min:.9f}, {r.r_corr_max:.9f}] | '
            f'{r.r_ind_mean:.9f} | [{r.r_ind_min:.9f}, {r.r_ind_max:.9f}] | {r.difference_mean:.7g} | '
            f'{r.difference_median:.7g} | {r.difference_p95:.7g} | {r.difference_max:.7g} |')
    lines += ['', '## 数值复核', '',
        '| GH每维阶数 | 对32阶最大R_corr误差 | 最大R_ind误差 | Corr/Ind安全判定变化 | 批量二维GH总耗时/s |',
        '|---:|---:|---:|---:|---:|']
    for order, g in validation.groupby('gh_order'):
        # Each integral is shared by the two thresholds; count runtime once.
        elapsed=g.drop_duplicates(['split','ell_ratio']).elapsed_s.sum()
        lines.append(f'| {order} | {g.max_abs_corr_error_vs_32.max():.3g} | {g.max_abs_ind_error_vs_32.max():.3g} | '
            f'{int(g.corr_mask_changes_vs_32.sum())}/{int(g.ind_mask_changes_vs_32.sum())} | {elapsed:.6f} |')
    lines += ['', f'- ell=0相关/独立最大差：{checks["ell_zero_max_abs_error"]:.3g}；R_ind跨ell变化：{checks["ind_change_with_ell"]:.3g}。',
        '- 概率有限且均在[0,1]，R_corr≤R_ind，边际下界成立，增大ell的单个任务—服务器对可靠性无异常增加；相关Mask为独立Mask子集。',
        '- ell=0的Mask与独立Mask完全一致；负数、NaN及无穷相关长度均按已有实现报错。28动作映射与第五阶段缓存逐项一致。',
        '- gh_validation.csv记录每个任务集/ell/阈值/阶数的误差、概率范围、边界最近任务/动作及可靠性。'
        '16/24/32阶全部候选对安全判定一致，包含所有距离阈值≤1e-6的候选对。32阶只是更高阶数值参考，不称作解析真值。',
        '- CPU故障只使用C/f；上传、CPU排队及任务到达不参与积分。本次没有导入torch、simpy或PPO/仿真执行模块。', '',
        '## 三张分析图', '', '![平均安全服务器对](mean_safe_pairs.png)', '',
        '![候选假安全对比例](false_safe_fraction.png)', '',
        '![Fallback与真实RSR上限](fallback_rsr_ceiling.png)', '',
        '图按训练/评估分行、两个阈值分列；假安全图同时明确全部候选对和独立模型安全对两个分母。', '',
        '## 八项问题的回答', '',
        '1. **可靠性随相关长度的变化：**固定任务和服务器对时，相关联合可靠性随ell增大不增加；'
        '空间风险更相关使双副本同时失败的期望增加，削弱冗余收益。检查对每个候选对执行，而非只比较均值。',
        '2. **两模型差异：**单节点边际不依赖ell，R_ind保持不变；R_ind−R_corr随ell增大总体扩大。'
        'ell=0的机器精度差异属于数值舍入，不应解释为模型差别。',
        '3. **候选假安全对：**独立安全集固定、相关安全集逐档收缩，候选假安全集合随ell扩大；'
        '数量及两种比例均不下降，详见各任务集独立统计表。',
        '4. **安全集合差异最大的位置：**']
    for (split, threshold), g in summary.groupby(['split','threshold']):
        largest=g.loc[g.false_safe_pairs.idxmax()]
        tied=g[g.false_safe_pairs==largest.false_safe_pairs].ell_ratio.tolist()
        lines.append(f'   {split}、{threshold:g}：候选假安全对最多的ell/ell0={tied}，'
            f'{int(largest.false_safe_pairs)}/5600={largest.false_safe_fraction_all_pairs:.3%}，'
            f'Mask不同任务比例{largest.mask_different_task_fraction:.1%}。')
    lines += ['5. **收缩是否过强：**不人为设一个“过度”百分比，也不自动调参；用空集和唯一动作占比衡量可选空间。']
    for (split, threshold), g in summary.groupby(['split','threshold']):
        first=g.iloc[0]; last=g.iloc[-1]
        lines.append(f'   {split}、{threshold:g}：ell从0到4ell0，平均安全对'
            f'{first.mean_safe_pairs_corr:.3f}→{last.mean_safe_pairs_corr:.3f}，'
            f'空集{first.tasks_0_corr}→{last.tasks_0_corr}，唯一动作{first.tasks_1_corr}→{last.tasks_1_corr}，'
            f'≥2动作{first.tasks_2plus_corr}→{last.tasks_2plus_corr}。')
    lines += ['   严格阈值下长相关长度进一步压缩可学习动作空间，需在策略实验中如实保留Fallback；'
        '宽松阈值是否出现空集应以上表为准，不能将安全对数减少直接等同于任务全部不可行。',
        '6. **Fallback与RSR上限：**相关空集比例随ell不减，RSR_ceiling=1−真实空集比例随ell不增；'
        '独立空集比例恒定。各档完整上限列在表中，它是可达理论上限，未推算Independent-Masked PPO的实际RSR。',
        '7. **独立退化：**ell=0使用现有独立风险矩阵，无除零；可靠性和安全Mask均正确退化到公平独立模型。',
        '8. **后续代表性长度：**优先使用ell/ell0=0、0.5、1、4：0为独立退化对照，0.5为短程，1保留第五阶段基准，4为长程端点。'
        '距离中位数处的潜在风险相关系数分别为0、0.25、0.5、约0.841；覆盖不同物理强度和实际安全集合。'
        '0.25与2档保留在完整扫描中，可用于补查曲线过渡；该选择依据零点、基准和相关强度覆盖，不依据尚未运行的PPO时延胜负。',
        '', '## 统计含义与执行范围', '',
        '- 假安全对是离线候选：R_ind≥阈值且R_corr<阈值。不能当作实际选中的假安全动作数、任务实际违约率或独立PPO实际RSR。',
        '- true_safe_with_extra_unsafe_tasks表示真实安全集非空但独立Mask额外放行不安全对的任务数，'
        '只说明存在可避免误选的可能；没有假设策略会选中它。',
        '- 所有数据为两个固定任务集的描述性扫描，不混合统计，不作任务总体或训练种子推断。',
        '- safe_pair_distribution.csv列出0～28的全部任务分布；false_safe_pairs.csv逐项列出候选假安全的Task_ID、动作编号、服务器对和可靠性。',
        '- reliability_tables.npz保留16/24/32阶原始可靠性和各阶边际失败概率；server_pairs.csv保留映射。',
        '- 本轮仅扩展tools/scan_reliability.py，未修改GH数学、PPO、主循环、任务执行、正式配置或原始数据。'
        '历史文件哈希及最终Git状态见source_and_output_audit.json；未commit或push。', '',
        '实际扫描命令：', '', '```text', metadata['command'], '```', '',
        '已有结果目录拒绝覆盖；复现时指定新的--output-dir。本阶段到此停止，不启动PPO或18组正式实验。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def mixed_statistics(corr, ind, tasks, tiers, candidate, split, server_pairs):
    """Apply task-specific thresholds to one shared probability table."""
    requirements = np.array(MIXED_CANDIDATES[candidate])[tiers]
    cm, im = corr >= requirements[:, None], ind >= requirements[:, None]
    assert not np.any(cm & ~im)
    summaries, distributions, false_rows, pair_rows, concentration = [], [], [], [], []
    for tier in range(5):
        keep = np.ones(len(tasks), dtype=bool) if tier == 0 else tiers == tier-1
        n = int(keep.sum())
        tags = dict(split=split, candidate=candidate, tier=tier,
                    r_req='mixed' if tier == 0 else MIXED_CANDIDATES[candidate][tier-1])
        summary, histogram = safe_pair_summary(corr[keep], ind[keep], requirements[keep, None])
        c, i = cm[keep], im[keep]
        false = i & ~c
        false_rows.append({**tags, 'task_count':n, 'candidate_pair_count':n*28,
            **{k:v for k,v in summary.items() if 'false_safe' in k},
            'independent_safe_pairs':int(i.sum()), 'false_unsafe_pairs':summary['false_unsafe_pairs'],
            'mask_different_task_fraction':float(np.any(c != i, axis=1).mean()),
            'true_safe_with_false_tasks':int((c.any(axis=1) & false.any(axis=1)).sum()),
            'true_safe_with_false_task_fraction':float((c.any(axis=1) & false.any(axis=1)).mean()),
            'true_empty_tasks':int((~c.any(axis=1)).sum()),
            'rsr_ceiling_true':float(c.any(axis=1).mean())})
        for model, mask, suffix in (('correlated', c, 'corr'), ('independent', i, 'ind')):
            counts = mask.sum(axis=1)
            summaries.append({**tags, 'model':model, 'task_count':n,
                'mean_safe_pairs':summary[f'mean_safe_pairs_{suffix}'],
                'std_safe_pairs':float(counts.std(ddof=0)),
                'min_safe_pairs':int(counts.min()), 'max_safe_pairs':int(counts.max()),
                'empty_tasks':summary[f'tasks_0_{suffix}'], 'empty_fraction':float((counts == 0).mean()),
                'single_tasks':summary[f'tasks_1_{suffix}'], 'single_fraction':float((counts == 1).mean()),
                'multi_tasks':summary[f'tasks_2plus_{suffix}'], 'multi_fraction':float((counts >= 2).mean()),
                'all_safe_tasks':int((counts == 28).sum()), 'all_safe_fraction':float((counts == 28).mean()),
                'nonempty_fraction':summary[f'feasible_fraction_{suffix}']})
            distributions.extend({**tags, 'model':model, 'safe_pair_count':h['safe_pair_count'],
                'task_count':h[f'tasks_{suffix}'], 'fraction':h[f'fraction_{suffix}']} for h in histogram)
            coverage = mask.sum(axis=0)
            singleton = mask[counts == 1].sum(axis=0)
            membership = int(coverage.sum())
            shares = coverage/membership if membership else np.zeros(28)
            for action, pair in enumerate(server_pairs):
                pair_rows.append({**tags, 'model':model, 'pair_index':action,
                    'server_m':int(pair[0]), 'server_n':int(pair[1]),
                    'safe_task_count':int(coverage[action]), 'coverage_fraction':float(coverage[action]/n),
                    'safe_membership_share':float(shares[action]),
                    'singleton_task_count':int(singleton[action])})
            top = int(coverage.argmax())
            dominant = int(singleton.argmax())
            concentration.append({**tags, 'model':model, 'safe_memberships':membership,
                'distinct_safe_pairs':int((coverage > 0).sum()), 'membership_hhi':float(np.sum(shares**2)),
                'top_pair_index':top, 'top_pair':str(tuple(int(s) for s in server_pairs[top])),
                'top_pair_coverage_fraction':float(coverage[top]/n), 'top_pair_membership_share':float(shares[top]),
                'top3_membership_share':float(np.sort(shares)[-3:].sum()),
                'singleton_tasks':int(singleton.sum()),
                'dominant_singleton_pair':str(tuple(int(s) for s in server_pairs[dominant])) if singleton.any() else 'none',
                'dominant_singleton_tasks':int(singleton[dominant]),
                'dominant_singleton_fraction':float(singleton[dominant]/singleton.sum()) if singleton.any() else 0.})
    details = pd.DataFrame(dict(split=split, candidate=candidate, task_id=tasks.Task_ID,
        tier=tiers+1, r_req=requirements, safe_pairs_corr=cm.sum(axis=1), safe_pairs_ind=im.sum(axis=1),
        false_safe_pairs=(im & ~cm).sum(axis=1),
        mask_corr=[''.join('1' if b else '0' for b in row) for row in cm],
        mask_ind=[''.join('1' if b else '0' for b in row) for row in im]))
    return summaries, distributions, false_rows, pair_rows, concentration, details


def choose_mixed_candidate(training_tiers):
    """Choose from static training feasibility; no evaluation data or PPO scores."""
    assert set(training_tiers.split) == {'train'}
    assessments = []
    for candidate, group in training_tiers[training_tiers.model == 'correlated'].groupby('candidate'):
        full = int((group.all_safe_tasks == 50).sum())
        collapsed = int(((group.empty_tasks+group.single_tasks) > 25).sum())
        multi = int((group.multi_tasks > 25).sum())
        assessments.append(dict(candidate=candidate, fully_unconstrained_tiers=full,
            majority_zero_or_one_tiers=collapsed, majority_multi_tiers=multi,
            eligible=full < 2 and collapsed < 3 and multi > 0))
    eligible = [r for r in assessments if r['eligible']]
    best = min((r['fully_unconstrained_tiers'] for r in eligible), default=None)
    finalists = [r['candidate'] for r in eligible if r['fully_unconstrained_tiers'] == best]
    selected = finalists[0] if len(finalists) == 1 else None
    return dict(selected_candidate=selected, selected_thresholds=MIXED_CANDIDATES.get(selected),
        selection_basis='training_only_static_feasibility', assessments=assessments,
        rule='Exclude >=2 entirely unconstrained tiers or >=3 majority-0/1 tiers; require a majority-multi tier. '
             'Prefer fewer entirely unconstrained tiers; do not break ties using evaluation or RSR.',
        training_tier_statistics=training_tiers[training_tiers.model == 'correlated'].to_dict('records'))


def run_mixed_sweep(servers, train_tasks, pairs, ids, distances, ell0, output):
    """Train-first calibration followed by held-out static verification only."""
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(p) for p in (SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH)]
    sources += [PROJECT_ROOT/p for p in ('tools/scan_reliability.py', 'core/spatial_risk.py',
        'core/env_state.py', 'core/task.py', 'core/main_loop.py', 'agents/ppo_agent.py',
        'config/configuration.py', 'config/params.py', 'tools/generate_server_and_task_parameters.py')]
    hashes = {str(p):sha256(p.read_bytes()).hexdigest() for p in sources}
    history = PROJECT_ROOT/'results/spatial_correlation_length_sweep'
    previous = json.loads((history/'run_metadata.json').read_text())
    for name, digest in previous['input_sha256'].items():
        if Path(name).resolve() != Path(__file__).resolve():
            assert sha256(Path(name).read_bytes()).hexdigest() == digest, f'6A input changed: {name}'
    baseline = np.load(history/'reliability_tables.npz')
    server_pairs = np.array(ids)[pairs]
    np.testing.assert_array_equal(server_pairs, np.array(list(combinations(ids, 2))))
    np.testing.assert_array_equal(server_pairs, baseline['pairs'])
    np.testing.assert_array_equal(distances, baseline['distances_km'])
    frequencies = servers.Processing_Frequency.to_numpy()
    rates = servers.Base_Failure_Rate.to_numpy() * LENGTH_SCALE
    np.testing.assert_array_equal(rates, baseline['scaled_rates'])
    rho = build_spatial_correlation_matrix(distances, ell0)[pairs[:, 0], pairs[:, 1]]
    thresholds = np.unique(np.array(list(MIXED_CANDIDATES.values())).ravel())
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), alpha=LENGTH_SCALE, beta=BETA,
        ell_ratio=1., ell0_km=ell0, gh_order=MIXED_ORDER, gh_check_orders=[16,24],
        candidates=MIXED_CANDIDATES, std_ddof=0, selection_uses_evaluation=False,
        git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT_ROOT,text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip(),
        git_status_before=subprocess.check_output(['git','status','--short'],cwd=PROJECT_ROOT,text=True),
        numpy=np.__version__, pandas=pd.__version__, python=sys.version)
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    hashes[str(history/'reliability_tables.npz')] = sha256((history/'reliability_tables.npz').read_bytes()).hexdigest()
    (output/'input_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    all_summaries, distributions, false_rows, pair_rows, concentration, details, mappings, gh_rows = ([] for _ in range(8))
    saved = dict(server_ids=np.array(ids), pairs=server_pairs, scaled_rates=rates,
                 distances_km=distances, candidate_thresholds=np.array(list(MIXED_CANDIDATES.values())))
    selection_digest = None
    check_edge_cases()
    for split in ('train', 'evaluation'):
        if split == 'evaluation':
            assert selection_digest == sha256((output/'selection.json').read_bytes()).hexdigest()
            tasks = pd.read_excel(EVALUATION_TASK_PATH).sort_values('Task_ID').reset_index(drop=True)
            train_attributes = train_tasks[['Task_Size','Computation_Demand']].to_numpy()
            evaluation_attributes = tasks[['Task_Size','Computation_Demand']].to_numpy()
            assert not np.array_equal(train_attributes, evaluation_attributes)
            assert not set(map(tuple,train_attributes)) & set(map(tuple,evaluation_attributes))
        else:
            tasks = train_tasks.reset_index(drop=True).copy()
        assert len(tasks) == 200 and tasks.Task_ID.is_unique
        levels, counts = np.unique(tasks.Reliability_Requirement, return_counts=True)
        np.testing.assert_array_equal(levels, MIXED_CANDIDATES['A'])
        np.testing.assert_array_equal(counts, [50]*4)
        tiers = np.searchsorted(levels, tasks.Reliability_Requirement)
        np.testing.assert_array_equal(tasks.Task_ID, baseline[f'{split}_task_ids'])
        mapping = tasks.rename(columns={'Reliability_Requirement':'original_r_req'}).assign(split=split,tier=tiers+1)
        for candidate, values in MIXED_CANDIDATES.items():
            mapping[f'r_req_{candidate}'] = np.array(values)[tiers]
        mappings.append(mapping)
        demand = tasks.Computation_Demand.to_numpy()[:,None]
        by_order = {}
        for order in LENGTH_ORDERS:
            start = perf_counter()
            marginal = compute_marginal_failure_probability(demand, frequencies, rates, BETA, quadrature_order=order)
            corr, ind = compute_pair_reliability_gh(demand, frequencies[pairs], rates[pairs], rho, BETA,
                                                   quadrature_order=order)
            elapsed = perf_counter()-start
            for values in (marginal, corr, ind):
                assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
            assert corr.shape == ind.shape == (200,28)
            np.testing.assert_allclose(ind, 1-marginal[:,pairs[:,0]]*marginal[:,pairs[:,1]],rtol=0,atol=TOLERANCE)
            lower = np.maximum(1-marginal[:,pairs[:,0]],1-marginal[:,pairs[:,1]])
            assert np.all(corr >= lower-TOLERANCE) and np.all(corr <= ind+TOLERANCE)
            for array, key in ((corr,f'{split}_correlated_{order}'),(ind,f'{split}_independent_{order}'),
                               (marginal,f'{split}_marginal_failure_{order}')):
                reference = baseline[key][3] if 'correlated' in key else baseline[key]
                np.testing.assert_array_equal(array, reference)
            by_order[order] = corr, ind, marginal, elapsed
        corr, ind, marginal, _ = by_order[MIXED_ORDER]
        for model_values in (corr, ind):
            nested = model_values[...,None] >= thresholds
            assert not np.any(nested[...,1:] & ~nested[...,:-1])
        for order, (c, i, q, elapsed) in by_order.items():
            row = dict(split=split, gh_order=order, elapsed_s=elapsed,
                max_abs_corr_error_vs_32=float(np.abs(c-corr).max()),
                max_abs_ind_error_vs_32=float(np.abs(i-ind).max()),
                max_abs_marginal_error_vs_32=float(np.abs(q-marginal).max()),
                corr_mask_changes=int(np.count_nonzero((c[...,None]>=thresholds)!=(corr[...,None]>=thresholds))),
                ind_mask_changes=int(np.count_nonzero((i[...,None]>=thresholds)!=(ind[...,None]>=thresholds))),
                min_probability=float(min(c.min(),i.min(),q.min())),
                max_probability=float(max(c.max(),i.max(),q.max())), max_abs_error_vs_6a=0.,
                min_corr_margin=float(np.abs(corr[...,None]-thresholds).min()),
                min_ind_margin=float(np.abs(ind[...,None]-thresholds).min()))
            assert max(row['max_abs_corr_error_vs_32'],row['max_abs_ind_error_vs_32']) <= TOLERANCE
            assert row['corr_mask_changes'] == row['ind_mask_changes'] == 0
            gh_rows.append(row)
        previous_masks = None
        for candidate in MIXED_CANDIDATES:
            frames = mixed_statistics(corr, ind, tasks, tiers, candidate, split, server_pairs)
            for target, rows in zip((all_summaries, distributions, false_rows, pair_rows, concentration),frames[:5]):
                target.extend(rows)
            details.append(frames[5])
            requirements = np.array(MIXED_CANDIDATES[candidate])[tiers,None]
            masks = np.stack((corr >= requirements, ind >= requirements))
            if previous_masks is not None:
                assert not np.any(masks & ~previous_masks)
            previous_masks = masks
            saved[f'{split}_{candidate}_requirements'] = requirements[:,0]
            saved[f'{split}_{candidate}_masks'] = masks
        saved.update({f'{split}_task_ids':tasks.Task_ID.to_numpy(), f'{split}_correlated':corr,
                      f'{split}_independent':ind, f'{split}_marginal_failure':marginal})
        if split == 'train':
            train_summary = pd.DataFrame(all_summaries)
            selection = choose_mixed_candidate(train_summary[train_summary.tier > 0])
            selection.update(train_task_sha256=hashes[str(Path(TRAIN_TASK_PATH))],
                train_probability_sha256=sha256(corr.tobytes()).hexdigest(),
                locked_before_evaluation=True)
            (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
            selection_digest = sha256((output/'selection.json').read_bytes()).hexdigest()
            print('Training-only recommendation locked:',selection['selected_candidate'],flush=True)
        print(f'{split}: 200 tasks, 4 x 50 fixed tiers; candidates A/B/C checked',flush=True)
    summary = pd.DataFrame(all_summaries)
    tables = dict(candidate_summary=summary[summary.tier == 0].drop(columns=['tier','r_req']),
        tier_summary=summary[summary.tier > 0], safe_pair_distribution=pd.DataFrame(distributions),
        false_safe_summary=pd.DataFrame(false_rows), safe_pair_membership=pd.DataFrame(pair_rows),
        safe_pair_concentration=pd.DataFrame(concentration), task_tier_mapping=pd.concat(mappings,ignore_index=True),
        task_mask_summary=pd.concat(details,ignore_index=True), gh_validation=pd.DataFrame(gh_rows))
    for name, frame in tables.items():
        assert not frame.isna().any().any(), name
        assert np.isfinite(frame.select_dtypes('number')).all().all(), name
        frame.to_csv(output/f'{name}.csv',index=False,float_format='%.17g')
    np.savez_compressed(output/'reliability_table.npz',**saved)
    group_map = {split:{str(int(tier)):dict(task_ids=g.Task_ID.astype(int).tolist(),
        task_ids_sha256=sha256(g.Task_ID.to_numpy(dtype=np.int64).tobytes()).hexdigest())
        for tier,g in table.groupby('tier')} for split,table in tables['task_tier_mapping'].groupby('split')}
    (output/'task_groups.json').write_text(json.dumps(group_map,indent=2)+'\n')
    simulation_modules = [m for m in ('torch','simpy','core.main_loop','core.task','agents.ppo_agent') if m in sys.modules]
    assert not simulation_modules
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    assert selection_digest == sha256((output/'selection.json').read_bytes()).hexdigest()
    validation = dict(passed=True, task_counts={'train':200,'evaluation':200},tier_counts=[50]*4,
        candidates=list(MIXED_CANDIDATES),pair_count=28,alpha=LENGTH_SCALE,beta=BETA,gh_order=MIXED_ORDER,
        gh_main_pair_integrations=2,gh_check_pair_integrations=4,selection_locked_before_evaluation=True,
        selection_sha256=selection_digest,selected_candidate=selection['selected_candidate'],
        grouping_and_attributes_fixed=True,train_evaluation_no_duplicate_attribute_rows=True,
        threshold_nesting_passed=True,correlated_mask_subset=True,gh_equal_to_6a=True,
        max_abs_corr_error_vs_32=float(tables['gh_validation'].max_abs_corr_error_vs_32.max()),
        corr_mask_changes=0,ind_mask_changes=0,no_probability_clipping=True,
        original_inputs_unchanged=True,simulation_modules_loaded=simulation_modules,
        ppo_training_episodes=0,simulation_episodes=0)
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (output/'commands.txt').write_text(metadata['command']+'\n')
    plot_mixed_sweep(output,tables['tier_summary'])
    write_mixed_report(output,tables,selection,metadata,validation)
    print('Saved mixed-reliability calibration:',output,flush=True)


def plot_mixed_sweep(output, tiers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    for filename in ('mean_safe_pairs','safe_action_counts','empty_rsr_ceiling'):
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
        for row,split in enumerate(('train','evaluation')):
            for col,candidate in enumerate(MIXED_CANDIDATES):
                ax=axes[row,col]
                g=tiers[(tiers.split==split)&(tiers.candidate==candidate)]
                for offset,(model,style) in enumerate((('correlated','-o'),('independent','--s'))):
                    data=g[g.model==model].sort_values('tier')
                    label='Corr' if offset==0 else 'Ind'
                    if filename=='mean_safe_pairs':
                        ax.plot(data.tier,data.mean_safe_pairs,style,label=label)
                    elif filename=='safe_action_counts':
                        x=data.tier.to_numpy()+(-.18 if offset==0 else .18)
                        base=np.zeros(4)
                        for field,color,text in (('empty_fraction','#b44747','0'),('single_fraction','#e0aa39','1'),
                                                 ('multi_fraction','#4d9b78','>=2')):
                            ax.bar(x,data[field],.32,bottom=base,color=color,hatch='//' if offset else None,
                                   label=f'{label}: {text}' if offset==0 else f'{label} (hatched)' if text=='0' else None)
                            base+=data[field].to_numpy()
                    else:
                        ax.plot(data.tier,data.empty_fraction,style,label=f'{label} empty')
                        ax.plot(data.tier,data.nonempty_fraction,style,alpha=.6,label=f'{label} nonempty')
                ax.set(title=f'{split}, candidate {candidate}',xlabel='Reliability requirement',
                       xticks=[1,2,3,4],xticklabels=[f'{v:g}' for v in MIXED_CANDIDATES[candidate]])
                ax.tick_params(axis='x',labelrotation=25)
                if filename=='mean_safe_pairs':ax.set_ylim(0,29)
                else:ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(0,1.05)
                ax.set_ylabel('Mean safe pairs' if filename=='mean_safe_pairs' else 'Task fraction')
                ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(output/f'{filename}.png',dpi=150);plt.close(fig)


def write_mixed_report(output, tables, selection, metadata, validation):
    summary,tiers,false=tables['candidate_summary'],tables['tier_summary'],tables['false_safe_summary']
    selected=selection['selected_candidate']
    lines=['# 第七阶段7A：四档混合可靠性静态校准','',
        f'分支`{metadata["git_branch"]}`，HEAD `{metadata["git_head"]}`。仅离线GH和Mask扫描；没有PPO训练、到达过程、排队、卸载动作或实际Fallback执行。',
        f'固定8台EUA Melbourne CBD服务器、28字典序服务器对；alpha=0.05、beta=0.5、ell=ell0={metadata["ell0_km"]:.12g} km。'
        '沿用第六阶段策略实验GH32，16/24阶复核全部任务—服务器对及七个候选阈值；原始故障率只在内存乘一次0.05。',
        '训练和评估文件原有四档各50个Task_ID直接作为等级1～4，所有候选只替换等级阈值；任务大小、计算需求、服务器属性均不变。'
        '两个文件中的Task_ID各自为1～200，复现键是(split, Task_ID)，不是要求两文件ID互斥。',
        '固定任务集的安全动作数SD使用ddof=0，是200或50个任务的描述性总体SD，不是训练种子不确定性。', '',
        '## 仅训练集选定方案', '',
        '先完成训练集的A/B/C统计，写入并锁定selection.json，之后才读取本轮评估任务表；校验锁定文件哈希不再变化。'
        '未使用第六阶段PPO性能或本轮评估结果选择参数。规则只考察完全无约束档次和可优化动作空间，不按RSR上限打分。',
        '将预设原则具体化为：排除至少两档全部50任务均28对安全，或至少三档中多数任务只有0～1动作的候选；'
        '保留有多数多动作任务的档次，并优先减少完全无约束档次。若仍并列则不以评估结果强行打破。', '',
        '| 方案 | 四档阈值 | 完全无约束档数 | 多数0/1动作档数 | 多数多动作档数 | 选择候选资格 |',
        '|---|---|---:|---:|---:|---|']
    for a in selection['assessments']:
        lines.append(f'| {a["candidate"]} | {list(MIXED_CANDIDATES[a["candidate"]])} | {a["fully_unconstrained_tiers"]} | '
            f'{a["majority_zero_or_one_tiers"]} | {a["majority_multi_tiers"]} | {a["eligible"]} |')
    lines += ['',f'推荐方案 **{selected}：{list(MIXED_CANDIDATES[selected]) if selected else "三组均不直接推荐，需另行校准"}**。',
        'A保留标准跨数量级含义，但前两档完全不受约束；B把第三档变为有作用的中间约束，但第一档仍完全不受约束；'
        'C没有整档全部28动作均安全，前三档仍全部具有多动作，严格档与其他方案完全相同，不会使多数档次坍缩。'
        'C的允许失败预算为0.001、0.0007、0.0004、0.0001，是可解释的逐级收紧；推荐不意味着其PPO时延最好。', '',
        '## 整体安全集合', '',
        '| 集合 | 方案 | 模型 | 平均±SD | 最小/最大 | 空集 | 唯一动作 | 多动作 | 全28对安全 | 自身非空比例 |',
        '|---|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.split} | {r.candidate} | {r.model} | {r.mean_safe_pairs:.3f}±{r.std_safe_pairs:.3f} | '
            f'{r.min_safe_pairs}/{r.max_safe_pairs} | {r.empty_tasks} ({r.empty_fraction:.1%}) | '
            f'{r.single_tasks} ({r.single_fraction:.1%}) | {r.multi_tasks} ({r.multi_fraction:.1%}) | '
            f'{r.all_safe_tasks} ({r.all_safe_fraction:.1%}) | {r.nonempty_fraction:.1%} |')
    lines += ['', '## 四档完整统计', '',
        '| 集合 | 方案 | 等级 | R_req | 模型 | 平均±SD | 最小/最大 | 空集 | 唯一 | ≥2 | 全28安全 |',
        '|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|']
    for r in tiers.itertuples():
        lines.append(f'| {r.split} | {r.candidate} | {r.tier} | {r.r_req:g} | {r.model} | '
            f'{r.mean_safe_pairs:.3f}±{r.std_safe_pairs:.3f} | {r.min_safe_pairs}/{r.max_safe_pairs} | '
            f'{r.empty_tasks} ({r.empty_fraction:.0%}) | {r.single_tasks} ({r.single_fraction:.0%}) | '
            f'{r.multi_tasks} ({r.multi_fraction:.0%}) | {r.all_safe_tasks} ({r.all_safe_fraction:.0%}) |')
    lines += ['', '## 候选假安全与理论可行上限', '',
        '这些是候选集合误判，没有实际策略动作。不能将假安全对数量换算成PPO选择次数、实际可避免违约或Independent PPO的实际RSR。'
        '下表上限始终来自真实相关Mask：整体分母200，分档分母50；独立Mask自身非空比例不是其真实RSR。', '',
        '| 集合 | 方案 | 等级(0=整体) | 假安全对 | 占独立安全对 | 有假安全任务 | 真安全且有假安全任务 | Mask不同任务 | 真实空集 | 真实RSR上限 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in false.itertuples():
        lines.append(f'| {r.split} | {r.candidate} | {r.tier} | {r.false_safe_pairs} | '
            f'{r.false_safe_fraction_ind_safe:.3%} | {r.false_safe_tasks} ({r.false_safe_task_fraction:.1%}) | '
            f'{r.true_safe_with_false_tasks} ({r.true_safe_with_false_task_fraction:.1%}) | '
            f'{r.mask_different_task_fraction:.1%} | {r.true_empty_tasks} | {r.rsr_ceiling_true:.1%} |')
    lines += ['', '## 安全动作集中程度', '',
        '每个安全(任务,动作)只计一次成员资格；HHI和前三对成员占比描述静态Mask集中度，不是策略选择分布、服务器副本分配量、CPU利用率或时延。'
        '完整两模型、各方案各档28对覆盖频率及唯一动作次数见safe_pair_membership.csv；汇总见safe_pair_concentration.csv。', '',
        '| 集合 | 方案 | 等级 | 有安全成员的不同对数 | 前三对成员占比 | HHI | 最常见唯一对 | 其唯一任务数/全部唯一任务 |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for r in tables['safe_pair_concentration'].query('model == "correlated" and tier > 0').itertuples():
        lines.append(f'| {r.split} | {r.candidate} | {r.tier} | {r.distinct_safe_pairs} | '
            f'{r.top3_membership_share:.1%} | {r.membership_hhi:.4f} | {r.dominant_singleton_pair} | '
            f'{r.dominant_singleton_tasks}/{r.singleton_tasks} |')
    lines += ['', '## 数值及一致性检查', '',
        '| 集合 | GH阶数 | Corr误差对32阶 | Ind误差对32阶 | Corr/Ind安全判定变化 | 对6A同阶基准误差 | 耗时/s |',
        '|---|---:|---:|---:|---|---:|---:|']
    for r in tables['gh_validation'].itertuples():
        lines.append(f'| {r.split} | {r.gh_order} | {r.max_abs_corr_error_vs_32:.3g} | '
            f'{r.max_abs_ind_error_vs_32:.3g} | {r.corr_mask_changes}/{r.ind_mask_changes} | '
            f'{r.max_abs_error_vs_6a:.3g} | {r.elapsed_s:.6f} |')
    lines += ['',
        '- 两个任务集各200个、四档各50；任务属性和分组跨候选固定，两个数据集属性行无重复，原生成器使用不同profile_stream=1/2。',
        '- 28动作字典序映射、距离、单次缩放故障率及16/24/32阶联合可靠性均与6A逐项一致。'
        '所有候选共用同一GH表，阈值不参与积分。不同阈值应用到同一任务时Mask嵌套，A→B→C逐项不扩张。',
        '- 概率有限且处于[0,1]；边际下界、公平独立边际及Corr≤Ind通过，相关Mask为独立Mask子集。'
        '未裁剪概率；既有解析极限及非法相关长度检查复用check_edge_cases。',
        '- 未导入torch、simpy、主仿真或PPO模块；PPO训练和动态卸载Episode均为0。',
        '- 原始文件和核心源码哈希未变；最终全工作区保全回查及历史产物检查见source_and_output_audit.json。', '',
        '## 三张静态分析图', '',
        '![四档平均安全动作数](mean_safe_pairs.png)', '',
        '![0/1/≥2动作任务比例](safe_action_counts.png)', '',
        '![空集及非空比例](empty_rsr_ceiling.png)', '',
        '图中Corr/Ind均区分；相关非空比例为真实RSR上限，独立非空比例只是自身模型可行率。', '',
        '## 九项问题与7B接口', '',
        '1. 原始A前两档均完全无约束，因此有两档缺乏约束区分；这来自给定alpha=0.05下的静态概率，不调整alpha来放大差异。',
        '2. 0.999已排除部分动作；0.9993、0.9995、0.9996进一步收紧；0.9、0.99在本场景全部安全。'
        '各档均值不是同一批任务，不能仅用跨档均值证明阈值单调性；代码另对每个相同任务检查全部阈值嵌套。',
        '3. 0.9999最易出现空集或唯一动作；前三档仍保留多动作。完整数量、比例和分布均在上表及CSV。',
        '4. 假安全候选集合和Mask不同任务按整体及每档分别列出；相关安全集合始终为独立子集，没有实际动作误判率的推断。',
        '5. 严格档的唯一动作集中情况见上表，可作为7B负载热点的待验证假设；本阶段不推断CPU利用率或实际时延。',
        '6. 三组方案的整体及分档真实RSR上限见候选假安全表；它们是静态可行上限而非已经实现的策略RSR。',
        f'7. 训练集推荐{selected}，依据为无整档完全无约束、仍有充足多动作空间且保留同一严格档；'
        '未根据评估集或PPO时延改变候选阈值。',
        '8. 推荐方案的独立评估验证只检查是否保留相近的静态可行性特征；无论结果如何都不反向重新选择，具体对照如下。',
        '9. 现有Task读取自身Reliability_Requirement，EnvironmentState按task.reliability_requirement比较缓存的28维可靠性，'
        'MainLoop接收逐任务profile并处理空集Fallback，因此无需修改PPO状态维度、Mask数学或任务生成逻辑。'
        '7B只需在实验编排内存中按已锁定Task_ID映射应用阈值，并按每任务要求预计算Mask。'
        '现有compare_ppo_greedy.py的单阈值预计算/实验入口仍需做必要的混合任务适配，不能继续把全部任务覆写为同一个阈值；'
        '本轮不进行该适配、不启动7B，也不据静态扫描承诺时延优势。', '']
    if selected:
        lines += ['| 集合 | 等级 | 平均安全对 Corr/Ind | Corr空集/唯一/多动作 | Corr全安全 | 真RSR上限 |',
                  '|---|---:|---|---|---:|---:|']
        for split in ('train','evaluation'):
            for tier in range(1,5):
                g=tiers[(tiers.candidate==selected)&(tiers.split==split)&(tiers.tier==tier)].set_index('model')
                c,i=g.loc['correlated'],g.loc['independent']
                lines.append(f'| {split} | {tier} | {c.mean_safe_pairs:.3f}/{i.mean_safe_pairs:.3f} | '
                    f'{int(c.empty_tasks)}/{int(c.single_tasks)}/{int(c.multi_tasks)} | {int(c.all_safe_tasks)} | {c.nonempty_fraction:.1%} |')
    lines += ['', '## 产物与执行', '',
        'candidate_summary.csv为整体，tier_summary.csv为四档（含ddof=0标准差）；safe_pair_distribution.csv保留每组0～28全部分布。'
        'task_tier_mapping.csv含原始属性、原等级及A/B/C阈值；task_groups.json保留每档50个Task_ID列表和摘要。'
        'task_mask_summary.csv及reliability_table.npz保留复现所需的逐任务Mask和共享GH概率。',
        'selection.json记录训练集选择及锁定哈希，validation.json与gh_validation.csv记录检查；源码、输入和输出SHA256一并保留。', '',
        '实际扫描命令：', '', '```text',metadata['command'],'```', '',
        '本轮仅扩展tools/scan_reliability.py；原有未提交修改保留，包括用户已删除的README.md。'
        '未修改正式配置、原始Excel/CSV、PPO或任务执行机制；已有目录拒绝覆盖。未commit、未push，完成后停止。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_deadline_sweep(servers, tasks, output, scales):
    """Training-only zero-queue checks; never select kappa from these bounds."""
    if len(set(scales)) != len(scales):
        raise ValueError('Use distinct deadline scales')
    source = PROJECT_ROOT/'results/mixed_reliability_calibration'
    for name,h in json.loads((source/'output_manifest.json').read_text()).items():
        if sha256((source/name).read_bytes()).hexdigest() != h:
            raise ValueError(f'Frozen 7A artifact changed: {name}')
    old_inputs = json.loads((source/'input_manifest.json').read_text())
    for path in (Path(SERVER_INFO_PATH),Path(TRAIN_TASK_PATH)):
        assert sha256(path.read_bytes()).hexdigest() == old_inputs[str(path)]
    metadata = json.loads((source/'run_metadata.json').read_text())
    assert (metadata['alpha'],metadata['beta'],metadata['ell_ratio'],metadata['gh_order']) == (.05,.5,1.,32)
    mapping = pd.read_csv(source/'task_tier_mapping.csv',float_precision='round_trip').query("split=='train'").reset_index(drop=True)
    np.testing.assert_array_equal(tasks[['Task_ID','Task_Size','Computation_Demand']].to_numpy(),mapping[['Task_ID','Task_Size','Computation_Demand']].to_numpy())
    arrays = np.load(source/'reliability_table.npz')
    pairs = np.array(list(combinations(range(len(servers)),2)))
    np.testing.assert_array_equal(arrays['server_ids'],servers.Server_ID)
    np.testing.assert_array_equal(arrays['pairs'],servers.Server_ID.to_numpy()[pairs])
    np.testing.assert_array_equal(arrays['train_task_ids'],tasks.Task_ID)
    np.testing.assert_array_equal(arrays['scaled_rates'],servers.Base_Failure_Rate.to_numpy()*.05)
    safe = arrays['train_correlated'] >= mapping.r_req_C.to_numpy()[:,None]
    np.testing.assert_array_equal(safe,arrays['train_C_masks'][0])
    sizes,demands = tasks.Task_Size.to_numpy(),tasks.Computation_Demand.to_numpy()
    frequencies,rates = servers.Processing_Frequency.to_numpy(),servers.Transmission_Rate.to_numpy()
    # These are fixed, unloaded service times, not predictions using future queues.
    node_latency = sizes[:,None]/rates+demands[:,None]/frequencies
    pair_latency = node_latency[:,pairs].min(axis=-1)
    best_safe = np.where(safe,pair_latency,np.inf).min(axis=1)
    has_safe = safe.any(axis=1)
    rows=[];summaries=[];reference=None
    for kappa in scales:
        ref,deadline = compute_task_deadlines(sizes,demands,frequencies,rates,kappa=kappa)
        if reference is None:reference=ref
        else:np.testing.assert_array_equal(reference,ref)
        np.testing.assert_allclose(ref,[min(size/rate+demand/frequency for frequency,rate in zip(frequencies,rates)) for size,demand in zip(sizes,demands)],rtol=0,atol=1e-12)
        np.testing.assert_allclose(deadline,kappa*ref,rtol=0,atol=0)
        joint = safe & (pair_latency<=deadline[:,None])
        frame=tasks.copy().assign(tier=mapping.tier.to_numpy(),mixed_reliability_requirement=mapping.r_req_C.to_numpy(),
            kappa=kappa,reference_latency_s=ref,relative_deadline_s=deadline,
            correlated_safe_pair_count=safe.sum(axis=1),joint_feasible_pair_count=joint.sum(axis=1),
            best_safe_unloaded_latency_s=np.where(has_safe,best_safe,np.nan),
            minimum_deadline_scale_for_safe_pair=np.where(has_safe,best_safe/ref,np.nan),
            reliability_empty=~has_safe,reliability_feasible_but_deadline_impossible=has_safe&~joint.any(axis=1))
        rows.append(frame)
        for tier,g in [(0,frame),*list(frame.groupby('tier'))]:
            eligible=~g.reliability_empty
            summaries.append(dict(kappa=kappa,tier=tier,task_count=len(g),
                reference_mean_s=g.reference_latency_s.mean(),deadline_min_s=g.relative_deadline_s.min(),
                deadline_mean_s=g.relative_deadline_s.mean(),deadline_max_s=g.relative_deadline_s.max(),
                reliability_empty_count=int(g.reliability_empty.sum()),
                joint_feasible_task_count=int(g.joint_feasible_pair_count.gt(0).sum()),
                joint_feasible_task_fraction=g.joint_feasible_pair_count.gt(0).mean(),
                deadline_impossible_given_safe_count=int(g.reliability_feasible_but_deadline_impossible.sum()),
                deadline_feasible_given_safe_fraction=g.loc[eligible,'joint_feasible_pair_count'].gt(0).mean(),
                mean_joint_feasible_pair_count=g.joint_feasible_pair_count.mean()))
    table=pd.concat(rows,ignore_index=True);summary=pd.DataFrame(summaries)
    assert np.isfinite(table[['reference_latency_s','relative_deadline_s']]).all().all()
    assert (table.relative_deadline_s>table.reference_latency_s).all()
    ordered=table.pivot(index='Task_ID',columns='kappa',values='joint_feasible_pair_count').sort_index(axis=1)
    assert (np.diff(ordered.to_numpy(),axis=1)>=0).all()
    assert summary[summary.tier==0].reliability_empty_count.eq(12).all()
    inputs=[Path(SERVER_INFO_PATH),Path(TRAIN_TASK_PATH),Path(__file__),PROJECT_ROOT/'core/task.py',
            source/'reliability_table.npz',source/'task_tier_mapping.csv',source/'output_manifest.json']
    hashes={str(path):sha256(path.read_bytes()).hexdigest() for path in inputs}
    output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'deadline_candidates.csv',index=False,float_format='%.17g')
    summary.to_csv(output/'summary.csv',index=False,float_format='%.17g')
    rows[0].drop(columns=['kappa','relative_deadline_s','joint_feasible_pair_count','reliability_feasible_but_deadline_impossible']).to_csv(output/'reference_latencies.csv',index=False,float_format='%.17g')
    meta=dict(command=shlex.join([sys.executable,*sys.argv]),split='train',deadline_scales=list(scales),
        selected_kappa=None,qos_grace_multiplier=None,reference='min_n(Task_Size/Transmission_Rate + Computation_Demand/Processing_Frequency)',
        gh_source='7A frozen GH32; mixed candidate C',alpha=.05,beta=.5,ell_ratio=1.,
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    validation=dict(passed=True,training_tasks=200,server_pairs=28,candidate_rows=len(table),
        evaluation_used=False,gh_new_integrations=0,simulation_episodes=0,training_episodes=0,
        deadlines_finite_positive=True,reference_independent_of_kappa=True,independent_formula_verified=True,
        joint_feasibility_monotone=True,reward_changed=False,formal_kappa_selected=False)
    (output/'input_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (output/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (output/'commands.txt').write_text(meta['command']+'\n')
    lines=['# 任务软期限：训练集静态候选检查','',
        '固定8服务器、200训练任务、28字典序服务器对；仅在内存使用7A方案C的四档可靠性要求及冻结相关GH32数组。',
        '参考时延=min_n(L_i/v_n+C_i/f_n)，相对软期限=kappa*参考时延。仅依赖任务大小、计算量和固定服务器参数，不依赖策略、队列、到达率、可靠性等级或未来完成时间。',
        '任务上传速率单位MB/s、数据MB，CPU频率cycles/s、计算量cycles，所有时延单位秒。每个任务共享同一期限，不能按28个动作各设不同期限。',
        'deadline_candidates.csv为200任务×候选κ；reference_latencies.csv为固定参考值；summary.csv按整体和四档分别统计。',
        '', '|κ|等级(0=整体)|期限最小/均值/最大 s|相关空集|同时满足期限与可靠性的任务数|安全集非空但静态期限不可行|平均联合可行动作数|',
        '|---|---|---|---|---|---|---|']
    for r in summary.itertuples():lines.append(f'|{r.kappa:g}|{r.tier}|{r.deadline_min_s:.6f}/{r.deadline_mean_s:.6f}/{r.deadline_max_s:.6f}|{r.reliability_empty_count}|{r.joint_feasible_task_count}/{r.task_count}|{r.deadline_impossible_given_safe_count}|{r.mean_joint_feasible_pair_count:.3f}|')
    lines+=['','这些都是无排队下界。联合可行只说明至少存在一个服务器对在空队列时满足条件，不代表PPO或SML在实际负载下能按时完成。',
        '相关空集的12个任务仍按既有规则Fallback执行；它们不能同时满足可靠性约束，不能通过放宽期限变成可靠任务，也不是拒绝任务。',
        '即使四个κ都静态可行，也不能据此选择最小值；需要固定κ候选后，仅在训练任务/到达流上观察真实队列下的按时、宽限区间和超时比例。当前尚未选择QoS宽限倍率，不能计算完整TORE式QoS分布。',
        '本轮不选择正式κ，不读取评估集做参数选择，不修改负时延奖励或正式配置，不增加deadline Mask，不调整现有可靠性Mask或Fallback。不修改原始Excel，不训练PPO。',
        'compute_task_deadlines提供显式κ的纯函数接口；后续实验可在运行开始对训练/评估任务各计算一次，并将结果共享给所有策略和到达率。当前不会自动注入历史仿真流程。',
        '', '实际命令：','', '```text',meta['command'],'```','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    assert all(sha256(Path(name).read_bytes()).hexdigest()==h for name,h in hashes.items())
    (output/'output_manifest.json').write_text(json.dumps({p.name:sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},indent=2)+'\n')
    print(summary[summary.tier==0].to_string(index=False));print(f'Saved {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true", help="Run the focused 0.999/0.9999 grid")
    parser.add_argument("--length-sweep", action="store_true", help="Scan six correlation lengths on both fixed task sets")
    parser.add_argument("--mixed-reliability-sweep", action="store_true", help="Calibrate three fixed four-tier mixtures without simulation")
    parser.add_argument("--deadline-sweep", action="store_true", help="Check training-only workload-scaled soft deadline candidates without queues")
    parser.add_argument("--deadline-scales", type=float, nargs="+", default=[1.2,1.5,2.,3.])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if sum((args.length_sweep, args.calibrate, args.mixed_reliability_sweep, args.deadline_sweep)) > 1:
        parser.error("Choose only one scan mode")
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / ("results/spatial_correlation_length_sweep" if args.length_sweep
                                          else "results/spatial_reliability_scan")
        if args.calibrate:
            args.output_dir /= "calibration_0p02_0p12"
        if args.mixed_reliability_sweep:
            args.output_dir = PROJECT_ROOT/'results/mixed_reliability_calibration'
        if args.deadline_sweep:
            args.output_dir = PROJECT_ROOT/'results/task_deadline_calibration'
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {args.output_dir}")
    paths = (Path(SERVER_INFO_PATH), Path(TRAIN_TASK_PATH))
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
    servers = pd.read_excel(paths[0], sheet_name="Servers").sort_values("Server_ID")
    tasks = pd.read_excel(paths[1]).sort_values("Task_ID")
    if len(servers) != 8 or len(tasks) != 200 or (
        not servers.Server_ID.is_unique or not tasks.Task_ID.is_unique
    ):
        raise ValueError("Expected 8 distinct servers and 200 distinct fixed training tasks")
    if args.deadline_sweep:
        run_deadline_sweep(servers,tasks,args.output_dir,args.deadline_scales)
        return
    profiles = [SimpleNamespace(server_id=int(row.Server_ID), latitude=row.Latitude,
                                longitude=row.Longitude) for row in servers.itertuples()]
    ids, distances = build_distance_matrix(profiles)
    pairs = np.array(list(combinations(range(8), 2)))
    ell = compute_baseline_correlation_length_km(distances)
    if args.mixed_reliability_sweep:
        run_mixed_sweep(servers,tasks,pairs,ids,distances,ell,args.output_dir)
        return
    if args.length_sweep:
        evaluation = pd.read_excel(EVALUATION_TASK_PATH).sort_values("Task_ID")
        run_length_sweep(servers, {"train":tasks,"evaluation":evaluation}, pairs, ids, distances, ell, args.output_dir)
        return
    matrix = build_spatial_correlation_matrix(distances, ell)
    check_edge_cases()
    scan, convergence, validation, distribution, differences = run_scan(
        servers, tasks, pairs, matrix[pairs[:, 0], pairs[:, 1]],
        scales=CALIBRATION_SCALES if args.calibrate else SCALES,
        thresholds=CALIBRATION_THRESHOLDS if args.calibrate else THRESHOLDS,
    )
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (("scan", scan), ("gh_convergence", convergence), ("validation", validation),
                        ("safe_pair_distribution", distribution), ("reliability_difference", differences)):
        table.to_csv(args.output_dir / f"{name}.csv", index=False, float_format="%.17g")
    pd.DataFrame({
        "pair_index": np.arange(28), "server_m": np.array(ids)[pairs[:, 0]],
        "server_n": np.array(ids)[pairs[:, 1]], "distance_km": distances[pairs[:, 0], pairs[:, 1]],
        "risk_correlation": matrix[pairs[:, 0], pairs[:, 1]],
    }).to_csv(args.output_dir / "server_pairs.csv", index=False, float_format="%.17g")
    if args.calibrate:
        write_calibration_report(args.output_dir, scan, convergence, validation, differences, ell, hashes)
    else:
        write_report(args.output_dir, scan, convergence, validation, ell, hashes, tasks)
    assert not validation[["nonfinite_values", "out_of_range_values", "lower_bound_violations",
                           "upper_bound_violations"]].to_numpy().any(), "See validation.csv"
    assert validation.ell_zero_max_abs_error.max() <= TOLERANCE
    assert validation.independent_baseline_max_abs_error.max() <= TOLERANCE
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
