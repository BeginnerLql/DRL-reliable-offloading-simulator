"""Validate existing GH reliability and fixed profiles without running the simulator.

Run from any directory with Python, NumPy, pandas, openpyxl and matplotlib:
    python tools/experiment0_reliability_validation.py

An existing nonempty output directory is never overwritten. Use --output-dir
for a separate reproduction. No configuration or training code is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "drl-experiment0-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.configuration import parameters
from config.paths import EVALUATION_TASK_PATH, SERVER_INFO_PATH, TRAIN_TASK_PATH
from core.spatial_risk import (
    GaussHermiteReliability,
    build_distance_matrix,
    build_spatial_correlation_matrix,
    compute_baseline_correlation_length_km,
)

OUTPUT = ROOT / "results" / "experiment0_reliability_validation"
MC_SAMPLES = 200_000
MC_SEED = 20261028
# Verified through the GitHub connector before implementing this experiment.
REVIEWED_MAIN_SHA = "f663ad3612ccb60e6a9a6302434b68f422344ea8"
DATASETS = {"train": Path(TRAIN_TASK_PATH), "evaluation": Path(EVALUATION_TASK_PATH)}
COLORS = {"train": "#2376b7", "evaluation": "#db7626"}
NAMES = {"train": "Training", "evaluation": "Evaluation"}
REQ_EDGES = np.linspace(*parameters.TASK_RELIABILITY_REQUIREMENT_RANGE, 5)
RHO_EDGES = np.linspace(0.0, 1.0, 5)
KEYS = ["dataset", "task_id", "server_a_id", "server_b_id"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def group_labels(edges, digits):
    return [
        f"[{lo:.{digits}f}, {hi:.{digits}f}{']' if i == len(edges) - 2 else ')'}"
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
    ]


def group_index(value, edges):
    if not edges[0] <= value <= edges[-1]:
        raise ValueError(f"Value {value} lies outside grouping range {edges}")
    return int(np.searchsorted(edges[1:-1], value, side="right"))


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator != 0 else np.nan


def load_profiles():
    servers = pd.read_excel(SERVER_INFO_PATH, sheet_name="Servers").sort_values("Server_ID")
    tasks = {name: pd.read_excel(path).sort_values("Task_ID") for name, path in DATASETS.items()}
    if len(servers) != 8 or servers.Server_ID.nunique() != 8:
        raise ValueError("Experiment 0 requires exactly eight distinct servers")
    for name, frame in tasks.items():
        if len(frame) != 200 or frame.Task_ID.nunique() != 200:
            raise ValueError(f"{name} requires exactly 200 distinct tasks")
        values = frame[["Task_Size", "Computation_Demand", "Reliability_Requirement"]].to_numpy()
        if not np.isfinite(values).all() or (values[:, :2] <= 0).any():
            raise ValueError(f"Invalid {name} task parameters")
        for requirement in frame.Reliability_Requirement:
            group_index(requirement, REQ_EDGES)
    numeric = servers[["Processing_Frequency", "Base_Failure_Rate", "Latitude", "Longitude"]].to_numpy()
    if not np.isfinite(numeric).all() or (numeric[:, 0] <= 0).any() or (numeric[:, 1] < 0).any():
        raise ValueError("Invalid server parameters")
    objects = [SimpleNamespace(server_id=int(s.Server_ID), latitude=s.Latitude, longitude=s.Longitude)
               for s in servers.itertuples()]
    server_ids, distances = build_distance_matrix(objects)
    ell = compute_baseline_correlation_length_km(distances)
    correlations = (build_spatial_correlation_matrix(distances, ell)
                    if parameters.SPATIAL_RISK_ENABLED else np.eye(8))
    return servers, tasks, server_ids, distances, correlations, ell


def enumerate_reliability(servers, tasks, distances, correlations):
    gh32 = GaussHermiteReliability(parameters.SPATIAL_RISK_BETA_P, 32)
    gh64 = GaussHermiteReliability(parameters.SPATIAL_RISK_BETA_P, 64)
    profiles = list(servers.itertuples())
    rows, accuracy = [], []
    for dataset, frame in tasks.items():
        for task in frame.itertuples():
            for a, b in combinations(range(8), 2):
                sa, sb = profiles[a], profiles[b]
                exposure_a = sa.Base_Failure_Rate * task.Computation_Demand / sa.Processing_Frequency
                exposure_b = sb.Base_Failure_Rate * task.Computation_Demand / sb.Processing_Frequency
                rho = float(correlations[a, b])
                r32 = gh32.compute_pair(exposure_a, exposure_b, rho)
                r64 = gh64.compute_pair(exposure_a, exposure_b, rho)
                # Preserve the existing public API, including its float64 rounding.
                qc, qi, q64 = 1.0 - r32.correlated, 1.0 - r32.independent, 1.0 - r64.correlated
                if not (0 <= qi <= 1 and 0 <= qc <= 1 and 0 <= q64 <= 1):
                    raise AssertionError("Reliability API returned an invalid probability")
                key = dict(dataset=dataset, task_id=int(task.Task_ID),
                           server_a_id=int(sa.Server_ID), server_b_id=int(sb.Server_ID))
                rows.append({
                    **key, "computation_demand_cycles": task.Computation_Demand,
                    "reliability_requirement": task.Reliability_Requirement,
                    "exposure_a": exposure_a, "exposure_b": exposure_b,
                    "distance_km": float(distances[a, b]), "rho": rho,
                    "q_corr": qc, "q_ind": qi, "r_corr": r32.correlated,
                    "r_ind": r32.independent, "af": divide(qc, qi),
                    "r_single_a": r32.replica_a, "r_single_b": r32.replica_b,
                    "safe_corr": r32.correlated >= task.Reliability_Requirement,
                    "safe_ind": r32.independent >= task.Reliability_Requirement,
                    "requirement_group": group_index(task.Reliability_Requirement, REQ_EDGES),
                    "correlation_group": group_index(rho, RHO_EDGES),
                })
                accuracy.append({**key, "rho": rho, "q_corr_32": qc, "q_corr_64": q64,
                                 "absolute_error": abs(qc - q64)})
        print(f"Enumerated {dataset}: {len(frame)} tasks x 28 pairs", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(accuracy)


def monte_carlo(pairs, accuracy):
    rows = []
    gh64_values = accuracy.set_index(KEYS).q_corr_64
    for dataset in DATASETS:
        ordered = pairs[pairs.dataset == dataset].sort_values(["q_corr", *KEYS[1:]])
        # Fifteen evenly spaced ranks include each dataset's minimum and maximum.
        ranks = np.rint(np.linspace(0, len(ordered) - 1, 15)).astype(int)
        for selection_index, rank in enumerate(ranks):
            case_id = len(rows) + 1
            seed = MC_SEED + case_id - 1
            row = ordered.iloc[rank]
            rng = np.random.default_rng(seed)
            x, y = rng.standard_normal((2, MC_SAMPLES))
            # Independent MC implementation: Cholesky rather than the GH transform.
            za = x
            zb = row.rho * x + np.sqrt(1.0 - row.rho**2) * y
            beta = parameters.SPATIAL_RISK_BETA_P
            qa = -np.expm1(-row.exposure_a * np.exp(beta * za - beta**2 / 2))
            qb = -np.expm1(-row.exposure_b * np.exp(beta * zb - beta**2 / 2))
            samples = qa * qb
            estimate = float(samples.mean())
            sample_std = float(samples.std(ddof=1))
            se = sample_std / np.sqrt(MC_SAMPLES)
            lower, upper = estimate - 1.96 * se, estimate + 1.96 * se
            reference = float(gh64_values.loc[tuple(row[k] for k in KEYS)])
            rows.append({
                "case_id": case_id, **{k: row[k] for k in KEYS},
                "selection_rank_0based": int(rank), "selection_quantile_target": selection_index / 14,
                "rho": row.rho, "exposure_a": row.exposure_a, "exposure_b": row.exposure_b,
                "mc_seed": seed, "mc_samples": MC_SAMPLES, "q_gh64": reference,
                "q_mc": estimate, "sample_std": sample_std, "standard_error": se,
                "ci95_lower": lower, "ci95_upper": upper,
                "absolute_error": abs(estimate - reference),
                "relative_error": divide(abs(estimate - reference), reference),
                "standardized_residual": divide(estimate - reference, se),
                "gh64_in_ci95": lower <= reference <= upper,
            })
    return pd.DataFrame(rows)


def feasibility_table(pairs):
    rows = []
    for (dataset, task_id), group in pairs.groupby(["dataset", "task_id"], sort=False):
        first = group.iloc[0]
        corr, ind = int(group.safe_corr.sum()), int(group.safe_ind.sum())
        rows.append({
            "dataset": dataset, "task_id": task_id,
            "reliability_requirement": first.reliability_requirement,
            "requirement_group": first.requirement_group,
            "safe_count_corr": corr, "safe_count_ind": ind,
            "empty_corr": corr == 0, "empty_ind": ind == 0,
            "false_safe_count": int((group.safe_ind & ~group.safe_corr).sum()),
            "max_r_corr": group.r_corr.max(), "max_r_ind": group.r_ind.max(),
            "single_feasible": bool(((group.r_single_a >= first.reliability_requirement)
                                     | (group.r_single_b >= first.reliability_requirement)).any()),
        })
    return pd.DataFrame(rows)


def summarize(pairs, feasibility, accuracy):
    rows = []
    req_labels, rho_labels = group_labels(REQ_EDGES, 5), group_labels(RHO_EDGES, 2)
    for dataset in DATASETS:
        frame = pairs[pairs.dataset == dataset]
        tasks = feasibility[feasibility.dataset == dataset]
        errors = accuracy[accuracy.dataset == dataset].set_index(KEYS).absolute_error
        groups = [("overall", -1, "all", np.nan, np.nan, frame)]
        groups += [("requirement", i, label, REQ_EDGES[i], REQ_EDGES[i + 1],
                    frame[frame.requirement_group == i]) for i, label in enumerate(req_labels)]
        groups += [("correlation", i, label, RHO_EDGES[i], RHO_EDGES[i + 1],
                    frame[frame.correlation_group == i]) for i, label in enumerate(rho_labels)]
        for kind, index, label, lower, upper, group in groups:
            defined_af = group.af.dropna()
            entry = dict(
                dataset=dataset, group_type=kind, group_index=index, group_label=label,
                group_lower=lower, group_upper=upper, n_tasks=group.task_id.nunique(),
                n_task_pairs=len(group), q_corr_sum=group.q_corr.sum(), q_ind_sum=group.q_ind.sum(),
                af_overall=divide(group.q_corr.sum(), group.q_ind.sum()),
                af_mean=defined_af.mean(), af_min=defined_af.min(), af_max=defined_af.max(),
                af_defined_count=len(defined_af), af_undefined_count=int(group.af.isna().sum()),
                rho_mean=group.rho.mean(),
                gh_error_max=errors.reindex(pd.MultiIndex.from_frame(group[KEYS])).max(),
                gh_error_mean=errors.reindex(pd.MultiIndex.from_frame(group[KEYS])).mean(),
            )
            if kind != "correlation":
                selected = tasks[tasks.task_id.isin(group.task_id)]
                count = len(selected)
                entry.update(
                    safe_actions_corr=int(selected.safe_count_corr.sum()),
                    safe_actions_ind=int(selected.safe_count_ind.sum()),
                    false_safe_actions=int(selected.false_safe_count.sum()),
                    fsr=divide(selected.false_safe_count.sum(), selected.safe_count_ind.sum()),
                    ass_corr=selected.safe_count_corr.mean(), ass_ind=selected.safe_count_ind.mean(),
                    empty_count_corr=int(selected.empty_corr.sum()),
                    empty_count_ind=int(selected.empty_ind.sum()),
                    esr_corr=divide(selected.empty_corr.sum(), count),
                    esr_ind=divide(selected.empty_ind.sum(), count),
                    rsr_max=1.0 - divide(selected.empty_corr.sum(), count),
                    single_feasible_task_ratio=divide(selected.single_feasible.sum(), count),
                )
            rows.append(entry)
    return pd.DataFrame(rows)


def figures(pairs, accuracy, feasibility, summary, mc, folder):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 110,
                         "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(folder / name, dpi=320, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    unit = np.finfo(float).eps
    error_locations = np.unique(accuracy.absolute_error.to_numpy() / unit)
    for ax, dataset in zip(axes, DATASETS):
        values = accuracy[accuracy.dataset == dataset].absolute_error
        locations, counts = np.unique(values.to_numpy() / unit, return_counts=True)
        ax.bar(locations, counts, width=0.10, color=COLORS[dataset], edgecolor="white")
        ax.set_xticks(error_locations)
        ax.set(title=NAMES[dataset], xlabel="Absolute error / float64 epsilon (dimensionless)",
               ylabel="Task-server-pair count")
        ax.text(0.98, 0.96, f"max = {values.max():.3e}\nmean = {values.mean():.3e}\nepsilon = {unit:.3e}",
                ha="right", va="top", transform=ax.transAxes)
    fig.suptitle("GH-32 versus GH-64: joint-failure probability error")
    save(fig, "gh_error.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        frame = pairs[pairs.dataset == dataset]
        groups = summary[(summary.dataset == dataset) & (summary.group_type == "correlation")]
        ax.scatter(frame.rho, frame.af, s=5, alpha=0.10, color=COLORS[dataset],
                   rasterized=True, label="All task-pair combinations")
        valid = groups[groups.n_task_pairs > 0]
        ax.plot(valid.rho_mean, valid.af_mean, "o-", color="#202832", label="Binned arithmetic mean AF")
        ax.plot(valid.rho_mean, valid.af_overall, "s--", color="#873d99", label="Binned ratio-of-sums AF")
        ax.axhline(1, color="gray", linewidth=0.8)
        ax.set(title=NAMES[dataset], xlabel="Latent spatial-risk correlation rho (dimensionless)",
               ylabel="Joint-failure amplification AF (dimensionless)")
        ax.legend(fontsize=8)
    fig.suptitle("Amplification versus latent spatial-risk correlation")
    save(fig, "af_vs_correlation.png")

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = np.arange(4)
    for offset, dataset in zip([-0.19, 0.19], DATASETS):
        frame = summary[(summary.dataset == dataset) & (summary.group_type == "requirement")]
        bars = ax.bar(x + offset, frame.fsr * 100, 0.36, color=COLORS[dataset], label=NAMES[dataset])
        ax.bar_label(bars, fmt="%.2f%%", fontsize=9, padding=3)
    ax.set(xticks=x, xticklabels=group_labels(REQ_EDGES, 5),
           xlabel="Task reliability requirement interval (dimensionless)",
           ylabel="False-safe rate (%)", title="False-safe decisions across four equal-width requirement bins")
    ax.set_ylim(0, max(1, summary.fsr.max() * 100) * 1.25)
    ax.legend()
    save(fig, "fsr_by_requirement.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        frame = feasibility[feasibility.dataset == dataset]
        bins = np.arange(-0.5, 29.5, 1)
        ax.hist([frame.safe_count_corr, frame.safe_count_ind], bins=bins,
                label=["Correlated model", "Independent model"],
                color=[COLORS[dataset], "#a9afb7"], edgecolor="white")
        ax.set(title=NAMES[dataset], xlabel="Safe server-pair count per task (pairs)",
               ylabel="Task count", xticks=np.arange(0, 29, 4))
        ax.legend(fontsize=9)
    fig.suptitle("Safe-set size distributions (28 candidate pairs per task)")
    save(fig, "safe_set_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.axhspan(-1.96, 1.96, color="#e5eee2", label="Pointwise 95% MC interval includes GH-64")
    ax.axhline(0, color="gray", linewidth=0.8)
    for dataset in DATASETS:
        frame = mc[mc.dataset == dataset]
        ax.scatter(frame.case_id, frame.standardized_residual, color=COLORS[dataset], label=NAMES[dataset])
    ax.set(xlabel="Representative combination ID (15 quantile ranks per dataset)",
           ylabel="(Q_MC - Q_GH64) / SE (dimensionless)",
           title="Independent Monte Carlo cross-check: 200,000 samples per combination")
    ax.legend(fontsize=8)
    save(fig, "mc_validation.png")


def markdown_table(frame, columns):
    lines = ["| " + " | ".join(columns.values()) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append("N/A" if pd.isna(value) else f"{value:.8g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(output, pairs, accuracy, feasibility, summary, mc, ell, source_hashes, initial_status):
    overall = summary[summary.group_type == "overall"]
    requirements = summary[summary.group_type == "requirement"]
    correlation = summary[summary.group_type == "correlation"]
    covered = int(mc.gh64_in_ci95.sum())
    coverage_wording = ("未出现超出点态 95% 区间的组合。" if covered == len(mc) else
                        "存在点态 95% 区间未覆盖的组合；30 次检验本身允许少量随机未覆盖，不单独据此判定 GH 有误。")
    missed = mc[~mc.gh64_in_ci95]
    missed_table = ("全部组合均覆盖。" if missed.empty else markdown_table(missed, {
        "case_id": "组合编号", "dataset": "任务集", "task_id": "任务ID",
        "server_a_id": "服务器a", "server_b_id": "服务器b", "q_gh64": "Q_GH64",
        "q_mc": "Q_MC", "standard_error": "SE", "ci95_lower": "CI下界",
        "ci95_upper": "CI上界", "standardized_residual": "标准化偏差"}))
    af_observations, safe_observations, req_observations, rho_observations = [], [], [], []
    for dataset in DATASETS:
        row = overall[overall.dataset == dataset].iloc[0]
        af_observations.append(f"{dataset} 的 AF_overall={row.af_overall:.8f}，对应联合失效风险总量增加 {(row.af_overall-1):.2%}")
        safe_observations.append(
            f"{dataset} 的 FSR={row.fsr:.4%}，独立模型比相关模型多认可 "
            f"{int(row.false_safe_actions)} 个安全动作，即相对相关安全动作数高估 "
            f"{divide(row.safe_actions_ind-row.safe_actions_corr, row.safe_actions_corr):.4%}")
        req = requirements[requirements.dataset == dataset].sort_values("group_index")
        req_observations.append(
            f"{dataset} 最低与最高要求组的 FSR 分别为 {req.iloc[0].fsr:.4%} 和 "
            f"{req.iloc[-1].fsr:.4%}；最高要求组 ESR_corr={req.iloc[-1].esr_corr:.4%}")
        rho = correlation[(correlation.dataset == dataset) & (correlation.n_task_pairs > 0)].sort_values("group_index")
        rho_observations.append(
            f"{dataset} 从最低到最高 rho 分组的 AF 总量比为 "
            + " → ".join(f"{v:.6f}" for v in rho.af_overall))
    columns = {"dataset": "任务集", "af_overall": "AF_overall", "fsr": "FSR", "ass_corr": "ASS_corr",
               "ass_ind": "ASS_ind", "esr_corr": "ESR_corr", "esr_ind": "ESR_ind", "rsr_max": "RSR_max"}
    text = f"""# 实验 0：GH 积分精度验证与可靠性可行性分析

## 1. 实验范围与可复现性

- 生成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}。
- GitHub `main` 审查基准：`{REVIEWED_MAIN_SHA}`（2026-10-28 通过 GitHub 连接器核验）；本次运行 HEAD：`{git('rev-parse', 'HEAD')}`。
- 运行前工作区状态见文末；既有改动未参与状态、策略或执行仿真。实验仅导入配置、路径和 `core/spatial_risk.py`，不导入 PPO、Mask、奖励、任务执行或训练入口。
- 输入：现有 8 台服务器、训练集 200 个任务、评估集 200 个任务；每集 28 个无序服务器对、5600 个组合，共 {len(pairs)} 行。未重新生成数据。
- 风险参数：空间风险开关 `{parameters.SPATIAL_RISK_ENABLED}`，β={parameters.SPATIAL_RISK_BETA_P}；基准相关长度 ℓ=median(d_ab)/ln(2)={ell:.12g} km。
- 潜在风险相关系数范围：[{pairs.rho.min():.9g}, {pairs.rho.max():.9g}]；故障暴露窗口仅为 CPU 执行时间 C/f，单位 s。
- 任务可靠性要求范围：[{REQ_EDGES[0]}, {REQ_EDGES[-1]}]；四个等宽区间宽度为 {(REQ_EDGES[-1]-REQ_EDGES[0])/4:.8g}。左闭右开，最后一组包含右端点。
- 空间相关系数使用四组固定区间：{'; '.join(group_labels(RHO_EDGES, 2))}。AF 算术均值与概率总量比均记录，二者不混用。
- MC 共选取 30 个组合：每个任务集按 GH-32 的 Q_corr、任务 ID、服务器 ID 稳定排序，取 `round(linspace(0, 5599, 15))` 对应的 15 个秩，包含各集最小和最大风险。每组 M={MC_SAMPLES}；基础种子 {MC_SEED}，第 k 组种子为 {MC_SEED}+k−1；使用 NumPy default_rng（PCG64）。完整选择秩与种子保存在 mc_validation.csv。
- 数值类型 float64；Python {platform.python_version()}，NumPy {np.__version__}，pandas {pd.__version__}，matplotlib {matplotlib.__version__}。图像使用真实 CSV 对应数据绘制，PNG 分辨率为 320 dpi；图中使用英文标注避免字体依赖。
- 重现命令：`python tools/experiment0_reliability_validation.py`。本次解释器：`{sys.executable}`。脚本拒绝覆盖非空输出目录；复算可指定 `--output-dir /tmp/experiment0_reproduction`。

## 2. 模型与指标公式

令 a_n=λ₀(n)C/f(n)，q_n(z)=1−exp[−a_n exp(βz−β²/2)]。潜在风险为标准高斯变量，ρ_ab=exp(−d_ab/ℓ)。ρ 是潜在风险场相关系数，不是故障事件 Pearson 相关系数。

- Q_corr=E[q_a(Z_a)q_b(Z_b)]；Q_ind=E[q_a(Z)]E[q_b(Z)]，独立模型保持相同边际分布，只去掉节点相关性。
- R_corr=1−Q_corr，R_ind=1−Q_ind；A_m(i)={{(a,b):R_m(i,a,b)≥R_req(i)}}。
- E_max=max|Q_corr,32−Q_corr,64|，E_mean=Σ|Q_corr,32−Q_corr,64|/(28N)。
- AF(i,a,b)=Q_corr/Q_ind；AF_overall=ΣQ_corr/ΣQ_ind，不能用逐项 AF 的平均值代替。
- ASS_m=Σ|A_m(i)|/N；ESR_m=Σ1[|A_m(i)|=0]/N；RSR_max=1−ESR_corr。
- FSR=Σ|A_ind(i)\\A_corr(i)|/Σ|A_ind(i)|，分母为独立模型认可的安全动作总数。
- MC 使用独立标准正态 X、Y 及 Cholesky 变换 Z_a=X，Z_b=ρX+√(1−ρ²)Y；与 GH 的对称变换实现不同。采样对象为 q_aq_b，不是伯努利故障事件。Q_MC=mean(q_aq_b)，SE=std(q_aq_b,ddof=1)/√M，CI95=[Q_MC−1.96SE,Q_MC+1.96SE]，不裁剪区间。
- 未定义的 AF/FSR 在 CSV 中写为 `N/A`，不置为零，不纳入 AF 的有效值统计。空分组的均值和比例同样为 N/A；rho 分组仅计算 AF 等组合级指标，不将组内候选集混作完整安全集。

## 3. 实验 A：GH 数值收敛与 MC 交叉验证

{markdown_table(overall, {'dataset': '任务集', 'gh_error_max': 'E_max', 'gh_error_mean': 'E_mean'})}

全部组合 E_max={accuracy.absolute_error.max():.12g}，E_mean={accuracy.absolute_error.mean():.12g}。现有公开接口返回 R，本实验保留该实现，以 Q=1−R 还原联合失败概率；因此最后几位存在消减与舍入误差。float64 epsilon 为 {np.finfo(float).eps:.12g}。这些数字衡量 GH-32/64 的实现输出一致性，不能解释为相对于数学真值的严格误差上界。

![GH-32/64 联合失败概率绝对误差分布](figures/gh_error.png)

图中横轴以 float64 epsilon 为单位，零误差也保留。误差位于浮点舍入量级，当前参数下增加求积阶数没有产生有实际意义的概率变化。

MC 检验中，GH-64 落入点态 95% 区间的组合为 **{covered}/{len(mc)}**；最大标准化偏差 |Q_MC−Q_GH64|/SE 为 {mc.standardized_residual.abs().max():.6g}。{coverage_wording} 这些区间是逐组合的点态区间，不是 30 组同时覆盖区间。

{markdown_table(mc.groupby('dataset', sort=False).agg(n_cases=('case_id', 'size'), covered=('gh64_in_ci95', 'sum'), max_absolute_error=('absolute_error', 'max'), mean_absolute_error=('absolute_error', 'mean'), max_standard_error=('standard_error', 'max')).reset_index(), {'dataset': '任务集', 'n_cases': '组合数', 'covered': 'CI覆盖数', 'max_absolute_error': '最大MC绝对偏差', 'mean_absolute_error': '平均MC绝对偏差', 'max_standard_error': '最大SE'})}

未覆盖组合如下；保留原始种子和结果，不更换样本以追求全部覆盖：

{missed_table}

![MC 交叉验证诊断](figures/mc_validation.png)

绿色带表示 |Q_MC−Q_GH64|≤1.96SE。MC 与 GH 的差异必须结合采样标准误判断，不能直接拿 MC 的绝对偏差作为 GH 的积分误差。

**精度判断：当前固定数据、β 和相关长度下，GH-32 与 GH-64 的差异处于浮点舍入量级；结合独立 MC 检验，GH-32 足以用于后续当前参数下的可靠性筛选。该判断不自动外推到更大的 β 或其他参数范围。**

## 4. 实验 B：整体可行性

以下比例为 0–1 的无量纲数；N/A 表示指标未定义。

{markdown_table(overall, columns)}

{markdown_table(overall, {'dataset': '任务集', 'safe_actions_corr': '相关安全动作总数', 'safe_actions_ind': '独立安全动作总数', 'false_safe_actions': '错误安全动作数', 'empty_count_corr': '相关空集任务数', 'empty_count_ind': '独立空集任务数'})}

AF_overall 大于 1，说明在相同边际风险下，当前空间相关性使联合失败风险总量增加。FSR 描述独立模型候选安全动作中的错误判定比例，不是某个调度策略实际选中动作的违约率。

{'；'.join(af_observations)}。{'；'.join(safe_observations)}。注意相对安全动作数高估率与 FSR 的分母不同。

![AF 与空间相关系数](figures/af_vs_correlation.png)

散点包含全部任务—服务器对。实线为分组算术均值，虚线为分组概率总量比；横坐标为各组实际 rho 均值。下面给出分组统计，避免把两种聚合方式混淆。

{markdown_table(correlation, {'dataset': '任务集', 'group_label': 'rho区间', 'n_task_pairs': '组合数', 'af_overall': 'AF总量比', 'af_mean': 'AF算术均值', 'af_min': 'AF最小值', 'af_max': 'AF最大值'})}

{'；'.join(rho_observations)}。当前两集的 AF 分组均值随 rho 区间增大而上升。

对固定任务和服务器边际风险，提高非负潜在相关性会提高联合失败概率。当前跨服务器对的图形还叠加了服务器异质性，分组趋势属于描述性证据，不单独等同于控制边际风险后的因果比较。

![相关与独立模型安全集规模分布](figures/safe_set_comparison.png)

相关模型安全集包含于独立模型安全集；独立模型给出的安全动作数量偏大。零安全动作的任务在固定双副本、必须分配的规则下不可满足阈值，不能把这部分归因于策略学习失败。

## 5. 按可靠性要求分层

{markdown_table(requirements, {'dataset': '任务集', 'group_label': '可靠性要求区间', 'n_tasks': '任务数', 'af_overall': 'AF总量比', 'fsr': 'FSR', 'ass_corr': 'ASS_corr', 'ass_ind': 'ASS_ind', 'esr_corr': 'ESR_corr', 'esr_ind': 'ESR_ind', 'rsr_max': 'RSR_max'})}

![四个等宽可靠性要求区间的 FSR](figures/fsr_by_requirement.png)

图中的 FSR 由各分组安全动作总数计算，不对任务级 FSR 做简单平均。应结合分组任务数、ASS 与 ESR 判断严格阈值下的可行性；分组 AF 的差异也受各组任务计算量分布影响。

{'；'.join(req_observations)}。当前空安全集任务集中在最高可靠性要求区间，后续算法评估应区分这一可行性限制。

## 6. 对后续 PPO 与 TORE-NET 实验的判断

当前整体 AF、非零 FSR、仍然存在的安全动作和有限 ESR，支持开展相关模型与独立模型的可靠性对比；无需先改变现有参数。但实验 0 没有调度、到达或训练，不能据此预言 PPO 优于 TORE-NET。

在相同相关模型、安全集和可靠性最大值回退下，严格使用安全集的算法应达到表中的 RSR_max。相同模型的 Corr-MPPO 与 TORE-2-Corr 应主要比较时延和资源消耗；RSR 上限由这批任务的可行性决定。FSR 非零也不保证任意策略都会实际选择错误安全动作，后续仍需记录策略选择与真实评分。

单副本可行任务比例：{'; '.join(f'{row.dataset}={row.single_feasible_task_ratio:.2%}' for row in overall.itertuples())}。当前最大单副本可靠性为 {max(pairs.r_single_a.max(), pairs.r_single_b.max()):.9g}，最低任务要求为 {pairs.reliability_requirement.min():.9g}。若单副本始终低于任务要求，至多双副本方法的单副本分支无法体现，单副本基线也需明确不可行时的规则。

这些结论仅适用于当前拓扑、任务集和风险模型；RSR 是理论阈值满足率，不是实际故障采样的成功率。MC 检验服务于期望积分校验，并未新增真实任务故障或修改执行机制。

## 7. 数据文件与源文件审计

- `gh_accuracy.csv`：{len(accuracy)} 行，全量 GH-32/64 联合失败概率与绝对误差。
- `mc_validation.csv`：{len(mc)} 行，分位点选择、种子、GH-64、MC、SE、点态 CI 与覆盖标记。
- `pair_reliability.csv`：{len(pairs)} 行，逐组合输入、边际可靠性、相关性、Q/R、AF、安全标记和分组编号。
- `feasibility.csv`：{len(feasibility)} 行，逐任务安全集、空集、错误安全动作和最大可靠性。
- `summary.csv`：{len(summary)} 行，整体、可靠性要求分组、rho 分组；分组编号均从 0 开始。rho 分组的安全集指标为 N/A（不适用）。
- `figures/`：上述五张 PNG，均为 320 dpi。

运行前工作区状态（实验未修改这些既有文件）：

```text
{initial_status or '(clean)'}
```

以下 SHA-256 在计算前取得，并在输出前重新核对一致，用于确认输入和复用模块未被实验修改：

| 文件 | SHA-256 |
| --- | --- |
"""
    text += "\n".join(f"| {path.relative_to(ROOT)} | `{digest}` |" for path, digest in source_hashes.items())
    (output / "report.md").write_text(text + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    output = parser.parse_args().output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty results: {output}")
    sources = [Path(SERVER_INFO_PATH), *DATASETS.values(), ROOT / "core/spatial_risk.py",
               ROOT / "config/configuration.py", ROOT / "config/paths.py", Path(__file__).resolve()]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    initial_status = git("status", "--short", "--untracked-files=no")
    servers, tasks, _, distances, correlations, ell = load_profiles()
    pairs, accuracy = enumerate_reliability(servers, tasks, distances, correlations)
    feasibility = feasibility_table(pairs)
    summary = summarize(pairs, feasibility, accuracy)
    mc = monte_carlo(pairs, accuracy)
    if (pairs.safe_corr & ~pairs.safe_ind).any() or (pairs.q_corr + 1e-14 < pairs.q_ind).any():
        raise AssertionError("Positive-correlation safety ordering is violated")
    if pairs.duplicated(KEYS).any() or len(pairs) != 11_200 or len(mc) != 30:
        raise AssertionError("Unexpected enumeration or MC selection coverage")
    for path, digest in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed while experiment was running: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("gh_accuracy", accuracy), ("mc_validation", mc),
                        ("pair_reliability", pairs), ("feasibility", feasibility), ("summary", summary)]:
        frame.to_csv(output / f"{name}.csv", index=False, na_rep="N/A", float_format="%.17g")
    figure_dir = output / "figures"
    figure_dir.mkdir()
    figures(pairs, accuracy, feasibility, summary, mc, figure_dir)
    report(output, pairs, accuracy, feasibility, summary, mc, ell, hashes, initial_status)
    print(summary[summary.group_type == "overall"][["dataset", "af_overall", "fsr", "ass_corr", "ass_ind", "esr_corr", "esr_ind", "rsr_max"]].to_string(index=False))
    print(f"GH max error: {accuracy.absolute_error.max():.6g}; MC coverage: {mc.gh64_in_ci95.sum()}/30")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
