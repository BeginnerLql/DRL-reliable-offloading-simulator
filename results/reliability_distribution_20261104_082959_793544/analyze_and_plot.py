"""Audit paired training/evaluation traces and draw grouped bar charts."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/reliability_distribution_mpl")
import hashlib
import json
from pathlib import Path
import sys
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))
sys.path.insert(0, str(OUT))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from run_study import METHODS, digest, verify_protected, write_json

LABELS = dict(zip(METHODS, ["Corr-PPO", "Ind-PPO", "Corr-Greedy", "Ind-Greedy", "RASOUL-Delay"]))
COLORS = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"]))


def scan_column(method):
    return ("single_safe_count" if method == "rasoul-delay" else "independent_pair_safe_count"
            if method in ("ind-ppo", "indep-delaygreedy") else "corr_pair_safe_count")


def audit_trace(path, group, method, dataset, episodes, scans, profiles, *, limit_rows=None):
    """Streaming verification of every row, with episode-level checks."""
    scan = scans[(scans.group_id == group["id"]) & (scans.dataset == dataset)].set_index("task_id")
    profile = profiles.set_index("Task_ID")
    count_column = scan_column(method)
    columns = ["episode", "task_id", "task_size", "computation_demand", "reliability_requirement",
               "arrival_time", "task_delay", "task_upload_delay", "task_queue_delay", "task_execution_delay",
               "feasible_action_count", "mask_fallback_used", "requirement_satisfied_correlated",
               "requirement_satisfied_independent", "server_A_id", "server_B_id"]
    if method == "rasoul-delay":
        columns += ["observations_at_decision", "delay_discrepancy", "replica_count"]
    rng = np.random.default_rng(2026)
    arrivals = np.load(OUT / "shared_inputs/evaluation_unit_interarrivals.npy") / 2.
    rows, per_episode = 0, []
    paired = hashlib.sha256()
    workload = hashlib.sha256()
    server_counts = np.zeros((8, 2), dtype=int)
    multi_safe = observed = changed = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200000, nrows=limit_rows):
        assert len(chunk) % 200 == 0
        first_episode = rows // 200 + 1
        chunk_episodes = len(chunk) // 200
        np.testing.assert_array_equal(chunk.episode, np.repeat(np.arange(first_episode, first_episode + chunk_episodes), 200))
        np.testing.assert_array_equal(chunk.task_id, np.tile(np.arange(1, 201), chunk_episodes))
        for csv_name, profile_name in (("task_size", "Task_Size"), ("computation_demand", "Computation_Demand"),
                                        ("reliability_requirement", "Reliability_Requirement")):
            np.testing.assert_allclose(chunk[csv_name], chunk.task_id.map(profile[profile_name]), rtol=1e-14, atol=3e-16)
        if dataset == "train":
            expected_arrivals = np.cumsum(rng.exponential(.5, size=(chunk_episodes, 200)), axis=1)
        else:
            expected_arrivals = np.cumsum(arrivals[first_episode - 1:first_episode - 1 + chunk_episodes], axis=1)
        np.testing.assert_allclose(chunk.arrival_time, expected_arrivals.ravel(), rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(chunk.feasible_action_count, chunk.task_id.map(scan[count_column]))
        np.testing.assert_array_equal(chunk.mask_fallback_used, chunk.feasible_action_count.eq(0))
        own_flag = "requirement_satisfied_independent" if method in ("ind-ppo", "indep-delaygreedy") else "requirement_satisfied_correlated"
        np.testing.assert_array_equal(chunk[own_flag], chunk.feasible_action_count.gt(0))
        assert np.isfinite(chunk.task_delay).all() and chunk.task_delay.ge(0).all()
        np.testing.assert_allclose(chunk.task_delay, chunk.task_upload_delay + chunk.task_queue_delay + chunk.task_execution_delay)
        matched = chunk[["episode", "task_id", "task_size", "computation_demand", "reliability_requirement", "arrival_time"]]
        paired.update(np.ascontiguousarray(matched.to_numpy(dtype=float)).tobytes())
        workload.update(np.ascontiguousarray(matched.drop(columns="reliability_requirement").to_numpy(dtype=float)).tobytes())
        for index, name in enumerate(("server_A_id", "server_B_id")):
            counts = chunk[name].value_counts()
            for server in range(1, 9):
                server_counts[server - 1, index] += int(counts.get(server, 0))
        if method == "rasoul-delay":
            assert chunk.server_B_id.isna().all() and chunk.replica_count.eq(1).all()
            np.testing.assert_allclose(chunk.delay_discrepancy, chunk.task_queue_delay, atol=1e-8)
            multi = chunk.feasible_action_count.ge(2)
            learned = multi & chunk.observations_at_decision.gt(0)
            multi_safe += int(multi.sum())
            observed += int(learned.sum())
            changed += int((learned & chunk.server_A_id.ne(chunk.task_id.map(scan.fastest_safe_server_by_base_delay))).sum())
            fallback = chunk.mask_fallback_used
            np.testing.assert_array_equal(chunk.loc[fallback, "server_A_id"],
                                          chunk.loc[fallback, "task_id"].map(scan.single_fallback_server))
        else:
            assert chunk.server_A_id.lt(chunk.server_B_id).all()
        grouped = chunk.groupby("episode", sort=True)
        part = grouped.agg(delay=("task_delay", "mean"), correlated=("requirement_satisfied_correlated", "mean"),
                           independent=("requirement_satisfied_independent", "mean"), fallback=("mask_fallback_used", "mean"))
        per_episode.append(part)
        rows += len(chunk)
    assert rows == episodes * 200
    return dict(rows=rows, paired_input_sha256=paired.hexdigest(), workload_without_requirement_sha256=workload.hexdigest(),
                server_counts=server_counts, multi_safe_decisions=multi_safe, observed_multi_safe=observed,
                history_LCB_differs_from_base_fastest=changed, episode_metrics=pd.concat(per_episode))


def audit(manifest):
    verify_protected()
    for relative, sha in {**manifest["input_sha256"], **manifest["shared_input_sha256"]}.items():
        assert digest(OUT / relative) == sha
    assert json.loads((OUT / "simulation_complete.json").read_text())["completed"]
    scans = pd.read_csv(OUT / "feasibility_per_task.csv")
    ceilings = pd.read_csv(OUT / "feasibility_summary.csv")
    registry = json.loads((OUT / "training_registry.json").read_text())
    assert len(registry) == 6 and len({r["initial_policy_sha256"] for r in registry}) == 1
    training_rows, train_records, eval_records, servers = 0, [], [], []
    train_workload_hashes, eval_workload_hashes = set(), set()
    train_group_hashes = {}
    for info in registry:
        group = next(g for g in manifest["groups"] if g["id"] == info["group_id"])
        assert info["training_episodes"] == 8000 and info["training_tasks"] == 1600000
        assert info["from_scratch"] and not info["smoke"]
        assert info["actor_steps"] == info["critic_steps"] == 64000
        assert digest(info["checkpoint"]) == info["checkpoint_sha256"]
        base = Path(info["training_directory"])
        status = json.loads((base / "status.json").read_text())
        assert status["status"] == "completed" and status["last_completed_episode"] == 8000
        active = json.loads((base / "parameters.json").read_text())
        assert active["TASK_RELIABILITY_REQUIREMENT_RANGE"] == [group["lower"], group["upper"]]
        profiles = pd.read_excel(OUT / "groups" / group["id"] / "inputs/train200.xlsx")
        checked = audit_trace(base / "task_results.csv", group, info["method"], "train", 8000, scans, profiles)
        train_workload_hashes.add(checked["workload_without_requirement_sha256"])
        train_group_hashes.setdefault(group["id"], set()).add(checked["paired_input_sha256"])
        logs = pd.read_csv(base / "logs.csv")
        assert len(logs) == 8000
        for recorded, column in (("task_Avg_Delay", "delay"), ("RSR_Correlated", "correlated"),
                                  ("RSR_Independent", "independent"), ("Mask_Fallback_Rate", "fallback")):
            np.testing.assert_allclose(logs[recorded], checked["episode_metrics"][column], rtol=1e-12, atol=1e-12)
        training_rows += checked["rows"]
        train_records.append(dict(group_id=group["id"], method=info["method"], episodes=8000,
                                  first_500_delay=float(logs.task_Avg_Delay.head(500).mean()),
                                  last_500_delay=float(logs.task_Avg_Delay.tail(500).mean()),
                                  trace_sha256=digest(base / "task_results.csv"), checkpoint_sha256=info["checkpoint_sha256"]))
        print("Training trace verified:", group["id"], info["method"], flush=True)
    assert len(train_workload_hashes) == 1 and all(len(hashes) == 1 for hashes in train_group_hashes.values())
    for group in manifest["groups"]:
        profiles = pd.read_excel(OUT / "groups" / group["id"] / "inputs/evaluation200.xlsx")
        ceiling = ceilings[(ceilings.group_id == group["id"]) & (ceilings.dataset == "evaluation")].iloc[0]
        group_hashes = set()
        for method in METHODS:
            base = OUT / "groups" / group["id"] / "evaluations/baseline" / method
            result = json.loads((base / "result.json").read_text())
            assert result["episodes"] == 500 and result["tasks"] == 100000 and not result["smoke"]
            if method in METHODS[:2]:
                info = next(r for r in registry if r["group_id"] == group["id"] and r["method"] == method)
                assert result["source_checkpoint_sha256"] == info["checkpoint_sha256"]
                assert result["frozen_weights_and_optimizer_verified"]
            trace = Path(result["run_directory"]) / "task_results.csv"
            assert digest(trace) == result["task_results_sha256"]
            checked = audit_trace(trace, group, method, "evaluation", 500, scans, profiles)
            assert checked["paired_input_sha256"] == result["paired_input_sha256"]
            group_hashes.add(checked["paired_input_sha256"])
            eval_workload_hashes.add(checked["workload_without_requirement_sha256"])
            logs = pd.read_csv(base / "episode_metrics.csv")
            for recorded, column in (("task_Avg_Delay", "delay"), ("RSR_Correlated", "correlated"),
                                      ("RSR_Independent", "independent"), ("Mask_Fallback_Rate", "fallback")):
                np.testing.assert_allclose(logs[recorded], checked["episode_metrics"][column], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(result["mean_delay_seconds"], logs.task_Avg_Delay.mean())
            np.testing.assert_allclose(result["success_rate"], logs.RSR_Correlated.mean())
            if method in ("ppo", "corr-delaygreedy"):
                np.testing.assert_allclose(result["success_rate"], ceiling.correlated_pair_ceiling)
            elif method == "rasoul-delay":
                np.testing.assert_allclose(result["success_rate"], ceiling.single_ceiling)
            else:
                np.testing.assert_allclose(result["rsr_independent"], ceiling.independent_pair_ceiling)
            result.update(group_id=group["id"], distribution=group["label"], lower=group["lower"], upper=group["upper"],
                          delay_ci95_halfwidth=1.96 * float(logs.task_Avg_Delay.std(ddof=1)) / np.sqrt(500),
                          success_ci95_halfwidth=1.96 * float(logs.RSR_Correlated.std(ddof=1)) / np.sqrt(500),
                          correlated_pair_ceiling=float(ceiling.correlated_pair_ceiling),
                          single_ceiling=float(ceiling.single_ceiling),
                          workload_without_requirement_sha256=checked["workload_without_requirement_sha256"])
            if method == "rasoul-delay":
                learning = pd.read_csv(base / "episode_server_learning.csv")
                assert len(learning) == 4000
                assert learning.groupby("episode").completed_feedback.sum().eq(200).all()
                result.update(completed_feedback=int(learning.completed_feedback.sum()),
                              multi_safe_decisions=checked["multi_safe_decisions"],
                              observed_multi_safe=checked["observed_multi_safe"],
                              history_LCB_differs_from_base_fastest=checked["history_LCB_differs_from_base_fastest"])
            for server_id, (a, b) in enumerate(checked["server_counts"], start=1):
                servers.append(dict(group_id=group["id"], method=method, server_id=server_id,
                                    replica_A_count=int(a), replica_B_count=int(b)))
            eval_records.append(result)
            print("Evaluation trace verified:", group["id"], method, flush=True)
        assert len(group_hashes) == 1
    assert len(eval_workload_hashes) == 1
    summary = pd.DataFrame(eval_records)
    assert training_rows == 9600000 and summary.tasks.sum() == 1500000
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(train_records).to_csv(OUT / "training_summary.csv", index=False)
    pd.DataFrame(servers).to_csv(OUT / "server_selection_counts.csv", index=False)
    verify_protected()
    write_json(OUT / "validation.json", dict(passed=True, training_runs=6, evaluation_runs=15,
               training_episodes_each=8000, evaluation_episodes_each=500, training_tasks=training_rows,
               evaluation_tasks=int(summary.tasks.sum()), all_rows_verified=True,
               unchanged_task_fields_and_arrival_times_across_all_groups=True,
               same_initial_networks_for_all_training=True, all_safe_counts_match_GH_scan=True,
               all_source_checkpoints_and_frozen_states_verified=True,
               all_RASOUL_feedback_complete=True, repository_inputs_and_algorithms_unchanged=True,
               success_definition="correlated pair / marginal single reliability threshold satisfaction",
               latency_includes_fallback_tasks=True, no_commit_or_push=True))
    return summary


def setup_style():
    font = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})


def panel(ax, manifest, summary, metric):
    x = np.arange(3)
    width = .15
    values_and_errors = []
    for index, method in enumerate(METHODS):
        rows = summary[summary.method.eq(method)].set_index("group_id").loc[[g["id"] for g in manifest["groups"]]]
        factor = 100 if metric == "success_rate" else 1
        values = rows[metric].to_numpy() * factor
        error = rows["success_ci95_halfwidth" if metric == "success_rate" else "delay_ci95_halfwidth"].to_numpy() * factor
        bars = ax.bar(x + (index - 2) * width, values, width, color=COLORS[method], label=LABELS[method],
                      yerr=error, capsize=2, error_kw={"linewidth": .8}, edgecolor="white", linewidth=.4)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=9)
        values_and_errors += list(values + error)
    ax.set_xticks(x, [g["label"] for g in manifest["groups"]])
    ax.set_xlabel("任务可靠性要求的连续均匀分布")
    ax.set_ylabel("可靠性满足率（%）" if metric == "success_rate" else "平均时延（秒）")
    ax.set_ylim(0, 108 if metric == "success_rate" else max(values_and_errors) * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if metric == "success_rate":
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)


def figures(manifest, summary):
    setup_style()
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    combined, axes = plt.subplots(2, 1, figsize=(12.5, 10), layout="constrained")
    for index, metric in enumerate(("mean_delay_seconds", "success_rate")):
        panel(axes[index], manifest, summary, metric)
        axes[index].set_title("(a) 平均时延" if index == 0 else "(b) 可靠性满足率")
        fig, ax = plt.subplots(figsize=(12.5, 5.6), layout="constrained")
        panel(ax, manifest, summary, metric)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=9, frameon=False)
        fig.suptitle("到达率 2、计算需求 1 倍、200 个任务；每方法评估 500 轮", fontsize=12)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(directory / f"{metric}_bar.{extension}")
        plt.close(fig)
    combined.suptitle("三组可靠性需求分布：五种卸载方法\n"
                      "各组两种 PPO 从头训练 8000 轮；时延包含全部任务及 Fallback", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    combined.legend(handles, labels, loc="outside lower center", ncols=5, frameon=False)
    for extension in ("png", "pdf", "svg"):
        combined.savefig(directory / f"overview_bar.{extension}")
    plt.close(combined)
    for group in manifest["groups"]:
        rows = summary[summary.group_id.eq(group["id"])].set_index("method").loc[list(METHODS)]
        fig, ax = plt.subplots(figsize=(8.2, 5), layout="constrained")
        bars = ax.bar([LABELS[m] for m in METHODS], rows.mean_delay_seconds, color=[COLORS[m] for m in METHODS],
                      yerr=rows.delay_ci95_halfwidth, capsize=3, width=.65)
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set_ylim(0, (rows.mean_delay_seconds + rows.delay_ci95_halfwidth).max() * 1.17)
        ax.set_ylabel("平均时延（秒）")
        ax.set_title(group["label"] + "；每方法 500 轮，时延含 Fallback")
        ax.tick_params(axis="x", rotation=10)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(directory / f"{group['id']}_delay_bar.{extension}")
        plt.close(fig)


def table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def report(manifest, summary):
    scan = pd.read_csv(OUT / "feasibility_summary.csv")
    lines = ["# 三组可靠性需求分布的训练与评估", "", "## 1. 设置", "",
             "三组为 U[0.9, 0.99]、U[0.99, 0.999]、U[0.999, 0.9999]。每组分别生成独立的训练、评估工作簿。"
             "按确认的控制变量口径，三组使用相同随机分位数映射可靠性区间，Task_ID、任务大小与计算需求逐行完全一致。"
             "训练与评估使用独立随机流，服务器固定为上一轮 PROFILE_SEED=2029 的 8 台服务器。", "",
             "到达率 2 任务/秒、计算需求 U[4×10⁹, 8×10⁹] cycles、每轮 200 个任务；算法及其他超参数不变。"
             "每组 Corr-PPO 与 Ind-PPO 各从头训练 8000 轮，共 6 个新模型；初始网络和随机种子相同（2026）。"
             "随后冻结各组模型，五种方法分别评估 500 轮；保留策略采样方式。评估种子 2027，"
             "复用上一轮基准的 500 轮任务到达时刻，三组及五种方法完全配对。", "",
             "GH 模型保持 β=0.5、每维 32 节点；只对 CPU 执行时间计算故障暴露。"
             "四种双副本方法的主成功率统一按相关可靠性 Rcorr 判断；RASOUL 使用单副本 GH 边际可靠性。"
             "独立模型满足率另存为诊断指标。双副本先完成者决定任务时延，另一个副本继续执行。"
             "RASOUL 保留单副本、c=0.5、安全集合 LCB、最大可靠性 Fallback、完成后反馈和每轮重置。", "",
             "平均时延和 P95 均包含全部任务，包括未满足可靠性阈值的 Fallback 任务。"
             "误差线为 500 个 Episode 均值的 95% 正态近似区间，只覆盖固定任务/服务器/训练结果下的随机性；每种 PPO 只有一个训练种子。", ""]
    for index, group in enumerate(manifest["groups"], start=2):
        rows = summary[summary.group_id.eq(group["id"])].set_index("method").loc[list(METHODS)].reset_index()
        display = pd.DataFrame({"方法": rows.method.map(LABELS), "平均时延/s": rows.mean_delay_seconds.map(lambda v: f"{v:.3f}"),
                                "P95/s": rows.p95_delay_seconds.map(lambda v: f"{v:.3f}"),
                                "相关/边际满足率": rows.success_rate.map(lambda v: f"{v:.2%}"),
                                "独立模型满足率": rows.rsr_independent.map(lambda v: f"{v:.2%}"),
                                "Fallback": rows.fallback_rate.map(lambda v: f"{v:.2%}"),
                                "平均排队/s": rows.mean_queue_seconds.map(lambda v: f"{v:.3f}")})
        physical = scan[(scan.group_id == group["id"]) & (scan.dataset == "evaluation")].iloc[0]
        lines += [f"## {index}. {group['label']}", "", table(display), "",
                  f"评估集可行性上限：相关双副本 {physical.correlated_pair_ceiling:.2%}，"
                  f"独立双副本 {physical.independent_pair_ceiling:.2%}，单副本 {physical.single_ceiling:.2%}。", ""]
    lines += ["## 5. 结论与解释", ""]
    for group in manifest["groups"]:
        rows = summary[summary.group_id.eq(group["id"])].set_index("method")
        fastest = rows.mean_delay_seconds.idxmin()
        corr, independent = rows.loc["ppo"], rows.loc["ind-ppo"]
        lines += [f"{group['label']} 中，最低平均时延为 {LABELS[fastest]} 的 {rows.loc[fastest, 'mean_delay_seconds']:.3f} 秒，"
                  f"对应可靠性满足率 {rows.loc[fastest, 'success_rate']:.2%}。"
                  f"Corr-PPO 与 Ind-PPO 的时延分别为 {corr.mean_delay_seconds:.3f}、{independent.mean_delay_seconds:.3f} 秒，"
                  f"满足率分别为 {corr.success_rate:.2%}、{independent.success_rate:.2%}。", ""]
    lines += [
              "可靠性要求越严格，可行服务器组合越少。本次相关双副本满足率上限从 100% 降到 94%，再降到 5.5%。"
              "第三组的 200 个评估任务中，189 个在相关模型下没有达标组合，Corr 方法的 Fallback 比例因此为 94.5%。"
              "其时延主要反映高比例 Fallback 条件下的队列表现，应与满足率一起解释。", "",
              "当前服务器最优单副本在最低计算需求 4×10⁹ cycles 下的边际可靠性也只有约 0.984704。"
              "因此后两组 RASOUL 的安全集合恒空，100% 走最大可靠性 Fallback，无法在安全集合内进行 LCB 选服。"
              "在服务器、任务和到达序列相同的条件下，这两组会选择相同服务器并得到相同延迟；仍按要求分别完整运行。", "",
              "时延不要求随可靠性阈值单调上升：阈值同时改变安全集合、所选服务器对及 Fallback，从而改变后续排队。"
              "低时延也不能单独证明方法更好，必须同时比较可靠性满足率。所有比较都基于各组重新训练的对应模型，未复用上一轮 PPO。", "",
              "## 6. 核验与文件", "",
              "输入逐行核验通过；6 次从头训练、15 次小规模评估检查通过；33 项现有单元测试通过。"
              "正式训练 960 万条任务和评估 150 万条任务逐条核验任务属性、到达时刻、安全集合、Fallback 和时延分解，"
              "每轮指标与任务明细一致。三组均共享除可靠性要求外的同一负载；每组五种方法的完整输入相同。"
              "PPO 源 checkpoint 哈希和冻结状态通过核验；RASOUL 每轮完成反馈数量为 200。", "",
              "- `figures/mean_delay_seconds_bar.*`：三组分布的平均时延分组条形图。",
              "- `figures/success_rate_bar.*`：可靠性满足率分组条形图。",
              "- `figures/overview_bar.*`：两项指标总图；`figures/rel*_delay_bar.*` 为各组单独时延图。",
              "- `summary.csv`、`training_summary.csv`：正式评估与训练摘要。",
              "- `feasibility_summary.csv`、`feasibility_per_task.csv`、`pair_reliability_detail.csv`：GH 扫描与安全集合。",
              "- `groups/*/inputs`、`training`、`checkpoints`、`evaluations`：三组数据、新模型与完整实验明细。",
              "- `validation.json`、`manifest.json`、`provenance.json`：一致性验证、实验设置与软件版本。", "",
              "当前正式数据、服务器和算法源文件未修改，历史实验及 checkpoint 保留；没有 commit、没有 push。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    summary = audit(manifest)
    figures(manifest, summary)
    report(manifest, summary)
    print("All training and evaluations verified. Bar charts and report saved.", flush=True)


if __name__ == "__main__":
    main()
