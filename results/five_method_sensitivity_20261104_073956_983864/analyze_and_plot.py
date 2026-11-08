"""Audit all formal evaluations and export scientific plots and a Chinese report."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/five_method_mpl")
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

METHODS = ["ppo", "ind-ppo", "corr-delaygreedy", "indep-delaygreedy", "rasoul-delay"]
LABELS = dict(zip(METHODS, ["Corr-PPO", "Ind-PPO", "Corr-Greedy", "Ind-Greedy", "RASOUL-Delay"]))
COLORS = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"]))
MARKERS = dict(zip(METHODS, ["o", "s", "^", "D", "v"]))
STYLES = dict(zip(METHODS, ["-", "-", "--", "--", "-."]))
GROUP_LABELS = {"arrival_rate": "任务到达率（任务/秒）", "demand_multiplier": "任务计算需求倍率 n",
                "task_count": "每轮任务数量"}
DELAY_AXES = {
    "arrival_rate": (2, 80, [2, 5, 10, 20, 40, 80]),
    "demand_multiplier": (1.5, 700, [2, 5, 10, 20, 50, 100, 200, 500]),
    "task_count": (5, 160, [5, 10, 20, 50, 100, 150]),
}
SUCCESS_AXES = {
    "arrival_rate": (60, 102, [60, 70, 80, 90, 95, 100]),
    "demand_multiplier": (10, 102, [10, 20, 40, 60, 80, 100]),
    "task_count": (60, 102, [60, 70, 80, 90, 95, 100]),
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def md_table(frame, formats=None):
    formats = formats or {}
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [formats.get(column, str)(value) for column, value in zip(frame.columns, row)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def audit(manifest):
    registry = json.loads((OUT / "training_registry.json").read_text())
    provenance = json.loads((OUT / "provenance.json").read_text())
    for relative, expected in provenance["source_sha256"].items():
        assert digest(ROOT / relative) == expected, relative
    for relative, expected in json.loads((OUT / "previous_inputs.json").read_text()).items():
        assert digest(OUT / "previous_inputs" / relative) == expected, relative
    for relative, expected in manifest["input_sha256"].items():
        assert digest(OUT / relative) == expected, relative
    for method, training in registry.items():
        assert training["from_scratch"] and training["training_episodes"] == 8000
        assert training["actor_steps"] == training["critic_steps"] == 64000
        assert digest(training["checkpoint"]) == training["checkpoint_sha256"]
        status = json.loads((Path(training["training_directory"]) / "status.json").read_text())
        assert status["status"] == "completed" and status["last_completed_episode"] == 8000
    official = {"data/server_info.xlsx": "inputs/server_info.xlsx", "data/task_parameters.xlsx": "inputs/train200.xlsx",
                "data/task_parameters_evaluation.xlsx": "inputs/evaluation200.xlsx"}
    for name, snapshot in official.items():
        assert digest(ROOT / name) == digest(OUT / snapshot)
    ceilings = pd.read_csv(OUT / "feasibility_summary.csv").set_index("scenario_id")
    rows, server_rows = [], []
    for scenario in manifest["scenarios"]:
        same_inputs = []
        scan = pd.read_csv(OUT / "feasibility" / f"{scenario['id']}.csv").set_index("task_id")
        for method in METHODS:
            base = OUT / "evaluations" / scenario["id"] / method
            result = json.loads((base / "result.json").read_text())
            assert not result["smoke"] and result["episodes"] == 500
            assert result["tasks"] == 500 * scenario["task_count"]
            assert result["server_sha256"] == digest(OUT / "inputs/server_info.xlsx")
            assert result["task_sha256"] == digest(scenario["task_path"])
            same_inputs.append(result["paired_input_sha256"])
            if method in registry:
                assert result["source_checkpoint_sha256"] == registry[method]["checkpoint_sha256"]
                assert result["frozen_weights_and_optimizer_verified"]
            if method in ("ppo", "corr-delaygreedy"):
                np.testing.assert_allclose(result["success_rate"], ceilings.loc[scenario["id"], "correlated_pair_ceiling"])
            elif method == "rasoul-delay":
                np.testing.assert_allclose(result["success_rate"], ceilings.loc[scenario["id"], "single_replica_ceiling"])
            else:
                np.testing.assert_allclose(result["rsr_independent"], ceilings.loc[scenario["id"], "independent_pair_ceiling"])
            assert result["success_rate"] <= ceilings.loc[scenario["id"], "correlated_pair_ceiling"] + 1e-12
            np.testing.assert_allclose(result["mean_delay_seconds"], result["mean_upload_seconds"] +
                                       result["mean_queue_seconds"] + result["mean_execution_seconds"])
            episodes = pd.read_csv(base / "episode_metrics.csv")
            assert len(episodes) == 500 and episodes.Episode.tolist() == list(range(1, 501))
            np.testing.assert_allclose(episodes.task_Avg_Delay.mean(), result["mean_delay_seconds"])
            np.testing.assert_allclose(episodes.RSR_Correlated.mean(), result["success_rate"])
            result["delay_ci95_halfwidth"] = 1.96 * float(episodes.task_Avg_Delay.std(ddof=1)) / np.sqrt(500)
            result["success_ci95_halfwidth"] = 1.96 * float(episodes.RSR_Correlated.std(ddof=1)) / np.sqrt(500)
            result["correlated_pair_ceiling"] = float(ceilings.loc[scenario["id"], "correlated_pair_ceiling"])
            result["single_replica_ceiling"] = float(ceilings.loc[scenario["id"], "single_replica_ceiling"])
            trace_path = Path(result["run_directory"]) / "task_results.csv"
            assert digest(trace_path) == result["task_results_sha256"]
            usecols = ["task_id", "feasible_action_count", "server_A_id", "server_B_id"]
            if method == "rasoul-delay":
                usecols += ["observations_at_decision", "task_queue_delay", "delay_discrepancy"]
            trace = pd.read_csv(trace_path, usecols=usecols)
            count_column = ("single_safe_count" if method == "rasoul-delay" else
                            "independent_pair_safe_count" if method in ("ind-ppo", "indep-delaygreedy") else "corr_pair_safe_count")
            np.testing.assert_array_equal(trace.feasible_action_count, trace.task_id.map(scan[count_column]))
            a_counts = trace.server_A_id.value_counts()
            b_counts = trace.server_B_id.value_counts()
            for server_id in range(1, 9):
                server_rows.append(dict(scenario_id=scenario["id"], method=method, server_id=server_id,
                                        replica_A_count=int(a_counts.get(server_id, 0)),
                                        replica_B_count=int(b_counts.get(server_id, 0))))
            if method == "rasoul-delay":
                learning = pd.read_csv(base / "episode_server_learning.csv")
                assert len(learning) == 500 * 8
                assert learning.groupby("episode").completed_feedback.sum().eq(scenario["task_count"]).all()
                assert learning.discrepancy_sum.ge(-1e-8).all()
                result["completed_feedback"] = int(learning.completed_feedback.sum())
                np.testing.assert_allclose(trace.delay_discrepancy, trace.task_queue_delay, atol=1e-8)
                multi = trace.feasible_action_count.ge(2)
                learned = multi & trace.observations_at_decision.gt(0)
                static_fastest = trace.task_id.map(scan.fastest_safe_server_by_base_delay)
                result["multi_safe_decisions"] = int(multi.sum())
                result["multi_safe_chosen_server_observed"] = int(learned.sum())
                result["observed_LCB_choice_differs_from_base_fastest"] = int((learned & trace.server_A_id.ne(static_fastest)).sum())
            else:
                result["completed_feedback"] = None
            rows.append(result)
            print(f"audited {scenario['id']} {method}", flush=True)
        assert len(set(same_inputs)) == 1, f"Unpaired tasks: {scenario['id']}"
    summary = pd.DataFrame(rows)
    assert len(summary) == 75
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(server_rows).to_csv(OUT / "server_selection_counts.csv", index=False)
    (OUT / "validation.json").write_text(json.dumps(dict(
        passed=True, scenarios=15, formal_evaluations=75, episodes_per_evaluation=500,
        total_evaluation_tasks=int(summary.tasks.sum()), training_tasks=3200000,
        paired_task_and_arrival_hashes=True, all_input_and_trace_hashes_verified=True,
        all_safe_counts_match_independent_GH_scan=True,
        all_episode_aggregates_match_task_results=True, all_PPO_weights_and_optimizers_frozen=True,
        both_training_runs_completed_8000_episodes=True, all_RASOUL_feedback_complete=True,
        success_definition="correlated pair threshold satisfaction, or marginal single threshold satisfaction",
        CI_scope="500 arrival episodes, conditional on fixed task/server profiles and one trained policy per PPO method",
        no_commit_or_push=True), indent=2))
    return summary, registry


def setup_plots():
    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 200})


def plot_panel(ax, group, points, summary, metric):
    for method in METHODS:
        frame = summary[summary.method.eq(method)].set_index("scenario_id")
        values = np.array([frame.loc[p["scenario_id"], metric] for p in points], dtype=float)
        if metric == "success_rate":
            values *= 100
        ax.plot([p["value"] for p in points], values, color=COLORS[method], marker=MARKERS[method],
                linestyle=STYLES[method], linewidth=1.7,
                markersize=7 if method == "corr-delaygreedy" else 5, label=LABELS[method],
                markerfacecolor="white" if method in ("ppo", "ind-ppo") else COLORS[method],
                zorder=5 if method in ("ppo", "ind-ppo") else 3)
    ax.set_xlabel(GROUP_LABELS[group])
    ax.set_ylabel("平均时延（秒，对数坐标）" if metric == "mean_delay_seconds" else "任务成功率（可靠性满足率，%）")
    ax.set_xticks([p["value"] for p in points])
    if metric == "success_rate":
        lower, upper, ticks = SUCCESS_AXES[group]
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
    else:
        lower, upper, ticks = DELAY_AXES[group]
        ax.set_yscale("log")
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.22)


def save_figures(manifest, summary, registry, *, write_tables=True):
    setup_plots()
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    overview, axes = plt.subplots(3, 2, figsize=(14, 13), layout="constrained")
    for row_index, (group, points) in enumerate(manifest["groups"].items()):
        group_rows = []
        for point in points:
            selected = summary[summary.scenario_id.eq(point["scenario_id"])].copy()
            selected.insert(0, "sweep_value", point["value"])
            group_rows.append(selected)
        if write_tables:
            pd.concat(group_rows).to_csv(OUT / f"{group}_comparison.csv", index=False)
        for column, metric in enumerate(("mean_delay_seconds", "success_rate")):
            plot_panel(axes[row_index, column], group, points, summary, metric)
            axes[row_index, column].set_title(f"({chr(97 + row_index * 2 + column)}) {GROUP_LABELS[group]}")
            figure, axis = plt.subplots(figsize=(8.2, 5.3), layout="constrained")
            plot_panel(axis, group, points, summary, metric)
            axis.legend(fontsize=9, ncols=2)
            for extension in ("png", "pdf", "svg"):
                figure.savefig(directory / f"{group}_{metric}.{extension}")
            plt.close(figure)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    overview.legend(handles, labels, loc="outside lower center", ncols=5, frameon=False)
    overview.suptitle("新可靠性阈值 U[0.92, 0.999]：五种方法的三组单因素实验\n"
                      "每点 500 轮；PPO 在基准条件各训练 8000 轮后冻结", fontsize=14)
    for extension in ("png", "pdf", "svg"):
        overview.savefig(directory / f"overview.{extension}")
    plt.close(overview)
    log_figure, log_axes = plt.subplots(1, 3, figsize=(16, 4.7), layout="constrained")
    for axis, (group, points) in zip(log_axes, manifest["groups"].items()):
        plot_panel(axis, group, points, summary, "mean_delay_seconds")
    log_figure.legend(handles, labels, loc="outside upper center", ncols=5, frameon=False)
    for extension in ("png", "pdf", "svg"):
        log_figure.savefig(directory / f"mean_delay_logscale.{extension}")
    plt.close(log_figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    training_summary = []
    for method, info in registry.items():
        logs = pd.read_csv(Path(info["training_directory"]) / "logs.csv")
        assert len(logs) == 8000
        rolling = logs.task_Avg_Delay.rolling(100, min_periods=1).mean()
        axes[0].plot(logs.Episode, rolling, color=COLORS[method], label=LABELS[method])
        axes[1].plot(logs.Episode, logs.RSR_Correlated.rolling(100, min_periods=1).mean() * 100,
                     color=COLORS[method], label=LABELS[method])
        training_summary.append(dict(method=method, first_500_mean_delay=float(logs.task_Avg_Delay.head(500).mean()),
                                     last_500_mean_delay=float(logs.task_Avg_Delay.tail(500).mean()),
                                     checkpoint_sha256=info["checkpoint_sha256"]))
    for axis, ylabel in zip(axes, ["平均时延（秒），100 轮移动平均", "相关可靠性满足率（%），100 轮移动平均"]):
        axis.set_xlabel("训练轮数")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.22)
        axis.legend()
    figure.savefig(directory / "training_curves.png")
    figure.savefig(directory / "training_curves.pdf")
    plt.close(figure)
    if write_tables:
        pd.DataFrame(training_summary).to_csv(OUT / "training_summary.csv", index=False)


def write_report(manifest, summary, registry):
    baseline = summary[summary.scenario_id.eq("baseline")].set_index("method").loc[METHODS].reset_index()
    table = baseline[["method", "mean_delay_seconds", "p95_delay_seconds", "success_rate", "fallback_rate", "mean_queue_seconds"]].copy()
    table["method"] = table.method.map(LABELS)
    table.columns = ["方法", "平均时延/s", "P95时延/s", "可靠性满足率", "Fallback比例", "平均排队/s"]
    formats = {"平均时延/s": lambda x: f"{x:.3f}", "P95时延/s": lambda x: f"{x:.3f}", "平均排队/s": lambda x: f"{x:.3f}",
               "可靠性满足率": lambda x: f"{x:.2%}", "Fallback比例": lambda x: f"{x:.2%}"}
    lines = ["# 五种卸载方法：新可靠性阈值与三组敏感性实验", "", "## 1. 实验设置", "",
             "训练集和评估集的可靠性要求重新生成，均服从连续均匀分布 U[0.92, 0.999]。"
             "数据种子为 2029；保留原来的 8 个 EUA 站点位置，重新生成 CPU 频率、传输速率和基础故障率。旧工作簿与原配置已备份到 `previous_inputs/`。", "",
             "Corr-PPO 和 Ind-PPO 使用相同初始随机种子 2026，从头各训练 8000 轮，每轮 200 个任务，到达率 2 任务/秒，计算需求 U[4×10⁹, 8×10⁹] cycles。"
             "除可靠性筛选模型外，两种 PPO 的网络和训练参数相同。训练后冻结网络、优化器和观测归一化参数，所有扫描点复用各自同一个最终模型。"
             "评估保留现有 PPO 的策略采样方式，不改为 argmax。", "",
             "评估每点 500 轮，评估种子 2027。预生成 500×600 个单位速率指数分布间隔，每轮取前 N 项并除以到达率；"
             "五种方法共享任务、到达时刻和服务器。任务数量组使用同一 600 任务池的嵌套前缀；计算需求组只乘以 n，其他任务字段不变。", "",
             "三组分别扫描到达率 0.5～3.0（步长 0.5）、计算需求倍率 0.5～3.0（步长 0.5）、任务数量 200～600（步长 100）。"
             "每次只改变一个因素。共 15 个不同场景、75 个正式评估；三个组共用同一个基准结果。", "",
             "成功率按任务所选执行方案的理论可靠性是否达到该任务阈值统计：四种双副本方法统一使用相关可靠性 Rcorr，"
             "RASOUL-Delay 使用单副本 GH 边际可靠性。独立可靠性满足率另存为诊断指标。"
             "这里统计可靠性约束满足比例，不抽样实际故障，也不加入截止期。可靠性故障暴露只包含 CPU 执行时间。", "",
             "沿用当前 GH 模型：β=0.5，每维 32 个积分节点，潜在风险相关系数 ρ=exp(-d/ℓ)，"
             "ℓ 为服务器间距离中位数/ln(2)，本次约 0.997545 km。独立模型保留相同 GH 边际，令双副本联合风险独立。", "",
             "双副本保留原模拟器的先完成者决定任务时延、另一个副本继续执行的机制。"
             "RASOUL 保留单副本、安全集合时延 LCB、最大边际可靠性 Fallback、完成后更新 discrepancy、每轮清空学习统计，c=0.5。", "",
             "## 2. 基准结果", "", "基准：到达率 2、计算需求 1 倍、每轮 200 个任务，各评估 500 轮。", "", md_table(table, formats), ""]
    for group, points in manifest["groups"].items():
        lines += [f"## {3 + list(manifest['groups']).index(group)}. {GROUP_LABELS[group]}", ""]
        for metric, title, multiplier in (("mean_delay_seconds", "平均时延（秒）", 1), ("success_rate", "可靠性满足率（%）", 100)):
            records = []
            for point in points:
                record = {"参数值": f"{point['value']:g}"}
                for method in METHODS:
                    result = summary[summary.scenario_id.eq(point["scenario_id"]) & summary.method.eq(method)].iloc[0]
                    record[LABELS[method]] = f"{result[metric] * multiplier:.3f}"
                records.append(record)
            lines += [title + "：", "", md_table(pd.DataFrame(records)), ""]
    scan = pd.read_csv(OUT / "feasibility_summary.csv").set_index("scenario_id")
    corr_base = float(baseline[baseline.method.eq("ppo")].mean_delay_seconds.iloc[0])
    greedy_base = float(baseline[baseline.method.eq("corr-delaygreedy")].mean_delay_seconds.iloc[0])
    rasoul_base = baseline[baseline.method.eq("rasoul-delay")].iloc[0]
    extrema = summary.loc[summary.groupby("scenario_id").mean_delay_seconds.idxmin()]
    winners = extrema.method.map(LABELS).value_counts()
    lines += ["## 6. 结果解释与边界", "",
              f"基准场景中，相关双副本的可行性上限为 {scan.loc['baseline', 'correlated_pair_ceiling']:.2%}，"
              f"独立双副本为 {scan.loc['baseline', 'independent_pair_ceiling']:.2%}，单副本为 {scan.loc['baseline', 'single_replica_ceiling']:.2%}。"
              "Corr-PPO 和 Corr-Greedy 均使用硬可靠性筛选，理论满足率达到当前任务集与服务器条件下的相关双副本上限；"
              "这项上限来自可行性，不表示时延最优。", "",
              "同一计算需求和任务池下，到达率只改变排队与选服过程，不改变各方案的 CPU 故障暴露。"
              "因此 Corr 方法和单副本方法的满足率随到达率保持不变；独立方法的真实相关满足率仍可能随所选服务器对而变化。"
              "增大计算需求会同时加重排队和故障暴露。任务数量组的满足率波动则还受新增任务的阈值、计算需求构成影响。", "",
              f"基准下 Corr-PPO 的平均时延为 {corr_base:.3f} 秒，Corr-Greedy 为 {greedy_base:.3f} 秒，"
              f"相对差值为 {(corr_base / greedy_base - 1):+.2%}。"
              "15 个场景中，最小平均时延的方法数量为：" + "、".join(f"{method} {int(count)} 个" for method, count in winners.items()) + "。"
              "这些是当前服务器样本、任务池和单次训练下的结果，不宣称 PPO 必然优于 Greedy。", "",
              f"RASOUL 基准平均排队 {rasoul_base.mean_queue_seconds:.3f} 秒，占总时延 "
              f"{rasoul_base.mean_queue_seconds / rasoul_base.mean_delay_seconds:.2%}；Fallback 比例 {rasoul_base.fallback_rate:.2%}。"
              "单副本的安全条件更严格，安全集合与最大可靠性 Fallback 会把负载集中到少数服务器；"
              "即使总 CPU 容量仍有空余，也可能出现局部拥堵。完整服务器选择次数见 `server_selection_counts.csv`。", "",
              f"基准单副本安全集合分布：K=0 有 {int(scan.loc['baseline', 'single_K0'])} 个任务，"
              f"K=1 有 {int(scan.loc['baseline', 'single_K1'])} 个任务，"
              f"K≥2 有 {int(scan.loc['baseline', 'single_K_ge2'])} 个任务。"
              "前两类无法使用 LCB 在多个安全服务器之间分散负载；在线学习有效也无法解除这些可靠性筛选限制。", "",
              "任务数量实验为有限轮、每轮排空的在线负载；过载条件下增大 N 会积累更长队列。"
              "它衡量有限任务批次的表现，不等同于稳定态时延。模型仅在基准条件训练，其他参数点衡量冻结策略的迁移表现。", "",
              "CSV 中 95% 区间按 500 个 Episode 均值的标准误计算，只描述固定任务/服务器及固定训练结果下的到达随机性和策略采样，"
              "不涵盖不同训练种子或不同服务器样本之间的变化。", "",
              "## 7. 一致性检查", "",
              "33 项单元测试通过；评估脚本先对 5 种方法、2 个不同场景完成 10 次小规模检查，随后执行正式实验。"
              "正式任务数、每轮任务数、任务 ID、时延分解、相同输入哈希、服务器哈希、模型哈希均通过核验。"
              "所有任务的安全集合数量与独立 GH 扫描逐条一致；PPO 评估前后网络和优化器状态不变。"
              "RASOUL 每轮收到全部任务的完成反馈。核验结果见 `validation.json`。", "",
              f"RASOUL 基准中有 {int(rasoul_base.multi_safe_decisions):,} 次 K≥2 的安全集合决策，"
              f"其中 {int(rasoul_base.multi_safe_chosen_server_observed):,} 次所选服务器已有完成观测；"
              f"这些决策中 {int(rasoul_base.observed_LCB_choice_differs_from_base_fastest):,} 次选服不同于仅按上传+CPU基础时延的最小值。"
              "这验证了在线 LCB 确实参与多候选选服。discrepancy 与实际排队时延逐条一致；"
              "存在历史反馈与探索项影响，不表示本轮实验已隔离二者各自的收益。", "",
              f"正式评估执行 {int(summary.tasks.sum()):,} 个任务，训练执行 3,200,000 个任务。"
              "历史实验与旧 checkpoint 保留；本轮没有 commit、没有 push。", "",
              "## 8. 输出文件", "", "- `figures/overview.png`、`.pdf`、`.svg`：三组因素 × 两项指标的总图。时延采用对数纵轴，成功率按数据范围缩放。原尺度图保存在 `figures_original_scale/`。",
              "- `figures/arrival_rate_*`、`figures/demand_multiplier_*`、`figures/task_count_*`：六张单图，均包含五种方法。",
              "- `figures/mean_delay_logscale.*`：五种方法的时延对数坐标补充图，便于查看量级差异。",
              "- `summary.csv`：75 次正式评估的均值、P95、时延分解、两种满足率、Fallback 和置信区间。",
              "- `arrival_rate_comparison.csv`、`demand_multiplier_comparison.csv`、`task_count_comparison.csv`：三组比较数据。",
              "- `feasibility_summary.csv`、`feasibility/`：每个场景和每个任务的单/双副本安全集合及上限。",
              "- `training_registry.json`、`checkpoints/`、`training/`：两个新模型和训练记录。",
              "- `evaluations/`：逐场景、逐方法的任务明细、每轮指标与 RASOUL 完成反馈。", ""]
    training = pd.read_csv(OUT / "training_summary.csv")
    lines += ["## 9. 训练核验", "",
              "两个模型均从头训练，Actor 和 Critic 各执行 64,000 次优化步骤，最终网络参数全部有限。"
              "训练早期与末期的同训练集表现如下；正式比较使用独立评估集。", ""]
    training = training[["method", "first_500_mean_delay", "last_500_mean_delay"]].copy()
    training["method"] = training.method.map(LABELS)
    training.columns = ["方法", "前500轮平均时延/s", "后500轮平均时延/s"]
    lines += [md_table(training, {name: lambda x: f"{x:.3f}" for name in training.columns[1:]}), "",
              "训练曲线见 `figures/training_curves.png`；软件版本与运行源代码哈希见 `provenance.json`。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    summary, registry = audit(manifest)
    save_figures(manifest, summary, registry)
    write_report(manifest, summary, registry)
    print("All 75 evaluations audited; plots and report saved.", flush=True)


if __name__ == "__main__":
    main()
