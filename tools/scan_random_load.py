"""Step 5: Uniform Random load scan with paired arrivals, actions and spatial risk."""

import argparse
from collections import defaultdict
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
from core.main_loop import MainLoop
from core.server import Server
from core.task import Task
from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    compute_baseline_correlation_length_km,
    map_spatial_risk_to_effective_failure_rates,
    sample_spatial_risk_field,
)
from tools.validate_spatial_reliability import markdown_table

TASK_METRICS = [
    "mean_latency_sec", "p95_latency_sec", "mean_upload_time_sec",
    "mean_cpu_waiting_time_sec", "mean_cpu_execution_time_sec",
]
SERVER_METRICS = [
    "cpu_utilization_arrival", "cpu_utilization_full", "mean_waiting_queue_length",
    "max_waiting_queue_length", "waiting_queue_length_at_last_arrival",
    "replica_count", "assigned_cpu_work_sec",
]


def make_environment(server_profiles):
    env, state = simpy.Environment(), EnvironmentState()
    for server_id, profile in server_profiles.items():
        server = Server(
            env, server_id, profile["Site_ID"], profile["Processing_Frequency"],
            profile["Base_Failure_Rate"], profile["Transmission_Rate"],
            profile["Latitude"], profile["Longitude"],
        )
        state.add_server_and_init_environment(server)
    return env, state


def episode_randomness(seed, episode_id, task_count, pair_count, correlation, base_rates):
    """Independent per-episode streams (5, episode_id, 0/1/2), reused across rates."""
    arrival_rng, action_rng, risk_rng = [
        np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(5, episode_id, stream)))
        for stream in range(3)
    ]
    arrival_base = arrival_rng.exponential(size=task_count)
    actions = action_rng.integers(pair_count, size=task_count)
    standard_normal = risk_rng.standard_normal(correlation.shape[0])
    risk = sample_spatial_risk_field(correlation, standard_normal=standard_normal)
    rates = map_spatial_risk_to_effective_failure_rates(base_rates, risk, parameters.SPATIAL_RISK_BETA_P)
    return {"arrival_base": arrival_base, "actions": actions, "standard_normal": standard_normal, "risk": risk, "effective_rates": rates}


def window_statistics(records, horizon):
    """Integrate actual CPU/wait intervals; coalesce equal-time queue events."""
    busy = sum(max(0., min(finish, horizon) - max(start, 0.)) for enter, start, finish in records)
    waiting_area = sum(max(0., min(start, horizon) - max(enter, 0.)) for enter, start, finish in records)
    events = defaultdict(int)
    for enter, start, finish in records:
        if enter < start and enter <= horizon:
            events[enter] += 1
            if start <= horizon:
                events[start] -= 1
    queue = maximum = 0
    for time in sorted(events):
        queue += events[time]
        maximum = max(maximum, queue)
    last_queue = sum(enter <= horizon < start for enter, start, finish in records)
    assert queue == last_queue
    return busy, waiting_area, maximum, last_queue


def run_episode(arrival_rate, episode_id, basis, server_profiles, task_profiles, pairs, distances, correlation, *, action_selector=None):
    env, state = make_environment(server_profiles)
    state.set_episode_spatial_risk_context(
        list(server_profiles), distances, correlation, basis["risk"], basis["effective_rates"]
    )
    tasks = []
    actions = []

    def arrivals():
        for index, (task_id, profile) in enumerate(task_profiles.items()):
            yield env.timeout(basis["arrival_base"][index] / arrival_rate)
            task = Task(env, state, task_id, profile)
            tasks.append(task)
            state.add_task(task)
            action = basis["actions"][index] if action_selector is None else action_selector(state.get_state(task))
            actions.append(action)
            a_id, b_id = pairs[action]
            a, b = state.get_server_by_id(a_id), state.get_server_by_id(b_id)
            task.replica_A_reliability = state.compute_replica_reliability(task, a)
            task.replica_B_reliability = state.compute_replica_reliability(task, b)
            task.pair_reliability = state.compute_pair_reliability(task, a, b)
            env.process(task.execute_task(a, b))

    env.process(arrivals())
    env.run()  # Drain every replica, including losers after first task completion.
    assert len(tasks) == len(task_profiles) == 200
    np.testing.assert_allclose([t.arrival_time for t in tasks], np.cumsum(basis["arrival_base"] / arrival_rate), rtol=0, atol=1e-10)
    for index, task in enumerate(tasks):
        assert (task.replica_A["server_id"], task.replica_B["server_id"]) == pairs[actions[index]]
    return collect_episode_metrics(arrival_rate, episode_id, tasks, state, env.now)


def collect_episode_metrics(arrival_rate, episode_id, tasks, state, episode_end):
    """Shared post-drain metrics for the existing training and evaluation loops."""
    assert len(tasks) == 200
    arrival_window = tasks[-1].arrival_time
    records = {server_id: [] for server_id in state.servers}
    assigned_work = dict.fromkeys(state.servers, 0.)
    latency_parts = []
    for task in tasks:
        winner = min([task.replica_A, task.replica_B], key=lambda replica: replica["finish_time"])
        delay = task.task_completion_time - task.arrival_time
        parts = [
            winner["queue_enter_time"] - task.arrival_time,
            winner["cpu_start_time"] - winner["queue_enter_time"],
            winner["finish_time"] - winner["cpu_start_time"],
        ]
        assert np.isfinite(delay) and delay > 0
        assert task.task_completion_time == winner["finish_time"]
        np.testing.assert_allclose(delay, sum(parts), rtol=0, atol=1e-10)
        latency_parts.append([delay, *parts])
        for replica in [task.replica_A, task.replica_B]:
            server_id = replica["server_id"]
            enter, start, finish = [replica[name] for name in ["queue_enter_time", "cpu_start_time", "finish_time"]]
            assert task.arrival_time < enter <= start < finish
            server = state.get_server_by_id(server_id)
            service_time = task.computation_demand / server.processing_frequency
            np.testing.assert_allclose(finish - start, service_time, rtol=0, atol=1e-10)
            records[server_id].append((enter, start, finish))
            assigned_work[server_id] += service_time
    assert sum(len(r) for r in records.values()) == 400
    assert episode_end == max(finish for r in records.values() for enter, start, finish in r)
    data = np.asarray(latency_parts)
    episode_metrics = {
        "arrival_rate": arrival_rate, "episode_id": episode_id,
        "mean_latency_sec": data[:, 0].mean(), "p95_latency_sec": np.percentile(data[:, 0], 95),
        "mean_upload_time_sec": data[:, 1].mean(), "mean_cpu_waiting_time_sec": data[:, 2].mean(),
        "mean_cpu_execution_time_sec": data[:, 3].mean(),
        "arrival_window_sec": arrival_window, "episode_end_sec": episode_end,
    }
    server_metrics = []
    for server_id, intervals in records.items():
        busy, waiting_area, maximum, last_queue = window_statistics(intervals, arrival_window)
        full_busy = sum(finish - start for enter, start, finish in intervals)
        np.testing.assert_allclose(full_busy, assigned_work[server_id], rtol=0, atol=1e-9)
        u_arrival, u_full = busy / arrival_window, full_busy / episode_end
        assert -1e-12 <= u_arrival <= 1 + 1e-12 and -1e-12 <= u_full <= 1 + 1e-12
        ordered = sorted(intervals, key=lambda interval: interval[1])
        assert all(left[2] <= right[1] for left, right in zip(ordered, ordered[1:]))
        server = state.get_server_by_id(server_id)
        assert server.queue.count == 0 and not server.queue.queue
        assert state.servers[server_id]["running_replica"] is None
        assert not state.servers[server_id]["waiting_replicas"]
        server_metrics.append({
            "arrival_rate": arrival_rate, "episode_id": episode_id, "server_id": server_id,
            "cpu_utilization_arrival": u_arrival, "cpu_utilization_full": u_full,
            "mean_waiting_queue_length": waiting_area / arrival_window,
            "max_waiting_queue_length": maximum,
            "waiting_queue_length_at_last_arrival": last_queue,
            "replica_count": len(intervals), "assigned_cpu_work_sec": assigned_work[server_id],
        })
    np.testing.assert_allclose(sum(assigned_work.values()), sum(finish-start for r in records.values() for enter, start, finish in r), rtol=0, atol=1e-8)
    return episode_metrics, server_metrics, tasks


def smoke_checks(bases, rates, server_profiles, task_profiles, pairs, distances, correlation, base_rates, seed):
    assert window_statistics([(0., 0., 2.), (.5, 2., 3.), (1.5, 3., 4.)], 1.) == (1., .5, 1, 1)
    assert window_statistics([(.5, 1., 1.5), (1., 1.5, 2.)], 1.) == (0., .5, 1, 1)
    assert window_statistics([(1., 1., 2.)], 1.) == (0., 0., 0, 0)
    assert len(pairs) == 28 and len(set(pairs)) == 28 and all(a < b for a, b in pairs)
    for episode_id, basis in enumerate(bases[:3], start=1):
        repeated = episode_randomness(seed, episode_id, 200, 28, correlation, base_rates)
        for key in basis:
            np.testing.assert_array_equal(basis[key], repeated[key])
        low = run_episode(rates[0], episode_id, basis, server_profiles, task_profiles, pairs, distances, correlation)
        high = run_episode(rates[-1], episode_id, basis, server_profiles, task_profiles, pairs, distances, correlation)
        repeated_low = run_episode(rates[0], episode_id, repeated, server_profiles, task_profiles, pairs, distances, correlation)
        assert low[:2] == repeated_low[:2]
        for result in [low, high]:
            tasks = result[2]
            assert [t.id for t in tasks] == list(task_profiles)
            assert [(t.task_size, t.computation_demand, t.reliability_requirement) for t in tasks] == [
                (p["Task_Size"], p["Computation_Demand"], p["Reliability_Requirement"]) for p in task_profiles.values()
            ]
        np.testing.assert_allclose([t.arrival_time*rates[0] for t in low[2]], [t.arrival_time*rates[-1] for t in high[2]], rtol=0, atol=1e-10)
        assert [(t.replica_A["server_id"], t.replica_B["server_id"]) for t in low[2]] == [(t.replica_A["server_id"], t.replica_B["server_id"]) for t in high[2]]
        for key in ["replica_A_reliability", "replica_B_reliability", "pair_reliability"]:
            np.testing.assert_array_equal([getattr(t, key) for t in low[2]], [getattr(t, key) for t in high[2]])
        altered = dict(basis)
        altered["risk"] = np.zeros(len(server_profiles))
        altered["effective_rates"] = map_spatial_risk_to_effective_failure_rates(base_rates, altered["risk"], parameters.SPATIAL_RISK_BETA_P)
        assert run_episode(rates[0], episode_id, altered, server_profiles, task_profiles, pairs, distances, correlation)[:2] == low[:2]
    for left, right in zip(bases[:3], bases[1:3]):
        for key in ["arrival_base", "actions", "standard_normal"]:
            assert not np.array_equal(left[key], right[key])
    print("Smoke passed: 200 tasks/400 completed replicas; CRN, reproducibility, queue boundaries and risk-independent physical delays.", flush=True)


def summarize(episodes, servers):
    rows = []
    for arrival_rate, group in episodes.groupby("arrival_rate", sort=True):
        for metric in TASK_METRICS:
            rows.append({"arrival_rate": arrival_rate, "scope": "episode", "server_id": "overall", "metric": metric, "mean": group[metric].mean(), "sd": group[metric].std(ddof=1)})
    for (arrival_rate, server_id), group in servers.groupby(["arrival_rate", "server_id"], sort=True):
        for metric in SERVER_METRICS:
            rows.append({"arrival_rate": arrival_rate, "scope": "server", "server_id": str(server_id), "metric": metric, "mean": group[metric].mean(), "sd": group[metric].std(ddof=1)})
    return pd.DataFrame(rows)


def plot_results(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = summary[summary.scope == "episode"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, label in [("mean_latency_sec", "Mean task latency"), ("p95_latency_sec", "Episode P95 latency")]:
        subset = overall[overall.metric == metric]
        ax.errorbar(subset.arrival_rate, subset["mean"], yerr=subset.sd, marker="o", capsize=3, label=label)
    ax.set(xlabel="Arrival rate (tasks/s)", ylabel="Latency (s)", title="Uniform Random: mean and Episode P95 (mean +/- SD)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(output_dir / "latency.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    subset = overall[overall.metric == "mean_cpu_waiting_time_sec"]
    ax.errorbar(subset.arrival_rate, subset["mean"], yerr=subset.sd, marker="o", capsize=3)
    ax.set(xlabel="Arrival rate (tasks/s)", ylabel="Winner CPU waiting time (s)", title="Uniform Random: CPU waiting (Episode mean +/- SD)")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(output_dir / "cpu_waiting.png", dpi=160); plt.close(fig)
    for filename, metrics, labels in [
        ("cpu_utilization.png", ["cpu_utilization_arrival", "cpu_utilization_full"], ["Arrival-window CPU utilization", "Full-Episode CPU utilization"]),
        ("waiting_queues.png", ["mean_waiting_queue_length", "waiting_queue_length_at_last_arrival"], ["Time-weighted waiting queue", "Waiting queue at last arrival"]),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, label in zip(axes, metrics, labels):
            for server_id in sorted(summary[summary.scope == "server"].server_id.unique(), key=int):
                subset = summary[(summary.scope == "server") & (summary.server_id == server_id) & (summary.metric == metric)]
                ax.plot(subset.arrival_rate, subset["mean"], marker="o", label=f"server{server_id}")
            ax.set(xlabel="Arrival rate (tasks/s)", ylabel=label, title=label + " (Episode means)")
            ax.grid(alpha=.3)
        axes[-1].legend(fontsize=8); fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160); plt.close(fig)
    return matplotlib.__version__


def write_report(args, server_frame, mean_cycles, critical_rate, ell, rates, bases, summary, mpl_version):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
    overall = summary[summary.scope == "episode"]
    task_rows = []
    for arrival_rate in rates:
        group = overall[overall.arrival_rate == arrival_rate].set_index("metric")
        task_rows.append({"arrival_rate": arrival_rate, **{metric: f"{group.loc[metric, 'mean']:.6f} ± {group.loc[metric, 'sd']:.6f}" for metric in TASK_METRICS}})
    task_table = pd.DataFrame(task_rows)
    queue_means = summary[(summary.scope == "server") & (summary.metric == "mean_waiting_queue_length")].pivot(index="arrival_rate", columns="server_id", values="mean")
    wait_means = overall[overall.metric == "mean_cpu_waiting_time_sec"].set_index("arrival_rate")["mean"]
    candidate_rates = [wait_means.idxmin(), queue_means.index[(queue_means >= 1).any(axis=1)][0], rates[rates > critical_rate][0]]
    slow_id = str(server_frame.loc[server_frame.Processing_Frequency.idxmin(), "Server_ID"])
    fast_id = str(server_frame.loc[server_frame.Processing_Frequency.idxmax(), "Server_ID"])

    def metric_mean(rate, metric, server_id="overall"):
        return summary[(summary.arrival_rate == rate) & (summary.server_id == server_id) & (summary.metric == metric)]["mean"].iloc[0]

    low_rate, high_rate = rates[0], rates[-1]
    delay_growth = metric_mean(high_rate, "mean_latency_sec") - metric_mean(low_rate, "mean_latency_sec")
    waiting_growth = metric_mean(high_rate, "mean_cpu_waiting_time_sec") - metric_mean(low_rate, "mean_cpu_waiting_time_sec")
    analysis = [
        f"- 低负载 {low_rate}：获胜副本平均等待 {metric_mean(low_rate, 'mean_cpu_waiting_time_sec'):.6f} s；排队较少但不是严格为零。",
        f"- 最低 CPU 的 server{slow_id} 最早形成明显积压：到达率 {candidate_rates[1]} 时其时间加权队列均值 {metric_mean(candidate_rates[1], 'mean_waiting_queue_length', slow_id):.6f}，末次到达队列均值 {metric_mean(candidate_rates[1], 'waiting_queue_length_at_last_arrival', slow_id):.6f}。这是描述性拐点，不是硬容量判定。",
        f"- 从 {low_rate} 到 {high_rate}，平均延迟增加 {delay_growth:.6f} s，其中等待增加 {waiting_growth:.6f} s（占增量 {waiting_growth/delay_growth:.2%}）；上传和获胜 CPU 服务均值仅小幅变化。",
        f"- P95 从 {metric_mean(low_rate, 'p95_latency_sec'):.6f} 到 {metric_mean(high_rate, 'p95_latency_sec'):.6f} s，倍率 {metric_mean(high_rate, 'p95_latency_sec')/metric_mean(low_rate, 'p95_latency_sec'):.3f}；平均延迟倍率 {metric_mean(high_rate, 'mean_latency_sec')/metric_mean(low_rate, 'mean_latency_sec'):.3f}，尾部恶化更明显。",
        f"- 在 {high_rate} 下，server{slow_id} 到达/完整利用率 {metric_mean(high_rate, 'cpu_utilization_arrival', slow_id):.6f}/{metric_mean(high_rate, 'cpu_utilization_full', slow_id):.6f}，平均队列 {metric_mean(high_rate, 'mean_waiting_queue_length', slow_id):.6f}，最大队列的 Episode 均值 {metric_mean(high_rate, 'max_waiting_queue_length', slow_id):.6f}，末次到达队列均值 {metric_mean(high_rate, 'waiting_queue_length_at_last_arrival', slow_id):.6f}。",
        f"- 同一高负载下，server{fast_id} 到达/完整利用率 {metric_mean(high_rate, 'cpu_utilization_arrival', fast_id):.6f}/{metric_mean(high_rate, 'cpu_utilization_full', fast_id):.6f}，平均队列 {metric_mean(high_rate, 'mean_waiting_queue_length', fast_id):.6f}，末次到达队列均值 {metric_mean(high_rate, 'waiting_queue_length_at_last_arrival', fast_id):.6f}。",
        f"- 工作量证据：server{slow_id}/server{fast_id} 平均分配副本 {metric_mean(high_rate, 'replica_count', slow_id):.6f}/{metric_mean(high_rate, 'replica_count', fast_id):.6f}，实际分配 CPU 服务时间 {metric_mean(high_rate, 'assigned_cpu_work_sec', slow_id):.6f}/{metric_mean(high_rate, 'assigned_cpu_work_sec', fast_id):.6f} s；副本数量接近，但异构 CPU 使慢服务器承受更多服务秒数。",
        "- 因此 Uniform Random 的均匀副本分配没有平衡服务负载：慢服务器积压、快服务器仍有闲置时段，排空尾段进一步暴露容量利用不均。这一判断同时基于分配工作量、实际运行区间和队列，不能据此量化未运行策略的改善幅度。",
        "- 参考临界率附近和之后，慢服务器积压增强；有限 Episode 有初始空闲期及最终排空期，所以实测利用率不会等于稳态 rho，也不会超过 1。",
    ]
    candidate_rows = []
    for label, rate, reason in zip(["低负载", "中等负载", "较高负载"], candidate_rates, ["扫描中获胜副本等待均值最低", "扫描中首次出现服务器时间加权平均等待队列至少一个副本", "扫描中首个超过最慢服务器参考临界率的到达率，并结合实际队列确认竞争差异"]):
        candidate_rows.append({
            "场景": label, "arrival_rate": rate,
            "mean_latency_sec": metric_mean(rate, "mean_latency_sec"),
            "p95_latency_sec": metric_mean(rate, "p95_latency_sec"),
            "mean_cpu_waiting_time_sec": metric_mean(rate, "mean_cpu_waiting_time_sec"),
            "最慢服务器平均队列": metric_mean(rate, "mean_waiting_queue_length", slow_id),
            "依据": reason,
        })
    counts = np.bincount(np.concatenate([basis["actions"] for basis in bases]), minlength=28)
    cpu_table = server_frame[["Server_ID", "Site_ID", "Processing_Frequency", "Transmission_Rate"]]
    lines = [
        "# Step 5：Uniform Random 负载扫描与服务器排队性能分析", "",
        "## A. 代码与实验版本", "",
        f"基准：`{branch}@{head}`。仅新增独立扫描脚本与本目录结果；输入 Excel、核心执行、风险模型、41 维状态和 PPO 均保留。",
        f"Python {sys.version.split()[0]}；NumPy {np.__version__}；pandas {pd.__version__}；SimPy {simpy.__version__}；matplotlib {mpl_version}。",
        f"固定种子 {args.seed}；每个 Episode e 独立派生 SeedSequence (5,e,0)/(5,e,1)/(5,e,2)，分别生成 Exp(1) 到达基础数、均匀动作索引、标准高斯向量。",
        "不同到达率复用同一 Episode 的三个基础轨迹，间隔为 X/lambda；不重置全局 np.random.seed，不创建 MainLoop 实例或调用 PPO。",
        f"命令：`{sys.executable} tools/scan_random_load.py`；正式配置为 {args.episodes} Episodes/到达率，200 固定任务/Episode。",
        "", "## B. 数据与负载参数", "",
        "固定源：data/server_info.xlsx、data/task_parameters.xlsx；服务器按 Server_ID、任务按 Task_ID 排序，复用 MainLoop 静态 profile builders 和动作字典序。",
        "CPU 为 cycles/s，上传速率为 MB/s：", "", markdown_table(cpu_table), "",
        f"平均需求 C_bar={mean_cycles:.12f} cycles；f_min={server_frame.Processing_Frequency.min():.12f} cycles/s；lambda_crit=4*f_min/C_bar={critical_rate:.12f} tasks/s。",
        f"最大扫描值=0.5*ceil((lambda_crit+1.5)/0.5)={rates[-1]}；扫描列表：{rates.tolist()}。",
        "lambda_crit 是均匀选择下平均工作量的稳态参考，不是有限 200 任务系统的硬容量界限。",
        f"beta={parameters.SPATIAL_RISK_BETA_P}；ell_0={ell:.12f} km，由当前坐标计算；风险场经现有函数映射到有效故障率。",
        "风险只用于条件理论可靠性计算，没有实际失败事件、动作 Mask 或取消副本，不改变物理延迟。",
        f"28 对的经验动作频数范围 {counts.min()}–{counts.max()}，每对理论期望 {len(bases)*200/28:.6f}；每个动作通过独立 RNG integers(28) 等概率抽样。",
        "", "## C. Episode 级任务延迟：Mean ± SD", "",
        "每个 Episode 先算 200 个任务的均值与 P95（NumPy 默认线性百分位），再跨 Episode 汇总；SD 使用 ddof=1，不使用 pooled P95。",
        "获胜副本以最早 finish 决定（完全相同时按 replica_A 优先），上传/等待/执行分解来自实际时间戳；另一个副本继续消耗 CPU。",
        "", markdown_table(task_table), "",
        "![任务延迟](latency.png)", "", "![CPU等待](cpu_waiting.png)", "",
        "## D. 服务器 CPU 与排队", "",
        "到达窗口 [0,T_arr]，T_arr 为第 200 个任务到达时间；完整窗口 [0,T_end]，T_end 为全部 400 个副本的最大 finish。",
        "实际 CPU 忙碌区间与各窗口取交集；分配工作量单独记录，不当成到达窗口利用率。",
        "等待区间采用 [queue_enter,cpu_start)，零时长等待排除，同一时刻的进入/退出先合并再更新队列数。",
        "时间加权队列均值为等待区间与到达窗口交集长度之和/T_arr；最大队列与末次到达积压仅包含已上传完、未开始 CPU 的副本。",
        "末次到达快照采用该时刻先前 CPU 事件已处理完的半开区间语义；当前第 200 个任务还没有上传结果。在途副本不计入。",
        "所有表项均为先按 Episode 计算，再跨 Episode 汇总的 mean±SD；副本数单位为个、工作量为秒、利用率为比例。",
    ]
    for arrival_rate in rates:
        rows = []
        for server_id in server_frame.Server_ID:
            group = summary[(summary.scope == "server") & (summary.arrival_rate == arrival_rate) & (summary.server_id == str(server_id))].set_index("metric")
            row = {"server_id": server_id, "rho_reference": arrival_rate*mean_cycles/(4*server_frame.loc[server_frame.Server_ID == server_id, 'Processing_Frequency'].iloc[0])}
            row.update({metric: f"{group.loc[metric, 'mean']:.6f} ± {group.loc[metric, 'sd']:.6f}" for metric in SERVER_METRICS})
            rows.append(row)
        lines.extend(["", f"### 到达率 {arrival_rate} tasks/s", "", markdown_table(pd.DataFrame(rows))])
    lines.extend([
        "", "![CPU利用率](cpu_utilization.png)", "", "![等待队列](waiting_queues.png)", "",
        "## E. 验证与解释范围", "",
        "先对前三个独立 Episode 做最低/最高负载 smoke，并重跑低负载；验证固定任务与顺序、200 次选择、400 副本完成、资源排空、正有限延迟、分解、CPU 利用率范围和工作量守恒。",
        "跨到达率的动作/风险/基础到达轨迹一致，不同 Episode 三个流不同；改变风险场不改变 Random 物理指标。",
        "另验证忙碌区间跨最终到达时刻正确截断、到达窗口之后进入的等待副本排除、同刻队列事件无瞬时虚假峰值。",
        "上传与执行均由固定物理参数决定，但获胜者可能随排队改变，因此获胜副本上传/执行均值可以随负载变化。",
        "CPU 到达窗口利用率接近饱和、等待积压和排空尾段应结合分配 CPU 工作量分析，不能只凭利用率差异宣布容量浪费。",
        "这是一种策略、固定 200 任务的有限 Episode 扫描，不证明其他策略改善幅度或系统稳态稳定性。",
        "", *analysis,
        "", "## F. 后续场景建议", "",
        "以下候选在扫描完成后依据实际结果选择。平均等待队列一个副本只作为中等竞争场景的解释依据，不是模型验收阈值或稳定性界限。",
        "", markdown_table(pd.DataFrame(candidate_rows)), "",
        "保留更高到达率结果作为严重积压参考；这些建议不启动 PPO、不证明 PPO 改善，也不自动修改正式训练到达率或可靠性参数。",
        "", "## 输出文件", "",
        "episode_metrics.csv、server_metrics.csv、summary.csv、report.md，以及 latency.png、cpu_waiting.png、cpu_utilization.png、waiting_queues.png。",
        "summary.csv 为长表：scope=episode 时 server_id=overall，scope=server 时为真实 Server_ID；每项 mean/sd 均以 Episode 为统计单位。",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=parameters.EXPERIMENT_SEED)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR) / "step5_random_load")
    args = parser.parse_args()
    server_frame = pd.read_excel(SERVER_INFO_PATH, sheet_name="Servers", dtype={"Site_ID": "string"}).sort_values("Server_ID").reset_index(drop=True)
    task_frame = pd.read_excel(TRAIN_TASK_PATH).sort_values("Task_ID").reset_index(drop=True)
    server_profiles = MainLoop._build_server_profiles(server_frame)
    task_profiles = MainLoop._build_task_profiles(task_frame)
    pairs = MainLoop.generate_combinations()
    env, state = make_environment(server_profiles)
    servers = [info["server_object"] for info in state.servers.values()]
    server_ids, distances = build_distance_matrix(servers)
    ell = compute_baseline_correlation_length_km(distances)
    correlation = build_spatial_correlation_matrix(distances, ell)
    base_rates = np.array([server_profiles[i]["Base_Failure_Rate"] for i in server_ids])
    mean_cycles = task_frame.Computation_Demand.mean()
    critical_rate = 4*server_frame.Processing_Frequency.min()/mean_cycles
    maximum_rate = .5*np.ceil((critical_rate+1.5)/.5)
    rates = np.arange(.5, maximum_rate+.25, .5)
    bases = [episode_randomness(args.seed, e, len(task_profiles), len(pairs), correlation, base_rates) for e in range(1, args.episodes+1)]
    smoke_checks(bases, rates, server_profiles, task_profiles, pairs, distances, correlation, base_rates, args.seed)
    episode_rows, server_rows = [], []
    for arrival_rate in rates:
        for episode_id, basis in enumerate(bases, start=1):
            episode_metrics, server_metrics, tasks = run_episode(arrival_rate, episode_id, basis, server_profiles, task_profiles, pairs, distances, correlation)
            episode_rows.append(episode_metrics)
            server_rows.extend(server_metrics)
        print(f"Completed rate={arrival_rate:.1f}: {args.episodes} Episodes, 200 tasks and 400 completed replicas each.", flush=True)
    episodes, server_metrics = pd.DataFrame(episode_rows), pd.DataFrame(server_rows)
    summary = summarize(episodes, server_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in [("episode_metrics.csv", episodes), ("server_metrics.csv", server_metrics), ("summary.csv", summary)]:
        frame.to_csv(args.output_dir / filename, index=False, float_format="%.17g")
    mpl_version = plot_results(summary, args.output_dir)
    write_report(args, server_frame, mean_cycles, critical_rate, ell, rates, bases, summary, mpl_version)
    print(f"Finished {len(rates)*args.episodes} Episodes; lambda_crit={critical_rate:.12f}; results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
