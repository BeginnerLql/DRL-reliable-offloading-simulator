"""Frozen Ind-PPO sweep; compare existing matched PPO and greedy evaluations."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))

import numpy as np
import pandas as pd
from scipy.stats import t
import torch

import Project_main
from config.paths import EVALUATION_TASK_PATH, SERVER_INFO_PATH
from tools import run_experiments as runner

TRAINING = ROOT / "results/ind_ppo_train_eval_20261103_165626_760475"
CHECKPOINT = TRAINING / "ind-ppo_20261103_165653_baseline.pt"
RATE2 = TRAINING / "ind-ppo_20261103_165653_766674_arrival2_eval500"
REFERENCE = ROOT / "results/three_method_comparison_20261102_225141_483260/summary.csv"
METHODS = ("Corr-PPO", "Ind-PPO", "Corr-delayGreedy", "Indep-delayGreedy")
RATES = (.5, 1, 1.5, 2, 2.5, 3)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(name, value):
    with (OUT / name).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)


def compare(registry):
    reference = pd.read_csv(REFERENCE)
    reference_map = {(row.arrival_rate, row.method): Path(row.source_directory)
                     for row in reference.itertuples()}
    summaries, paired_rows, differences, audits = [], [], [], []
    common_keys = ("total_episodes", "taskno", "TASK_ARRIVAL_RATE", "EXPERIMENT_SEED", "NUM_SERVERS",
                   "SPATIAL_RISK_ENABLED", "SPATIAL_RISK_BETA_P", "SPATIAL_RISK_GH_ORDER",
                   "RELIABILITY_ACTION_MASK_ENABLED")
    for rate in RATES:
        reference_tasks, reference_settings, reference_hashes = None, None, None
        episode_delays = {}
        for method in METHODS:
            directory = registry[float(rate)] if method == "Ind-PPO" else reference_map[rate, method]
            settings = json.loads((directory / "parameters.json").read_text())
            status = json.loads((directory / "status.json").read_text())
            assert status["status"] == "completed" and status["last_completed_episode"] == 500
            assert settings["model_summary"].lower() == {"Corr-PPO": "ppo", "Ind-PPO": "ind-ppo"}.get(method, method.lower())
            common = {key: settings[key] for key in common_keys}
            assert common["TASK_ARRIVAL_RATE"] == rate and common["EXPERIMENT_SEED"] == 2027
            hashes = {name: digest(directory / name) for name in ("server_info.xlsx", "task_parameters.xlsx")}
            tasks = pd.read_csv(directory / "task_results.csv")
            assert len(tasks) == 100000 and not tasks.duplicated(["episode", "task_id"]).any()
            assert tasks.groupby("episode").size().tolist() == [200] * 500
            assert tasks.server_A_id.ne(tasks.server_B_id).all() and tasks.task_delay.ge(0).all()
            np.testing.assert_allclose(tasks.task_completion_time,
                                       np.minimum(tasks.replica_A_finish_time, tasks.replica_B_finish_time))
            np.testing.assert_allclose(tasks.task_delay, tasks.task_completion_time - tasks.arrival_time)
            np.testing.assert_allclose(tasks.task_delay,
                                       tasks.task_upload_delay + tasks.task_queue_delay + tasks.task_execution_delay)
            np.testing.assert_allclose(tasks.pair_reliability_independent,
                                       1 - (1 - tasks.replica_A_reliability) * (1 - tasks.replica_B_reliability))
            for field in ("correlated", "independent"):
                np.testing.assert_array_equal(tasks[f"requirement_satisfied_{field}"],
                                              tasks[f"pair_reliability_{field}"] >= tasks.reliability_requirement)
            mask_model = "independent" if method in ("Ind-PPO", "Indep-delayGreedy") else "correlated"
            np.testing.assert_array_equal(tasks.feasible_action_count > 0,
                                          tasks[f"requirement_satisfied_{mask_model}"])
            np.testing.assert_array_equal(tasks.mask_fallback_used, tasks.feasible_action_count == 0)
            matched = tasks[["episode", "task_id", "task_size", "computation_demand",
                             "reliability_requirement", "arrival_time"]]
            if reference_tasks is None:
                reference_tasks, reference_settings, reference_hashes = matched, common, hashes
            else:
                np.testing.assert_allclose(matched, reference_tasks, rtol=1e-12, atol=1e-12)
                assert common == reference_settings and hashes == reference_hashes
            per_episode = tasks.groupby("episode").agg(
                mean_delay_seconds=("task_delay", "mean"),
                rsr_correlated=("requirement_satisfied_correlated", "mean"),
                rsr_independent=("requirement_satisfied_independent", "mean"),
                fallback_rate=("mask_fallback_used", "mean"),
            )
            episode_delays[method] = per_episode.mean_delay_seconds
            paired_rows.append(per_episode.assign(arrival_rate=rate, method=method).reset_index())
            summaries.append(dict(
                arrival_rate=rate, method=method, episodes=500, tasks=len(tasks),
                mean_delay_seconds=float(tasks.task_delay.mean()),
                p95_delay_seconds=float(tasks.task_delay.quantile(.95)),
                mean_upload_seconds=float(tasks.task_upload_delay.mean()),
                mean_queue_seconds=float(tasks.task_queue_delay.mean()),
                mean_execution_seconds=float(tasks.task_execution_delay.mean()),
                rsr_correlated=float(tasks.requirement_satisfied_correlated.mean()),
                rsr_independent=float(tasks.requirement_satisfied_independent.mean()),
                fallback_rate=float(tasks.mask_fallback_used.mean()), source_directory=str(directory),
            ))
            record = dict(arrival_rate=rate, method=method, source_directory=str(directory),
                          task_results_sha256=digest(directory / "task_results.csv"),
                          input_sha256=hashes, matched_settings=common, paired_arrivals_verified=True)
            if method == "Ind-PPO":
                summary = json.loads((directory / "summary.json").read_text())
                assert summary["weights_and_optimizer_unchanged"]
                assert summary["source_checkpoint_sha256"] == digest(CHECKPOINT)
                runner.assert_identical(torch.load(CHECKPOINT, map_location="cpu", weights_only=True),
                                         torch.load(directory / "checkpoint_latest.pt", map_location="cpu", weights_only=True))
                record.update(checkpoint_sha256=summary["source_checkpoint_sha256"], frozen_verified=True)
            audits.append(record)
        for method in METHODS:
            if method == "Ind-PPO":
                continue
            delta = episode_delays["Ind-PPO"] - episode_delays[method]
            half = float(t.ppf(.975, len(delta) - 1) * delta.std() / np.sqrt(len(delta)))
            differences.append(dict(
                arrival_rate=rate, reference=method, paired_episodes=500,
                mean_ind_minus_reference_seconds=float(delta.mean()), ci95_lower_seconds=float(delta.mean() - half),
                ci95_upper_seconds=float(delta.mean() + half),
                ind_delay_reduction_percent=float(-delta.mean() / episode_delays[method].mean() * 100),
            ))
    frame, delta_frame = pd.DataFrame(summaries), pd.DataFrame(differences)
    frame.to_csv(OUT / "summary.csv", index=False)
    pd.concat(paired_rows, ignore_index=True).to_csv(OUT / "paired_episode_metrics.csv", index=False)
    delta_frame.to_csv(OUT / "paired_differences.csv", index=False)
    for metric, name in (("mean_delay_seconds", "mean_delay"), ("p95_delay_seconds", "p95_delay"),
                         ("rsr_correlated", "rsr_correlated"), ("mean_queue_seconds", "queue_delay")):
        frame.pivot(index="arrival_rate", columns="method", values=metric)[list(METHODS)].to_csv(OUT / f"{name}_comparison.csv")
    json_file("validation.json", dict(
        runs=audits, all_24_runs_verified=True, total_compared_task_rows=2400000,
        independent_ppo_checkpoint=str(CHECKPOINT), independent_ppo_checkpoint_sha256=digest(CHECKPOINT),
        same_ind_ppo_checkpoint_at_all_rates=True, evaluation_only=True,
        reliability_metric="Fraction whose computed reliability meets its task requirement, not sampled successes",
        interval_scope="Paired Student-t 95% intervals over 500 episodes; fixed profiles and one checkpoint per PPO variant",
    ))
    return frame, delta_frame


def report(frame, differences):
    lines = ["# Ind-PPO 到达率评估与四方法对比", "",
             "## 1. 实验口径", "",
             "复用同一个 Ind-PPO checkpoint，不重新训练。新增到达率 0.5、1、1.5、2.5、3；到达率 2 的已有冻结评估结果直接复用。每个到达率均为原评估集的 500 轮、每轮 200 个任务，评估种子 2027。", "",
             "与已有 Corr-PPO、Corr-delayGreedy、Indep-delayGreedy 的结果逐任务核对任务属性、到达时刻，核对输入文件哈希与风险配置。各 PPO 在所有负载下使用各自同一个 checkpoint；评估期间网络与优化器状态均不改变。RASOUL-Delay 此前使用不同可靠性阈值，未纳入本表。", "",
             "## 2. 平均任务完成时延（秒）", "",
             "| 到达率（任务/秒） | Corr-PPO | Ind-PPO | Corr-delayGreedy | Indep-delayGreedy |", "|---:|---:|---:|---:|---:|"]
    for rate in RATES:
        rows = frame[frame.arrival_rate.eq(rate)].set_index("method")
        lines.append(f"| {rate:g} | " + " | ".join(f"{rows.loc[method, 'mean_delay_seconds']:.6f}" for method in METHODS) + " |")
    lines += ["", "## 3. 相关模型 RSR", "",
              "| 到达率 | Corr-PPO | Ind-PPO | Corr-delayGreedy | Indep-delayGreedy |", "|---:|---:|---:|---:|---:|"]
    for rate in RATES:
        rows = frame[frame.arrival_rate.eq(rate)].set_index("method")
        lines.append(f"| {rate:g} | " + " | ".join(f"{rows.loc[method, 'rsr_correlated']:.2%}" for method in METHODS) + " |")
    lines += ["", "所有方法的独立模型 RSR 均为 93%。Corr-PPO、Corr-delayGreedy 的 Fallback 为 8.5%；Ind-PPO、Indep-delayGreedy 为 7%。相关可靠性动作掩码保证前两者达到当前固定任务集的相关 RSR 上限 91.5%。RSR 为计算可靠性满足任务要求的比例。", "",
              "## 4. Ind-PPO 相对各参照的时延变化", "",
              "下表正值表示 Ind-PPO 平均时延更低，负值表示更高。", "",
              "| 到达率 | 相对 Corr-PPO | 相对 Corr-delayGreedy | 相对 Indep-delayGreedy |", "|---:|---:|---:|---:|"]
    for rate in RATES:
        rows = differences[differences.arrival_rate.eq(rate)].set_index("reference")
        lines.append(f"| {rate:g} | " + " | ".join(f"{rows.loc[method, 'ind_delay_reduction_percent']:.2f}%" for method in METHODS if method != "Ind-PPO") + " |")
    lines += ["", "每轮配对平均时延差及其 95% 区间见 paired_differences.csv。区间只反映固定任务集和固定训练 checkpoint 下的到达流及动作采样变化，不包含训练种子间的不确定性。", "",
              "## 5. 文件与一致性", "",
              "- summary.csv：24 组结果的平均/P95 时延、分阶段时延、两套 RSR 和 Fallback。",
              "- mean_delay_comparison.csv、p95_delay_comparison.csv、rsr_correlated_comparison.csv：宽表。",
              "- paired_episode_metrics.csv、paired_differences.csv：每轮配对指标与差异。",
              "- validation.json：逐组输入与任务结果哈希、到达流匹配、冻结验证。",
              "- comparison.png、comparison.pdf：平均时延、P95 及相关 RSR 随到达率的变化。",
              "- evaluate.py：复用现有评估入口的独立运行与分析脚本。", "",
              "本次未修改算法、任务原始文件、已有 checkpoint 或历史评估结果，未 commit、push。", ""]
    with (OUT / "report.md").open("x", encoding="utf-8") as output:
        output.write("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    colors, markers = ("#2563eb", "#15803d", "#d97706", "#9333ea"), ("o", "^", "s", "D")
    for axis, metric, title, multiplier in zip(
            axes, ("mean_delay_seconds", "p95_delay_seconds", "rsr_correlated"),
            ("Mean task delay (s)", "P95 task delay (s)", "Requirement met: correlated model (%)"), (1, 1, 100)):
        for method, color, marker in zip(METHODS, colors, markers):
            data = frame[frame.method.eq(method)].sort_values("arrival_rate")
            axis.plot(data.arrival_rate, data[metric] * multiplier, color=color, marker=marker, label=method, linewidth=1.8)
        axis.set(title=title, xlabel="Arrival rate (tasks/s)", xticks=RATES)
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False)
    axes[2].set_ylim(0, 100)
    fig.suptitle("Original evaluation profiles | 500 × 200 tasks per rate | matched arrivals, seed 2027")
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)


def main():
    with (OUT / "started.json").open("x", encoding="utf-8") as output:
        protected = {str(path): digest(path) for path in
                     (CHECKPOINT, Path(EVALUATION_TASK_PATH), Path(SERVER_INFO_PATH), RATE2 / "task_results.csv", REFERENCE)}
        json.dump(dict(rates_new=[.5, 1, 1.5, 2.5, 3], rate2_reused=str(RATE2), episodes=500,
                       seed=2027, protected_sha256=protected), output, indent=2)
    torch.set_num_threads(1)
    logger = importlib.import_module("io_utils.long_run_logger")
    writer = importlib.import_module("io_utils.save_parameters_and_logs")
    with patch.object(runner, "RESULTS_DIR", OUT), patch.object(Project_main, "RESULTS_DIR", OUT), \
            patch.object(logger, "RESULTS_DIR", OUT), patch.object(writer, "RESULTS_DIR", OUT):
        directory = runner.main(["ppo", "--ppo-agent", "ind-ppo", "--checkpoint", str(CHECKPOINT),
                                 "--rates", ".5", "1", "1.5", "2.5", "3", "--episodes", "500",
                                 "--seed", "2027", "--device", "cpu", "--task-path", str(EVALUATION_TASK_PATH)])
    registry = {float(rate): Path(path) for rate, path in json.loads((directory / "ppo_runs.json").read_text()).items()}
    registry[2.0] = RATE2
    json_file("ind_ppo_runs_all_rates.json", {str(rate): str(path) for rate, path in sorted(registry.items())})
    frame, differences = compare(registry)
    report(frame, differences)
    assert all(digest(path) == value for path, value in protected.items())
    json_file("completed.json", dict(status="completed", newly_evaluated_tasks=500000,
                                     total_ind_ppo_evaluation_tasks=600000, protected_files_unchanged=True))
    print(frame[["arrival_rate", "method", "mean_delay_seconds", "rsr_correlated"]].to_string(index=False))
    print("Report:", OUT / "report.md", flush=True)


if __name__ == "__main__":
    main()
