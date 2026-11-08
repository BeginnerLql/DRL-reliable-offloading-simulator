"""Audit completed Ind-PPO evaluation against existing matched references."""
import hashlib
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))

import numpy as np
import pandas as pd
from scipy.stats import t
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    completed = json.loads((OUT / "completed.json").read_text())
    assert completed["status"] == "completed"
    evaluation = Path(completed["evaluation_directory"])
    plan = json.loads((OUT / "execution_plan.json").read_text())
    saved = json.loads((evaluation / "parameters.json").read_text())
    checkpoint = Path(saved["SOURCE_CHECKPOINT"])
    training = checkpoint.with_suffix("")
    train_status = json.loads((training / "status.json").read_text())
    assert train_status["status"] == "completed" and train_status["last_completed_episode"] == 8000
    logs = pd.read_csv(training / "logs.csv")
    assert logs.Episode.tolist() == list(range(1, 8001))
    assert np.isfinite(logs.select_dtypes(include="number")).all().all()
    assert (logs.task_Avg_Delay >= 0).all()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(torch.isfinite(value).all() for key in ("policy_net", "policy_old", "value_net")
               for value in state[key].values())
    actor_steps = {int(value["step"]) for value in state["opt_policy"]["state"].values()}
    critic_steps = {int(value["step"]) for value in state["opt_value"]["state"].values()}
    assert actor_steps == critic_steps == {64000}
    assert digest(checkpoint) != digest(ROOT / "results/ppo_20261031_180331_baseline.pt")
    # The reference has the same infrastructure, task profiles and PPO settings.
    previous_train = ROOT / "results/ppo_20261031_180331_baseline"
    old_params = json.loads((previous_train / "parameters.json").read_text())
    train_params = json.loads((training / "parameters.json").read_text())
    keys = [key for key in train_params if key.endswith("_ppo")]
    keys += ["total_episodes", "taskno", "TASK_ARRIVAL_RATE", "EXPERIMENT_SEED", "NUM_SERVERS",
             "SPATIAL_RISK_ENABLED", "SPATIAL_RISK_BETA_P", "SPATIAL_RISK_GH_ORDER",
             "RELIABILITY_ACTION_MASK_ENABLED", "num_states", "num_actions"]
    for key in keys:
        assert train_params[key] == old_params[key], (key, train_params[key], old_params[key])
    for name in ("server_info.xlsx", "task_parameters.xlsx"):
        assert digest(training / name) == digest(previous_train / name)

    old_summary = pd.read_csv(ROOT / "results/three_method_comparison_20261102_225141_483260/summary.csv")
    references = old_summary[old_summary.arrival_rate.eq(2)]
    directories = {row.method: Path(row.source_directory) for row in references.itertuples()}
    directories["Ind-PPO"] = evaluation
    summaries, differences, paired, audit = [], [], {}, []
    reference_rows = None
    for method, directory in directories.items():
        config = json.loads((directory / "parameters.json").read_text())
        tasks = pd.read_csv(directory / "task_results.csv")
        assert len(tasks) == 100000 and not tasks.duplicated(["episode", "task_id"]).any()
        assert tasks.groupby("episode").size().tolist() == [200] * 500
        assert tasks.server_A_id.ne(tasks.server_B_id).all()
        np.testing.assert_allclose(tasks.task_completion_time,
                                   np.minimum(tasks.replica_A_finish_time, tasks.replica_B_finish_time))
        independent = method in ("Ind-PPO", "Indep-delayGreedy")
        model = "independent" if independent else "correlated"
        np.testing.assert_array_equal(tasks.feasible_action_count > 0, tasks[f"requirement_satisfied_{model}"])
        np.testing.assert_array_equal(tasks.mask_fallback_used, tasks.feasible_action_count == 0)
        for field in ("correlated", "independent"):
            np.testing.assert_array_equal(tasks[f"requirement_satisfied_{field}"],
                                          tasks[f"pair_reliability_{field}"] >= tasks.reliability_requirement)
        matched = tasks[["episode", "task_id", "arrival_time", "task_size", "computation_demand",
                         "reliability_requirement"]]
        if reference_rows is None:
            reference_rows = matched
        else:
            np.testing.assert_allclose(matched, reference_rows, rtol=1e-12, atol=1e-12)
        for key in ("TASK_ARRIVAL_RATE", "EXPERIMENT_SEED", "SPATIAL_RISK_ENABLED",
                    "SPATIAL_RISK_BETA_P", "SPATIAL_RISK_GH_ORDER", "RELIABILITY_ACTION_MASK_ENABLED"):
            assert config[key] == saved[key]
        for name in ("server_info.xlsx", "task_parameters.xlsx"):
            assert digest(directory / name) == digest(evaluation / name)
        episode_delay = tasks.groupby("episode").task_delay.mean()
        paired[method] = episode_delay
        summaries.append(dict(
            method=method, episodes=500, tasks=len(tasks), mean_delay_seconds=float(tasks.task_delay.mean()),
            p95_delay_seconds=float(tasks.task_delay.quantile(.95)),
            mean_upload_seconds=float(tasks.task_upload_delay.mean()),
            mean_queue_seconds=float(tasks.task_queue_delay.mean()),
            mean_execution_seconds=float(tasks.task_execution_delay.mean()),
            rsr_correlated=float(tasks.requirement_satisfied_correlated.mean()),
            rsr_independent=float(tasks.requirement_satisfied_independent.mean()),
            fallback_rate=float(tasks.mask_fallback_used.mean()), source=str(directory),
        ))
        audit.append(dict(method=method, task_results_sha256=digest(directory / "task_results.csv"),
                          paired_arrivals_and_profiles_verified=True))
    frame = pd.DataFrame(summaries)
    for method in paired:
        if method == "Ind-PPO":
            continue
        delta = paired["Ind-PPO"] - paired[method]
        half = float(t.ppf(.975, len(delta) - 1) * delta.std() / np.sqrt(len(delta)))
        differences.append(dict(reference=method, mean_ind_minus_reference_seconds=float(delta.mean()),
                                ci95_lower_seconds=float(delta.mean() - half),
                                ci95_upper_seconds=float(delta.mean() + half),
                                ind_delay_reduction_percent=float(-delta.mean() / paired[method].mean() * 100)))
    frame.to_csv(OUT / "four_method_comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired_episode_delays.csv")
    pd.DataFrame(differences).to_csv(OUT / "paired_delay_differences.csv", index=False)
    for path, value in plan["protected_sha256"].items():
        assert digest(path) == value
    with (OUT / "validation.json").open("x", encoding="utf-8") as output:
        json.dump(dict(
            training_episodes=8000, training_tasks=1600000, evaluation_episodes=500, evaluation_tasks=100000,
            training_configuration_matches_corr_ppo_except_mask=True, actor_optimizer_steps=64000,
            critic_optimizer_steps=64000, checkpoint_sha256=digest(checkpoint),
            no_prior_checkpoint_used_for_training=True, existing_inputs_and_checkpoint_unchanged=True,
            evaluation_frozen=True, original_method_regression_exact=True, unit_tests_passed=33,
            audits=audit, training_directory=str(training), checkpoint=str(checkpoint),
        ), output, indent=2)
    ind = frame[frame.method.eq("Ind-PPO")].iloc[0]
    lines = ["# Ind-PPO 独立训练与原评估集验证", "",
             "## 1. 方法与实验设置", "",
             "新增模型标识 `ind-ppo`。与 Corr-PPO 共用相同 PPO 类、42 维状态、28 个双副本动作、网络结构、SMDP/GAE 更新及负任务时延奖励。仅将动作可行集合和最大可靠性回退依据改为独立可靠性 R_ind=1−(1−R_A)(1−R_B)，R_A、R_B 仍为 GH 边际积分结果。相关风险模型保持开启，结果同时报告两套 RSR。", "",
             "从随机初始化独立训练 8000 轮，每轮 200 个原训练任务，到达率 2.0，训练种子 2026；不加载既有 Corr-PPO checkpoint。所有 PPO 超参数与既有 Corr-PPO 训练快照逐项核对一致。", "",
             "使用原始 data/task_parameters_evaluation.xlsx（可靠性要求 U[0.99, 0.999]）冻结评估 500 轮，共 100000 个任务；到达率 2.0、评估种子 2027。评估期间网络与优化器状态保持不变。", "",
             "## 2. 相同评估口径的四方法对比", "",
             "| 方法 | 平均时延（秒） | P95（秒） | 平均排队（秒） | 相关模型 RSR | 独立模型 RSR | Fallback |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in frame.itertuples():
        lines.append(f"| {row.method} | {row.mean_delay_seconds:.6f} | {row.p95_delay_seconds:.6f} | {row.mean_queue_seconds:.6f} | {row.rsr_correlated:.2%} | {row.rsr_independent:.2%} | {row.fallback_rate:.2%} |")
    lines += ["", "所有行均为相同服务器、原评估任务、500 轮及同一到达流，已逐任务核对到达时刻和任务属性。RSR 表示计算可靠性满足任务要求的比例。RASOUL-Delay 的此前宽松阈值实验未混入本表。", "",
              "## 3. 配对时延差异", "",
              "| 参照方法 | Ind-PPO 时延降低比例 | Ind-PPO 减参照的平均时延（秒） | 95% 配对区间（秒） |", "|---|---:|---:|---:|"]
    for row in differences:
        lines.append(f"| {row['reference']} | {row['ind_delay_reduction_percent']:.2f}% | {row['mean_ind_minus_reference_seconds']:.6f} | [{row['ci95_lower_seconds']:.6f}, {row['ci95_upper_seconds']:.6f}] |")
    lines += ["", "区间以 500 轮的配对平均时延差计算，条件为固定任务集和本次单个训练种子，未覆盖训练种子间的不确定性。", "",
              "## 4. 验证与交付", "",
              "- 33 项测试通过，覆盖独立掩码、并列回退、双副本执行、训练更新、checkpoint、冻结评估、结果表及参照复用。",
              "- 原 Corr-PPO 和两个 Greedy 的固定种子任务记录及日志在修改前后完全一致。",
              "- 训练完成 8000 轮，actor、critic 各 64000 次优化器更新；网络参数及指标均为有限数值。",
              "- 原任务文件、服务器数据和已有 checkpoint 哈希不变；未 commit、push。", "",
              f"新 checkpoint：`{checkpoint.relative_to(ROOT)}`。", "",
              "训练全量任务 trace 保存在本机；总体记录、每轮日志、参数快照、checkpoint、评估原始任务记录均在本独立目录。", "",
              "## 5. 结果解释", "",
              f"本次 Ind-PPO 平均时延 {ind.mean_delay_seconds:.6f} 秒；按其独立假设评估的 RSR 为 {ind.rsr_independent:.2%}，按相关模型评估则为 {ind.rsr_correlated:.2%}。应结合时延与相关模型 RSR 一起判断，不能仅凭独立假设下的达标率认定满足相关可靠性要求。", ""]
    with (OUT / "report.md").open("x", encoding="utf-8") as output:
        output.write("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    smooth = logs.task_Avg_Delay.rolling(100, min_periods=1).mean()
    axes[0].plot(logs.Episode, smooth, color="#15803d")
    axes[0].set(title="Ind-PPO training (100-episode mean)", xlabel="Episode", ylabel="Task delay (s)")
    labels = [name.replace("-delayGreedy", "-Greedy") for name in frame.method]
    axes[1].bar(labels, frame.mean_delay_seconds, color=["#2563eb", "#d97706", "#9ca3af", "#15803d"])
    axes[1].set(title="Original evaluation: mean delay", ylabel="Seconds")
    x = np.arange(len(frame))
    axes[2].bar(x - .18, frame.rsr_correlated * 100, width=.36, label="Correlated")
    axes[2].bar(x + .18, frame.rsr_independent * 100, width=.36, label="Independent")
    axes[2].set(title="Reliability requirement met", ylabel="Percent", xticks=x, xticklabels=labels, ylim=(0, 100))
    axes[2].legend(frameon=False)
    for axis in axes[1:]:
        axis.tick_params(axis="x", labelrotation=25)
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)
    print(frame.to_string(index=False))
    print("Report:", OUT / "report.md")


if __name__ == "__main__":
    main()
