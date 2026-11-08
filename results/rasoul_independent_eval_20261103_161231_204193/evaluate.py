"""Independent evaluation and passive audit; repository algorithms are unchanged.

Run with the repository's Python environment. Outputs are exclusive to this
directory; a completed or started evaluation cannot be overwritten by rerunning.
"""

from collections import Counter, defaultdict
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))

import numpy as np
import pandas as pd
from openpyxl import load_workbook
import torch

from Project_main import build_model
from config.params import params
from config.paths import EVALUATION_TASK_PATH, SERVER_INFO_PATH
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.spatial_risk import GaussHermiteReliability
from core.task import Task
from io_utils.long_run_logger import LongRunLogger
from io_utils import long_run_logger as logger_module
from io_utils import save_parameters_and_logs as writer_module
from tools.run_experiments import temporary_params

EXPECTED_HEAD = "a6e5aafb534ddb98cc80a8be215b01b510bf981b"
PROFILE_SEED, ARRIVAL_SEED, EPISODES, ARRIVAL_RATE, EXPLORATION_C = 2029, 2027, 20, 2.0, .5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(name, value):
    with (OUT / name).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)


def save_csv(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    with (OUT / name).open("x", encoding="utf-8", newline="") as output:
        frame.to_csv(output, index=False)
    return frame


def prepare_input():
    inputs = OUT / "inputs"
    inputs.mkdir()
    original = pd.read_excel(EVALUATION_TASK_PATH)
    assert len(original) == 200 and original.Task_ID.is_unique
    generated = np.random.default_rng(PROFILE_SEED).uniform(.9, .99, len(original))
    path = inputs / "task_parameters_evaluation_rel090_099_seed2029.xlsx"
    shutil.copy2(EVALUATION_TASK_PATH, path)
    book = load_workbook(path)
    sheet = book.worksheets[0]
    column = [cell.value for cell in sheet[1]].index("Reliability_Requirement") + 1
    for row, requirement in enumerate(generated, start=2):
        sheet.cell(row, column).value = float(requirement)
    book.save(path)
    book.close()
    revised = pd.read_excel(path)
    pd.testing.assert_frame_equal(original.drop(columns="Reliability_Requirement"),
                                  revised.drop(columns="Reliability_Requirement"), check_exact=True)
    np.testing.assert_allclose(revised.Reliability_Requirement, generated, rtol=0, atol=1e-15)
    assert revised.Reliability_Requirement.between(.9, .99, inclusive="both").all()
    first, second = load_workbook(EVALUATION_TASK_PATH), load_workbook(path)
    assert first.sheetnames == second.sheetnames
    checked_cells = 0
    for sheet_a, sheet_b in zip(first.worksheets, second.worksheets):
        assert sheet_a.max_row == sheet_b.max_row and sheet_a.max_column == sheet_b.max_column
        for row in sheet_a:
            for cell in row:
                if sheet_a.title == first.worksheets[0].title and cell.column == column and cell.row > 1:
                    continue
                other = sheet_b.cell(cell.row, cell.column)
                assert (cell.value, cell.data_type, cell.number_format) == (other.value, other.data_type, other.number_format)
                checked_cells += 1
    first.close()
    second.close()
    save_csv("input_row_verification.csv", [dict(
        row=index + 2, task_id=int(original.iloc[index].Task_ID), other_fields_exactly_equal=True,
        original_requirement=float(original.iloc[index].Reliability_Requirement),
        new_requirement=float(revised.iloc[index].Reliability_Requirement),
    ) for index in range(len(original))])
    save_json("input_verification.json", dict(
        original=str(EVALUATION_TASK_PATH), original_sha256=digest(EVALUATION_TASK_PATH),
        generated=str(path), generated_sha256=digest(path), rows=len(original),
        unchanged_columns=[name for name in original if name != "Reliability_Requirement"],
        non_target_cells_checked=checked_cells, all_non_target_values_types_formats_equal=True,
        rng="numpy.random.default_rng / PCG64", seed=PROFILE_SEED, distribution="U[0.9, 0.99]",
    ))
    shutil.copy2(SERVER_INFO_PATH, inputs / "server_info.xlsx")
    return path, revised


def feasibility(tasks):
    servers = pd.read_excel(SERVER_INFO_PATH, sheet_name="Servers").sort_values("Server_ID")
    gh = GaussHermiteReliability(params.SPATIAL_RISK_BETA_P, params.SPATIAL_RISK_GH_ORDER)
    reliability, scans, detail = {}, [], []
    for task in tasks.itertuples():
        values = {}
        for server in servers.itertuples():
            exposure = float(server.Base_Failure_Rate * task.Computation_Demand / server.Processing_Frequency)
            marginal = gh.compute_marginal(exposure)
            assert marginal == gh.compute_pair(exposure, 0, 0).replica_a
            values[int(server.Server_ID)] = marginal
            detail.append(dict(task_id=int(task.Task_ID), server_id=int(server.Server_ID),
                               exposure=exposure, marginal_reliability=marginal,
                               requirement=float(task.Reliability_Requirement), safe=marginal >= task.Reliability_Requirement))
        reliability[int(task.Task_ID)] = values
        safe = [server for server, value in values.items() if value >= task.Reliability_Requirement]
        scans.append(dict(task_id=int(task.Task_ID), requirement=float(task.Reliability_Requirement),
                          safe_server_count=len(safe), safe_server_ids=json.dumps(safe),
                          **{f"R_{server}": value for server, value in values.items()}))
    frame = save_csv("feasibility_per_task.csv", scans)
    save_csv("marginal_reliability_per_server_task.csv", detail)
    save_csv("safe_count_distribution.csv", [dict(
        safe_server_count=count, tasks=int(frame.safe_server_count.eq(count).sum()),
        proportion=float(frame.safe_server_count.eq(count).mean()),
    ) for count in range(len(servers) + 1)])
    summary = {"K=0": int(frame.safe_server_count.eq(0).sum()),
               "K=1": int(frame.safe_server_count.eq(1).sum()),
               "K>=2": int(frame.safe_server_count.ge(2).sum())}
    save_json("feasibility_summary.json", summary)
    print("Feasibility:", summary, flush=True)
    return reliability, frame


class PassiveAudit:
    def __init__(self, loop, reliabilities):
        self.loop, self.model, self.reliabilities = loop, loop.model, reliabilities
        self.decisions, self.candidates, self.feedback, self.events = [], [], [], []
        self.tasks, self.episodes, self.server_episodes, self.resets = [], [], [], []
        self.observed, self.assigned = Counter(), Counter()
        self.execution_counts = defaultdict(Counter)
        self.feedback_keys = set()

    def event(self, kind, task_id=None, server_id=None):
        sequence = len(self.events) + 1
        self.events.append(dict(sequence=sequence, episode=self.loop.this_episode,
                                time=float(self.loop.env.now), event=kind, task_id=task_id, server_id=server_id))
        return sequence

    def install(self, stack):
        original_select = self.loop._select_rasoul_server
        original_feedback = self.loop._update_rasoul_feedback
        original_reset = self.model.reset

        def reset():
            original_reset()
            assert self.model.decision_count == 0
            assert not any(self.model.completed_counts.values()) and not any(self.model.discrepancy_sums.values())
            self.resets.append(dict(episode=self.loop.this_episode, all_statistics_zero=True))

        def select(task):
            episode, decision = self.loop.this_episode, self.model.decision_count + 1
            values = self.reliabilities[task.id]
            bases, scores, bonuses = {}, {}, {}
            safe = [server for server in self.loop.index_of_actions if values[server] >= task.reliability_requirement]
            counts, sums = dict(self.model.completed_counts), dict(self.model.discrepancy_sums)
            for server in self.loop.index_of_actions:
                profile = self.loop.server_profiles[server]
                base = task.task_size / profile["Transmission_Rate"] + task.computation_demand / profile["Processing_Frequency"]
                count = counts[server]
                assert count == self.observed[episode, server]
                mean = sums[server] / count if count else None
                bonus = EXPLORATION_C * math.sqrt(2 * math.log(max(1, decision)) / count) if count else None
                score = base + mean - bonus if count else -math.inf
                bases[server], scores[server], bonuses[server] = base, score, bonus
                self.candidates.append(dict(
                    episode=episode, task_id=task.id, decision=decision, server_id=server,
                    decision_time=float(task.env.now), safe=server in safe, marginal_reliability=values[server],
                    base_delay=base, completed_observations=count, assigned_before=self.assigned[episode, server],
                    pending_before=self.assigned[episode, server] - count, discrepancy_sum=sums[server],
                    mean_discrepancy=mean, exploration_bonus=bonus,
                    lcb_score=None if count == 0 else score, unobserved_priority=count == 0,
                ))
            if not safe:
                reason = "fallback"
                expected = min(values, key=lambda server: (-values[server], bases[server], server))
            else:
                reason = ("single_safe" if len(safe) == 1 else
                          "unobserved_exploration" if any(counts[server] == 0 for server in safe) else "lcb")
                expected = min(safe, key=lambda server: (scores[server], bases[server], server))
            selected = original_select(task).server_id
            assert selected == expected and self.model.decision_count == decision
            assert counts == self.model.completed_counts and sums == self.model.discrepancy_sums
            assert task.feasible_action_count == len(safe) and task.mask_fallback_used == (not safe)
            assert task.replica_A_reliability == values[selected]
            no_discrepancy, basic = None, None
            if reason == "lcb":
                no_discrepancy = min(safe, key=lambda server: (bases[server] - bonuses[server], bases[server], server))
                basic = min(safe, key=lambda server: (bases[server], server))
                assert task.predicted_delay_at_decision == scores[selected]
            sequence = self.event("decision", task.id, selected)
            self.decisions.append(dict(
                episode=episode, task_id=task.id, decision=decision, event_sequence=sequence,
                decision_time=float(task.env.now), safe_server_count=len(safe), branch=reason,
                selected_server=selected, selected_base=bases[selected],
                selected_mean_discrepancy=sums[selected] / counts[selected] if counts[selected] else None,
                selected_lcb=scores[selected] if counts[selected] else None,
                no_discrepancy_server=no_discrepancy, basic_delay_server=basic,
                discrepancy_changed_choice=reason == "lcb" and selected != no_discrepancy,
                differs_from_basic_delay=reason == "lcb" and selected != basic,
            ))
            self.assigned[episode, selected] += 1
            return self.loop.env_state.get_server_by_id(selected)

        def feedback(task):
            before = task.rasoul_feedback_processed
            counts = dict(self.model.completed_counts)
            original_feedback(task)
            if before or not task.rasoul_feedback_processed:
                assert counts == self.model.completed_counts
                return
            episode, server = self.loop.this_episode, task.replica_A["server_id"]
            key = (episode, task.id)
            assert key not in self.feedback_keys and task.task_completion_event.triggered
            assert task.task_completion_time == self.loop.env.now
            assert self.model.completed_counts[server] == counts[server] + 1
            assert all(self.model.completed_counts[other] == value for other, value in counts.items() if other != server)
            self.feedback_keys.add(key)
            self.observed[episode, server] += 1
            assert self.observed[episode, server] <= self.assigned[episode, server]
            sequence = self.event("feedback", task.id, server)
            self.feedback.append(dict(
                episode=episode, task_id=task.id, server_id=server, event_sequence=sequence,
                feedback_time=float(self.loop.env.now), completion_time=task.task_completion_time,
                actual_delay=task.task_completion_time - task.arrival_time,
                saved_base_delay=task.base_delay_at_decision,
                discrepancy=task.task_completion_time - task.arrival_time - task.base_delay_at_decision,
                completed_count_before=counts[server], completed_count_after=self.model.completed_counts[server],
                mean_discrepancy_after=self.model.discrepancy_sums[server] / self.model.completed_counts[server],
            ))

        stack.enter_context(patch.object(self.model, "reset", reset))
        stack.enter_context(patch.object(self.loop, "_select_rasoul_server", select))
        stack.enter_context(patch.object(self.loop, "_update_rasoul_feedback", feedback))
        for cls, name, stage in ((Task, "execute_task", "execute"), (Task, "_run_replica", "upload"),
                                  (EnvironmentState, "register_waiting_replica", "queue"),
                                  (EnvironmentState, "start_replica_execution", "cpu")):
            original = getattr(cls, name)

            def wrapper(instance, *args, _original=original, _stage=stage, **kwargs):
                task = instance if _stage in ("execute", "upload") else args[1]
                if _stage == "execute":
                    assert len(args) < 2 or args[1] is None
                if _stage == "upload":
                    assert args[1] == "replica_A"
                self.execution_counts[self.loop.this_episode, task.id][_stage] += 1
                return _original(instance, *args, **kwargs)

            stack.enter_context(patch.object(cls, name, wrapper))

    def finish_episode(self, loop):
        rows = list(loop.task_results)
        frame = pd.DataFrame(rows)
        episode = loop.this_episode
        assert len(rows) == 200 and len(frame.task_id.unique()) == 200
        assert not loop.pendingList and sum(self.model.completed_counts.values()) == 200
        assert self.model.decision_count == 200
        feedback = [item for item in self.feedback if item["episode"] == episode]
        assert len(feedback) == 200
        assert frame.server_B_id.isna().all() and frame.pair_reliability_correlated.isna().all()
        assert frame.pair_reliability_independent.isna().all() and frame.replica_count.eq(1).all()
        assert frame.task_delay.ge(0).all()
        np.testing.assert_allclose(frame.task_delay, frame.task_completion_time - frame.arrival_time)
        np.testing.assert_allclose(frame.task_delay, frame.task_upload_delay + frame.task_queue_delay + frame.task_execution_delay)
        np.testing.assert_allclose(frame.delay_discrepancy, frame.task_queue_delay, atol=1e-12, rtol=1e-12)
        for task in loop.env_state.tasks.values():
            assert task.rasoul_feedback_processed
            assert self.execution_counts[episode, task.id] == Counter(execute=1, upload=1, queue=1, cpu=1)
        for server in loop.index_of_actions:
            chosen = frame.server_A_id.eq(server)
            observed = [item for item in feedback if item["server_id"] == server]
            count = self.model.completed_counts[server]
            assert count == int(chosen.sum()) == len(observed)
            total = sum(item["discrepancy"] for item in observed)
            assert math.isclose(total, self.model.discrepancy_sums[server], rel_tol=1e-12, abs_tol=1e-12)
            info = loop.env_state.servers[server]
            assert not info["waiting_replicas"] and info["running_replica"] is None
            self.server_episodes.append(dict(
                episode=episode, server_id=server, selections=int(chosen.sum()), completed_feedback=count,
                discrepancy_sum=self.model.discrepancy_sums[server], mean_discrepancy=total / count if count else None,
                mean_queue_delay=float(frame.loc[chosen, "task_queue_delay"].mean()) if count else None,
            ))
        rsr, fallback = float(frame.requirement_satisfied.mean()), float(frame.mask_fallback_used.mean())
        assert math.isclose(rsr, .73) and math.isclose(fallback, .27)
        assert math.isclose(loop.log_data[-1][3], rsr) and math.isclose(loop.log_data[-1][5], fallback)
        self.episodes.append(dict(
            episode=episode, tasks=len(rows), feedback=len(feedback), mean_delay=float(frame.task_delay.mean()),
            p95_delay=float(frame.task_delay.quantile(.95)), mean_queue_delay=float(frame.task_queue_delay.mean()),
            theoretical_rsr=rsr, fallback_rate=fallback,
        ))
        self.tasks.extend(rows)


def summarize(audit, scan, checks):
    tasks = pd.DataFrame(audit.tasks)
    decisions = pd.DataFrame(audit.decisions)
    feedback = pd.DataFrame(audit.feedback)
    episodes = save_csv("episode_metrics.csv", audit.episodes)
    server_episodes = save_csv("episode_server_learning.csv", audit.server_episodes)
    save_csv("decisions.csv", decisions)
    save_csv("decision_candidates.csv", audit.candidates)
    save_csv("completion_feedback.csv", feedback)
    save_csv("event_order.csv", audit.events)
    save_csv("episode_resets.csv", audit.resets)
    save_csv("execution_counts.csv", [dict(episode=key[0], task_id=key[1], **counts)
                                       for key, counts in audit.execution_counts.items()])
    expected_arrivals = np.cumsum(np.random.default_rng(ARRIVAL_SEED).exponential(
        1 / ARRIVAL_RATE, (EPISODES, 200)), axis=1).ravel()
    np.testing.assert_array_equal(tasks.arrival_time, expected_arrivals)
    assert len(tasks) == len(decisions) == len(feedback) == 4000
    assert len(audit.resets) == EPISODES
    assert not feedback.duplicated(["episode", "task_id"]).any()
    np.testing.assert_array_equal(tasks.feasible_action_count, decisions.safe_server_count)
    np.testing.assert_array_equal(tasks.server_A_id, decisions.selected_server)
    np.testing.assert_array_equal(tasks.mask_fallback_used, decisions.branch.eq("fallback"))
    for episode in range(1, EPISODES + 1):
        current = tasks[tasks.episode.eq(episode)]
        np.testing.assert_array_equal(current.task_id, scan.task_id)
        np.testing.assert_array_equal(current.feasible_action_count, scan.safe_server_count)
    # Verify FIFO execution from actual queue-enter/start timestamps.
    for (_, _), group in tasks.groupby(["episode", "server_A_id"]):
        by_start = group.sort_values("replica_A_cpu_start_time")
        assert by_start.replica_A_queue_enter_time.is_monotonic_increasing
        assert by_start.replica_A_cpu_start_time.ge(by_start.replica_A_queue_enter_time).all()
    server_rows = []
    for server in audit.loop.index_of_actions:
        selected = tasks.server_A_id.eq(server)
        observed = feedback.server_id.eq(server)
        count = int(observed.sum())
        values = feedback.loc[observed, "discrepancy"]
        server_rows.append(dict(
            server_id=server, selections=int(selected.sum()), selection_share=float(selected.mean()),
            completed_feedback=count, discrepancy_sum=float(values.sum()),
            mean_discrepancy=float(values.mean()) if count else None,
            mean_task_delay=float(tasks.loc[selected, "task_delay"].mean()) if count else None,
            mean_queue_delay=float(tasks.loc[selected, "task_queue_delay"].mean()) if count else None,
        ))
    servers = save_csv("server_statistics.csv", server_rows)
    branches = decisions.branch.value_counts().to_dict()
    save_csv("decision_branch_counts.csv", [dict(branch=name, decisions=int(count), share=count / len(decisions))
                                             for name, count in branches.items()])
    lcb = decisions[decisions.branch.eq("lcb")]
    changed = lcb[lcb.discrepancy_changed_choice]
    save_csv("discrepancy_changes_choice.csv", changed)
    example_keys = set(zip(changed.head(10).episode, changed.head(10).task_id))
    save_csv("learning_examples_candidates.csv", [row for row in audit.candidates
                                                 if (row["episode"], row["task_id"]) in example_keys])
    summary = dict(
        episodes=EPISODES, tasks_per_episode=200, measured_task_executions=len(tasks),
        arrival_rate=ARRIVAL_RATE, exploration_c_seconds=EXPLORATION_C,
        requirement_range=[.9, .99], requirement_generation_seed=PROFILE_SEED, arrival_seed=ARRIVAL_SEED,
        mean_task_delay_seconds=float(tasks.task_delay.mean()), p95_task_delay_seconds=float(tasks.task_delay.quantile(.95)),
        mean_upload_delay_seconds=float(tasks.task_upload_delay.mean()),
        mean_queue_delay_seconds=float(tasks.task_queue_delay.mean()),
        mean_execution_delay_seconds=float(tasks.task_execution_delay.mean()),
        episode_mean_delay_std_seconds=float(episodes.mean_delay.std()),
        theoretical_rsr=float(tasks.requirement_satisfied.mean()), fallback_rate=float(tasks.mask_fallback_used.mean()),
        mean_single_reliability=float(tasks.task_reliability.mean()), completion_feedback_count=len(feedback),
        decision_branches={key: int(value) for key, value in branches.items()},
        lcb_decisions=len(lcb), discrepancy_changed_choice_decisions=len(changed),
        discrepancy_changed_choice_fraction_of_lcb=float(len(changed) / len(lcb)) if len(lcb) else None,
        lcb_differs_from_basic_delay_decisions=int(lcb.differs_from_basic_delay.sum()),
        negative_candidate_lcb_scores=sum(row["safe"] and row["lcb_score"] is not None and row["lcb_score"] < 0
                                          for row in audit.candidates),
        final_episode_learning_scope="Counters reset every episode; aggregate sums are reporting only",
    )
    checks.update(
        task_count_and_unique_ids=True, one_upload_queue_cpu_execution_per_task=True,
        no_second_replica=True, completed_feedback_once_per_task=True, feedback_at_completion_time=True,
        only_completed_observations_used=True, episode_resets_verified=EPISODES,
        all_decisions_match_formula_and_tie_rules=True, feasibility_matches_task_results=True,
        rsr_and_fallback_match_scan_every_episode=True, discrepancy_equals_observed_queue_delay=True,
        fifo_execution_verified=True, arrival_stream_matches_seed=True,
    )
    save_json("summary.json", summary)
    return summary, decisions, servers, episodes


def write_report(summary, scan, decisions, servers, episodes, checks):
    # Follow the supplied technical prompt's concise, numbered Chinese format.
    lines = [
        "# RASOUL-Delay 独立评估验证报告", "",
        "## 1. 评估配置", "",
        f"基于 GitHub main `{EXPECTED_HEAD}`。仅运行 RASOUL-Delay，不训练 PPO，不修改算法源码和正式训练配置。",
        "原评估集共 200 个任务，仅在副本中重新生成 Reliability_Requirement，采用 NumPy PCG64、种子 2029、连续均匀分布 U[0.9, 0.99]。其余单元格的值、类型及数字格式逐项核对一致，原文件 SHA256 保持不变。",
        "到达率 2.0 任务/秒，探索系数 c=0.5 秒，到达流种子 2027，评估 20 个 Episode，每轮复用相同 200 个任务及顺序，并清空 Bandit 历史统计。", "",
        "## 2. 可行性扫描", "", "| 安全服务器数 | 任务数 | 比例 |", "|---|---:|---:|",
    ]
    for name, condition in (("K=0", scan.safe_server_count.eq(0)), ("K=1", scan.safe_server_count.eq(1)),
                            ("K≥2", scan.safe_server_count.ge(2))):
        lines.append(f"| {name} | {int(condition.sum())} | {condition.mean():.1%} |")
    lines.extend(["", "K=0 的任务必定执行最大边际可靠性 Fallback；其余任务只能选择安全服务器。因此理论 RSR=73%、Fallback Rate=27%，每轮一致，不能把这两个固定比例的变化作为学习效果。", "",
                  "## 3. 运行结果", "", "| 指标 | 结果 |", "|---|---:|",
                  f"| 任务执行次数 | {summary['measured_task_executions']} |",
                  f"| 平均任务完成时延 | {summary['mean_task_delay_seconds']:.6f} 秒 |",
                  f"| P95 任务完成时延 | {summary['p95_task_delay_seconds']:.6f} 秒 |",
                  f"| 平均上传时延 | {summary['mean_upload_delay_seconds']:.6f} 秒 |",
                  f"| 平均排队时延 | {summary['mean_queue_delay_seconds']:.6f} 秒 |",
                  f"| 平均执行时延 | {summary['mean_execution_delay_seconds']:.6f} 秒 |",
                  f"| 理论 RSR | {summary['theoretical_rsr']:.2%} |",
                  f"| Fallback Rate | {summary['fallback_rate']:.2%} |",
                  f"| 完成反馈数量 | {summary['completion_feedback_count']} |", "",
                  "理论 RSR 指计算出的单副本边际可靠性满足任务要求的比例，不是随机失败仿真中的成功率。P95 按全部 4000 条任务时延计算；各轮指标另存于 episode_metrics.csv。", "",
                  "## 4. 服务器选择及历史 discrepancy", "",
                  "| Server ID | 选择次数 | 完成反馈 | 平均 discrepancy（秒） |", "|---:|---:|---:|---:|"])
    for row in servers.itertuples():
        mean = f"{row.mean_discrepancy:.6f}" if row.completed_feedback else "无反馈"
        lines.append(f"| {row.server_id} | {row.selections} | {row.completed_feedback} | {mean} |")
    lines.extend(["", "上表按 20 轮聚合观测，仅用于汇总。策略本身每轮重置；每轮、每台服务器的 n_m、discrepancy 累计和及均值见 episode_server_learning.csv。", "",
                  "## 5. 在线学习是否实际参与选择", "",
                  "| 决策分支 | 次数 |", "|---|---:|"])
    labels = {"fallback": "K=0：可靠性 Fallback", "single_safe": "K=1：单一安全候选",
              "unobserved_exploration": "K≥2：存在未观测候选，优先探索", "lcb": "K≥2：全部安全候选已有反馈，比较有限 LCB 分数"}
    for branch in ("fallback", "single_safe", "unobserved_exploration", "lcb"):
        lines.append(f"| {labels[branch]} | {summary['decision_branches'].get(branch, 0)} |")
    lines.extend(["",
        "LCB 采用 T_base + 平均 discrepancy − c√(2 ln N / n_m)。审计在调用原选择函数之前读取已完成统计，不读取真实 CPU 积压，不将未完成任务当成观测，也不改变选择函数返回值、事件调度或随机流。",
        f"在 {summary['lcb_decisions']} 次有限 LCB 选择中，有 {summary['discrepancy_changed_choice_decisions']} 次（{summary['discrepancy_changed_choice_fraction_of_lcb']:.2%}）在保持当前候选集合、N、n_m 和探索项不变、仅把平均 discrepancy 置零后，最优服务器发生变化。该结果直接说明历史 discrepancy 在这些决策中改变了排序。",
        f"另外，{summary['lcb_differs_from_basic_delay_decisions']} 次有限 LCB 决策与仅按基础时延排序的结果不同；这一差异同时包含 discrepancy 与探索项的作用。",
        "这些反事实只重新计算同一决策状态下的排序，不是另一次策略执行，也不代表全程时延收益。示例见 learning_examples_candidates.csv，全部受影响决策见 discrepancy_changes_choice.csv。", "",
        "## 6. 一致性检查", "",
        "- 每轮 200 次决策和 200 次完成反馈，共 4000 个任务，无重复任务或重复反馈。",
        "- 每个任务恰好一次上传、一次 CPU 入队和一次 CPU 执行；所有第二副本字段为空。",
        "- 逐决策检查安全集合、Fallback 最大可靠性规则、LCB 公式及平局处理。",
        "- 所有反馈均在实际完成时刻发生；每次决策的 n_m 与此前已处理反馈数一致。",
        "- 每轮 n_m 与 discrepancy 累计和从零开始；结束时所有任务均完成，CPU 等待队列为空。",
        "- 逐任务确认 discrepancy=实际完成时延−决策时保存的基础时延，并与实际排队时延一致。",
        "- 使用原算法、相同输入和种子进行无审计重放，20 轮全部任务记录及 episode 日志与带审计运行完全一致，证明观察包装未改变运行结果。",
        "- 原任务表、服务器数据、算法源码、正式配置及已有 checkpoint 的哈希不变；没有 commit 或 push。", "",
        "## 7. 结论与范围", "",
        "较宽松可靠性阈值产生了多个安全候选，正常 LCB 分支实际运行，历史完成反馈也确实影响后续选择。单副本执行、可靠性筛选、异步反馈及每轮重置机制的一致性检查通过。",
        "本轮仅验证在线学习机制有效参与决策。20 轮均使用同一固定任务集，仅到达间隔变化，不能据此认定 c=0.5 最优，也不能直接与旧可靠性分布下的 PPO 或贪心结果比较。", "",
        "## 8. 结果文件", "",
        "- inputs/task_parameters_evaluation_rel090_099_seed2029.xlsx：新评估任务文件。",
        "- input_row_verification.csv、input_verification.json：逐行及非目标单元格核对。",
        "- feasibility_per_task.csv、safe_count_distribution.csv：每任务安全服务器数量及分布。",
        "- runs/：原仿真日志、任务结果 CSV、配置快照及 Excel。",
        "- summary.json、episode_metrics.csv、server_statistics.csv：总体、每轮、每服务器结果。",
        "- decisions.csv、decision_candidates.csv：全部决策及各候选的历史统计和 LCB 分数。",
        "- completion_feedback.csv、episode_server_learning.csv、episode_resets.csv：反馈、学习统计与重置。",
        "- validation.json、protected_files_manifest.json：一致性与源码、数据完整性检查。",
        "- overview.png、overview.pdf：可行性、时延、服务器选择与决策分支概览。",
        "- evaluate.py：独立评估与审计脚本；不修改仓库算法文件。", "",
    ])
    with (OUT / "report.md").open("x", encoding="utf-8") as output:
        output.write("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    counts = scan.safe_server_count.value_counts().reindex(range(9), fill_value=0)
    axes[0, 0].bar(counts.index, counts.values, color="#2563eb")
    axes[0, 0].set(title="Feasible servers per task", xlabel="K", ylabel="Tasks", xticks=range(9))
    axes[0, 1].bar(servers.server_id, servers.selections, color="#15803d")
    axes[0, 1].set(title="Server selections across 20 episodes", xlabel="Server ID", ylabel="Selections", xticks=servers.server_id)
    axes[1, 0].plot(episodes.episode, episodes.mean_delay, label="Mean task delay", marker="o", markersize=3)
    axes[1, 0].plot(episodes.episode, episodes.mean_queue_delay, label="Mean queue delay", marker="s", markersize=3)
    axes[1, 0].set(title="Per-episode latency", xlabel="Episode", ylabel="Seconds")
    axes[1, 0].legend(frameon=False)
    branches = ["fallback", "single_safe", "unobserved_exploration", "lcb"]
    axes[1, 1].bar(["Fallback", "Single safe", "Unobserved", "LCB"],
                   [summary["decision_branches"].get(branch, 0) for branch in branches], color="#d97706")
    axes[1, 1].set(title="Decision branches", ylabel="Decisions")
    figure.suptitle("RASOUL-Delay | U[0.90, 0.99] | arrival=2/s | c=0.5 s")
    figure.savefig(OUT / "overview.png", dpi=180)
    figure.savefig(OUT / "overview.pdf")
    plt.close(figure)


def main():
    with (OUT / "started.json").open("x", encoding="utf-8") as output:
        json.dump(dict(status="started", expected_head=EXPECTED_HEAD), output)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == EXPECTED_HEAD
    assert not subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    protected = [name for name in tracked if
                 (name.endswith(".py") and not name.startswith("results/"))
                 or name.startswith("data/") or name.endswith(".pt")]
    protected_before = {name: digest(ROOT / name) for name in protected}
    save_json("protected_files_manifest.json", dict(head=head, before_sha256=protected_before))
    original_params = {key: value for key, value in vars(params).items() if not key.startswith("_")}
    task_path, tasks = prepare_input()
    reliabilities, scan = feasibility(tasks)
    (OUT / "runs").mkdir()
    torch.set_num_threads(1)
    np.random.seed(ARRIVAL_SEED)
    torch.manual_seed(ARRIVAL_SEED)
    checks = dict(github_main=head, original_task_rows_preserved=True)
    with temporary_params(model_summary="rasoul-delay", total_episodes=EPISODES, taskno=len(tasks),
                          TASK_ARRIVAL_RATE=ARRIVAL_RATE, EXPERIMENT_SEED=ARRIVAL_SEED,
                          RASOUL_DELAY_EXPLORATION_C=EXPLORATION_C,
                          EXPERIMENT_TAG="rel090_099_seed2029_arrival2_eval20"):
        model = build_model(device="cpu")
        settings = {key: value for key, value in vars(params).items() if not key.startswith("_")}
        settings.update(SOURCE_TRAINING_RELIABILITY_RANGE=settings["TASK_RELIABILITY_REQUIREMENT_RANGE"],
                        TASK_RELIABILITY_REQUIREMENT_RANGE=(.9, .99), EVALUATION_PROFILE_SEED=PROFILE_SEED,
                        EVALUATION_ONLY=True, REPLICA_COUNT=1, RELIABILITY_SEMANTICS="GH single-server marginal",
                        ACTION_SELECTION="single-replica delay LCB with completed feedback", SOURCE_HEAD=head)
        recorded = SimpleNamespace(**settings)
        with patch.object(logger_module, "RESULTS_DIR", OUT / "runs"), \
                patch.object(writer_module, "RESULTS_DIR", OUT / "runs"):
            with LongRunLogger(recorded, model, "independent20") as logger:
                (logger.run_dir / "configuration.py").rename(logger.run_dir / "source_training_configuration.py")
                (logger.run_dir / "task_parameters.xlsx").rename(logger.run_dir / "source_training_task_parameters.xlsx")
                shutil.copy2(task_path, logger.run_dir / "task_parameters.xlsx")
                loop = MainLoop(model, EPISODES, len(tasks), params.num_states, params.num_actions,
                                task_parameters_path=task_path)
                audit = PassiveAudit(loop, reliabilities)

                def finish(current):
                    audit.finish_episode(current)
                    logger(current)

                loop.episode_end_callback = finish
                with ExitStack() as stack:
                    audit.install(stack)
                    loop.EP()
            writer_module.save_params_and_logs(recorded, loop.log_data, loop.task_results, "independent20",
                                              task_parameters_path=task_path,
                                              task_results_csv_path=logger.task_results_csv_path)
        # Same 20 episodes replayed only to check passive observation, not new samples.
        with redirect_stdout(io.StringIO()):
            replay = MainLoop(build_model(device="cpu"), EPISODES, len(tasks), params.num_states, params.num_actions,
                              task_parameters_path=task_path)
            replay.EP()
        assert replay.task_results == audit.tasks and replay.log_data == loop.log_data
        checks.update(observer_transparency_exact_replay=True, replay_episodes=EPISODES,
                      replay_task_executions=len(replay.task_results), run_directory=str(logger.run_dir))
        summary, decisions, servers, episodes = summarize(audit, scan, checks)
    assert original_params == {key: value for key, value in vars(params).items() if not key.startswith("_")}
    protected_after = {name: digest(ROOT / name) for name in protected}
    assert protected_before == protected_after
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == head
    assert not subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).strip()
    checks.update(all_protected_file_hashes_unchanged=True, runtime_params_restored=True,
                  git_head_unchanged=True, tracked_worktree_unchanged=True, commit_or_push_performed=False)
    save_json("validation.json", checks)
    write_report(summary, scan, decisions, servers, episodes, checks)
    save_json("completed.json", dict(status="completed", measured_tasks=4000, all_checks_passed=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Report:", OUT / "report.md", flush=True)


if __name__ == "__main__":
    main()
