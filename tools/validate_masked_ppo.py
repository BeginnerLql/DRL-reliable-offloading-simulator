"""Small real PPO/SimPy checks for GH safety masks and deterministic fallback."""

import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import core.main_loop as loop_module
from agents.ppo_agent import masked_categorical
from compare_ppo_greedy import build_agent
from config.params import params
from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH
from core.env_state import EnvironmentState, select_latency_greedy_action
from core.main_loop import MainLoop
from io_utils.save_parameters_and_logs import summarize_task_results


def check_ties_and_masks():
    pairs = MainLoop.generate_combinations()
    state = np.zeros(params.num_states)
    assert select_latency_greedy_action(state, pairs) == 0
    state[3*params.NUM_SERVERS:4*params.NUM_SERVERS] = 4
    state[3*params.NUM_SERVERS + 1:3*params.NUM_SERVERS + 3] = 1
    assert select_latency_greedy_action(state, pairs) == pairs.index((2, 3))
    mask = np.zeros(params.num_actions, dtype=bool)
    assert select_latency_greedy_action(state, pairs, mask) == pairs.index((2, 3))
    mask[-1] = True
    assert select_latency_greedy_action(state, pairs, mask) == len(pairs)-1
    dist = masked_categorical(torch.randn(3, params.num_actions), mask)
    assert torch.all(dist.sample() == len(pairs)-1)
    assert torch.all(torch.isfinite(dist.entropy()))
    try:
        masked_categorical(torch.zeros(params.num_actions), np.zeros_like(mask))
    except ValueError:
        pass
    else:
        raise AssertionError("Empty masks must be handled before calling PPO")


def run_case(name, masked, scenario, output):
    torch.manual_seed(params.EXPERIMENT_SEED)
    np.random.seed(params.EXPERIMENT_SEED)
    agent = build_agent()
    loop = MainLoop(agent, 1, 200, params.num_states, params.num_actions,
                    use_safety_mask=masked, failure_rate_scale=0.05)
    for tid, profile in loop.task_profiles.items():
        if scenario in ("0.999", "0.9999"):
            threshold = float(scenario)
        else:
            normal = (tid > 2 and tid % 2 == 1) if scenario == "interleaved" else (tid == 1 and scenario == "single_ppo")
            threshold = 0.999 if normal else 0.99999
        profile["Reliability_Requirement"] = threshold
    if scenario in ("interleaved", "single_ppo"):
        loop._sample_interarrival_time = lambda: 0.1
    observations, traces = {}, []
    fallback_with_later_ppo = set()
    selected, updates, discounts = [], [], []
    original_state = EnvironmentState.get_state
    original_fallback = loop_module.select_latency_greedy_action
    original_select, original_train = agent.select_action, agent.train_step
    initial = copy.deepcopy({"actor": agent.policy_net.state_dict(), "critic": agent.value_net.state_dict(),
                             "actor_optimizer": agent.optimizer_policy.state_dict(),
                             "critic_optimizer": agent.optimizer_value.state_dict()})

    def observe(state, task):
        value = original_state(state, task)
        np.testing.assert_allclose(value[:params.NUM_SERVERS], [
            state.get_server_by_id(sid).base_failure_rate * 0.05 / params.STATE_FAILURE_RATE_SCALE
            for sid in sorted(state.servers)
        ], rtol=1e-7)
        observations[task.id] = value.copy()
        active_fallback = [
            item["task"].id for info in state.servers.values()
            for item in ([info["running_replica"]] if info["running_replica"] is not None else []) + info["waiting_replicas"]
            if item["task"].is_fallback
        ]
        if active_fallback and state.get_safe_action_mask(task).any():
            fallback_with_later_ppo.update(active_fallback)
        return value

    def fallback(observation, pairs, action_mask=None):
        task = loop.env_state.tasks[loop.taskCounter]
        assert not task.safe_mask.any()
        # Independent oracle uses current server metadata, before dispatch.
        predicted = {}
        for sid in loop.env_state.servers:
            server = loop.env_state.get_server_by_id(sid)
            running, waiting = loop.env_state.get_server_backlog_components(sid, loop.env.now)
            predicted[sid] = max(task.task_size/server.transmission_rate, running+waiting) + task.computation_demand/server.processing_frequency
        expected = min(range(len(pairs)), key=lambda i: (
            min(predicted[sid] for sid in pairs[i]), sum(predicted[sid] for sid in pairs[i]), i))
        action = original_fallback(observation, pairs, action_mask)
        assert action == expected
        return action

    def select(observation, epsilon, action_mask=None):
        task = loop.env_state.tasks[loop.taskCounter]
        assert task.safe_mask.any() and not task.is_fallback
        np.testing.assert_array_equal(action_mask, task.safe_mask if masked else np.ones(params.num_actions, dtype=bool))
        action = original_select(observation, epsilon, action_mask)
        assert not masked or task.safe_mask[action]
        selected.append(task.id)
        return action

    def train():
        ids = loop.ppo_decision_task_ids
        assert ids == agent.task_ids == selected
        assert ids == [tid for tid, task in loop.env_state.tasks.items() if not task.is_fallback]
        assert len(ids) > 0 and not updates
        updates.append(len(ids))
        for index, tid in enumerate(ids):
            task = loop.env_state.tasks[tid]
            next_tid = ids[index+1] if index+1 < len(ids) else None
            end_time = (loop.env_state.tasks[next_tid].arrival_time if next_tid is not None else
                        max(t.task_completion_time for t in loop.env_state.tasks.values()))
            dt = end_time - task.arrival_time
            assert agent.actions[index] == task.action_index
            expected_utility = sum(
                np.interp(item.task_completion_time-item.arrival_time,
                          [item.deadline, params.QOS_GRACE_MULTIPLIER*item.deadline], [1., 0.])
                if item.pair_reliability >= item.reliability_requirement else 0.
                for item in loop.env_state.tasks.values() if item.ppo_reward_owner_task_id == tid)
            np.testing.assert_allclose(agent.rewards[index], expected_utility, rtol=0, atol=1e-12)
            assert agent.dones[index] == (next_tid is None)
            np.testing.assert_allclose(agent.delta_times[index], dt, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(agent.states[index], observations[tid])
            np.testing.assert_array_equal(agent.next_states[index], observations[next_tid] if next_tid else np.zeros(params.num_states))
            expected_mask = task.safe_mask if masked else np.ones(params.num_actions, dtype=bool)
            expected_next = ((loop.env_state.tasks[next_tid].safe_mask if masked else np.ones(params.num_actions, dtype=bool))
                             if next_tid else np.zeros(params.num_actions, dtype=bool))
            np.testing.assert_array_equal(agent.action_masks[index], expected_mask)
            np.testing.assert_array_equal(agent.next_action_masks[index], expected_next)
            with torch.no_grad():
                logits = agent.policy_old(agent._to_tensor(agent.states[index]))
                expected_log = masked_categorical(logits, expected_mask).log_prob(torch.tensor(task.action_index)).item()
            np.testing.assert_allclose(agent.old_log_probs[index], expected_log, rtol=0, atol=1e-6)
            traces.append({"case": name, "task_id": tid, "next_task_id": next_tid,
                           "action": task.action_index, "reward": agent.rewards[index], "delta_t": dt,
                           "discount": agent.gamma**dt, "done": next_tid is None,
                           "mask": "".join(map(lambda v: str(int(v)), expected_mask)),
                           "next_mask": "".join(map(lambda v: str(int(v)), expected_next)),
                           "old_log_prob": agent.old_log_probs[index]})
        np.testing.assert_allclose(sum(agent.rewards), loop.episodic_reward, rtol=0, atol=1e-9)
        original_pow = torch.pow

        def power(base, exponent, *args, **kwargs):
            result = original_pow(base, exponent, *args, **kwargs)
            if isinstance(base, torch.Tensor) and base.shape == (len(ids),) and torch.all(base == agent.gamma):
                np.testing.assert_allclose(exponent.cpu().numpy(), agent.delta_times, rtol=1e-6)
                np.testing.assert_allclose(result.cpu().numpy(), np.power(agent.gamma, agent.delta_times), rtol=2e-5, atol=1e-8)
                discounts.append(True)
            return result

        torch.pow = power
        try:
            original_train()
        finally:
            torch.pow = original_pow
        assert discounts and not agent.states and not agent.action_masks and not agent.task_ids

    EnvironmentState.get_state = observe
    loop_module.select_latency_greedy_action = fallback
    agent.select_action, agent.train_step = select, train
    try:
        loop.EP()
    finally:
        EnvironmentState.get_state = original_state
        loop_module.select_latency_greedy_action = original_fallback
    tasks = list(loop.env_state.tasks.values())
    assert len(tasks) == len(loop.task_results) == 200
    fallback_count = sum(task.is_fallback for task in tasks)
    assert fallback_count + len(selected) == 200
    assert bool(updates) == bool(selected)
    for task in tasks:
        assert task.is_fallback == (not task.safe_mask.any())
        assert (task.ppo_transition_index is None) == task.is_fallback
        if task.is_fallback:
            assert task.pair_reliability < task.reliability_requirement
        for replica in (task.replica_A, task.replica_B):
            server = loop.env_state.get_server_by_id(replica["server_id"])
            np.testing.assert_allclose(replica["queue_enter_time"], task.arrival_time+task.task_size/server.transmission_rate)
            assert replica["cpu_start_time"] >= replica["queue_enter_time"]
            np.testing.assert_allclose(replica["finish_time"]-replica["cpu_start_time"], task.computation_demand/server.processing_frequency)
        assert task.task_completion_time == min(task.replica_A["finish_time"], task.replica_B["finish_time"])
    episode, counts, loads = summarize_task_results(loop.task_results)
    direct_rsr = np.mean([task.pair_reliability >= task.reliability_requirement for task in tasks])
    assert episode.theoretical_rsr.iloc[0] == direct_rsr
    assert counts.task_count.sum() == 200 and loads.replica_count.sum() == 400
    assert (loads.cpu_utilization <= 1+1e-12).all()
    if masked:
        assert direct_rsr == (200-fallback_count)/200
    elif scenario == "0.9999":
        assert any(not task.safe_mask[task.action_index] for task in tasks if not task.is_fallback)
    if scenario == "all_fallback":
        assert fallback_count == 200 and not agent.optimizer_policy.state and not agent.optimizer_value.state
        assert all(torch.equal(value, initial["actor"][key]) for key, value in agent.policy_net.state_dict().items())
        assert all(torch.equal(value, initial["critic"][key]) for key, value in agent.value_net.state_dict().items())
    if scenario == "single_ppo":
        assert selected == [1] and updates == [1] and agent.optimizer_value.state
    if scenario == "interleaved":
        assert fallback_with_later_ppo and selected == list(range(3, 200, 2))
        assert any(trace["next_task_id"] and trace["delta_t"] > 0.19 for trace in traces)
        assert [task.id for task in sorted(tasks, key=lambda t: t.task_completion_time)] != list(range(1, 201))
    assert all(torch.isfinite(parameter).all() for parameter in agent.policy_net.parameters())
    case = output / name
    case.mkdir()
    transitions = pd.DataFrame(traces, columns=(
        "case", "task_id", "next_task_id", "action", "reward", "delta_t", "discount",
        "done", "mask", "next_mask", "old_log_prob",
    ))
    for filename, table in (("tasks", pd.DataFrame(loop.task_results)), ("episode_metrics", episode),
                            ("safe_pair_counts", counts), ("server_load", loads), ("transitions", transitions)):
        table.to_csv(case / f"{filename}.csv", index=False, float_format="%.17g")
    return {"case": name, **episode.iloc[0].to_dict(), "update_count": len(updates),
            "verified_fallbacks_affecting_later_ppo": len(fallback_with_later_ppo)}


def write_report(output, summary, hashes):
    checks = [
        "非空安全集内Masked PPO只选安全动作；No-Mask使用全部28动作",
        "空集Fallback与当前可观测状态的28对穷举结果一致，含两级并列规则",
        "所有200任务和400副本上传、排队、CPU执行及完成时间一致",
        "Fallback无PPO样本，且其副本出现在后续PPO决策的真实队列中",
        "完成任务有效QoS归属到被观察到的PPO决策区间，区间奖励总和等于全任务有效QoS总和",
        "Fallback数量加正常PPO决策数量等于200，副本分配量之和等于400",
        "理论RSR等于全部实际任务的逐任务相关可靠性判定均值",
        "全Fallback正常结束，不调用采样或更新，网络及优化器不变",
        "交错任务的下一状态、下一Mask、实际gamma**delta_t与终止排空间隔正确",
    ]
    pd.DataFrame({"check": checks, "passed": True}).to_csv(output / "checks.csv", index=False)
    (output / "input_manifest.json").write_text(json.dumps({str(path): digest for path, digest in hashes.items()}, indent=2) + "\n")
    lines = [
        "# 第二阶段：GH Mask与最小时延Fallback验证", "",
        "实际执行10个单Episode场景，每个200任务、400副本；使用真实PPO优化和SimPy流程。",
        f"两种PPO复用同一类，状态{params.num_states}维、动作{params.num_actions}维。所有场景运行时故障率缩放均为0.05，beta=0.5，ell0沿用现有坐标。",
        "前四个主要场景统一阈值0.999或0.9999；全Fallback使用0.99999；交错和单样本场景仅在内存中组合这两个阈值。原始文件不修改。",
        "默认到达率沿用0.5；交错及单样本场景固定0.1秒间隔以覆盖队列影响和终止尾部。", "",
        "## 实测结果", "",
        "| 场景 | 任务 | Fallback | PPO决策 | 更新次数 | 平均时延/s | 理论RSR | 双副本平均CPU等待/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.case} | {int(row.task_count)} | {int(row.fallback_count)} | {int(row.ppo_decision_count)} "
                     f"| {row.update_count} | {row.mean_task_latency_s:.6f} | {row.theoretical_rsr:.1%} | {row.mean_cpu_wait_s:.6f} |")
    lines += ["", "## 检查结果", ""] + [f"{index}. 通过：{check}。" for index, check in enumerate(checks, 1)]
    lines += [
        "", "另检查单个有效样本也执行一次PPO更新，未沿用旧有少于8条样本直接丢弃的门槛；学习率、网络、epoch、batch size、GAE和时间折扣公式保持原值。",
        "PPO Transition的下一状态对应下一次有效PPO决策；末条使用原有任务首副本完成的终止排空时刻，包含尾部Fallback。第二个副本继续执行到CPU完成，并计入负载。",
        "Fallback通过队列和后续状态影响PPO，不产生PPO动作或log_prob；可靠性不达标的任务有效QoS为0，原始时延QoS单独保留。",
        "Masked PPO理论RSR=1-Fallback比例；No-Mask的非Fallback任务也可能不满足阈值。两种方法在相同任务和参数下Fallback任务集合一致，但队列不同可导致Fallback动作不同。",
        "这些单轮结果用于实现验证，不能作为训练收敛或方法性能结论。未开展正式训练、Monte Carlo、commit或push。", "",
        "## 输出与复现", "",
        "`/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python tools/validate_masked_ppo.py`",
        "复跑需用`--output-dir`指定新目录，已有结果拒绝覆盖。",
        "summary.csv和checks.csv为汇总；各场景tasks.csv、transitions.csv记录任务及PPO关联，含当前/下一Mask、真实时间间隔及折扣。",
        "各场景另存episode_metrics.csv、safe_pair_counts.csv、server_load.csv；CPU等待对全部400副本取平均，执行负载含cycles、CPU忙时和排空前利用率。",
        "正式实验的默认运行时故障率缩放仍为1.0；0.05仅用于本次显式验证，未自动改写正式故障率或任务阈值。",
        "历史扫描中的空安全集比例在新机制下对应Fallback比例，原有历史文件保持原样；新日志不记录拒绝率。",
    ]
    if (output / "integration_checks.csv").exists():
        lines += ["", "额外集成检查见integration_checks.csv：旧对比入口仅1训练轮及1轮冻结评估、Excel写入回读、Safe Random/Safe Min-Latency共用Fallback、网络结构及历史文件哈希核对。",
                  "heuristic_smoke.csv记录两个安全启发式的200任务验证；comparison_smoke/保留旧对比入口的小规模产物。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/spatial_reliability_scan/phase2_fallback_validation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = (Path(SERVER_INFO_PATH), Path(TRAIN_TASK_PATH), PROJECT_ROOT / "core/spatial_risk.py")
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
    check_ties_and_masks()
    rows = []
    for masked in (True, False):
        for scenario in ("0.999", "0.9999", "all_fallback", "interleaved", "single_ppo"):
            name = ("masked_" if masked else "no_mask_") + scenario
            rows.append(run_case(name, masked, scenario, args.output_dir))
            print(f"PASS {name}", flush=True)
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False, float_format="%.17g")
    write_report(args.output_dir, summary, hashes)
    print(f"All checks passed: {args.output_dir}")


if __name__ == "__main__":
    main()
