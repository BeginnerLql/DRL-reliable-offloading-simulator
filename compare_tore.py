"""Paired TORE/SML/RASOUL evaluation, optionally against an existing frozen PPO.

RASOUL warms up on training tasks and learns online within each evaluation
episode. No PPO training or workbook regeneration is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import simpy
import torch

from compare_ppo_greedy import (
    build_agent, derived_seed, frozen_snapshot, make_environment, masked_evaluation,
    same_tree, set_correlation_length,
)
from config.params import params, effective_spatial_risk_beta
from config.paths import EVALUATION_TASK_PATH, RESULTS_DIR, SERVER_INFO_PATH, TRAIN_TASK_PATH
from core.main_loop import MainLoop
from core.rasoul import RASOULPolicy
from core.task import Task, compute_task_utility
from io_utils.save_parameters_and_logs import summarize_satisfied_reliability


def reliable_qos_metrics(data):
    """Legacy conditional metric retained for historical experiment scripts."""
    eligible = data.loc[data.requirement_satisfied, 'task_qos']
    return dict(mean_qos=float(eligible.mean()), qos_eligible_count=len(eligible),
                qos_eligible_sum=float(eligible.sum()))


def reliability_adjusted_qos(data):
    """Zero QoS below the unified reliability requirement; preserve raw outcomes."""
    return pd.Series(compute_task_utility(data.task_qos, data.requirement_satisfied),
                     index=data.index, name='task_qos')


def all_task_qos_metrics(data):
    """Count every task and assign zero QoS to reliability-unsatisfied tasks."""
    qos = reliability_adjusted_qos(data)
    return dict(mean_qos=float(qos.mean()), qos_task_count=len(data),
                qos_sum=float(qos.sum()))


def aggregate_comparison_metrics(data, keys, fields):
    """Pool QoS by task count; also support historical conditional-metric files."""
    grouped = data.groupby(keys)
    result = grouped[fields].mean()
    count, total = (('qos_task_count', 'qos_sum') if 'qos_task_count' in data
                    else ('qos_eligible_count', 'qos_eligible_sum'))
    totals = grouped[[count, total]].sum()
    result = result.join(totals)
    result['mean_qos'] = result[total] / result[count].replace(0, np.nan)
    if 'satisfied_reliability_sum' in data:
        reliability = grouped[['satisfied_task_count', 'satisfied_reliability_sum']].sum()
        result = result.join(reliability)
        result['mean_satisfied_reliability'] = (result.satisfied_reliability_sum /
                                               result.satisfied_task_count.replace(0, np.nan))
    return result.reset_index()


def summarize_task_groups(records, keys, *, only_satisfied=False):
    """Include all tasks, zero unsatisfied QoS, and retain actual task latency."""
    all_groups = records.groupby(keys)
    base = all_groups.agg(tier_task_count=('task_qos', 'size'),
                          theoretical_rsr=('requirement_satisfied', 'mean'))
    selected = records.loc[records.requirement_satisfied] if only_satisfied else records
    selected = selected.assign(reported_qos=reliability_adjusted_qos(selected))
    grouped = selected.groupby(keys)
    frame = grouped.agg(qos_task_count=('reported_qos', 'size'), qos_sum=('reported_qos', 'sum'),
        mean_latency_s=('task_latency', 'mean'), fallback_fraction=('is_fallback', 'mean'))
    frame['p95_latency_s'] = grouped.task_latency.quantile(.95)
    frame = base.join(frame)
    frame['qos_task_count'] = frame.qos_task_count.fillna(0).astype(int)
    frame['qos_sum'] = frame.qos_sum.fillna(0.)
    frame['mean_qos'] = frame.qos_sum / frame.qos_task_count.replace(0, np.nan)
    if 'false_safe_selection' in records:
        frame['false_safe_fraction'] = all_groups.false_safe_selection.mean()
    frame = frame.reset_index()
    if 'pair_reliability' in records:
        frame = frame.merge(summarize_satisfied_reliability(records, keys), on=keys, validate='one_to_one')
    return frame


def summarize_reliability_tiers(records, *, only_satisfied=False):
    """Group by requested reliability and include all tasks by default."""
    frames = []
    for keys in (['seed_id', 'episode', 'reliability_requirement', 'method'],
                 ['seed_id', 'reliability_requirement', 'method'],
                 ['reliability_requirement', 'method']):
        frames.append(summarize_task_groups(records, keys, only_satisfied=only_satisfied))
    return tuple(frames)


def cache_evaluation_profiles(profiles, tasks, *, failure_rate_scale, ell_ratio,
                              reliability_requirement=None):
    """Share marginals and both models; retain correlated masks for PPO/SML."""
    if reliability_requirement is not None and (
            not np.isfinite(reliability_requirement) or not 0 < reliability_requirement <= 1):
        raise ValueError('Reliability requirement must be finite and in (0, 1]')
    _, state = make_environment(profiles, failure_rate_scale=failure_rate_scale)
    state.initialize_spatial_reliability()
    set_correlation_length(state, ell_ratio)
    cached = {}
    for tid, profile in tasks.items():
        # Rebuild masks for the effective threshold and physical parameters;
        # never reuse a cache from an earlier experiment or mutate its inputs.
        profile = {key: value for key, value in profile.items()
                   if key not in ('reliability_snapshot', 'safe_mask')}
        if reliability_requirement is not None:
            profile['Reliability_Requirement'] = float(reliability_requirement)
        task = Task(simpy.Environment(), state, tid, profile)
        mask = state.get_safe_action_mask(task)
        for value in (*task.reliability_snapshot, mask):
            value.setflags(write=False)
        cached[tid] = {**profile, 'reliability_snapshot': task.reliability_snapshot, 'safe_mask': mask}
    return cached


def warmup_rasoul(profiles, tasks, *, seed, episodes, arrival_rate, failure_rate_scale, ell_ratio):
    """Retain learning across fresh queues; return only a fully drained history."""
    policy = RASOULPolicy(sorted(profiles))
    metrics, records = [], []
    for episode in range(1, episodes + 1):
        arrival_seed = derived_seed((104, seed, episode, 0))
        basis = np.random.default_rng(arrival_seed).exponential(size=len(tasks))
        context = SimpleNamespace(this_episode=episode, use_safety_mask=True, num_actions=params.num_actions)
        values, _ = masked_evaluation('RASOUL', None, profiles, tasks, basis, episode, 0,
            lambda task: MainLoop.get_task_outcome_info(context, task), task_records=records,
            arrival_rate=arrival_rate, failure_rate_scale=failure_rate_scale, ell_ratio=ell_ratio,
            rasoul_policy=policy)
        metrics.append(dict(seed_id=seed, arrival_seed=arrival_seed, **values))
        if episode % 20 == 0:
            print(f'RASOUL warmup seed {seed}: {episode}/{episodes}', flush=True)
    history = policy.snapshot()
    assert history['decision_count'] == episodes * len(tasks)
    assert sum(history['feedback_counts']) == history['decision_count']
    return history, metrics, records


def run_comparison(args):
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    agent = None
    if args.ppo_checkpoint is not None:
        agent = build_agent(device=args.device)
        agent.load_model(args.ppo_checkpoint)
        for network in (agent.policy_net, agent.policy_old, agent.value_net):
            network.eval().requires_grad_(False)
    snapshot = frozen_snapshot(agent) if agent is not None else None
    input_paths = [Path(SERVER_INFO_PATH), Path(TRAIN_TASK_PATH), Path(EVALUATION_TASK_PATH)]
    if args.ppo_checkpoint is not None:
        input_paths.append(args.ppo_checkpoint)
    source_paths = [Path(p) for p in (
        'compare_tore.py', 'compare_ppo_greedy.py', 'core/tore.py', 'core/rasoul.py', 'core/env_state.py',
        'core/task.py', 'core/main_loop.py', 'core/server.py', 'core/spatial_risk.py',
        'io_utils/save_parameters_and_logs.py', 'agents/ppo_agent.py',
        'config/configuration.py', 'config/params.py',
    )]
    source_paths = [Path(__file__).resolve().parent / p for p in source_paths]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in input_paths + source_paths}
    profiles = MainLoop._build_server_profiles(pd.read_excel(
        SERVER_INFO_PATH, sheet_name='Servers', dtype={'Site_ID': 'string'}))
    task_profiles = MainLoop._build_task_profiles(pd.read_excel(EVALUATION_TASK_PATH))
    if len(task_profiles) != params.taskno:
        raise ValueError('Evaluation workbook task count must match the current configuration')
    tasks = cache_evaluation_profiles(profiles, task_profiles,
        failure_rate_scale=args.failure_rate_scale, ell_ratio=args.ell_ratio,
        reliability_requirement=args.reliability_requirement)
    training_profiles = MainLoop._build_task_profiles(pd.read_excel(TRAIN_TASK_PATH))
    if len(training_profiles) != params.taskno:
        raise ValueError('Training workbook task count must match the current configuration')
    # Preserve the existing evaluation-only threshold override. Training keeps
    # its original task requirements; both physical models use the same scale.
    training_tasks = cache_evaluation_profiles(profiles, training_profiles,
        failure_rate_scale=args.failure_rate_scale, ell_ratio=args.ell_ratio)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        command=shlex.join([sys.executable, *sys.argv]), seeds=args.seeds, episodes=args.episodes,
        arrival_rate=args.arrival_rate, failure_rate_scale=args.failure_rate_scale,
        ell_ratio=args.ell_ratio, beta=effective_spatial_risk_beta(), gh_order=32,
        spatial_risk_enabled=params.SPATIAL_RISK_ENABLED,
        mean_qos_scope='All tasks in denominator; QoS is zero for requirement_satisfied=False under the unified evaluation model; raw task_qos preserved',
        mean_satisfied_reliability_scope='Mean selected reliability among requirement_satisfied tasks under the unified model; NaN when none; reliable fallback included',
        latency_scope='All task execution latencies, including unsatisfied tasks and fallback',
        requirements=sorted({p['Reliability_Requirement'] for p in tasks.values()}),
        reliability_requirement_override=args.reliability_requirement,
        original_requirements=sorted({p['Reliability_Requirement'] for p in task_profiles.values()}),
        requirement_override_scope='evaluation only; source workbook unchanged',
        ppo_checkpoint=str(args.ppo_checkpoint.resolve()) if args.ppo_checkpoint is not None else None,
        ppo_device=args.device if agent is not None else None,
        methods=['TORE', 'SML', 'RASOUL'] + (['PPO'] if agent is not None else []),
        selection_models={'TORE': 'independent', 'SML': 'correlated', 'RASOUL': 'marginal', 'PPO': 'correlated'},
        evaluation_model='correlated', maximum_tore_replicas=2,
        tore_priority=['node predicted QoS descending', 'node predicted completion ascending',
                       'server ID ascending', 'add one node and check reliability', 'at most two replicas'],
        availability_in_selection=False,
        rasoul_warmup_episodes=args.rasoul_warmup_episodes,
        rasoul_warmup_requirements=sorted({p['Reliability_Requirement'] for p in training_tasks.values()}),
        rasoul_protocol='Training-task warmup per seed; deep-copy full history into each fresh evaluation queue; online completed feedback within episode.',
        rasoul_score='predicted QoS - mean raw prediction error + sqrt(2*ln(max(2,N))/n); unobserved nodes have first priority; scores are not clipped.',
        rasoul_admission='sum(full execution time / own deadline) over running and CPU-waiting tasks + new execution / deadline <= 1; exclude uploads in flight.',
        rasoul_fallbacks={'admission': 'fastest reliable node', 'no_reliable_node': 'fastest node'},
        note='Seeds pair RASOUL warmup and arrival scenarios; one optional frozen PPO checkpoint is reused, with no PPO training.',
    )
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (output/'input_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    pd.DataFrame([
        {**{key: value for key, value in profile.items()
            if key not in ('reliability_snapshot', 'safe_mask')},
         'Task_ID': tid,
         'Original_Reliability_Requirement': task_profiles[tid]['Reliability_Requirement']}
        for tid, profile in tasks.items()
    ]).to_csv(output/'effective_task_profiles.csv', index=False, float_format='%.17g')
    np.savez_compressed(output/'reliability_table.npz', task_ids=list(tasks),
        server_ids=sorted(profiles), pairs=MainLoop.generate_combinations(),
        **{name: np.stack([p['reliability_snapshot'][i] for p in tasks.values()])
           for i, name in enumerate(('marginal', 'correlated', 'independent'))})
    rows, all_tasks, all_loads, warmup_rows = [], [], [], []
    methods = [('TORE', 'TORE', None), ('SML', 'safe_min_latency', None), ('RASOUL', 'RASOUL', None)]
    if agent is not None:
        methods.append(('PPO', 'trained_masked_ppo', agent))
    completed_replicas = 0
    for seed in args.seeds:
        history, warmup_metrics, warmup_tasks = warmup_rasoul(profiles, training_tasks, seed=seed,
            episodes=args.rasoul_warmup_episodes, arrival_rate=args.arrival_rate,
            failure_rate_scale=args.failure_rate_scale, ell_ratio=args.ell_ratio)
        history_hash = RASOULPolicy.history_digest(history)
        (output/f'rasoul_history_seed_{seed}.json').write_text(json.dumps(history, indent=2)+'\n')
        pd.DataFrame(warmup_tasks).to_csv(output/f'rasoul_warmup_tasks_seed_{seed}.csv.gz',
                                         index=False, float_format='%.17g')
        warmup_rows.extend(warmup_metrics)
        for episode in range(1, args.episodes+1):
            arrival_seed = derived_seed((103, seed, episode, 0))
            basis = np.random.default_rng(arrival_seed).exponential(size=len(tasks))
            arrivals = []
            for label, method, policy in methods:
                context = SimpleNamespace(this_episode=episode, use_safety_mask=True,
                                          num_actions=params.num_actions)
                records = []
                action_seed = derived_seed((103, seed, episode, 1))
                rasoul_policy = RASOULPolicy.from_snapshot(history) if label == 'RASOUL' else None
                if rasoul_policy is not None:
                    assert RASOULPolicy.history_digest(rasoul_policy.snapshot()) == history_hash
                values, loads = masked_evaluation(method, policy, profiles, tasks, basis,
                    episode, action_seed, lambda task: MainLoop.get_task_outcome_info(context, task),
                    task_records=records, arrival_rate=args.arrival_rate,
                    failure_rate_scale=args.failure_rate_scale, ell_ratio=args.ell_ratio,
                    rasoul_policy=rasoul_policy)
                data = pd.DataFrame(records)
                data['reported_qos'] = reliability_adjusted_qos(data)
                values.update(**all_task_qos_metrics(data),
                              mean_latency_s=float(data.task_latency.mean()),
                              p95_latency_s=float(data.task_latency.quantile(.95)))
                if rasoul_policy is not None:
                    final_history = rasoul_policy.snapshot()
                    assert final_history['decision_count'] == history['decision_count'] + len(tasks)
                    assert sum(final_history['feedback_counts']) == final_history['decision_count']
                    assert RASOULPolicy.history_digest(history) == history_hash
                    values.update(rasoul_initial_history_sha256=history_hash,
                                  rasoul_final_history_sha256=RASOULPolicy.history_digest(final_history),
                                  rasoul_initial_decision_count=history['decision_count'],
                                  rasoul_final_decision_count=final_history['decision_count'])
                if label not in ('TORE', 'RASOUL'):
                    data['selection_reliability_model'] = 'correlated'
                    data['selection_reliability'] = data.pair_reliability
                    data['selection_requirement_satisfied'] = data.requirement_satisfied
                data['false_safe_selection'] = (data.selection_requirement_satisfied
                                                & ~data.requirement_satisfied)
                values['false_safe_fraction'] = float(data.false_safe_selection.mean())
                rows.append(dict(seed_id=seed, method=label, arrival_seed=arrival_seed,
                                 action_seed=action_seed, **values))
                all_tasks.append(data.assign(seed_id=seed, method=label))
                all_loads.append(loads.assign(seed_id=seed, method=label))
                completed_replicas += int(loads.replica_count.sum())
                arrivals.append(values['arrival_sha256'])
                if agent is not None:
                    same_tree(snapshot, frozen_snapshot(agent))
            assert len(set(arrivals)) == 1
            print(f'Completed seed {seed}, paired episode {episode}/{args.episodes}', flush=True)
    evaluation = pd.DataFrame(rows)
    task_columns = list(dict.fromkeys(column for frame in all_tasks for column in frame))
    task_rows = pd.concat([frame.dropna(axis=1, how='all') for frame in all_tasks],
                          ignore_index=True).reindex(columns=task_columns)
    load_rows = pd.concat(all_loads, ignore_index=True)
    fields = ['mean_qos', 'mean_latency_s', 'p95_latency_s', 'theoretical_rsr',
              'fallback_fraction', 'false_safe_fraction', 'mean_satisfied_reliability']
    seed_summary = summarize_task_groups(task_rows, ['seed_id', 'method'])
    summary = summarize_task_groups(task_rows, ['method'])
    tier_episodes, tier_seeds, tier_summary = summarize_reliability_tiers(task_rows)
    differences = []
    reference = evaluation[evaluation.method == 'TORE'].set_index(['seed_id', 'episode_id'])
    for label in metadata['methods'][1:]:
        other = evaluation[evaluation.method == label].set_index(['seed_id', 'episode_id']).loc[reference.index]
        for key in reference.index:
            differences.append(dict(seed_id=key[0], episode_id=key[1], method=label, reference='TORE',
                **{f'delta_{field}': float(other.loc[key, field] - reference.loc[key, field])
                   for field in fields}))
    for name, frame in (
        ('eval_episode_metrics', evaluation), ('eval_task_metrics', task_rows),
        ('server_load_metrics', load_rows), ('eval_seed_summary', seed_summary),
        ('summary', summary), ('paired_differences', pd.DataFrame(differences)),
        ('reliability_tier_episode_metrics', tier_episodes),
        ('reliability_tier_seed_summary', tier_seeds), ('reliability_tier_summary', tier_summary),
        ('rasoul_warmup_episode_metrics', pd.DataFrame(warmup_rows)),
    ):
        frame.to_csv(output/f'{name}.csv', index=False, float_format='%.17g')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value for p, value in hashes.items())
    validation = dict(passed=True, paired_arrivals=True, frozen_ppo=agent is not None,
        completed_tasks=len(task_rows), completed_replicas=completed_replicas,
        rasoul_warmup_tasks=len(args.seeds) * args.rasoul_warmup_episodes * len(training_tasks),
        rasoul_evaluation_history_reset=True, rasoul_all_completed_feedback_collected=True,
        source_and_inputs_unchanged=True, elapsed_s=time.monotonic()-started)
    (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    lines = ['# ' + ' / '.join(metadata['methods']) + ' 配对对比', '',
        'TORE 使用独立可靠性，最多两个副本；SML/PPO 使用相关安全集合；RASOUL 固定单副本，使用相同节点边际可靠性。最终可靠性均按统一模型评价。',
        'TORE 按单节点预测 QoS 降序排列；同分按节点预测完成时延及节点编号排序。逐个加入副本，每次加入后检查独立可靠性，达标即停止。',
        '最多两个副本；不先遍历全部单节点，不搜索替代节点对。达到上限仍不达标时进入现有无约束 SML Fallback。',
        'Fallback 不等于全局无可行节点对；逐任务记录贪心已选节点、可靠性及是否遗漏其他可行组合。',
        'TORE 选择依据预测 QoS、完成时延和可靠性；RASOUL 额外使用历史 QoS 预测误差与置信修正及单核负载准入。不使用可用性估计。',
        'RASOUL 先在训练任务上预热，评估每轮从同一预热历史副本开始，并仅在任务完成后在线更新；各方法分别重建真实队列。',
        'RASOUL 准入回退在可靠性可行节点中选最快者；无可行单节点时在全部节点中选最快者。两类回退分别记录。',
        '按任务自身的可靠性要求分档；QoS 分母为全部任务，统一评价模型下可靠性不达标的任务 QoS 记为 0，达标任务保留原始 QoS。逐任务 task_qos 保留原始时延评分。平均时延及 P95 使用全部任务的实际时延，包括回退任务。RSR 分母为全部任务。',
        f'RASOUL 每种子预热 {args.rasoul_warmup_episodes} 轮，预热不计入评估指标。阈值覆盖仅用于评估，预热保留训练工作簿的原始任务要求。',
        f'到达率={args.arrival_rate:g} tasks/s，故障率缩放={args.failure_rate_scale:g}，ell/ell0={args.ell_ratio:g}。',
        f'本次评估任务可靠性要求={metadata["requirements"]}；只在本次评估中覆盖阈值，原始工作簿不变。',
        'RSR 为理论可靠性达标率，没有采样实际失败事件。种子对应 RASOUL 预热和到达场景；复用一个 PPO 检查点不代表多个独立训练模型。', '',
        '达标任务平均可靠性只平均实际所选方案在统一模型下的可靠性，包含达标的回退任务；无达标任务记为“无样本”。', '',
        '|方法|全部任务平均QoS|全部任务平均时延/s|相关模型RSR|达标任务平均可靠性|全部任务Fallback|全部任务假安全选择|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'|{row.method}|{row.mean_qos:.6f}|{row.mean_latency_s:.6f}|'
                     f'{row.theoretical_rsr:.1%}|'
                     f'{"无样本" if pd.isna(row.mean_satisfied_reliability) else f"{row.mean_satisfied_reliability:.8f}"}|'
                     f'{row.fallback_fraction:.1%}|{row.false_safe_fraction:.1%}|')
    lines += ['', '## 按可靠性要求分档（每档全部任务）', '',
              '|可靠性要求|方法|统计任务数|平均QoS|平均时延/s|P95时延/s|达标任务平均可靠性|',
              '|---:|---|---:|---:|---:|---:|---:|']
    for row in tier_summary.itertuples():
        lines.append(f'|{row.reliability_requirement:g}|{row.method}|{row.qos_task_count}|'
                     f'{row.mean_qos:.6f}|{row.mean_latency_s:.6f}|{row.p95_latency_s:.6f}|'
                     f'{"无样本" if pd.isna(row.mean_satisfied_reliability) else f"{row.mean_satisfied_reliability:.8f}"}|')
    lines += ['', '本次不训练 PPO；RASOUL 按约定预热并在线学习。输入工作簿不变，结果不构成收敛或稳定优势的证据。',
              '', '```text', metadata['command'], '```', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'output_manifest.json').write_text(json.dumps({
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir() if p.is_file() and p.name != 'output_manifest.json'
    }, indent=2)+'\n')
    print(summary.drop(columns='p95_latency_s').to_string(index=False))
    print(f'Complete: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--rasoul-warmup-episodes', type=int, default=params.total_episodes,
                        help='RASOUL training-task warmup per seed (default: configured total_episodes)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    parser.add_argument('--arrival-rate', type=float, default=params.TASK_ARRIVAL_RATE)
    parser.add_argument('--failure-rate-scale', type=float, default=1.)
    parser.add_argument('--ell-ratio', type=float, default=1.)
    parser.add_argument('--reliability-requirement', type=float,
                        help='Override every evaluation task requirement; leave source workbooks unchanged')
    parser.add_argument('--ppo-checkpoint', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output-dir', type=Path, default=Path(RESULTS_DIR)/'tore_comparison')
    args = parser.parse_args()
    if (args.episodes < 1 or args.rasoul_warmup_episodes < 1 or not args.seeds or min(args.seeds) < 1
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error('Use positive episode counts and distinct positive seed IDs')
    if not np.isfinite(args.arrival_rate) or args.arrival_rate <= 0:
        parser.error('Arrival rate must be positive and finite')
    if any(not np.isfinite(v) or v < 0 for v in (args.failure_rate_scale, args.ell_ratio)):
        parser.error('Failure-rate scale and correlation-length ratio must be finite and nonnegative')
    if args.reliability_requirement is not None and (
            not np.isfinite(args.reliability_requirement) or not 0 < args.reliability_requirement <= 1):
        parser.error('Reliability requirement must be finite and in (0, 1]')
    run_comparison(args)


if __name__ == '__main__':
    main()
