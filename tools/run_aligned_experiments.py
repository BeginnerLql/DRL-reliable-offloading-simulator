"""Train and pair four methods under the effective-QoS objective, or save preflight only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch

from compare_ppo_greedy import (build_agent, derived_seed, frozen_snapshot,
                                masked_evaluation, same_tree, train_masked_agent)
from compare_tore import cache_evaluation_profiles, summarize_task_groups, warmup_rasoul
from config.params import params, effective_spatial_risk_beta
from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH
from core.main_loop import MainLoop
from core.rasoul import RASOULPolicy


# Exploratory, specified from feasibility rather than method rankings.
# The scale multiplies base failure rates; it is not spatial-risk beta.
SCENARIOS = {
    'low_requirement': (.5, .9),
    'partial_feasibility': (.5, .99),
    'tight_feasible': (.1, .999),
    'high_requirement': (.05, .9999),
    'infeasible_boundary': (.1, .9999),
}
METHODS = [('PPO', 'trained_masked_ppo'), ('SML', 'safe_min_latency'),
           ('TORE', 'TORE'), ('RASOUL', 'RASOUL')]
METRICS = ['mean_qos', 'mean_latency_s', 'p95_latency_s', 'theoretical_rsr',
           'deadline_satisfaction', 'joint_satisfaction', 'mean_replicas', 'mean_satisfied_reliability']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def input_hashes():
    sources = [Path(__file__), ROOT/'compare_ppo_greedy.py', ROOT/'compare_tore.py', ROOT/'requirements.txt']
    for folder in ('core', 'agents', 'config', 'io_utils'):
        sources.extend((ROOT/folder).glob('*.py'))
    sources.extend(map(Path, (SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH)))
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}


def prepare_scenarios(names):
    servers = MainLoop._build_server_profiles(pd.read_excel(
        SERVER_INFO_PATH, sheet_name='Servers', dtype={'Site_ID': 'string'}))
    originals = {split: MainLoop._build_task_profiles(pd.read_excel(path))
                 for split, path in (('train', TRAIN_TASK_PATH), ('evaluation', EVALUATION_TASK_PATH))}
    if any(len(tasks) != params.taskno for tasks in originals.values()):
        raise ValueError('Each workbook must contain params.taskno tasks')
    cases, feasibility = {}, []
    for name in names:
        scale, requirement = SCENARIOS[name]
        splits = {}
        for split, original in originals.items():
            tasks = cache_evaluation_profiles(servers, original, failure_rate_scale=scale,
                ell_ratio=1., reliability_requirement=requirement)
            counts = np.array([p['safe_mask'].sum() for p in tasks.values()])
            feasibility.append(dict(scenario=name, split=split, failure_rate_scale=scale,
                reliability_requirement=requirement, task_count=len(tasks),
                no_safe_action_count=int((counts == 0).sum()),
                single_safe_action_count=int((counts == 1).sum()),
                multi_safe_action_count=int((counts > 1).sum()),
                expected_rsr=float((counts > 0).mean())))
            splits[split] = tasks
        if not any(p['safe_mask'].any() for p in splits['train'].values()) and any(
                p['safe_mask'].any() for p in splits['evaluation'].values()):
            raise ValueError(f'{name}: no train decisions but evaluation requires a policy')
        cases[name] = splits
    return servers, originals, cases, pd.DataFrame(feasibility)


def paired_statistics(seeds):
    """Use independently trained seeds as the sampling unit, not individual tasks."""
    rows = []
    for scenario, group in seeds.groupby('scenario'):
        ppo = group[group.method == 'PPO'].set_index('seed_id')
        for reference in ('SML', 'TORE', 'RASOUL'):
            other = group[group.method == reference].set_index('seed_id').loc[ppo.index]
            for metric in METRICS:
                delta = (ppo[metric] - other[metric]).dropna()
                half = float(student_t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.nan
                rows.append(dict(scenario=scenario, reference=reference, metric=metric,
                    seed_count=len(delta), delta_ppo_minus_reference=float(delta.mean()),
                    ci95_lower=float(delta.mean()-half), ci95_upper=float(delta.mean()+half),
                    ppo_better_seeds=int((delta < 0 if metric in ('mean_latency_s', 'p95_latency_s', 'mean_replicas')
                                         else delta > 0).sum())))
    return pd.DataFrame(rows)


def task_summary(records, keys):
    summary = summarize_task_groups(records, keys)
    extras = records.groupby(keys).agg(mean_raw_qos=('task_qos', 'mean'),
        deadline_satisfaction=('on_time', 'mean'), joint_satisfaction=('joint', 'mean'),
        mean_replicas=('replica_count', 'mean'))
    return summary.merge(extras.reset_index(), on=keys, validate='one_to_one')


def run_seed(name, splits, servers, args, seed, case):
    scale, _ = SCENARIOS[name]
    scenario_id = list(SCENARIOS).index(name)
    learnable = any(profile['safe_mask'].any() for profile in splits['train'].values())
    agent = None
    training_updates = 0
    if learnable:
        torch.manual_seed(derived_seed((121, scenario_id, seed, 0)))
        agent = build_agent(device=args.device)
        agent.save_model(case/'initial.pt')
        torch.manual_seed(derived_seed((121, scenario_id, seed, 1)))
        np.random.seed(derived_seed((121, scenario_id, seed, 2)))
        loop, training, train_load = train_masked_agent(
            agent, servers, splits['train'], scenario_id, seed, args.train_episodes,
            arrival_rate=args.arrival_rate, failure_rate_scale=scale,
            arrival_stream=(121, scenario_id, seed, 3), progress_label=name)
        np.testing.assert_allclose(training.episode_reward, params.taskno*training.mean_qos,
                                   rtol=0, atol=1e-9)
        training.to_csv(case/'train_episode_metrics.csv', index=False)
        train_load.to_csv(case/'train_server_load.csv', index=False)
        pd.DataFrame(loop.task_results).to_csv(case/'train_task_metrics.csv.gz', index=False)
        training_updates = int(training.actor_optimizer_steps.sum())
        agent.save_model(case/'final.pt')
        del loop
        for network in (agent.policy_net, agent.policy_old, agent.value_net):
            network.eval().requires_grad_(False)
    snapshot = frozen_snapshot(agent) if agent is not None else None
    history, warmup, warmup_records = warmup_rasoul(servers, splits['train'], seed=seed,
        episodes=args.rasoul_warmup_episodes, arrival_rate=args.arrival_rate,
        failure_rate_scale=scale, ell_ratio=1.)
    write_json(case/'rasoul_history.json', history)
    pd.DataFrame(warmup).to_csv(case/'rasoul_warmup_episode_metrics.csv', index=False)
    pd.DataFrame(warmup_records).to_csv(case/'rasoul_warmup_tasks.csv.gz', index=False)
    records, load_frames = [], []
    for episode in range(1, args.eval_episodes+1):
        # Same arrivals for every method; evaluated arrivals never enter training/warmup.
        arrival_seed = derived_seed((122, scenario_id, seed, episode))
        basis = np.random.default_rng(arrival_seed).exponential(size=params.taskno)
        arrival_hashes = []
        for label, method in METHODS:
            context = SimpleNamespace(this_episode=episode)
            rows = []
            values, loads = masked_evaluation(method, agent if label == 'PPO' else None,
                servers, splits['evaluation'], basis, episode,
                derived_seed((123, scenario_id, seed, episode)),
                lambda task: MainLoop.get_task_outcome_info(context, task), task_records=rows,
                arrival_rate=args.arrival_rate, failure_rate_scale=scale,
                rasoul_policy=RASOULPolicy.from_snapshot(history) if label == 'RASOUL' else None)
            arrival_hashes.append(values['arrival_sha256'])
            frame = pd.DataFrame(rows).assign(scenario=name, seed_id=seed, method=label,
                arrival_seed=arrival_seed, ppo_trained=learnable if label == 'PPO' else False)
            frame['on_time'] = frame.task_latency <= frame.deadline_s
            frame['joint'] = frame.on_time & frame.requirement_satisfied
            frame['replica_count'] = 1 + frame.server_B_id.notna().astype(int)
            np.testing.assert_allclose(values['mean_qos'], frame.task_utility.mean(), rtol=0, atol=1e-12)
            records.append(frame)
            load_frames.append(loads.assign(scenario=name, seed_id=seed, method=label))
            if agent is not None:
                same_tree(snapshot, frozen_snapshot(agent))
        assert len(set(arrival_hashes)) == 1
        print(f'{name}: seed {seed}, paired evaluation {episode}/{args.eval_episodes}', flush=True)
    columns = list(dict.fromkeys(column for frame in records for column in frame))
    data = pd.concat([frame.dropna(axis=1, how='all') for frame in records],
                     ignore_index=True).reindex(columns=columns)
    data.to_csv(case/'eval_task_metrics.csv.gz', index=False)
    pd.concat(load_frames, ignore_index=True).to_csv(case/'server_load_metrics.csv', index=False)
    episodes = task_summary(data, ['scenario', 'seed_id', 'episode', 'method'])
    seeds = task_summary(data, ['scenario', 'seed_id', 'method'])
    episodes.to_csv(case/'eval_episode_metrics.csv', index=False)
    seeds.to_csv(case/'eval_seed_summary.csv', index=False)
    write_json(case/'validation.json', dict(passed=True, ppo_trained=learnable,
        actor_optimizer_steps=training_updates, paired_arrivals=True, frozen_evaluation=True,
        task_count=len(data), reward_equals_total_effective_qos=True))
    return seeds, episodes


def run(args):
    if args.device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    hashes = input_hashes()
    servers, originals, cases, feasibility = prepare_scenarios(args.scenarios)
    needs_policy = any(any(p['safe_mask'].any() for p in splits['train'].values()) for splits in cases.values())
    if not args.preflight_only and needs_policy and args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --preflight-only to prepare the suite; formal GPU training has not started.')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'input_manifest.json', hashes)
    for source in hashes:
        path = Path(source)
        target = output/'source_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    feasibility.to_csv(output/'feasibility.csv', index=False)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]),
        objective='mean effective QoS over all tasks', reward_assignment='completion interval',
        gamma=params.gamma_ppo, gae_lambda=params.gae_lambda_ppo,
        seeds=args.seeds, train_episodes=args.train_episodes, eval_episodes=args.eval_episodes,
        rasoul_warmup_episodes=args.rasoul_warmup_episodes, arrival_rate=args.arrival_rate,
        scenarios={name: dict(failure_rate_scale=SCENARIOS[name][0], requirement=SCENARIOS[name][1]) for name in args.scenarios},
        device=args.device, cuda_available=torch.cuda.is_available(), torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        beta=effective_spatial_risk_beta(), ell_ratio=1., spatial_risk_enabled=params.SPATIAL_RISK_ENABLED,
        parameter_status='Exploratory feasibility scenarios; no literature calibration claimed',
        checkpoint_selection='Final episode; no evaluation-based checkpoint selection',
        threshold_override='Both training and evaluation, in memory only; workbooks unchanged',
        statistics='Paired differences across independent training seeds; uncorrected 95% t intervals',
        mean_satisfied_reliability_scope='Selected unified-model reliability averaged over satisfied tasks; empty groups NaN; pooled by satisfied count',
        hyperparameters={key:value for key,value in vars(params).items() if key.endswith('_ppo')},
        status='preflight_only' if args.preflight_only else 'running')
    write_json(output/'run_metadata.json', metadata)
    for name, splits in cases.items():
        folder = output/name
        folder.mkdir()
        effective = []
        arrays = {}
        for split, tasks in splits.items():
            for tid, profile in tasks.items():
                effective.append(dict(split=split, Task_ID=tid,
                    **{key:value for key,value in profile.items() if key not in ('Task_ID','safe_mask','reliability_snapshot')},
                    Original_Reliability_Requirement=originals[split][tid]['Reliability_Requirement']))
            for i, label in enumerate(('marginal', 'correlated', 'independent')):
                arrays[f'{split}_{label}'] = np.stack([p['reliability_snapshot'][i] for p in tasks.values()])
        pd.DataFrame(effective).to_csv(folder/'effective_task_profiles.csv', index=False)
        np.savez_compressed(folder/'reliability_table.npz', **arrays)
    if args.preflight_only:
        print(feasibility.to_string(index=False))
        lines = ['# 有效 QoS 实验预检', '',
            '仅扫描可行性并固定探索性场景，没有启动训练。所有阈值覆盖同时用于训练和评估，原始工作簿不变。', '',
            '|场景|任务集|故障率缩放|可靠性要求|无解|单动作|多动作|', '|---|---|---:|---:|---:|---:|---:|']
        for row in feasibility.itertuples():
            lines.append(f'|{row.scenario}|{row.split}|{row.failure_rate_scale:g}|{row.reliability_requirement:g}|'
                         f'{row.no_safe_action_count}|{row.single_safe_action_count}|{row.multi_safe_action_count}|')
        lines.extend(['', f'CUDA 可用：{torch.cuda.is_available()}。正式训练默认每场景10个独立种子×300轮，评估20轮，RASOUL预热100轮。',
            '全无解场景只比较回退行为，不创建训练模型，也不计为独立训练能力证据。场景根据可行性定义，不根据方法成绩筛选。'])
        (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    else:
        all_seeds, all_episodes = [], []
        for name, splits in cases.items():
            for seed in args.seeds:
                case = output/name/f'seed_{seed}'
                case.mkdir()
                seeds, episodes = run_seed(name, splits, servers, args, seed, case)
                all_seeds.append(seeds)
                all_episodes.append(episodes)
        seeds = pd.concat(all_seeds, ignore_index=True)
        seeds.to_csv(output/'eval_seed_summary.csv', index=False)
        pd.concat(all_episodes, ignore_index=True).to_csv(output/'eval_episode_metrics.csv', index=False)
        paired_statistics(seeds).to_csv(output/'paired_statistics.csv', index=False)
        summary = seeds.groupby(['scenario','method'])[METRICS].agg(['mean','std'])
        totals = seeds.groupby(['scenario','method'])[['satisfied_task_count','satisfied_reliability_sum']].sum()
        summary[('mean_satisfied_reliability','mean')] = (
            totals.satisfied_reliability_sum / totals.satisfied_task_count.replace(0, np.nan))
        summary.to_csv(output/'summary.csv')
        lines = ['# 有效 QoS 四方法配对实验', '',
            '主指标为全部任务平均有效QoS；可靠性不达标置0。平均时延、P95、按时率、副本数为辅助指标。',
            '每种子训练最终检查点冻结评估；RASOUL每个评估回合从本种子预热历史重置。',
            'paired_statistics.csv 使用种子作为统计单位。全无解场景的PPO仅为固定回退，没有策略训练或推理。', '',
            '达标任务平均可靠性按达标任务数汇集计算；无达标任务显示“无样本”，其种子间SD只使用有样本的种子。', '',
            '|场景|方法|有效QoS均值|种子间SD|平均时延/s|RSR|达标任务平均可靠性|', '|---|---|---:|---:|---:|---:|---:|']
        for (name, method), row in summary.iterrows():
            lines.append(f'|{name}|{method}|{row[("mean_qos","mean")]:.6f}|{row[("mean_qos","std")]:.6f}|'
                         f'{row[("mean_latency_s","mean")]:.6f}|{row[("theoretical_rsr","mean")]:.2%}|'
                         f'{"无样本" if pd.isna(row[("mean_satisfied_reliability","mean")]) else format(row[("mean_satisfied_reliability","mean")], ".8f")}|')
        lines.append('\n单种子或短训练仅供功能验证，不构成稳定优势证据。原始任务和历史结果保留。')
        (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    assert hashes == input_hashes(), 'Inputs or sources changed during the experiment'
    if not args.preflight_only:
        metadata['status'] = 'complete'
        write_json(output/'run_metadata.json', metadata)
    write_json(output/'validation.json', dict(passed=True, preflight_only=args.preflight_only,
        sources_and_workbooks_unchanged=True, elapsed_s=time.monotonic()-started))
    write_json(output/'output_manifest.json', {str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob('*') if p.is_file() and p.name != 'output_manifest.json'})
    print(f'Complete: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--scenarios', choices=list(SCENARIOS), nargs='+', default=list(SCENARIOS))
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda', help='CPU is explicit and intended for integration checks')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1,11)))
    parser.add_argument('--train-episodes', type=int, default=300)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--rasoul-warmup-episodes', type=int, default=100)
    parser.add_argument('--arrival-rate', type=float, default=2.)
    args = parser.parse_args()
    if any(value <= 0 for value in (args.train_episodes,args.eval_episodes,args.rasoul_warmup_episodes)):
        parser.error('Episode counts must be positive')
    if not np.isfinite(args.arrival_rate) or args.arrival_rate <= 0:
        parser.error('Arrival rate must be positive and finite')
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('Seeds must be distinct nonnegative integers')
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error('Scenarios must be distinct')
    run(args)


if __name__ == '__main__':
    main()
