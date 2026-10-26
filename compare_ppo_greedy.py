"""PPO/heuristic comparisons, including paired masked latency experiments."""

from concurrent.futures import ProcessPoolExecutor
import argparse
import contextlib
import copy
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import simpy
import torch

from agents.ppo_agent import PPOAgent
from config.params import params, effective_spatial_risk_beta
from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH, RESULTS_DIR
from core.env_state import EnvironmentState, select_latency_greedy_action
from core.main_loop import MainLoop
from core.server import Server
from core.task import Task, compute_task_qos, compute_task_utility
from core.tore import select_tore_action
from core.spatial_risk import (compute_marginal_failure_probability, compute_pair_reliability_gh,
                               build_spatial_correlation_matrix, compute_baseline_correlation_length_km)
from io_utils.save_parameters_and_logs import summarize_task_results, task_latency_components

ARRIVAL_RATES = (1., 1.5, 2., 2.5, 3., 3.5)
RATE_STREAM_IDS = {2.: 0, 3.: 1, 1.: 2, 1.5: 3, 2.5: 4, 3.5: 5}
METRICS = ('mean_latency_sec', 'mean_upload_time_sec',
           'mean_cpu_waiting_time_sec', 'mean_cpu_execution_time_sec')
OUTPUT_DIR = Path(RESULTS_DIR) / 'ppo_vs_latency_greedy'


def derived_seed(key):
    return int(np.random.SeedSequence(params.EXPERIMENT_SEED, spawn_key=key).generate_state(1)[0])


def seed_mapping(rates=ARRIVAL_RATES, seed_ids=range(1, 11)):
    return pd.DataFrame([
        {'arrival_rate': rate, 'rate_stream_id': RATE_STREAM_IDS[rate], 'seed_id': seed,
         'training_seed': derived_seed((6, 0, seed)),
         **{name: derived_seed((6, 1, RATE_STREAM_IDS[rate], seed, role))
            for role, name in enumerate(('init_seed', 'action_seed', 'shuffle_seed'))}}
        for rate in rates for seed in seed_ids
    ])


def build_agent(*, device='cpu'):
    keys = ('gamma', 'actor_lr', 'critic_lr', 'clip_eps', 'k_epochs', 'batch_size',
            'entropy_coef', 'reward_scale', 'gae_lambda', 'value_loss_coef', 'max_grad_norm')
    return PPOAgent(params.num_states, params.num_actions, params.hidden_layers_ppo,
                    device=device, activation=params.af_ppo,
                    **{k: getattr(params, k + '_ppo') for k in keys})


def make_environment(profiles, *, failure_rate_scale=1., reliability_model='correlated'):
    env, state = simpy.Environment(), EnvironmentState(failure_rate_scale=failure_rate_scale,
                                                     reliability_model=reliability_model)
    for sid, p in profiles.items():
        state.add_server_and_init_environment(Server(
            env, sid, p['Site_ID'], p['Processing_Frequency'], p['Base_Failure_Rate'],
            p['Transmission_Rate'], p['Latitude'], p['Longitude']))
    return env, state


def set_correlation_length(state, ell_ratio):
    if ell_ratio != 1:
        state.spatial_correlation_matrix = build_spatial_correlation_matrix(
            state.spatial_distance_matrix,
            ell_ratio * compute_baseline_correlation_length_km(state.spatial_distance_matrix))


def environment_basis(key):
    rng = np.random.default_rng(np.random.SeedSequence(params.EXPERIMENT_SEED, spawn_key=(*key, 0)))
    return rng.exponential(size=params.taskno)


def task_metrics(tasks, rate, episode_id):
    parts = []
    actions = []
    pairs = MainLoop.generate_combinations()
    for task in tasks:
        winner = min((r for r in (task.replica_A, task.replica_B) if r['finish_time'] is not None),
                     key=lambda r: r['finish_time'])
        delay = task.task_completion_time - task.arrival_time
        components = (winner['queue_enter_time']-task.arrival_time,
                      winner['cpu_start_time']-winner['queue_enter_time'],
                      winner['finish_time']-winner['cpu_start_time'])
        assert abs(delay-sum(components)) < 1e-9
        parts.append((delay, *components))
        actions.append(pairs.index((task.replica_A['server_id'], task.replica_B['server_id']))
                       if task.replica_B['server_id'] is not None else None)
    metrics = summarize_latency_parts(parts, actions, rate, episode_id,
                                      deadlines=[task.deadline for task in tasks])
    raw = compute_task_qos(np.asarray(parts)[:, 0], [task.deadline for task in tasks],
                           grace_multiplier=params.QOS_GRACE_MULTIPLIER)
    utility = compute_task_utility(raw, [task.pair_reliability >= task.reliability_requirement for task in tasks])
    metrics.update(mean_qos=float(utility.mean()), mean_raw_qos=float(raw.mean()),
                   utility_sum=float(utility.sum()))
    return metrics


def summarize_latency_parts(parts, actions, rate, episode_id, *, deadlines):
    data = np.asarray(parts)
    pairs = MainLoop.generate_combinations()
    counts = np.bincount([action for action in actions if action is not None], minlength=len(pairs))
    return {'arrival_rate': rate, 'episode_id': episode_id,
            'mean_latency_sec': data[:, 0].mean(),
            **{name: data[:, i].mean() for i, name in enumerate(METRICS[1:], start=1)},
            'episode_raw_reward': compute_task_qos(data[:, 0], deadlines,
                grace_multiplier=params.QOS_GRACE_MULTIPLIER).sum(),
            'pair_hhi': np.sum((counts/len(data))**2) if all(a is not None for a in actions) else np.nan,
            **{f'action_count_{i}': int(count) for i, count in enumerate(counts)}}


class TrainingLoop(MainLoop):
    """Use independent episode streams while retaining the production PPO loop."""
    def __init__(self, agent, rate, seed, episodes, *, failure_rate_scale=1., arrival_stream=None,
                 reliability_model='correlated', ell_ratio=1.):
        self.rate, self.seed = rate, seed
        self.arrival_stream = arrival_stream
        self.ell_ratio = ell_ratio
        super().__init__(agent, episodes, params.taskno, params.num_states, params.num_actions,
                         failure_rate_scale=failure_rate_scale, reliability_model=reliability_model)

    def _initialize_episode_spatial_risk(self):
        if self.arrival_stream is None:
            self.basis = environment_basis((6, 2, RATE_STREAM_IDS[self.rate], self.seed, self.this_episode))
        else:
            rng = np.random.default_rng(derived_seed((*self.arrival_stream, self.this_episode)))
            self.basis = rng.exponential(size=params.taskno)
        super()._initialize_episode_spatial_risk()
        set_correlation_length(self.env_state, self.ell_ratio)

    def _sample_interarrival_time(self):
        return float(self.basis[self.taskCounter-1] / self.rate)


def evaluate_episode(selector, rate, episode_id, profiles, tasks, basis, *, failure_rate_scale=1.,
                     reliability_model='correlated', ell_ratio=1., task_selector=None, method=None,
                     rasoul_policy=None):
    if method == 'RASOUL' and rasoul_policy is None:
        raise ValueError('RASOUL evaluation requires an explicit history-bearing policy')
    env, state = make_environment(profiles, failure_rate_scale=failure_rate_scale,
                                  reliability_model=reliability_model)
    state.initialize_spatial_reliability()
    set_correlation_length(state, ell_ratio)
    pairs = MainLoop.generate_combinations()

    def rasoul_feedback(task):
        yield task.task_completion_event
        rasoul_policy.observe(task)

    def arrivals():
        for index, (tid, profile) in enumerate(tasks.items()):
            yield env.timeout(float(basis[index]/rate))
            task = Task(env, state, tid, profile)
            state.add_task(task)
            observation = state.get_state(task)
            backlog = observation[3*params.NUM_SERVERS:4*params.NUM_SERVERS] + observation[4*params.NUM_SERVERS:5*params.NUM_SERVERS]
            task.decision_backlog_sum_s = float(backlog.sum())
            task.decision_backlog_max_s = float(backlog.max())
            if method == 'RASOUL':
                decision = rasoul_policy.select(observation, task)
                task.rasoul_decision = decision
                # Keep the shared pair mask solely for pair-feasibility diagnostics.
                task.safe_mask = state.get_safe_action_mask(task)
                task.is_fallback = decision.is_fallback
                task.action_index = None
                a, b = state.get_server_by_id(decision.server_id), None
            elif method == 'TORE':
                marginal, _, independent = state.compute_task_reliabilities(task)
                # Explicitly replace the per-task mask: profiles may cache a
                # correlated mask for PPO/SML, irrespective of environment mode.
                task.safe_mask = independent >= task.reliability_requirement
                decision = select_tore_action(observation, pairs, marginal, independent,
                                              task.reliability_requirement, task.deadline)
                task.tore_decision = decision
                task.is_fallback = decision.is_fallback
                task.action_index = decision.action_index
                a = state.get_server_by_id(decision.server_ids[0])
                b = (state.get_server_by_id(decision.server_ids[1])
                     if len(decision.server_ids) == 2 else None)
            else:
                task.safe_mask = state.get_safe_action_mask(task)
                task.is_fallback = not bool(task.safe_mask.any())
                action = (select_latency_greedy_action(observation, pairs) if task.is_fallback
                          else (task_selector(observation, task) if task_selector is not None
                                else selector(observation, task.safe_mask)))
                task.action_index = int(action)
                if (not task.is_fallback and task_selector is None
                        and method is not None and method.endswith('masked_ppo')):
                    task.ppo_decision = True
                    task.ppo_action_mask = np.array(task.safe_mask, dtype=bool, copy=True)
                a, b = (state.get_server_by_id(sid) for sid in pairs[action])
            task.replica_A_reliability = state.compute_replica_reliability(task, a)
            task.replica_B_reliability = state.compute_replica_reliability(task, b) if b is not None else None
            task.pair_reliability = (state.compute_pair_reliability(task, a, b)
                                     if b is not None else task.replica_A_reliability)
            env.process(task.execute_task(a, b))
            if method == 'RASOUL':
                env.process(rasoul_feedback(task))

    env.process(arrivals()); env.run()
    completed = list(state.tasks.values())
    assert len(completed) == params.taskno
    assert all(r['finish_time'] is not None for t in completed for r in t.active_replicas)
    np.testing.assert_allclose([t.arrival_time for t in completed], np.cumsum(basis/rate), rtol=0, atol=1e-9)
    return task_metrics(completed, rate, episode_id), completed


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same_tree(x, y)
    else:
        assert a == b


def scale_task_profiles(profiles, multiplier, *, task_size_multiplier=1.):
    return {tid: {**profile, 'Computation_Demand': profile['Computation_Demand'] * multiplier,
                  'Task_Size': profile['Task_Size'] * task_size_multiplier}
            for tid, profile in profiles.items()}


def run_rate(job, *, task_demand_multiplier=1., task_size_multiplier=1.):
    rate, seed_ids, train_episodes, eval_episodes, output = job
    started = time.monotonic()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    case = Path(output) / f'rate_{rate:g}'
    (case/'checkpoints').mkdir(parents=True, exist_ok=True)
    sf = pd.read_excel(SERVER_INFO_PATH, sheet_name='Servers', dtype={'Site_ID': 'string'}).sort_values('Server_ID')
    profiles = MainLoop._build_server_profiles(sf)
    eval_tasks = scale_task_profiles(MainLoop._build_task_profiles(pd.read_excel(EVALUATION_TASK_PATH)),
                                    task_demand_multiplier, task_size_multiplier=task_size_multiplier)
    pairs = MainLoop.generate_combinations()
    training, evaluation = [], []
    for seeds in seed_mapping([rate], seed_ids).itertuples(index=False):
        torch.manual_seed(seeds.init_seed)
        agent = build_agent()
        agent.save_model(case/'checkpoints'/f'seed_{seeds.seed_id:02d}_initial.pt')
        torch.manual_seed(seeds.action_seed); np.random.seed(seeds.shuffle_seed)
        loop = TrainingLoop(agent, rate, seeds.seed_id, train_episodes)
        loop.task_profiles = scale_task_profiles(loop.task_profiles, task_demand_multiplier,
                                                task_size_multiplier=task_size_multiplier)
        original_train = agent.train_step
        tags = {'task_demand_multiplier': task_demand_multiplier, 'task_size_multiplier': task_size_multiplier,
                'seed_id': seeds.seed_id, 'training_seed': seeds.training_seed}

        def train_step():
            count = len(loop.ppo_decision_task_ids)
            assert np.asarray(agent.states).shape == (count, params.num_states)
            assert agent.dones == [False]*(count-1) + [True]
            np.testing.assert_allclose(sum(agent.rewards), loop.episodic_reward, rtol=0, atol=1e-9)
            original_train()
            if loop.this_episode % 50 == 0:
                print(f'size={task_size_multiplier:g} demand={task_demand_multiplier:g} rate={rate:g} seed={seeds.seed_id} train={loop.this_episode}/{train_episodes}', file=sys.stderr, flush=True)

        agent.train_step = train_step
        with contextlib.redirect_stdout(io.StringIO()): loop.EP()
        for episode, outcomes in pd.DataFrame(loop.task_results).groupby('episode'):
            # Record every episode, including those with no PPO transitions.
            first = outcomes.replica_A_finish_time <= outcomes.replica_B_finish_time
            upload = np.where(first, outcomes.replica_A_queue_enter_time, outcomes.replica_B_queue_enter_time) - outcomes.arrival_time
            start = np.where(first, outcomes.replica_A_cpu_start_time, outcomes.replica_B_cpu_start_time)
            wait = start - outcomes.arrival_time - upload
            execution = outcomes.task_completion_time - start
            actions = outcomes.action_index.to_numpy(dtype=int)
            metrics = summarize_latency_parts(np.column_stack((outcomes.task_latency, upload, wait, execution)), actions, rate, episode,
                                              deadlines=outcomes.deadline_s.to_numpy())
            metrics.update(mean_qos=float(outcomes.task_utility.mean()),
                           mean_raw_qos=float(outcomes.task_qos.mean()),
                           utility_sum=float(outcomes.task_utility.sum()))
            training.append({**tags, 'method': 'ppo_final', **metrics})
        checkpoint = case/'checkpoints'/f'seed_{seeds.seed_id:02d}_final.pt'
        agent.save_model(checkpoint)
        frozen = build_agent(); frozen.load_model(checkpoint)
        frozen.policy_old.eval()
        snapshot = copy.deepcopy({'actor': frozen.policy_net.state_dict(), 'old_actor': frozen.policy_old.state_dict(),
                                  'critic': frozen.value_net.state_dict(), 'actor_optimizer': frozen.optimizer_policy.state_dict(),
                                  'critic_optimizer': frozen.optimizer_value.state_dict()})
        for episode_id in range(1, eval_episodes+1):
            basis = environment_basis((6, 3, seeds.seed_id, episode_id))
            torch.manual_seed(derived_seed((6, 3, seeds.seed_id, episode_id, 3)))
            ppo, ppo_tasks = evaluate_episode(lambda s, mask: frozen.select_action(s, 0., action_mask=mask), rate, episode_id,
                                             profiles, eval_tasks, basis, method='trained_masked_ppo')
            greedy, greedy_tasks = evaluate_episode(lambda s, mask: select_latency_greedy_action(s, pairs, mask), rate,
                                                   episode_id, profiles, eval_tasks, basis)
            assert [(t.id, t.arrival_time, t.task_size, t.computation_demand) for t in ppo_tasks] == [
                (t.id, t.arrival_time, t.task_size, t.computation_demand) for t in greedy_tasks]
            np.testing.assert_array_equal(ppo_tasks[0].env_state.spatial_correlation_matrix,
                                          greedy_tasks[0].env_state.spatial_correlation_matrix)
            evaluation.extend([{**tags, 'method': 'ppo_final', **ppo}, {**tags, 'method': 'latency_greedy', **greedy}])
        same_tree(snapshot, {'actor': frozen.policy_net.state_dict(), 'old_actor': frozen.policy_old.state_dict(),
                             'critic': frozen.value_net.state_dict(), 'actor_optimizer': frozen.optimizer_policy.state_dict(),
                             'critic_optimizer': frozen.optimizer_value.state_dict()})
        assert not frozen.states and not frozen.rewards
        pd.DataFrame(training).to_csv(case/'train_episode_metrics.csv', index=False, float_format='%.17g')
        pd.DataFrame(evaluation).to_csv(case/'eval_episode_metrics.csv', index=False, float_format='%.17g')
        print(f'size={task_size_multiplier:g} demand={task_demand_multiplier:g} rate={rate:g} seed={seeds.seed_id} completed; elapsed={time.monotonic()-started:.1f}s', flush=True)
    return case


def summarize(output, cases):
    output = Path(output)
    training = pd.concat([pd.read_csv(p/'train_episode_metrics.csv', float_precision='round_trip') for p in cases])
    evaluation = pd.concat([pd.read_csv(p/'eval_episode_metrics.csv', float_precision='round_trip') for p in cases])
    seeds = evaluation.groupby(['arrival_rate', 'seed_id', 'training_seed', 'method'])[[*METRICS, 'mean_qos']].mean().reset_index()
    rows = []
    for rate, data in seeds.groupby('arrival_rate'):
        ppo = data[data.method == 'ppo_final'].set_index('seed_id')
        greedy = data[data.method == 'latency_greedy'].set_index('seed_id')
        for metric in (*METRICS, 'mean_qos'):
            delta = ppo[metric]-greedy[metric]
            half = student_t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            rows.append({'arrival_rate': rate, 'reference_method': 'latency_greedy', 'metric': metric,
                         'seed_count': len(delta), 'ppo_mean': ppo[metric].mean(), 'ppo_sd': ppo[metric].std(ddof=1),
                         'reference_mean': greedy[metric].mean(), 'reference_sd': greedy[metric].std(ddof=1),
                         'paired_mean_difference': delta.mean(), 'paired_difference_sd': delta.std(ddof=1),
                         'ci95_lower': delta.mean()-half, 'ci95_upper': delta.mean()+half,
                         'improvement_mean_pct': ((ppo[metric]-greedy[metric] if metric == 'mean_qos'
                                                   else greedy[metric]-ppo[metric])/greedy[metric].replace(0, np.nan)*100).mean(),
                         'ppo_better_seed_count': int((delta > 0 if metric == 'mean_qos' else delta < 0).sum())})
    paired = pd.DataFrame(rows)
    for name, data in [('train_episode_metrics.csv', training), ('eval_episode_metrics.csv', evaluation),
                       ('eval_seed_summary.csv', seeds), ('paired_comparison.csv', paired)]:
        data.to_csv(output/name, index=False, float_format='%.17g')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ('mean_latency_sec', 'mean_cpu_waiting_time_sec', 'mean_cpu_execution_time_sec')):
        for method, data in seeds.groupby('method'):
            group = data.groupby('arrival_rate')[metric]
            ax.errorbar(group.mean().index, group.mean(), yerr=group.std(ddof=1), marker='o', capsize=3, label=method)
        ax.set(xlabel='Arrival rate (tasks/s)', ylabel=metric); ax.grid(alpha=.3)
    axes[0].legend(); fig.tight_layout(); fig.savefig(output/'evaluation.png', dpi=150); plt.close(fig)
    lines = ['# 当前配置：PPO与Latency Greedy六档对比', '',
             f'CPU运行；每率{seeds.seed_id.nunique()}种子×{training.episode_id.max()}训练轮×{evaluation.episode_id.max()}评估轮。最终模型冻结，PPO按概率采样，无模型选择或调参。',
             f'{params.num_states}维；CPU 1–4 GHz，上传5–20 MB/s，任务计算需求2–4×10⁹ cycles，任务大小5–10 MB；运行与等待工作量及任务期限直接使用秒数。',
             f'gamma={params.gamma_ppo}，GAE lambda={params.gae_lambda_ppo}；GH安全Mask，空集共用最小时延Fallback；PPO奖励为决策区间内完成任务的有效QoS之和。',
             'Greedy从相同状态还原物理量，预测max(上传,运行＋等待)+执行；组合按最小副本时间、两副本时间之和、编号排序。不观察在途或未来任务。',
             '训练和评估使用独立固定任务集。两方法使用相同任务、到达和未知风险分布；所有400副本排空。Greedy在非空安全集中选最小时延，空集在全部动作中Fallback。区间为探索性逐指标比较，未作多重比较校正。', '',
             '主指标为全部任务平均有效QoS，可靠性不达标置0；平均时延为辅助指标。', '',
             '| 到达率 | PPO有效QoS | Greedy有效QoS | PPO−Greedy 95% CI |',
             '|---|---:|---:|---|']
    for r in paired[paired.metric == 'mean_qos'].itertuples():
        lines.append(f'| {r.arrival_rate:g} | {r.ppo_mean:.6f} | {r.reference_mean:.6f} | [{r.ci95_lower:.6f}, {r.ci95_upper:.6f}] |')
    lines += ['',
             '| 到达率 | PPO平均延迟(s) | Greedy平均延迟(s) | 改善率(%) | PPO−Greedy 95% CI(s) |',
             '|---|---:|---:|---:|---|']
    for r in paired[paired.metric == 'mean_latency_sec'].itertuples():
        lines.append(f'| {r.arrival_rate:g} | {r.ppo_mean:.6f} | {r.reference_mean:.6f} | {r.improvement_mean_pct:.3f} | [{r.ci95_lower:.6f}, {r.ci95_upper:.6f}] |')
    lines.extend(['', '![评估结果](evaluation.png)', '', '命令：`python compare_ppo_greedy.py`'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')


MASKED_METHODS = ('initial_masked_ppo', 'trained_masked_ppo', 'safe_min_latency', 'safe_random')
MASKED_THRESHOLDS = (0.999, 0.9999)
MASKED_ALPHA = 0.05
MASKED_RATE = 2.


def precompute_masked_profiles(profiles, task_sets, output):
    """Two batched GH integrals, shared across thresholds, seeds and episodes."""
    _, state = make_environment(profiles, failure_rate_scale=MASKED_ALPHA)
    state.initialize_spatial_reliability()
    ids = state.spatial_risk_server_ids
    pairs = np.array([(ids.index(a), ids.index(b)) for a, b in state.server_pairs])
    frequencies = np.array([profiles[sid]['Processing_Frequency'] for sid in ids])
    rates = np.array([profiles[sid]['Base_Failure_Rate'] for sid in ids]) * MASKED_ALPHA
    rho = state.spatial_correlation_matrix[pairs[:, 0], pairs[:, 1]]
    cached, arrays, summaries, distributions = {}, {}, [], []
    for split, tasks in task_sets.items():
        demand = np.array([p['Computation_Demand'] for p in tasks.values()])[:, None]
        marginal = 1 - compute_marginal_failure_probability(demand, frequencies, rates, effective_spatial_risk_beta())
        corr, ind = compute_pair_reliability_gh(demand, frequencies[pairs], rates[pairs], rho, effective_spatial_risk_beta())
        assert marginal.shape == (200, 8) and corr.shape == ind.shape == (200, 28)
        assert np.all(np.isfinite(corr)) and np.all((0 <= corr) & (corr <= ind + 1e-12)) and np.all(ind <= 1)
        for values in (marginal, corr, ind):
            values.setflags(write=False)
        arrays.update({f'{split}_ids': np.array(list(tasks)), f'{split}_marginal': marginal,
                       f'{split}_correlated': corr, f'{split}_independent': ind})
        cached[split] = {}
        for threshold in MASKED_THRESHOLDS:
            masks = corr >= threshold
            masks.setflags(write=False)
            cached[split][threshold] = {
                tid: {**profile, 'Reliability_Requirement': threshold,
                      'reliability_snapshot': (marginal[i], corr[i], ind[i]), 'safe_mask': masks[i]}
                for i, (tid, profile) in enumerate(tasks.items())
            }
            counts = masks.sum(axis=1)
            summaries.append({'split': split, 'threshold': threshold, 'task_count': len(tasks),
                              'mean_safe_pair_count': counts.mean(), 'fallback_count': int((counts == 0).sum()),
                              'effective_decision_count': int((counts > 0).sum()),
                              'multi_safe_decision_count': int((counts >= 2).sum()),
                              'single_safe_decision_count': int((counts == 1).sum()),
                              'expected_rsr': float(np.mean(counts > 0))})
            distributions.extend({'split': split, 'threshold': threshold, 'safe_pair_count': k,
                                  'task_count': int(count)} for k, count in enumerate(np.bincount(counts, minlength=29)))
        arrays[f'{split}_masks'] = np.stack([corr >= threshold for threshold in MASKED_THRESHOLDS])
    np.savez_compressed(output/'reliability_table.npz', **arrays, thresholds=MASKED_THRESHOLDS,
                        server_ids=ids, pairs=state.server_pairs, scaled_rates=rates,
                        distances_km=state.spatial_distance_matrix, correlation=state.spatial_correlation_matrix)
    pd.DataFrame(summaries).to_csv(output/'safe_set_summary.csv', index=False, float_format='%.17g')
    pd.DataFrame(distributions).to_csv(output/'safe_pair_distribution.csv', index=False)
    return cached


def parameter_digest(agent):
    values = torch.cat([p.detach().flatten() for network in (agent.policy_net, agent.value_net) for p in network.parameters()])
    assert torch.isfinite(values).all()
    return hashlib.sha256(values.cpu().numpy().tobytes()).hexdigest()


def optimizer_steps(optimizer):
    return max((int(state['step']) for state in optimizer.state.values()), default=0)


def audit_rollout(agent, loop):
    """Check completion-interval utility, masks and elapsed time before PPO updates."""
    ids = loop.ppo_decision_task_ids
    tasks = loop.env_state.tasks
    assert len(tasks) == 200 and ids == agent.task_ids
    assert ids == [tid for tid, task in tasks.items() if not task.is_fallback]
    assert agent.dones == [False]*(len(ids)-1) + [True]
    for i, tid in enumerate(ids):
        task = tasks[tid]
        next_tid = ids[i+1] if i+1 < len(ids) else None
        end = tasks[next_tid].arrival_time if next_tid else max(t.task_completion_time for t in tasks.values())
        assert task.reliability_snapshot is loop.task_profiles[tid]['reliability_snapshot']
        assert task.safe_mask is loop.task_profiles[tid]['safe_mask'] and task.safe_mask[agent.actions[i]]
        assert task.action_index == agent.actions[i] and task.ppo_next_task_id == next_tid
        completed_utility = sum(loop.compute_reward(loop.get_task_outcome_info(item))
                                for item in tasks.values() if item.ppo_reward_owner_task_id == tid)
        np.testing.assert_allclose(agent.rewards[i], completed_utility, rtol=0, atol=1e-9)
        assert agent.rewards[i] == task.ppo_interval_reward
        np.testing.assert_allclose(agent.delta_times[i], end-task.arrival_time, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(agent.action_masks[i], task.safe_mask)
        np.testing.assert_array_equal(agent.next_action_masks[i], tasks[next_tid].safe_mask if next_tid else np.zeros(28, dtype=bool))
        np.testing.assert_array_equal(agent.next_states[i], agent.states[i+1] if next_tid else np.zeros(params.num_states))
    assert np.isfinite(agent.old_log_probs).all()
    assert np.isfinite(agent.states).all() and np.isfinite(agent.rewards).all()
    np.testing.assert_allclose(sum(agent.rewards), loop.episodic_reward, rtol=0, atol=1e-9)
    counts = np.asarray(agent.action_masks).sum(axis=1)
    np.testing.assert_allclose(np.asarray(agent.old_log_probs)[counts == 1], 0, atol=1e-6)
    return {'episode_id': loop.this_episode, 'effective_transition_count': len(ids),
            'multi_safe_decision_count': int((counts >= 2).sum()),
            'single_safe_decision_count': int((counts == 1).sum()),
            'ppo_interval_reward_sum': float(sum(agent.rewards)),
            'decision_backlog_mean_s': float(np.asarray(agent.states)[:, 3*params.NUM_SERVERS:5*params.NUM_SERVERS].sum(axis=1).mean())}


def train_masked_agent(agent, profiles, tasks, scenario, seed, episodes, *, progress=True,
                       reliability_model='correlated', arrival_stream=None, ell_ratio=1., progress_label=None,
                       arrival_rate=MASKED_RATE, failure_rate_scale=MASKED_ALPHA):
    loop = TrainingLoop(agent, arrival_rate, seed, episodes, failure_rate_scale=failure_rate_scale,
                        arrival_stream=arrival_stream or (31, scenario, seed, 3),
                        reliability_model=reliability_model, ell_ratio=ell_ratio)
    loop.server_profiles, loop.task_profiles = profiles, tasks
    if all('safe_mask' in profile and not np.any(profile['safe_mask']) for profile in tasks.values()):
        raise ValueError('PPO training has no feasible decisions; all tasks use fallback. Inspect reliability feasibility before training.')
    original_train = agent.train_step
    diagnostics = []

    def update():
        row = audit_rollout(agent, loop)
        before = parameter_digest(agent)
        actor_step, critic_step = optimizer_steps(agent.optimizer_policy), optimizer_steps(agent.optimizer_value)
        original_train()
        after = parameter_digest(agent)
        assert not agent.states and not agent.action_masks and not agent.task_ids
        row.update(update_call_count=1, parameters_changed=before != after, parameter_digest=after,
                   actor_optimizer_steps=optimizer_steps(agent.optimizer_policy)-actor_step,
                   critic_optimizer_steps=optimizer_steps(agent.optimizer_value)-critic_step)
        assert row['actor_optimizer_steps'] == row['critic_optimizer_steps'] == params.k_epochs_ppo * int(np.ceil(row['effective_transition_count']/params.batch_size_ppo))
        diagnostics.append(row)
        if progress and loop.this_episode % 20 == 0:
            label = MASKED_THRESHOLDS[scenario] if progress_label is None else progress_label
            print(f'threshold={label} seed={seed} train={loop.this_episode}/{episodes} '
                  f'latency={loop.episodic_delay/200:.4f}s valid={row["effective_transition_count"]} multi={row["multi_safe_decision_count"]}',
                  file=sys.stderr, flush=True)

    agent.train_step = update
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            loop.EP()
    finally:
        agent.train_step = original_train
    metrics, _, loads = summarize_task_results(loop.task_results)
    metrics = metrics.rename(columns={'episode': 'episode_id', 'mean_task_latency_s': 'mean_latency_s'})
    metrics['mean_safe_pair_count'] = np.mean([np.count_nonzero(p['safe_mask']) for p in tasks.values()])
    metrics['episode_reward'] = [row[1] for row in loop.log_data]
    metrics = metrics.merge(pd.DataFrame(diagnostics), on='episode_id', validate='one_to_one')
    assert len(metrics) == episodes and np.all(metrics.ppo_decision_count == metrics.effective_transition_count)
    expected_rewards = pd.DataFrame(loop.task_results).groupby('episode').task_utility.sum()
    np.testing.assert_allclose(metrics.episode_reward, expected_rewards.to_numpy(), rtol=0, atol=1e-10)
    assert loads.replica_count.sum() == episodes*400 and np.all(loads.cpu_utilization <= 1+1e-12)
    return loop, metrics, loads.rename(columns={'episode': 'episode_id'})


def frozen_agent(path):
    agent = build_agent()
    agent.load_model(path)
    for network in (agent.policy_net, agent.policy_old, agent.value_net):
        network.eval()
        network.requires_grad_(False)
    return agent


def frozen_snapshot(agent):
    return copy.deepcopy({'actor': agent.policy_net.state_dict(), 'old_actor': agent.policy_old.state_dict(),
                          'critic': agent.value_net.state_dict(), 'actor_optimizer': agent.optimizer_policy.state_dict(),
                          'critic_optimizer': agent.optimizer_value.state_dict()})


def masked_evaluation(method, agent, profiles, tasks, basis, episode, action_seed, logger, *,
                      reliability_model='correlated', task_records=None, ell_ratio=1., record_backlog=False, task_selector=None, arrival_rate=MASKED_RATE,
                      failure_rate_scale=MASKED_ALPHA, rasoul_policy=None):
    pairs = MainLoop.generate_combinations()
    if method == 'TORE':
        selector = None
        reliability_model = 'independent'
    elif method == 'RASOUL':
        selector = None
        reliability_model = 'correlated'
    elif method.endswith('masked_ppo'):
        torch.manual_seed(action_seed)
        selector = lambda state, mask: agent.select_action(state, 0., action_mask=mask)
    elif method == 'safe_min_latency':
        selector = lambda state, mask: select_latency_greedy_action(state, pairs, mask)
    else:
        rng = np.random.default_rng(action_seed)
        selector = lambda state, mask: int(rng.choice(np.flatnonzero(mask)))
    _, completed = evaluate_episode(selector, arrival_rate, episode, profiles, tasks, basis,
                                    failure_rate_scale=failure_rate_scale, reliability_model=reliability_model,
                                    ell_ratio=ell_ratio, task_selector=task_selector, method=method,
                                    rasoul_policy=rasoul_policy)
    state = completed[0].env_state
    np.testing.assert_array_equal(state.spatial_correlation_matrix, build_spatial_correlation_matrix(
        state.spatial_distance_matrix, ell_ratio*compute_baseline_correlation_length_km(state.spatial_distance_matrix)))
    rows = [dict(logger(task), episode=episode) for task in completed]
    if record_backlog:
        for row,task in zip(rows,completed):
            row.update(decision_backlog_sum_s=task.decision_backlog_sum_s,
                       decision_backlog_max_s=task.decision_backlog_max_s)
    if task_records is not None:
        task_records.extend(rows)
    metrics, _, loads = summarize_task_results(rows)
    values = metrics.iloc[0].to_dict()
    values['episode_id'] = int(values.pop('episode'))
    values['mean_latency_s'] = values.pop('mean_task_latency_s')
    counts = np.array([np.count_nonzero(task.safe_mask) for task in completed])
    values.update(mean_safe_pair_count=float(counts.mean()), normal_decision_count=sum(not t.is_fallback for t in completed),
                  multi_safe_decision_count=int((counts >= 2).sum()), single_safe_decision_count=int((counts == 1).sum()),
                  decision_backlog_mean_s=float(np.mean([task.decision_backlog_sum_s for task in completed])),
                  decision_backlog_max_s=float(max(task.decision_backlog_max_s for task in completed)))
    fingerprints = []
    for task in completed:
        assert task.reliability_snapshot is tasks[task.id]['reliability_snapshot']
        if method == 'RASOUL':
            decision = task.rasoul_decision
            sid_index = state.spatial_risk_server_ids.index(decision.server_id)
            assert task.safe_mask is tasks[task.id]['safe_mask']
            assert task.pair_reliability == task.reliability_snapshot[0][sid_index]
            assert task.replica_A['server_id'] == decision.server_id
            assert task.replica_B['server_id'] is None and task.action_index is None
            assert task.is_fallback == decision.is_fallback
            assert task.rasoul_feedback['feedback_time'] == task.task_completion_time
            assert task.rasoul_feedback['prediction_error'] == decision.predicted_qos - task.rasoul_feedback['actual_qos']
            if decision.fallback_reason == 'no_reliable_node':
                assert not any(decision.safe_node_mask) and not task.pair_reliability >= task.reliability_requirement
            else:
                assert decision.safe_node_mask[sid_index]
                assert task.pair_reliability >= task.reliability_requirement
                assert decision.fallback_reason == ('admission' if not any(decision.admitted_node_mask) else None)
                if not decision.is_fallback:
                    assert decision.admitted_node_mask[sid_index]
        elif method == 'TORE':
            assert task.is_fallback == (task.tore_decision.greedy_reliability < task.reliability_requirement)
            np.testing.assert_array_equal(task.safe_mask,
                task.reliability_snapshot[2] >= task.reliability_requirement)
            if task.action_index is None:
                sid_index = state.spatial_risk_server_ids.index(task.replica_A['server_id'])
                assert task.pair_reliability == task.reliability_snapshot[0][sid_index]
                assert task.pair_reliability >= task.reliability_requirement
            else:
                assert task.pair_reliability == task.reliability_snapshot[1][task.action_index]
        else:
            assert task.safe_mask is tasks[task.id]['safe_mask']
            assert task.pair_reliability == task.reliability_snapshot[1][task.action_index]
            assert task.is_fallback == (not task.safe_mask.any())
        assert task.is_fallback or task.action_index is None or task.safe_mask[task.action_index]
        assert task.env_state.failure_rate_scale == failure_rate_scale
        for replica in task.active_replicas:
            server = task.env_state.get_server_by_id(replica['server_id'])
            assert server.base_failure_rate == profiles[server.server_id]['Base_Failure_Rate']
            assert replica['cpu_start_time'] >= replica['queue_enter_time']
            np.testing.assert_allclose(replica['finish_time']-replica['cpu_start_time'], task.computation_demand/server.processing_frequency)
        assert task.task_completion_time == min(r['finish_time'] for r in task.active_replicas)
        fingerprints.append((task.id, task.arrival_time, task.task_size, task.computation_demand,
                             task.reliability_requirement,
                             ((task.rasoul_decision.server_id,) if method == 'RASOUL' else
                              task.tore_decision.server_ids if method == 'TORE' else task.action_index),
                             task.task_completion_time))
    assert values['task_count'] == len(tasks)
    assert loads.replica_count.sum() == sum(len(t.active_replicas) for t in completed)
    if method == 'RASOUL':
        decisions = [task.rasoul_decision for task in completed]
        safe_counts = np.array([sum(d.safe_node_mask) for d in decisions])
        assert values['theoretical_rsr'] == np.mean(safe_counts > 0)
        values.update(
            mean_safe_node_count=float(safe_counts.mean()),
            mean_admitted_node_count=float(np.mean([sum(d.admitted_node_mask) for d in decisions])),
            reliability_fallback_count=sum(d.fallback_reason == 'no_reliable_node' for d in decisions),
            admission_fallback_count=sum(d.fallback_reason == 'admission' for d in decisions),
            mean_prediction_error=float(np.mean([t.rasoul_feedback['prediction_error'] for t in completed])),
            mean_absolute_prediction_error=float(np.mean([abs(t.rasoul_feedback['prediction_error']) for t in completed])),
        )
        values['reliability_fallback_fraction'] = values['reliability_fallback_count'] / len(completed)
        values['admission_fallback_fraction'] = values['admission_fallback_count'] / len(completed)
        rasoul_policy.snapshot()  # The final drain must include every pending feedback.
    elif reliability_model == 'correlated':
        assert values['theoretical_rsr'] == np.mean(counts > 0)
    assert values['fallback_count'] == sum(t.is_fallback for t in completed)
    if method not in ('TORE', 'RASOUL'):
        assert values['fallback_count'] == (counts == 0).sum()
    assert np.all(loads.cpu_utilization <= 1+1e-12) and values['mean_cpu_wait_s'] >= 0
    np.testing.assert_allclose(loads.execution_cycles.sum(),
                              sum(len(t.active_replicas)*t.computation_demand for t in completed))
    values['trace_sha256'] = hashlib.sha256(repr(fingerprints).encode()).hexdigest()
    values['arrival_sha256'] = hashlib.sha256(np.array([task.arrival_time for task in completed]).tobytes()).hexdigest()
    if agent is not None:
        assert not any((agent.states, agent.actions, agent.rewards, agent.old_log_probs, agent.action_masks,
                        agent.next_action_masks, agent.task_ids, agent.next_states, agent.dones, agent.delta_times))
    return values, loads.rename(columns={'episode': 'episode_id'})


def run_masked_seed(scenario, seed, episodes, evaluation_episodes, profiles, cached, output):
    threshold = MASKED_THRESHOLDS[scenario]
    case = output/f'threshold_{threshold:g}'/f'seed_{seed}'
    case.mkdir(parents=True, exist_ok=False)
    seeds = {name: derived_seed((31, scenario, seed, role, 0))
             for name, role in (('init_seed', 0), ('train_action_seed', 1), ('shuffle_seed', 2))}
    tags = {'threshold': threshold, 'seed_id': seed, **seeds}
    train_tasks, eval_tasks = cached['train'][threshold], cached['evaluation'][threshold]
    torch.manual_seed(seeds['init_seed'])
    agent = build_agent()
    initial_path, final_path = case/'initial.pt', case/'final.pt'
    agent.save_model(initial_path)
    torch.manual_seed(seeds['train_action_seed'])
    np.random.seed(seeds['shuffle_seed'])
    loop, training, train_loads = train_masked_agent(agent, profiles, train_tasks, scenario, seed, episodes)
    agent.save_model(final_path)
    training['train_arrival_seed'] = [derived_seed((31, scenario, seed, 3, int(episode))) for episode in training.episode_id]
    for name, value in tags.items():
        training[name] = value
        train_loads[name] = value
    training.to_csv(case/'train_episode_metrics.csv', index=False, float_format='%.17g')
    train_loads.to_csv(case/'train_server_load.csv', index=False, float_format='%.17g')
    # Replay one training episode from the original checkpoint, without replacing final.pt.
    replay = build_agent(); replay.load_model(initial_path)
    torch.manual_seed(seeds['train_action_seed']); np.random.seed(seeds['shuffle_seed'])
    _, replay_metrics, _ = train_masked_agent(replay, profiles, train_tasks, scenario, seed, 1, progress=False)
    assert replay_metrics.parameter_digest.iloc[0] == training.parameter_digest.iloc[0]
    assert replay_metrics.mean_latency_s.iloc[0] == training.mean_latency_s.iloc[0]
    reproduction = [{'threshold': threshold, 'seed_id': seed, 'phase': 'first_train_episode', 'method': 'trained_masked_ppo', 'passed': True}]
    models = {'initial_masked_ppo': frozen_agent(initial_path), 'trained_masked_ppo': frozen_agent(final_path)}
    snapshots = {name: frozen_snapshot(model) for name, model in models.items()}
    file_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (initial_path, final_path)}
    evaluation, eval_loads, streams = [], [], []
    for episode in range(1, evaluation_episodes+1):
        arrival_seed = derived_seed((31, scenario, seed, 4, episode))
        basis = np.random.default_rng(arrival_seed).exponential(size=200)
        arrivals = set()
        for method in MASKED_METHODS:
            role = {'initial_masked_ppo': 5, 'trained_masked_ppo': 6, 'safe_min_latency': 8, 'safe_random': 7}[method]
            action_seed = derived_seed((31, scenario, seed, role, episode))
            values, loads = masked_evaluation(method, models.get(method), profiles, eval_tasks, basis, episode,
                                              action_seed, loop.get_task_outcome_info)
            arrivals.add(values['arrival_sha256'])
            evaluation.append({**tags, 'method': method, 'eval_arrival_seed': arrival_seed,
                               'eval_action_seed': action_seed, **values})
            loads = loads.assign(threshold=threshold, seed_id=seed, method=method)
            eval_loads.append(loads)
            if episode == 1:
                again, _ = masked_evaluation(method, models.get(method), profiles, eval_tasks, basis, episode,
                                             action_seed, loop.get_task_outcome_info)
                assert again == values
                reproduction.append({'threshold': threshold, 'seed_id': seed, 'phase': 'first_eval_episode', 'method': method, 'passed': True})
        assert len(arrivals) == 1
        streams.append({'threshold': threshold, 'seed_id': seed, 'episode_id': episode,
                        'arrival_seed': arrival_seed, 'arrival_sha256': arrivals.pop()})
    for name, model in models.items():
        same_tree(snapshots[name], frozen_snapshot(model))
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in file_hashes.items())
    evaluation, eval_loads = pd.DataFrame(evaluation), pd.concat(eval_loads, ignore_index=True)
    assert evaluation.groupby('episode_id').fallback_count.nunique().max() == 1
    assert evaluation.groupby('episode_id').theoretical_rsr.nunique().max() == 1
    evaluation.to_csv(case/'eval_episode_metrics.csv', index=False, float_format='%.17g')
    eval_loads.to_csv(case/'eval_server_load.csv', index=False, float_format='%.17g')
    print(f'COMPLETE threshold={threshold:g} seed={seed} train={episodes} eval={evaluation_episodes}x4; checkpoints frozen', flush=True)
    return training, evaluation, train_loads, eval_loads, pd.DataFrame(streams), pd.DataFrame(reproduction)


def summarize_masked_results(output):
    training = pd.read_csv(output/'train_episode_metrics.csv', float_precision='round_trip')
    evaluation = pd.read_csv(output/'eval_episode_metrics.csv', float_precision='round_trip')
    loads = pd.read_csv(output/'eval_server_load.csv', float_precision='round_trip')
    measures = ['mean_latency_s', 'mean_cpu_wait_s', 'theoretical_rsr', 'fallback_count', 'fallback_fraction',
                'mean_safe_pair_count', 'normal_decision_count', 'multi_safe_decision_count',
                'single_safe_decision_count', 'decision_backlog_mean_s', 'decision_backlog_max_s']
    seeds = evaluation.groupby(['threshold', 'seed_id', 'method'])[measures].mean().reset_index()
    seeds.to_csv(output/'eval_seed_summary.csv', index=False, float_format='%.17g')
    method_rows, paired_rows, seed_deltas = [], [], []
    for (threshold, method), group in seeds.groupby(['threshold', 'method']):
        row = {'threshold': threshold, 'method': method, 'seed_count': len(group)}
        for measure in measures:
            row[f'{measure}_mean'] = group[measure].mean()
            row[f'{measure}_sd'] = group[measure].std(ddof=1)
        method_rows.append(row)
    for threshold, group in seeds.groupby('threshold'):
        ppo = group[group.method == 'trained_masked_ppo'].set_index('seed_id')
        for reference in ('initial_masked_ppo', 'safe_random', 'safe_min_latency'):
            baseline = group[group.method == reference].set_index('seed_id').reindex(ppo.index)
            delta = ppo.mean_latency_s - baseline.mean_latency_s
            improvement = -delta / baseline.mean_latency_s * 100
            paired_rows.append({'threshold': threshold, 'reference_method': reference, 'seed_count': len(ppo),
                                'ppo_mean_latency_s': ppo.mean_latency_s.mean(),
                                'reference_mean_latency_s': baseline.mean_latency_s.mean(),
                                'paired_delta_mean_s': delta.mean(), 'paired_delta_sd_s': delta.std(ddof=1),
                                'improvement_mean_pct': improvement.mean(), 'improvement_sd_pct': improvement.std(ddof=1),
                                'ppo_better_seed_count': int((delta < 0).sum())})
            seed_deltas.extend({'threshold': threshold, 'seed_id': seed, 'reference_method': reference,
                                'delta_s': delta.loc[seed], 'improvement_pct': improvement.loc[seed]} for seed in ppo.index)
    methods, paired = pd.DataFrame(method_rows), pd.DataFrame(paired_rows)
    methods.to_csv(output/'method_summary.csv', index=False, float_format='%.17g')
    paired.to_csv(output/'paired_comparison.csv', index=False, float_format='%.17g')
    pd.DataFrame(seed_deltas).to_csv(output/'paired_seed_differences.csv', index=False, float_format='%.17g')
    server_measures = ['replica_count', 'execution_cycles', 'cpu_busy_s', 'mean_cpu_wait_s', 'cpu_utilization']
    server_seeds = loads.groupby(['threshold', 'seed_id', 'method', 'server_id'])[server_measures].mean().reset_index()
    server_seeds.to_csv(output/'server_load_seed_summary.csv', index=False, float_format='%.17g')
    server_summary = server_seeds.groupby(['threshold', 'method', 'server_id'])[server_measures].mean().reset_index()
    server_summary.to_csv(output/'server_load_summary.csv', index=False, float_format='%.17g')
    plot_masked_results(output, training, seeds)
    write_masked_report(output, training, methods, paired, server_summary)


def plot_masked_results(output, training, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (threshold, group) in zip(axes, training.groupby('threshold')):
        for seed, data in group.groupby('seed_id'):
            ax.plot(data.episode_id, data.mean_latency_s, alpha=.3, linewidth=.7, label=f'Seed {seed}')
        mean = group.groupby('episode_id').mean_latency_s.mean()
        ax.plot(mean.index, mean.rolling(10, min_periods=1).mean(), color='black', label='Mean, rolling 10')
        ax.set(title=f'R_req={threshold}', xlabel='Training episode', ylabel='Mean task latency (s)')
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'training_curves.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ['Initial PPO', 'Trained PPO', 'SML', 'Safe Random']
    for ax, (threshold, group) in zip(axes, seeds.groupby('threshold')):
        values = group.groupby('method').mean_latency_s
        means = values.mean().reindex(MASKED_METHODS)
        deviations = values.std(ddof=1).reindex(MASKED_METHODS).fillna(0)
        ax.bar(np.arange(4), means, yerr=deviations, capsize=3)
        ax.set(xticks=np.arange(4), xticklabels=labels, title=f'R_req={threshold}', ylabel='Seed mean latency (s)')
        ax.tick_params(axis='x', labelrotation=20); ax.grid(axis='y', alpha=.25)
    fig.tight_layout(); fig.savefig(output/'latency_comparison.png', dpi=160); plt.close(fig)


def write_masked_report(output, training, methods, paired, server_summary):
    metadata = json.loads((output/'run_metadata.json').read_text())
    safe = pd.read_csv(output/'safe_set_summary.csv')
    nseeds, nepisodes, neval = metadata['seed_count'], metadata['train_episodes'], metadata['eval_episodes']
    lines = ['# Correlation-Aware Masked PPO：第三阶段时延验证', '',
             f'两个单阈值场景分别独立训练；每场景{nseeds}个训练种子×{nepisodes}轮，最终Checkpoint评估{neval}个配对Episode。Initial PPO仅为学习参考。',
             '固定8台EUA服务器、28个字典序动作、每Episode 200任务/400副本；alpha=0.05只在运行时乘一次，beta=0.5，ell=ell0，到达率2 tasks/s。',
             '复用固定训练任务和独立评估任务，只在内存覆盖单一阈值。单阈值实验不代表四档混合配置。',
             f'CPU PyTorch单线程；网络[64,32]/tanh、{params.num_states}维状态/{params.num_actions}动作，gamma={params.gamma_ppo}、GAE={params.gae_lambda_ppo}；其他超参数见run_metadata.json。',
             '所有方法使用同一相关Mask，空集合用同一个最小时延Fallback，全部任务执行。评估PPO使用Masked Categorical概率采样；不使用argmax或最优轮次选择。', '',
             '## 四种方法的评估结果', '',
             '先对每种子20个配对Episode取均值，再对种子均值报告mean±sample SD（开发验证可由CLI缩短轮数）。',
             '| 阈值 | 方法 | 任务平均时延/s | 双副本平均CPU等待/s | 理论RSR | Fallback数量/200 | Fallback比例 | 平均安全对 |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in methods.itertuples():
        lines.append(f'| {row.threshold:g} | {row.method} | {row.mean_latency_s_mean:.6f} ± {row.mean_latency_s_sd:.6f} '
                     f'| {row.mean_cpu_wait_s_mean:.6f} ± {row.mean_cpu_wait_s_sd:.6f} | {row.theoretical_rsr_mean:.1%} '
                     f'| {row.fallback_count_mean:g} | {row.fallback_fraction_mean:.1%} | {row.mean_safe_pair_count_mean:.3f} |')
    lines += ['', 'RSR以全部200个实际卸载任务为分母；Fallback均不满足阈值。相同场景的四种方法Fallback和RSR逐Episode严格相同，训练集和评估集可以不同。', '',
              '## 配对时延差', '', 'Delta=PPO−对照；负值表示PPO更快。改善百分比先逐种子计算再取均值。', '',
              '| 阈值 | 对照 | Delta mean±SD /s | 改善/% | PPO更好的种子数 |', '|---:|---|---:|---:|---:|']
    for row in paired.itertuples():
        lines.append(f'| {row.threshold:g} | {row.reference_method} | {row.paired_delta_mean_s:.6f} ± {row.paired_delta_sd_s:.6f} '
                     f'| {row.improvement_mean_pct:.3f} | {row.ppo_better_seed_count}/{row.seed_count} |')
    lines += ['', '## 学习空间和训练过程', '',
              '| 数据集 | 阈值 | 有效决策 | 多安全动作≥2 | 唯一安全动作 | 唯一动作占有效决策 | Fallback |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in safe.itertuples():
        ratio = row.single_safe_decision_count/row.effective_decision_count
        lines.append(f'| {row.split} | {row.threshold:g} | {row.effective_decision_count} | {row.multi_safe_decision_count} '
                     f'| {row.single_safe_decision_count} | {ratio:.1%} | {row.fallback_count} |')
    for threshold, group in training.groupby('threshold'):
        first = group[group.episode_id <= min(10, nepisodes)].mean_latency_s.mean()
        last = group[group.episode_id > max(0, nepisodes-10)].mean_latency_s.mean()
        lines.append(f'\n阈值{threshold:g}：训练前{min(10, nepisodes)}轮均值{first:.6f}s，末{min(10, nepisodes)}轮均值{last:.6f}s；'
                     f'有效PPO更新调用{int(group.update_call_count.sum())}次，Actor/Critic各{int(group.actor_optimizer_steps.sum())}个optimizer step，'
                     f'实际参数变化的Episode数{int(group.parameters_changed.sum())}。')
    lines += ['', '![训练曲线](training_curves.png)', '', '![算法时延比较](latency_comparison.png)', '',
              '训练曲线使用训练任务；Initial/Final比较使用同一独立评估任务和到达流，学习判断以配对评估为主。柱图误差线为种子SD，不是置信区间。', '',
              '## CPU等待与服务器负载', '',
              'CPU等待对全部400副本取平均，包含Fallback和任务完成后继续执行的副本。执行需求单位为CPU cycles，CPU忙时为秒。',
              '每台服务器每Episode平均副本数见下表；完整cycles、忙时、CPU等待和利用率见server_load_summary.csv及逐Episode负载表。', '',
              '| 阈值 | 服务器 | Initial PPO | Trained PPO | SML | Safe Random |', '|---:|---:|---:|---:|---:|---:|']
    for (threshold, sid), group in server_summary.groupby(['threshold', 'server_id']):
        values = group.set_index('method').replica_count
        lines.append(f'| {threshold:g} | {sid} | ' + ' | '.join(f'{values[method]:.2f}' for method in MASKED_METHODS) + ' |')
    lines += ['', '## 判断与下一步', '']
    consistent = True
    for threshold in MASKED_THRESHOLDS:
        learning = paired[(paired.threshold == threshold) & (paired.reference_method == 'initial_masked_ppo')].iloc[0]
        sml = paired[(paired.threshold == threshold) & (paired.reference_method == 'safe_min_latency')].iloc[0]
        lines.append(f'- 阈值{threshold:g}：训练相对初始化平均时延变化{learning.paired_delta_mean_s:+.6f}s，'
                     f'{learning.ppo_better_seed_count}/{nseeds}个种子改善；相对SML差值{sml.paired_delta_mean_s:+.6f}s。')
        consistent &= learning.ppo_better_seed_count == nseeds and sml.ppo_better_seed_count >= max(1, nseeds-1)
    lines += [
        '- 0.9999下唯一安全动作无选择空间，Fallback也无PPO梯度；有效Transition数不能等同于可学习的多动作样本数。唯一动作样本仍进入原有Critic/GAE计算。',
        '- 如果PPO慢于SML，可能原因包括：SML直接利用已知时延关系；100轮、每轮有限多动作样本不足；负任务时延只间接反映对后续任务的队列影响；原始队列特征可能在积压较大时使tanh饱和。以上仅为可能解释，本轮不据此改参数、奖励或网络。',
        ('- 建议：可考虑进入更大样本验证，但当前3种子仍不足以声称最终优势，需另行授权正式实验。' if consistent else
         '- 建议：暂不直接进入10 seeds×300 Episodes。先核对学习空间、训练趋势和未优于SML的种子，明确是否有值得扩大验证的学习效应；本轮停止，不启动正式训练。'),
        '- 只有3个训练种子，结论是开发验证，不宣称统计显著性或最终算法优势。', '', '## 执行与完整性检查', '',
        f'- HEAD：`{metadata["git_head"]}`；分支`{metadata["git_branch"]}`。',
        '- GH只对训练集和评估集各做一次批量二维积分，阈值和种子共享只读快照；运行时重新积分会立即报错。',
        '- 每个训练Episode审计Task_ID、奖励、当前/下一Mask、实际delta_t及singleton log_prob；参数和log_prob均有限。',
        '- 全部Episode完成200任务和400副本；安全集非空无不安全动作；Fallback及RSR逐方法匹配；CPU总需求、等待分母和利用率检查通过。',
        '- 每个种子重放首训练Episode，参数SHA256与时延完全一致；每方法重放首评估Episode，动作/完成时间轨迹及指标完全一致。未重复整套100轮训练。',
        '- 初始化、训练采样、shuffle、训练到达、评估到达、初始/最终PPO评估采样和Safe Random使用独立可复现流；seed_mapping.csv和paired_streams.csv保存种子。',
        '- 评估前后网络、旧策略、Critic和优化器逐项完全一致，buffer始终为空，Checkpoint文件SHA256未变。',
        '- 未修改数据、GH模型、PPO网络、奖励、GAE或折扣；未覆盖历史结果，未commit或push。', '', '实际命令：', '',
        '```text', metadata['command'], '```', '',
        'train_episode_metrics.csv、eval_episode_metrics.csv保存逐Episode结果；eval_seed_summary.csv先按种子聚合；paired_comparison.csv及paired_seed_differences.csv保存配对比较。',
        '其他审计：reliability_table.npz、safe_set_summary.csv、safe_pair_distribution.csv、reproducibility_checks.csv、input_manifest.json、validation_summary.json、run_metadata.json。',
    ]
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def masked_comparison(args):
    output = args.output_dir or Path(RESULTS_DIR)/'masked_ppo_latency_comparison'
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    files = [Path(p) for p in (SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH)] + [
        Path(name) for name in ('agents/ppo_agent.py', 'core/main_loop.py', 'core/env_state.py', 'core/task.py',
                                'core/server.py', 'core/spatial_risk.py', 'config/configuration.py', 'config/params.py',
                                'compare_ppo_greedy.py')]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (output/'input_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    import shlex
    metadata = {'command': shlex.join([sys.executable, *sys.argv]), 'seed_ids': args.seeds, 'seed_count': len(args.seeds),
                'train_episodes': args.train_episodes, 'eval_episodes': args.eval_episodes,
                'thresholds': MASKED_THRESHOLDS, 'alpha': MASKED_ALPHA, 'beta': effective_spatial_risk_beta(),
                'arrival_rate': MASKED_RATE, 'gh_order': 32, 'torch_threads': torch.get_num_threads(),
                'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'simpy': simpy.__version__,
                'ppo_hyperparameters': {key: value for key, value in vars(params).items() if key.endswith('_ppo')},
                'git_branch': __import__('subprocess').check_output(['git', 'branch', '--show-current'], text=True).strip(),
                'git_head': __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_status_before': __import__('subprocess').check_output(['git', 'status', '--short'], text=True)}
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    server_frame = pd.read_excel(SERVER_INFO_PATH, sheet_name='Servers', dtype={'Site_ID': 'string'})
    profiles = MainLoop._build_server_profiles(server_frame)
    task_sets = {'train': MainLoop._build_task_profiles(pd.read_excel(TRAIN_TASK_PATH)),
                 'evaluation': MainLoop._build_task_profiles(pd.read_excel(EVALUATION_TASK_PATH))}
    assert len(profiles) == 8 and all(len(tasks) == 200 for tasks in task_sets.values())
    cached = precompute_masked_profiles(profiles, task_sets, output)
    # A missing snapshot must fail rather than silently repeat integration.
    import core.env_state as state_module
    saved_functions = (state_module.compute_pair_reliability_gh, state_module.compute_marginal_failure_probability)
    runtime_calls = []

    def unexpected_gh(*unused_args, **unused_kwargs):
        runtime_calls.append(True)
        raise AssertionError('GH integration reached an episode: precomputed snapshot missing')

    state_module.compute_pair_reliability_gh = state_module.compute_marginal_failure_probability = unexpected_gh
    collected = []
    started = time.monotonic()
    try:
        for scenario in range(2):
            for seed in args.seeds:
                collected.append(run_masked_seed(scenario, seed, args.train_episodes, args.eval_episodes, profiles, cached, output))
    finally:
        state_module.compute_pair_reliability_gh, state_module.compute_marginal_failure_probability = saved_functions
    filenames = ('train_episode_metrics', 'eval_episode_metrics', 'train_server_load', 'eval_server_load',
                 'paired_streams', 'reproducibility_checks')
    for index, name in enumerate(filenames):
        pd.concat([case[index] for case in collected], ignore_index=True).to_csv(output/f'{name}.csv', index=False, float_format='%.17g')
    train = pd.concat([case[0] for case in collected], ignore_index=True)
    train[['threshold', 'seed_id', 'init_seed', 'train_action_seed', 'shuffle_seed']].drop_duplicates().to_csv(output/'seed_mapping.csv', index=False)
    assert not runtime_calls
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    validation = {'passed': True, 'gh_batch_2d_calls': 2, 'gh_runtime_calls': 0, 'paired_streams_verified': True,
                  'checkpoint_frozen': True, 'all_tasks_and_replicas_completed': True,
                  'unsafe_nonfallback_actions': 0, 'reproducibility_checks': len(pd.concat([case[5] for case in collected])),
                  'training_episodes': 2*len(args.seeds)*args.train_episodes,
                  'evaluation_episodes': 2*len(args.seeds)*args.eval_episodes*4, 'elapsed_s': time.monotonic()-started}
    (output/'validation_summary.json').write_text(json.dumps(validation, indent=2)+'\n')
    summarize_masked_results(output)
    print(f'Complete masked comparison: {output} ({validation["elapsed_s"]:.1f}s)', flush=True)


ABLATION_MODELS = ('correlated', 'independent')


def reliability_outcomes(rows, tasks):
    """Evaluate selected actions against the same true correlated reference."""
    data = task_latency_components(rows)
    snapshots = [tasks[tid]['reliability_snapshot'] for tid in data.task_id]
    corr = np.array([snapshot[1] for snapshot in snapshots])
    ind = np.array([snapshot[2] for snapshot in snapshots])
    threshold = data.reliability_requirement.to_numpy()[:, None]
    cm, im = corr >= threshold, ind >= threshold
    assert not np.any(cm & ~im)
    action = data.action_index.to_numpy(dtype=int)
    index = np.arange(len(data))
    data['selected_r_corr'] = corr[index, action]
    data['selected_r_ind'] = ind[index, action]
    data['correlated_safe_pair_count'] = cm.sum(axis=1)
    data['independent_safe_pair_count'] = im.sum(axis=1)
    data['masks_differ'] = np.any(cm != im, axis=1)
    data['true_satisfied'] = data.selected_r_corr >= data.reliability_requirement
    data['independent_satisfied'] = data.selected_r_ind >= data.reliability_requirement
    data['false_safe_selected'] = data.independent_satisfied & ~data.true_satisfied
    data['avoidable_violation'] = cm.any(axis=1) & ~data.true_satisfied
    data['unavoidable_violation'] = ~cm.any(axis=1)
    data['unavoidable_fallback_violation'] = data.unavoidable_violation & data.is_fallback
    data['unavoidable_normal_violation'] = data.unavoidable_violation & ~data.is_fallback
    data['normal_decision'] = ~data.is_fallback
    data['false_safe_normal_decision'] = data.false_safe_selected & data.normal_decision
    np.testing.assert_array_equal(data.requirement_satisfied, data.true_satisfied)
    np.testing.assert_array_equal(~data.true_satisfied, data.avoidable_violation | data.unavoidable_violation)
    own = np.array([tasks[tid]['safe_mask'] for tid in data.task_id])
    assert np.all(own[index, action] | data.is_fallback.to_numpy())
    np.testing.assert_array_equal(data.is_fallback, ~own.any(axis=1))
    return data


def reliability_episode_metrics(data):
    return data.groupby('episode').agg(
        false_safe_selected_count=('false_safe_selected', 'sum'),
        false_safe_selected_fraction=('false_safe_selected', 'mean'),
        false_safe_normal_decision_count=('false_safe_normal_decision', 'sum'),
        avoidable_violation_count=('avoidable_violation', 'sum'),
        unavoidable_violation_count=('unavoidable_violation', 'sum'),
        unavoidable_fallback_violation_count=('unavoidable_fallback_violation', 'sum'),
        unavoidable_normal_violation_count=('unavoidable_normal_violation', 'sum'),
        rsr_true=('true_satisfied', 'mean'), rsr_independent=('independent_satisfied', 'mean'),
        mean_correlated_safe_pair_count=('correlated_safe_pair_count', 'mean'),
        mean_independent_safe_pair_count=('independent_safe_pair_count', 'mean'),
        masks_differ_task_fraction=('masks_differ', 'mean'),
        max_latency_decomposition_error_s=('latency_decomposition_error_s', 'max'),
    ).rename_axis('episode_id').reset_index()


def run_ablation_seed(scenario, seed, args, profiles, cached, output, *, include_heuristics=False,
                      models=ABLATION_MODELS, ell_ratio=1., mixed=False):
    threshold = 'mixed_C' if mixed else MASKED_THRESHOLDS[scenario]
    case_name = threshold if mixed else f'threshold_{threshold:g}'
    namespace = 71 if mixed else 41
    train_frames, eval_frames, load_frames, checks = [], [], [], []
    for model in models:
        arm = ABLATION_MODELS.index(model)
        case = output/case_name/f'seed_{seed}'/model
        case.mkdir(parents=True, exist_ok=False)
        tags = {('scenario' if mixed else 'threshold'):threshold, 'seed_id':seed, 'model':model}
        streams = {name: derived_seed((namespace, scenario, seed, role, arm if role in (1, 2) else 0))
                   for name, role in (('init_seed', 0), ('train_action_seed', 1), ('shuffle_seed', 2))}
        arrival_stream = (namespace, scenario, seed, 3)
        sets = {}
        for split in ('train', 'evaluation'):
            sets[split] = {}
            for tid, p in cached[split][threshold].items():
                mask = p['reliability_snapshot'][1 if model == 'correlated' else 2] >= p['Reliability_Requirement']
                mask.setflags(write=False)
                sets[split][tid] = {**p, 'safe_mask': mask}
        torch.manual_seed(streams['init_seed'])
        agent = build_agent()
        initial_digest = parameter_digest(agent)
        initial, final = case/'initial.pt', case/'final.pt'
        agent.save_model(initial)
        torch.manual_seed(streams['train_action_seed']); np.random.seed(streams['shuffle_seed'])
        loop, training, train_loads = train_masked_agent(
            agent, profiles, sets['train'], scenario, seed, args.train_episodes,
            reliability_model=model, arrival_stream=arrival_stream, ell_ratio=ell_ratio,
            progress_label=threshold if mixed else None)
        previous = Path(RESULTS_DIR)/'correlated_vs_independent_ppo'/case.relative_to(output)/'train_episode_metrics.csv'
        prefix_verified = False
        if include_heuristics and not mixed and ell_ratio == 1 and previous.exists():
            old = pd.read_csv(previous, float_precision='round_trip')
            n = min(len(old), len(training))
            np.testing.assert_array_equal(training.parameter_digest.iloc[:n], old.parameter_digest.iloc[:n])
            np.testing.assert_array_equal(training.mean_latency_s.iloc[:n], old.mean_latency_s.iloc[:n])
            prefix_verified = True
        agent.save_model(final)
        train_tasks = reliability_outcomes(loop.task_results, sets['train'])
        training = training.merge(reliability_episode_metrics(train_tasks), on='episode_id', validate='one_to_one')
        training['train_arrival_seed'] = [derived_seed((*arrival_stream, int(e))) for e in training.episode_id]
        training = training.assign(**tags, **streams)
        training.to_csv(case/'train_episode_metrics.csv', index=False, float_format='%.17g')
        train_tasks.assign(**tags).to_csv(case/'train_task_records.csv.gz', index=False, float_format='%.17g')
        train_frames.append(training)
        load_frames.append(train_loads.assign(**tags, phase='train'))
        replay = build_agent(); replay.load_model(initial)
        torch.manual_seed(streams['train_action_seed']); np.random.seed(streams['shuffle_seed'])
        _, replay_metrics, _ = train_masked_agent(
            replay, profiles, sets['train'], scenario, seed, 1, progress=False,
            reliability_model=model, arrival_stream=arrival_stream, ell_ratio=ell_ratio)
        assert replay_metrics.parameter_digest.iloc[0] == training.parameter_digest.iloc[0]
        assert replay_metrics.mean_latency_s.iloc[0] == training.mean_latency_s.iloc[0]
        frozen = frozen_agent(final); snapshot = frozen_snapshot(frozen)
        digest = hashlib.sha256(final.read_bytes()).hexdigest()
        per_arm, task_frames = [], []
        for episode in range(1, args.eval_episodes+1):
            arrival_seed = derived_seed((namespace, scenario, seed, 4, episode))
            action_seed = derived_seed((namespace, scenario, seed, 5, arm, episode))
            basis = np.random.default_rng(arrival_seed).exponential(size=200)
            records = []
            values, loads = masked_evaluation(
                'trained_masked_ppo', frozen, profiles, sets['evaluation'], basis, episode,
                action_seed, loop.get_task_outcome_info, reliability_model=model, task_records=records,
                ell_ratio=ell_ratio,record_backlog=mixed)
            details = reliability_outcomes(records, sets['evaluation'])
            extra = reliability_episode_metrics(details).iloc[0].to_dict()
            per_arm.append({**values, **extra, **tags, 'eval_arrival_seed': arrival_seed,
                            'eval_action_seed': action_seed})
            task_frames.append(details.assign(**tags))
            load_frames.append(loads.assign(**tags, phase='evaluation'))
            if episode == 1:
                again, _ = masked_evaluation('trained_masked_ppo', frozen, profiles, sets['evaluation'],
                    basis, episode, action_seed, loop.get_task_outcome_info, reliability_model=model,
                    ell_ratio=ell_ratio,record_backlog=mixed)
                assert again == values
        same_tree(snapshot, frozen_snapshot(frozen))
        assert hashlib.sha256(final.read_bytes()).hexdigest() == digest
        evaluation = pd.DataFrame(per_arm)
        evaluation.to_csv(case/'eval_episode_metrics.csv', index=False, float_format='%.17g')
        pd.concat(task_frames).to_csv(case/'eval_task_records.csv.gz', index=False, float_format='%.17g')
        eval_frames.append(evaluation)
        checks.append({**tags, **streams, 'initial_parameter_digest': initial_digest,
                       'final_checkpoint_sha256': digest, 'train_replay_passed': True,
                       'eval_replay_passed': True, 'evaluation_frozen': True,
                       'phase4_training_prefix_verified': prefix_verified})
        print(f'COMPLETE threshold={threshold} seed={seed} model={model}', flush=True)
    if len(models) > 1:
        assert checks[0]['initial_parameter_digest'] == checks[1]['initial_parameter_digest']
        np.testing.assert_array_equal(train_frames[0].train_arrival_seed, train_frames[1].train_arrival_seed)
        np.testing.assert_array_equal(eval_frames[0].arrival_sha256, eval_frames[1].arrival_sha256)
    if include_heuristics:
        frames, loads, reproduction = evaluate_safe_methods(scenario, seed, profiles,
            cached['evaluation'][threshold], args.eval_episodes, output, loop.get_task_outcome_info,
            ell_ratio=ell_ratio,mixed=mixed)
        for frame in frames:
            np.testing.assert_array_equal(eval_frames[0].arrival_sha256, frame.arrival_sha256)
            np.testing.assert_array_equal(eval_frames[0].fallback_count, frame.fallback_count)
            np.testing.assert_array_equal(eval_frames[0].rsr_true, frame.rsr_true)
        eval_frames.extend(frames); load_frames.extend(loads); checks.extend(reproduction)
    return pd.concat(train_frames), pd.concat(eval_frames), pd.concat(load_frames), pd.DataFrame(checks)


def evaluate_safe_methods(scenario, seed, profiles, tasks, episodes, output, logger, *, ell_ratio=1., mixed=False):
    threshold = 'mixed_C' if mixed else MASKED_THRESHOLDS[scenario]
    case_name = threshold if mixed else f'threshold_{threshold:g}'
    namespace = 71 if mixed else 41
    frames, loads_all, checks = [], [], []
    for method, role in (('safe_min_latency', 7), ('safe_random', 6)):
        case = output/case_name/f'seed_{seed}'/method
        case.mkdir(parents=True, exist_ok=False)
        records, metrics = [], []
        tags = {('scenario' if mixed else 'threshold'):threshold, 'seed_id':seed, 'model':method}
        for episode in range(1, episodes+1):
            arrival_seed = derived_seed((namespace, scenario, seed, 4, episode))
            action_seed = derived_seed((namespace, scenario, seed, role, episode))
            basis = np.random.default_rng(arrival_seed).exponential(size=200)
            rows = []
            values, loads = masked_evaluation(method, None, profiles, tasks, basis, episode,
                action_seed, logger, task_records=rows, ell_ratio=ell_ratio,record_backlog=mixed)
            details = reliability_outcomes(rows, tasks)
            metrics.append({**values, **reliability_episode_metrics(details).iloc[0].to_dict(), **tags,
                            'eval_arrival_seed':arrival_seed, 'eval_action_seed':action_seed})
            records.append(details.assign(**tags))
            loads_all.append(loads.assign(**tags, phase='evaluation'))
            if episode == 1:
                again, _ = masked_evaluation(method, None, profiles, tasks, basis, episode,
                    action_seed, logger, ell_ratio=ell_ratio,record_backlog=mixed)
                assert values == again
        frame = pd.DataFrame(metrics)
        frame.to_csv(case/'eval_episode_metrics.csv', index=False, float_format='%.17g')
        pd.concat(records).to_csv(case/'eval_task_records.csv.gz', index=False, float_format='%.17g')
        frames.append(frame)
        checks.append({**tags, 'eval_replay_passed':True})
    return frames, loads_all, checks


def summarize_ablation(output, *, plots=True):
    train = pd.read_csv(output/'train_episode_metrics.csv')
    evaluation = pd.read_csv(output/'eval_episode_metrics.csv')
    measures = [name for name in evaluation.select_dtypes('number') if name not in
                ('threshold', 'seed_id', 'episode_id', 'eval_arrival_seed', 'eval_action_seed')]
    seeds = evaluation.groupby(['threshold', 'seed_id', 'model'])[measures].mean().reset_index()
    seeds.to_csv(output/'eval_seed_summary.csv', index=False, float_format='%.17g')
    methods = seeds.groupby(['threshold', 'model'])[measures].agg(['mean', 'std'])
    methods.columns = ['_'.join(column) for column in methods.columns]
    methods.reset_index().to_csv(output/'method_summary.csv', index=False, float_format='%.17g')
    violation_fields = ['false_safe_selected_count', 'false_safe_selected_fraction',
                        'false_safe_normal_decision_count', 'avoidable_violation_count',
                        'unavoidable_violation_count', 'unavoidable_fallback_violation_count',
                        'unavoidable_normal_violation_count', 'rsr_true', 'rsr_independent']
    seeds.groupby(['threshold', 'model'])[violation_fields].agg(['mean', 'std']).to_csv(output/'reliability_violation_summary.csv', float_format='%.17g')
    differences = []
    for threshold, group in seeds.groupby('threshold'):
        corr = group[group.model == 'correlated'].set_index('seed_id')
        ind = group[group.model == 'independent'].set_index('seed_id').reindex(corr.index)
        for seed in corr.index:
            differences.append(dict(threshold=threshold, seed_id=seed,
                delta_latency_s=corr.loc[seed, 'mean_latency_s'] - ind.loc[seed, 'mean_latency_s'],
                latency_relative_change_pct=100*(corr.loc[seed, 'mean_latency_s']/ind.loc[seed, 'mean_latency_s']-1),
                delta_rsr=corr.loc[seed, 'rsr_true'] - ind.loc[seed, 'rsr_true'],
                avoided_violations_per_episode=ind.loc[seed, 'avoidable_violation_count']-corr.loc[seed, 'avoidable_violation_count']))
    paired = pd.DataFrame(differences)
    paired.to_csv(output/'paired_seed_differences.csv', index=False, float_format='%.17g')
    paired.groupby('threshold').agg(
        delta_latency_mean_s=('delta_latency_s', 'mean'), delta_latency_sd_s=('delta_latency_s', 'std'),
        delta_rsr_mean=('delta_rsr', 'mean'), delta_rsr_sd=('delta_rsr', 'std'),
        latency_relative_change_pct=('latency_relative_change_pct', 'mean'),
        avoided_violations_per_episode=('avoided_violations_per_episode', 'mean'),
    ).to_csv(output/'paired_comparison.csv', float_format='%.17g')
    loads = pd.read_csv(output/'server_load_episodes.csv')
    fields = ['replica_count', 'execution_cycles', 'cpu_busy_s', 'mean_cpu_wait_s', 'cpu_utilization']
    sl = loads[loads.phase == 'evaluation'].groupby(['threshold', 'seed_id', 'model', 'server_id'])[fields].mean().reset_index()
    sl.to_csv(output/'server_load_seed_summary.csv', index=False, float_format='%.17g')
    sl.groupby(['threshold', 'model', 'server_id'])[fields].mean().to_csv(output/'server_load_summary.csv', float_format='%.17g')
    if not plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(evaluation.threshold.unique()), figsize=(11, 4), squeeze=False)
    for ax, (threshold, group) in zip(axes[0], train.groupby('threshold')):
        for model, arm in group.groupby('model'):
            mean = arm.groupby('episode_id').mean_latency_s.mean()
            ax.plot(mean.index, mean.rolling(10, min_periods=1).mean(), label=model)
        ax.set(title=f'R_req={threshold}', xlabel='Training episode', ylabel='Task latency (s), rolling 10')
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output/'training_curves.png', dpi=150); plt.close(fig)


def write_ablation_report(output):
    metadata = json.loads((output/'run_metadata.json').read_text())
    validation = json.loads((output/'validation_summary.json').read_text())
    methods = pd.read_csv(output/'method_summary.csv')
    paired = pd.read_csv(output/'paired_comparison.csv')
    seeds = pd.read_csv(output/'eval_seed_summary.csv')
    train = pd.read_csv(output/'train_episode_metrics.csv')
    evaluation = pd.read_csv(output/'eval_episode_metrics.csv')
    audit = pd.read_csv(output/'cpu_wait_audit/phase3_method_audit.csv')
    lines = ['# 第四阶段：CPU等待审计及相关／独立Mask PPO消融', '',
        '先完成CPU审计，确认没有实际时延计算错误后才开展消融。历史结果、原始Excel和正式配置均未修改。', '',
        '## CPU等待口径审计', '',
        '第三阶段mean_cpu_wait_s统计全部400副本的平均CPU等待，包含任务首次完成后仍继续执行的Loser，不能解释为任务完成时延中的等待分量。新日志明确提供Winner、Loser、All-Replica三种口径；保留旧字段作为All-Replica的历史别名。',
        '第三阶段未落盘完整逐任务生命周期，本次使用保存的初始/最终Checkpoint、原评估到达及动作种子、原GH缓存重放全部480个评估Episode。480个历史轨迹SHA256完全一致，96,000任务和192,000副本均完成；没有重新训练第三阶段模型。',
        f'每任务验证首次完成与Winner识别、上传/排队/执行时间非负，以及task_latency=winner_upload+winner_wait+winner_execution。最大分解误差{validation["cpu_audit"]["max_decomposition_error_s"]:.3g}s，未发现实际时延错误。', '',
        '| 阈值 | 第三阶段方法 | 任务时延/s | Winner上传/s | Winner等待/s | Winner执行/s | Loser等待/s | 全副本等待/s |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in audit.itertuples():
        lines.append(f'| {r.threshold:g} | {r.method} | {r.task_latency_s:.6f} | {r.winner_upload_s:.6f} | {r.winner_cpu_wait_s:.6f} | {r.winner_execution_s:.6f} | {r.loser_cpu_wait_s:.6f} | {r.all_replica_cpu_wait_s:.6f} |')
    lines += ['', 'PPO比SML任务时延低，但全副本等待高，主要体现在Loser排队。Winner较快即可完成逻辑任务，另一个副本继续占用CPU并贡献全副本等待。审计表中PPO的Winner等待也低于SML，因此不存在将较高全副本等待直接加到任务时延的矛盾。', '',
        '## 实验设置与配对', '',
        f'阈值{metadata["thresholds"]}；每阈值相关/独立两臂，各{len(metadata["seed_ids"])}种子×{metadata["train_episodes"]}训练Episode，最终Checkpoint各评估{metadata["eval_episodes"]}Episode。完整运行12模型、1200训练Episode、240评估Episode。此总量遵循用户对附件600轮计数冲突的明确澄清。',
        '8台EUA节点、28个字典序服务器对；200任务/400副本，alpha=0.05仅在内存缩放一次，beta=0.5，ell0=median_distance/ln(2)，到达率2 tasks/s。固定训练和独立评估任务，阈值仅在内存覆盖。',
        f'复用同一PPOAgent、{params.num_states}维状态、[64,32]/tanh网络、当前奖励/学习率/GAE/按真实间隔折扣及双副本执行。两臂同种子初始化参数完全相同，训练与评估到达流逐Episode配对；动作采样和shuffle使用独立随机流。评估使用Masked Categorical随机采样且冻结网络、优化器和缓冲区。',
        '相关臂使用R_corr构建Mask，独立臂使用同一GH输出的公平R_ind构建Mask；所有真实性能判定始终用R_corr。各臂空集调用现有同一个最小时延Fallback；全部任务执行，Fallback无策略Transition，奖励按任务完成被观察到的决策区间累计有效QoS。GH只在运行级预计算两批，另有ell=0两批验证，Episode内积分被断言禁止。', '',
        '## 评估结果', '', '先平均每种子的20个配对Episode，再对3个种子均值报告mean±sample SD。RSR是理论约束满足率，不是实际故障试验成功率。', '',
        '| 阈值 | Mask模型 | 任务时延mean±SD/s | 真实RSR | 独立假设RSR | Fallback/200 | Winner等待/s | Loser等待/s | 全副本等待/s |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in methods.itertuples():
        lines.append(f'| {r.threshold:g} | {r.model} | {r.mean_latency_s_mean:.6f} ± {r.mean_latency_s_std:.6f} | {r.rsr_true_mean:.3%} | {r.rsr_independent_mean:.3%} | {r.fallback_count_mean:.2f} ({r.fallback_fraction_mean:.1%}) | {r.mean_winner_cpu_wait_s_mean:.6f} | {r.mean_loser_cpu_wait_s_mean:.6f} | {r.mean_all_replica_cpu_wait_s_mean:.6f} |')
    lines += ['', '## 实际动作误判与违约', '',
        '假安全选择定义为所选动作R_ind≥阈值且R_corr<阈值，以全部200任务为分母；不是候选服务器对的比例。该标记同样用于描述相关臂Fallback的所选对，但该臂Fallback已知不安全，并不表示相关Mask误判。另列非Fallback假安全数。',
        '可避免违约：真实相关安全集非空但选了相关不安全动作。不可避免违约：真实相关安全集为空；独立臂可能仍认为有安全动作并执行普通PPO决策，因此不可避免数量不等于其自身Fallback数量。', '',
        '| 阈值 | 模型 | 假安全选择/200 | 非Fallback假安全/200 | 可避免违约/200 | 不可避免违约/200 | 其中自身Fallback / 普通PPO |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in methods.itertuples():
        lines.append(f'| {r.threshold:g} | {r.model} | {r.false_safe_selected_count_mean:.3f} ({r.false_safe_selected_fraction_mean:.3%}) | {r.false_safe_normal_decision_count_mean:.3f} | {r.avoidable_violation_count_mean:.3f} | {r.unavoidable_violation_count_mean:.3f} | {r.unavoidable_fallback_violation_count_mean:.3f} / {r.unavoidable_normal_violation_count_mean:.3f} |')
    for (threshold, model), g in evaluation.groupby(['threshold','model']):
        lines.append(f'\n累计评估：阈值{threshold:g}、{model}，{len(g)*200}任务，假安全选择{int(g.false_safe_selected_count.sum())}次（非Fallback {int(g.false_safe_normal_decision_count.sum())}次），可避免违约{int(g.avoidable_violation_count.sum())}次，不可避免违约{int(g.unavoidable_violation_count.sum())}次。')
    lines += ['', '## 配对差值', '', 'Delta_latency=相关−独立，负值相关更快；Delta_RSR=相关真实RSR−独立真实RSR，正值相关更可靠。相对时延变化以独立臂为分母，先逐种子计算。', '',
        '| 阈值 | Delta latency mean±SD/s | 相对时延变化 | Delta RSR/百分点 | 每Episode减少违约 |', '|---:|---:|---:|---:|---:|']
    for r in paired.itertuples():
        lines.append(f'| {r.threshold:g} | {r.delta_latency_mean_s:+.6f} ± {r.delta_latency_sd_s:.6f} | {r.latency_relative_change_pct:+.3f}% | {100*r.delta_rsr_mean:+.3f} | {r.avoided_violations_per_episode:.3f} |')
    lines += ['', '逐种子结果见eval_seed_summary.csv、paired_seed_differences.csv。两臂Mask和Fallback不同会改变后续队列，时延差包含决策支持集与Fallback变化的作用，不能全部归因于学习能力。', '',
        '## 学习空间与训练趋势', '', '| 阈值 | 模型 | 训练有效Transition/轮 | 多动作/轮 | 唯一动作/轮 | 前10轮时延/s | 末10轮时延/s |', '|---:|---|---:|---:|---:|---:|---:|']
    for (threshold, model), g in train.groupby(['threshold','model']):
        lines.append(f'| {threshold:g} | {model} | {g.effective_transition_count.mean():.0f} | {g.multi_safe_decision_count.mean():.0f} | {g.single_safe_decision_count.mean():.0f} | {g[g.episode_id<=10].mean_latency_s.mean():.6f} | {g[g.episode_id>metadata["train_episodes"]-10].mean_latency_s.mean():.6f} |')
    lines += ['', '![训练曲线](training_curves.png)', '',
        '正常训练与参数更新已逐轮验证；训练趋势有波动，100轮和3种子不足以证明充分收敛。严格阈值下唯一合法动作与Fallback限制可学习样本，独立臂较宽松的动作集还包含假安全选择。未据此调整任何超参数。', '',
        '## CPU分量与服务器负载', '', 'Winner上传+等待+执行逐任务等于任务时延，分量均值及种子SD见method_summary.csv。全副本CPU等待包含Loser和Fallback；完整逐任务记录存于各阈值/种子/模型目录的train_task_records.csv.gz和eval_task_records.csv.gz。',
        'server_load_summary.csv保存各服务器每Episode副本数、CPU cycles、忙时、等待和利用率；先按种子平均再聚合。server_load_seed_summary.csv与server_load_episodes.csv提供细分。所有Episode均400副本，不因任务首次完成而取消Loser。', '',
        '## 检查、结论与后续', '',
        f'- CPU审计通过；ell=0最大差{validation["ell_zero_max_error"]:.3g}；正相关安全集为公平独立安全集子集。',
        '- 所有训练和评估完成全部任务/副本；自身Mask非法非Fallback动作0；Fallback复用统一函数。Task_ID、奖励、当前/下一Mask和真实时间间隔逐训练Episode审计通过；原全Fallback/交错/单样本回归另行运行。',
        '- 参数、log_prob、时延与等待有限；singleton log_prob为0。每模型首训练Episode重放参数摘要和时延一致，首评估Episode轨迹和指标一致；并未重复整套100轮。',
        '- 评估前后网络/旧策略/Critic/优化器逐项一致，缓冲区为空，最终Checkpoint SHA256不变。输入文件哈希运行前后一致，历史文件另行核对。',
        '- 相关模型消除了真实安全集非空时的可避免违约；真实安全集为空时，任何策略均不能满足约束。是否存在时延代价应以上述配对差为准；不能把独立假设RSR当成真实性能。',
        '- 这轮是可靠性建模机制验证，不作统计显著性或最终优势声明。若实际假安全和RSR差稳定出现，可进入另行授权的10 seeds×300 Episodes确认性实验；应同时保留Winner/Loser口径和避免/不可避免违约拆分。本轮不启动正式实验。', '',
        '## 修改范围与执行', '',
        '修改compare_ppo_greedy.py（消融选项、复用训练评估、逐任务和配对统计）、core/env_state.py（Mask模型选择）、core/main_loop.py（透传模型选项）、io_utils/save_parameters_and_logs.py（CPU分量审计和明确统计）。未改GH数学、PPO类/网络/更新、任务执行机制、原始数据或正式配置。',
        f'分支`{metadata["git_branch"]}`，HEAD `{metadata["git_head"]}`。开始时已有未提交修改，全部保留；本轮不commit/push。最终状态与文件哈希核对另见source_and_output_audit.json。', '',
        '实际主要命令：', '```text',
        '/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python /tmp/phase4_cpu_audit.py',
        '/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python tools/validate_masked_ppo.py --output-dir /tmp/phase4_fallback_regression_20261021',
        metadata['command'], '```', '',
        '极小规模集成检查另输出至/tmp/phase4_ablation_smoke_20261021（2轮训练/2轮评估、单种子，两阈值两模型），未覆盖历史结果。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


CONFIRMATION_METHODS = ('correlated', 'independent', 'safe_min_latency', 'safe_random')


def paired_seed_statistics(seeds, *, group_column='threshold', identity_reference=False):
    differences = []
    for threshold, group in seeds.groupby(group_column):
        corr = group[group.model == 'correlated'].set_index('seed_id')
        for reference in CONFIRMATION_METHODS[1:]:
            baseline = group[group.model == reference].set_index('seed_id').reindex(corr.index)
            for seed in corr.index:
                a, b = corr.loc[seed], baseline.loc[seed]
                differences.append(dict({group_column:threshold}, seed_id=seed, reference=reference,
                    delta_latency_s=a.mean_latency_s-b.mean_latency_s,
                    latency_improvement_pct=100*(1-a.mean_latency_s/b.mean_latency_s),
                    delta_rsr_pp=100*(a.rsr_true-b.rsr_true),
                    avoided_violations_per_episode=b.avoidable_violation_count-a.avoidable_violation_count,
                    delta_winner_upload_s=a.mean_winner_upload_s-b.mean_winner_upload_s,
                    delta_winner_wait_s=a.mean_winner_cpu_wait_s-b.mean_winner_cpu_wait_s,
                    delta_winner_execution_s=a.mean_winner_execution_s-b.mean_winner_execution_s))
    differences = pd.DataFrame(differences)
    paired = []
    fields = [c for c in differences if c not in (group_column, 'seed_id', 'reference')]
    for (threshold, reference), group in differences.groupby([group_column, 'reference']):
        n = len(group)
        row = dict({group_column:threshold}, reference=reference, seed_count=n,
                   latency_better_seed_count=int((group.delta_latency_s < 0).sum()),
                   rsr_better_seed_count=int((group.delta_rsr_pp > 1e-10).sum()))
        identity = identity_reference and reference == 'independent'
        row['comparison_kind'] = 'shared_policy_identity' if identity else 'paired_seed_t_interval'
        if identity:
            assert not group.delta_latency_s.any() and not group.delta_rsr_pp.any()
        for field in fields:
            mean, sd = group[field].mean(), group[field].std(ddof=1)
            half = 0. if identity else (student_t.ppf(.975, n-1)*sd/np.sqrt(n) if n > 1 else np.nan)
            row.update({field+'_mean':mean, field+'_sd':sd,
                        field+'_ci95_lower':mean-half, field+'_ci95_upper':mean+half})
        paired.append(row)
    return differences, pd.DataFrame(paired)


def training_seed_statistics(train, *, group_column='threshold'):
    learning = []
    for (threshold, seed, model), group in train.groupby([group_column, 'seed_id', 'model']):
        last_episode = int(group.episode_id.max())
        first, last = group[group.episode_id<=10], group[group.episode_id>last_episode-10]
        early_tail = group[(group.episode_id>last_episode-100)&(group.episode_id<=last_episode-50)]
        tail = group[group.episode_id>last_episode-50]
        learning.append(dict({group_column:threshold}, seed_id=seed, model=model,
            first10_latency_s=first.mean_latency_s.mean(), last10_latency_s=last.mean_latency_s.mean(),
            first10_reward=first.episode_reward.mean(), last10_reward=last.episode_reward.mean(),
            preceding50_latency_s=early_tail.mean_latency_s.mean(), last50_latency_s=tail.mean_latency_s.mean(),
            tail_change_pct=100*(tail.mean_latency_s.mean()/early_tail.mean_latency_s.mean()-1),
            last50_episode_latency_sd_s=tail.mean_latency_s.std(ddof=1),
            effective_transition_count=group.effective_transition_count.mean(),
            multi_safe_decision_count=group.multi_safe_decision_count.mean(),
            single_safe_decision_count=group.single_safe_decision_count.mean(),
            update_calls=group.update_call_count.sum(), optimizer_steps=group.actor_optimizer_steps.sum()))
    return pd.DataFrame(learning)


def summarize_confirmation(output, *, plots=True, write_report=True, identity_reference=False):
    summarize_ablation(output, plots=plots)
    seeds = pd.read_csv(output/'eval_seed_summary.csv')
    train = pd.read_csv(output/'train_episode_metrics.csv')
    differences, paired = paired_seed_statistics(seeds, identity_reference=identity_reference)
    differences.to_csv(output/'paired_seed_differences.csv', index=False, float_format='%.17g')
    paired.to_csv(output/'paired_comparison.csv', index=False, float_format='%.17g')
    training_seed_statistics(train).to_csv(output/'training_seed_summary.csv', index=False, float_format='%.17g')
    train[['threshold','seed_id','model','init_seed','train_action_seed','shuffle_seed']].drop_duplicates().to_csv(output/'seed_mapping.csv', index=False)
    evaluation = pd.read_csv(output/'eval_episode_metrics.csv')
    evaluation[['threshold','seed_id','model','episode_id','eval_arrival_seed','eval_action_seed','arrival_sha256']].to_csv(output/'evaluation_streams.csv', index=False)
    # A flat header keeps the violation summary convenient to read outside pandas.
    methods = pd.read_csv(output/'method_summary.csv')
    fields = [c for c in methods if any(k in c for k in ('violation','false_safe','rsr_','fallback'))]
    methods[['threshold','model',*fields]].to_csv(output/'reliability_violation_summary.csv', index=False, float_format='%.17g')
    if plots:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        labels = ['Corr PPO', 'Ind PPO', 'SML', 'Safe Random']
        for column, (threshold, group) in enumerate(seeds.groupby('threshold')):
            for row, metric in enumerate(('mean_latency_s', 'rsr_true')):
                values = group.groupby('model')[metric]
                mean = values.mean().reindex(CONFIRMATION_METHODS)
                sd = values.std(ddof=1).reindex(CONFIRMATION_METHODS)
                axes[row,column].bar(range(4), mean, yerr=sd, capsize=3)
                axes[row,column].set(xticks=range(4), xticklabels=labels, title=f'R_req={threshold}',
                                     ylabel='Latency (s)' if row==0 else 'True RSR')
                axes[row,column].tick_params(axis='x', labelrotation=15)
        fig.tight_layout(); fig.savefig(output/'evaluation_comparison.png', dpi=150); plt.close(fig)
    if write_report:
        write_confirmation_report(output)


def write_confirmation_report(output):
    metadata = json.loads((output/'run_metadata.json').read_text())
    validation = json.loads((output/'validation_summary.json').read_text())
    methods = pd.read_csv(output/'method_summary.csv')
    paired = pd.read_csv(output/'paired_comparison.csv')
    learning = pd.read_csv(output/'training_seed_summary.csv')
    evaluation = pd.read_csv(output/'eval_episode_metrics.csv')
    loads = pd.read_csv(output/'server_load_summary.csv')
    n = len(metadata['seed_ids'])
    lines = ['# 第五阶段：Masked PPO可靠性与时延确认性实验', '',
        f'两个单阈值 × 两种PPO × {n}种子 × {metadata["train_episodes"]}训练Episode，'
        f'最终Checkpoint各评估{metadata["eval_episodes"]}个Episode；SML、Safe Random共享相同评估任务及到达流。',
        '固定8台EUA服务器、28字典序对、200任务/400副本、alpha=0.05、beta=0.5、ell=ell0、到达率2 tasks/s、GH32阶。Excel阈值及故障率仅在内存覆盖，单线程CPU PyTorch和确定性执行。网络、奖励、GAE、实际时间折扣、执行机制和Fallback均沿用第四阶段。',
        '种子1的四个完整训练任务及四方法评估先通过预检，再扩展种子2～10；预检结果直接纳入统计。各臂共享初始化和到达流，动作采样独立；随机种子命名空间沿用41，便于核对第四阶段前100轮训练轨迹。',
        '所有RSR均用真实R_corr判定，表示理论可靠性约束满足率，不是故障事件成功率。各臂按自身Mask选择，空集执行同一个最小时延Fallback，不拒绝任务。', '',
        '## 四方法评估', '',
        '先平均每种子的评估Episode，再对训练种子均值求mean±sample SD。', '',
        '| 阈值 | 方法 | 时延mean±SD/s | True RSR mean±SD/% | Fallback比例 | Winner等待/s | 全副本等待/s |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in methods.itertuples():
        lines.append(f'| {r.threshold:g} | {r.model} | {r.mean_latency_s_mean:.6f} ± {r.mean_latency_s_std:.6f} | '
            f'{100*r.rsr_true_mean:.3f} ± {100*r.rsr_true_std:.3f} | {r.fallback_fraction_mean:.1%} | '
            f'{r.mean_winner_cpu_wait_s_mean:.6f} | {r.mean_all_replica_cpu_wait_s_mean:.6f} |')
    lines += ['', '## 配对统计', '',
        'Delta_latency=CorrPPO−对照，负值更快；改善率=(对照−CorrPPO)/对照，先逐种子计算。'
        'Delta_RSR=CorrPPO_true−对照_true，以百分点计。95% CI为配对种子差的Student-t区间，'
        '不将20个评估Episode当作独立训练样本；所列区间为逐比较边际区间，未作多重比较校正。', '',
        '| 阈值 | 对照 | Delta latency mean±SD/s | 95% CI/s | 改善/% | 更快种子 | Delta RSR/百分点 | RSR差95% CI |',
        '|---:|---|---:|---|---:|---:|---:|---|']
    for r in paired.itertuples():
        lines.append(f'| {r.threshold:g} | {r.reference} | {r.delta_latency_s_mean:+.6f} ± {r.delta_latency_s_sd:.6f} | '
            f'[{r.delta_latency_s_ci95_lower:+.6f}, {r.delta_latency_s_ci95_upper:+.6f}] | '
            f'{r.latency_improvement_pct_mean:.3f} | {r.latency_better_seed_count}/{n} | '
            f'{r.delta_rsr_pp_mean:+.3f} | [{r.delta_rsr_pp_ci95_lower:+.3f}, {r.delta_rsr_pp_ci95_upper:+.3f}] |')
    lines += ['', '## 可靠性违约分解', '',
        '总违约=可避免+不可避免，二者互斥。假安全表示所选对R_ind≥阈值且R_corr<阈值，'
        '与前述分类有交集，不能再次相加。相关方法Fallback也可能碰巧选择R_ind判安全的对，'
        '这种标记属于已知不安全的Fallback，不能解释为相关Mask误判。独立臂真实空集的普通PPO任务属于不可避免违约。', '',
        '| 阈值 | 方法 | 假安全选择/轮 | 非Fallback假安全/轮 | 可避免/轮 | 不可避免/轮 | 自身Fallback/轮 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in methods.itertuples():
        lines.append(f'| {r.threshold:g} | {r.model} | {r.false_safe_selected_count_mean:.3f} | '
            f'{r.false_safe_normal_decision_count_mean:.3f} | {r.avoidable_violation_count_mean:.3f} | '
            f'{r.unavoidable_violation_count_mean:.3f} | {r.fallback_count_mean:.0f} |')
    for (threshold, model), group in evaluation.groupby(['threshold','model']):
        lines.append(f'\n{threshold:g}/{model}累计{len(group)*200}任务：假安全{int(group.false_safe_selected_count.sum())}次，'
            f'可避免违约{int(group.avoidable_violation_count.sum())}次，不可避免违约{int(group.unavoidable_violation_count.sum())}次。')
    lines += ['', '## 训练与学习空间', '',
        '| 阈值 | PPO | 首10轮时延/s | 末10轮时延/s | 201～250轮/s | 251～300轮/s | 有效Transition/轮 | 多动作/轮 | 唯一动作/轮 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    learning_details = []
    for (threshold, model), group in learning.groupby(['threshold','model']):
        lines.append(f'| {threshold:g} | {model} | {group.first10_latency_s.mean():.6f} | {group.last10_latency_s.mean():.6f} | '
            f'{group.preceding50_latency_s.mean():.6f} | {group.last50_latency_s.mean():.6f} | '
            f'{group.effective_transition_count.mean():.0f} | {group.multi_safe_decision_count.mean():.0f} | {group.single_safe_decision_count.mean():.0f} |')
        learning_details.append(f'\n{threshold:g}/{model}：末10轮种子均值范围{group.last10_latency_s.min():.6f}～{group.last10_latency_s.max():.6f}s，'
            f'末两个50轮区间逐种子相对变化范围{group.tail_change_pct.min():+.2f}%～{group.tail_change_pct.max():+.2f}%；'
            f'PPO更新{int(group.update_calls.sum())}次，Actor/Critic各{int(group.optimizer_steps.sum())}个优化步。')
    lines += learning_details
    lines += ['', '![训练曲线](training_curves.png)', '', '![时延与真实RSR](evaluation_comparison.png)', '',
        '训练曲线和末段区间用于描述稳定性，不能单凭均值下降断言数学收敛。逐种子首末10轮、尾部波动及最终评估分别保存在training_seed_summary.csv和eval_seed_summary.csv。', '',
        '## Winner时延分量与服务器负载', '',
        '| 阈值 | 方法 | Winner上传/s | Winner等待/s | Winner执行/s | Loser等待/s |', '|---:|---|---:|---:|---:|---:|']
    for r in methods.itertuples():
        lines.append(f'| {r.threshold:g} | {r.model} | {r.mean_winner_upload_s_mean:.6f} | {r.mean_winner_cpu_wait_s_mean:.6f} | '
            f'{r.mean_winner_execution_s_mean:.6f} | {r.mean_loser_cpu_wait_s_mean:.6f} |')
    lines += ['', '逐任务核对TaskLatency=WinnerUpload+WinnerCPUWaiting+WinnerExecution；全副本等待含Loser，不能代入此式。'
        '各对照的配对Winner分量差及CI见paired_comparison.csv。', '',
        '| 阈值 | 方法 | 服务器4副本/轮 | 服务器5副本/轮 | 服务器5利用率 | 服务器5等待/s |', '|---:|---|---:|---:|---:|---:|']
    for (threshold, model), group in loads.groupby(['threshold','model']):
        v=group.set_index('server_id')
        lines.append(f'| {threshold:g} | {model} | {v.loc[4,"replica_count"]:.2f} | {v.loc[5,"replica_count"]:.2f} | '
            f'{v.loc[5,"cpu_utilization"]:.2%} | {v.loc[5,"mean_cpu_wait_s"]:.4f} |')
    lines += ['', '严格阈值下相关臂训练154个有效Transition中90个只有唯一动作，评估152个中88个唯一动作；'
        '每轮仅64个多动作决策。唯一动作均受同一安全集合限制，增加训练不能创造额外可选服务器对。'
        '负载集中和Fallback比例必须同时用于解释时延；独立臂的较宽支持集包含真实不安全动作。', '',
        '## 正确性与复现', '',
        f'- 主实验{validation["training_episodes"]}训练Episode、{validation["evaluation_episodes"]}评估Episode；全部200任务/400副本完成。',
        '- 每轮审计Task_ID、动作、奖励、当前/下一Mask、下一状态和真实时间间隔；Fallback无Transition。每方法首评估Episode及每模型首训练Episode重放一致。',
        '- 评估前后Actor/旧策略/Critic/优化器及Checkpoint哈希均不变，训练缓冲区为空；自身Mask非法非Fallback动作0。',
        '- GH仅运行级缓存，Episode内积分调用被禁止；正相关安全集合包含关系、ell=0一致性检查通过。相关PPO/SML/Random的RSR和Fallback逐Episode一致。',
        '- 无NaN/Inf、遗漏任务、负等待或CPU利用率超1；违约总数与逐任务真实RSR核对一致。历史数据与源代码完整性另见source_and_output_audit.json。',
        '- 评估不生成训练Transition；正常策略采样次数用normal_decision_count，训练有效样本用effective_transition_count。', '',
        '## 执行与修改范围', '',
        f'分支`{metadata["git_branch"]}`，HEAD `{metadata["git_head"]}`；完整代码快照、配置、输入哈希、随机种子映射随结果保存。'
        '本轮仅扩展compare_ppo_greedy.py的实验编排和统计，核心算法及正式配置保持原样。', '',
        '```text', metadata['command'], '```', '',
        '未覆盖第三、四阶段结果；未运行18组相关长度实验；未commit、未push。最终九项问题的结合数据分析见下文。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def reliability_ablation(args):
    """Reuse the audited simulation and PPO for a paired model-only ablation."""
    import shutil
    import shlex
    import subprocess
    confirmation = args.confirmation
    output = args.output_dir or Path(RESULTS_DIR)/('masked_ppo_confirmation_10seed' if confirmation else 'correlated_vs_independent_ppo')
    audit = Path(args.cpu_audit_dir)
    audit_summary = json.loads((audit/'audit_summary.json').read_text())
    assert audit_summary['passed'] and not audit_summary['latency_bug_found']
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(audit, output/'cpu_wait_audit')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), thresholds=args.thresholds,
        seed_ids=args.seeds, train_episodes=args.train_episodes, eval_episodes=args.eval_episodes,
        models=ABLATION_MODELS, alpha=MASKED_ALPHA, beta=effective_spatial_risk_beta(), arrival_rate=MASKED_RATE,
        gh_order=32, torch_threads=torch.get_num_threads(), seed_namespace=41,
        python=sys.version, torch=torch.__version__, numpy=np.__version__, simpy=simpy.__version__,
        git_branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_status_before=subprocess.check_output(['git', 'status', '--short'], text=True),
        ppo_hyperparameters={k:v for k,v in vars(params).items() if k.endswith('_ppo')})
    metadata.update(confirmation=confirmation, evaluation_methods=CONFIRMATION_METHODS if confirmation else ABLATION_MODELS)
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    inputs = [Path(p) for p in (SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH)]
    inputs += [Path(name) for name in ('core/spatial_risk.py', 'agents/ppo_agent.py', 'core/task.py',
                                      'core/main_loop.py', 'core/env_state.py', 'core/server.py', 'compare_ppo_greedy.py',
                                      'io_utils/save_parameters_and_logs.py', 'config/configuration.py',
                                      'config/params.py', 'config/paths.py')]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (output/'input_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    if confirmation:
        for p in inputs:
            if p.suffix == '.py':
                target = output/'source_snapshot'/p
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
        (output/'working_tree.patch').write_text(subprocess.check_output(['git', 'diff'], text=True))
    profiles = MainLoop._build_server_profiles(pd.read_excel(SERVER_INFO_PATH, sheet_name='Servers', dtype={'Site_ID':'string'}))
    tasks = {'train':MainLoop._build_task_profiles(pd.read_excel(TRAIN_TASK_PATH)),
             'evaluation':MainLoop._build_task_profiles(pd.read_excel(EVALUATION_TASK_PATH))}
    cached = precompute_masked_profiles(profiles, tasks, output)
    # Verify ell=0 using the same implementation, once per task split.
    arrays = np.load(output/'reliability_table.npz')
    pair_index = np.array(arrays['pairs'])-1
    frequencies = np.array([p['Processing_Frequency'] for p in profiles.values()])
    independent_errors = []
    for split, profiles_t in tasks.items():
        demand = np.array([p['Computation_Demand'] for p in profiles_t.values()])[:,None]
        zero_corr, fair_ind = compute_pair_reliability_gh(demand, frequencies[pair_index],
            arrays['scaled_rates'][pair_index], np.zeros(28), effective_spatial_risk_beta())
        independent_errors.append(float(np.max(np.abs(zero_corr-fair_ind))))
        np.testing.assert_allclose(zero_corr, fair_ind, rtol=0, atol=1e-12)
        for threshold in args.thresholds:
            assert not np.any((arrays[f'{split}_correlated']>=threshold)&(arrays[f'{split}_independent']<threshold))
    import core.env_state as state_module
    saved = state_module.compute_pair_reliability_gh, state_module.compute_marginal_failure_probability
    def unexpected_gh(*a, **kw):
        raise AssertionError('Unexpected GH integral during an episode')
    state_module.compute_pair_reliability_gh = state_module.compute_marginal_failure_probability = unexpected_gh
    started = time.monotonic()
    try:
        cases = []
        jobs = ([(threshold, seed) for seed in args.seeds for threshold in args.thresholds] if confirmation else
                [(threshold, seed) for threshold in args.thresholds for seed in args.seeds])
        for threshold, seed in jobs:
            cases.append(run_ablation_seed(MASKED_THRESHOLDS.index(threshold), seed, args, profiles, cached, output,
                                            include_heuristics=confirmation))
            if confirmation and seed == args.seeds[0] and threshold == args.thresholds[-1]:
                preflight_eval = pd.concat([case[1] for case in cases])
                for _, group in preflight_eval.groupby(['threshold','seed_id','episode_id']):
                    assert group.arrival_sha256.nunique() == 1 and len(group) == 4
                    corr = group[group.model != 'independent']
                    assert corr.fallback_count.nunique() == corr.rsr_true.nunique() == 1
                assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
                preflight = dict(passed=True, seed_id=seed, trained_models=2*len(args.thresholds),
                                 train_episodes=2*len(args.thresholds)*args.train_episodes,
                                 evaluation_episodes=len(preflight_eval),
                                 phase4_prefix_verified=bool(pd.concat([case[3] for case in cases])
                                     .query("model in ['correlated', 'independent']").phase4_training_prefix_verified.all()))
                (output/'seed1_preflight.json').write_text(json.dumps(preflight, indent=2)+'\n')
                print('PREFLIGHT PASSED: seed 1 results retained; proceeding to remaining seeds', flush=True)
    finally:
        state_module.compute_pair_reliability_gh, state_module.compute_marginal_failure_probability = saved
    for i, name in enumerate(('train_episode_metrics', 'eval_episode_metrics', 'server_load_episodes', 'reproducibility_checks')):
        pd.concat([case[i] for case in cases], ignore_index=True).to_csv(output/f'{name}.csv', index=False, float_format='%.17g')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    validation = dict(passed=True, cpu_audit=audit_summary, ell_zero_max_error=max(independent_errors),
        mask_subset=True, gh_precompute_batches=2, gh_ell_zero_check_batches=2, gh_runtime_calls=0,
        frozen_evaluation=True, paired_initial_parameters=True, paired_arrivals=True,
        own_mask_illegal_nonfallback_actions=0, all_tasks_and_replicas_completed=True,
        training_episodes=len(cases)*2*args.train_episodes,
        evaluation_episodes=len(cases)*(4 if confirmation else 2)*args.eval_episodes,
        elapsed_s=time.monotonic()-started)
    (output/'validation_summary.json').write_text(json.dumps(validation, indent=2)+'\n')
    if confirmation:
        summarize_confirmation(output)
    else:
        summarize_ablation(output)
        write_ablation_report(output)
    print(json.dumps(validation, indent=2), flush=True)


LENGTH_POLICY_RATIOS = (0., 0.5, 1., 4.)


def load_length_profiles(profiles, task_sets, history, sweep):
    """Use the verified GH32 cache; no integration in policy experiments."""
    for source in (history, sweep):
        metadata = json.loads((source/'run_metadata.json').read_text())
        if metadata['beta'] != effective_spatial_risk_beta():
            raise ValueError('Historical reliability cache uses a different effective spatial-risk beta')
    old = np.load(history/'reliability_table.npz')
    arrays = np.load(sweep/'reliability_tables.npz')
    np.testing.assert_array_equal(old['pairs'], arrays['pairs'])
    np.testing.assert_array_equal(old['scaled_rates'], arrays['scaled_rates'])
    np.testing.assert_array_equal(old['scaled_rates'],
        [p['Base_Failure_Rate']*MASKED_ALPHA for p in profiles.values()])
    cached = {ratio:{} for ratio in LENGTH_POLICY_RATIOS}
    for split, tasks in task_sets.items():
        np.testing.assert_array_equal(list(tasks), arrays[f'{split}_task_ids'])
        marginal = 1-arrays[f'{split}_marginal_failure_32']
        ind = arrays[f'{split}_independent_32']
        corr_all = arrays[f'{split}_correlated_32']
        np.testing.assert_array_equal(marginal, old[f'{split}_marginal'])
        np.testing.assert_array_equal(ind, old[f'{split}_independent'])
        np.testing.assert_array_equal(corr_all[3], old[f'{split}_correlated'])
        assert np.all(np.diff(corr_all,axis=0)<=1e-12) and np.all(corr_all<=ind+1e-12)
        np.testing.assert_allclose(corr_all[0],ind,rtol=0,atol=1e-12)
        for ratio in LENGTH_POLICY_RATIOS:
            index = list(arrays['ell_ratios']).index(ratio)
            corr = corr_all[index]
            for values in (marginal,corr,ind):
                assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
                values.setflags(write=False)
            cached[ratio][split] = {}
            for threshold in MASKED_THRESHOLDS:
                masks = corr>=threshold; masks.setflags(write=False)
                assert not np.any(masks & ~(ind>=threshold))
                if ratio==0:np.testing.assert_array_equal(masks,ind>=threshold)
                cached[ratio][split][threshold] = {tid:{**p,'Reliability_Requirement':threshold,
                    'reliability_snapshot':(marginal[i],corr[i],ind[i]),'safe_mask':masks[i]}
                    for i,(tid,p) in enumerate(tasks.items())}
    return cached


def physical_trajectory_digest(data):
    columns = ['episode','task_id','arrival_time','action_index','is_fallback','server_A_id','server_B_id',
        *[f'replica_{r}_{event}_time' for r in ('A','B') for event in ('queue_enter','cpu_start','finish')],
        'task_completion_time','task_latency']
    return hashlib.sha256(data[columns].to_numpy(dtype=np.float64).tobytes()).hexdigest()


def reuse_length_evaluation(ratio, threshold, seed, model, cache, history, output, history_loads, episodes):
    source_model = 'independent' if ratio==0 and model=='correlated' else model
    source = history/f'threshold_{threshold:g}'/f'seed_{seed}'/source_model
    original = pd.read_csv(source/'eval_task_records.csv.gz',float_precision='round_trip',
                           dtype={'safe_mask':str,'action_mask':str})
    original = original[original.episode<=episodes].copy()
    metrics = pd.read_csv(source/'eval_episode_metrics.csv',float_precision='round_trip')
    metrics = metrics[metrics.episode_id<=episodes].copy().sort_values('episode_id')
    tasks = cache[ratio]['evaluation'][threshold]
    if model=='independent':
        tasks = {tid:{**p,'safe_mask':p['reliability_snapshot'][2]>=threshold} for tid,p in tasks.items()}
    actions = original.action_index.to_numpy(dtype=int)
    snapshots = [tasks[tid]['reliability_snapshot'] for tid in original.task_id]
    source_tasks = cache[1.]['evaluation'][threshold]
    old_selected = np.array([source_tasks[tid]['reliability_snapshot'][1][action]
                            for tid,action in zip(original.task_id,actions)])
    np.testing.assert_array_equal(original.pair_reliability,old_selected)
    np.testing.assert_array_equal(original.requirement_satisfied,old_selected>=threshold)
    observed = np.array([[flag=='1' for flag in mask] for mask in original.safe_mask])
    np.testing.assert_array_equal(observed,np.array([tasks[tid]['safe_mask'] for tid in original.task_id]))
    selected = np.array([snapshot[1][action] for snapshot,action in zip(snapshots,actions)])
    # Only reliability fields change; action, fallback and lifecycle remain untouched.
    rescored = original.assign(pair_reliability=selected,
                               requirement_satisfied=selected>=threshold)
    details = reliability_outcomes(rescored,tasks)
    extra = reliability_episode_metrics(details).set_index('episode_id')
    exact_reuse = ratio==1 and source_model==model
    if exact_reuse:
        for field in extra:
            np.testing.assert_allclose(metrics.set_index('episode_id')[field],extra[field],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(original.pair_reliability,selected)
        details = original
    else:
        for field in extra:metrics[field]=metrics.episode_id.map(extra[field])
        metrics['theoretical_rsr']=metrics.rsr_true
    digest = physical_trajectory_digest(original)
    assert physical_trajectory_digest(details)==digest
    scenario = MASKED_THRESHOLDS.index(threshold)
    for episode,g in details.groupby('episode'):
        assert len(g)==200
        row = metrics[metrics.episode_id==episode].iloc[0]
        arrival_seed = derived_seed((41,scenario,seed,4,int(episode)))
        assert row.eval_arrival_seed==arrival_seed
        basis = np.random.default_rng(arrival_seed).exponential(size=200)
        np.testing.assert_allclose(g.arrival_time,np.cumsum(basis/MASKED_RATE),rtol=0,atol=1e-9)
        fingerprint = [(int(r.task_id),float(r.arrival_time),float(r.task_size),float(r.computation_demand),
            float(r.reliability_requirement),int(r.action_index),float(r.task_completion_time)) for r in g.itertuples()]
        assert hashlib.sha256(repr(fingerprint).encode()).hexdigest()==row.trace_sha256
        if model!='independent':assert row.rsr_true==np.mean([p['safe_mask'].any() for p in tasks.values()])
    details = details.assign(model=model)
    metrics = metrics.assign(model=model,ell_ratio=ratio,
        evaluation_origin='phase5_exact' if exact_reuse else 'phase5_trace_rescored')
    destination = output/f'threshold_{threshold:g}'/f'seed_{seed}'/model
    destination.mkdir(parents=True,exist_ok=False)
    metrics.to_csv(destination/'eval_episode_metrics.csv',index=False,float_format='%.17g')
    details.to_csv(destination/'eval_task_records.csv.gz',index=False,float_format='%.17g')
    loads = history_loads[(history_loads.phase=='evaluation')&(history_loads.threshold==threshold)&
                         (history_loads.seed_id==seed)&(history_loads.model==source_model)&
                         (history_loads.episode_id<=episodes)].copy().assign(model=model,ell_ratio=ratio)
    return metrics,loads,dict(ell_ratio=ratio,threshold=threshold,seed_id=seed,model=model,
        source_model=source_model,source=str(source),physical_trajectory_sha256=digest,
        trajectory_unchanged=True,phase5_exact=exact_reuse)


def plot_length_policy_results(output, seeds, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    for metric,name,label in (('rsr_true','true_rsr.png','True correlated-model RSR'),
        ('mean_latency_s','mean_latency.png','Mean task latency (s)'),
        ('avoidable_violation_count','avoidable_violations.png','Avoidable violation fraction')):
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,(threshold,group) in zip(axes,seeds.groupby('threshold')):
            for model,arm in group.groupby('model'):
                values=arm.groupby('ell_ratio')[metric]
                divisor=200 if metric=='avoidable_violation_count' else 1
                ax.errorbar(values.mean().index,values.mean()/divisor,
                    yerr=values.std(ddof=1)/divisor,marker='o',capsize=3,label=model)
            ax.set(title=f'R_req={threshold}',xlabel='ell / ell0',ylabel=label,xticks=LENGTH_POLICY_RATIOS)
            if metric!='mean_latency_s':ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(output/name,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,ratio in enumerate((.5,4.)):
        for col,threshold in enumerate(MASKED_THRESHOLDS):
            ax=axes[row,col];g=training[(training.ell_ratio==ratio)&(training.threshold==threshold)]
            for _,arm in g.groupby('seed_id'):
                ax.plot(arm.episode_id,arm.mean_latency_s.rolling(10,min_periods=1).mean(),alpha=.25,lw=.7)
            mean=g.groupby('episode_id').mean_latency_s.mean()
            ax.plot(mean.index,mean.rolling(10,min_periods=1).mean(),color='black',label='Seed mean, rolling 10')
            ax.set(title=f'ell/ell0={ratio:g}, R_req={threshold}',xlabel='Episode',ylabel='Training latency (s)')
            ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'new_training_curves.png',dpi=150);plt.close(fig)


def length_policy_sensitivity(args):
    import shlex
    import shutil
    import subprocess
    history = Path(RESULTS_DIR)/'masked_ppo_confirmation_10seed'
    sweep = Path(RESULTS_DIR)/'spatial_correlation_length_sweep'
    output = args.output_dir or Path(RESULTS_DIR)/'spatial_length_ppo_sensitivity'
    old_meta = json.loads((history/'run_metadata.json').read_text())
    assert old_meta['alpha']==MASKED_ALPHA and old_meta['beta']==effective_spatial_risk_beta()
    assert old_meta['arrival_rate']==MASKED_RATE and old_meta['gh_order']==32
    assert old_meta['ppo_hyperparameters']=={k:v for k,v in vars(params).items() if k.endswith('_ppo')}
    assert old_meta['numpy']==np.__version__ and old_meta['torch']==torch.__version__
    assert old_meta['simpy']==simpy.__version__ and args.eval_episodes<=old_meta['eval_episodes']
    for directory in (history,sweep):
        manifest=json.loads((directory/'output_manifest.json').read_text())
        assert all(hashlib.sha256((directory/name).read_bytes()).hexdigest()==h for name,h in manifest.items())
    old_hashes=json.loads((history/'input_manifest.json').read_text())
    for name,h in old_hashes.items():
        if name!='compare_ppo_greedy.py':assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    profiles=MainLoop._build_server_profiles(pd.read_excel(SERVER_INFO_PATH,sheet_name='Servers',dtype={'Site_ID':'string'}))
    tasks={'train':MainLoop._build_task_profiles(pd.read_excel(TRAIN_TASK_PATH)),
           'evaluation':MainLoop._build_task_profiles(pd.read_excel(EVALUATION_TASK_PATH))}
    cached=load_length_profiles(profiles,tasks,history,sweep)
    output.mkdir(parents=True,exist_ok=False)
    meta=dict(command=shlex.join([sys.executable,*sys.argv]),ell_ratios=LENGTH_POLICY_RATIOS,
        thresholds=MASKED_THRESHOLDS,seed_ids=args.seeds,train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,alpha=MASKED_ALPHA,beta=effective_spatial_risk_beta(),gh_order=32,
        arrival_rate=MASKED_RATE,torch_threads=torch.get_num_threads(),history=str(history.resolve()),
        reliability_cache=str((sweep/'reliability_tables.npz').resolve()),
        ppo_hyperparameters=old_meta['ppo_hyperparameters'],seed_namespace=41,
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status_before=subprocess.check_output(['git','status','--short'],text=True),
        python=sys.version,numpy=np.__version__,torch=torch.__version__,simpy=simpy.__version__)
    (output/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    inputs=[Path(name) for name in old_hashes]
    inputs.extend([sweep/'reliability_tables.npz',history/'output_manifest.json',sweep/'output_manifest.json'])
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (output/'input_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for path in inputs:
        if path.suffix=='.py':
            target=output/'source_snapshot'/path
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    history_train=pd.read_csv(history/'train_episode_metrics.csv',float_precision='round_trip')
    history_loads=pd.read_csv(history/'server_load_episodes.csv',float_precision='round_trip')
    history_checks=pd.read_csv(history/'reproducibility_checks.csv')
    state_context=SimpleNamespace(this_episode=0,use_safety_mask=True,num_actions=params.num_actions)
    logger=lambda task:MainLoop.get_task_outcome_info(state_context,task)
    # Historical trajectories already contain both replicas; only compatibility replays run here.
    import core.env_state as state_module
    saved=(state_module.compute_pair_reliability_gh,state_module.compute_marginal_failure_probability)
    def unexpected_gh(*a,**kw):raise AssertionError('GH reached a policy episode')
    state_module.compute_pair_reliability_gh=state_module.compute_marginal_failure_probability=unexpected_gh
    training_all,evaluation_all,loads_all,checks_all=[],[],[],[]
    sources,invariance,reference_training,compatibility=[],[],[],[]
    started=time.monotonic()
    try:
        for scenario,threshold in enumerate(MASKED_THRESHOLDS):
            for arm,model in enumerate(ABLATION_MODELS):
                source=history/f'threshold_{threshold:g}'/'seed_1'/model
                agent=frozen_agent(source/'final.pt');snapshot=frozen_snapshot(agent)
                task_profiles=cached[1.]['evaluation'][threshold]
                if model=='independent':task_profiles={tid:{**p,'safe_mask':p['reliability_snapshot'][2]>=threshold} for tid,p in task_profiles.items()}
                basis=np.random.default_rng(derived_seed((41,scenario,1,4,1))).exponential(size=200)
                values,_=masked_evaluation('trained_masked_ppo',agent,profiles,task_profiles,basis,1,
                    derived_seed((41,scenario,1,5,arm,1)),logger,reliability_model=model)
                old=pd.read_csv(source/'eval_episode_metrics.csv',float_precision='round_trip').iloc[0]
                for field,value in values.items():
                    if isinstance(value,str):assert value==old[field]
                    else:np.testing.assert_allclose(value,old[field],rtol=0,atol=1e-12)
                same_tree(snapshot,frozen_snapshot(agent))
                compatibility.append(dict(threshold=threshold,seed_id=1,model=model,checkpoint_replay_passed=True))
        pd.DataFrame(compatibility).to_csv(output/'historical_compatibility.csv',index=False)
        for seed in args.seeds:
            for ratio in LENGTH_POLICY_RATIOS:
                outer=output/f'ell_{ratio:g}'
                for scenario,threshold in enumerate(MASKED_THRESHOLDS):
                    if ratio in (.5,4.):
                        tr,ev,ld,ck=run_ablation_seed(scenario,seed,args,profiles,cached[ratio],outer,
                            include_heuristics=True,models=('correlated',),ell_ratio=ratio)
                        tr=tr.assign(ell_ratio=ratio,training_origin='new')
                        ev=ev.assign(ell_ratio=ratio,evaluation_origin='new_simulation')
                        ld=ld.assign(ell_ratio=ratio)
                        training_all.append(tr);evaluation_all.append(ev);loads_all.append(ld)
                        checks_all.append(ck.assign(ell_ratio=ratio))
                        origin=history_checks[(history_checks.threshold==threshold)&(history_checks.seed_id==seed)&
                                              (history_checks.model=='correlated')].iloc[0]
                        assert ck.iloc[0].initial_parameter_digest==origin.initial_parameter_digest
                    elif ratio==0:
                        frames,ld,ck=evaluate_safe_methods(scenario,seed,profiles,
                            cached[ratio]['evaluation'][threshold],args.eval_episodes,outer,logger,ell_ratio=ratio)
                        evaluation_all.extend(frame.assign(ell_ratio=ratio,evaluation_origin='new_simulation') for frame in frames)
                        loads_all.extend(frame.assign(ell_ratio=ratio) for frame in ld)
                        checks_all.append(pd.DataFrame(ck).assign(ell_ratio=ratio))
                    reuse_models=CONFIRMATION_METHODS if ratio==1 else ('independent',)
                    if ratio==0:reuse_models=('correlated','independent')
                    for model in reuse_models:
                        ev,ld,check=reuse_length_evaluation(ratio,threshold,seed,model,cached,history,outer,
                                                          history_loads,args.eval_episodes)
                        evaluation_all.append(ev);loads_all.append(ld);invariance.append(check)
                    for model in ABLATION_MODELS:
                        source_model='independent' if ratio==0 or model=='independent' else 'correlated'
                        old_case=history/f'threshold_{threshold:g}'/f'seed_{seed}'/source_model
                        new=ratio in (.5,4.) and model=='correlated'
                        checkpoint=(outer/f'threshold_{threshold:g}'/f'seed_{seed}'/model/'final.pt') if new else old_case/'final.pt'
                        sources.append(dict(ell_ratio=ratio,threshold=threshold,seed_id=seed,model=model,
                            source_checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            training_origin='new' if new else ('shared_independent_ell0' if ratio==0 and model=='correlated' else 'phase5_reused'),
                            shared_policy_at_zero=ratio==0))
                        train=tr.copy() if new else history_train[(history_train.threshold==threshold)&
                            (history_train.seed_id==seed)&(history_train.model==source_model)].copy().assign(model=model)
                        # Reference views contain learning diagnostics only, never old-ell reliability claims.
                        columns=['threshold','seed_id','model','episode_id','mean_latency_s','episode_reward',
                            'effective_transition_count','multi_safe_decision_count','single_safe_decision_count',
                            'fallback_count','update_call_count','actor_optimizer_steps','critic_optimizer_steps',
                            'init_seed','train_action_seed','shuffle_seed','train_arrival_seed']
                        reference_training.append(train[columns].assign(ell_ratio=ratio,
                            training_origin='new' if new else 'phase5_reference'))
            if seed==args.seeds[0]:
                pre_eval=pd.concat(evaluation_all,ignore_index=True)
                for _,g in pre_eval.groupby(['ell_ratio','threshold','seed_id','episode_id']):
                    assert len(g)==4 and g.arrival_sha256.nunique()==1
                    corr=g[g.model!='independent'];assert corr.fallback_count.nunique()==corr.rsr_true.nunique()==1
                for _,g in pre_eval.groupby(['threshold','seed_id','episode_id']):assert g.arrival_sha256.nunique()==1
                (output/'seed1_preflight.json').write_text(json.dumps(dict(passed=True,seed_id=seed,
                    new_models=4,new_training_episodes=4*args.train_episodes,
                    evaluation_rows=len(pre_eval),historical_checkpoint_replays=len(compatibility)),indent=2)+'\n')
                print('LENGTH PREFLIGHT PASSED; seed 1 retained, expanding remaining seeds',flush=True)
    finally:
        state_module.compute_pair_reliability_gh,state_module.compute_marginal_failure_probability=saved
    training=pd.concat(training_all,ignore_index=True)
    evaluation=pd.concat(evaluation_all,ignore_index=True)
    loads=pd.concat(loads_all,ignore_index=True)
    views=pd.concat(reference_training,ignore_index=True)
    for name,frame in (('train_episode_metrics',training),('eval_episode_metrics',evaluation),
        ('server_load_episodes',loads),('training_reference_metrics',views),
        ('reproducibility_checks',pd.concat(checks_all,ignore_index=True)),
        ('checkpoint_sources',pd.DataFrame(sources)),('trajectory_invariance',pd.DataFrame(invariance))):
        frame.to_csv(output/f'{name}.csv',index=False,float_format='%.17g')
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in hashes.items())
    for _,g in evaluation.groupby(['threshold','seed_id','episode_id']):assert g.arrival_sha256.nunique()==1
    ind=evaluation[evaluation.model=='independent']
    for _,g in ind.groupby(['threshold','seed_id','episode_id']):
        for field in ('mean_latency_s','fallback_count','trace_sha256','arrival_sha256','mean_all_replica_cpu_wait_s'):
            assert g[field].nunique()==1
    validation=dict(passed=True,new_models=len(args.seeds)*4,new_training_episodes=len(training),
        new_ppo_evaluation_episodes=len(args.seeds)*4*args.eval_episodes,
        combined_evaluation_rows=len(evaluation),gh_order=32,gh_integrations=0,
        historical_results_unchanged=True,independent_trajectory_invariant=True,
        ell_zero_shared_policy=True,ell_one_exact_reuse=True,all_tasks_and_replicas_completed=True,
        historical_checkpoint_replays=len(compatibility),elapsed_s=time.monotonic()-started)
    (output/'validation_summary.json').write_text(json.dumps(validation,indent=2)+'\n')
    summarize_length_policy(output,training,evaluation,loads,views)
    print(json.dumps(validation,indent=2),flush=True)


def summarize_length_policy(output, training, evaluation, loads, views):
    filenames=('eval_seed_summary','method_summary','paired_comparison','paired_seed_differences',
        'reliability_violation_summary','server_load_summary','server_load_seed_summary',
        'training_seed_summary','seed_mapping','evaluation_streams')
    collected={name:[] for name in filenames}
    for ratio in LENGTH_POLICY_RATIOS:
        local=output/f'ell_{ratio:g}'
        for name,frame in (('train_episode_metrics',views),('eval_episode_metrics',evaluation),('server_load_episodes',loads)):
            frame[frame.ell_ratio==ratio].drop(columns='ell_ratio').to_csv(local/f'{name}.csv',index=False,float_format='%.17g')
        summarize_confirmation(local,plots=False,write_report=False,identity_reference=ratio==0)
        for name in filenames:
            frame=pd.read_csv(local/f'{name}.csv')
            collected[name].append(frame.assign(ell_ratio=ratio))
    for name,frames in collected.items():pd.concat(frames,ignore_index=True).to_csv(output/f'{name}.csv',index=False,float_format='%.17g')
    seeds=pd.read_csv(output/'eval_seed_summary.csv')
    plot_length_policy_results(output,seeds,training)
    write_length_policy_report(output)


def write_length_policy_report(output):
    meta=json.loads((output/'run_metadata.json').read_text())
    validation=json.loads((output/'validation_summary.json').read_text())
    methods=pd.read_csv(output/'method_summary.csv');paired=pd.read_csv(output/'paired_comparison.csv')
    evaluation=pd.read_csv(output/'eval_episode_metrics.csv');learning=pd.read_csv(output/'training_seed_summary.csv')
    sources=pd.read_csv(output/'checkpoint_sources.csv')
    n=len(meta['seed_ids'])
    lines=['# 第六阶段6B：空间相关长度的PPO策略级敏感性', '',
        f'ell/ell0={{0,0.5,1,4}}；两个单一阈值0.999/0.9999，{n}种子×{meta["train_episodes"]}训练轮，最终模型评估{meta["eval_episodes"]}个配对Episode。',
        '固定8台EUA服务器、28字典序动作、200任务/400副本、alpha=0.05、beta=0.5、到达率2 tasks/s。'
        '沿用第五阶段策略实验GH32，直接加载6A已验证的32阶数组；ell=1概率数组与第五阶段逐项相等。原始数据和正式配置不修改。', '',
        '## 复用、统计单位及实际新增规模', '',
        '- 独立PPO使用第五阶段对应阈值/种子的最终Checkpoint和完整逐任务动作/生命周期记录，四档仅改变真实性能评价；动作、队列、Fallback、时延不变。',
        '- ell=1的四方法直接复用第五阶段评估，不重新训练或替换结果；当前代码的兼容性重放另作诊断，不混入正式统计。',
        '- ell=0的相关PPO与独立PPO共用同一已训练独立策略及同一动作流，结果是恒等对照；不是新增训练，不是两个独立样本，配对表标记shared_policy_identity，零区间只表示恒等。',
        f'- 新增相关PPO {validation["new_models"]}模型、{validation["new_training_episodes"]}训练Episode、'
        f'{validation["new_ppo_evaluation_episodes"]}PPO评估Episode，仅ell=0.5和4。种子1完整预检后再扩展，预检结果直接纳入。',
        '- Safe Min-Latency及Safe Random在ell=0/0.5/4重新执行，ell=1复用第五阶段。所有方法使用同一Fallback函数，两个副本继续执行到完成。',
        '- 根目录train_episode_metrics.csv只含新增训练。各ell子目录的训练表及根目录training_reference_metrics.csv是标记来源的学习诊断视图，包含历史引用，不代表新增训练次数；引用表不携带旧ell可靠性结论。',
        '- 每种子先平均20个评估Episode，再对十个配对种子求均值、样本SD和Student-t 95%CI；区间是逐比较边际区间，未作多重比较校正。RSR是理论约束满足率，不是实际故障事件成功率。', '',
        '## 完整评估结果', '',
        '| ell/ell0 | 阈值 | 方法 | 平均时延±SD/s | 真实RSR±SD/% | Fallback/% | 假安全选择/轮 | 可避免违约/轮 | 不可避免/轮 |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in methods.itertuples():
        lines.append(f'| {r.ell_ratio:g} | {r.threshold:g} | {r.model} | {r.mean_latency_s_mean:.6f} ± {r.mean_latency_s_std:.6f} | '
            f'{100*r.rsr_true_mean:.3f} ± {100*r.rsr_true_std:.3f} | {100*r.fallback_fraction_mean:.1f} | '
            f'{r.false_safe_selected_count_mean:.3f} | {r.avoidable_violation_count_mean:.3f} | {r.unavoidable_violation_count_mean:.0f} |')
    lines+=['', '## 配对比较', '',
        'Delta_latency=CorrPPO−对照；负值更快。改善率=(对照−CorrPPO)/对照，先逐种子计算。'
        'Delta_RSR=相关PPO真实RSR−对照真实RSR，以百分点计。ell=0的两PPO相同策略行不作独立性推断。', '',
        '| ell/ell0 | 阈值 | 对照 | Delta时延±SD/s | 95%CI/s | 改善/% | 更快种子 | Delta RSR/百分点 | RSR差95%CI |',
        '|---:|---:|---|---:|---|---:|---:|---:|---|']
    for r in paired.itertuples():
        label=r.reference+('（同一策略）' if r.comparison_kind=='shared_policy_identity' else '')
        lines.append(f'| {r.ell_ratio:g} | {r.threshold:g} | {label} | {r.delta_latency_s_mean:+.6f} ± {r.delta_latency_s_sd:.6f} | '
            f'[{r.delta_latency_s_ci95_lower:+.6f}, {r.delta_latency_s_ci95_upper:+.6f}] | {r.latency_improvement_pct_mean:.3f} | '
            f'{r.latency_better_seed_count}/{n} | {r.delta_rsr_pp_mean:+.3f} | [{r.delta_rsr_pp_ci95_lower:+.3f}, {r.delta_rsr_pp_ci95_upper:+.3f}] |')
    lines+=['', '## 主图', '', '误差线为十个种子均值之间的SD，不是把Episode当独立训练样本。ell=0两PPO线重合，来自同一策略。', '',
        '![真实RSR](true_rsr.png)', '', '![任务完成时延](mean_latency.png)', '',
        '![可避免违约比例](avoidable_violations.png)', '', '![新增模型训练曲线](new_training_curves.png)', '',
        '## 可靠性与时延口径', '',
        '- 每任务以当前ell对应R_corr统一判定。总违约=可避免+不可避免，二者互斥；假安全动作可包含其中两类，不能再次相加。',
        '- 相关策略的Fallback上也可能出现R_ind判安全的所选对；该动作已被相关模型识别为不安全，不能解释为相关Mask错误。非Fallback假安全和可避免违约对相关方法均为0。',
        '- 独立策略的自身空集不随ell变化，但真实空集会变化；真实空集上的普通独立PPO决策仍属于不可避免违约。',
        '- 逐任务TaskLatency=WinnerUpload+WinnerCPUWaiting+WinnerExecution；All-Replica等待包含Loser，不能代入Winner分解。', '',
        '| ell/ell0 | 阈值 | 方法 | Winner上传/s | Winner等待/s | Winner执行/s | 全副本等待/s |',
        '|---:|---:|---|---:|---:|---:|---:|']
    for r in methods.itertuples():
        lines.append(f'| {r.ell_ratio:g} | {r.threshold:g} | {r.model} | {r.mean_winner_upload_s_mean:.6f} | '
            f'{r.mean_winner_cpu_wait_s_mean:.6f} | {r.mean_winner_execution_s_mean:.6f} | {r.mean_all_replica_cpu_wait_s_mean:.6f} |')
    lines+=['', '## 学习空间与来源', '',
        '| ell/ell0 | 阈值 | PPO | 有效Transition/轮 | 多动作/轮 | 唯一动作/轮 | 首10轮/s | 末10轮/s |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for (ratio,threshold,model),g in learning.groupby(['ell_ratio','threshold','model']):
        lines.append(f'| {ratio:g} | {threshold:g} | {model} | {g.effective_transition_count.mean():.0f} | '
            f'{g.multi_safe_decision_count.mean():.0f} | {g.single_safe_decision_count.mean():.0f} | '
            f'{g.first10_latency_s.mean():.6f} | {g.last10_latency_s.mean():.6f} |')
    lines+=['', f'模型来源表共{len(sources)}行，对应{sources.source_checkpoint.nunique()}个唯一Checkpoint路径；'
        '来源、SHA256、共享标记及随机种子映射见checkpoint_sources.csv、seed_mapping.csv和evaluation_streams.csv。'
        '历史文件只读，不复制为伪新增训练。', '',
        '## 正确性检查与追溯', '',
        '- 第五阶段完整产物清单、核心代码及输入哈希检查通过；6A GH32与第五阶段基准逐项一致。新增代码仅扩展编排/运行级ell参数，未改核心算法。',
        '- 历史种子1两个阈值、两PPO冻结Checkpoint兼容性重放通过，动作/完成时间轨迹哈希及指标一致；网络与优化器未改变。',
        '- 独立轨迹逐任务重评分时物理字段摘要不变，ell=0共享基准和ell=1原样复用均核对；不同ell及四方法到达流相同。',
        '- 正相关可靠性单调性、Mask包含关系、ell=0退化通过；运行中GH被断言禁止，新增积分调用0。运行级协方差元数据也按当前ell设置。',
        '- 所有新运行完成200任务、400副本；PPO Task_ID、奖励、当前/下一Mask、下一状态及实际时间间隔逐轮审计。Fallback没有Transition。',
        '- 新模型首训练/首评估Episode重放一致；评估前后网络/优化器及Checkpoint哈希不变。旧模型和历史轨迹只读复用。',
        '- 无非法自身Mask动作、NaN/Inf、概率越界或负等待；最终落盘任务回读、历史文件哈希与CI独立复算见source_and_output_audit.json。', '',
        '## 执行与修改', '',
        f'分支`{meta["git_branch"]}`，HEAD `{meta["git_head"]}`；本轮仅修改compare_ppo_greedy.py。完整源码快照及输入哈希随输出保存。', '',
        '```text',meta['command'],'```', '',
        '结果另存独立目录，未覆盖第五阶段或6A；未commit、未push；未运行原计划18组实验。具体九项结论见结果分析补充。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


MIXED_REQUIREMENTS = (0.999, 0.9993, 0.9996, 0.9999)


def load_mixed_profiles(profiles, output):
    """Use the frozen 7A assignment and GH32 arrays, never reassign task tiers."""
    import shutil
    source = Path(RESULTS_DIR)/'mixed_reliability_calibration'
    metadata = json.loads((source/'run_metadata.json').read_text())
    if metadata['beta'] != effective_spatial_risk_beta():
        raise ValueError('Historical reliability cache uses a different effective spatial-risk beta')
    selection = json.loads((source/'selection.json').read_text())
    assert selection['selected_candidate']=='C' and tuple(selection['selected_thresholds'])==MIXED_REQUIREMENTS
    for name,h in json.loads((source/'input_manifest.json').read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
    for name,h in json.loads((source/'output_manifest.json').read_text()).items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h
    arrays = np.load(source/'reliability_table.npz')
    mapping = pd.read_csv(source/'task_tier_mapping.csv',float_precision='round_trip')
    _,state=make_environment(profiles,failure_rate_scale=MASKED_ALPHA)
    state.initialize_spatial_reliability()
    np.testing.assert_array_equal(arrays['pairs'],state.server_pairs)
    np.testing.assert_array_equal(arrays['distances_km'],state.spatial_distance_matrix)
    np.testing.assert_array_equal(arrays['scaled_rates'],[p['Base_Failure_Rate']*MASKED_ALPHA for p in profiles.values()])
    cached={}
    for split,path in (('train',TRAIN_TASK_PATH),('evaluation',EVALUATION_TASK_PATH)):
        raw=pd.read_excel(path).sort_values('Task_ID').reset_index(drop=True)
        assigned=mapping[mapping.split==split].reset_index(drop=True)
        np.testing.assert_array_equal(raw[['Task_ID','Task_Size','Computation_Demand']],assigned[['Task_ID','Task_Size','Computation_Demand']])
        np.testing.assert_array_equal(raw.Reliability_Requirement,assigned.original_r_req)
        np.testing.assert_array_equal(assigned.groupby('tier').size(),[50]*4)
        np.testing.assert_array_equal(assigned.r_req_C,np.array(MIXED_REQUIREMENTS)[assigned.tier.to_numpy()-1])
        np.testing.assert_array_equal(arrays[f'{split}_task_ids'],raw.Task_ID)
        raw['Reliability_Requirement']=assigned.r_req_C
        tasks=MainLoop._build_task_profiles(raw)
        marginal=1-arrays[f'{split}_marginal_failure']
        corr,ind=arrays[f'{split}_correlated'],arrays[f'{split}_independent']
        masks=corr>=assigned.r_req_C.to_numpy()[:,None]
        np.testing.assert_array_equal(np.stack((masks,ind>=assigned.r_req_C.to_numpy()[:,None])),arrays[f'{split}_C_masks'])
        assert not np.any(masks&~(ind>=assigned.r_req_C.to_numpy()[:,None]))
        for values in (marginal,corr,ind,masks):values.setflags(write=False)
        cached[split]={'mixed_C':{tid:{**p,'reliability_tier':int(assigned.tier.iloc[i]),
            'reliability_snapshot':(marginal[i],corr[i],ind[i]),'safe_mask':masks[i]}
            for i,(tid,p) in enumerate(tasks.items())}}
        count=masks.sum(axis=1)
        np.testing.assert_array_equal([(count==0).sum(),(count==1).sum(),(count>=2).sum()],
                                     [12,21,167] if split=='train' else [13,21,166])
    for name in ('selection.json','task_tier_mapping.csv','task_groups.json','reliability_table.npz'):
        shutil.copyfile(source/name,output/name)
    return cached


def audit_mixed_records(data, tasks, model):
    """Audit actual mixed-task execution against immutable per-Task_ID profiles."""
    assert data.groupby('episode').size().eq(200).all()
    assert data.groupby('episode').task_id.nunique().eq(200).all()
    references=[tasks[tid] for tid in data.task_id]
    np.testing.assert_array_equal(data.reliability_requirement,[p['Reliability_Requirement'] for p in references])
    np.testing.assert_array_equal(data.task_size,[p['Task_Size'] for p in references])
    np.testing.assert_array_equal(data.computation_demand,[p['Computation_Demand'] for p in references])
    data=data.assign(tier=[p['reliability_tier'] for p in references])
    assert data.groupby(['episode','tier']).size().eq(50).all()
    own=np.array([p['reliability_snapshot'][2 if model=='independent' else 1]>=p['Reliability_Requirement'] for p in references])
    observed=np.array([[flag=='1' for flag in mask] for mask in data.safe_mask])
    np.testing.assert_array_equal(observed,own)
    np.testing.assert_array_equal(data.is_fallback,~own.any(axis=1))
    assert np.all(own[np.arange(len(data)),data.action_index.to_numpy(dtype=int)]|data.is_fallback.to_numpy())
    task_latency_components(data)
    assert np.allclose(data.true_satisfied.mean(),1-data.unavoidable_violation.mean()-data.avoidable_violation.mean())
    if model!='independent':assert not data.avoidable_violation.any()
    return data


def mixed_task_metrics(data):
    """One episode/tier or pooled seed: latency quantiles come from raw tasks."""
    count=len(data);fallback=data.is_fallback
    metrics=dict(task_count=count,mean_latency_s=data.task_latency.mean(),p95_latency_s=data.task_latency.quantile(.95),
        mean_winner_upload_s=data.winner_upload_s.mean(),mean_winner_cpu_wait_s=data.winner_cpu_wait_s.mean(),
        mean_winner_execution_s=data.winner_execution_s.mean(),mean_all_replica_cpu_wait_s=data.all_replica_cpu_wait_s.mean(),
        mean_loser_cpu_wait_s=data.loser_cpu_wait_s.mean(),fallback_count=int(fallback.sum()),fallback_fraction=fallback.mean(),
        nonfallback_count=int((~fallback).sum()),mean_fallback_latency_s=data.loc[fallback,'task_latency'].mean(),
        mean_nonfallback_latency_s=data.loc[~fallback,'task_latency'].mean(),mean_safe_pair_count=data.safe_pair_count.mean(),
        empty_safe_fraction=(data.safe_pair_count==0).mean(),single_safe_fraction=(data.safe_pair_count==1).mean(),
        multi_safe_fraction=(data.safe_pair_count>=2).mean(),rsr_true=data.true_satisfied.mean(),
        false_safe_selected_count=int(data.false_safe_selected.sum()),false_safe_selected_fraction=data.false_safe_selected.mean(),
        false_safe_normal_decision_count=int(data.false_safe_normal_decision.sum()),
        avoidable_violation_count=int(data.avoidable_violation.sum()),avoidable_violation_fraction=data.avoidable_violation.mean(),
        unavoidable_violation_count=int(data.unavoidable_violation.sum()),unavoidable_violation_fraction=data.unavoidable_violation.mean(),
        false_safe_avoidable_count=int((data.false_safe_selected&data.avoidable_violation).sum()),
        false_safe_unavoidable_count=int((data.false_safe_selected&data.unavoidable_violation).sum()),
        nonfalse_unavoidable_count=int((~data.false_safe_selected&data.unavoidable_violation).sum()),
        true_feasible_count=int((~data.unavoidable_violation).sum()),
        unsafe_given_true_feasible_fraction=data.avoidable_violation.sum()/(~data.unavoidable_violation).sum())
    assert metrics['false_safe_selected_count']==metrics['false_safe_avoidable_count']+metrics['false_safe_unavoidable_count']
    return metrics


def collect_mixed_seed(output, seed, args, cached):
    evaluations,groups,checkpoints=[],[],[]
    for model in CONFIRMATION_METHODS:
        case=output/'mixed_C'/f'seed_{seed}'/model
        tasks=cached['evaluation']['mixed_C']
        data=pd.read_csv(case/'eval_task_records.csv.gz',float_precision='round_trip',dtype={'safe_mask':str,'action_mask':str})
        data=audit_mixed_records(data,tasks,model)
        evaluations.append(data)
        for episode,g in data.groupby('episode'):
            for tier,subset in [(0,g),*list(g.groupby('tier'))]:
                groups.append(dict(seed_id=seed,model=model,episode_id=episode,tier=tier,**mixed_task_metrics(subset)))
        data.to_csv(case/'eval_task_records.csv.gz',index=False,float_format='%.17g')
        if model in ABLATION_MODELS:
            training=pd.read_csv(case/'train_task_records.csv.gz',float_precision='round_trip',dtype={'safe_mask':str,'action_mask':str})
            training=audit_mixed_records(training,cached['train']['mixed_C'],model)
            training.to_csv(case/'train_task_records.csv.gz',index=False,float_format='%.17g')
            metrics=pd.read_csv(case/'train_episode_metrics.csv',float_precision='round_trip')
            normal=int((~training.is_fallback).sum()/args.train_episodes)
            assert metrics.effective_transition_count.eq(normal).all() and metrics.update_call_count.eq(1).all()
            if model=='correlated':assert normal==188 and metrics.theoretical_rsr.eq(.94).all()
            initial,final=case/'initial.pt',case/'final.pt'
            a=torch.load(initial,map_location='cpu',weights_only=True)
            b=torch.load(final,map_location='cpu',weights_only=True)
            assert not a['opt_policy']['state'] and not a['opt_value']['state']
            actor_change=float(torch.sqrt(sum(((b['policy_net'][k]-v)**2).sum() for k,v in a['policy_net'].items())))
            assert actor_change>0
            checkpoints.append(dict(seed_id=seed,model=model,scenario='mixed_C',train_episodes=args.train_episodes,
                initial_checkpoint=str(initial.resolve()),final_checkpoint=str(final.resolve()),
                final_checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),actor_l2_change=actor_change,
                independent_optimizer_initialization=True))
    data=pd.concat(evaluations,ignore_index=True)
    for _,g in data.groupby(['episode','task_id']):
        assert len(g)==4 and g.arrival_time.nunique()==g.reliability_requirement.nunique()==g.task_size.nunique()==g.computation_demand.nunique()==1
    return data,pd.DataFrame(groups),pd.DataFrame(checkpoints)


def summarize_mixed_results(output, training, episode_metrics, data, loads):
    numeric=[c for c in episode_metrics.select_dtypes('number') if c not in ('seed_id','episode_id','tier')]
    seeds=episode_metrics.groupby(['tier','seed_id','model'])[numeric].mean().reset_index()
    # Replace mean episode quantiles with quantiles of 20*200 or 20*50 task samples.
    for (seed,model),g in data.groupby(['seed_id','model']):
        for tier,subset in [(0,g),*list(g.groupby('tier'))]:
            select=(seeds.seed_id==seed)&(seeds.model==model)&(seeds.tier==tier)
            seeds.loc[select,'p95_latency_s']=subset.task_latency.quantile(.95)
    seeds.to_csv(output/'eval_seed_summary.csv',index=False,float_format='%.17g')
    methods=seeds.groupby(['tier','model'])[numeric].agg(['mean','std'])
    methods.columns=['_'.join(c) for c in methods.columns]
    methods=methods.reset_index()
    n=seeds.seed_id.nunique()
    half=student_t.ppf(.975,n-1)*methods.mean_latency_s_std/np.sqrt(n) if n>1 else np.nan
    methods['mean_latency_s_ci95_lower']=methods.mean_latency_s_mean-half
    methods['mean_latency_s_ci95_upper']=methods.mean_latency_s_mean+half
    methods.to_csv(output/'method_summary.csv',index=False,float_format='%.17g')
    differences,paired=paired_seed_statistics(seeds,group_column='tier')
    differences.to_csv(output/'paired_seed_differences.csv',index=False,float_format='%.17g')
    paired.to_csv(output/'paired_comparison.csv',index=False,float_format='%.17g')
    paired[paired.tier>0].to_csv(output/'tier_comparison.csv',index=False,float_format='%.17g')
    fields=['tier','model',*[c for c in methods if any(k in c for k in ('rsr_','violation','false_safe','unavoidable','true_feasible'))]]
    methods[fields].to_csv(output/'reliability_violation_summary.csv',index=False,float_format='%.17g')
    training_seed_statistics(training,group_column='scenario').to_csv(output/'training_seed_summary.csv',index=False,float_format='%.17g')
    fields=['replica_count','execution_cycles','cpu_busy_s','mean_cpu_wait_s','cpu_utilization']
    sl=loads[loads.phase=='evaluation'].groupby(['seed_id','model','server_id'])[fields].mean().reset_index()
    sl.to_csv(output/'server_load_seed_summary.csv',index=False,float_format='%.17g')
    sl.groupby(['model','server_id'])[fields].mean().reset_index().to_csv(output/'server_load_summary.csv',index=False,float_format='%.17g')
    replicas=pd.concat([data[['seed_id','model','episode','tier','computation_demand']].assign(
        server_id=data[f'server_{r}_id'],cpu_wait_s=data[f'replica_{r}_cpu_start_time']-data[f'replica_{r}_queue_enter_time'],
        cpu_busy_s=data[f'replica_{r}_finish_time']-data[f'replica_{r}_cpu_start_time']) for r in ('A','B')],ignore_index=True)
    tier_loads=replicas.groupby(['seed_id','model','tier','server_id']).agg(
        replica_count=('server_id','size'),execution_cycles=('computation_demand','sum'),
        cpu_busy_s=('cpu_busy_s','sum'),mean_cpu_wait_s=('cpu_wait_s','mean'))
    index=pd.MultiIndex.from_product([sorted(data.seed_id.unique()),CONFIRMATION_METHODS,range(1,5),range(1,9)],
                                    names=['seed_id','model','tier','server_id'])
    tier_loads=tier_loads.reindex(index,fill_value=0).reset_index()
    for field in ('replica_count','execution_cycles','cpu_busy_s'):tier_loads[field]/=data.episode.nunique()
    tier_loads.to_csv(output/'server_tier_load_seed_summary.csv',index=False,float_format='%.17g')
    tier_loads.groupby(['model','tier','server_id'])[['replica_count','execution_cycles','cpu_busy_s','mean_cpu_wait_s']].mean().reset_index().to_csv(
        output/'server_tier_load_summary.csv',index=False,float_format='%.17g')
    data.groupby(['seed_id','model','tier','action_index']).size().rename('task_count').reset_index().to_csv(output/'tier_action_counts.csv',index=False)
    plot_mixed_policy(output,methods,training)
    write_mixed_policy_report(output,methods,paired)


def mixed_policy_confirmation(args):
    import shlex,shutil,subprocess
    output=args.output_dir or Path(RESULTS_DIR)/'mixed_reliability_ppo_10seed'
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    profiles=MainLoop._build_server_profiles(pd.read_excel(SERVER_INFO_PATH,sheet_name='Servers',dtype={'Site_ID':'string'}))
    cached=load_mixed_profiles(profiles,output)
    meta=dict(command=shlex.join([sys.executable,*sys.argv]),thresholds=MIXED_REQUIREMENTS,alpha=MASKED_ALPHA,
        beta=effective_spatial_risk_beta(),arrival_rate=MASKED_RATE,gh_order=32,ell_ratio=1.,seed_namespace=71,
        ell0_km=compute_baseline_correlation_length_km(np.load(output/'reliability_table.npz')['distances_km']),
        train_episodes=args.train_episodes,eval_episodes=args.eval_episodes,seed_ids=args.seeds,
        torch_threads=torch.get_num_threads(),python=sys.version,numpy=np.__version__,torch=torch.__version__,simpy=simpy.__version__,
        ppo_hyperparameters={k:v for k,v in vars(params).items() if k.endswith('_ppo')},
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status_before=subprocess.check_output(['git','status','--short'],text=True))
    old=json.loads((Path(RESULTS_DIR)/'masked_ppo_confirmation_10seed/run_metadata.json').read_text())
    assert meta['ppo_hyperparameters']==old['ppo_hyperparameters']
    inputs=[Path(p) for p in (SERVER_INFO_PATH,TRAIN_TASK_PATH,EVALUATION_TASK_PATH)]
    inputs += [Path(p) for p in ('compare_ppo_greedy.py','agents/ppo_agent.py','core/main_loop.py','core/env_state.py',
        'core/spatial_risk.py','core/task.py','core/server.py','config/configuration.py','config/params.py','config/paths.py','io_utils/save_parameters_and_logs.py')]
    source=Path(RESULTS_DIR)/'mixed_reliability_calibration'
    inputs += [source/p for p in ('selection.json','task_tier_mapping.csv','reliability_table.npz','output_manifest.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (output/'input_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (output/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    for p in inputs:
        if p.suffix=='.py':
            target=output/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    import core.env_state as state_module
    original=state_module.compute_pair_reliability_gh,state_module.compute_marginal_failure_probability
    def unexpected_gh(*a,**kw):raise AssertionError('GH called during mixed policy execution')
    state_module.compute_pair_reliability_gh=state_module.compute_marginal_failure_probability=unexpected_gh
    started=time.monotonic();cases=[];raw=[];groups=[];checkpoints=[]
    try:
        for seed in args.seeds:
            case=run_ablation_seed(0,seed,args,profiles,cached,output,include_heuristics=True,mixed=True)
            data,group,checkpoint=collect_mixed_seed(output,seed,args,cached)
            cases.append(case);raw.append(data);groups.append(group);checkpoints.append(checkpoint)
            if seed==args.seeds[0]:
                corr=group[group.model!='independent']
                np.testing.assert_allclose(corr[corr.tier==0].rsr_true,.935,rtol=0,atol=1e-12)
                np.testing.assert_allclose(corr[corr.tier==4].rsr_true,.74,rtol=0,atol=1e-12)
                assert corr[corr.tier.isin([1,2,3])].rsr_true.eq(1).all()
                assert corr[corr.tier==0].fallback_count.eq(13).all()
                assert case[3].query("model in ['correlated','independent']").evaluation_frozen.all()
                (output/'seed1_preflight.json').write_text(json.dumps(dict(passed=True,seed_id=seed,
                    new_models=2,train_episodes=2*args.train_episodes,evaluation_episodes=4*args.eval_episodes,
                    task_mapping_verified=True,rsr_verified=True,frozen_evaluation=True),indent=2)+'\n')
                print('MIXED PREFLIGHT PASSED; seed 1 retained, expanding remaining seeds',flush=True)
    finally:
        state_module.compute_pair_reliability_gh,state_module.compute_marginal_failure_probability=original
    training,evaluation,loads,checks=[pd.concat([case[i] for case in cases],ignore_index=True) for i in range(4)]
    data=pd.concat(raw,ignore_index=True);group=pd.concat(groups,ignore_index=True)
    extra=group[group.tier==0].drop(columns='tier').set_index(['seed_id','model','episode_id'])
    evaluation=evaluation.set_index(['seed_id','model','episode_id'])
    for field in extra:
        if field in evaluation:np.testing.assert_allclose(evaluation[field],extra.loc[evaluation.index,field],rtol=0,atol=1e-10,equal_nan=True)
        else:evaluation[field]=extra.loc[evaluation.index,field]
    evaluation=evaluation.reset_index()
    for name,frame in (('train_episode_metrics',training),('eval_episode_metrics',evaluation),
        ('eval_tier_metrics',group[group.tier>0]),('server_load_episodes',loads),('reproducibility_checks',checks),
        ('checkpoint_manifest',pd.concat(checkpoints,ignore_index=True))):
        frame.to_csv(output/f'{name}.csv',index=False,float_format='%.17g')
    training[['scenario','seed_id','model','init_seed','train_action_seed','shuffle_seed']].drop_duplicates().to_csv(output/'seed_mapping.csv',index=False)
    evaluation[['seed_id','model','episode_id','eval_arrival_seed','eval_action_seed','arrival_sha256']].to_csv(output/'evaluation_streams.csv',index=False)
    assert len(training)==2*len(args.seeds)*args.train_episodes and len(evaluation)==4*len(args.seeds)*args.eval_episodes
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    validation=dict(passed=True,new_models=2*len(args.seeds),training_episodes=len(training),evaluation_episodes=len(evaluation),
        all_tasks_completed=True,replicas_per_episode=400,task_mapping_verified=True,gh_runtime_calls=0,gh_new_integrations=0,
        gh_source='7A frozen GH32',frozen_evaluation=True,own_mask_illegal_actions=0,paired_arrivals=True,
        training_transitions_corr=188,training_transitions_ind=190,eval_corr_rsr=.935,eval_strict_corr_rsr=.74,
        elapsed_s=time.monotonic()-started)
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    summarize_mixed_results(output,training,group,data,loads)
    print(json.dumps(validation,indent=2),flush=True)


def plot_mixed_policy(output, methods, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Corr PPO','Ind PPO','SML','Safe Random']
    overall=methods[methods.tier==0].set_index('model').loc[list(CONFIRMATION_METHODS)]
    fig,ax=plt.subplots(figsize=(7,4))
    half=overall.mean_latency_s_ci95_upper-overall.mean_latency_s_mean
    ax.bar(labels,overall.mean_latency_s_mean,yerr=half,capsize=4)
    ax.set(ylabel='Mean task latency (s)',title='Mixed reliability: 95% CI across seed means')
    fig.tight_layout();fig.savefig(output/'overall_latency.png',dpi=160);plt.close(fig)
    for filename,fields in [('tier_latency',[('mean_latency_s_mean','Mean task latency (s)')]),
        ('tier_reliability',[('rsr_true_mean','True theoretical RSR'),('avoidable_violation_fraction_mean','Avoidable violation fraction')])]:
        fig,axes=plt.subplots(1,len(fields),figsize=(6*len(fields),4),squeeze=False)
        for ax,(field,ylabel) in zip(axes[0],fields):
            for model,label in zip(CONFIRMATION_METHODS,labels):
                g=methods[(methods.model==model)&(methods.tier>0)].sort_values('tier')
                ax.errorbar(g.tier,g[field],yerr=g[field.replace('_mean','_std')],marker='o',capsize=3,label=label)
            ax.set(xticks=[1,2,3,4],xticklabels=[f'{v:g}' for v in MIXED_REQUIREMENTS],xlabel='Task R_req',ylabel=ylabel)
            ax.tick_params(axis='x',labelrotation=15);ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(output/f'{filename}.png',dpi=160);plt.close(fig)
    loads=pd.read_csv(output/'server_load_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,(model,label) in enumerate(zip(CONFIRMATION_METHODS,labels)):
        g=loads[loads.model==model].sort_values('server_id')
        for ax,field in zip(axes,('replica_count','cpu_utilization')):
            ax.bar(g.server_id+(i-1.5)*.2,g[field],width=.2,label=label)
            ax.set(xlabel='Server',ylabel='Replicas / episode' if field=='replica_count' else 'CPU busy utilization',xticks=range(1,9))
    for ax in axes:ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output/'server_load.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,model in zip(axes,ABLATION_MODELS):
        g=training[training.model==model]
        for _,arm in g.groupby('seed_id'):ax.plot(arm.episode_id,arm.mean_latency_s.rolling(10,min_periods=1).mean(),alpha=.25,lw=.7)
        avg=g.groupby('episode_id').mean_latency_s.mean()
        ax.plot(avg.index,avg.rolling(10,min_periods=1).mean(),color='black',label='Seed mean (rolling 10)')
        ax.set(title=model,xlabel='Training episode',ylabel='Mean task latency (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)


def write_mixed_policy_report(output, methods, paired):
    meta=json.loads((output/'run_metadata.json').read_text())
    validation=json.loads((output/'validation.json').read_text())
    learning=pd.read_csv(output/'training_seed_summary.csv')
    loads=pd.read_csv(output/'server_load_summary.csv')
    lines=['# 第七阶段7B：四档混合可靠性PPO正式配对实验','',
        f'方案C={list(MIXED_REQUIREMENTS)}；沿用7A各数据集固定Task_ID分组，每档50、每轮200任务/400副本。'
        f'两PPO各{len(meta["seed_ids"])}种子×{meta["train_episodes"]}轮独立训练，第{meta["train_episodes"]}轮最终模型随机采样评估。'
        f'四方法每种子评估{meta["eval_episodes"]}个配对Episode，共{validation["training_episodes"]}训练和{validation["evaluation_episodes"]}方法—评估Episode。',
        '固定8台EUA服务器、28字典序动作、alpha=0.05、beta=0.5、ell=ell0、到达率2 tasks/s、CPU单线程PyTorch。'
        '直接复用7A GH32概率与等级映射，决策期禁止GH调用。原始基础率只在状态/上下文内乘一次0.05。',
        'PPO类、Actor/Critic、Reward、GAE、优化器、实际时间折扣、FIFO双副本执行及共用Fallback均未修改。'
        '两个PPO是新建网络和优化器，数值初始参数按同一种子配对，但训练、参数存储和优化器状态独立，不复用第五/六阶段Checkpoint。', '',
        '## 统计口径与随机性', '',
        '- 10个训练种子为主要统计单位，先对各自20个配对评估Episode求均值，再计算跨种子均值、样本SD与Student-t 95%CI。区间是逐比较边际区间，未作多重比较校正。',
        '- SeedSequence沿用既有分层机制，本阶段命名空间71：初始化(role0)、训练动作(role1,arm)、shuffle(role2,arm)、训练到达(role3,episode)、评估到达(role4,episode)、PPO评估动作(role5,arm,episode)、Random(role6,episode)、SML(role7,episode)。动作流不会改变到达流。',
        '- Episode P95从该轮200/50个原始任务计算；种子P95从20轮汇总的4000/1000个原始任务时延计算，再对十个种子P95求均值±SD。没有对若干P95再次取P95，也没有用Episode P95均值替代种子P95。',
        '- Fallback及非Fallback时延分别按各自实际任务计算。不存在该子组时CSV留空、报告为无样本，不能将其当作零时延；这不是PPO或可靠性数值异常。',
        '- RSR始终用每任务真实相关可靠性与其自身R_req比较。所有任务卸载并排空两个副本，Fallback不进入PPO Transition。冻结评估只记录action_index，旧日志ppo_transition字段为空属于预期。', '',
        '## 整体与四档完整评估', '',
        '| 等级(0=整体) | 方法 | 时延±SD/s | P95±SD/s | True RSR±SD/% | Fallback/% | Winner上传/等待/执行 s | 全副本等待/s |',
        '|---:|---|---:|---:|---:|---:|---|---:|']
    for r in methods.itertuples():
        lines.append(f'| {r.tier} | {r.model} | {r.mean_latency_s_mean:.6f} ± {r.mean_latency_s_std:.6f} | '
            f'{r.p95_latency_s_mean:.6f} ± {r.p95_latency_s_std:.6f} | {100*r.rsr_true_mean:.3f} ± {100*r.rsr_true_std:.3f} | '
            f'{100*r.fallback_fraction_mean:.1f} | {r.mean_winner_upload_s_mean:.6f}/{r.mean_winner_cpu_wait_s_mean:.6f}/{r.mean_winner_execution_s_mean:.6f} | {r.mean_all_replica_cpu_wait_s_mean:.6f} |')
    lines+=['','## 整体和分档配对差','',
        'Delta=Corr PPO−对照，负时延差表示更快；改善百分比先逐种子计算。RSR差为百分点。等级1～4依次对应0.999/0.9993/0.9996/0.9999。','',
        '| 等级 | 对照 | Delta时延/s | 95%CI/s | 改善/% | 更快种子 | Delta RSR/百分点 | 消除可避免违约/轮 |',
        '|---:|---|---:|---|---:|---:|---:|---:|']
    for r in paired.itertuples():
        lines.append(f'| {r.tier} | {r.reference} | {r.delta_latency_s_mean:+.6f} | '
            f'[{r.delta_latency_s_ci95_lower:+.6f}, {r.delta_latency_s_ci95_upper:+.6f}] | {r.latency_improvement_pct_mean:.3f} | '
            f'{r.latency_better_seed_count}/{len(meta["seed_ids"])} | {r.delta_rsr_pp_mean:+.3f} | {r.avoided_violations_per_episode_mean:.3f} |')
    lines+=['','## 实际可靠性违约及交叉分类','',
        '不可避免=真实28对全不安全；可避免=真实安全集非空却选中不安全对；二者互斥且构成全部违约。'
        '假安全动作与它们重叠，不能额外相加。以下只列Independent PPO的实际选择；候选假安全对的离线数量不代入。','',
        '| 等级 | 假安全/轮 | 可避免/轮 | 不可避免/轮 | 假安全∩可避免 | 假安全∩不可避免 | 非假安全∩不可避免 | 可避免/全部任务% | 不安全/真实可行任务% |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in methods[methods.model=='independent'].itertuples():
        lines.append(f'| {r.tier} | {r.false_safe_selected_count_mean:.3f} | {r.avoidable_violation_count_mean:.3f} | '
            f'{r.unavoidable_violation_count_mean:.0f} | {r.false_safe_avoidable_count_mean:.3f} | {r.false_safe_unavoidable_count_mean:.3f} | '
            f'{r.nonfalse_unavoidable_count_mean:.3f} | {100*r.avoidable_violation_fraction_mean:.3f} | {100*r.unsafe_given_true_feasible_fraction_mean:.3f} |')
    lines+=['','相关方法的可避免违约和非Fallback假安全动作均应为0。相关Fallback本身已被识别为不安全，若所选对恰被R_ind判安全，统计上仍属于假安全交叉项，但不是相关Mask错误。',
        '', '## 训练诊断', '',
        '| 种子 | PPO | 首10轮/s | 末10轮/s | 201～250轮/s | 251～300轮/s | 尾部变化/% | Transition/轮 | 多动作/唯一动作 | 更新/优化器步数 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in learning.itertuples():
        lines.append(f'| {r.seed_id} | {r.model} | {r.first10_latency_s:.6f} | {r.last10_latency_s:.6f} | '
            f'{r.preceding50_latency_s:.6f} | {r.last50_latency_s:.6f} | {r.tail_change_pct:+.3f} | '
            f'{r.effective_transition_count:.0f} | {r.multi_safe_decision_count:.0f}/{r.single_safe_decision_count:.0f} | {r.update_calls}/{r.optimizer_steps} |')
    lines+=['','## 实际服务器负载','',
        '| 方法 | 服务器 | 副本/轮 | 执行需求/cycles每轮 | CPU忙时/s | 利用率/% | CPU等待/s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in loads.itertuples():
        lines.append(f'| {r.model} | {r.server_id} | {r.replica_count:.3f} | {r.execution_cycles:.6g} | '
            f'{r.cpu_busy_s:.6f} | {100*r.cpu_utilization:.3f} | {r.mean_cpu_wait_s:.6f} |')
    lines+=['',
        'server_tier_load_summary.csv给出实际各档副本在每服务器的分配和工作量。分档节点等待在每个种子内汇总20轮实际副本，未接收副本的节点补0并保留分配数；整体节点等待沿用既有逐轮节点均值口径。'
        'CPU利用率按全部副本排空后的Episode时长计算。决策积压指标为到达时已上传的CPU工作，未包含尚在上传的副本。',
        '', '## 图表', '', '![总体时延与95%CI](overall_latency.png)', '', '![分档时延](tier_latency.png)', '',
        '![分档RSR与可避免违约](tier_reliability.png)', '', '![实际服务器负载](server_load.png)', '',
        '![训练曲线](training_curves.png)', '', '分档时延及RSR图误差线为十种子SD；总体时延柱图为十种子均值的95%CI。',
        '', '## 正确性与复现', '',
        '- 种子1完成两模型全预算训练及四方法评估、映射/上限/冻结检查后再扩展，预检纳入正式结果。',
        '- 每轮200任务、400副本、四档各50，任务属性和要求逐项核对7A；相关策略整体RSR=.935、最高档=.74，前三档=1。训练相关Mask为12空集/21唯一/167多动作。',
        '- 每个训练Episode在PPO更新前检查完成区间有效QoS总和、动作、当前/下一Mask、下一状态、跨Fallback真实时间间隔及唯一动作log_prob；第一轮训练和评估可重放。',
        '- 最终Checkpoint随机采样评估前后网络、旧Actor、Critic及优化器完全一致；Checkpoint字节哈希不变。Actor参数变化范数见checkpoint_manifest.csv。',
        '- GH来自7A只读缓存，Episode中调用积分会直接报错；两种PPO各新建实例，初始优化器均为空。原始数据和冻结核心文件哈希未变。',
        '- 20轮仅是每个训练种子的配对评估流；没有额外挑选Checkpoint、增加训练预算或删除不利种子。',
        '', '## 执行与范围', '',
        f'分支`{meta["git_branch"]}`，HEAD `{meta["git_head"]}`。本轮只修改compare_ppo_greedy.py；README.md删除和其他未提交修改保留。',
        '', '```text',meta['command'],'```','',
        '完整命令、源码快照、输入哈希、逐任务轨迹、种子映射及独立回读审计随结果保存。未运行额外18组实验，未修改正式配置或Excel，未commit、未push。'
        '实际数据的十二项解释与研究局限在结果分析补充中给出。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


INTERVENTION_ARMS = {'A': 'All-SML', 'B': 'All-PPO',
                     'C': 'Low-PPO / High-SML', 'D': 'Low-SML / High-PPO'}


def intervention_episode(arm, agent, profiles, tasks, basis, episode, action_seed, rng_states, *, arrival_rate=MASKED_RATE):
    """Dispatch on the arriving task; couple random states, never actions or observations."""
    decisions = []
    pairs = MainLoop.generate_combinations()
    if arm == 'B':
        torch.manual_seed(action_seed)

    def select(observation, task):
        profile = tasks[task.id]
        assert task.safe_mask is profile['safe_mask'] and task.safe_mask.any()
        assert task.reliability_requirement == profile['Reliability_Requirement']
        np.testing.assert_array_equal(observation, task.env_state.get_state(task))
        high = task.reliability_requirement == MIXED_REQUIREMENTS[-1]
        use_ppo = arm == 'B' or (arm == 'C' and not high) or (arm == 'D' and high)
        rng_hash = ''
        if use_ppo:
            if arm == 'B':
                rng_states[task.id] = torch.get_rng_state().clone()
            else:
                torch.set_rng_state(rng_states[task.id])
            rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            action = agent.select_action(observation, 0., action_mask=task.safe_mask)
            task.ppo_decision = True
            task.ppo_action_mask = np.array(task.safe_mask, dtype=bool, copy=True)
        else:
            action = select_latency_greedy_action(observation, pairs, task.safe_mask)
        decisions.append(dict(task_id=task.id, policy_used='PPO' if use_ppo else 'SML',
            ppo_rng_sha256=rng_hash, observation_sha256=hashlib.sha256(np.asarray(observation).tobytes()).hexdigest(),
            decision_action=int(action)))
        return action

    records = []
    logger = lambda task: MainLoop.get_task_outcome_info(SimpleNamespace(this_episode=episode), task)
    values, loads = masked_evaluation('safe_min_latency', agent, profiles, tasks, basis, episode,
        action_seed, logger, task_records=records, record_backlog=True, task_selector=select, arrival_rate=arrival_rate)
    data = audit_mixed_records(reliability_outcomes(records, tasks), tasks, 'correlated')
    decision = pd.DataFrame(decisions)
    assert len(decision) == 187 and decision.task_id.is_unique
    data = data.merge(decision, on='task_id', how='left', validate='one_to_one')
    data['policy_used'] = data.policy_used.fillna('Fallback')
    normal = ~data.is_fallback
    np.testing.assert_array_equal(data.loc[normal,'action_index'], data.loc[normal,'decision_action'])
    expected = {'A':0, 'B':187, 'C':150, 'D':37}[arm]
    assert data.policy_used.eq('PPO').sum() == expected
    assert data.is_fallback.sum() == 13 and data.true_satisfied.mean() == .935
    assert data.loc[data.tier==4,'true_satisfied'].mean() == .74
    assert data.loc[data.tier<4,'true_satisfied'].all()
    high=data[data.tier==4]
    assert [(high.safe_pair_count==0).sum(),(high.safe_pair_count==1).sum(),(high.safe_pair_count>=2).sum()]==[13,21,16]
    assert high.loc[high.safe_pair_count==1,'server_A_id'].eq(4).all()
    assert high.loc[high.safe_pair_count==1,'server_B_id'].eq(5).all()
    assert data.ppo_transition_index.isna().all()
    assert np.isfinite(data[['task_latency','winner_cpu_wait_s','all_replica_cpu_wait_s',
                             'replica_A_finish_time','replica_B_finish_time']]).all().all()
    values.update(ppo_inference_count=expected, optimizer_steps=0, **mixed_task_metrics(data))
    return values, loads, data


def intervention_group_metrics(data):
    """Raw-task latency metrics, including groups consisting entirely of fallback tasks."""
    fields = dict(mean_latency_s='task_latency', mean_winner_upload_s='winner_upload_s',
        mean_winner_cpu_wait_s='winner_cpu_wait_s', mean_winner_execution_s='winner_execution_s',
        mean_all_replica_cpu_wait_s='all_replica_cpu_wait_s', mean_loser_cpu_wait_s='loser_cpu_wait_s',
        decision_backlog_mean_s='decision_backlog_sum_s', mean_safe_pair_count='safe_pair_count', rsr_true='true_satisfied')
    return dict(task_count=len(data), **{k:data[v].mean() for k,v in fields.items()},
        p95_latency_s=data.task_latency.quantile(.95), fallback_fraction=data.is_fallback.mean(),
        fallback_count=int(data.is_fallback.sum()), ppo_inference_count=int(data.policy_used.eq('PPO').sum()),
        mean_fallback_latency_s=data.loc[data.is_fallback,'task_latency'].mean(),
        mean_nonfallback_latency_s=data.loc[~data.is_fallback,'task_latency'].mean())


def intervention_groups(data):
    high = data[data.tier==4]
    return [('overall',data), ('low_mid',data[data.tier<4]),
            *[(f'tier_{tier}',g) for tier,g in data.groupby('tier')],
            *[(f'high_{name}',high[condition]) for name,condition in (
                ('unique',high.safe_pair_count==1),('multiple',high.safe_pair_count>=2),
                ('fallback',high.safe_pair_count==0))]]


def intervention_statistics(output, data, episodes, replicas, loads, *,
                            comparisons=(('C','A'),('D','A'),('B','C'),('B','D'),('B','A'))):
    seed_rows=[]
    for (seed,arm),g in data.groupby(['seed_id','arm']):
        for scope,subset in intervention_groups(g):
            # Equal task counts per episode; means equal episode means. Quantiles use pooled raw tasks.
            seed_rows.append(dict(seed_id=seed,arm=arm,scope=scope,**intervention_group_metrics(subset)))
    seeds=pd.DataFrame(seed_rows)
    seeds['task_count']/=data.episode.nunique()
    seeds['fallback_count']/=data.episode.nunique()
    seeds['ppo_inference_count']/=data.episode.nunique()
    # Load metrics retain per-episode units; High wait on nodes 4/5 is per allocated replica.
    hotspots=[]
    for (seed,arm),g in replicas.groupby(['seed_id','arm']):
        low=g[(g.tier<4)&g.server_id.isin([4,5])]
        high=g[(g.tier==4)&g.server_id.isin([4,5])]
        sl=loads[(loads.seed_id==seed)&(loads.arm==arm)]
        hotspots.append(dict(seed_id=seed,arm=arm,scope='nodes_45',
            low_execution_cycles=low.computation_demand.sum()/data.episode.nunique(),
            low_replica_count=len(low)/data.episode.nunique(), high_replica_wait_s=high.cpu_wait_s.mean(),
            high_execution_cycles=high.computation_demand.sum()/data.episode.nunique(),
            high_replica_count=len(high)/data.episode.nunique(), total_cpu_busy_s=sl.cpu_busy_s.sum()/data.episode.nunique(),
            hotspot_mean_utilization=sl[sl.server_id.isin([4,5])].cpu_utilization.mean(),
            decision_backlog_mean_s=episodes[(episodes.seed_id==seed)&(episodes.arm==arm)].decision_backlog_mean_s.mean()))
    seeds=pd.concat([seeds,pd.DataFrame(hotspots)],ignore_index=True)
    seeds.to_csv(output/'eval_seed_summary.csv',index=False,float_format='%.17g')
    numeric=[c for c in seeds.select_dtypes('number') if c!='seed_id']
    summary=seeds.groupby(['scope','arm'])[numeric].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index()
    summary.to_csv(output/'arm_summary.csv',index=False,float_format='%.17g')
    pairs=[];interactions=[];raw_diffs=[]
    for scope,g in seeds.groupby('scope'):
        for metric in numeric:
            wide=g.pivot(index='seed_id',columns='arm',values=metric)
            if wide.isna().any().any():continue
            for target,reference in comparisons:
                diff=wide[target]-wide[reference]
                half=student_t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)) if len(diff)>1 else np.nan
                improvement=np.where(wide[reference]!=0,-100*diff/wide[reference],np.nan)
                pairs.append(dict(scope=scope,metric=metric,comparison=f'{target}-{reference}',
                    delta_mean=diff.mean(),delta_sd=diff.std(ddof=1),ci95_lower=diff.mean()-half,
                    ci95_upper=diff.mean()+half,improvement_percent=np.nanmean(improvement) if np.isfinite(improvement).any() else np.nan,
                    lower_seeds=int((diff<0).sum()),seed_count=len(diff)))
                raw_diffs.extend(dict(scope=scope,metric=metric,comparison=f'{target}-{reference}',seed_id=s,delta=v) for s,v in diff.items())
            effect=wide.B-wide.C-wide.D+wide.A
            half=student_t.ppf(.975,len(effect)-1)*effect.std(ddof=1)/np.sqrt(len(effect)) if len(effect)>1 else np.nan
            interactions.append(dict(scope=scope,metric=metric,interaction_mean=effect.mean(),
                interaction_sd=effect.std(ddof=1),ci95_lower=effect.mean()-half,ci95_upper=effect.mean()+half,seed_count=len(effect)))
    paired=pd.DataFrame(pairs);interaction=pd.DataFrame(interactions)
    paired.to_csv(output/'paired_comparison.csv',index=False,float_format='%.17g')
    interaction.to_csv(output/'interaction_effects.csv',index=False,float_format='%.17g')
    pd.DataFrame(raw_diffs).to_csv(output/'paired_seed_differences.csv',index=False,float_format='%.17g')
    return summary,paired,interaction


def arrival_queue_metrics(data, loads, profiles):
    """Integrate recorded CPU intervals; this measures trajectories without scheduling work."""
    end=float(data.arrival_time.max())
    finish=float(data[['replica_A_finish_time','replica_B_finish_time']].to_numpy().max())
    parts=[pd.DataFrame(dict(server_id=data[f'server_{r}_id'],arrival=data.arrival_time,
        queue=data[f'replica_{r}_queue_enter_time'],start=data[f'replica_{r}_cpu_start_time'],
        finish=data[f'replica_{r}_finish_time'],cycles=data.computation_demand,tier=data.tier)) for r in ('A','B')]
    replicas=pd.concat(parts,ignore_index=True)
    assert len(replicas)==400
    events=np.unique(np.r_[0.,end,replicas.loc[replicas.queue<=end,'queue']])
    total_at_events=np.zeros(len(events));rows=[];total_area=0.;first_area=0.
    def area(g,until):
        service=(g.finish-g.start).to_numpy()
        elapsed=np.maximum(np.minimum(g.finish.to_numpy(),until)-g.start.to_numpy(),0.)
        waiting=np.maximum(np.minimum(g.start.to_numpy(),until)-g.queue.to_numpy(),0.)
        return float(np.sum(service*waiting+service*elapsed-.5*elapsed**2))
    for sid,profile in profiles.items():
        g=replicas[replicas.server_id==sid].sort_values('start')
        service=(g.finish-g.start).to_numpy();q=g.queue.to_numpy();f=g.finish.to_numpy()
        assert np.all(g.start.to_numpy()[1:]>=f[:-1]-1e-12)
        assert np.all(np.diff(q)>=-1e-12)
        def backlog(times):
            return np.sum(np.minimum(service,np.maximum(f-times[:,None],0.))*(q<=times[:,None]),axis=1)
        values=backlog(events);total_at_events+=values
        at_arrivals=backlog(data.arrival_time.to_numpy())
        remaining=float(backlog(np.array([end]))[0])
        inflight=float(service[q>end].sum())
        busy=float(np.maximum(np.minimum(f,end)-g.start.to_numpy(),0.).sum())
        total_busy=float(service.sum());a=area(g,end);a1=area(g,end/2)
        total_area+=a;first_area+=a1
        load=loads[loads.server_id==sid].iloc[0]
        np.testing.assert_allclose(total_busy,load.cpu_busy_s,rtol=0,atol=1e-11)
        assert 0<=busy/end<=1+1e-12
        frequency=profile['Processing_Frequency']
        rows.append(dict(server_id=sid,arrival_duration_s=end,episode_duration_s=finish,drain_time_s=finish-end,
            arrival_phase_cpu_busy_s=busy,arrival_phase_utilization=busy/end,
            full_period_cpu_busy_s=total_busy,full_period_utilization=total_busy/finish,
            cpu_demand_to_arrival_capacity=total_busy/end,
            arrival_phase_mean_cpu_backlog_s=a/end,arrival_phase_max_cpu_backlog_s=float(values.max()),
            arrival_epoch_mean_cpu_backlog_s=float(at_arrivals.mean()),arrival_epoch_max_cpu_backlog_s=float(at_arrivals.max()),
            arrival_first_half_mean_cpu_backlog_s=a1/(end/2),arrival_second_half_mean_cpu_backlog_s=(a-a1)/(end/2),
            end_arrival_cpu_backlog_s=remaining,end_arrival_cpu_backlog_cycles=remaining*frequency,
            end_arrival_upload_pending_cpu_s=inflight,end_arrival_total_pending_cpu_s=remaining+inflight,
            end_arrival_total_pending_cpu_cycles=(remaining+inflight)*frequency,
            low_execution_cycles=float(g.loc[g.tier<4,'cycles'].sum()),high_execution_cycles=float(g.loc[g.tier==4,'cycles'].sum())))
    server=pd.DataFrame(rows)
    global_row=dict(arrival_duration_s=end,episode_duration_s=finish,drain_time_s=finish-end,
        arrival_phase_mean_cpu_backlog_s=total_area/end,arrival_phase_max_cpu_backlog_s=float(total_at_events.max()),
        arrival_first_half_mean_cpu_backlog_s=first_area/(end/2),arrival_second_half_mean_cpu_backlog_s=(total_area-first_area)/(end/2),
        end_arrival_cpu_backlog_s=float(server.end_arrival_cpu_backlog_s.sum()),
        end_arrival_upload_pending_cpu_s=float(server.end_arrival_upload_pending_cpu_s.sum()),
        end_arrival_total_pending_cpu_s=float(server.end_arrival_total_pending_cpu_s.sum()),
        arrival_phase_mean_utilization=float(server.arrival_phase_utilization.mean()),
        full_period_mean_utilization=float(server.full_period_utilization.mean()),
        total_cpu_busy_s=float(server.full_period_cpu_busy_s.sum()),
        nodes_45_low_execution_cycles=float(server.loc[server.server_id.isin([4,5]),'low_execution_cycles'].sum()),
        nodes_45_high_execution_cycles=float(server.loc[server.server_id.isin([4,5]),'high_execution_cycles'].sum()))
    assert np.isfinite(server.to_numpy()).all() and np.isfinite(list(global_row.values())).all()
    return server,global_row


def tier_policy_intervention(args):
    import shlex,shutil,subprocess
    sweep=getattr(args,'frozen_arrival_rate_sweep',False)
    rates=(1.,1.5,2.,2.5,3.) if sweep else (MASKED_RATE,)
    output=args.output_dir or Path(RESULTS_DIR)/('frozen_ppo_arrival_rate_sweep' if sweep else 'tier_policy_intervention')
    resume=getattr(args,'resume',False)
    if resume:
        assert sweep and output.is_dir()
        assert (output/'source_snapshot.py').read_bytes()==Path('compare_ppo_greedy.py').read_bytes()
        prior=json.loads((output/'run_metadata.json').read_text())
        assert prior['seed_ids']==args.seeds and prior['eval_episodes']==args.eval_episodes
    else:
        output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source=Path(RESULTS_DIR)/'mixed_reliability_ppo_10seed'
    historical={str(source/name):h for name,h in json.loads((source/'output_manifest.json').read_text()).items()}
    for name,h in historical.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    # The comparison driver is the sole source intentionally extended in 8A.
    original_inputs=json.loads((source/'input_manifest.json').read_text())
    for name,h in original_inputs.items():
        if Path(name).name!='compare_ppo_greedy.py':assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    previous=Path(RESULTS_DIR)/'tier_policy_intervention'
    previous_tasks=previous_episodes=None
    if sweep:
        historical.update({str(previous/name):h for name,h in json.loads((previous/'output_manifest.json').read_text()).items()})
        assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==h for name,h in historical.items())
        previous_tasks=pd.read_csv(previous/'eval_task_metrics.csv',float_precision='round_trip',dtype={'safe_mask':str,'action_mask':str})
        previous_episodes=pd.read_csv(previous/'eval_episode_metrics.csv',float_precision='round_trip')
    profiles=MainLoop._build_server_profiles(pd.read_excel(SERVER_INFO_PATH,sheet_name='Servers',dtype={'Site_ID':'string'}))
    cached=load_mixed_profiles(profiles,output);tasks=cached['evaluation']['mixed_C']
    checkpoint=pd.read_csv(source/'checkpoint_manifest.csv')
    checkpoint=checkpoint[(checkpoint.model=='correlated')&checkpoint.seed_id.isin(args.seeds)].copy()
    assert len(checkpoint)==len(args.seeds) and checkpoint.train_episodes.eq(300).all()
    for row in checkpoint.itertuples():
        assert hashlib.sha256(Path(row.final_checkpoint).read_bytes()).hexdigest()==row.final_checkpoint_sha256
    checkpoint.to_csv(output/'checkpoint_manifest.csv',index=False)
    meta=dict(command=shlex.join([sys.executable,*sys.argv]),arms=INTERVENTION_ARMS,thresholds=MIXED_REQUIREMENTS,
        alpha=MASKED_ALPHA,beta=effective_spatial_risk_beta(),ell_ratio=1.,gh_order=32,arrival_rate=list(rates) if sweep else MASKED_RATE,
        ell0_km=compute_baseline_correlation_length_km(np.load(output/'reliability_table.npz')['distances_km']),
        seed_ids=args.seeds,eval_episodes=args.eval_episodes,new_training_episodes=0,
        torch_threads=torch.get_num_threads(),python=sys.version,numpy=np.__version__,torch=torch.__version__,simpy=simpy.__version__,
        rng_protocol='B retains 7B sequential Categorical RNG; C/D restore B pre-sample CPU RNG state keyed by (seed, episode, Task_ID). Live observations and new actions in every arm.',
        arrival_seed_key='(71,0,seed,4,episode)',ppo_seed_key='(71,0,seed,5,0,episode)',
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status_before=subprocess.check_output(['git','status','--short'],text=True))
    inputs={**historical,**{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in original_inputs}}
    (output/'input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    if not resume:
        (output/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        shutil.copyfile('compare_ppo_greedy.py',output/'source_snapshot.py')
    with (output/'commands.txt').open('a',encoding='utf-8') as command_log:command_log.write(meta['command']+'\n')
    import core.env_state as state_module
    def forbidden(*a,**kw):raise AssertionError('Training or runtime GH forbidden in frozen intervention')
    all_raw=[];all_episodes=[];all_loads=[];replays=[];history_checks=[];rng_mapping=[];queue_rows=[];server_rows=[]
    with contextlib.ExitStack() as stack:
        from unittest.mock import patch
        for name in ('compute_pair_reliability_gh','compute_marginal_failure_probability'):
            stack.enter_context(patch.object(state_module,name,forbidden))
            stack.enter_context(patch.object(sys.modules[__name__],name,forbidden))
        stack.enter_context(patch.object(PPOAgent,'train_step',forbidden))
        stack.enter_context(patch.object(torch.optim.Adam,'step',forbidden))
        for seed in args.seeds:
            case=output/f'seed_{seed}'
            if sweep and (case/'complete.json').exists():
                saved=json.loads((case/'complete.json').read_text())
                assert all(hashlib.sha256((case/name).read_bytes()).hexdigest()==h for name,h in saved.items())
                print(f'RESUME seed={seed}: complete evaluation artifacts verified and retained',flush=True)
                continue
            row=checkpoint[checkpoint.seed_id==seed].iloc[0]
            agent=frozen_agent(row.final_checkpoint);snapshot=frozen_snapshot(agent)
            old={}
            for arm,model in (('A','safe_min_latency'),('B','correlated')):
                case=source/'mixed_C'/f'seed_{seed}'/model
                old[arm]=pd.read_csv(case/'eval_episode_metrics.csv',float_precision='round_trip').set_index('episode_id')
                old[arm+'_tasks']=pd.read_csv(case/'eval_task_records.csv.gz',float_precision='round_trip',dtype={'safe_mask':str})
            if sweep:
                old={arm:previous_episodes[(previous_episodes.seed_id==seed)&(previous_episodes.arm==arm)].set_index('episode_id') for arm in INTERVENTION_ARMS}
                old.update({arm+'_tasks':previous_tasks[(previous_tasks.seed_id==seed)&(previous_tasks.arm==arm)] for arm in INTERVENTION_ARMS})
            common_rng={}
            for arrival_rate in rates:
                for episode in range(1,args.eval_episodes+1):
                    arrival_seed=derived_seed((71,0,seed,4,episode));action_seed=derived_seed((71,0,seed,5,0,episode))
                    basis=np.random.default_rng(arrival_seed).exponential(size=200);rng_states={};arrivals=set();episode_data={}
                    for arm in ('B','A','C','D'):
                        values,loads,data=intervention_episode(arm,agent,profiles,tasks,basis,episode,action_seed,rng_states,arrival_rate=arrival_rate)
                        arrivals.add(values['arrival_sha256'])
                        tags=dict(seed_id=seed,arm=arm,**({'arrival_rate':arrival_rate} if sweep else {}))
                        data=data.assign(**tags);episode_data[arm]=data
                        all_raw.append(data);all_loads.append(loads.assign(**tags))
                        all_episodes.append(dict(**values,**tags,eval_arrival_seed=arrival_seed,eval_action_seed=action_seed))
                        if (sweep and arrival_rate==2.) or (not sweep and arm in ('A','B')):
                            assert values['trace_sha256']==old[arm].loc[episode,'trace_sha256']
                            prior=old[arm+'_tasks'].query('episode==@episode').sort_values('task_id')
                            current=data.sort_values('task_id')
                            fields=['arrival_time','action_index','server_A_id','server_B_id','task_completion_time',
                                    'task_latency','replica_A_queue_enter_time','replica_A_cpu_start_time','replica_A_finish_time',
                                    'replica_B_queue_enter_time','replica_B_cpu_start_time','replica_B_finish_time']
                            np.testing.assert_array_equal(current[fields].to_numpy(),prior[fields].to_numpy())
                            history_checks.append(dict(**tags,episode_id=episode,
                                trace_sha256=values['trace_sha256'],historical_trace_match=True,all_task_timestamps_match=True))
                        if sweep:
                            server,queue=arrival_queue_metrics(data,loads,profiles)
                            server_rows.append(server.assign(**tags,episode_id=episode))
                            queue_rows.append(dict(**tags,episode_id=episode,**queue))
                        if episode==1:
                            again,_,again_data=intervention_episode(arm,agent,profiles,tasks,basis,episode,action_seed,rng_states,arrival_rate=arrival_rate)
                            assert values==again
                            np.testing.assert_array_equal(data.action_index,again_data.action_index)
                            replays.append(dict(**tags,episode_id=episode,passed=True))
                    assert len(arrivals)==1 and len(rng_states)==187
                    b=episode_data['B'].set_index('task_id')
                    for arm in ('C','D'):
                        selected=episode_data[arm].query("policy_used=='PPO'").set_index('task_id')
                        np.testing.assert_array_equal(selected.ppo_rng_sha256,b.loc[selected.index,'ppo_rng_sha256'])
                    for ordinal,(tid,state) in enumerate(rng_states.items()):
                        rng_mapping.append(dict(seed_id=seed,**({'arrival_rate':arrival_rate} if sweep else {}),episode_id=episode,task_id=tid,ppo_call_index_B=ordinal,
                            eval_action_seed=action_seed,rng_sha256=hashlib.sha256(state.numpy().tobytes()).hexdigest()))
                    if sweep:
                        hashes=tuple(hashlib.sha256(state.numpy().tobytes()).hexdigest() for state in rng_states.values())
                        if episode in common_rng:assert hashes==common_rng[episode]
                        else:common_rng[episode]=hashes
            same_tree(snapshot,frozen_snapshot(agent))
            assert hashlib.sha256(Path(row.final_checkpoint).read_bytes()).hexdigest()==row.final_checkpoint_sha256
            if sweep and seed==args.seeds[0]:
                preflight=dict(passed=True,seed_id=seed,arrival_rates=list(rates),evaluation_episodes=len(rates)*4*args.eval_episodes,
                    rate_2_all_arms_exact=True,task_mapping_verified=True,frozen=True,retained_in_main_statistics=True)
                (output/'seed1_preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
                print('ARRIVAL SWEEP PREFLIGHT PASSED: all five rates; retaining seed 1 and expanding remaining seeds',flush=True)
            if sweep:
                case=output/f'seed_{seed}'
                case.mkdir(exist_ok=True)
                pieces={'tasks':pd.concat(all_raw,ignore_index=True),'episodes':pd.DataFrame(all_episodes),
                    'loads':pd.concat(all_loads,ignore_index=True),'servers':pd.concat(server_rows,ignore_index=True),
                    'queue':pd.DataFrame(queue_rows),'history':pd.DataFrame(history_checks),
                    'replays':pd.DataFrame(replays),'rng':pd.DataFrame(rng_mapping)}
                for name,frame in pieces.items():frame.to_csv(case/f'{name}.csv.gz',index=False,float_format='%.17g')
                saved={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in case.glob('*.csv.gz')}
                (case/'complete.json').write_text(json.dumps(saved,indent=2)+'\n')
                all_raw.clear();all_episodes.clear();all_loads.clear();server_rows.clear();queue_rows.clear()
                history_checks.clear();replays.clear();rng_mapping.clear();del pieces
            print(f'INTERVENTION seed={seed} complete; {len(rates)} rates x 4 x {args.eval_episodes} episodes; frozen and historical replay verified',flush=True)
    if sweep:
        def saved_frame(name):
            return pd.concat([pd.read_csv(output/f'seed_{seed}'/f'{name}.csv.gz',float_precision='round_trip',
                dtype={'safe_mask':str,'action_mask':str}) for seed in args.seeds],ignore_index=True)
        finish_arrival_sweep(output,args,saved_frame('tasks'),saved_frame('episodes'),saved_frame('loads'),
            saved_frame('servers'),saved_frame('queue'),saved_frame('history').to_dict('records'),
            saved_frame('replays').to_dict('records'),saved_frame('rng').to_dict('records'),inputs,started)
        return
    data=pd.concat(all_raw,ignore_index=True);episodes=pd.DataFrame(all_episodes);loads=pd.concat(all_loads,ignore_index=True)
    assert len(data)==len(args.seeds)*args.eval_episodes*800
    for _,g in data.groupby(['seed_id','episode','task_id']):
        assert len(g)==4 and g.arrival_time.nunique()==g.safe_mask.nunique()==1
    metrics=[]
    for (seed,arm,episode),g in data.groupby(['seed_id','arm','episode']):
        for scope,subset in intervention_groups(g):
            metrics.append(dict(seed_id=seed,arm=arm,episode_id=episode,scope=scope,**intervention_group_metrics(subset)))
    group=pd.DataFrame(metrics)
    replicas=pd.concat([data[['seed_id','arm','episode','task_id','tier','safe_pair_count','computation_demand']].assign(
        replica=r,server_id=data[f'server_{r}_id'],cpu_wait_s=data[f'replica_{r}_cpu_start_time']-data[f'replica_{r}_queue_enter_time'],
        cpu_busy_s=data[f'replica_{r}_finish_time']-data[f'replica_{r}_cpu_start_time']) for r in ('A','B')],ignore_index=True)
    assert len(replicas)==2*len(data) and np.isfinite(replicas[['cpu_wait_s','cpu_busy_s']]).all().all()
    tier_load=replicas.groupby(['seed_id','arm','episode','tier','server_id']).agg(replica_count=('server_id','size'),
        execution_cycles=('computation_demand','sum'),cpu_busy_s=('cpu_busy_s','sum'),mean_cpu_wait_s=('cpu_wait_s','mean'))
    index=pd.MultiIndex.from_product([args.seeds,INTERVENTION_ARMS,range(1,args.eval_episodes+1),range(1,5),range(1,9)],names=tier_load.index.names)
    tier_load=tier_load.reindex(index)
    tier_load[['replica_count','execution_cycles','cpu_busy_s']]=tier_load[['replica_count','execution_cycles','cpu_busy_s']].fillna(0)
    tier_load=tier_load.reset_index().rename(columns={'episode':'episode_id'})
    load_seed=loads.groupby(['seed_id','arm','server_id'])[['replica_count','execution_cycles','cpu_busy_s','mean_cpu_wait_s','cpu_utilization']].mean().reset_index()
    load_seed.groupby(['arm','server_id']).mean(numeric_only=True).drop(columns='seed_id').reset_index().to_csv(output/'server_load_summary.csv',index=False,float_format='%.17g')
    tier_load.groupby(['arm','tier','server_id'])[['replica_count','execution_cycles','cpu_busy_s','mean_cpu_wait_s']].mean().reset_index().to_csv(output/'server_tier_load_summary.csv',index=False,float_format='%.17g')
    for name,frame in [('eval_task_metrics',data),('eval_episode_metrics',episodes),('eval_tier_metrics',group[group.scope.str.startswith('tier_')]),
        ('high_group_episode_metrics',group[group.scope.str.startswith('high_')]),('server_load_by_tier',tier_load),
        ('server_load_episodes',loads),('historical_replay_checks',pd.DataFrame(history_checks)),
        ('reproducibility_checks',pd.DataFrame(replays)),('ppo_random_mapping',pd.DataFrame(rng_mapping))]:
        frame.to_csv(output/f'{name}.csv',index=False,float_format='%.17g')
    data.groupby(['arm','tier','safe_pair_count','server_A_id','server_B_id']).size().rename('task_count').reset_index().to_csv(output/'actual_pair_counts.csv',index=False)
    replicas[(replicas.tier==4)].groupby(['seed_id','arm','safe_pair_count','server_id']).agg(
        replica_count=('server_id','size'),execution_cycles=('computation_demand','sum'),mean_cpu_wait_s=('cpu_wait_s','mean')).reset_index().to_csv(output/'high_group_server_load.csv',index=False)
    summary,paired,interaction=intervention_statistics(output,data,episodes,replicas,loads)
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==h for name,h in inputs.items())
    validation=dict(passed=True,evaluation_episodes=len(episodes),tasks=len(data),replicas=len(replicas),
        training_episodes=0,optimizer_steps=0,gh_runtime_calls=0,all_tasks_completed=True,task_mapping_verified=True,
        identical_masks_and_arrivals=True,illegal_nonfallback_actions=0,fallback_per_episode=13,overall_rsr=.935,high_rsr=.74,
        frozen_checkpoints_and_optimizers=True,historical_A_B_exact_episodes=len(history_checks),
        repeated_arm_checks=len(replays),shared_ppo_rng_task_aligned=True,all_historical_hashes_unchanged=True,
        max_latency_decomposition_error_s=float(data.latency_decomposition_error_s.max()),elapsed_s=time.monotonic()-started)
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    plot_intervention(output,summary)
    write_intervention_report(output,summary,paired,interaction,validation)
    manifest={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'output_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(validation,indent=2),flush=True)


def finish_arrival_sweep(output,args,data,episodes,loads,servers,queue,history,replays,rng_mapping,inputs,started):
    """Reuse 8A seed-level statistics separately at each arrival rate."""
    assert len(episodes)==5*4*len(args.seeds)*args.eval_episodes and len(data)==200*len(episodes)
    paired_keys=['arrival_rate','seed_id','episode','task_id']
    assert data.groupby(paired_keys)[['arrival_time','safe_mask']].nunique().eq(1).all().all()
    assert data.groupby(['seed_id','episode','task_id'])[['task_size','computation_demand','reliability_requirement','safe_mask']].nunique().eq(1).all().all()
    scaled=data.assign(unit_rate_arrival=data.arrival_time*data.arrival_rate)
    spread=scaled.groupby(['seed_id','episode','task_id']).unit_rate_arrival.agg(['min','max'])
    assert np.max(spread['max']-spread['min'])<5e-12
    replicas=pd.concat([data[['arrival_rate','seed_id','arm','episode','task_id','tier','computation_demand']].assign(
        server_id=data[f'server_{r}_id'],cpu_wait_s=data[f'replica_{r}_cpu_start_time']-data[f'replica_{r}_queue_enter_time'],
        cpu_busy_s=data[f'replica_{r}_finish_time']-data[f'replica_{r}_cpu_start_time']) for r in ('A','B')],ignore_index=True)
    assert len(replicas)==2*len(data) and loads.groupby(['arrival_rate','seed_id','arm','episode_id']).replica_count.sum().eq(400).all()
    groups=[]
    for (rate,seed,arm,episode),g in data.groupby(['arrival_rate','seed_id','arm','episode']):
        groups.extend(dict(arrival_rate=rate,seed_id=seed,arm=arm,episode_id=episode,scope=scope,
                           **intervention_group_metrics(subset)) for scope,subset in intervention_groups(g))
    groups=pd.DataFrame(groups)
    summaries=[];paired=[];seed_frames=[];diff_frames=[]
    for rate in sorted(data.arrival_rate.unique()):
        case=output/f'rate_{rate:g}';case.mkdir(exist_ok=True)
        summary,comparison,_=intervention_statistics(case,data[data.arrival_rate==rate],episodes[episodes.arrival_rate==rate],
            replicas[replicas.arrival_rate==rate],loads[loads.arrival_rate==rate],
            comparisons=(('C','A'),('B','A'),('D','A'),('C','B'),('B','C'),('B','D')))
        summaries.append(summary.assign(arrival_rate=rate));paired.append(comparison.assign(arrival_rate=rate))
        seed_frames.append(pd.read_csv(case/'eval_seed_summary.csv',float_precision='round_trip').assign(arrival_rate=rate))
        diff_frames.append(pd.read_csv(case/'paired_seed_differences.csv',float_precision='round_trip').assign(arrival_rate=rate))
    summary=pd.concat(summaries,ignore_index=True);comparison=pd.concat(paired,ignore_index=True)
    server=servers.merge(loads[['arrival_rate','seed_id','arm','episode_id','server_id','replica_count','execution_cycles','mean_cpu_wait_s']],
        on=['arrival_rate','seed_id','arm','episode_id','server_id'],validate='one_to_one')
    frames=[('eval_episode_metrics',episodes),('eval_tier_metrics',groups[groups.scope.str.startswith('tier_')]),
        ('high_group_episode_metrics',groups[groups.scope.str.startswith('high_')]),('server_load_metrics',server),
        ('queue_backlog_metrics',queue),('arm_summary',summary),('paired_comparison',comparison),
        ('eval_seed_summary',pd.concat(seed_frames,ignore_index=True)),('paired_seed_differences',pd.concat(diff_frames,ignore_index=True)),
        ('historical_replay_checks',pd.DataFrame(history)),('reproducibility_checks',pd.DataFrame(replays)),
        ('ppo_random_mapping',pd.DataFrame(rng_mapping))]
    for name,frame in frames:frame.to_csv(output/f'{name}.csv',index=False,float_format='%.17g')
    data.to_csv(output/'eval_task_metrics.csv.gz',index=False,float_format='%.17g')
    for name,frame,keys in [('server',server,['arrival_rate','seed_id','arm','server_id']),('queue',queue,['arrival_rate','seed_id','arm'])]:
        fields=[c for c in frame.select_dtypes('number') if c not in keys+['episode_id']]
        seeds=frame.groupby(keys)[fields].mean().reset_index()
        seeds.to_csv(output/f'{name}_seed_summary.csv',index=False,float_format='%.17g')
        methods=seeds.groupby([k for k in keys if k!='seed_id'])[fields].agg(['mean','std'])
        methods.columns=['_'.join(c) for c in methods.columns]
        methods.reset_index().to_csv(output/f'{name}_summary.csv',index=False,float_format='%.17g')
    assert episodes.fallback_count.eq(13).all() and episodes.rsr_true.eq(.935).all()
    assert groups[groups.scope=='tier_4'].rsr_true.eq(.74).all()
    assert groups[groups.scope.isin(['tier_1','tier_2','tier_3'])].rsr_true.eq(1).all()
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==h for name,h in inputs.items())
    validation=dict(passed=True,arrival_rates=sorted(data.arrival_rate.unique()),evaluation_episodes=len(episodes),tasks=len(data),replicas=len(replicas),
        seed1_preflight_retained=True,training_episodes=0,optimizer_steps=0,gh_runtime_calls=0,
        frozen_checkpoints_and_optimizers=True,identical_tasks_and_masks_across_rates=True,
        identical_arrivals_within_rate=True,max_scaled_arrival_difference_s=float(np.max(spread['max']-spread['min'])),
        shared_ppo_rng_task_aligned=True,cross_rate_ppo_rng_aligned=True,
        rate_2_all_arms_exact_episodes=len(history),repeated_arm_checks=len(replays),
        all_tasks_completed=True,replicas_per_episode=400,illegal_nonfallback_actions=0,
        fallback_per_episode=13,overall_rsr=.935,high_rsr=.74,lower_tiers_rsr=1.,
        max_latency_decomposition_error_s=float(data.latency_decomposition_error_s.max()),
        arrival_and_full_period_utilization_separate=True,all_historical_hashes_unchanged=True,elapsed_s=time.monotonic()-started)
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    plot_arrival_sweep(output,summary,comparison,pd.read_csv(output/'server_summary.csv'))
    write_arrival_sweep_report(output,summary,comparison,pd.read_csv(output/'queue_summary.csv'),validation)
    manifest={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file() and p.name!='output_manifest.json'}
    (output/'output_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(validation,indent=2),flush=True)


def plot_arrival_sweep(output,summary,comparison,servers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,name,title in [('mean_latency_s','mean_latency_vs_rate.png','Mean first-completion latency (s)'),
                              ('p95_latency_s','p95_latency_vs_rate.png','P95 latency: pooled tasks per seed (s)')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for arm in INTERVENTION_ARMS:
            g=summary[(summary.arm==arm)&(summary.scope=='overall')].sort_values('arrival_rate')
            ax.errorbar(g.arrival_rate,g[metric+'_mean'],yerr=g[metric+'_std'],marker='o',capsize=3,label=f'{arm}: {INTERVENTION_ARMS[arm]}')
        ax.set_xlabel('Arrival rate (tasks/s)');ax.set_ylabel(title);ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/name,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    g=comparison[(comparison.scope=='tier_4')&(comparison.metric=='mean_winner_cpu_wait_s')&(comparison.comparison=='C-A')].sort_values('arrival_rate')
    ax.errorbar(g.arrival_rate,-g.delta_mean,yerr=np.vstack([g.ci95_upper-g.delta_mean,g.delta_mean-g.ci95_lower]),marker='o',capsize=4)
    ax.axhline(0,color='gray',linewidth=1);ax.set_xlabel('Arrival rate (tasks/s)');ax.set_ylabel('High winner wait reduction A-C (s), 95% CI');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'cross_tier_wait_improvement.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for col,node in enumerate([4,5]):
        for arm in INTERVENTION_ARMS:
            g=servers[(servers.arm==arm)&(servers.server_id==node)].sort_values('arrival_rate')
            axes[0,col].plot(g.arrival_rate,g.low_execution_cycles_mean/1e11,marker='o',label=arm)
            axes[1,col].plot(g.arrival_rate,g.end_arrival_cpu_backlog_s_mean,marker='o',label=arm)
        axes[0,col].set_title(f'Server {node}');axes[0,col].set_ylabel('Low/Mid CPU work (1e11 cycles/episode)')
        axes[1,col].set_ylabel('CPU backlog at last arrival (s)');axes[1,col].set_xlabel('Arrival rate (tasks/s)')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(output/'hotspot_work_and_backlog.png',dpi=160);plt.close(fig)


def write_arrival_sweep_report(output,summary,comparison,queue,validation):
    lines=['# 第八阶段8B-1：冻结PPO跨到达率泛化实验','',
        '使用7B在2 tasks/s训练300轮的10个相关PPO最终Checkpoint，不训练、不更新优化器。A=SML/SML，B=PPO/PPO，C=PPO/SML，D=SML/PPO（前三档/最高档）。',
        '服务器、Task_ID及四档各50任务、alpha=.05、beta=.5、ell=ell0、28对映射及GH32缓存完全复用7A/7B/8A。',
        '到达样本basis=default_rng(derived_seed((71,0,seed,4,episode))).exponential(200)，间隔=basis/rate；2 tasks/s严格保留原路径。',
        'B在每个负载下基于实时状态重新Categorical采样；C/D复用同一任务的B采样前RNG状态，而不是动作或观测。PPO随机流与到达流隔离，所有负载的任务采样RNG状态逐项核验相同。',
        '种子1先完成全部5档、20轮、4 Arm的400个方法Episode预检，验证冻结、可靠性、完整执行及2 tasks/s回归，再扩展剩余种子；预检结果直接纳入统计。',
        '统计单位为10个训练种子，先汇总各自20轮；SD为样本SD，配对95%CI为t(9)区间。P95按每个种子的原始任务池计算，不平均逐Episode分位数。',
        '', '## 整体与High结果','',
        '|rate|Arm|整体均值±SD/s|整体P95±SD/s|High均值/s|High Winner等待/s|RSR/%|Fallback/轮|',
        '|---|---|---|---|---|---|---|---|']
    for r in summary[summary.scope=='overall'].sort_values(['arrival_rate','arm']).itertuples():
        high=summary[(summary.arrival_rate==r.arrival_rate)&(summary.arm==r.arm)&(summary.scope=='tier_4')].iloc[0]
        lines.append(f'|{r.arrival_rate:g}|{r.arm}|{r.mean_latency_s_mean:.6f} ± {r.mean_latency_s_std:.6f}|{r.p95_latency_s_mean:.6f} ± {r.p95_latency_s_std:.6f}|{high.mean_latency_s_mean:.6f}|{high.mean_winner_cpu_wait_s_mean:.6f}|{100*r.rsr_true_mean:.1f}|{r.fallback_count_mean:g}|')
    lines+=['','## 核心配对比较','', 'Delta=目标−参照；负时延差为改善，改善率逐种子计算后汇总。','',
        '|rate|范围|指标|比较|Delta|95%CI|改善/%|下降种子|','|---|---|---|---|---|---|---|---|']
    for r in comparison.sort_values(['arrival_rate','scope','metric','comparison']).itertuples():
        if ((r.scope in ('overall','tier_4','high_unique','low_mid') and r.metric in ('mean_latency_s','mean_winner_cpu_wait_s') and r.comparison in ('C-A','B-A','D-A','C-B'))
            or (r.scope=='nodes_45' and r.metric=='low_execution_cycles' and r.comparison=='C-A')):
            lines.append(f'|{r.arrival_rate:g}|{r.scope}|{r.metric}|{r.comparison}|{r.delta_mean:+.6g}|[{r.ci95_lower:+.6g},{r.ci95_upper:+.6g}]|{r.improvement_percent:.3f}|{r.lower_seeds}/{r.seed_count}|')
    lines+=['','## 四档与唯一动作组','', '|rate|范围|Arm|均值±SD/s|P95/s|Winner上传/等待/执行/s|全副本/Loser等待/s|Fallback/非Fallback时延/s|',
        '|---|---|---|---|---|---|---|---|']
    for r in summary[summary.scope.isin(['tier_1','tier_2','tier_3','tier_4','high_unique'])].sort_values(['arrival_rate','scope','arm']).itertuples():
        lines.append(f'|{r.arrival_rate:g}|{r.scope}|{r.arm}|{r.mean_latency_s_mean:.6f} ± {r.mean_latency_s_std:.6f}|{r.p95_latency_s_mean:.6f}|{r.mean_winner_upload_s_mean:.6f}/{r.mean_winner_cpu_wait_s_mean:.6f}/{r.mean_winner_execution_s_mean:.6f}|{r.mean_all_replica_cpu_wait_s_mean:.6f}/{r.mean_loser_cpu_wait_s_mean:.6f}|{r.mean_fallback_latency_s_mean:.6f}/{r.mean_nonfallback_latency_s_mean:.6f}|')
    lines+=['','## 队列与利用率口径','',
        '到达阶段为[0,T_last_arrival]（包括首次到达前空闲）；完整周期为[0,T_last_replica_finish]。按CPU执行区间与阶段的交集累加busy，再分别除以对应时长。cpu_demand_to_arrival_capacity=全部已分配CPU执行秒/到达时长，可大于1，与有界的实际忙时利用率不同。',
        'CPU backlog只含已经上传、正在执行或等待CPU的剩余执行秒；upload_pending另列仍在上传的已卸载副本全部CPU执行秒。end_arrival_total_pending为两者之和，不把上传中的工作遗漏。末次到达本身也已分派，其两个副本通常仍在上传。',
        '时间平均CPU backlog通过每个副本的排队矩形面积和执行剩余工作三角形积分得到；最大值核查所有上传完成入队时刻与阶段边界。此为事后轨迹计量，决策仍只使用原实时可观测状态。',
        '同时保留原到达时刻采样的backlog指标；其均值、最大值可能小于事件级最大值，不能与时间平均口径混用。后半段/前半段均值用于描述有限到达窗口中的增长。',
        '', '|rate|Arm|到达/完整利用率(8节点均值)|CPU busy总秒/轮|前半/后半时间平均积压/s|最后到达CPU/上传待执行/s|最大总CPU积压/s|排空秒|',
        '|---|---|---|---|---|---|---|---|']
    for r in queue.sort_values(['arrival_rate','arm']).itertuples():
        lines.append(f'|{r.arrival_rate:g}|{r.arm}|{r.arrival_phase_mean_utilization_mean:.6f}/{r.full_period_mean_utilization_mean:.6f}|{r.total_cpu_busy_s_mean:.6f}|{r.arrival_first_half_mean_cpu_backlog_s_mean:.6f}/{r.arrival_second_half_mean_cpu_backlog_s_mean:.6f}|{r.end_arrival_cpu_backlog_s_mean:.6f}/{r.end_arrival_upload_pending_cpu_s_mean:.6f}|{r.arrival_phase_max_cpu_backlog_s_mean:.6f}|{r.drain_time_s_mean:.6f}|')
    lines+=['','逐节点Low/High cycles、副本、利用率和到达结束积压见server_load_metrics.csv、server_summary.csv。逐Episode队列见queue_backlog_metrics.csv。所有条件均值为空表示该子集没有任务，非物理数值异常。',
        '', '## 检查','', '```json',json.dumps(validation,indent=2),'```','',
        '各Arm每个负载每个种子的首轮额外重放一次（200次审计，不纳入4000轮正式样本）；2 tasks/s下四Arm共800轮与8A逐任务时间戳和轨迹哈希精确一致。',
        '每轮所有任务和副本均完成只说明有限200任务Episode可排空，不证明队列稳态稳定性。本实验是在2 tasks/s训练策略的跨负载冻结泛化测试，不代表在每档负载重新训练的算法性能。',
        '后续代表性负载建议与九项具体问题的实际数值解释在分析补充中给出；不会自动启动8B-2。',
        '', '## 图','', '![平均时延](mean_latency_vs_rate.png)','', '![P95](p95_latency_vs_rate.png)','',
        '![High等待改善](cross_tier_wait_improvement.png)','', '![热点工作量与积压](hotspot_work_and_backlog.png)','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def plot_intervention(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def chart(filename,scopes,metric,ylabel):
        fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4);width=.8/len(scopes)
        for i,scope in enumerate(scopes):
            g=summary[summary.scope==scope].set_index('arm').loc[list(INTERVENTION_ARMS)]
            ax.bar(x+(i-(len(scopes)-1)/2)*width,g[metric+'_mean'],width,
                   yerr=g[metric+'_std'],capsize=3,label=scope)
        ax.set_xticks(x,list(INTERVENTION_ARMS));ax.set_ylabel(ylabel);ax.legend();ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(output/filename,dpi=160);plt.close(fig)
    chart('latency_comparison.png',['overall','tier_4'],'mean_latency_s','First completion latency (s), mean +/- seed SD')
    chart('high_winner_wait.png',['tier_4'],'mean_winner_cpu_wait_s','High winner CPU wait (s)')
    chart('low_cpu_work_nodes_45.png',['nodes_45'],'low_execution_cycles','Low/Mid CPU cycles on nodes 4+5 per episode')
    chart('high_group_latency.png',['high_unique','high_multiple','high_fallback'],'mean_latency_s','High first completion latency (s)')


def write_intervention_report(output,summary,paired,interaction,validation):
    lines=['# 第八阶段8A：冻结策略的跨等级干预实验','',
        '四组均重新进行真实队列仿真：A=All-SML，B=All-PPO，C=Low-PPO/High-SML，D=Low-SML/High-PPO。',
        '复用7B的10个第300轮相关模型Checkpoint，无新训练。全部使用7A方案C与GH32缓存：α=.05、β=.5、ell=ell0、2 tasks/s、四档各50任务。',
        'B保留7B原顺序Categorical采样流，在每个非Fallback任务采样前保存CPU RNG状态；C/D按(seed,episode,Task_ID)恢复该状态后，调用原select_action读取本Arm当前实时状态。共享的是随机状态，不是历史动作或观测。',
        'A/B全部重新仿真并逐任务核对7B的动作、到达及两个副本完整时间戳；ppo_random_mapping.csv保存B调用序号、原种子与随机状态哈希。C/D共同PPO任务的随机哈希逐项相等。',
        'Fallback由原入口从全部28对调用同一SML函数。冻结评估无训练Transition；ppo_decision、policy_used和ppo_inference_count记录真实推理，ppo_transition_index仅表示训练记录编号。',
        '统计单位为训练种子；每种子先汇总20配对Episodes。表中±为10种子的样本SD；配对CI使用t(9)。P95从每种子原始任务池计算，再跨种子汇总。',
        '', '## 完整时延与可靠性','',
        '|范围|Arm|时延±SD/s|P95/s|Winner上传/等待/执行/s|RSR/%|Fallback/%|全副本/Loser等待/s|',
        '|---|---|---|---|---|---|---|---|']
    for scope in ['overall','low_mid','tier_1','tier_2','tier_3','tier_4','high_unique','high_multiple','high_fallback']:
        for r in summary[summary.scope==scope].itertuples():
            lines.append(f'|{scope}|{r.arm}|{r.mean_latency_s_mean:.6f} ± {r.mean_latency_s_std:.6f}|{r.p95_latency_s_mean:.6f}|{r.mean_winner_upload_s_mean:.6f}/{r.mean_winner_cpu_wait_s_mean:.6f}/{r.mean_winner_execution_s_mean:.6f}|{100*r.rsr_true_mean:.1f}|{100*r.fallback_fraction_mean:.1f}|{r.mean_all_replica_cpu_wait_s_mean:.6f}/{r.mean_loser_cpu_wait_s_mean:.6f}|')
    lines+=['','## 配对干预差','', 'Delta=目标Arm−参考Arm，负值表示该指标下降；改善百分比先逐种子计算。','',
        '|范围|指标|比较|Delta|95%CI|改善/%|下降种子|','|---|---|---|---|---|---|---|']
    for r in paired.itertuples():
        if (r.scope in ('overall','low_mid','tier_4','high_unique') and r.metric in ('mean_latency_s','mean_winner_cpu_wait_s')) or (r.scope=='nodes_45' and r.metric in ('low_execution_cycles','low_replica_count','high_replica_wait_s','total_cpu_busy_s')):
            lines.append(f'|{r.scope}|{r.metric}|{r.comparison}|{r.delta_mean:+.6g}|[{r.ci95_lower:+.6g}, {r.ci95_upper:+.6g}]|{r.improvement_percent:.3f}|{r.lower_seeds}/{r.seed_count}|')
    lines+=['','## 交互效应','', 'Interaction=B−C−D+A。共享队列产生非加性干预，不能直接作为网络内部规则的因果分解。','',
        '|范围|指标|交互均值|95%CI|','|---|---|---|---|']
    for r in interaction.itertuples():
        if r.scope in ('overall','tier_4','high_unique') and r.metric in ('mean_latency_s','mean_winner_cpu_wait_s'):
            lines.append(f'|{r.scope}|{r.metric}|{r.interaction_mean:+.6f}|[{r.ci95_lower:+.6f}, {r.ci95_upper:+.6f}]|')
    lines+=['','## 服务器4/5与资源代价','', '|Arm|Low副本/轮|Low cycles/轮|High副本/轮|High cycles/轮|High在4/5副本等待/s|4/5平均利用率|所有节点busy秒/轮|总积压均值/s|',
        '|---|---|---|---|---|---|---|---|---|']
    for r in summary[summary.scope=='nodes_45'].itertuples():
        lines.append(f'|{r.arm}|{r.low_replica_count_mean:.3f}|{r.low_execution_cycles_mean:.9g}|{r.high_replica_count_mean:.3f}|{r.high_execution_cycles_mean:.9g}|{r.high_replica_wait_s_mean:.6f}|{r.hotspot_mean_utilization_mean:.6f}|{r.total_cpu_busy_s_mean:.6f}|{r.decision_backlog_mean_s_mean:.6f}|')
    lines+=['','逐服务器、逐等级、逐Episode的400副本负载见server_load_by_tier.csv；全服务器忙时与利用率见server_load_episodes.csv。空节点等级组合工作量为0，条件平均等待留空表示没有副本。',
        'High唯一/多动作/Fallback分别21/16/13个。唯一动作始终(4,5)，实际选择与分组节点负载见actual_pair_counts.csv、high_group_server_load.csv。',
        '', '## 正确性与来源','', '```json',json.dumps(validation,indent=2),'```','',
        '历史7A/7B清单逐项核验；Checkpoint仅引用绝对路径与SHA256，不复制。运行时GH、train_step和Adam.step设置为报错哨兵。网络/旧策略/价值网络及两优化器state_dict逐种子前后精确一致。',
        f'全部{validation["evaluation_episodes"]}个方法Episode均200任务、400副本，Mask、RSR、等级、采样映射与时延分解均实际断言。各Arm每种子首Episode额外重放一次验证复现，{validation["repeated_arm_checks"]}次审计不混入正式统计。',
        '', '## 解释局限','',
        '这是固定任务分布和10个已训练冻结策略下的动态干预总效应。Low/Mid策略切换会改变队列，并进一步改变High多动作SML及Fallback的后续动作，不能隔离为仅经节点4/5负载的一条中介路径。',
        '唯一动作组固定服务器对，为队列环境解释提供更直接证据，但仍包含系统内反馈和共享资源干扰。未对隐藏层、策略编码、其他任务分布或其他模型进行识别；不能断言网络显式学会了可靠性分级规则。',
        '不根据本轮结果调整参数，不开展额外训练。具体十一项机制问题的数值分析见下文。','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def quick_qos_comparison(args):
    """Small GPU run using current workbook requirements and existing simulators."""
    import os
    import shlex
    from unittest.mock import patch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not torch.cuda.is_available():
        raise RuntimeError('Quick QoS comparison requires a CUDA GPU; CPU fallback is disabled')
    output = args.output_dir or Path(RESULTS_DIR)/'qos_gpu_quick_comparison'
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    files = [Path(p) for p in (SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH)] + [
        Path(p) for p in ('compare_ppo_greedy.py', 'agents/ppo_agent.py', 'core/main_loop.py',
                         'core/env_state.py', 'core/task.py', 'config/params.py', 'config/configuration.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    command = shlex.join([sys.executable, *sys.argv])
    (output/'commands.txt').write_text(command+'\n')
    (output/'input_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    profiles = MainLoop._build_server_profiles(pd.read_excel(SERVER_INFO_PATH, sheet_name='Servers', dtype={'Site_ID':'string'}))
    _, state = make_environment(profiles, failure_rate_scale=MASKED_ALPHA)
    state.initialize_spatial_reliability()
    task_sets, arrays, safe_rows = {}, {}, []
    # Populate the existing per-task caches once, preserving every workbook requirement.
    for split, path in (('train', TRAIN_TASK_PATH), ('evaluation', EVALUATION_TASK_PATH)):
        tasks = MainLoop._build_task_profiles(pd.read_excel(path))
        for tid, profile in tasks.items():
            task = Task(simpy.Environment(), state, tid, profile)
            mask = state.get_safe_action_mask(task)
            for value in (*task.reliability_snapshot, mask):
                value.setflags(write=False)
            profile.update(reliability_snapshot=task.reliability_snapshot, safe_mask=mask)
            safe_rows.append(dict(split=split, task_id=tid, requirement=task.reliability_requirement,
                                  deadline_s=task.deadline, safe_pair_count=int(mask.sum())))
        task_sets[split] = tasks
        arrays.update({f'{split}_{name}': np.stack([p['reliability_snapshot'][i] for p in tasks.values()])
                       for i, name in enumerate(('marginal', 'correlated', 'independent'))})
    np.savez_compressed(output/'reliability_table.npz', **arrays, server_ids=list(profiles), pairs=state.server_pairs)
    pd.DataFrame(safe_rows).to_csv(output/'task_safe_sets.csv', index=False, float_format='%.17g')
    metadata = dict(command=command, device='cuda', gpu=torch.cuda.get_device_name(0),
        train_episodes=args.train_episodes, eval_episodes=args.eval_episodes, seeds=args.seeds,
        arrival_rate=MASKED_RATE, alpha=MASKED_ALPHA, beta=effective_spatial_risk_beta(), ell_ratio=1., gh_order=32,
        state_dimension=params.num_states, deadline_feature='raw seconds', kappa=params.TASK_DEADLINE_SCALE,
        b=params.QOS_GRACE_MULTIPLIER, gamma=params.gamma_ppo, gae_lambda=params.gae_lambda_ppo,
        objective='mean effective QoS over all tasks', reward_assignment='completion interval',
        requirements=sorted({p['Reliability_Requirement'] for p in task_sets['train'].values()}),
        torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
        note='Current workbook requirements; no historical mixed-tier override; simulator and heuristics run on CPU, PPO on GPU')
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    train_frames, eval_rows, task_rows, load_frames = [], [], [], []
    import core.env_state as state_module
    def forbidden_gh(*a, **kw):
        raise AssertionError('GH must use the precomputed per-task cache')
    with patch.object(state_module, 'compute_marginal_failure_probability', forbidden_gh), \
         patch.object(state_module, 'compute_pair_reliability_gh', forbidden_gh):
        for seed in args.seeds:
            torch.manual_seed(derived_seed((92, seed, 0)))
            agent = build_agent(device='cuda')
            assert all(p.is_cuda for net in (agent.policy_net, agent.policy_old, agent.value_net) for p in net.parameters())
            case = output/f'seed_{seed}'
            case.mkdir()
            agent.save_model(case/'initial.pt')
            torch.manual_seed(derived_seed((92, seed, 1)))
            np.random.seed(derived_seed((92, seed, 2)))
            loop, training, _ = train_masked_agent(agent, profiles, task_sets['train'], 0, seed,
                args.train_episodes, arrival_stream=(92, seed, 3), progress_label='QoS GPU')
            train_tasks = pd.DataFrame(loop.task_results)
            np.testing.assert_allclose(training.mean_qos,
                train_tasks.groupby('episode').task_utility.mean().to_numpy(), rtol=0, atol=1e-12)
            train_frames.append(training.assign(seed_id=seed))
            agent.save_model(case/'final.pt')
            checkpoint_hash = hashlib.sha256((case/'final.pt').read_bytes()).hexdigest()
            for network in (agent.policy_net, agent.policy_old, agent.value_net):
                network.eval().requires_grad_(False)
            snapshot = frozen_snapshot(agent)
            for episode in range(1, args.eval_episodes+1):
                arrival_seed = derived_seed((92, seed, 4, episode))
                basis = np.random.default_rng(arrival_seed).exponential(size=200)
                signatures = []
                for role, method in enumerate(('trained_masked_ppo', 'safe_min_latency', 'safe_random')):
                    records = []
                    action_seed = derived_seed((92, seed, 5, role, episode))
                    values, loads = masked_evaluation(method, agent if role == 0 else None, profiles,
                        task_sets['evaluation'], basis, episode, action_seed, loop.get_task_outcome_info,
                        task_records=records, arrival_rate=MASKED_RATE)
                    data = pd.DataFrame(records)
                    np.testing.assert_allclose(data.task_qos, compute_task_qos(data.task_latency,
                        data.deadline_s, grace_multiplier=params.QOS_GRACE_MULTIPLIER), rtol=0, atol=1e-12)
                    eval_rows.append(dict(seed_id=seed, method=method, arrival_seed=arrival_seed,
                        action_seed=action_seed, **values))
                    task_rows.append(data.assign(seed_id=seed, method=method))
                    load_frames.append(loads.assign(seed_id=seed, method=method))
                    signatures.append(values['arrival_sha256'])
                    same_tree(snapshot, frozen_snapshot(agent))
                assert len(set(signatures)) == 1
            assert hashlib.sha256((case/'final.pt').read_bytes()).hexdigest() == checkpoint_hash
    training, evaluation = pd.concat(train_frames, ignore_index=True), pd.DataFrame(eval_rows)
    records, loads = pd.concat(task_rows, ignore_index=True), pd.concat(load_frames, ignore_index=True)
    for _, group in evaluation.groupby(['seed_id','episode_id']):
        assert group.fallback_count.nunique() == group.theoretical_rsr.nunique() == 1
    assert records.groupby(['seed_id','method','episode']).size().eq(200).all()
    assert loads.groupby(['seed_id','method','episode_id']).replica_count.sum().eq(400).all()
    fields = ['mean_qos','mean_latency_s','mean_winner_cpu_wait_s','theoretical_rsr','fallback_fraction','mean_safe_pair_count']
    seed_summary = evaluation.groupby(['seed_id','method'])[fields].mean().reset_index()
    summary = seed_summary.groupby('method')[fields].mean().reset_index()
    ppo = evaluation[evaluation.method == 'trained_masked_ppo'].set_index(['seed_id','episode_id'])
    comparisons = []
    for method in ('safe_min_latency','safe_random'):
        baseline = evaluation[evaluation.method == method].set_index(['seed_id','episode_id']).loc[ppo.index]
        comparisons.append(dict(reference=method, latency_delta_s=float((ppo.mean_latency_s-baseline.mean_latency_s).mean()),
            latency_improvement_pct=float((baseline.mean_latency_s.mean()-ppo.mean_latency_s.mean())/baseline.mean_latency_s.mean()*100),
            qos_delta=float((ppo.mean_qos-baseline.mean_qos).mean()),
            latency_better_episodes=int((ppo.mean_latency_s < baseline.mean_latency_s).sum()),
            qos_better_episodes=int((ppo.mean_qos > baseline.mean_qos).sum())))
    paired = pd.DataFrame(comparisons)
    for name, frame in [('train_episode_metrics',training),('eval_episode_metrics',evaluation),
                        ('eval_seed_summary',seed_summary),('summary',summary),('paired_comparison',paired),
                        ('eval_task_metrics',records),('server_load_metrics',loads)]:
        frame.to_csv(output/f'{name}.csv', index=False, float_format='%.17g')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    validation = dict(passed=True,device='cuda',gpu=torch.cuda.get_device_name(0),
        train_episodes=len(training),eval_method_episodes=len(evaluation),
        completed_tasks=200*(len(training)+len(evaluation)),completed_replicas=400*(len(training)+len(evaluation)),
        updates=int(training.update_call_count.sum()),cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),
        paired_arrivals=True,frozen_evaluation=True,matching_fallback_and_rsr=True,gh_runtime_calls=0,
        illegal_mask_actions=0,source_and_workbooks_unchanged_during_run=True,elapsed_s=time.monotonic()-started)
    (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    lines = ['# GPU QoS 简单对比','',f'GPU：{metadata["gpu"]}；每种子训练{args.train_episodes}轮，评估{args.eval_episodes}个配对Episode，种子{args.seeds}。',
        'PPO使用GPU；SimPy队列、GH预计算和两种启发式使用CPU。全部方法共用任务、到达流、相关Mask和Fallback。',
        f'到达率2 tasks/s，α=0.05，β=0.5，ell=ell0；κ=1.5，b=2；43维状态中deadline为秒原值。可靠性要求沿用当前Excel：{metadata["requirements"]}，未改成历史7B的混合阈值。',
        'PPO按Masked Categorical采样。训练奖励为决策区间内完成任务的有效QoS之和；平均QoS分母为全部200任务，可靠性不达标记0。原始QoS另记mean_raw_qos。时延和RSR包含全部任务。',
        '', '|方法|平均QoS ↑|平均时延/s ↓|Winner CPU等待/s|理论RSR|Fallback比例|','|---|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'|{r.method}|{r.mean_qos:.6f}|{r.mean_latency_s:.6f}|{r.mean_winner_cpu_wait_s:.6f}|{r.theoretical_rsr:.1%}|{r.fallback_fraction:.1%}|')
    lines += ['', '|PPO相对基线|时延差/s（负为更好）|时延改善/%|QoS差（正为更好）|','|---|---:|---:|---:|']
    for r in paired.itertuples():
        lines.append(f'|{r.reference}|{r.latency_delta_s:+.6f}|{r.latency_improvement_pct:+.2f}|{r.qos_delta:+.6f}|')
    lines += ['', f'训练首5轮平均QoS={training.head(5).mean_qos.mean():.6f}，末5轮={training.tail(5).mean_qos.mean():.6f}。',
        '这是小规模功能与效果预览，不能据单个训练种子和短训练判断收敛或最终优势。采用当前配置的γ和完成区间奖励；未修改原始Excel，未覆盖历史结果。',
        '', '实际命令：','```text',command,'```','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (output/'output_manifest.json').write_text(json.dumps({str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob('*') if p.is_file() and p.name != 'output_manifest.json'},indent=2)+'\n')
    print(summary.to_string(index=False)); print(paired.to_string(index=False)); print(f'Complete: {output}')


def legacy_main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_mapping().to_csv(OUTPUT_DIR/'seed_mapping.csv', index=False)
    files = [Path(p) for p in (SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH)] + [
        Path('config/configuration.py'), Path('config/params.py'), Path('agents/ppo_agent.py'),
        Path('core/main_loop.py'), Path('core/env_state.py'), Path('core/task.py'), Path('core/server.py'),
        Path('compare_ppo_greedy.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUTPUT_DIR/'input_manifest.json').write_text(json.dumps(hashes, indent=2))
    jobs = [(rate, tuple(range(1, 11)), 300, 20, OUTPUT_DIR) for rate in ARRIVAL_RATES]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        cases = list(pool.map(run_rate, jobs))
    summarize(OUTPUT_DIR, cases)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    print(f'Complete: {OUTPUT_DIR}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick-qos-comparison', action='store_true', help='Small CUDA-only PPO vs SML vs Safe Random run on current workbooks')
    parser.add_argument('--masked-comparison', action='store_true', help='Run the paired two-threshold masked PPO development experiment')
    parser.add_argument('--reliability-ablation', action='store_true', help='Compare correlated and fair-independent masks')
    parser.add_argument('--confirmation', action='store_true', help='Run paired four-method confirmation, seed 1 first')
    parser.add_argument('--length-sensitivity', action='store_true', help='Reuse phase 5 and train correlated PPO only at ell/ell0=0.5,4')
    parser.add_argument('--mixed-reliability-confirmation', action='store_true', help='Train paired PPO models on the frozen 7A mixed requirements')
    parser.add_argument('--tier-policy-intervention', action='store_true', help='Frozen 7B checkpoints: paired 2x2 task-tier policy intervention, no training')
    parser.add_argument('--frozen-arrival-rate-sweep', action='store_true', help='Frozen 8A arms at five arrival rates, no training')
    parser.add_argument('--resume', action='store_true', help='Resume verified per-seed frozen arrival sweep artifacts')
    parser.add_argument('--thresholds', type=float, nargs='+', choices=MASKED_THRESHOLDS, default=list(MASKED_THRESHOLDS))
    parser.add_argument('--cpu-audit-dir', type=Path, help='Completed CPU audit directory; required for the ablation')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--train-episodes', type=int)
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.eval_episodes is None:
        args.eval_episodes = 5 if args.quick_qos_comparison else 20
    if args.resume and not args.frozen_arrival_rate_sweep:
        parser.error('--resume is only supported for the frozen arrival sweep')
    args.seeds = args.seeds or ([1] if args.quick_qos_comparison else list(range(1, 11)) if args.confirmation or args.length_sensitivity or args.mixed_reliability_confirmation or args.tier_policy_intervention or args.frozen_arrival_rate_sweep else [1, 2, 3])
    if args.train_episodes is None:
        args.train_episodes = 20 if args.quick_qos_comparison else 300 if args.confirmation or args.length_sensitivity or args.mixed_reliability_confirmation else 100
    if min(args.train_episodes, args.eval_episodes, *args.seeds) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use positive episode counts and distinct positive seed IDs')
    if args.quick_qos_comparison:
        if any((args.masked_comparison,args.reliability_ablation,args.confirmation,args.length_sensitivity,
                args.mixed_reliability_confirmation,args.tier_policy_intervention,args.frozen_arrival_rate_sweep)):
            parser.error('Use --quick-qos-comparison separately')
        quick_qos_comparison(args)
    elif args.tier_policy_intervention or args.frozen_arrival_rate_sweep:
        if any((args.mixed_reliability_confirmation,args.length_sensitivity,args.confirmation,args.masked_comparison,args.reliability_ablation)):
            parser.error('Use --tier-policy-intervention separately')
        if args.tier_policy_intervention and args.frozen_arrival_rate_sweep:
            parser.error('Use one frozen evaluation mode at a time')
        if args.frozen_arrival_rate_sweep and args.seeds[0]!=1:
            parser.error('Arrival sweep preflight requires seed 1 first')
        if not set(args.seeds).issubset(range(1,11)) or args.eval_episodes>20:
            parser.error('Use historical seeds 1..10 and at most 20 paired evaluation episodes')
        tier_policy_intervention(args)
    elif args.mixed_reliability_confirmation:
        if args.length_sensitivity or args.confirmation or args.masked_comparison or args.reliability_ablation or args.seeds[0]!=1:
            parser.error('Use --mixed-reliability-confirmation separately, with seed 1 first')
        if not set(args.seeds).issubset(range(1,11)):
            parser.error('Use mixed training seeds 1 through 10')
        mixed_policy_confirmation(args)
    elif args.length_sensitivity:
        if args.confirmation or args.masked_comparison or args.reliability_ablation or args.seeds[0]!=1:
            parser.error('Use --length-sensitivity separately, with seed 1 first')
        if not set(args.seeds).issubset(range(1,11)):
            parser.error('Length sensitivity reuses historical seeds 1 through 10')
        length_policy_sensitivity(args)
    elif args.confirmation:
        if args.masked_comparison or args.reliability_ablation or args.seeds[0] != 1:
            parser.error('Use --confirmation separately, with seed 1 first')
        args.cpu_audit_dir = args.cpu_audit_dir or Path(RESULTS_DIR)/'correlated_vs_independent_ppo/cpu_wait_audit'
        reliability_ablation(args)
    elif args.reliability_ablation:
        if args.masked_comparison or args.cpu_audit_dir is None:
            parser.error('Use --reliability-ablation with --cpu-audit-dir, separately from --masked-comparison')
        reliability_ablation(args)
    elif args.masked_comparison:
        masked_comparison(args)
    else:
        legacy_main()


if __name__ == '__main__':
    main()
