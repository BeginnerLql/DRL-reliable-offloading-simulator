"""Six arrival rates: fixed-config PPO training and paired frozen Greedy evaluation."""

from concurrent.futures import ProcessPoolExecutor
import contextlib
import copy
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import simpy
import torch

from agents.ppo_agent import PPOAgent
from config.params import params
from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH, RESULTS_DIR
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.server import Server
from core.task import Task
from core.spatial_risk import (
    build_distance_matrix, compute_baseline_correlation_length_km,
    build_spatial_correlation_matrix, sample_spatial_risk_field,
    map_spatial_risk_to_effective_failure_rates,
)

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


def build_agent():
    keys = ('gamma', 'actor_lr', 'critic_lr', 'clip_eps', 'k_epochs', 'batch_size',
            'entropy_coef', 'reward_scale', 'gae_lambda', 'value_loss_coef', 'max_grad_norm')
    return PPOAgent(params.num_states, params.num_actions, params.hidden_layers_ppo,
                    device='cpu', activation=params.af_ppo,
                    **{k: getattr(params, k + '_ppo') for k in keys})


def make_environment(profiles):
    env, state = simpy.Environment(), EnvironmentState()
    for sid, p in profiles.items():
        state.add_server_and_init_environment(Server(
            env, sid, p['Site_ID'], p['Processing_Frequency'], p['Base_Failure_Rate'],
            p['Transmission_Rate'], p['Latitude'], p['Longitude']))
    return env, state


def environment_basis(key, correlation, base_rates):
    streams = [np.random.default_rng(np.random.SeedSequence(
        params.EXPERIMENT_SEED, spawn_key=(*key, role))) for role in range(2)]
    arrivals = streams[0].exponential(size=params.taskno)
    risk = rates = None
    if params.SPATIAL_RISK_ENABLED:
        risk = sample_spatial_risk_field(correlation, standard_normal=streams[1].standard_normal(params.NUM_SERVERS))
        rates = map_spatial_risk_to_effective_failure_rates(base_rates, risk, params.SPATIAL_RISK_BETA_P)
    return arrivals, risk, rates


def set_risk(state, basis, distances, correlation):
    if params.SPATIAL_RISK_ENABLED:
        state.set_episode_spatial_risk_context(list(state.servers), distances, correlation, basis[1], basis[2])


def select_latency_greedy_action(state, pairs):
    """Invert the current 42D observation; CPU work is already in seconds."""
    n = params.NUM_SERVERS
    fmin, fmax = params.SERVER_PROCESSING_FREQ_RANGE
    rmin, rmax = params.SERVER_TRANSMISSION_RATE_RANGE
    lmin, lmax = params.TASK_SIZE_RANGE
    cmin, cmax = params.TASK_DEMAND_RANGE
    state = np.asarray(state, dtype=np.float64)
    frequency = fmin + state[n:2*n] * (fmax-fmin)
    rate = rmin + state[2*n:3*n] * (rmax-rmin)
    work = state[3*n:4*n] + state[4*n:5*n]
    size = lmin + state[5*n] * (lmax-lmin)
    demand = cmin + state[5*n+1] * (cmax-cmin)
    predicted = np.maximum(size/rate, work) + demand/frequency
    return min(range(len(pairs)), key=lambda i: (
        min(predicted[pairs[i][0]-1], predicted[pairs[i][1]-1]),
        predicted[pairs[i][0]-1] + predicted[pairs[i][1]-1], i))


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
        actions.append(pairs.index((task.replica_A['server_id'], task.replica_B['server_id'])))
    data = np.asarray(parts)
    counts = np.bincount(actions, minlength=len(pairs))
    return {'arrival_rate': rate, 'episode_id': episode_id,
            'mean_latency_sec': data[:, 0].mean(),
            **{name: data[:, i].mean() for i, name in enumerate(METRICS[1:], start=1)},
            'episode_raw_reward': -data[:, 0].sum(), 'pair_hhi': np.sum((counts/len(tasks))**2),
            **{f'action_count_{i}': int(count) for i, count in enumerate(counts)}}


class TrainingLoop(MainLoop):
    """Use independent episode streams while retaining the production PPO loop."""
    def __init__(self, agent, rate, seed, episodes, distances, correlation, base_rates):
        self.rate, self.seed = rate, seed
        self.distances, self.correlation, self.base_rates = distances, correlation, base_rates
        super().__init__(agent, episodes, params.taskno, params.num_states, params.num_actions)

    def _initialize_episode_spatial_risk(self):
        self.basis = environment_basis((6, 2, RATE_STREAM_IDS[self.rate], self.seed, self.this_episode),
                                       self.correlation, self.base_rates)
        set_risk(self.env_state, self.basis, self.distances, self.correlation)

    def _sample_interarrival_time(self):
        return float(self.basis[0][self.taskCounter-1] / self.rate)


def evaluate_episode(selector, rate, episode_id, profiles, tasks, basis, distances, correlation):
    env, state = make_environment(profiles)
    set_risk(state, basis, distances, correlation)
    pairs = MainLoop.generate_combinations()

    def arrivals():
        for index, (tid, profile) in enumerate(tasks.items()):
            yield env.timeout(float(basis[0][index]/rate))
            task = Task(env, state, tid, profile)
            state.add_task(task)
            action = selector(state.get_state(task))
            a, b = (state.get_server_by_id(sid) for sid in pairs[action])
            task.replica_A_reliability = state.compute_replica_reliability(task, a)
            task.replica_B_reliability = state.compute_replica_reliability(task, b)
            task.pair_reliability = state.compute_pair_reliability(task, a, b)
            env.process(task.execute_task(a, b))

    env.process(arrivals()); env.run()
    completed = list(state.tasks.values())
    assert len(completed) == params.taskno
    assert all(r['finish_time'] is not None for t in completed for r in (t.replica_A, t.replica_B))
    np.testing.assert_allclose([t.arrival_time for t in completed], np.cumsum(basis[0]/rate), rtol=0, atol=1e-9)
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
    _, state = make_environment(profiles)
    _, distances = build_distance_matrix([v['server_object'] for v in state.servers.values()])
    correlation = build_spatial_correlation_matrix(distances, compute_baseline_correlation_length_km(distances))
    base_rates = sf.Base_Failure_Rate.to_numpy()
    pairs = MainLoop.generate_combinations()
    training, evaluation = [], []
    for seeds in seed_mapping([rate], seed_ids).itertuples(index=False):
        torch.manual_seed(seeds.init_seed)
        agent = build_agent()
        agent.save_model(case/'checkpoints'/f'seed_{seeds.seed_id:02d}_initial.pt')
        torch.manual_seed(seeds.action_seed); np.random.seed(seeds.shuffle_seed)
        loop = TrainingLoop(agent, rate, seeds.seed_id, train_episodes, distances, correlation, base_rates)
        loop.task_profiles = scale_task_profiles(loop.task_profiles, task_demand_multiplier,
                                                task_size_multiplier=task_size_multiplier)
        original_train = agent.train_step
        tags = {'task_demand_multiplier': task_demand_multiplier, 'task_size_multiplier': task_size_multiplier,
                'seed_id': seeds.seed_id, 'training_seed': seeds.training_seed}

        def train_step():
            assert np.asarray(agent.states).shape == (params.taskno, params.num_states)
            assert agent.dones == [False]*(params.taskno-1) + [True]
            tasks = list(loop.env_state.tasks.values())
            np.testing.assert_array_equal(agent.rewards, [-(t.task_completion_time-t.arrival_time) for t in tasks])
            metrics = task_metrics(tasks, rate, loop.this_episode)
            original_train()
            training.append({**tags, 'method': 'ppo_final', **metrics})
            if loop.this_episode % 50 == 0:
                print(f'size={task_size_multiplier:g} demand={task_demand_multiplier:g} rate={rate:g} seed={seeds.seed_id} train={loop.this_episode}/{train_episodes}', file=sys.stderr, flush=True)

        agent.train_step = train_step
        with contextlib.redirect_stdout(io.StringIO()): loop.EP()
        checkpoint = case/'checkpoints'/f'seed_{seeds.seed_id:02d}_final.pt'
        agent.save_model(checkpoint)
        frozen = build_agent(); frozen.load_model(checkpoint)
        frozen.policy_old.eval()
        snapshot = copy.deepcopy({'actor': frozen.policy_net.state_dict(), 'old_actor': frozen.policy_old.state_dict(),
                                  'critic': frozen.value_net.state_dict(), 'actor_optimizer': frozen.optimizer_policy.state_dict(),
                                  'critic_optimizer': frozen.optimizer_value.state_dict()})
        for episode_id in range(1, eval_episodes+1):
            basis = environment_basis((6, 3, seeds.seed_id, episode_id), correlation, base_rates)
            torch.manual_seed(derived_seed((6, 3, seeds.seed_id, episode_id, 3)))
            ppo, ppo_tasks = evaluate_episode(lambda s: frozen.select_action(s, 0.), rate, episode_id,
                                             profiles, eval_tasks, basis, distances, correlation)
            greedy, greedy_tasks = evaluate_episode(lambda s: select_latency_greedy_action(s, pairs), rate,
                                                   episode_id, profiles, eval_tasks, basis, distances, correlation)
            assert [(t.id, t.arrival_time, t.task_size, t.computation_demand) for t in ppo_tasks] == [
                (t.id, t.arrival_time, t.task_size, t.computation_demand) for t in greedy_tasks]
            same_tree(ppo_tasks[0].env_state.effective_failure_rates, greedy_tasks[0].env_state.effective_failure_rates)
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
    seeds = evaluation.groupby(['arrival_rate', 'seed_id', 'training_seed', 'method'])[list(METRICS)].mean().reset_index()
    rows = []
    for rate, data in seeds.groupby('arrival_rate'):
        ppo = data[data.method == 'ppo_final'].set_index('seed_id')
        greedy = data[data.method == 'latency_greedy'].set_index('seed_id')
        for metric in METRICS:
            delta = ppo[metric]-greedy[metric]
            half = student_t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            rows.append({'arrival_rate': rate, 'reference_method': 'latency_greedy', 'metric': metric,
                         'seed_count': len(delta), 'ppo_mean': ppo[metric].mean(), 'ppo_sd': ppo[metric].std(ddof=1),
                         'reference_mean': greedy[metric].mean(), 'reference_sd': greedy[metric].std(ddof=1),
                         'paired_mean_difference': delta.mean(), 'paired_difference_sd': delta.std(ddof=1),
                         'ci95_lower': delta.mean()-half, 'ci95_upper': delta.mean()+half,
                         'improvement_mean_pct': ((greedy[metric]-ppo[metric])/greedy[metric]*100).mean(),
                         'ppo_better_seed_count': int((delta < 0).sum())})
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
             '42维；CPU 1–4 GHz，上传5–20 MB/s，任务计算需求2–4×10⁹ cycles，任务大小5–10 MB；运行与等待工作量直接使用秒数。',
             f'gamma={params.gamma_ppo}（按秒折扣），GAE lambda={params.gae_lambda_ppo}；单任务负延迟回填。空间风险开启={params.SPATIAL_RISK_ENABLED}，可靠性不筛选动作或惩罚奖励。',
             'Greedy从相同状态还原物理量，预测max(上传,运行＋等待)+执行；组合按最小副本时间、两副本时间之和、编号排序。不观察在途或未来任务。',
             '训练和评估使用独立固定任务集。两方法使用相同任务、到达和空间场；所有400副本排空。每种子先平均20评估轮，再配对训练种子，计算t区间。改善率逐种子计算后平均。区间为探索性逐指标比较，未作多重比较校正。', '',
             '| 到达率 | PPO平均延迟(s) | Greedy平均延迟(s) | 改善率(%) | PPO−Greedy 95% CI(s) |',
             '|---|---:|---:|---:|---|']
    for r in paired[paired.metric == 'mean_latency_sec'].itertuples():
        lines.append(f'| {r.arrival_rate:g} | {r.ppo_mean:.6f} | {r.reference_mean:.6f} | {r.improvement_mean_pct:.3f} | [{r.ci95_lower:.6f}, {r.ci95_upper:.6f}] |')
    lines.extend(['', '![评估结果](evaluation.png)', '', '命令：`python compare_ppo_greedy.py`'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
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


if __name__ == '__main__':
    main()
