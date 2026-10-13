"""Seven task-size multipliers at 2 tasks/s, using the existing PPO/Greedy runner."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import compare_ppo_greedy as experiment

TASK_SIZE_MULTIPLIERS = (.5, 1., 1.5, 2., 2.5, 3., 3.5)
ARRIVAL_RATE = 2.
OUTPUT_DIR = Path(experiment.RESULTS_DIR) / 'task_size_ppo_vs_latency_greedy'


def run_multiplier(job):
    multiplier, seed_ids, train_episodes, eval_episodes, output = job
    case_root = Path(output) / f'size_{multiplier:g}'
    return experiment.run_rate((ARRIVAL_RATE, seed_ids, train_episodes, eval_episodes, case_root),
                               task_size_multiplier=multiplier)


def summarize(output, cases):
    output = Path(output)
    training = pd.concat([pd.read_csv(p/'train_episode_metrics.csv', float_precision='round_trip') for p in cases])
    evaluation = pd.concat([pd.read_csv(p/'eval_episode_metrics.csv', float_precision='round_trip') for p in cases])
    keys = ['task_size_multiplier', 'arrival_rate', 'seed_id', 'training_seed', 'method']
    seeds = evaluation.groupby(keys)[list(experiment.METRICS)].mean().reset_index()
    rows = []
    for multiplier, data in seeds.groupby('task_size_multiplier'):
        ppo = data[data.method == 'ppo_final'].set_index('seed_id')
        greedy = data[data.method == 'latency_greedy'].set_index('seed_id')
        for metric in experiment.METRICS:
            delta = ppo[metric]-greedy[metric]
            half = student_t.ppf(.975, len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
            rows.append({'task_size_multiplier': multiplier, 'arrival_rate': ARRIVAL_RATE,
                         'reference_method': 'latency_greedy', 'metric': metric, 'seed_count': len(delta),
                         'ppo_mean': ppo[metric].mean(), 'ppo_sd': ppo[metric].std(ddof=1),
                         'reference_mean': greedy[metric].mean(), 'reference_sd': greedy[metric].std(ddof=1),
                         'paired_mean_difference': delta.mean(), 'paired_difference_sd': delta.std(ddof=1),
                         'ci95_lower': delta.mean()-half, 'ci95_upper': delta.mean()+half,
                         'improvement_mean_pct': ((greedy[metric]-ppo[metric])/greedy[metric]*100).mean(),
                         'ppo_better_seed_count': int((delta < 0).sum())})
    paired = pd.DataFrame(rows)
    for name, frame in [('train_episode_metrics.csv', training), ('eval_episode_metrics.csv', evaluation),
                        ('eval_seed_summary.csv', seeds), ('paired_comparison.csv', paired)]:
        frame.to_csv(output/name, index=False, float_format='%.17g')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, experiment.METRICS):
        for method, data in seeds.groupby('method'):
            group = data.groupby('task_size_multiplier')[metric]
            ax.errorbar(group.mean().index, group.mean(), yerr=group.std(ddof=1), marker='o', capsize=3, label=method)
        ax.set(xlabel='Task size multiplier', ylabel=metric); ax.grid(alpha=.3)
    axes[0].legend(); fig.tight_layout(); fig.savefig(output/'evaluation.png', dpi=150); plt.close(fig)
    comparison = paired[paired.metric == 'mean_latency_sec']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(comparison.task_size_multiplier, comparison.improvement_mean_pct, marker='o')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(xlabel='Task size multiplier', ylabel='PPO mean latency improvement (%)')
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(output/'improvement.png', dpi=150); plt.close(fig)

    sf = pd.read_excel(experiment.SERVER_INFO_PATH, sheet_name='Servers')
    tasks = pd.read_excel(experiment.EVALUATION_TASK_PATH)
    base_load = 2 * ARRIVAL_RATE * tasks.Computation_Demand.mean() / sf.Processing_Frequency.sum()
    lines = ['# 任务大小倍率：PPO与Latency Greedy平均延迟对比', '',
             f'到达率固定{ARRIVAL_RATE:g} tasks/s；每档{seeds.seed_id.nunique()}种子×{training.episode_id.max()}训练轮×{evaluation.episode_id.max()}评估轮。',
             '每档从头初始化、独立训练到最后一轮后冻结，PPO按概率采样，不选模型、不调参。各倍率同seed共享初始权重及动作/shuffle/到达/空间场的随机流。',
             '训练和评估分别复用独立固定的200任务。只有Task_Size乘倍率k；计算需求、任务顺序、服务器和可靠性要求保持不变，Excel不改写。',
             f'42维，CPU 1–4 GHz，上传5–20 MB/s，基础需求2–4×10⁹ cycles；大小min–max范围固定为{experiment.params.TASK_SIZE_RANGE} MB。运行与等待工作量均用原始秒数，不压缩、不裁剪、不随k调整输入映射。',
             f'gamma={experiment.params.gamma_ppo}（按秒折扣），GAE lambda={experiment.params.gae_lambda_ppo}，单任务负延迟回填，空间风险开启={experiment.params.SPATIAL_RISK_ENABLED}。可靠性不筛选动作或惩罚奖励。',
             'Greedy使用相同状态还原物理量，预测max(上传,运行＋等待)+执行；按组合的首完成预测、两副本时间之和、编号排序。两方法评估任务、到达和风险场配对；评估不更新网络或优化器。',
             '每种子先平均评估Episode，再按10个种子配对计算t区间；改善率先逐种子算后平均。区间为探索性逐倍率比较，未作多重比较校正。只研究平均延迟及平均上传、等待、执行时间，不计算P95。', '',
             '| 倍率 | 名义CPU负载 | PPO平均延迟(s) | Greedy平均延迟(s) | 改善率(%) | PPO−Greedy 95% CI(s) |',
             '|---|---:|---:|---:|---:|---|']
    for r in comparison.itertuples():
        lines.append(f'| {r.task_size_multiplier:g} | {100*base_load:.1f}% | {r.ppo_mean:.6f} | {r.reference_mean:.6f} | {r.improvement_mean_pct:.3f} | [{r.ci95_lower:.6f}, {r.ci95_upper:.6f}] |')
    lines.extend(['', '各档名义CPU负载相同（约55.9%）；上传速率固定，每个副本独立上传，未建模共享链路带宽竞争。所有档均为200任务到达后排空副本的有限任务集结果。', '',
                  '![平均延迟及分解](evaluation.png)', '', '![平均延迟改善率](improvement.png)', '',
                  '命令：`python scan_task_size.py`'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for multiplier in TASK_SIZE_MULTIPLIERS:
        frame = experiment.seed_mapping([ARRIVAL_RATE])
        frame.insert(0, 'task_size_multiplier', multiplier)
        frames.append(frame)
    pd.concat(frames).to_csv(OUTPUT_DIR/'seed_mapping.csv', index=False)
    files = [Path(p) for p in (experiment.SERVER_INFO_PATH, experiment.TRAIN_TASK_PATH, experiment.EVALUATION_TASK_PATH)] + [
        Path('config/configuration.py'), Path('config/params.py'), Path('agents/ppo_agent.py'),
        Path('core/main_loop.py'), Path('core/env_state.py'), Path('core/task.py'), Path('core/server.py'),
        Path('compare_ppo_greedy.py'), Path('scan_task_size.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUTPUT_DIR/'input_manifest.json').write_text(json.dumps(hashes, indent=2))
    jobs = [(k, tuple(range(1, 11)), 300, 20, OUTPUT_DIR) for k in TASK_SIZE_MULTIPLIERS]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        cases = list(pool.map(run_multiplier, jobs))
    summarize(OUTPUT_DIR, cases)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    print(f'Complete: {OUTPUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
