"""Create the formal masked Pair PPO integration report from one smoke run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'diagnostics/results/masked_pair_ppo_smoke'
REPORT=ROOT/'diagnostics/results/MASKED_PAIR_PPO_INTEGRATION.md'


def markdown(frame):
    cols=list(frame.columns)
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{item:.4f}' if isinstance(item,(float,np.floating)) else str(item) for item in row)+' |')
    return '\n'.join(lines)


def run(directory=DEFAULT):
    directory=Path(directory)
    meta=json.loads((directory/'run_metadata.json').read_text())
    overall=pd.read_csv(directory/'smoke_policy_summary.csv').set_index('policy')
    updates=pd.read_csv(directory/'masked_pair_updates.csv')
    masked_train=pd.read_csv(directory/'masked_pair_train_decisions.csv')
    train_tasks=pd.read_csv(directory/'masked_pair_train_task_assignments.csv')
    train_ep=pd.read_csv(directory/'masked_pair_training_episode_metrics.csv')
    baseline=pd.read_csv(directory/'legacy_pair_training_curve.csv')
    masked_curve=pd.read_csv(directory/'masked_pair_training_curve.csv')
    loads=pd.concat([pd.read_csv(directory/f'{name}_server_load.csv') for name in overall.index])
    mask_check=meta['mask_comparison']
    deployed_path=directory/'deployment/masked_pair_stochastic_eval_task_assignments.csv'
    if deployed_path.exists():
        original=pd.read_csv(directory/'masked_pair_stochastic_eval_task_assignments.csv')
        deployed=pd.read_csv(deployed_path)
        exact=bool(original.equals(deployed))
        deployment={'reloaded_checkpoint_default_mode':'masked_stochastic',
            'evaluation_rows':len(deployed),'exact_task_assignment_match':exact,
            'action_mismatch_count':int((original.action_index!=deployed.action_index).sum())}
        if not exact:raise RuntimeError('Reloaded default deployment differs from original evaluation')
        (directory/'deployment_validation.json').write_text(json.dumps(deployment,indent=2),encoding='utf-8')
    else:
        deployment=json.loads((directory/'deployment_validation.json').read_text())
    if not len(updates)==len(train_ep)==meta['train_episodes']:
        raise RuntimeError('Training episode/update summaries are incomplete')
    if not np.isfinite(updates.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError('Non-finite update diagnostics')
    if len(masked_train)!=len(train_tasks):
        raise RuntimeError('Incomplete masked training decision log')
    pair78=loads[loads.server_id.isin([7,8])][['policy','server_id','selection_count',
        'utilization','mean_queue_length','p95_queue_length','mean_waiting_time']]
    summary=overall[['mean_reward','mean_latency','overall_rsr','highest_rsr','conditional_rsr',
        'feasibility_rate','empty_safe_set_rate','avoidable_violation_count',
        'unavoidable_violation_count','pair_78_frequency','pair_selection_hhi',
        'pair_selection_entropy']].reset_index()
    max_first_ratio=float(updates.first_minibatch_ratio_max_abs_error.max())
    min_ratio=float(updates.ratio_min.min());max_ratio=float(updates.ratio_max.max())
    empty=masked_train[masked_train.safe_set_empty.astype(bool)]
    fallback_stats={'training_empty_count':len(empty),
        'training_empty_rate':len(empty)/len(masked_train),
        'mean_deficit_when_empty':float(empty.reliability_deficit.mean()) if len(empty) else 0.0,
        'max_deficit_when_empty':float(empty.reliability_deficit.max()) if len(empty) else 0.0,
        'all_empty_have_positive_deficit':bool((empty.reliability_deficit>0).all()) if len(empty) else True}
    (directory/'fallback_summary.json').write_text(json.dumps(fallback_stats,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(baseline.episode,baseline.episode_reward,alpha=.4,label='Legacy Pair PPO')
    ax.plot(masked_curve.episode,masked_curve.episode_reward,alpha=.4,label='Masked Pair PPO')
    ax.plot(baseline.episode,baseline.episode_reward.rolling(10,min_periods=1).mean(),label='Legacy 10-ep mean')
    ax.plot(masked_curve.episode,masked_curve.episode_reward.rolling(10,min_periods=1).mean(),label='Masked 10-ep mean')
    ax.set(xlabel='Training episode',ylabel='Episode reward');ax.legend()
    fig.tight_layout();fig.savefig(directory/'training_reward_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,column,title in zip(axes,('mean_queue_length','p95_queue_length'),('Mean waiting queue','P95 waiting queue')):
        for policy,group in loads.groupby('policy'):
            ax.plot(group.server_id,group[column],marker='o',label=policy)
        ax.set(xlabel='Server ID',ylabel=title)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(directory/'server_queue_comparison.png');plt.close(fig)
    feasible=~masked_train.safe_set_empty.astype(bool)
    satisfied=train_tasks.Reliability_Satisfied.astype(bool)
    train_avoidable=int((feasible & ~satisfied).sum())
    if train_avoidable:
        raise RuntimeError(f'Masked training selected {train_avoidable} avoidably unreliable tasks')
    stochastic=overall.loc['masked_pair_stochastic']
    legacy=overall.loc['legacy_pair_greedy']
    greedy=overall.loc['masked_pair_greedy']
    stable=bool(max_first_ratio<1e-5 and train_avoidable==0
        and stochastic.avoidable_violation_count==0
        and mask_check['safe_mask_mismatch_count']==0
        and stochastic.pair_selection_hhi < greedy.pair_selection_hhi)
    lines=['# Reliability-Masked Pair PPO Integration','',
        f'独立新算法采用 Pair Actor 与原 Critic、reward、SMDP-GAE/PPO 超参数；仅将状态对应的有效动作 mask 贯穿采样和 PPO 更新。单 seed、{meta["train_episodes"]}×{meta["tasks_per_episode"]} 训练任务；{meta["eval_episodes"]}×{meta["tasks_per_episode"]} 冻结评估任务。旧 Pair PPO 与新 Agent 从逐张量相同的 Actor/Critic 初始化开始，训练使用相同到达与空间风险流，评估也共享外生流。此处不是正式 10-seed 结论。','',
        '## A. 实现改动','',
        '`agents/masked_pair_ppo_agent.py` 提供独立 Agent；`config/masked_pair_ppo.py` 保持新算法配置独立；`tools/run_masked_pair_ppo.py evaluate` 默认 **masked stochastic**，`--deployment greedy` 仅用于消融。`MainLoop` 只增加新 Agent 可选的当前任务上下文 hook；旧 Flat/Pair PPO 不实现该 hook。新 Agent 对每个合法 pair 调用生产 `Task.initialize_reliability_evaluation()`，不新增可靠性公式或 rho 阈值。','',
        '## B. PPO 数学一致性','',
        '安全集非空时 effective mask 为 safe mask；空集时为所有 max-reliability pair（数值容差 1e-12）。Actor 原始 logits 对 mask 外动作赋 `-inf`，由同一 `Categorical` 分布产生 action、old log probability、entropy。rollout 按 task ID 保存 effective mask；PPO minibatch 对应行将**相同 mask**施加于新 logits，再计算 new log probability、ratio 和仅安全支持上的 entropy。Critic、GAE、reward 与 PPO loss 系数保持原样。','',
        f'{len(updates)} 次训练更新均有限；每回合首次 minibatch 的 ratio 相对 1 最大误差 {max_first_ratio:.3g}；全部 minibatch ratio 范围 [{min_ratio:.4f}, {max_ratio:.4f}]。固定 task/requirement/episode hazard 样本的 safe mask 对照 {mask_check["states_compared"]} 条，mismatch **{mask_check["safe_mask_mismatch_count"]}**；从磁盘独立加载 checkpoint 后默认 stochastic 部署的全部 {deployment["evaluation_rows"]} 条评估任务与原评估逐条相同。','',
        '## C. 单 seed smoke 结果','',markdown(summary),'',
        f'训练奖励前 10 回合均值：旧 {baseline.episode_reward.head(10).mean():.2f}、新 {masked_curve.episode_reward.head(10).mean():.2f}；后 10 回合均值：旧 {baseline.episode_reward.tail(10).mean():.2f}、新 {masked_curve.episode_reward.tail(10).mean():.2f}。逐回合曲线见 `training_reward_comparison.png`。', '',
        '## D. Safe-set 与队列行为','',
        f'新算法训练集的 mean safe-set size {train_ep.mean_safe_set_size.mean():.2f}，feasibility rate {train_ep.feasibility_rate.mean():.1%}，conditional RSR {train_ep.conditional_rsr.mean():.1%}；可避免违约总数 {train_avoidable}。评估时默认 masked stochastic 的 feasibility rate {stochastic.feasibility_rate:.1%}、conditional RSR {stochastic.conditional_rsr:.1%}、overall RSR {stochastic.overall_rsr:.1%}。', '',
        markdown(pair78),'',
        f'(7,8) 选择率：旧 Greedy {legacy.pair_78_frequency:.1%}，新 masked stochastic {stochastic.pair_78_frequency:.1%}，masked greedy {greedy.pair_78_frequency:.1%}；HHI 分别 {legacy.pair_selection_hhi:.3f}/{stochastic.pair_selection_hhi:.3f}/{greedy.pair_selection_hhi:.3f}。服务器 7/8 的真实等待队列均值、P95 来自入队、获 CPU、完成事件的时间加权统计，不是 backlog proxy。', '',
        '## E. Empty-safe-set 与 policy/system 分界','',
        f'训练阶段有 {fallback_stats["training_empty_count"]} 次空安全集（{fallback_stats["training_empty_rate"]:.1%}）；这些状态全部使用 max-reliability fallback，平均 deficit {fallback_stats["mean_deficit_when_empty"]:.6g}、最大 deficit {fallback_stats["max_deficit_when_empty"]:.6g}。并列最可靠动作保留 Actor 概率重新归一化抽样。评估时的 `avoidable_violation_count` 区分有可行 pair 却选择违约的策略失败；`unavoidable_violation_count` 是空安全集的系统不可行。`conditional_rsr` 只以非空安全集任务为分母，不能与 overall RSR 混用。', '',
        '## F. Legacy baseline 完整性','',
        '旧 PPOAgent、Pair Actor、Flat PPO 的采样/更新函数未修改。新算法为独立类和入口。旧 Pair PPO 的训练前 100 回合 episode reward 与已保存的正式 300 回合 trial 0 前缀逐值相同；初始化权重逐张量相同，外生流一致。旧评估端仍使用原 greedy 行为；新入口明确默认 stochastic，未调用旧的 greedy-only 冻结评估器。', '',
        '## G. 是否进入正式 10-seed','',
        ('**具备进入正式 10-seed 比较的工程条件。**' if stable else '**尚不具备进入正式 10-seed 的条件。**')+
        ' 本结论只依据此单 seed smoke 的数值稳定性、mask 一致性、可避免违约、动作集中度和旧版回归；正式论文结论仍需多 seed 评估。', '',
        '运行命令（仓库根目录）：','','```bash',
        '/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python tools/run_masked_pair_ppo.py train-smoke --train-episodes 100 --eval-episodes 20',
        '/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python tools/run_masked_pair_ppo.py evaluate --checkpoint-dir diagnostics/results/masked_pair_ppo_smoke',
        '/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python diagnostics/write_masked_pair_ppo_integration_report.py',
        '```','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    return REPORT

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results-dir',type=Path,default=DEFAULT)
    args=parser.parse_args()
    print(run(args.results_dir))
