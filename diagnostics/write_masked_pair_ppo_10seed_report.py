"""Write evidence-driven report from completed formal paired-seed CSVs."""
from __future__ import annotations

import sys
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from diagnostics.aggregate_masked_pair_ppo_10seed import OUT, POLICIES


def fmt_mean_std(frame,metric,policy):
    values=frame.loc[frame.policy.eq(policy),metric].to_numpy(dtype=float)
    if len(values)!=10 or not np.isfinite(values).all():raise RuntimeError(f'Missing {policy} {metric}')
    return f'{values.mean():.4f} ± {values.std(ddof=1):.4f}'


def ci_line(ci,wlt,comparison,metric,unit=''):
    row=ci[(ci.comparison==comparison)&(ci.metric==metric)].iloc[0]
    win=wlt[(wlt.comparison==comparison)&(wlt.metric==metric)].iloc[0]
    return (f'| {metric} | {row.mean_delta:+.4f}{unit} | '
            f'[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | '
            f'{int(win.positive_seeds)}/{int(win.negative_seeds)}/{int(win.ties)} | '
            f'{"否" if row.ci95_low<=0<=row.ci95_high else "是"} |')


def main():
    result=pd.read_csv(OUT/'evaluation_seed_summary.csv')
    tiers=pd.read_csv(OUT/'requirement_level_results.csv')
    ci=pd.read_csv(OUT/'bootstrap_ci.csv')
    wlt=pd.read_csv(OUT/'win_loss_tie.csv')
    train=pd.read_csv(OUT/'training_seed_summary.csv')
    loads=pd.read_csv(OUT/'server_load_results.csv')
    paired=pd.read_csv(OUT/'paired_comparisons.csv')
    if len(result)!=40 or result.trial_id.nunique()!=10:raise RuntimeError('Formal experiment is incomplete')
    lines=['# Reliability-Masked Pair PPO: Formal Paired 10-Seed Experiment','',
           '## 1. Experimental Setup','',
           'Flat PPO、Pair PPO 和 Reliability-Masked Pair PPO 分别按此前正式实验的相同 10 个 seed 训练。'
           '每个 seed：300 回合训练、每回合 200 任务；20 回合评估、每回合 200 任务。'
           'Flat/Pair 采用此前正式定义的 greedy 部署；Masked Pair 的正式部署是 masked stochastic。'
           'Masked greedy 仅使用**同一个 masked checkpoint**作 evaluation-time 消融。'
           '训练初始化、任务到达、空间风险与各算法配对；masked 动作抽样使用独立、固定的 Torch seed。本目录保存 `formal_seed_plan.csv` 和 `formal_reference_metadata.json` 快照，以便远端复现。'
           '脚本逐 seed 核验 Flat/Pair 的训练奖励曲线和评估动作、reward 与历史正式结果一致。',
           '', '统计单位是训练 seed（n=10），以下所有标准差为 seed 间 sample std。'
           '配对差值的 95% CI 由 20,000 次 seed-level paired bootstrap 得到（seed=2043）。'
           'win/loss/tie 容差为 1e-10；latency、HHI 和负载越低越好。','',
           '## 2. Main Results','',
           '| 指标 | Flat PPO | Pair PPO | Masked Pair stochastic |',
           '|---|---:|---:|---:|']
    primary=('mean_reward','mean_latency','p50_latency','p90_latency','p95_latency',
             'overall_rsr','highest_rsr')
    for metric in primary:
        lines.append('| '+metric+' | '+' | '.join(fmt_mean_std(result,metric,p) for p in ('flat','pair','masked_stochastic'))+' |')
    lines += ['', '各可靠性需求档的 RSR（seed-level mean ± std）：', '',
              '| R_req | Flat | Pair | Masked stochastic |',
              '|---:|---:|---:|---:|']
    for requirement in (.9,.99,.999,.9999):
        group=tiers[np.isclose(tiers.R_req,requirement)]
        lines.append(f'| {requirement} | '+' | '.join(fmt_mean_std(group,'overall_rsr',p) for p in ('flat','pair','masked_stochastic'))+' |')
    lines += ['', '## 3. Paired Statistical Results','',
              'Delta 均定义为 Masked stochastic − Pair greedy；表中 win/loss/tie 按指标改善方向计算。','',
              '| 指标 | mean delta | paired bootstrap 95% CI | win/loss/tie | CI 不跨 0 |',
              '|---|---:|---:|---:|---|']
    for metric in ('mean_reward','mean_latency','overall_rsr','highest_rsr',
                   'pair_selection_hhi','maximum_mean_queue_length','avoidable_violation_rate'):
        lines.append(ci_line(ci,wlt,'masked_minus_pair',metric))
    lines += ['', '## 4. Reliability Feasibility Analysis','',
              'Pair PPO 的 safe-set/AVR 数值是按其实际决策状态**事后重放**生产可靠性规则的诊断，'
              '不是旧 Pair PPO 的内部约束。','',
              '| 指标 | Pair post-hoc | Masked stochastic |',
              '|---|---:|---:|']
    for metric in ('feasibility_rate','conditional_rsr','overall_rsr',
                   'avoidable_violation_rate','unavoidable_violation_rate',
                   'empty_safe_set_rate','mean_empty_deficit','p95_empty_deficit'):
        lines.append(f'| {metric} | {fmt_mean_std(result,metric,"pair")} | {fmt_mean_std(result,metric,"masked_stochastic")} |')
    high=tiers[np.isclose(tiers.R_req,.9999)]
    lines += ['', '最高需求档 R_req=0.9999：','',
              '| 指标 | Pair post-hoc | Masked stochastic |',
              '|---|---:|---:|']
    for metric in ('feasibility_rate','conditional_rsr','overall_rsr',
                   'avoidable_violation_rate','unavoidable_violation_rate',
                   'empty_safe_set_rate','mean_empty_deficit','p95_empty_deficit'):
        lines.append(f'| {metric} | {fmt_mean_std(high,metric,"pair")} | {fmt_mean_std(high,metric,"masked_stochastic")} |')
    lines += ['', '## 5. Deployment Ablation','',
              'Masked stochastic 与 masked greedy 在每个 seed 中复用同一训练 checkpoint、相同任务/环境流和相同 safe mask。'
              'Delta=stochastic−greedy。','',
              '| 指标 | mean delta | paired bootstrap 95% CI | win/loss/tie | CI 不跨 0 |',
              '|---|---:|---:|---:|---|']
    for metric in ('mean_reward','mean_latency','overall_rsr','conditional_rsr',
                   'pair_selection_hhi','top1_pair_frequency','maximum_server_selection_share',
                   'mean_server_queue_length','mean_server_p95_queue_length',
                   'maximum_mean_queue_length','maximum_p95_queue_length'):
        lines.append(ci_line(ci,wlt,'stochastic_minus_greedy',metric))
    lines += ['', '## 6. Pair / Server Concentration','',
              '| 指标 | Pair PPO | Masked stochastic | Masked greedy |',
              '|---|---:|---:|---:|']
    for metric in ('pair_selection_hhi','pair_selection_entropy','top1_pair_frequency',
                   'unique_selected_pairs','maximum_server_selection_share',
                   'maximum_server_utilization','maximum_mean_queue_length',
                   'maximum_p95_queue_length','mean_server_waiting_time','pair_78_frequency'):
        lines.append('| '+metric+' | '+' | '.join(fmt_mean_std(result,metric,p) for p in ('pair','masked_stochastic','masked_greedy'))+' |')
    lines += ['', '服务器逐 seed 的 selection count、utilization、真实等待队列长度均值/P95、平均等待时间'
              '见 `server_load_results.csv`；动作分布见 `pair_selection_results.csv`。',
              '', '## 7. Seed Stability','']
    for metric in ('mean_reward','mean_latency','overall_rsr','highest_rsr','pair_selection_hhi'):
        row=ci[(ci.comparison=='masked_minus_pair')&(ci.metric==metric)].iloc[0]
        values=paired[paired.comparison.eq('masked_minus_pair')][f'delta_{metric}'].to_numpy(dtype=float)
        w=wlt[(wlt.comparison=='masked_minus_pair')&(wlt.metric==metric)].iloc[0]
        lines.append(f'- {metric}: mean delta {row.mean_delta:+.4f}; seed delta 范围 '
                     f'[{values.min():+.4f}, {values.max():+.4f}]; '
                     f'win/loss/tie {int(w.positive_seeds)}/{int(w.negative_seeds)}/{int(w.ties)}; '
                     f'CI {"跨 0" if row.ci95_low<=0<=row.ci95_high else "不跨 0"}。')
    lines += ['', '训练稳定性：','',
              '| 指标 | 10-seed mean ± std |', '|---|---:|']
    masked_train=train[train.policy.eq('masked')]
    for metric in ('first30_masked_entropy','last30_masked_entropy','first30_pair_hhi',
                   'last30_pair_hhi','final30_top1_frequency','mean_safe_set_size',
                   'empty_safe_set_rate','ratio_min','ratio_max','max_first_ratio_error'):
        lines.append(f'| {metric} | {fmt_mean_std(masked_train,metric,"masked")} |')
    sanity=json.loads((OUT/'sanity_checks.json').read_text())
    lines.append(f'')
    lines.append(f'独立 sanity checks：{sanity["trials"]} 个 seed×{sanity["policies_per_trial"]} 种部署均有 4000 条评估记录；每个 Masked seed 有 {sanity["masked_updates_per_seed"]} 次有限 PPO 更新；有效 mask 支持、最大可靠性 fallback、同 checkpoint/同 safe mask 均通过；可避免违约总数 {sanity["masked_avoidable_violations"]}。')
    lines += ['', '## 8. Failure Cases','']
    abnormal=[]
    for trial in range(10):
        m=result[(result.trial_id==trial)&result.policy.eq('masked_stochastic')].iloc[0]
        h=high[(high.trial_id==trial)&high.policy.eq('masked_stochastic')].iloc[0]
        if m.avoidable_violation_count or h.avoidable_violation_count:
            abnormal.append(f'trial {trial}: 非零可避免违约')
        if m.conditional_rsr < 1-1e-10:
            abnormal.append(f'trial {trial}: conditional RSR<1')
    lines.append('Masked 约束异常 seed：'+('；'.join(abnormal) if abnormal else '无（所有 10 seed 可避免违约为 0）。'))
    worst=high[high.policy.eq('masked_stochastic')].sort_values('overall_rsr').iloc[0]
    lines.append(f'最高需求档最低 RSR 为 trial {int(worst.trial_id)} 的 {worst.overall_rsr:.4f}；'
                 f'该 seed 可行率 {worst.feasibility_rate:.4f}、空安全集率 {worst.empty_safe_set_rate:.4f}。')
    lines.append('具体空安全集 deficit 与各 seed 的最高需求档可行上限见 `requirement_level_results.csv`。')
    lines += ['', '## 9. Conclusion','']
    def good(metric):
        r=ci[(ci.comparison=='masked_minus_pair')&(ci.metric==metric)].iloc[0]
        w=wlt[(wlt.comparison=='masked_minus_pair')&(wlt.metric==metric)].iloc[0]
        sign=-1 if metric in ('mean_latency','pair_selection_hhi','maximum_mean_queue_length') else 1
        return sign*r.mean_delta>0 and int(w.positive_seeds)>=8 and (
            r.ci95_low>0 if sign>0 else r.ci95_high<0)
    masked=result[result.policy.eq('masked_stochastic')]
    pair=result[result.policy.eq('pair')]
    if all(good(m) for m in ('mean_reward','overall_rsr','mean_latency','maximum_mean_queue_length')):
        choice='A'
        reason='可靠性、reward、latency 与最大服务器平均队列均在多数 seed 同向改善，且各配对 CI 不跨 0。'
    elif (masked.mean_reward.mean()<=pair.mean_reward.mean() and
          masked.overall_rsr.mean()<=pair.overall_rsr.mean()):
        choice='C'
        reason='单 seed smoke 中的主要 reward/RSR 收益未在 10 seed 均值中保持。'
    else:
        choice='B'
        reason='mask 对可行任务的可靠性约束稳定生效，但至少一项主要性能指标未同时满足多数 seed 改善和配对 CI 不跨 0 的严格标准。'
    lines.append(f'**结论 {choice}。** {reason} 稳定性判断同时依据配对 CI 和逐 seed 方向，并非只看均值。')
    lines += ['', '所有原始 per-seed 值、paired delta、bootstrap CI 与 win/loss/tie 分别见同目录 CSV；'
              '正式图见同目录 PNG。此前 Flat-vs-Pair 正式输出未被覆盖。','']
    path=OUT/'MASKED_PAIR_PPO_10SEED_REPORT.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(path)
    return choice

if __name__=='__main__':main()
