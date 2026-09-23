"""Render the instrumented Pair PPO mechanism diagnosis from saved metrics."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'diagnostics/results/policy_oracle_alignment'

def markdown(frame):
    frame=frame.reset_index() if frame.index.name is not None else frame
    columns=list(frame.columns)
    rows=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for values in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in values)+' |')
    return '\n'.join(rows)

def run(directory:Path):
    aligned=pd.read_csv(directory/'actor_oracle_alignment_by_requirement.csv').set_index('R_req')
    advantages=pd.read_csv(directory/'advantage_alignment_by_requirement.csv')
    signs=pd.read_csv(directory/'advantage_sign_action_quality.csv')
    visits=pd.read_csv(directory/'pair_visitation_exploration.csv').set_index('action_index')
    trajectory=pd.read_csv(directory/'pair_78_training_trajectory.csv')
    untrained=pd.read_csv(directory/'untrained_actor_top1_frequency.csv').set_index('action_index')
    surface=pd.read_csv(directory/'rho_sensitivity_by_requirement.csv').set_index('R_req')
    mode=pd.read_csv(directory/'frozen_policy_mode_comparison.csv')
    action=pd.read_csv(directory/'evaluation_action_vectors.csv')
    train=pd.read_csv(directory/'training_advantage_oracle_regret.csv')
    sample=mode[(mode.policy_mode=='sampled')&(mode.R_req=='all')]
    greedy=mode[(mode.policy_mode=='greedy')&(mode.R_req=='all')].iloc[0]
    sampled_high=mode[(mode.policy_mode=='sampled')&(mode.R_req=='0.9999')]
    greedy_high=mode[(mode.policy_mode=='greedy')&(mode.R_req=='0.9999')].iloc[0]
    visit78=visits.loc[27]
    start=trajectory.head(20).mean(numeric_only=True);end=trajectory.tail(20).mean(numeric_only=True)
    init78=untrained.loc[27]
    init_rank=int(untrained.greedy_top1_fraction.rank(ascending=False,method='min').loc[27])
    row=[]
    for req in (0.9,0.99,0.999,0.9999):
        g=aligned.loc[req]
        norm=advantages[(np.isclose(advantages.R_req,req))&(advantages.advantage_type=='normalized_advantage')].iloc[0]
        row.append({'R_req':req,'Spearman':g.spearman_logit_reward_mean,
            'Kendall':g.kendall_logit_reward_mean,'Top1_hit':g.top1_hit_mean,
            'Top3_hit':g.top3_hit_mean,'Top5_hit':g.top5_hit_mean,
            'M_star':g.optimal_set_probability_mass_mean,'Uniform_mass':g.uniform_optimal_set_mass_mean,
            'M_star_lift':g.optimal_mass_lift_over_uniform_mean,
            'Mean_greedy_regret':g.greedy_regret_mean,
            'Corr_normA_negRegret':norm.spearman_with_negative_regret})
    compact=pd.DataFrame(row)
    mass=aligned[["optimal_set_probability_mass_mean","optimal_set_probability_mass_median",
        "optimal_set_probability_mass_p25","optimal_set_probability_mass_p75",
        "optimal_set_probability_mass_p90","optimal_set_probability_mass_ci_low",
        "optimal_set_probability_mass_ci_high"]].reset_index()
    mass.columns=["R_req","mean","median","P25","P75","P90","CI_low","CI_high"]
    compact.to_csv(directory/'alignment_key_metrics.csv',index=False)
    by_sign=signs[(signs.advantage_type=='normalized_advantage')&np.isclose(signs.R_req,.9999)].set_index('sign')
    high=aligned.loc[.9999];low=aligned.loc[.9]
    lines=['# Policy–Oracle Alignment Mechanism Diagnosis','',
        '本报告仅分析正式 seed plan 的 trial 0：300 个训练 episode × 200 task，20 个冻结评估 episode × 200 task。诊断子类调用原 PPO 的采样与更新；2-episode 对照的 Actor/Critic 权重逐张量相同，完整 300-episode 训练曲线及 4,000 条评估动作、延迟、奖励与正式 trial 0 完全一致。以下 bootstrap CI 为单次 run 内 state 重采样的描述性区间，不代表跨 seed 推断。','',
        '以下数值数据、checkpoint 与图均存放在 diagnostics/results/policy_oracle_alignment/。','',
        '## A. Actor 是否知道哪些 pair 真正高 reward？','',
        '在 1,000 个按四档分层选取的固定 evaluation state 上，对全部 28 个合法 pair 重放即时 task reward。每档 rank/Top-K/最优集合概率质量：','',markdown(compact),'',
        f'Actor 对宽泛排序有信息：Spearman 从 R=.9 的 {low.spearman_logit_reward_mean:.3f} 到 R=.9999 的 {high.spearman_logit_reward_mean:.3f}；但高档 Top-1 命中仅 {high.top1_hit_mean:.1%}。M_star 从 {low.optimal_set_probability_mass_mean:.3f} 降至 {high.optimal_set_probability_mass_mean:.3f}。部分下降来自最优集合缩小（均匀策略基线 {low.uniform_optimal_set_mass_mean:.3f}→{high.uniform_optimal_set_mass_mean:.3f}），但高于均匀的质量提升也从 {low.optimal_mass_lift_over_uniform_mean:.3f} 降至 {high.optimal_mass_lift_over_uniform_mean:.3f}。因此并非 Actor 完全不了解 reward 排序，而是对最高档最优动作的概率集中不足。完整均值/中位数/P25/P75/P90/bootstrap CI 见 actor_oracle_alignment_by_requirement.csv。',
        'M_star 的完整分布（CI 为 state 重采样区间）：','',markdown(mass),'',
        'probability-vs-regret 与 logit-vs-reward 的 Spearman 在同一 state 中相同，因为 softmax 保序。逐 pair 数值与分箱图见 evaluation_action_vectors.csv / probability_vs_regret.png。','',
        '## B. PPO advantage 是否给正确动作正确 credit？','',
        '4,800 个按需求平衡抽取的训练状态重放所选动作的即时 Oracle regret。raw/normalized advantage 与负 regret 的 Pearson/Spearman 见 advantage_alignment_by_requirement.csv。真正进入 PPO 更新的是每个 episode 标准化后的 advantage。','',
        markdown(advantages[advantages.advantage_type=='normalized_advantage'][['R_req','N','positive_fraction','oracle_regret_mean','pearson_with_negative_regret','spearman_with_negative_regret']]),'',
        f'最高档 normalized A 的 Spearman={compact.loc[compact.R_req.eq(.9999),"Corr_normA_negRegret"].iloc[0]:.3f}，不是零；A>0 组的平均 regret {by_sign.loc["positive","mean_oracle_regret"]:.2f}，A<0 组 {by_sign.loc["negative","mean_oracle_regret"]:.2f}，最优集合命中分别 {by_sign.loc["positive","optimal_set_hit_rate"]:.1%} / {by_sign.loc["negative","optimal_set_hit_rate"]:.1%}。因此高档 credit 信号存在，但并不足以保证最终贪心动作最优。R=.99 的相关性接近零且略负；原始 A 大多为正，必须以 normalized A 判断实际更新方向。即时 Oracle regret 与长期 SMDP return 不同，相关性不能独自证明 PPO 公式错误。','',
        '## C. 为什么 (7,8) 成为 policy attractor？','',
        f'100 个独立初始化的未训练 Actor × 同一 1,000 个 evaluation state 中，(7,8) Top-1 频率 {init78.greedy_top1_fraction:.2%}（28 对中排第 {init_rank}，均匀参考 3.57%）；它有轻微输入特征偏好，但不是结构上必定胜出的索引。Actor 对 28 对共享同一个 scorer，动作索引是 combinations(1..8,2) 的固定顺序；没有额外 mask，也没有 action-specific 输出权重。',
        f'训练前 20 回合，(7,8) 平均采样率 {start.sampled_78_frequency:.1%}、贪心率 {start.greedy_78_frequency:.1%}、平均概率 {start.mean_78_probability:.3f}；最后 20 回合分别为 {end.sampled_78_frequency:.1%}、{end.greedy_78_frequency:.1%}、{end.mean_78_probability:.3f}。全程采样 {int(visit78.sampled_count)}/60,000，记录的 4,800 个训练 Oracle 子样本里，当它被采样时平均 regret {visit78.mean_oracle_regret_replayed_subset:.2f}（median 0），这是在训练随机策略所形成的队列状态中的条件结果。此时的正优势更新可以强化它，但时间轨迹本身不证明单向因果。',
        f'冻结权重、相同到达/空间 seed 下，贪心评估把 {greedy.pair_78_fraction:.2%} 任务派到 (7,8)，平均延迟 {greedy.mean_task_delay:.3f}s、reward {greedy.mean_task_reward:.2f}；三次独立采样式评估中 (7,8) 平均占 {sample.pair_78_fraction.mean():.2%}，延迟 {sample.mean_task_delay.mean():.3f}s、reward {sample.mean_task_reward.mean():.2f}。这是策略诱导的队列分布变化，说明训练时“偶尔使用 (7,8) 的好结果”不能外推为“几乎所有任务贪心使用 (7,8) 仍然好”。',
        f'代价也必须说明：最高档 R=.9999 的可靠性满足率在贪心评估中 {greedy_high.reliability_satisfaction_rate:.1%}，采样评估平均 {sampled_high.reliability_satisfaction_rate.mean():.1%}。采样虽改善当前 reward/延迟，却不能直接作为高可靠部署策略。','',
        '可能原因按证据排序：1) **策略模式/队列分布错位（SUPPORTED）**：同权重、同外生 seed 的采样/贪心评估产生大幅不同 pair 集中度、延迟、reward。2) **训练选择偏差与策略吸引（PARTIALLY SUPPORTED）**：被采样时 (7,8) 低 regret，随后其概率和贪心率升高；尚未隔离每次更新的因果贡献。3) **advantage credit 弱且需求相关（PARTIALLY SUPPORTED）**：相关性有限，.99 近零；最高档仍存在正确方向信号。4) **输入特征初始偏好（PARTIALLY SUPPORTED）**：未训练 (7,8) 有轻微偏好，但远非最优先索引。5) **纯 action-index/mask 错误（NOT SUPPORTED）**：共享 scorer、合法 pair 映射一致、无额外 mask。6) **环境中的 (7,8) universal optimum（NOT SUPPORTED）**：此前 Oracle regret 与本轮贪心评估均反驳。','',
        '## D. 高 R_req 为什么 regret 更大？','',
        f'上一轮 exact-optimal set 均值随需求从 7.00 缩到 3.47；本轮 M_star {low.optimal_set_probability_mass_mean:.3f}→{high.optimal_set_probability_mass_mean:.3f}、Top-1 命中 {low.top1_hit_mean:.1%}→{high.top1_hit_mean:.1%}，最高档 greedy regret 均值 {high.greedy_regret_mean:.2f}。最高档违约会跳到失败分支并有 log10 惩罚，使非最优动作的 reward gap 更大。Pair (7,8) 的全局偏好未按高需求与拥塞状态充分调整；这比“R_req 输入丢失”更符合观测。','',
        '## E. 是否存在 R_req × rho 的 Oracle interaction？','',
        markdown(surface),'',
        '在固定同一个 evaluation state、同一 pair (7,8)、同一组 episode 有效节点故障率的实验中，只改变 Actor 的 rho 特征；生产 task reward 使用固定有效故障率和两个副本的条件故障概率，因此条件 Oracle reward 对这个单一 rho 输入为常数。Actor score 对 rho 有非零响应，并随 R_req 略增。**这不等于“空间风险模型没有相关性”**：rho 控制跨节点 episode 风险场的联合分布。仅替换一个 pair 的 rho、同时固定已实现的节点故障率，不是一个完整的风险场反事实；不能据此认定 Actor 的 rho 偏好总体错误。二维数据与图见 rreq_rho_surface.csv、rreq_rho_score_heatmap.png、oracle_rreq_rho_reward_heatmap.png。','',
        '## 最值得修改的一个机制','',
        '优先解决**训练时随机动作分布与部署时几乎确定性贪心动作造成的队列反馈错位**。建议下一步单独设计并评估一个与部署一致、同时保持任务级可靠性约束的动作选择机制；先在冻结模型上比较不同选择温度/约束规则对延迟、reward、最高档满足率的共同影响，再决定是否改训练。此报告不修改 PPO、Actor 或 reward，除本轮一次同配置 instrumented run 外，没有额外训练模型。','']
    report=ROOT/'diagnostics/results/POLICY_ORACLE_ALIGNMENT.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    args=parser.parse_args();print(run(args.run_dir))
