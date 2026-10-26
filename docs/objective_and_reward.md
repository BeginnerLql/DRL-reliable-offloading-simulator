# 问题、目标、奖励与实验指标

本实现研究在线固定双副本卸载：每个任务选择两个不同服务器，首次副本完成定义任务响应时间，另一个副本继续执行。空间相关瞬态故障可靠性只取决于任务执行需求和固定物理参数，不包含排队时间，也不采样实际故障事件。

任务 i 的相关安全集合为 A_i={a: R_corr(i,a)>=R_req(i)}。集合非空时，Masked PPO 在集合内选择；集合为空时使用固定最小时延回退。可靠性约束只要求在存在可行解时满足。无解任务继续实际执行并影响队列，仍进入全部指标的分母。

相对期限为 D_i，响应时间为 T_i。原始时延评分 Q_i=clip((b-T_i/D_i)/(b-1),0,1)，当前 b=2。有效效用 u_i=1[R_corr(i,a_i)>=R_req(i)]*Q_i。目标为最大化 E[sum_i u_i/N]，N 是固定到达任务数。它不等同于最小化平均秒数、P95 或最大化严格按时率。

## 训练奖励

PPO 按有效策略决策排列 transition。每次下一次策略决策前收集已经完成的任务，其 u_i 累加到上一个决策区间；中间的回退不创建策略动作或 log_prob。末个区间包括末次到达后的首副本完成排空。并列时间按 SimPy 实际事件处理顺序决定哪个区间已经观察到完成事件。

因此每个任务效用只计一次，包括由早先决策派发、在后续决策区间完成的任务。gamma=1 时，sum_k reward_k=sum_i u_i=N*mean_effective_qos。仍记录 delta_t，可显式改变 gamma 做折扣消融；gamma<1 时不再与等权平均效用目标完全一致。GAE 和 PPO 的价值函数近似仍会影响估计误差，目标相同不保证最优或超过基线。

记录字段：task_qos 是原始时延评分；task_utility 是 u_i；ppo_reward_owner_task_id 标识完成事件被哪个策略决策区间收集；ppo_interval_reward 是该决策区间的总奖励，不是该决策所派发任务自己的 QoS。Episode Reward 为所有任务有效效用总和。评估时这些训练归属字段为空。

## 实验指标与可行性

主指标 mean_qos=sum_i u_i/N；原始平均评分另记 mean_raw_qos。可靠性达标率、全部任务平均时延/P95、按时率、副本资源消耗为辅助指标。可靠性判定采用统一相关模型，包括可靠性可行但因其他原因回退的基线任务。不能仅根据 is_fallback 把评分置零。RSR 是理论要求满足率，不是经验成功率。

新增辅助指标 mean_satisfied_reliability：只对 requirement_satisfied=True 的任务，平均其实际所选方案的 pair_reliability。该兼容字段在双副本下为统一相关模型可靠性，在单副本下为相同模型的节点边际可靠性；不使用独立模型选择评分或任务要求阈值代替实际可靠性。包含可靠性达标的回退任务，不额外要求按时完成。无达标任务时记 NaN，报告显示“无样本”。保留 satisfied_task_count 和 satisfied_reliability_sum，跨回合和种子汇总时按达标任务数加权；配对区间只使用双方都有达标样本的种子。该条件均值应与可靠性达标率一同解读，不改变奖励或主目标。

训练前运行只读检查：

```bash
python tools/audit_objective_alignment.py
python tools/audit_objective_alignment.py --requirements 0.9 0.99 0.999 0.9999
```

输出每个训练/评估场景的无安全动作、单安全动作、多安全动作任务数，不改工作簿。全部无解用于失效边界实验，不能作为 PPO 学习性能实验；全部只有一个动作也没有策略选择空间。正式主入口对全部无解的 PPO 训练提前报错，直接 MainLoop 仍支持全回退仿真供边界验证。

目前持久化任务全部要求 0.9999，默认故障率缩放 1 时全部无解。需要根据有依据的物理参数或服务等级选定实验场景，然后同步生成训练/评估输入再训练；这里没有自动降低要求。比较时至少同时包含共享相关安全集合的 SML，用来区分可靠性模型优势与 PPO 调度优势。

本轮改动不扩展固定双副本动作空间，不重写历史结果。历史检查点可供冻结推理，但不代表经过当前完成区间奖励和 gamma=1 训练；新的性能结论需要独立重训和配对评估。

## 正式对比入口

`tools/run_aligned_experiments.py` 使用相同的生产训练与评估函数，默认固定以下探索性场景。它们根据可行性覆盖范围定义，没有根据方法胜负筛选，也不声称故障率已经过文献标定。缩放系数乘基础故障率，与空间风险 beta 不同。

|场景|基础故障率缩放|任务可靠性要求|作用|
|---|---:|---:|---|
|low_requirement|0.5|0.9|较低要求，观察复制开销与调度|
|partial_feasibility|0.5|0.99|部分任务无解、单动作和多动作并存|
|tight_feasible|0.1|0.999|要求较紧，但全部任务存在选择空间|
|high_requirement|0.05|0.9999|高要求下的混合可行性|
|infeasible_boundary|0.1|0.9999|全回退边界；无PPO训练和推理|

所有场景默认到达率为 2 tasks/s，空间风险和期限沿用当前配置。每场景在内存中同时覆盖训练和评估阈值，包括 RASOUL 的预热任务；不写入原始 Excel。每个可训练场景使用 10 个独立模型各训练 300 轮，最终检查点冻结评估 20 轮；RASOUL 每种子预热 100 轮，每个评估回合从其预热历史重新开始。每轮所有方法使用完全相同的到达序列，训练、预热、评估采用不同随机流。

运行预检（无需 GPU）：

```bash
python tools/run_aligned_experiments.py --preflight-only --output-dir results/aligned_objective_preflight_new
```

GPU 可用后运行正式比较（输出目录必须尚不存在）：

```bash
python tools/run_aligned_experiments.py --output-dir results/aligned_objective_formal_20261026
```

单轮 CPU 集成检查需显式选择 CPU，不能用于替代正式多种子实验：

```bash
python tools/run_aligned_experiments.py --device cpu --scenarios tight_feasible infeasible_boundary --seeds 1 --train-episodes 2 --eval-episodes 2 --rasoul-warmup-episodes 2 --output-dir results/aligned_objective_smoke_new
```

输出包含源码和工作簿快照、输入输出哈希、每个场景的有效任务与可靠性数组、每种子的训练曲线/检查点/逐任务记录、服务器资源、评估指标、配对差和报告。每个种子的 P95 从该种子所有评估任务汇集计算；95% t 区间的采样单位是独立种子，单种子时区间留空，多指标区间未校正多重比较。
