# Reliability-Masked Pair PPO Integration

独立新算法采用 Pair Actor 与原 Critic、reward、SMDP-GAE/PPO 超参数；仅将状态对应的有效动作 mask 贯穿采样和 PPO 更新。单 seed、100×200 训练任务；20×200 冻结评估任务。旧 Pair PPO 与新 Agent 从逐张量相同的 Actor/Critic 初始化开始，训练使用相同到达与空间风险流，评估也共享外生流。此处不是正式 10-seed 结论。

## A. 实现改动

`agents/masked_pair_ppo_agent.py` 提供独立 Agent；`config/masked_pair_ppo.py` 保持新算法配置独立；`tools/run_masked_pair_ppo.py evaluate` 默认 **masked stochastic**，`--deployment greedy` 仅用于消融。`MainLoop` 只增加新 Agent 可选的当前任务上下文 hook；旧 Flat/Pair PPO 不实现该 hook。新 Agent 对每个合法 pair 调用生产 `Task.initialize_reliability_evaluation()`，不新增可靠性公式或 rho 阈值。

## B. PPO 数学一致性

安全集非空时 effective mask 为 safe mask；空集时为所有 max-reliability pair（数值容差 1e-12）。Actor 原始 logits 对 mask 外动作赋 `-inf`，由同一 `Categorical` 分布产生 action、old log probability、entropy。rollout 按 task ID 保存 effective mask；PPO minibatch 对应行将**相同 mask**施加于新 logits，再计算 new log probability、ratio 和仅安全支持上的 entropy。Critic、GAE、reward 与 PPO loss 系数保持原样。

100 次训练更新均有限；每回合首次 minibatch 的 ratio 相对 1 最大误差 0；全部 minibatch ratio 范围 [0.9916, 1.0091]。固定 task/requirement/episode hazard 样本的 safe mask 对照 1000 条，mismatch **0**；从磁盘独立加载 checkpoint 后默认 stochastic 部署的全部 4000 条评估任务与原评估逐条相同。

## C. 单 seed smoke 结果

| policy | mean_reward | mean_latency | overall_rsr | highest_rsr | conditional_rsr | feasibility_rate | empty_safe_set_rate | avoidable_violation_count | unavoidable_violation_count | pair_78_frequency | pair_selection_hhi | pair_selection_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_pair_greedy | 40.7204 | 3.3656 | 0.8748 | 0.5270 | 0.9143 | 0.9567 | 0.0432 | 328 | 173 | 0.0452 | 0.9055 | 0.2115 |
| masked_pair_stochastic | 59.5345 | 2.0416 | 0.9567 | 0.8270 | 1.0000 | 0.9567 | 0.0432 | 0 | 173 | 0.0780 | 0.0440 | 3.2276 |
| masked_pair_greedy | 49.1681 | 2.9405 | 0.9567 | 0.8270 | 1.0000 | 0.9567 | 0.0432 | 0 | 173 | 0.0648 | 0.6975 | 0.7232 |

训练奖励前 10 回合均值：旧 9047.26、新 12170.38；后 10 回合均值：旧 9934.39、新 12155.04。逐回合曲线见 `training_reward_comparison.png`。

## D. Safe-set 与队列行为

新算法训练集的 mean safe-set size 21.66，feasibility rate 97.9%，conditional RSR 100.0%；可避免违约总数 0。评估时默认 masked stochastic 的 feasibility rate 95.7%、conditional RSR 100.0%、overall RSR 95.7%。

| policy | server_id | selection_count | utilization | mean_queue_length | p95_queue_length | mean_waiting_time |
|---|---|---|---|---|---|---|
| legacy_pair_greedy | 7 | 198 | 0.0354 | 0.0388 | 0.0000 | 1.6073 |
| legacy_pair_greedy | 8 | 3983 | 0.6440 | 0.7953 | 4.0000 | 1.6361 |
| masked_pair_stochastic | 7 | 1315 | 0.2436 | 0.0493 | 0.0000 | 0.3062 |
| masked_pair_stochastic | 8 | 1495 | 0.2578 | 0.0565 | 0.0000 | 0.3088 |
| masked_pair_greedy | 7 | 556 | 0.1163 | 0.0573 | 0.0000 | 0.8432 |
| masked_pair_greedy | 8 | 3703 | 0.5935 | 0.6571 | 4.0000 | 1.4515 |

(7,8) 选择率：旧 Greedy 4.5%，新 masked stochastic 7.8%，masked greedy 6.5%；HHI 分别 0.906/0.044/0.697。服务器 7/8 的真实等待队列均值、P95 来自入队、获 CPU、完成事件的时间加权统计，不是 backlog proxy。

## E. Empty-safe-set 与 policy/system 分界

训练阶段有 418 次空安全集（2.1%）；这些状态全部使用 max-reliability fallback，平均 deficit 0.000103531、最大 deficit 0.000721743。并列最可靠动作保留 Actor 概率重新归一化抽样。评估时的 `avoidable_violation_count` 区分有可行 pair 却选择违约的策略失败；`unavoidable_violation_count` 是空安全集的系统不可行。`conditional_rsr` 只以非空安全集任务为分母，不能与 overall RSR 混用。

## F. Legacy baseline 完整性

旧 PPOAgent、Pair Actor、Flat PPO 的采样/更新函数未修改。新算法为独立类和入口。旧 Pair PPO 的训练前 100 回合 episode reward 与已保存的正式 300 回合 trial 0 前缀逐值相同；初始化权重逐张量相同，外生流一致。旧评估端仍使用原 greedy 行为；新入口明确默认 stochastic，未调用旧的 greedy-only 冻结评估器。

## G. 是否进入正式 10-seed

**具备进入正式 10-seed 比较的工程条件。** 本结论只依据此单 seed smoke 的数值稳定性、mask 一致性、可避免违约、动作集中度和旧版回归；正式论文结论仍需多 seed 评估。

运行命令（仓库根目录）：

```bash
/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python tools/run_masked_pair_ppo.py train-smoke --train-episodes 100 --eval-episodes 20
/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python tools/run_masked_pair_ppo.py evaluate --checkpoint-dir diagnostics/results/masked_pair_ppo_smoke
/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python diagnostics/write_masked_pair_ppo_integration_report.py
```
