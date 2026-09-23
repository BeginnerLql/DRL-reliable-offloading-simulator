# Reliability-Masked Frozen Pair PPO Selection

本实验复用已保存的 Pair PPO Actor/Critic checkpoint；四个独立仿真副本使用完全相同的 20×200 次到达间隔与 episode 空间风险场，Actor/Critic 权重在前后逐张量一致。Greedy 与上一轮正式 trial 0 的 4,000 条评估记录逐条一致；Native Sample 与上一轮 action seed 2026 的 4,000 条记录逐条一致。所有策略仅改变冻结 Actor 的动作选择规则。完整任务级日志、mask 前后 28 维概率、安全集及图均在 `diagnostics/results/reliability_masked_policy/`。

## 四策略总体结果

| policy | mean_reward | total_reward | mean_latency | p50_latency | p90_latency | p95_latency | rsr | pair_selection_hhi | pair_selection_entropy | unique_pair_count | pair_78_frequency | safe_set_empty_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 43.8843 | 175537.0069 | 3.3586 | 2.6456 | 6.5472 | 8.7776 | 0.9227 | 0.9975 | 0.0096 | 2 | 0.9988 | 0.0432 |
| native_sample | 53.6900 | 214760.1998 | 1.9837 | 1.9556 | 3.0843 | 3.5086 | 0.8450 | 0.1101 | 2.5529 | 28 | 0.1910 | 0.0432 |
| masked_sample | 60.7583 | 243033.0004 | 2.0216 | 1.9649 | 3.1656 | 3.6587 | 0.9567 | 0.1229 | 2.4468 | 28 | 0.2243 | 0.0432 |
| masked_greedy | 48.7687 | 195074.6927 | 3.0738 | 2.4952 | 5.8025 | 8.0304 | 0.9567 | 0.8528 | 0.4021 | 8 | 0.9227 | 0.0432 |

mean_reward 为平均每任务 reward，total_reward 为 4,000 任务 reward 总和；latency 单位秒；entropy 使用自然对数（nats），HHI 为实际 pair 选择频率平方和。`safe_set_empty_rate` 是在任何选择规则之前按当次 episode 有效故障率计算，因此四策略相同。

选择 pair 的风险属性（effective failure probability 为两个副本联合失败概率）：

| policy | selected_rho_mean | selected_effective_failure_probability_mean | selected_pair_reliability_mean | reliability_violation_rate |
|---|---|---|---|---|
| greedy | 0.1270 | 0.0001 | 0.9999 | 0.0772 |
| native_sample | 0.2341 | 0.0005 | 0.9995 | 0.1550 |
| masked_sample | 0.2307 | 0.0003 | 0.9997 | 0.0432 |
| masked_greedy | 0.1353 | 0.0001 | 0.9999 | 0.0432 |

各档需求结果：

| policy | R_req | rsr | mean_latency | mean_reward | safe_set_empty_rate | reliability_violation_rate |
|---|---|---|---|---|---|---|
| greedy | 0.9000 | 1.0000 | 3.4819 | 45.6721 | 0.0000 | 0.0000 |
| greedy | 0.9900 | 1.0000 | 3.0098 | 53.5717 | 0.0000 | 0.0000 |
| greedy | 0.9990 | 0.9890 | 3.1895 | 48.8516 | 0.0000 | 0.0110 |
| greedy | 0.9999 | 0.7020 | 3.7532 | 27.4416 | 0.1730 | 0.2980 |
| native_sample | 0.9000 | 1.0000 | 2.1024 | 60.5567 | 0.0000 | 0.0000 |
| native_sample | 0.9900 | 0.9970 | 1.8688 | 67.0470 | 0.0000 | 0.0030 |
| native_sample | 0.9990 | 0.9020 | 1.8933 | 60.5232 | 0.0000 | 0.0980 |
| native_sample | 0.9999 | 0.4810 | 2.0702 | 26.6334 | 0.1730 | 0.5190 |
| masked_sample | 0.9000 | 1.0000 | 2.1090 | 60.5450 | 0.0000 | 0.0000 |
| masked_sample | 0.9900 | 1.0000 | 1.9080 | 66.5752 | 0.0000 | 0.0000 |
| masked_sample | 0.9990 | 1.0000 | 1.8940 | 66.2948 | 0.0000 | 0.0000 |
| masked_sample | 0.9999 | 0.8270 | 2.1754 | 49.6180 | 0.1730 | 0.1730 |
| masked_greedy | 0.9000 | 1.0000 | 3.2244 | 48.7944 | 0.0000 | 0.0000 |
| masked_greedy | 0.9900 | 1.0000 | 2.8012 | 55.5384 | 0.0000 | 0.0000 |
| masked_greedy | 0.9990 | 1.0000 | 2.8975 | 53.1987 | 0.0000 | 0.0000 |
| masked_greedy | 0.9999 | 0.8270 | 3.3719 | 37.5431 | 0.1730 | 0.1730 |

## 最高可靠性需求 R_req=0.9999

| policy | rsr | mean_latency | mean_reward | pair_78_frequency | safe_set_empty_rate | selected_unsafe_rate |
|---|---|---|---|---|---|---|
| greedy | 0.7020 | 3.7532 | 27.4416 | 1.0000 | 0.1730 | 0.2980 |
| native_sample | 0.4810 | 2.0702 | 26.6334 | 0.1880 | 0.1730 | 0.5190 |
| masked_sample | 0.8270 | 2.1754 | 49.6180 | 0.2940 | 0.1730 | 0.1730 |
| masked_greedy | 0.8270 | 3.3719 | 37.5431 | 0.7200 | 0.1730 | 0.1730 |

在该档 1,000 个任务中，173 个（17.3%）没有任何满足要求的 pair；因此当前合法动作和风险场下，任意可靠性 mask 的满足率上限为 82.7%。空集时两种 masked 策略都明确回退到可靠性最高的合法 pair，并单独标记，不计作安全选择。Native Sample 有 519 个违约，其中 346 个发生在**存在安全 pair** 的情况下；Greedy 分别为 298 / 125。Masked Sample 的违约恰为 173 个空集任务，最高档满足率 82.7%。相对 Native Sample 的同 episode 配对提升 34.6%（20 episode bootstrap 95% CI 27.5% 至 42.0%）。

Masked Sample 相比 Native Sample 最高档平均延迟 2.070→2.175s，平均 reward 26.63→49.62；它仍明显优于 Greedy 的 3.753s 延迟，且 reward 更高。与 Masked Greedy 相比，两者最高档 RSR 同为 82.7%，但 Masked Sample 延迟 2.175s vs 3.372s、reward 49.62 vs 37.54。

## 队列与动作集中度

| policy | server_id | selection_count | utilization | mean_queue_length | p95_queue_length | mean_waiting_time | busy_time |
|---|---|---|---|---|---|---|---|
| greedy | 7 | 3995 | 0.7185 | 1.2437 | 6.0000 | 2.5473 | 5879.4444 |
| greedy | 8 | 4000 | 0.6475 | 0.8015 | 4.0000 | 1.6395 | 5298.0000 |
| native_sample | 7 | 2169 | 0.3922 | 0.1449 | 1.0000 | 0.5457 | 3204.0000 |
| native_sample | 8 | 1880 | 0.3074 | 0.0814 | 1.0000 | 0.3536 | 2510.9500 |
| masked_sample | 7 | 2256 | 0.4140 | 0.1797 | 1.0000 | 0.6508 | 3383.6111 |
| masked_sample | 8 | 2084 | 0.3418 | 0.1324 | 1.0000 | 0.5191 | 2793.0500 |
| masked_greedy | 7 | 3836 | 0.6817 | 1.0449 | 5.0000 | 2.2282 | 5576.5556 |
| masked_greedy | 8 | 3855 | 0.6172 | 0.7164 | 4.0000 | 1.5203 | 5049.2500 |

`mean_queue_length` 和 `p95_queue_length` 是从实际等待副本集合在每次入队、获得 CPU、完成时的状态变化按 SimPy 时间积分得到；`busy_time` 是实际运行副本的 CPU 占用时长，`utilization=busy_time/episode horizon` 跨 20 回合汇总；`mean_waiting_time` 是从进入 CPU 等待队列到获得 CPU 的真实时长。两种 greedy 选择都让服务器 7/8 长时间积压，而两种 sampling 明显分散到更多节点。

Masked Sample 与 Greedy 的任务级动作不同率 77.6%，与 Native Sample 的不同率 88.1%（两次随机抽样即使都可行，也未必选同一 pair）。Masked Sample 相比 Masked Greedy 总体 reward 的 episode 配对差 +11.99（95% CI +10.03 至 +14.00），latency 差 -1.052s（95% CI -1.331 至 -0.785）。这说明仅加 mask 不足以解决负载集中；保留分散选择是关键。

## 机制判断

- **H1 SUPPORTED**：Greedy 的 (7,8) 选择率 99.9%、HHI 0.998，服务器 7 的时间加权平均队列 1.244；Native Sample 对应 19.1%、0.110、0.145。
- **H2 SUPPORTED**：Native Sample 降低集中度与平均延迟，但最高档 RSR 70.2%→48.1%；在有可行 pair 的 827 个最高档任务中仍选了 346 个不安全动作。
- **H3 SUPPORTED（受空集限制）**：Masked Sample 保持 28 对均被选择，HHI 0.123，平均延迟 2.022s；最高档 RSR 恢复到可行性上限 82.7%。空集率 17.3% 是剩余违约的全部来源。

下一步**值得在独立里程碑中研究**可靠性 mask 与训练/部署选择方式的一致化；本轮证据不支持仅把 mask 加到贪心部署端，因为 Masked Greedy 仍高度集中。正式整合前还应检验高需求空集任务的处理方案及多 seed 稳定性。本轮没有重新训练，也没有修改 PPO、Actor、reward 或 simulator 生产逻辑。

复现命令（仓库根目录）：

```bash
/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python diagnostics/evaluate_reliability_masked_policy.py
/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python diagnostics/write_reliability_masked_report.py
/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python diagnostics/validate_reliability_masked_policy.py
```

20 个 episode 是同一 checkpoint/同一外生 trace 上的配对重复；bootstrap CI 按 episode 重采样，不能当作跨训练 seed 推断。完整配对差与 CI 见 `paired_episode_effects.csv`。
