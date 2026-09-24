# Deployment 2×2 Ablation

## 1. Experimental Control

本轮没有训练模型。十个 Pair checkpoint 均从原始正式实验目录加载，Actor/Critic SHA256 与完成清单核对；Pair greedy 每个 seed 都在相同 arrival/spatial seed 下重放，并逐任务复现已保存的 action、reward、latency 与可靠性。每个 seed 的 4,000 项 Pair stochastic 评估均与同 seed greedy 重放的到达间隔及完整空间风险轨迹逐项相同。

Pair stochastic 从原 Pair Actor 的完整 28-action `Categorical(logits)` 分布采样；没有 reliability mask、logit 修改或温度调整。动作采样使用私有、由 Trial ID 和独立标签派生的 CPU Torch generator。Pair 的 feasibility / AVR / conditional RSR 是在其访问状态上按生产可靠性规则事后诊断，不是 Pair policy 的限制。Pair 与 Masked 使用各自正式训练得到的 checkpoint；因此 B 是两个正式 stochastic policy 版本的成对比较，不能解释成对同一个 Actor 只切换 inference mask 的单因素实验。统计单位为训练 seed（n=10），所有 ± 为 seed 间 sample std。A/B 的 CI 使用 20,000 次 seed-level paired bootstrap（seed=2043）；C 直接复用此前正式 Masked deployment ablation 的 per-seed paired deltas 与 bootstrap CI。

## 2. 2×2 Main Table

| Policy | Reward | Mean latency | P95 latency | Overall RSR | Highest-tier RSR | Conditional RSR | AVR | UVR | Pair HHI | Top-1 pair | Max mean queue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pair Greedy | 46.0569 ± 2.1147 | 3.2516 ± 0.2775 | 7.7571 ± 1.2696 | 95.37% ± 1.62% | 81.85% ± 6.21% | 98.04% ± 0.86% | 1.90% ± 0.82% | 2.73% ± 1.16% | 0.9783 ± 0.0361 | 98.87% ± 1.92% | 1.1475 ± 0.2175 |
| Pair Stochastic | 57.0973 ± 1.4562 | 1.9942 ± 0.0183 | 3.5626 ± 0.0610 | 89.72% ± 2.25% | 63.33% ± 6.78% | 92.23% ± 1.48% | 7.55% ± 1.37% | 2.73% ± 1.16% | 0.1352 ± 0.0134 | 24.54% ± 2.80% | 0.1679 ± 0.0300 |
| Masked Greedy | 50.6739 ± 3.6442 | 2.9138 ± 0.4070 | 6.6945 ± 1.6053 | 97.27% ± 1.16% | 89.16% ± 4.59% | 100.00% ± 0.00% | 0.00% ± 0.00% | 2.73% ± 1.16% | 0.8319 ± 0.1242 | 90.56% ± 7.83% | 0.9265 ± 0.2118 |
| Masked Stochastic | 61.4291 ± 0.6899 | 2.0107 ± 0.0102 | 3.6006 ± 0.0471 | 97.27% ± 1.16% | 89.16% ± 4.59% | 100.00% ± 0.00% | 0.00% ± 0.00% | 2.73% ± 1.16% | 0.0555 ± 0.0056 | 11.34% ± 1.71% | 0.0844 ± 0.0136 |

Pair Greedy / Pair Stochastic 的 Conditional RSR 与 AVR 为 post-hoc diagnostic。AVR/UVR 按全部任务数作分母；Conditional RSR 的分母为 safe set 非空任务。可靠性比率以百分数显示。

## 3. Sampling Effect Without Mask

Comparison A 定义为 Pair Stochastic − Pair Greedy。

| Metric | Mean delta [95% paired CI] | Win/Loss/Tie |
|---|---:|---:|
| Reward | +11.0405 [+10.0558, +11.9009] | 10/0/0 |
| Latency | -1.2575 [-1.4093, -1.0893] | 10/0/0 |
| Overall RSR | -0.0565 [-0.0620, -0.0518] | 0/10/0 |
| Highest-tier RSR | -0.1852 [-0.1958, -0.1752] | 0/10/0 |
| Pair HHI | -0.8431 [-0.8629, -0.8147] | 10/0/0 |
| Top-1 frequency | -0.7433 [-0.7646, -0.7206] | 10/0/0 |
| Max server selection share | -0.2093 [-0.2161, -0.2027] | 10/0/0 |
| Maximum mean queue | -0.9796 [-1.0971, -0.8491] | 10/0/0 |
| Maximum P95 queue | -4.1000 [-4.6000, -3.5000] | 10/0/0 |

Latency, pair concentration, server selection concentration, and queue metrics are favorable when their deltas are negative; other displayed deltas are favorable when positive. W/L/T counts use that metric's favorable direction and 1e-10 tie tolerance.

## 4. Mask Effect Under Stochastic Deployment

Comparison B 定义为 Masked Stochastic − Pair Stochastic；两者都采用 stochastic deployment。Pair 的 reliability values are evaluated post hoc using `Task.initialize_reliability_evaluation`; Masked outputs use the same production rule during action selection.

| Metric | Mean delta [95% paired CI] | Win/Loss/Tie |
|---|---:|---:|
| Overall RSR | +0.0755 [+0.0686, +0.0847] | 10/0/0 |
| Highest-tier RSR | +0.2583 [+0.2414, +0.2804] | 10/0/0 |
| Conditional RSR | +0.0777 [+0.0703, +0.0876] | 10/0/0 |
| AVR | -0.0755 [-0.0847, -0.0686] | 10/0/0 |
| UVR | +0.0000 [+0.0000, +0.0000] | 0/0/10 |
| Reward | +4.3318 [+3.8682, +4.9724] | 10/0/0 |
| Latency | +0.0165 [+0.0082, +0.0248] | 1/9/0 |
| Pair HHI | -0.0797 [-0.0895, -0.0691] | 10/0/0 |
| Maximum mean queue | -0.0836 [-0.1052, -0.0634] | 10/0/0 |

RSR、AVR、UVR 等比例指标的 paired delta 使用 [0,1] 比例单位；例如 `+0.01` 等于增加 1 个百分点。

## 5. Sampling Effect With Mask

Comparison C 定义为 Masked Stochastic − Masked Greedy。数值逐项复用已有正式报告中的 bootstrap 输出，不重跑 checkpoint 或重新抽样。

| Metric | Mean delta [95% paired CI] | Win/Loss/Tie |
|---|---:|---:|
| Reward | +10.7552 [+8.2707, +12.6667] | 10/0/0 |
| Latency | -0.9031 [-1.1264, -0.6508] | 10/0/0 |
| Overall RSR | +0.0000 [+0.0000, +0.0000] | 0/0/10 |
| Highest-tier RSR | +0.0000 [+0.0000, +0.0000] | 0/0/10 |
| Pair HHI | -0.7764 [-0.8422, -0.6953] | 10/0/0 |
| Top-1 frequency | -0.7922 [-0.8359, -0.7391] | 10/0/0 |
| Max server selection share | -0.2700 [-0.2823, -0.2550] | 10/0/0 |
| Maximum mean queue | -0.8421 [-0.9594, -0.7031] | 10/0/0 |
| Maximum P95 queue | -3.1000 [-3.7000, -2.4000] | 10/0/0 |
| AVR | +0.0000 [+0.0000, +0.0000] | 0/0/10 |
| Conditional RSR | +0.0000 [+0.0000, +0.0000] | 0/0/10 |

## 6. Reliability Decomposition

| Policy | feasibility_rate | conditional_rsr | overall_rsr | avoidable_violation_rate | unavoidable_violation_rate | empty_safe_set_rate |
|---|---:|---:|---:|---:|---:|---:|
| Pair Greedy | 97.27% ± 1.16% | 98.04% ± 0.86% | 95.37% ± 1.62% | 1.90% ± 0.82% | 2.73% ± 1.16% | 2.73% ± 1.16% |
| Pair Stochastic | 97.27% ± 1.16% | 92.23% ± 1.48% | 89.72% ± 2.25% | 7.55% ± 1.37% | 2.73% ± 1.16% | 2.73% ± 1.16% |
| Masked Greedy | 97.27% ± 1.16% | 100.00% ± 0.00% | 97.27% ± 1.16% | 0.00% ± 0.00% | 2.73% ± 1.16% | 2.73% ± 1.16% |
| Masked Stochastic | 97.27% ± 1.16% | 100.00% ± 0.00% | 97.27% ± 1.16% | 0.00% ± 0.00% | 2.73% ± 1.16% | 2.73% ± 1.16% |

Pair metrics in this table remain post hoc. The production reliability test is `execution_reliability >= reliability_requirement`, evaluated through `Task.initialize_reliability_evaluation`; safe-set feasibility is whether any of the 28 legal pairs passes it.

### Per-requirement RSR and feasibility

| R_req | Pair Greedy RSR | Pair Stochastic RSR | Masked Greedy RSR | Masked Stochastic RSR | Pair Stochastic feasible | Masked Stochastic feasible |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% |
| 0.99 | 100.00% ± 0.00% | 99.87% ± 0.16% | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% |
| 0.999 | 99.62% ± 0.67% | 95.68% ± 2.23% | 99.93% ± 0.22% | 99.93% ± 0.22% | 99.93% ± 0.22% | 99.93% ± 0.22% |
| 0.9999 | 81.85% ± 6.21% | 63.33% ± 6.78% | 89.16% ± 4.59% | 89.16% ± 4.59% | 89.16% ± 4.59% | 89.16% ± 4.59% |

Pair Stochastic 是否产生 avoidable violation 的结果可直接从 AVR / Conditional RSR 观察；Masked 的 avoidable violation 为零是 action mask 保证的属性，并经现有 formal sanity checks 核验。

## 7. Queue / Concentration Analysis

| Policy | Pair HHI | Pair entropy | Top-1 pair | Unique pairs | (7,8) frequency | Max server selection share | Max utilization | Max mean queue | Max P95 queue | Mean server wait (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pair Greedy | 0.9783 ± 0.0361 | 0.0510 ± 0.0724 | 98.87% ± 1.92% | 1.8000 ± 0.4216 | 0.9887 ± 0.0192 | 50.00% ± 0.00% | 0.7134 ± 0.0183 | 1.1475 ± 0.2175 | 5.1000 ± 0.9944 | 0.5120 ± 0.0782 |
| Pair Stochastic | 0.1352 ± 0.0134 | 2.3660 ± 0.0909 | 24.54% ± 2.80% | 27.9000 ± 0.3162 | 0.2454 ± 0.0280 | 29.07% ± 1.15% | 0.4095 ± 0.0228 | 0.1679 ± 0.0300 | 1.0000 ± 0.0000 | 0.2574 ± 0.0151 |
| Masked Greedy | 0.8319 ± 0.1242 | 0.3946 ± 0.2264 | 90.56% ± 7.83% | 6.9000 ± 0.9944 | 0.9056 ± 0.0783 | 48.47% ± 1.70% | 0.6770 ± 0.0212 | 0.9265 ± 0.2118 | 4.1000 ± 1.1005 | 0.4808 ± 0.1238 |
| Masked Stochastic | 0.0555 ± 0.0056 | 3.0813 ± 0.0608 | 11.34% ± 1.71% | 28.0000 ± 0.0000 | 0.1131 ± 0.0176 | 21.47% ± 1.15% | 0.3057 ± 0.0196 | 0.0844 ± 0.0136 | 1.0000 ± 0.0000 | 0.3005 ± 0.0131 |

服务器逐节点 selection count、utilization、mean/P95 queue、waiting time 见 `deployment_2x2_server_load_results.csv`。各 seed 的完整 Pair stochastic 原始任务、post-hoc mask、policy entropy/action probability 和 28 维概率向量也随本报告保存。

## 8. Seed Stability

| Comparison | Metric | Seed delta min…max | Beneficial W/L/T | Paired CI excludes zero |
|---|---|---:|---:|---|
| Pair Stochastic - Pair Greedy | mean_latency | [-1.6577, -0.6978] | 10/0/0 | yes |
| Pair Stochastic - Pair Greedy | overall_rsr | [-0.0755, -0.0468] | 0/10/0 | yes |
| Pair Stochastic - Pair Greedy | highest_rsr | [-0.2150, -0.1610] | 0/10/0 | yes |
| Pair Stochastic - Pair Greedy | avoidable_violation_rate | [+0.0467, +0.0755] | 0/10/0 | yes |
| Pair Stochastic - Pair Greedy | pair_selection_hhi | [-0.8844, -0.7315] | 10/0/0 | yes |
| Pair Stochastic - Pair Greedy | maximum_mean_queue_length | [-1.3002, -0.5795] | 10/0/0 | yes |
| Masked Stochastic - Pair Stochastic | mean_latency | [-0.0049, +0.0347] | 1/9/0 | yes |
| Masked Stochastic - Pair Stochastic | overall_rsr | [+0.0618, +0.1095] | 10/0/0 | yes |
| Masked Stochastic - Pair Stochastic | highest_rsr | [+0.2220, +0.3400] | 10/0/0 | yes |
| Masked Stochastic - Pair Stochastic | avoidable_violation_rate | [-0.1095, -0.0617] | 10/0/0 | yes |
| Masked Stochastic - Pair Stochastic | pair_selection_hhi | [-0.1068, -0.0459] | 10/0/0 | yes |
| Masked Stochastic - Pair Stochastic | maximum_mean_queue_length | [-0.1369, -0.0404] | 10/0/0 | yes |
| Masked Stochastic - Masked Greedy | mean_latency | [-1.4354, -0.2263] | 10/0/0 | yes |
| Masked Stochastic - Masked Greedy | overall_rsr | [+0.0000, +0.0000] | 0/0/10 | no |
| Masked Stochastic - Masked Greedy | highest_rsr | [+0.0000, +0.0000] | 0/0/10 | no |
| Masked Stochastic - Masked Greedy | avoidable_violation_rate | [+0.0000, +0.0000] | 0/0/10 | no |
| Masked Stochastic - Masked Greedy | pair_selection_hhi | [-0.9021, -0.5204] | 10/0/0 | yes |
| Masked Stochastic - Masked Greedy | maximum_mean_queue_length | [-1.1412, -0.3653] | 10/0/0 | yes |

## 9. Mechanism Conclusion

**Q1 — Pair PPO 由 greedy 改为 stochastic 后是否降低 concentration、queue 和 latency？** 是，三项 paired 95% CI 均完全低于 0。 Pair Stochastic − Pair Greedy：latency -1.2575 [-1.4093, -1.0893]；HHI -0.8431 [-0.8629, -0.8147]；max mean queue -0.9796 [-1.0971, -0.8491]。

**Q2 — 没有 reliability mask 的 Pair stochastic 是否产生更多 avoidable violations？** 是，Masked stochastic 相对 Pair stochastic 的 AVR 显著降低且 Conditional RSR 提高。 Pair Stochastic 的 AVR 为 7.55% ± 1.37%；Masked Stochastic 为 0.00% ± 0.00%。

**Q3 — stochastic 部署下 mask 是否提高 RSR / highest-RSR / Conditional RSR？** 是，三项 paired 95% CI 均完全高于 0。 Overall RSR +0.0755 [+0.0686, +0.0847]；highest-tier RSR +0.2583 [+0.2414, +0.2804]；Conditional RSR +0.0777 [+0.0703, +0.0876]；AVR -0.0755 [-0.0847, -0.0686]。

这些结果支持机制解释：stochastic selection 主要缓解 pair/server concentration 与 queueing；reliability masking 消除可避免的可靠性违约；两者结合带来最终表现提升。

### Sampling × mask interaction（描述性）

| Metric | Sampling delta without mask (Pair stochastic − Pair greedy) | Sampling delta with mask (Masked stochastic − Masked greedy) | With-mask minus no-mask delta |
|---|---:|---:|---:|
| Mean latency | -1.2575 | -0.9031 | +0.3544 |
| Pair HHI | -0.8431 | -0.7764 | +0.0668 |
| Top-1 frequency | -0.7433 | -0.7922 | -0.0489 |
| Max mean queue | -0.9796 | -0.8421 | +0.1375 |

Interaction 数值仅作描述，不进行 two-way ANOVA。
