# Reliability-Masked Pair PPO: Formal Paired 10-Seed Experiment

## 1. Experimental Setup

Flat PPO、Pair PPO 和 Reliability-Masked Pair PPO 分别按此前正式实验的相同 10 个 seed 训练。每个 seed：300 回合训练、每回合 200 任务；20 回合评估、每回合 200 任务。Flat/Pair 采用此前正式定义的 greedy 部署；Masked Pair 的正式部署是 masked stochastic。Masked greedy 仅使用**同一个 masked checkpoint**作 evaluation-time 消融。训练初始化、任务到达、空间风险与各算法配对；masked 动作抽样使用独立、固定的 Torch seed。本目录保存 `formal_seed_plan.csv` 和 `formal_reference_metadata.json` 快照，以便远端复现。脚本逐 seed 核验 Flat/Pair 的训练奖励曲线和评估动作、reward 与历史正式结果一致。

统计单位是训练 seed（n=10），以下所有标准差为 seed 间 sample std。配对差值的 95% CI 由 20,000 次 seed-level paired bootstrap 得到（seed=2043）。win/loss/tie 容差为 1e-10；latency、HHI 和负载越低越好。

## 2. Main Results

| 指标 | Flat PPO | Pair PPO | Masked Pair stochastic |
|---|---:|---:|---:|
| mean_reward | 44.6860 ± 2.5080 | 46.0569 ± 2.1147 | 61.4291 ± 0.6899 |
| mean_latency | 3.3823 ± 0.3625 | 3.2516 ± 0.2775 | 2.0107 ± 0.0102 |
| p50_latency | 2.7121 ± 0.1542 | 2.6500 ± 0.0850 | 1.9613 ± 0.0079 |
| p90_latency | 6.4667 ± 0.8792 | 6.1277 ± 0.7359 | 3.1402 ± 0.0244 |
| p95_latency | 8.2496 ± 1.3792 | 7.7571 ± 1.2696 | 3.6006 ± 0.0471 |
| overall_rsr | 0.9465 ± 0.0162 | 0.9537 ± 0.0162 | 0.9727 ± 0.0116 |
| highest_rsr | 0.7919 ± 0.0617 | 0.8185 ± 0.0621 | 0.8916 ± 0.0459 |

各可靠性需求档的 RSR（seed-level mean ± std）：

| R_req | Flat | Pair | Masked stochastic |
|---:|---:|---:|---:|
| 0.9 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 0.99 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 0.999 | 0.9941 ± 0.0087 | 0.9962 ± 0.0067 | 0.9993 ± 0.0022 |
| 0.9999 | 0.7919 ± 0.0617 | 0.8185 ± 0.0621 | 0.8916 ± 0.0459 |

## 3. Paired Statistical Results

Delta 均定义为 Masked stochastic − Pair greedy；表中 win/loss/tie 按指标改善方向计算。

| 指标 | mean delta | paired bootstrap 95% CI | win/loss/tie | CI 不跨 0 |
|---|---:|---:|---:|---|
| mean_reward | +15.3723 | [+14.3445, +16.2357] | 10/0/0 | 是 |
| mean_latency | -1.2409 | [-1.3927, -1.0738] | 10/0/0 | 是 |
| overall_rsr | +0.0191 | [+0.0144, +0.0240] | 10/0/0 | 是 |
| highest_rsr | +0.0731 | [+0.0554, +0.0917] | 10/0/0 | 是 |
| pair_selection_hhi | -0.9228 | [-0.9403, -0.8976] | 10/0/0 | 是 |
| maximum_mean_queue_length | -1.0631 | [-1.1795, -0.9329] | 10/0/0 | 是 |
| avoidable_violation_rate | -0.0190 | [-0.0240, -0.0144] | 10/0/0 | 是 |

## 4. Reliability Feasibility Analysis

Pair PPO 的 safe-set/AVR 数值是按其实际决策状态**事后重放**生产可靠性规则的诊断，不是旧 Pair PPO 的内部约束。

| 指标 | Pair post-hoc | Masked stochastic |
|---|---:|---:|
| feasibility_rate | 0.9727 ± 0.0116 | 0.9727 ± 0.0116 |
| conditional_rsr | 0.9804 ± 0.0086 | 1.0000 ± 0.0000 |
| overall_rsr | 0.9537 ± 0.0162 | 0.9727 ± 0.0116 |
| avoidable_violation_rate | 0.0190 ± 0.0082 | 0.0000 ± 0.0000 |
| unavoidable_violation_rate | 0.0273 ± 0.0116 | 0.0273 ± 0.0116 |
| empty_safe_set_rate | 0.0273 ± 0.0116 | 0.0273 ± 0.0116 |
| mean_empty_deficit | 0.0001 ± 0.0001 | 0.0001 ± 0.0001 |
| p95_empty_deficit | 0.0003 ± 0.0002 | 0.0003 ± 0.0002 |

最高需求档 R_req=0.9999：

| 指标 | Pair post-hoc | Masked stochastic |
|---|---:|---:|
| feasibility_rate | 0.8916 ± 0.0459 | 0.8916 ± 0.0459 |
| conditional_rsr | 0.9173 ± 0.0370 | 1.0000 ± 0.0000 |
| overall_rsr | 0.8185 ± 0.0621 | 0.8916 ± 0.0459 |
| avoidable_violation_rate | 0.0731 ± 0.0312 | 0.0000 ± 0.0000 |
| unavoidable_violation_rate | 0.1084 ± 0.0459 | 0.1084 ± 0.0459 |
| empty_safe_set_rate | 0.1084 ± 0.0459 | 0.1084 ± 0.0459 |
| mean_empty_deficit | 0.0001 ± 0.0001 | 0.0001 ± 0.0001 |
| p95_empty_deficit | 0.0003 ± 0.0002 | 0.0003 ± 0.0002 |

## 5. Deployment Ablation

Masked stochastic 与 masked greedy 在每个 seed 中复用同一训练 checkpoint、相同任务/环境流和相同 safe mask。Delta=stochastic−greedy。

| 指标 | mean delta | paired bootstrap 95% CI | win/loss/tie | CI 不跨 0 |
|---|---:|---:|---:|---|
| mean_reward | +10.7552 | [+8.2707, +12.6667] | 10/0/0 | 是 |
| mean_latency | -0.9031 | [-1.1264, -0.6508] | 10/0/0 | 是 |
| overall_rsr | +0.0000 | [+0.0000, +0.0000] | 0/0/10 | 否 |
| conditional_rsr | +0.0000 | [+0.0000, +0.0000] | 0/0/10 | 否 |
| pair_selection_hhi | -0.7764 | [-0.8422, -0.6953] | 10/0/0 | 是 |
| top1_pair_frequency | -0.7922 | [-0.8359, -0.7391] | 10/0/0 | 是 |
| maximum_server_selection_share | -0.2700 | [-0.2823, -0.2550] | 10/0/0 | 是 |
| mean_server_queue_length | -0.1540 | [-0.1826, -0.1214] | 10/0/0 | 是 |
| mean_server_p95_queue_length | -0.6250 | [-0.7625, -0.4750] | 10/0/0 | 是 |
| maximum_mean_queue_length | -0.8421 | [-0.9594, -0.7031] | 10/0/0 | 是 |
| maximum_p95_queue_length | -3.1000 | [-3.7000, -2.4000] | 10/0/0 | 是 |

## 6. Pair / Server Concentration

| 指标 | Pair PPO | Masked stochastic | Masked greedy |
|---|---:|---:|---:|
| pair_selection_hhi | 0.9783 ± 0.0361 | 0.0555 ± 0.0056 | 0.8319 ± 0.1242 |
| pair_selection_entropy | 0.0510 ± 0.0724 | 3.0813 ± 0.0608 | 0.3946 ± 0.2264 |
| top1_pair_frequency | 0.9887 ± 0.0192 | 0.1134 ± 0.0171 | 0.9056 ± 0.0783 |
| unique_selected_pairs | 1.8000 ± 0.4216 | 28.0000 ± 0.0000 | 6.9000 ± 0.9944 |
| maximum_server_selection_share | 0.5000 ± 0.0000 | 0.2147 ± 0.0115 | 0.4847 ± 0.0170 |
| maximum_server_utilization | 0.7134 ± 0.0183 | 0.3057 ± 0.0196 | 0.6770 ± 0.0212 |
| maximum_mean_queue_length | 1.1475 ± 0.2175 | 0.0844 ± 0.0136 | 0.9265 ± 0.2118 |
| maximum_p95_queue_length | 5.1000 ± 0.9944 | 1.0000 ± 0.0000 | 4.1000 ± 1.1005 |
| mean_server_waiting_time | 0.5120 ± 0.0782 | 0.3005 ± 0.0131 | 0.4808 ± 0.1238 |
| pair_78_frequency | 0.9887 ± 0.0192 | 0.1131 ± 0.0176 | 0.9056 ± 0.0783 |

服务器逐 seed 的 selection count、utilization、真实等待队列长度均值/P95、平均等待时间见 `server_load_results.csv`；动作分布见 `pair_selection_results.csv`。

## 7. Seed Stability

- mean_reward: mean delta +15.3723; seed delta 范围 [+11.7630, +17.2887]; win/loss/tie 10/0/0; CI 不跨 0。
- mean_latency: mean delta -1.2409; seed delta 范围 [-1.6498, -0.6922]; win/loss/tie 10/0/0; CI 不跨 0。
- overall_rsr: mean delta +0.0191; seed delta 范围 [+0.0067, +0.0340]; win/loss/tie 10/0/0; CI 不跨 0。
- highest_rsr: mean delta +0.0731; seed delta 范围 [+0.0270, +0.1250]; win/loss/tie 10/0/0; CI 不跨 0。
- pair_selection_hhi: mean delta -0.9228; seed delta 范围 [-0.9498, -0.8256]; win/loss/tie 10/0/0; CI 不跨 0。

训练稳定性：

| 指标 | 10-seed mean ± std |
|---|---:|
| first30_masked_entropy | 2.8574 ± 0.0252 |
| last30_masked_entropy | 2.7926 ± 0.0619 |
| first30_pair_hhi | 0.0532 ± 0.0018 |
| last30_pair_hhi | 0.0652 ± 0.0055 |
| final30_top1_frequency | 0.1414 ± 0.0156 |
| mean_safe_set_size | 21.7339 ± 0.0757 |
| empty_safe_set_rate | 0.0226 ± 0.0025 |
| ratio_min | 0.9809 ± 0.0045 |
| ratio_max | 1.0147 ± 0.0033 |
| max_first_ratio_error | 0.0000 ± 0.0000 |

独立 sanity checks：10 个 seed×4 种部署均有 4000 条评估记录；每个 Masked seed 有 300 次有限 PPO 更新；有效 mask 支持、最大可靠性 fallback、同 checkpoint/同 safe mask 均通过；可避免违约总数 0。

## 8. Failure Cases

Masked 约束异常 seed：无（所有 10 seed 可避免违约为 0）。
最高需求档最低 RSR 为 trial 0 的 0.8270；该 seed 可行率 0.8270、空安全集率 0.1730。
具体空安全集 deficit 与各 seed 的最高需求档可行上限见 `requirement_level_results.csv`。

## 9. Conclusion

**结论 A。** 可靠性、reward、latency 与最大服务器平均队列均在多数 seed 同向改善，且各配对 CI 不跨 0。 稳定性判断同时依据配对 CI 和逐 seed 方向，并非只看均值。

所有原始 per-seed 值、paired delta、bootstrap CI 与 win/loss/tie 分别见同目录 CSV；正式图见同目录 PNG。此前 Flat-vs-Pair 正式输出未被覆盖。
