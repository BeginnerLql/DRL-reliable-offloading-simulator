# Pair-Scoring PPO：可靠性需求条件化诊断

## 方法与可验证范围

正式实验的 10 个 seed × 20 个评估 episode × 每 episode 5 个 task，合计 1000 个固定决策上下文。每个上下文穷举 28 个合法无序异节点 pair × 四档 R_req（112,000 条反事实）。其他任务的到达和动作保持已记录值，仅改变目标任务的 pair 和 R_req。
Oracle 用无副作用的 FIFO CPU / 固定上行轨迹重建目标任务的首副本完成时间；episode 有效故障率按正式 seed 重现；可靠性和奖励分别调用生产代码 Task.initialize_reliability_evaluation 和 MainLoop.calcReward。它是**目标任务即时 reward** Oracle，不包含对后续任务队列及长期 PPO return 的反事实影响。
逐条重放全部 40,000 条正式 Pair 评估记录：最大延迟误差 1.7e-13 秒，有效故障率误差 9.97e-17，reward 误差 7.11e-14。各档除 R_req 外的 context SHA256、延迟及 pair reliability 完全相同。当前没有 deadline、额外 action mask、直接 rho 或能耗 reward 项。

## 1. Oracle 是否要求随 R_req 改变 pair？

按 action-index 打破并列时，.9→.9999 的 argmax 改变率为 **80.9%**；但两端最优集合完全不相交的状态只有 **7.1%**。因此判定 **A：真正迫使动作改变的需求边界稀少**。单看 80.9% 会把并列最优的索引切换误当成强监督。
相邻档 argmax / 最优集合不相交比例：0.9->0.99: 0.1% / 0.0%; 0.99->0.999: 21.9% / 0.0%; 0.999->0.9999: 80.2% / 7.1%。
四档唯一 argmax 数 mean=2.022、median=2；分布 {'1': 191, '2': 597, '3': 211, '4': 1}。存在多个并列最优的状态比例依次为 [1.0, 1.0, 0.998, 0.778]。
Reward margin（best−second）如下；前三档几乎都是 0，最高档 median 仍为 0：

| R_req | Mean | Median | P25 | P75 | P90 |
|---|---|---|---|---|---|
| 0.9 | 0 | 0 | 0 | 0 | 0 |
| 0.99 | 0 | 0 | 0 | 0 | 0 |
| 0.999 | 0.012228 | 0 | 0 | 0 | 0 |
| 0.9999 | 4.2497 | 0 | 0 | 0 | 7.4019 |

切换明细见 oracle_switch_matrix.csv；并列最优集合和每行 reward 见 oracle_pair_results.csv。

## 2. Pair PPO 是否利用 R_req？

正式实验持久化的 10×100 个 probe 显示，.9→.9999 的 greedy argmax 改变率 **0.6%**，TV_Rreq=0.02854，而 TV_rho(true→zero)=0.03518。两种 TV 的扰动尺度不同，数值比仅供描述；TV_rho 只约为 TV_Rreq 的 1.23 倍，不能断言分布层面对 R_req 完全无响应。Greedy 动作的确基本不变。
Oracle 强制切换 7.1% 与 policy greedy 切换 0.6% 之间仍有差距；按并列索引计算的 Oracle 80.9% 不能直接当成要求 policy 切换的比例。注意 Oracle 的 1,000 个 state 与已保存的 1,000 个 policy probe 不是逐状态匹配，只能做分布级比较。每 seed 的 TV/argmax 与跨 seed 的 mean、std、median、bootstrap CI 分别见 tv_requirement_summary.csv / tv_requirement_aggregate.csv。

| Metric | Num_Seeds | Mean | Std | Median | Bootstrap_95CI_Low | Bootstrap_95CI_High |
|---|---|---|---|---|---|---|
| TV_Rreq_09_09999 | 10 | 0.028537 | 0.012567 | 0.0278 | 0.02153 | 0.036469 |
| TV_Rho_True_Zero | 10 | 0.035182 | 0.012508 | 0.035218 | 0.027981 | 0.042613 |
| Argmax_09_09999 | 10 | 0.006 | 0.013499 | 0 | 0 | 0.014 |

正式运行器未保存训练后的 checkpoint、逐动作 logits/probability 或完整 probe state，因此无法补出四档相邻 TV、全部动作分数和更大样本的模型 sweep。policy_requirement_sweep.csv 是当时保存的 probe **摘要**，不是重载权重得到的完整分布。

## 3. (7,8) 是否为 universal shortcut？

(7,8) 的 rho=0.12671、平均 pair reliability=0.9999546、平均 latency=3.258s，在样本均值的三目标 Pareto front 上；这表示它没有被另一 pair 同时在 rho、延迟、可靠性严格支配，**不表示**它近乎总是 reward 最优。
它的严格单一 argmax 频率依次为 [0.0, 0.0, 0.0, 0.03]，包含于并列最优集合的频率依次为 [0.28, 0.28, 0.28, 0.255]；各档即时 regret 见下表。

| R_req | Mean_Regret | Median_Regret | P90_Regret | Fraction_Strictly_Suboptimal |
|---|---|---|---|---|
| 0.9 | 23.121 | 21.368 | 51.28 | 0.72 |
| 0.99 | 23.121 | 21.368 | 51.28 | 0.72 |
| 0.999 | 23.121 | 21.368 | 51.28 | 0.72 |
| 0.9999 | 26.089 | 21.726 | 64.667 | 0.745 |

因此环境中的 (7,8) universal optimum **不成立**，但 Actor 在 10/10 seed 都以它作为 Top-1，说明策略层的全局 pair 捷径 **有证据支持**。Pareto 比较基于样本均值，不能替代逐状态 reward 比较。

## 4. Reward 是否提供 requirement-specific signal？

当前 reward 是满足阈值时的延迟奖励、未满足时的失败分支调整，再减 log10 违约惩罚。rho 仅影响 episode 风险场的联合采样；给定各节点有效 hazard 后，任务级 joint failure 为两个条件故障概率的乘积，rho 不直接进入奖励。

正式评估轨迹分层分解（不同档是不同任务，不能当作固定状态的因果差）：

| R_req | reward_total | reward_delay_component | reward_reliability_gate_component | reward_reliability_penalty_component | reward_reliability_component | reward_correlation_component | reward_energy_component | reliability_violation | Latency | Task_Count | Violation_Rate | Mean_Absolute_Reliability_Component |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 46.475 | 46.475 | 0 | 0 | 0 | 0 | 0 | 0 | 3.3229 | 10000 | 0 | 0 |
| 0.99 | 53.275 | 53.275 | 0 | 0 | 0 | 0 | 0 | 0 | 2.9641 | 10000 | 0 | 0 |
| 0.999 | 48.688 | 48.864 | -0.17116 | -0.0047111 | -0.17587 | 0 | 0 | 0.00047111 | 3.2384 | 10000 | 0.0038 | 0.17587 |
| 0.9999 | 35.79 | 45.194 | -8.7686 | -0.63509 | -9.4037 | 0 | 0 | 0.063509 | 3.4811 | 10000 | 0.1815 | 9.4037 |

.9 档 reliability component 均值 0.000；.9999 档为 -9.404，同时其成功延迟奖励均值 45.194。低档无违约信号，高档约 18.1% 任务违约且存在显著的阈值分支信号。不能笼统说高档可靠性 reward 太弱。
固定同一 state、同一 pair，仅将 .9 改为 .9999 时，reward component 的绝对改变量如下（延迟分量不变）：

| Component | Matched_State_Pairs | Mean_Absolute_Change | Median_Absolute_Change | Fraction_Nonzero |
|---|---|---|---|---|
| reward_total | 28000 | 45.985 | 62.876 | 0.65036 |
| reward_delay_component | 28000 | 0 | 0 | 0 |
| reward_reliability_component | 28000 | 45.985 | 62.876 | 0.65036 |
| reward_reliability_gate_component | 28000 | 41.108 | 53.448 | 0.65032 |
| reward_reliability_penalty_component | 28000 | 4.8768 | 3.0875 | 0.65036 |

## 5. Pair Actor 是否学到 R_req × rho interaction？

**UNKNOWN。** 没有训练后权重，无法计算 raw/masked logits、R_req×rho score 热图或 ∂score/∂rho。已有 rho counterfactual TV 非零只证明分布对 rho 输入有响应，不能证明交互。r_req_input_trace.txt 证明四档规范化值 0、1/3、2/3、1 以 float32 进入全部 28 个 pair feature；本次没有把未训练网络的分数冒充已训练模型结果。

## 6. 最可能的原因及证据强度

| 优先级 | 假设 | 结论 | 依据 |
|---|---|---|---|
| 1 | 环境缺少清晰的需求条件化最优边界 | **SUPPORTED** | 强制最优集合切换仅 7.1%；大量 reward 并列、margin=0。 |
| 2 | 策略形成 (7,8) 全局捷径 | **SUPPORTED** | 10/10 seed Top-1 为 (7,8)，但其 Oracle 最优频率低；环境的 universal optimum 本身 **NOT SUPPORTED**。 |
| 3 | 可靠性 reward 信号弱/不平滑 | **PARTIALLY SUPPORTED** | .9/.99 档无违约；.9999 档平均 reliability component -9.404，存在较强但稀疏的阈值信号。 |
| 4 | R_req 输入被损坏/压缩 | **NOT SUPPORTED** | 生成、state、float32、28 个 pair feature 中四档保持区分。 |
| 5 | Actor 没学到 R_req×rho 交互 | **UNKNOWN** | 缺训练后 checkpoint，无法检查二维 score 曲面。 |
| 6 | 探索、延迟信用分配或优化等其他原因 | **UNKNOWN** | 本次即时 Oracle 无法隔离 PPO 长期训练过程。 |

## 其他验证与限制

每状态匹配低/高 rho pair 后，DeltaQ 均值从 -0.034 变为 -2.903，**没有**随 R_req 升高而系统增大；四档 bootstrap CI 见 delta_q_rho_summary.csv，折线与分布见 PNG。匹配成本 median=0.067、P90=0.217，因此不是严格同质 pair 的因果 rho 效应；尤其 rho 在当前模型中不直接进入任务 reward。
完整权重级 P3/P4 所需数据不存在；diagnostic_status.json 列明未生成的图。其余 CSV/JSON/PNG 均经过 finite 值和结构检查。smoke 与 1,000-state full run 均已执行。

## 下一项最有价值的实验

未来一次**同配置**正式训练运行，保存最终 Pair policy_old.state_dict 及完整评估 states，然后在同一批 state 上进行四档 R_req×rho 的 logits/probability sweep，并报告相邻 TV 与有限差分交互；这能直接区分‘输入虽有效但网络未学到交互’和‘环境边界稀少’。当前不重新训练。
