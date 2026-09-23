# Policy–Oracle Alignment Mechanism Diagnosis

本报告仅分析正式 seed plan 的 trial 0：300 个训练 episode × 200 task，20 个冻结评估 episode × 200 task。诊断子类调用原 PPO 的采样与更新；2-episode 对照的 Actor/Critic 权重逐张量相同，完整 300-episode 训练曲线及 4,000 条评估动作、延迟、奖励与正式 trial 0 完全一致。以下 bootstrap CI 为单次 run 内 state 重采样的描述性区间，不代表跨 seed 推断。

以下数值数据、checkpoint 与图均存放在 diagnostics/results/policy_oracle_alignment/。

## A. Actor 是否知道哪些 pair 真正高 reward？

在 1,000 个按四档分层选取的固定 evaluation state 上，对全部 28 个合法 pair 重放即时 task reward。每档 rank/Top-K/最优集合概率质量：

| R_req | Spearman | Kendall | Top1_hit | Top3_hit | Top5_hit | M_star | Uniform_mass | M_star_lift | Mean_greedy_regret | Corr_normA_negRegret |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9000 | 0.4962 | 0.4066 | 0.4731 | 1.0000 | 1.0000 | 0.4721 | 0.2500 | 0.2221 | 12.8920 | 0.1189 |
| 0.9900 | 0.4951 | 0.4005 | 0.4923 | 0.9385 | 1.0000 | 0.4660 | 0.2475 | 0.2185 | 14.9546 | -0.0282 |
| 0.9990 | 0.5582 | 0.4455 | 0.3958 | 0.9458 | 1.0000 | 0.4394 | 0.2098 | 0.2296 | 18.2278 | 0.1483 |
| 0.9999 | 0.6391 | 0.4959 | 0.2625 | 0.7333 | 0.8500 | 0.2679 | 0.1085 | 0.1594 | 23.6430 | 0.3125 |

Actor 对宽泛排序有信息：Spearman 从 R=.9 的 0.496 到 R=.9999 的 0.639；但高档 Top-1 命中仅 26.2%。M_star 从 0.472 降至 0.268。部分下降来自最优集合缩小（均匀策略基线 0.250→0.108），但高于均匀的质量提升也从 0.222 降至 0.159。因此并非 Actor 完全不了解 reward 排序，而是对最高档最优动作的概率集中不足。完整均值/中位数/P25/P75/P90/bootstrap CI 见 actor_oracle_alignment_by_requirement.csv。
M_star 的完整分布（CI 为 state 重采样区间）：

| R_req | mean | median | P25 | P75 | P90 | CI_low | CI_high |
|---|---|---|---|---|---|---|---|
| 0.9000 | 0.4721 | 0.4485 | 0.4421 | 0.4826 | 0.5349 | 0.4678 | 0.4765 |
| 0.9900 | 0.4660 | 0.4505 | 0.4452 | 0.5358 | 0.5411 | 0.4572 | 0.4744 |
| 0.9990 | 0.4394 | 0.4467 | 0.4226 | 0.4868 | 0.5457 | 0.4276 | 0.4500 |
| 0.9999 | 0.2679 | 0.2092 | 0.1589 | 0.4296 | 0.4873 | 0.2480 | 0.2882 |

probability-vs-regret 与 logit-vs-reward 的 Spearman 在同一 state 中相同，因为 softmax 保序。逐 pair 数值与分箱图见 evaluation_action_vectors.csv / probability_vs_regret.png。

## B. PPO advantage 是否给正确动作正确 credit？

4,800 个按需求平衡抽取的训练状态重放所选动作的即时 Oracle regret。raw/normalized advantage 与负 regret 的 Pearson/Spearman 见 advantage_alignment_by_requirement.csv。真正进入 PPO 更新的是每个 episode 标准化后的 advantage。

| R_req | N | positive_fraction | oracle_regret_mean | pearson_with_negative_regret | spearman_with_negative_regret |
|---|---|---|---|---|---|
| 0.9000 | 1200 | 0.5183 | 5.7190 | 0.1749 | 0.1189 |
| 0.9900 | 1200 | 0.5458 | 7.0886 | -0.0406 | -0.0282 |
| 0.9990 | 1200 | 0.4508 | 13.7726 | 0.2416 | 0.1483 |
| 0.9999 | 1200 | 0.4492 | 32.5090 | 0.2563 | 0.3125 |

最高档 normalized A 的 Spearman=0.312，不是零；A>0 组的平均 regret 20.88，A<0 组 41.99，最优集合命中分别 39.9% / 18.6%。因此高档 credit 信号存在，但并不足以保证最终贪心动作最优。R=.99 的相关性接近零且略负；原始 A 大多为正，必须以 normalized A 判断实际更新方向。即时 Oracle regret 与长期 SMDP return 不同，相关性不能独自证明 PPO 公式错误。

## C. 为什么 (7,8) 成为 policy attractor？

100 个独立初始化的未训练 Actor × 同一 1,000 个 evaluation state 中，(7,8) Top-1 频率 5.39%（28 对中排第 9，均匀参考 3.57%）；它有轻微输入特征偏好，但不是结构上必定胜出的索引。Actor 对 28 对共享同一个 scorer，动作索引是 combinations(1..8,2) 的固定顺序；没有额外 mask，也没有 action-specific 输出权重。
训练前 20 回合，(7,8) 平均采样率 3.4%、贪心率 0.4%、平均概率 0.039；最后 20 回合分别为 18.4%、99.9%、0.187。全程采样 6571/60,000，记录的 4,800 个训练 Oracle 子样本里，当它被采样时平均 regret 2.47（median 0），这是在训练随机策略所形成的队列状态中的条件结果。此时的正优势更新可以强化它，但时间轨迹本身不证明单向因果。
冻结权重、相同到达/空间 seed 下，贪心评估把 99.88% 任务派到 (7,8)，平均延迟 3.359s、reward 43.88；三次独立采样式评估中 (7,8) 平均占 19.84%，延迟 1.990s、reward 53.77。这是策略诱导的队列分布变化，说明训练时“偶尔使用 (7,8) 的好结果”不能外推为“几乎所有任务贪心使用 (7,8) 仍然好”。
代价也必须说明：最高档 R=.9999 的可靠性满足率在贪心评估中 70.2%，采样评估平均 48.1%。采样虽改善当前 reward/延迟，却不能直接作为高可靠部署策略。

可能原因按证据排序：1) **策略模式/队列分布错位（SUPPORTED）**：同权重、同外生 seed 的采样/贪心评估产生大幅不同 pair 集中度、延迟、reward。2) **训练选择偏差与策略吸引（PARTIALLY SUPPORTED）**：被采样时 (7,8) 低 regret，随后其概率和贪心率升高；尚未隔离每次更新的因果贡献。3) **advantage credit 弱且需求相关（PARTIALLY SUPPORTED）**：相关性有限，.99 近零；最高档仍存在正确方向信号。4) **输入特征初始偏好（PARTIALLY SUPPORTED）**：未训练 (7,8) 有轻微偏好，但远非最优先索引。5) **纯 action-index/mask 错误（NOT SUPPORTED）**：共享 scorer、合法 pair 映射一致、无额外 mask。6) **环境中的 (7,8) universal optimum（NOT SUPPORTED）**：此前 Oracle regret 与本轮贪心评估均反驳。

## D. 高 R_req 为什么 regret 更大？

上一轮 exact-optimal set 均值随需求从 7.00 缩到 3.47；本轮 M_star 0.472→0.268、Top-1 命中 47.3%→26.2%，最高档 greedy regret 均值 23.64。最高档违约会跳到失败分支并有 log10 惩罚，使非最优动作的 reward gap 更大。Pair (7,8) 的全局偏好未按高需求与拥塞状态充分调整；这比“R_req 输入丢失”更符合观测。

## E. 是否存在 R_req × rho 的 Oracle interaction？

| R_req | mean_abs_actor_score_slope | mean_abs_oracle_reward_slope |
|---|---|---|
| 0.9000 | 0.5642 | 0.0000 |
| 0.9900 | 0.5889 | 0.0000 |
| 0.9990 | 0.6135 | 0.0000 |
| 0.9999 | 0.6370 | 0.0000 |

在固定同一个 evaluation state、同一 pair (7,8)、同一组 episode 有效节点故障率的实验中，只改变 Actor 的 rho 特征；生产 task reward 使用固定有效故障率和两个副本的条件故障概率，因此条件 Oracle reward 对这个单一 rho 输入为常数。Actor score 对 rho 有非零响应，并随 R_req 略增。**这不等于“空间风险模型没有相关性”**：rho 控制跨节点 episode 风险场的联合分布。仅替换一个 pair 的 rho、同时固定已实现的节点故障率，不是一个完整的风险场反事实；不能据此认定 Actor 的 rho 偏好总体错误。二维数据与图见 rreq_rho_surface.csv、rreq_rho_score_heatmap.png、oracle_rreq_rho_reward_heatmap.png。

## 最值得修改的一个机制

优先解决**训练时随机动作分布与部署时几乎确定性贪心动作造成的队列反馈错位**。建议下一步单独设计并评估一个与部署一致、同时保持任务级可靠性约束的动作选择机制；先在冻结模型上比较不同选择温度/约束规则对延迟、reward、最高档满足率的共同影响，再决定是否改训练。此报告不修改 PPO、Actor 或 reward，除本轮一次同配置 instrumented run 外，没有额外训练模型。
