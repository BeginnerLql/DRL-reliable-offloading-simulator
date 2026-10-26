# RASOUL 适配版

方法标识为 `RASOUL`。来源为 [RASOUL](https://doi.org/10.1109/JIOT.2025.3592330)，
原文第 III–V 节、式 (8)、(18)–(23) 和 Algorithm 1。
这是当前系统模型中的明确改编，不是原文完整算法的复现。

## 适配范围

- 固定单副本，保持当前单核 FIFO、上传与 CPU 队列模型。
- 使用与其他方法相同的当前运行和等待工作量预测完成时间。
- 保留单节点可靠性筛选、QoS 误差历史、置信修正及负载准入。
- 不建模原文的可用性学习，使用当前统一的任务 QoS。
- 历史修正对象从原文的无排队基础 QoS 差距，改为含当前队列预测的剩余误差。
- 补充冷启动、平分规则、两类回退及预热/评估历史隔离协议。

## 决策规则

节点顺序为 Server_ID 升序，状态与其他方法一致。预测时延为：

`T_hat_j = max(upload_j, running_work_j + waiting_work_j) + demand/frequency_j`。

用现有期限函数计算未修正的 `Q_hat_j`。节点边际可靠性来自与 TORE、PPO、SML
相同的 GH 快照；单节点不存在跨节点独立组合或相关组合之分。

1. 过滤边际可靠性低于任务要求的节点。
2. 对每个节点计算 `score_j = Q_hat_j - mean_error_j + sqrt(2*ln(max(2,N))/n_j)`。
   `N` 是包含当前分配在内的累计决策次数，`n_j` 只计已完成反馈。
   `n_j=0` 时优先级为正无穷；日志用 `rasoul_cold_start=true` 与空分数字段表示。
   有限评分不裁剪，可以大于 1，仅用于排序，不解释为实际 QoS 或成功概率。
3. 按评分降序、未修正预测时延升序、Server_ID 升序排列可靠性可行节点。
4. 按此顺序选择第一个通过单核负载准入的节点。

准入条件为 `U_j + (demand/frequency_j)/deadline <= 1`。
`U_j` 是当前运行与 CPU 等待任务各自的 `完整执行时间/自身相对期限` 之和。
运行任务不改用剩余时间，完成后移除；尚在上传中的任务不计入。
这是约定的负载规则，不是任意任务严格满足期限的保证。当前队列的各任务参数来自
本方法先前已分配的任务及 CPU 驻留信息，不使用未来到达或未来完成结果。

两类回退均继续执行一个副本，不拒绝任务：

| 条件 | 选择 | `fallback_reason` |
|---|---|---|
| 有可靠性可行节点，但全部未通过准入 | 可行节点中预测最快者 | `admission` |
| 没有可靠性可行节点 | 全部节点中预测最快者 | `no_reliable_node` |

回退的时延平分按 Server_ID 升序。准入回退仍然满足可靠性要求，不能与可靠性违约混计。

## 反馈与历史

在决策时保存所选节点未修正的 `Q_hat_j`。任务完成事件触发后，仅更新该节点一次：

`error = Q_hat_at_decision - actual_task_qos`。

不得使用加过历史修正或探索项的评分作为误差基准。正常任务和两类回退任务全部提供
反馈。采用累计样本数及误差和，无滑动窗口、衰减系数或额外调参项。
反馈跟随 SimPy 的实际事件处理顺序，同一时刻仍只使用已处理的完成反馈。

每个种子在训练工作簿任务上独立预热，使用独立于评估的到达随机流。预热轮间保留历史、
重建空队列；末轮全部任务和反馈排空后保存完整历史，包括累计决策数、各节点反馈数和误差和。

每个评估回合从空队列和同一份预热历史的独立副本开始，回合内在线更新。
评估结束后的历史不带入下一回合，也不修改预热快照。只有完全排空的历史允许保存和恢复。
不同到达率或故障率场景应分别运行并预热。

## 运行与输出

`compare_tore.py` 现在默认比较 TORE、SML、RASOUL；提供 PPO 检查点后比较四种方法。
不重新训练 PPO。RASOUL 的预热轮数由 `--rasoul-warmup-episodes` 显式设置，默认取
`params.total_episodes`。预热使用与本次评估相同的到达率、故障率缩放及物理配置。
所有方法使用相同评估任务及到达序列，各自重建队列。

```bash
python compare_tore.py --episodes 5 --seeds 1 \
  --rasoul-warmup-episodes 20 \
  --output-dir results/rasoul_comparison
```

加入已有的、状态维度兼容的 PPO 检查点：

```bash
python compare_tore.py --episodes 5 --seeds 1 \
  --rasoul-warmup-episodes 20 \
  --ppo-checkpoint results/qos_gpu_quick_comparison/seed_1/final.pt \
  --output-dir results/four_method_comparison
```

默认物理参数来自当前配置；前期实验的 `--failure-rate-scale 0.05` 和到达率需明确传入，
不会从检查点自动推断。`--reliability-requirement` 保持现有的仅覆盖评估任务语义，
RASOUL 预热保留训练工作簿原始阈值，二者均写入运行元数据；输入工作簿不变。
输出目录必须不存在。

- `rasoul_history_seed_*.json`：完全排空的预热历史快照。
- `rasoul_warmup_tasks_seed_*.csv.gz`：预热逐任务反馈与决策。
- `rasoul_warmup_episode_metrics.csv`：预热回合指标，与评估分开。
- `eval_task_metrics.csv`：所选节点、未修正预测、学习评分、历史均值、置信项、
  决策前反馈数、单节点可行/准入掩码、负载、回退原因及完成反馈。
- `eval_episode_metrics.csv`：评估指标、两类回退、预测误差、评估前后历史哈希和决策数。
- 另有每种子汇总、总体汇总、相对 TORE 的配对差值、服务器负载及验证报告。

兼容字段 `pair_reliability` 在单副本下存所选节点边际可靠性。第二副本字段与 pair action
index 为空；不伪造第二份执行。已有 `safe_mask`/`safe_pair_count` 仅是相关节点对可行性诊断，
不参与 RASOUL 选择；真正的单节点条件写入 `safe_node_mask`/`safe_node_count`。

主要指标仍为全部任务平均 QoS/时延和理论可靠性要求满足率。所有回退任务计入，
预热任务不计入。没有引入随机故障事件采样，RSR 不解释为经验任务成功率。
在固定任务和物理参数下，RASOUL 的 RSR 等于存在可靠性可行单节点的任务比例，
低达标率不能单独归因于学习算法。复用一个 PPO 检查点的多个种子不是多个独立 PPO 模型。

验证命令：`python -m unittest discover -s tests -v`。
