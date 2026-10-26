# TORE 对比方法

唯一方法标识为 **TORE**。这是 TORE-NET 在当前系统模型中的适配版，
不是原论文完整算法的复现。来源：[TORE-NET](https://doi.org/10.1109/JIOT.2026.3722657)，
式(12)–(13)、(18)、(26)–(28)及 Algorithm 1。

## 保留与适配

- 保留完整任务复制、首次完成对应的任务 QoS、独立副本可靠性组合、
  找到满足可靠性要求的选择即停止、所有实际调度副本执行到结束。
- 不建模可用性估计，按单节点预测 QoS 评分。
- 最多两个副本；首选节点足够时提前停止。PPO/SML 仍使用原固定双副本动作空间。
- 保留原文按单节点服务评分逐步增选、检查当前已选集合的机制。
  两副本上限、确定性平分规则及达到上限后的 Fallback 属于本文适配；
  不继承原文完整问题下的近似保证。

## 选择规则

每次任务到达，仅使用与现有 SML 相同的可观测信息预测各节点完成时间：

`T_i = max(upload_i, running_work_i + waiting_work_i) + demand / frequency_i`。

不考虑可用性后，原文的单节点服务评分简化为预测 QoS。服务器排序为：

1. 单节点预测 QoS 降序。
2. 同分时，单节点预测完成时间升序。
3. 仍相同时，服务器 ID 升序。

从空集合开始，加入排名最高的节点并检查可靠性；若不足，保留该节点并加入排名第二的节点，
再次检查。达标即停止，最多两个副本。任务 QoS 仍是所选节点 QoS 的最大值。
**不先遍历所有单节点，不搜索其他节点对，不按节点对时延之和评分。**
因此，即使低排名节点单独可达标，也可能返回两个高排名节点；
即使其他节点对可行，前两个节点不达标也会触发上限处理。

原文 Algorithm 1 第8行将候选范围写为 `M`，第9行加入集合 `X`，但没有显式排除
已经加入 `X` 的节点。按字面反复求相同最大值可能重复选择同一节点，导致集合不增长。
本实现依据正文“逐步增加服务器”的描述，将增选范围解释为尚未选择的节点 `M \ X`。
这是为使伪代码可执行而作的明确解释，并非经作者代码确认的细节。

每个任务使用同一 GH 快照的边际可靠性 `R_i`。
单副本选择可靠性为 `R_i`，双副本选择可靠性为 `1-(1-R_i)(1-R_j)`。
不把基础故障率直接代入指数公式来替换已积分的边际可靠性。
TORE 明确忽略配置文件中已缓存的相关安全 Mask；不会修改供 PPO/SML 共用的缓存。

达到两个副本上限仍不达标时，沿用现有无约束 SML Fallback（包括其原有平分规则），
实际执行两个副本并记录 Fallback。这个分支不保证可靠性达标。
Fallback 表示贪心增选达到上限，不等于全局无可行组合。
日志记录 `greedy_server_ids`（增选顺序）、`greedy_reliability`、
`fallback_reason=replica_limit`；`independent_feasible_pair_exists` 和
`greedy_missed_feasible_pair` 区分全局无解与贪心遗漏。
安全节点对数量仅供诊断，不参与 TORE 的选择或停止条件。

最终可靠性达标率统一使用相关模型；单副本则使用相同边际可靠性。
`pair_reliability` 是兼容历史日志的字段名：单副本任务中其值为该节点边际可靠性。
单副本没有 pair action index，`action_index`、第二副本 ID 和生命周期字段为空；
不会伪造第二份任务，负载、等待统计只计实际执行的副本。
未新增平均副本数指标。

## 运行

比较入口包含 TORE、SML 和 [RASOUL 适配版](rasoul_baseline.md)，使用相同评估任务、
到达序列、服务器和期限，分别重建真实队列。RASOUL 会先使用训练任务预热：

```bash
python compare_tore.py --episodes 5 --seeds 1 --rasoul-warmup-episodes 20 \
  --output-dir results/tore_comparison
```

可加载已有、维度兼容的 PPO 检查点作冻结配对评估，不重新训练：

```bash
python compare_tore.py --episodes 5 --seeds 1 --arrival-rate 2 \
  --failure-rate-scale 0.05 \
  --rasoul-warmup-episodes 20 \
  --ppo-checkpoint results/qos_gpu_quick_comparison/seed_1/final.pt \
  --output-dir results/tore_ppo_comparison
```

默认到达率跟随当前配置，默认故障率缩放为 1。历史实验中的 0.05 缩放需要显式传入。
输出目录必须不存在，避免覆盖历史结果。默认冻结 PPO 在 CPU 推理，可用 `--device cuda`。
RASOUL 默认预热轮数取当前 `total_episodes`，每个评估回合重置到该种子预热后的历史并在线更新。
`--ell-ratio` 改变相关长度，节点边际保持不变。
`--reliability-requirement 0.999` 将本次所有评估任务的可靠性要求统一为 0.999，
在构建安全集合前生效，不修改原始任务工作簿。`effective_task_profiles.csv` 保存实际评估任务
及原始阈值；不改变已有 PPO 检查点，也不会重新训练。
评估种子代表配对到达场景；复用一个检查点不代表多个独立训练模型。

输出包括逐任务结果、Episode/种子汇总、节点负载、相对 TORE 的配对差值、
相关/独立可靠性快照、输入和源码哈希、验证结果及报告。
主要指标为 QoS、平均时延、相关模型可靠性达标率、Fallback 和假安全选择比例。
没有实际故障事件采样，因此 RSR 不是经验任务成功率。

验证：`python -m unittest discover -s tests -v`。
