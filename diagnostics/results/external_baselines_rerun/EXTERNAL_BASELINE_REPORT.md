# External Reliability Baseline Report

## 1. Baseline Definitions

* **Max-Reliability:** enumerate every legal pair and use the production reliability initializer; sample uniformly among maximum-reliability ties.
* **Reliability-Constrained Min-Latency:** select minimum decision-time estimated first-result latency among the exact production safe set. If it is empty, use the exact Masked PPO maximum-reliability fallback set, then minimize estimated latency within it.
* **Reliability-Constrained Random:** sample uniformly from the exact safe set, or uniformly from the shared maximum-reliability fallback when empty.
* PPO reference policies are the stored formal Pair greedy/stochastic and Masked greedy/stochastic deployments. The Masked stochastic trajectory was replayed only for telemetry and had to match its original recorded actions, masks, and outcomes.

Every method uses 10 paired environment seeds × 20 evaluation episodes × 200 tasks. Exogenous arrivals and spatial-risk streams match per seed. Policy-dependent server queues are expected to differ.

## 2. Information Fairness

See [baseline_information_audit.md](baseline_information_audit.md) for the exact observation variables and estimate equation. Safe-set inputs use the same production feasibility functions as Masked PPO. The heuristic does not receive realized delay or simulate candidate actions.

## 3. Main Results

Mean ± sample SD across the 10 environment seeds; tasks are not treated as independent samples.

| Method | Reward | Mean latency (s) | P95 latency (s) | Overall RSR | Highest-tier RSR | Pair HHI | Maximum mean queue |
|---|---:|---:|---:|---:|---:|---:|---:|
| Max-Reliability | 44.9026 ± 1.4792 | 3.5338 ± 0.2579 | 8.8875 ± 1.2718 | 0.9727 ± 0.0116 | 0.8916 ± 0.0459 | 0.2850 ± 0.0834 | 1.1227 ± 0.3897 |
| Safe Min-Latency | 65.6410 ± 0.6736 | 1.8451 ± 0.0058 | 3.1367 ± 0.0247 | 0.9727 ± 0.0116 | 0.8916 ± 0.0459 | 0.0799 ± 0.0020 | 0.0813 ± 0.0055 |
| Safe Random | 59.9867 ± 0.6574 | 2.0737 ± 0.0069 | 3.7574 ± 0.0509 | 0.9727 ± 0.0116 | 0.8916 ± 0.0459 | 0.0407 ± 0.0011 | 0.0559 ± 0.0054 |
| Masked PPO stochastic | 61.4291 ± 0.6899 | 2.0107 ± 0.0102 | 3.6006 ± 0.0471 | 0.9727 ± 0.0116 | 0.8916 ± 0.0459 | 0.0555 ± 0.0056 | 0.0844 ± 0.0136 |
| Pair PPO greedy | 46.0569 ± 2.1147 | 3.2516 ± 0.2775 | 7.7571 ± 1.2696 | 0.9537 ± 0.0162 | 0.8185 ± 0.0621 | 0.9783 ± 0.0361 | 1.1475 ± 0.2175 |
| Pair PPO stochastic | 57.0973 ± 1.4562 | 1.9942 ± 0.0183 | 3.5626 ± 0.0610 | 0.8972 ± 0.0225 | 0.6333 ± 0.0678 | 0.1352 ± 0.0134 | 0.1679 ± 0.0300 |
| Masked PPO greedy | 50.6739 ± 3.6442 | 2.9138 ± 0.4070 | 6.6945 ± 1.6053 | 0.9727 ± 0.0116 | 0.8916 ± 0.0459 | 0.8319 ± 0.1242 | 0.9265 ± 0.2118 |

## 4. PPO vs Max-Reliability

Max-Reliability achieves mean overall/highest-tier RSR of 0.9727/0.8916. The paired table tests whether reliability-first choices pay for that reliability with reward, latency, queue, or pair concentration. Tie counts are present in the per-decision logs.

## 5. PPO vs Reliability-Constrained Min-Latency

Mean overall RSR is 0.9727 for Masked PPO stochastic and 0.9727 for Safe Min-Latency. At requirement 0.9999, mean feasibility is 0.8916 for Safe Min-Latency and 0.8916 for PPO; corresponding RSR is 0.8916/0.8916. See paired reward/latency/queue comparisons before judging whether the myopic heuristic is sufficient.

## 6. PPO vs Safe Random

Safe Random's mean overall RSR is 0.9727; its comparison isolates the value of actor preference beyond the shared reliability mask.

## 7. Queue and Load Analysis

Per-server utilization, time-weighted mean/P95 queues, and waiting time are in baseline_server_load.csv. Decision telemetry records all current server backlog values and the next task-arrival backlog for the selected servers. Per-seed concentration metrics include pair HHI, maximum server selection share, maximum utilization, and maximum queues.

## 8. Reliability Analysis

Feasibility rate is P(safe set nonempty). Conditional RSR is success among feasible decisions. Avoidable violation means the safe set was nonempty but a selected pair failed the production reliability threshold. Unavoidable violation means the safe set was empty and the chosen fallback could not meet the requirement. The constrained methods are checked for zero avoidable violations. Selected-action shortfall and best-achievable shortfall are both recorded separately.

## 9. Myopic vs Long-Term Behavior

The compressed baseline_decision_telemetry.csv.gz joins each selected-action estimate to realized Task_Delay, current backlog, and selected-server backlog at the next task arrival. Across seeds, Safe Min-Latency had estimated/realized latency 1.7651 ± 0.0061 / 1.8451 ± 0.0058 s, mean realized-minus-estimated error 0.0801 ± 0.0055 s, estimate/realized correlation 0.9295 ± 0.0072, and mean next-arrival selected-pair backlog change 0.4751 ± 0.0093 s. Its selected-pair backlog increased by 0.0477 s from the first to fourth within-episode task quartile. Same-pair consecutive-decision frequency was 0.0661 ± 0.0039. These measurements describe whether the myopic choice accumulates congestion; they do not feed future queue outcomes back into earlier choices.

| Policy | Estimated latency | Realized latency | Realized − estimated | Estimate/realized corr. | Next-arrival backlog Δ | Same pair consecutively |
|---|---:|---:|---:|---:|---:|---:|
| Safe Min-Latency | 1.7651 ± 0.0061 | 1.8451 ± 0.0058 | 0.0801 ± 0.0055 | 0.9295 ± 0.0072 | 0.4751 ± 0.0093 | 0.0661 ± 0.0039 |
| Masked PPO stochastic | 1.9651 ± 0.0098 | 2.0107 ± 0.0102 | 0.0456 ± 0.0043 | 0.9658 ± 0.0040 | 0.4483 ± 0.0168 | 0.0567 ± 0.0099 |

## 10. Seed Stability

The intervals below use 20,000 bootstrap resamples of paired seed-level deltas (n=10), seed 2043. Delta is Masked PPO minus baseline. For reward/RSR, positive favors PPO; for latency/HHI/queue, negative favors PPO. W/L/T counts are seed-level.

| Comparison | Metric | Mean delta [95% CI] | PPO win/loss/tie |
|---|---|---:|---:|
| Masked PPO stochastic vs Safe Min-Latency | mean_reward | -4.21183 [-4.35053, -4.07183] | 0/10/0 |
| Masked PPO stochastic vs Safe Min-Latency | mean_latency | 0.16558 [0.15919, 0.17218] | 0/10/0 |
| Masked PPO stochastic vs Safe Min-Latency | p95_latency | 0.46391 [0.43847, 0.48783] | 0/10/0 |
| Masked PPO stochastic vs Safe Min-Latency | overall_rsr | 0.00000 [0.00000, 0.00000] | 0/0/10 |
| Masked PPO stochastic vs Safe Min-Latency | highest_rsr | 0.00000 [0.00000, 0.00000] | 0/0/10 |
| Masked PPO stochastic vs Safe Min-Latency | pair_selection_hhi | -0.02435 [-0.02772, -0.02072] | 10/0/0 |
| Masked PPO stochastic vs Safe Min-Latency | maximum_mean_queue_length | 0.00310 [-0.00444, 0.01102] | 4/6/0 |
| Masked PPO stochastic vs Max-Reliability | mean_reward | 16.52653 [15.76707, 17.26144] | 10/0/0 |
| Masked PPO stochastic vs Max-Reliability | mean_latency | -1.52308 [-1.67678, -1.37483] | 10/0/0 |
| Masked PPO stochastic vs Max-Reliability | p95_latency | -5.28685 [-6.03153, -4.55486] | 10/0/0 |
| Masked PPO stochastic vs Max-Reliability | overall_rsr | 0.00000 [0.00000, 0.00000] | 0/0/10 |
| Masked PPO stochastic vs Max-Reliability | highest_rsr | 0.00000 [0.00000, 0.00000] | 0/0/10 |
| Masked PPO stochastic vs Max-Reliability | pair_selection_hhi | -0.22949 [-0.27810, -0.17975] | 10/0/0 |
| Masked PPO stochastic vs Max-Reliability | maximum_mean_queue_length | -1.03830 [-1.27818, -0.82386] | 10/0/0 |
| Masked PPO stochastic vs Safe Random | mean_reward | 1.44243 [1.33093, 1.55395] | 10/0/0 |
| Masked PPO stochastic vs Safe Random | mean_latency | -0.06298 [-0.06786, -0.05830] | 10/0/0 |
| Masked PPO stochastic vs Safe Random | p95_latency | -0.15676 [-0.19540, -0.12338] | 10/0/0 |
| Masked PPO stochastic vs Safe Random | overall_rsr | 0.00000 [0.00000, 0.00000] | 0/0/10 |
| Masked PPO stochastic vs Safe Random | highest_rsr | 0.00000 [0.00000, 0.00000] | 0/0/10 |
| Masked PPO stochastic vs Safe Random | pair_selection_hhi | 0.01479 [0.01185, 0.01813] | 0/10/0 |
| Masked PPO stochastic vs Safe Random | maximum_mean_queue_length | 0.02852 [0.02229, 0.03628] | 0/10/0 |

## 11. Conclusion

**Q1.** Max-Reliability is reliability-first by construction; its actual reward, latency, queue, and RSR results are shown above and compared by paired bootstrap.

**Q2.** Safe Min-Latency reaches mean RSR 0.9727; whether it can replace PPO depends on paired reward, mean/tail latency, and queue results, not RSR alone.

**Q3.** The same safe action rule is used by PPO and Safe Min-Latency. Their paired queue/reward/tail-latency deltas test whether the learned stochastic preference adds long-term allocation value.

**Q4.** Safe Random tests whether a mask alone plus random sampling is enough; its measured paired gaps are reported above.

**Does PPO remain necessary given a reliability-safe action set?** Evidence is mixed: some paired metrics favor one method, but the predeclared reward/queue/tail criteria do not establish general PPO necessity over Safe Min-Latency. This is a ten-seed experimental conclusion, not a universal claim. If intervals overlap zero, report the evidence as inconclusive/competitive rather than asserting a PPO advantage.
