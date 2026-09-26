# Safe Min-Latency vs Masked Pair PPO: Rerun Reproducibility

## Run identity and scope

- Rerun commit: `3e93404214a2fd027e7073eceee30d34e2d9e3fa`; current `origin/main` at start: `3e93404214a2fd027e7073eceee30d34e2d9e3fa`.
- Formal training reference: `7ddef1af40aef6e42c91f692946fda847a465024`.
- Fresh evaluation only: 10 seeds × 20 episodes × 200 tasks = 40,000 tasks per policy. No PPO training was run.
- Masked PPO was deployed stochastically. Each trial's saved actor and critic hashes matched `completed.json`; weights were unchanged by evaluation.
- Data SHA256 values matched the formal metadata. Legal action ordering matched `MainLoop.generate_combinations()` for all 28 pairs.
- The pre-existing `external_baselines/` directory was verified unchanged by per-file SHA256.

## Main results (rerun, seed is the statistical unit)

Mean ± sample SD over 10 seeds:

| Metric | Safe Min-Latency | Masked PPO stochastic |
|---|---:|---:|
| Reward | 65.64095917 ± 0.67361747 | 61.42913172 ± 0.68994836 |
| Mean latency (s) | 1.84512802 ± 0.00576079 | 2.01070608 ± 0.01021946 |
| P50 latency (s) | 1.81464118 ± 0.01441705 | 1.96134022 ± 0.00787868 |
| P90 latency (s) | 2.83089306 ± 0.00538581 | 3.14024599 ± 0.02437341 |
| P95 latency (s) | 3.13670085 ± 0.02468073 | 3.60060779 ± 0.04705471 |
| Overall RSR | 0.97272500 ± 0.01157131 | 0.97272500 ± 0.01157131 |
| Highest-tier RSR | 0.89160000 ± 0.04590134 | 0.89160000 ± 0.04590134 |
| Conditional RSR | 1.00000000 ± 0.00000000 | 1.00000000 ± 0.00000000 |
| Feasibility rate | 0.97272500 ± 0.01157131 | 0.97272500 ± 0.01157131 |
| AVR | 0.00000000 ± 0.00000000 | 0.00000000 ± 0.00000000 |
| UVR | 0.02727500 ± 0.01157131 | 0.02727500 ± 0.01157131 |
| Pair HHI | 0.07985491 ± 0.00204452 | 0.05550901 ± 0.00557850 |
| Pair entropy | 2.72781531 ± 0.01458309 | 3.08129246 ± 0.06080767 |
| Top-1 pair frequency | 0.16770000 ± 0.00931904 | 0.11335000 ± 0.01708280 |
| Unique selected pairs | 26.30000000 ± 0.82327260 | 28.00000000 ± 0.00000000 |
| Max server selection share | 0.29556250 ± 0.00268176 | 0.21468750 ± 0.01154058 |
| Max utilization | 0.38716589 ± 0.00589179 | 0.30571957 ± 0.01958721 |
| Max mean queue | 0.08128495 ± 0.00554374 | 0.08438866 ± 0.01359457 |
| Max P95 queue | 1.00000000 ± 0.00000000 | 1.00000000 ± 0.00000000 |
| Mean server waiting time | 0.23386384 ± 0.01941365 | 0.30054713 ± 0.01310451 |

## Paired bootstrap: Masked PPO − Safe Min-Latency

20,000 seed-level paired bootstrap resamples, seed 2043; W/L/T counts are PPO wins / Safe Min-Latency wins / ties under each metric's favorable direction.

| Metric | Mean delta | 95% CI | W/L/T |
|---|---:|---:|---:|
| mean_reward | -4.21182745 | [-4.35053078, -4.07183047] | 0/10/0 |
| mean_latency | 0.16557806 | [0.15918635, 0.17218416] | 0/10/0 |
| p95_latency | 0.46390693 | [0.43847237, 0.48783278] | 0/10/0 |
| overall_rsr | 0.00000000 | [0.00000000, 0.00000000] | 0/0/10 |
| highest_rsr | 0.00000000 | [0.00000000, 0.00000000] | 0/0/10 |
| pair_selection_hhi | -0.02434590 | [-0.02772297, -0.02071643] | 10/0/0 |
| maximum_mean_queue_length | 0.00310371 | [-0.00444491, 0.01102083] | 4/6/0 |

## Old vs rerun means

| Metric | Old Safe | Rerun Safe | Abs diff | Old PPO | Rerun PPO | Abs diff |
|---|---:|---:|---:|---:|---:|---:|
| Reward | 65.6409591735 | 65.6409591735 | 0 | 61.4291317216 | 61.4291317216 | 0 |
| Mean latency (s) | 1.84512802323 | 1.84512802323 | 0 | 2.01070607952 | 2.01070607952 | 0 |
| P95 latency (s) | 3.13670085441 | 3.13670085441 | 0 | 3.60060778799 | 3.60060778799 | 0 |
| Overall RSR | 0.972725 | 0.972725 | 0 | 0.972725 | 0.972725 | 0 |
| Highest-tier RSR | 0.8916 | 0.8916 | 0 | 0.8916 | 0.8916 | 0 |
| Pair HHI | 0.0798549125 | 0.0798549125 | 0 | 0.0555090125 | 0.0555090125 | 0 |
| Max mean queue | 0.0812849527602 | 0.0812849527602 | 0 | 0.0843886603952 | 0.0843886603952 | 0 |

`rerun_seed_reproducibility.csv` includes the required per-trial rows for both policies and all available seed metrics. `rerun_metric_reproducibility_summary.csv` gives each metric's maximum absolute difference across the 10 seeds. Match threshold is exactly `1e-12` (`rtol=0`); no tolerance was widened.

| Policy | Metric | Max abs diff across seeds | Exact matches |
|---|---|---:|---:|
| Safe Min-Latency | Reward | 0 | 10/10 |
| Masked PPO stochastic | Reward | 0 | 10/10 |
| Safe Min-Latency | Mean latency (s) | 0 | 10/10 |
| Masked PPO stochastic | Mean latency (s) | 0 | 10/10 |
| Safe Min-Latency | P95 latency (s) | 0 | 10/10 |
| Masked PPO stochastic | P95 latency (s) | 0 | 10/10 |
| Safe Min-Latency | Overall RSR | 0 | 10/10 |
| Masked PPO stochastic | Overall RSR | 0 | 10/10 |
| Safe Min-Latency | Highest-tier RSR | 0 | 10/10 |
| Masked PPO stochastic | Highest-tier RSR | 0 | 10/10 |
| Safe Min-Latency | Pair HHI | 0 | 10/10 |
| Masked PPO stochastic | Pair HHI | 0 | 10/10 |
| Safe Min-Latency | Max mean queue | 0 | 10/10 |
| Masked PPO stochastic | Max mean queue | 0 | 10/10 |

## Per-task replay and paired exogenous streams

- Safe Min-Latency: 0 action mismatches / 40000 tasks.
- Masked PPO stochastic: 0 action mismatches / 40000 tasks.

- Masked PPO replay: 0 action mismatches and 0 safe-mask mismatches across 10 trials × 4000 tasks. The existing runner compared task IDs/actions, safe masks, reward, delay, and execution reliability, failing on any mismatch beyond `1e-12`.
- Safe Min-Latency vs its old task telemetry: action mismatches = 0; selected-pair mismatches = 0; maximum estimate difference = 0; maximum selected-reliability difference = 0.
- Within-rerun environment trace checks: 30 complete policy-stream comparisons passed; arrival mismatches = 0, spatial-risk mismatches = 0. These are assertions in the unmodified production runner.
- Reliability metrics equal between Safe Min-Latency and Masked PPO on every seed at `1e-12`: **True**. This includes feasibility, conditional RSR, AVR, UVR, and mean safe-set size; both methods use the same production reliability vector and effective-mask helper.

## Information fairness and latency estimator

Safe Min-Latency uses only current task size/demand, current server backlog seconds, uplink, and processing frequency. It uses the same production reliability vector and exact safe/effective mask as PPO; empty safe sets use the same maximum-reliability fallback, then minimum estimated latency. The estimate uses current backlog, upload and service time, with pair latency equal to the first replica result. It does not use realized task delay, future arrivals, candidate-action stepping, or cloned-environment rollouts. PPO's normalized backlog is reversible, so raw seconds do not add information.

| Policy | Mean estimated (s) | Mean realized (s) | Realized − estimate (s) | P50 abs error | P90 abs error | P95 abs error | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Safe Min-Latency | 1.76505188 | 1.84512802 | 0.08007614 | 0.00000000 | 0.06208783 | 0.65630513 | 0.92943323 | 0.95959715 |
| Masked PPO stochastic | 1.96507108 | 2.01070608 | 0.04563500 | 0.00000000 | 0.00000000 | 0.18614314 | 0.96568996 | 0.97923108 |

For tasks whose reliability requirement was satisfied, delay/reward Spearman correlations are:

| Policy | Satisfied tasks | Spearman(Task_Delay, Task_Reward) |
|---|---:|---:|
| Safe Min-Latency | 38909 | -0.99999996 |
| Masked PPO stochastic | 38909 | -1.00000000 |

The reward is almost perfectly decreasing with realized delay on satisfied tasks, confirming that Safe Min-Latency is a strong baseline aligned with immediate reward; this is not an information-fairness violation.

## Ten-item fairness and replication audit

1. **Same reliability rule?** Yes; both call the production reliability-vector implementation.
2. **Same safe-set construction?** Yes; both call the same production `effective_action_mask`, including the same max-reliability fallback.
3. **Same arrivals/spatial-risk streams?** Yes within each trial; 30 cross-policy full-stream assertions passed.
4. **Future information used by Safe Min-Latency?** No.
5. **System information unavailable to PPO?** No; raw backlog seconds correspond to PPO's reversible normalization.
6. **Checkpoint frozen?** Yes; all 10 actor/critic hashes matched and evaluation left them unchanged.
7. **Masked PPO replayed old evaluation?** Yes; 40,000/40,000 actions and masks matched, task outcome comparison passed.
8. **Safe Min-Latency old baseline reproduced?** Yes; see per-task and per-seed tables, using the strict `1e-12` threshold.
9. **Old reward/mean/P95 direction reproduced?** Yes: Safe Min-Latency is higher on reward and lower on mean and P95 latency on all 10 seeds.
10. **Old equal-RSR conclusion reproduced?** Yes; overall and highest-tier RSR are equal on every seed.

## Verdict

**A. EXACT / NUMERICALLY IDENTICAL REPRODUCTION**

Maximum absolute old-vs-rerun seed-metric difference: `0`. Maximum per-task outcome difference among Task_Reward, Task_Delay, and Execution_Reliability: `0`. The result independently reproduces the earlier conclusion that Safe Min-Latency has higher reward and lower mean/P95 latency while reliability metrics remain identical.
