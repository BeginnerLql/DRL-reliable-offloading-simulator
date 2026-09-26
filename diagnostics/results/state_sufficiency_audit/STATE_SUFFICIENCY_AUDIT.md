# State Sufficiency Audit

Parent frozen-policy value audit commit: `5fc305361404c7fe0f1a8b5f6efb9105007e753c`. This diagnostic uses one frozen formal evaluation seed and 60,000 task decisions; all fit and state-aliasing comparisons are descriptive.

## Decision

Primary classification: **B — RELIABILITY CONTEXT IS MISSING STATE**. Formal observation expansion: **NO**. PPO retraining: **NO**.
An augmented predictor alone does not show that the state representation causes the performance gap to Safe Min-Latency. The coupled-state GAE H20 alignment gate is required for a state expansion recommendation.
## Frozen trajectory regression

Actor file and parameter hashes match before/after; mismatches: action 0, state 0, mask 0, probability 0, reliability 0. Maximum reward and task-delay differences are 0 and 0. All production/historical protected hashes match.
The upload observer wrapped only the existing input-delay calculation and queue-registration callback. It forwarded the same return values and events. Every episode still resolves 200 tasks and records 400 upload-to-CPU-queue transitions.
## Value prediction on held-out test episodes

| state_variant | model | mae | rmse | pearson | spearman | r2 | explained_variance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | MLP | 82.91 | 105.2 | 0.2955 | 0.2827 | 0.07004 | 0.08724 |
| S0 | Ridge | 82.65 | 104.4 | 0.2947 | 0.2815 | 0.08466 | 0.08627 |
| S1 | MLP | 82 | 104 | 0.3145 | 0.294 | 0.09022 | 0.0989 |
| S1 | Ridge | 81.99 | 103.4 | 0.3183 | 0.2918 | 0.1013 | 0.1013 |
| S2 | MLP | 83.38 | 105.9 | 0.2876 | 0.2756 | 0.05792 | 0.08241 |
| S2 | Ridge | 82.46 | 104.1 | 0.3026 | 0.2909 | 0.08918 | 0.09083 |
| S3 | MLP | 82.07 | 104.1 | 0.3139 | 0.2943 | 0.08887 | 0.09824 |
| S3 | Ridge | 81.83 | 103.2 | 0.3239 | 0.2993 | 0.1048 | 0.1048 |

| model | r2_s0 | r2_s1 | r2_s2 | r2_s3 | delta_r2_reliability | delta_r2_inflight | delta_r2_both | interaction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.07004 | 0.09022 | 0.05792 | 0.08887 | 0.02017 | -0.01212 | 0.01882 | 0.01077 |
| Ridge | 0.08466 | 0.1013 | 0.08918 | 0.1048 | 0.01659 | 0.004517 | 0.0201 | -0.001003 |

S0 exactly reproduces the previous V3 prediction baseline within the tolerance stored in the manifest. MLPs share a [64,32] tanh architecture, target scale, initialization seed and 2,400-update budget; only input dimensions and TRAIN-only scaling of appended features differ. Ridge uses alpha=1 and train-only feature scaling.
## Conditional return variance (same k=20 estimator)

| state_variant | k | n_test | conditional_global_variance_ratio | mean_neighbor_distance | estimator |
| --- | --- | --- | --- | --- | --- |
| S0 | 20 | 12000 | 0.7706 | 0.873 | Mean within 20 TRAIN-neighbor MC-return variance / global TEST variance. |
| S1 | 20 | 12000 | 0.8227 | 3.314 | Mean within 20 TRAIN-neighbor MC-return variance / global TEST variance. |
| S2 | 20 | 12000 | 0.8046 | 1.716 | Mean within 20 TRAIN-neighbor MC-return variance / global TEST variance. |
| S3 | 20 | 12000 | 0.8398 | 4.342 | Mean within 20 TRAIN-neighbor MC-return variance / global TEST variance. |
## State aliasing and hidden workload

### In-flight workload at decision time

| scope | server_id | decisions | decisions_with_inflight | inflight_decision_rate | mean_inflight_replicas | max_inflight_replicas | mean_inflight_future_service_workload | max_inflight_future_service_workload |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggregate_per_decision | all | 60000 | 12298 | 0.205 | 0.3875 | 9 | 0.6606 | 16.98 |
| per_server_per_decision | 1 | 60000 | 1248 | 0.0208 | 0.021 | 2 | 0.05186 | 7.2 |
| per_server_per_decision | 2 | 60000 | 1149 | 0.01915 | 0.01923 | 2 | 0.04599 | 4.909 |
| per_server_per_decision | 3 | 60000 | 1776 | 0.0296 | 0.03012 | 2 | 0.06379 | 6.417 |
| per_server_per_decision | 4 | 60000 | 2065 | 0.03442 | 0.03495 | 3 | 0.06261 | 5.571 |
| per_server_per_decision | 5 | 60000 | 3887 | 0.06478 | 0.06702 | 3 | 0.1171 | 9.333 |
| per_server_per_decision | 6 | 60000 | 3666 | 0.0611 | 0.06358 | 3 | 0.1004 | 5.941 |
| per_server_per_decision | 7 | 60000 | 3299 | 0.05498 | 0.0567 | 3 | 0.0864 | 5.222 |
| per_server_per_decision | 8 | 60000 | 5419 | 0.09032 | 0.09485 | 3 | 0.1324 | 6 |

The S0 nearest-neighbor 10th-percentile distance is 0; ties at distance zero are all retained, yielding 3965 pairs (33.0% of 12,000 test states). Safe/effective masks differ in 51.0%/51.0%; mean Hamming distances are 2.771/2.772. Effective-mask aliases have mean return gap 101.200; all near-neighbor pairs have mean return gap 97.788, and mean policy-distribution TV 0.134.
| group | n_pairs | mean_s0_distance | mean_effective_mask_hamming | effective_mask_alias_rate | mean_safe_mask_hamming | safe_mask_alias_rate | mean_flight_standardized_distance | flight_alias_rate | mean_return_gap | median_return_gap | p90_return_gap | mean_policy_probability_tv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_mask_only | 1458 | 0 | 5.358 | 1 | 5.354 | 1 | 0.0003473 | 0 | 98.8 | 81.85 | 207.7 | 0.2391 |
| B_flight_only | 528 | 0 | 0 | 0 | 0.003788 | 0.003788 | 15.17 | 1 | 93.33 | 74.13 | 201.3 | 0 |
| C_both | 565 | 0 | 5.623 | 1 | 5.627 | 0.9965 | 15.5 | 1 | 107.4 | 84.92 | 233.2 | 0.3237 |
| D_neither | 1414 | 0 | 0 | 0 | 0.0007072 | 0.0007072 | 0 | 0 | 94.57 | 75.81 | 205 | 0 |

| group | n_pairs | mean_flight_distance | mean_return_gap | median_return_gap | mean_future_queue_change_gap | mean_next_task_backlog_gap | mean_realized_latency_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| distinct_inflight | 1093 | 15.34 | 100.6 | 78.62 | 2.166 | 2.17 | 0.2956 |
| similar_inflight | 2872 | 0.0001763 | 96.72 | 78.72 | 1.067 | 1.074 | 0.1917 |
High-load MLP S0/S2 R²: 0.1615/0.1386. Highest-requirement S0/S1 R²: 0.0021/-0.0471. Safe-set 6–15 S0/S3 R²: -0.0168/-0.1906.
Ten nominal-state aliasing examples are saved with both states’ mask, reliability vector, backlog, in-flight count/workload, upload-time summary and realized delay. Pairwise follow-on queue and latency differences are diagnostic outcomes only and never training features.
## H20 action-credit alignment

| state_variant | spearman_overall | sign_agreement_overall | spearman_coupled | sign_agreement_coupled |
| --- | --- | --- | --- | --- |
| S0 | 0.1477 | 0.519 | 0.1057 | 0.5402 |
| S1 | 0.1447 | 0.5118 | 0.1131 | 0.5575 |
| S2 | 0.142 | 0.5221 | 0.1265 | 0.5747 |
| S3 | 0.143 | 0.519 | 0.1306 | 0.5747 |

| state_variant | spearman | sign_agreement |
| --- | --- | --- |
| S0 | 0.2247 | 0.5356 |
| S1 | 0.2265 | 0.5153 |
| S2 | 0.2217 | 0.5492 |
| S3 | 0.2309 | 0.5322 |

| state_variant | spearman | sign_agreement |
| --- | --- | --- |
| S0 | 0.1628 | 0.4426 |
| S1 | 0.1598 | 0.4298 |
| S2 | 0.1372 | 0.4596 |
| S3 | 0.138 | 0.4383 |

| state_variant | spearman | sign_agreement |
| --- | --- | --- |
| S0 | 0.09811 | 0.4898 |
| S1 | 0.06884 | 0.4796 |
| S2 | 0.0656 | 0.5 |
| S3 | 0.04698 | 0.5 |
## Interpretation limits

Reliability masks alter the stochastic policy distribution even though they are not Actor MLP inputs. In-flight work is physically committed to server queues but is absent from the S0 CPU-backlog vector until upload completes. The audit tests whether adding these decision-time features predicts one realized frozen-policy return and whether the corresponding fitted-value GAE better aligns with the already-created matched counterfactual reference. It does not prove a POMDP or a causal explanation for the benchmark gap.
Full feature, regression, stratified prediction, aliasing, case-study and model outputs are in this directory. Future queue changes and latency are labels for diagnosis only; no future information enters S0–S4.
