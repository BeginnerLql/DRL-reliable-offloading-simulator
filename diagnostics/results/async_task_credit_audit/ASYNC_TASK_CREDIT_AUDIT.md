# Asynchronous Task Credit and Future Externality Audit

Matched source: `diagnostics/results/value_gae_identifiability/matched_state_manifest.csv`; exactly 1000 frozen states and 23065 candidate-action branches.
Frozen Masked Pair PPO stochastic continuation; gamma=0.9; no retraining and no production code changed.

## Timing and reward semantics

Decision time `t_i` is the task arrival/action time in the frozen rollout. The actual reward resolution time `tau_i` is production's `Primary_Start + Task_Delay`, validated against the earlier actual replica completion `min(Primary_End, Backup_End)`. Decision-index returns use the originating task order: `r_i + sum_{j>i} gamma^(t_j-t_i) r_j`. Event-time returns use actual resolution times: `sum_{j>=i} gamma^(tau_j-t_i) r_j`. Own/future membership depends only on decision index.
Prior tasks `j<i` still pending at `t_i` are logged as pre-existing provenance and excluded from the root return. Only `j>i` contributes to future externality.

## Integrity

Matched states: 1000; candidate branches: 23065; task reward provenance rows: 2429051; duplicate rewards: 0; pre-existing leakage: 0.
State/mask/probability/backlog/task/hazard/reliability mismatch counts: 0/0/0/0/0/0/0.
Arrival/Torch stream mismatch counts: 0/0. There are no separate runtime per-replica Bernoulli draws in the current simulator; environmental randomness is the fixed arrival/spatial streams plus stochastic policy actions.
Original-action full-episode return max error (decision/event): 1.02e-12/0; original outcome row mismatches: 0; Qevent=Oevent+Fevent max residual: 5.68e-14.
Policy-centering weighted residual mean/max: 8.84e-15/2.47e-13. Actor/Critic checkpoint SHA-256: `f8e729db614052e3acd20aab6c12f7e1ef6698668f750eb0d77e9eceae9404d0` / `7a674efd2dffd689af1d920fc89fa23b81d1a88e9f4269c5db832e3db8228019`.

## Event-time externality size

Own spread mean/median/P90/P95: 26.8755/24.5524/44.2210/56.8076.
Future spread mean/median/P90/P95: 10.5357/0.0000/33.9512/49.5334.
Future/own spread ratio median/P75/P90: 0.0000/0.6767/1.8181.
Own/total exact set agreement 54.9%; mean optimal-set Jaccard 0.812; disjoint rate 11.2%; conflict rate (disjoint or own-optimal total regret > 0.01) 11.2%.
SafeMin event total regret mean/all and conflict: 2.3711/10.6046; PPO sampled action: 8.5881/8.6659.

## GAE and time-semantic comparison

| gae_kind | semantics | component | pearson | spearman | sign_agreement | sign_agreement_including_zero |
| --- | --- | --- | --- | --- | --- | --- |
| actor_normalized | decision | future | 0.0835 | 0.0405 | 0.5773 | 0.2690 |
| actor_normalized | decision | own | 0.0926 | 0.1109 | 0.5159 | 0.5030 |
| actor_normalized | decision | total | 0.1242 | 0.1494 | 0.5313 | 0.5180 |
| actor_normalized | event | future | 0.0850 | 0.0761 | 0.5815 | 0.2710 |
| actor_normalized | event | own | 0.0922 | 0.1144 | 0.5169 | 0.5040 |
| actor_normalized | event | total | 0.1247 | 0.1523 | 0.5344 | 0.5210 |
| raw | decision | future | 0.0864 | 0.0411 | 0.6459 | 0.3010 |
| raw | decision | own | 0.0902 | 0.1111 | 0.6051 | 0.5900 |
| raw | decision | total | 0.1237 | 0.1487 | 0.6000 | 0.5850 |
| raw | event | future | 0.0884 | 0.0758 | 0.6502 | 0.3030 |
| raw | event | own | 0.0897 | 0.1142 | 0.6041 | 0.5890 |
| raw | event | total | 0.1244 | 0.1517 | 0.5990 | 0.5840 |

For raw GAE, event-time own/future/total Spearman=0.1142/0.0758/0.1517.
Decision-vs-event total median Spearman/Kendall=1.0000/1.0000; sign agreement=99.6%; mean optimal-set Jaccard=0.998; exact optimal-set agreement=99.8%.

## Frozen H20 coupled/easy and fixed strata

Coupled is unchanged from the previous H20 audit: the best H20 decision-index return exceeds SafeMin's H20 return by more than 0.01. Load, requirement, and safe-set bins reuse the prior frozen manifest.
| stratum | count | gae_kind | mean_future_spread | conflict_rate | gae_future_spearman | gae_future_sign | safemin_total_mean_regret | ppo_total_mean_regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coupled | 174 | actor_normalized | 24.6028 | 0.6264 | 0.0906 | 0.5506 | 13.6262 | 11.6825 |
| coupled | 174 | raw | 24.6028 | 0.6264 | 0.0651 | 0.5823 | 13.6262 | 11.6825 |
| easy | 826 | actor_normalized | 7.5724 | 0.0036 | 0.0721 | 0.5974 | 0.0001 | 7.9363 |
| easy | 826 | raw | 7.5724 | 0.0036 | 0.0777 | 0.6851 | 0.0001 | 7.9363 |
| high | 303 | actor_normalized | 13.0810 | 0.1485 | 0.1362 | 0.5484 | 3.2132 | 10.8558 |
| high | 303 | raw | 13.0810 | 0.1485 | 0.1263 | 0.6559 | 3.2132 | 10.8558 |
| 0.9999 | 260 | actor_normalized | 3.1946 | 0.0654 | 0.1714 | 0.5976 | 1.2290 | 7.3971 |
| 0.9999 | 260 | raw | 3.1946 | 0.0654 | 0.1719 | 0.6707 | 1.2290 | 7.3971 |
| 6-15 | 98 | actor_normalized | 6.0287 | 0.1224 | 0.0726 | 0.4884 | 2.2123 | 7.2419 |
| 6-15 | 98 | raw | 6.0287 | 0.1224 | 0.0625 | 0.5116 | 2.2123 | 7.2419 |

Requirement:
| requirement | states | future_spread_mean | conflict_rate | own_total_jaccard |
| --- | --- | --- | --- | --- |
| 0.9000 | 300 | 12.9910 | 0.1100 | 0.7971 |
| 0.9900 | 220 | 10.4415 | 0.1091 | 0.8342 |
| 0.9990 | 220 | 15.9576 | 0.1727 | 0.7316 |
| 0.9999 | 260 | 3.1946 | 0.0654 | 0.8793 |

Load:
| load | states | future_spread_mean | conflict_rate |
| --- | --- | --- | --- |
| high | 303 | 13.0810 | 0.1485 |
| low | 398 | 7.6709 | 0.0905 |
| medium | 299 | 11.7697 | 0.1037 |

Effective safe-set:
| safe_set_bin | states | future_spread_mean | conflict_rate |
| --- | --- | --- | --- |
| 1-5 | 55 | 1.3864 | 0.0727 |
| 16-28 | 840 | 11.7484 | 0.1143 |
| 6-15 | 98 | 6.0287 | 0.1224 |
| empty | 7 | 0.0000 | 0.0000 |

## Future reward by downstream distance

These groups are descriptive propagation distances, not learned horizon parameters. Event-time bins are descriptive only.
| band | mean_event_discounted_contribution |
| --- | --- |
| next_1 | 45.3306 |
| next_2_5 | 112.7402 |
| next_6_10 | 52.5840 |
| next_11_20 | 29.6413 |
| next_gt20 | 4.5688 |

## Action-ranking comparison

Old-Q scoring status: scored using the existing target-network checkpoint. Centered Advantage and Q checkpoints were read only.
| reference | method | state_count | mean_spearman | median_spearman | top1_hit_rate | top3_optimal_overlap | top5_optimal_overlap | mean_regret | median_regret | p90_regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FutureEvent | CenteredAdvantage | 1000 | -0.2541 | -0.3116 | 0.7590 | 0.7307 | 0.7468 | 2.8572 | 0.0000 | 10.7545 |
| FutureEvent | OldQ | 1000 | -0.1141 | -0.0868 | 0.7660 | 0.7578 | 0.7728 | 2.6577 | 0.0000 | 10.2816 |
| FutureEvent | PPOActor | 1000 | -0.3052 | -0.3655 | 0.6670 | 0.6973 | 0.7172 | 4.7279 | 0.0000 | 16.5887 |
| FutureEvent | SafeMin | 1000 | -0.1689 | -0.0727 | 0.8350 | 0.7710 | 0.7730 | 2.0848 | 0.0000 | 7.7959 |
| OwnEvent | CenteredAdvantage | 1000 | 0.6654 | 0.7723 | 0.5430 | 0.6263 | 0.5808 | 3.1348 | 0.0000 | 12.0533 |
| OwnEvent | OldQ | 1000 | 0.2335 | 0.3512 | 0.5830 | 0.5311 | 0.4429 | 5.7155 | 0.0000 | 23.7769 |
| OwnEvent | PPOActor | 1000 | 0.5440 | 0.6345 | 0.8780 | 0.6497 | 0.5968 | 2.4049 | 0.0000 | 2.7170 |
| OwnEvent | SafeMin | 1000 | 0.9394 | 1.0000 | 0.9990 | 0.9322 | 0.9118 | 0.9653 | 0.0000 | 0.0000 |
| TotalEvent | CenteredAdvantage | 1000 | 0.5423 | 0.6290 | 0.4670 | 0.5020 | 0.4854 | 5.3129 | 0.4773 | 14.6096 |
| TotalEvent | OldQ | 1000 | 0.2033 | 0.3130 | 0.5300 | 0.4863 | 0.4093 | 7.6942 | 0.0000 | 27.7339 |
| TotalEvent | PPOActor | 1000 | 0.4134 | 0.5083 | 0.6190 | 0.4933 | 0.4766 | 6.4538 | 0.0000 | 22.0356 |
| TotalEvent | SafeMin | 1000 | 0.7848 | 0.9897 | 0.8740 | 0.7593 | 0.7458 | 2.3711 | 0.0000 | 8.5264 |

## Conclusion and go/no-go

Primary = **B**, secondary = **None**. Future spread, own/total conflicts, and SafeMin conflict regret are present, while raw PPO GAE aligns less with future than own advantage.
Evidence: mean future spread=10.5357; own/total disjoint=11.2%; conflict SafeMin regret=10.6046; raw GAE own/future rho=0.1142/0.0758; median decision/event rho=1.0000; optimal-set Jaccard=0.998.
Worthwhile long-term externality: **YES**. Recommend Externality-Aware/Counterfactual Advantage: **YES**. Recommend changing current PPO temporal GAE semantics: **NO**.

## Figures

- Own vs future spread: `own_vs_future_spread.png`
- Future/own ratio: `future_own_spread_ratio.png`
- Optimal-set conflict: `own_total_optimal_conflict.png`
- SafeMin/PPO regret: `safemin_total_regret_conflicts.png`, `ppo_total_regret_conflicts.png`
- GAE alignment: `gae_component_alignment.png`
- Decision/event landscape: `decision_vs_event_total.png`
- Load: `future_spread_by_load.png`
- Task-distance accumulation: `externality_accumulation_by_task_distance.png`
- Coupled/easy: `coupled_easy_externality_spread.png`
- Top-5 cases: `representative_top5_action_components.png`
