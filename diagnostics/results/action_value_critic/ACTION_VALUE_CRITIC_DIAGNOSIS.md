# Action-Value Critic Diagnosis

## 1. Motivation

The prior long-horizon experiment found action-dependent future effects: Safe Min-Latency was in the H=1 oracle optimal set for 98.6% of states, falling to about 78.4% by H≥10. This motivates action-specific credit assignment. The finite horizons below are held-out evaluation lenses only; no fixed horizon enters the learned algorithm.

## 2. Method

The side learner predicts all 28 pair values with a shared scorer over the same decision-time node, task, and pair-correlation features as the Pair Actor. Its weights and optimizer are independent. For selected rollout action `a_t`, it minimizes Huber loss against `r_t + gamma**delta_t * E_pi_old[Q_target(s',.)]`; terminal rows use `r_t`. The effective action mask is saved at collection time, and each next-state masked policy distribution is stored before PPO updates. The target network uses soft-update tau=0.005, interval=1; Q learning runs 4 updates per rollout. Only simulator-generated online transitions train Q.

## 3. Training Stability

- All Q training metrics finite / explosion guard passed: **True**.
- Maximum absolute online/target prediction: **43.46**.
- Final episode selected-action Q standard deviation: **0.002404**.
- Pair visitation minimum: **797**; never visited: **0/28**.
- Bellman-target decile calibration MAE: **51.88**.
- Mean held-out effective-set Q spread: **0.0043693** on mean absolute Q scale **43.45** (relative spread **0.000101**); action-value collapse flag (<1e-3 relative): **True**.
- Mean rollout target vs selected-Q prediction in the final episode: **84.29** vs **43.32**; mean TD error remains **40.97**.

## 4. Counterfactual Alignment

The existing 1,000 diagnostic states are held out. Table reports per-state Q-vs-Q_H and Actor-vs-Q_H alignment. H=5/10/20/50 are all shown; no horizon was selected post hoc.

| horizon | state_count | q_spearman | q_kendall | q_top1_hit_rate | q_top3_hit_rate | q_top5_hit_rate | q_mean_regret | actor_spearman | actor_top3_hit_rate | actor_mean_regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1000 | 0.1738 | 0.1343 | 0.4880 | 0.6040 | 0.8100 | 8.0950 | 0.4242 | 0.8310 | 6.5154 |
| 10 | 1000 | 0.1675 | 0.1290 | 0.4800 | 0.5950 | 0.7960 | 8.1376 | 0.4163 | 0.8100 | 6.6757 |
| 20 | 1000 | 0.1673 | 0.1289 | 0.4800 | 0.5950 | 0.7960 | 8.1630 | 0.4161 | 0.8100 | 6.7041 |
| 50 | 1000 | 0.1673 | 0.1289 | 0.4800 | 0.5950 | 0.7960 | 8.1630 | 0.4161 | 0.8100 | 6.7041 |

## 5. Easy vs Coupled States

States are called easy when the existing H=20 myopic regret is ≤0.01, and coupled otherwise.

| horizon | coupling_group | state_count | q_spearman | q_kendall | q_top1_hit_rate | q_top3_hit_rate | q_top5_hit_rate | q_mean_regret | q_median_regret | q_spread_mean | q_safe_std_mean | q_mean_effective | actor_spearman | actor_kendall | actor_top1_hit_rate | actor_top3_hit_rate | actor_top5_hit_rate | actor_mean_regret | random_top1_hit_rate | random_top3_hit_rate | random_top5_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | coupled | 216 | 0.0167 | 0.0126 | 0.2083 | 0.3889 | 0.5787 | 14.2094 | 10.2414 | 0.0035 | 0.0010 | 43.4535 | 0.1388 | 0.1188 | 0.1806 | 0.5694 | 0.7731 | 14.4915 | 0.2245 | 0.5306 | 0.7088 |
| 5 | easy | 784 | 0.2211 | 0.1710 | 0.5651 | 0.6633 | 0.8737 | 6.4104 | 0.0000 | 0.0046 | 0.0013 | 43.4530 | 0.5101 | 0.4185 | 0.7079 | 0.9031 | 0.9758 | 4.3179 | 0.3375 | 0.6544 | 0.8223 |
| 10 | coupled | 216 | -0.0092 | -0.0094 | 0.1620 | 0.3380 | 0.5139 | 14.6477 | 11.2601 | 0.0035 | 0.0010 | 43.4535 | 0.1166 | 0.1022 | 0.1389 | 0.5093 | 0.6991 | 14.8696 | 0.1969 | 0.4930 | 0.6735 |
| 10 | easy | 784 | 0.2211 | 0.1710 | 0.5676 | 0.6658 | 0.8737 | 6.3439 | 0.0000 | 0.0046 | 0.0013 | 43.4530 | 0.5073 | 0.4155 | 0.7003 | 0.8929 | 0.9681 | 4.4181 | 0.3357 | 0.6510 | 0.8190 |
| 20 | coupled | 216 | -0.0094 | -0.0097 | 0.1620 | 0.3380 | 0.5139 | 14.7634 | 11.2601 | 0.0035 | 0.0010 | 43.4535 | 0.1160 | 0.1018 | 0.1389 | 0.5093 | 0.6991 | 14.9993 | 0.1963 | 0.4914 | 0.6713 |
| 20 | easy | 784 | 0.2210 | 0.1710 | 0.5676 | 0.6658 | 0.8737 | 6.3445 | 0.0000 | 0.0046 | 0.0013 | 43.4530 | 0.5072 | 0.4154 | 0.7003 | 0.8929 | 0.9681 | 4.4187 | 0.3357 | 0.6510 | 0.8190 |
| 50 | coupled | 216 | -0.0094 | -0.0097 | 0.1620 | 0.3380 | 0.5139 | 14.7634 | 11.2601 | 0.0035 | 0.0010 | 43.4535 | 0.1160 | 0.1018 | 0.1389 | 0.5093 | 0.6991 | 14.9993 | 0.1963 | 0.4914 | 0.6713 |
| 50 | easy | 784 | 0.2210 | 0.1710 | 0.5676 | 0.6658 | 0.8737 | 6.3445 | 0.0000 | 0.0046 | 0.0013 | 43.4530 | 0.5072 | 0.4154 | 0.7003 | 0.8929 | 0.9681 | 4.4187 | 0.3357 | 0.6510 | 0.8190 |

## 6. Load / Safe-set Stratification

### Load

| horizon | load_tertile | state_count | q_spearman | q_kendall | q_top1_hit_rate | q_top3_hit_rate | q_top5_hit_rate | q_mean_regret | q_median_regret | q_spread_mean | q_safe_std_mean | q_mean_effective | actor_spearman | actor_kendall | actor_top1_hit_rate | actor_top3_hit_rate | actor_top5_hit_rate | actor_mean_regret | random_top1_hit_rate | random_top3_hit_rate | random_top5_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | high | 337 | 0.0121 | 0.0125 | 0.4095 | 0.5401 | 0.6914 | 11.8234 | 4.6750 | 0.0040 | 0.0011 | 43.4534 | 0.3328 | 0.2692 | 0.5638 | 0.8071 | 0.8932 | 7.6484 | 0.3243 | 0.6290 | 0.7933 |
| 5 | low | 345 | 0.3369 | 0.2586 | 0.5855 | 0.6841 | 0.9304 | 5.0952 | 0.0000 | 0.0047 | 0.0013 | 43.4528 | 0.4944 | 0.4147 | 0.6319 | 0.8609 | 0.9565 | 5.6816 | 0.3015 | 0.6189 | 0.7953 |
| 5 | medium | 318 | 0.1627 | 0.1242 | 0.4654 | 0.5849 | 0.8050 | 7.3983 | 0.9561 | 0.0044 | 0.0012 | 43.4531 | 0.4417 | 0.3600 | 0.5849 | 0.8239 | 0.9465 | 6.2193 | 0.3139 | 0.6358 | 0.8052 |
| 10 | high | 337 | -0.0014 | 0.0014 | 0.3976 | 0.5252 | 0.6706 | 12.0530 | 4.9543 | 0.0040 | 0.0011 | 43.4534 | 0.3164 | 0.2567 | 0.5341 | 0.7745 | 0.8546 | 8.0863 | 0.3151 | 0.6150 | 0.7796 |
| 10 | low | 345 | 0.3326 | 0.2552 | 0.5739 | 0.6725 | 0.9188 | 5.0129 | 0.0000 | 0.0047 | 0.0013 | 43.4528 | 0.4916 | 0.4115 | 0.6261 | 0.8522 | 0.9391 | 5.5915 | 0.2969 | 0.6115 | 0.7879 |
| 10 | medium | 318 | 0.1618 | 0.1233 | 0.4654 | 0.5849 | 0.7956 | 7.3782 | 0.9561 | 0.0044 | 0.0012 | 43.4531 | 0.4372 | 0.3557 | 0.5755 | 0.8019 | 0.9371 | 6.3569 | 0.3052 | 0.6248 | 0.7956 |
| 20 | high | 337 | -0.0017 | 0.0011 | 0.3976 | 0.5252 | 0.6706 | 12.0873 | 4.9543 | 0.0040 | 0.0011 | 43.4534 | 0.3157 | 0.2563 | 0.5341 | 0.7745 | 0.8546 | 8.1296 | 0.3151 | 0.6150 | 0.7796 |
| 20 | low | 345 | 0.3325 | 0.2551 | 0.5739 | 0.6725 | 0.9188 | 5.0530 | 0.0000 | 0.0047 | 0.0013 | 43.4528 | 0.4917 | 0.4115 | 0.6261 | 0.8522 | 0.9391 | 5.6316 | 0.2965 | 0.6105 | 0.7865 |
| 20 | medium | 318 | 0.1618 | 0.1232 | 0.4654 | 0.5849 | 0.7956 | 7.3782 | 0.9561 | 0.0044 | 0.0012 | 43.4531 | 0.4372 | 0.3556 | 0.5755 | 0.8019 | 0.9371 | 6.3569 | 0.3052 | 0.6248 | 0.7956 |
| 50 | high | 337 | -0.0017 | 0.0011 | 0.3976 | 0.5252 | 0.6706 | 12.0873 | 4.9543 | 0.0040 | 0.0011 | 43.4534 | 0.3157 | 0.2563 | 0.5341 | 0.7745 | 0.8546 | 8.1296 | 0.3151 | 0.6150 | 0.7796 |
| 50 | low | 345 | 0.3325 | 0.2551 | 0.5739 | 0.6725 | 0.9188 | 5.0530 | 0.0000 | 0.0047 | 0.0013 | 43.4528 | 0.4917 | 0.4115 | 0.6261 | 0.8522 | 0.9391 | 5.6316 | 0.2965 | 0.6105 | 0.7865 |
| 50 | medium | 318 | 0.1618 | 0.1232 | 0.4654 | 0.5849 | 0.7956 | 7.3782 | 0.9561 | 0.0044 | 0.0012 | 43.4531 | 0.4372 | 0.3556 | 0.5755 | 0.8019 | 0.9371 | 6.3569 | 0.3052 | 0.6248 | 0.7956 |

### Effective safe-set size

| horizon | safe_set_bin | state_count | q_spearman | q_kendall | q_top1_hit_rate | q_top3_hit_rate | q_top5_hit_rate | q_mean_regret | q_median_regret | q_spread_mean | q_safe_std_mean | q_mean_effective | actor_spearman | actor_kendall | actor_top1_hit_rate | actor_top3_hit_rate | actor_top5_hit_rate | actor_mean_regret | random_top1_hit_rate | random_top3_hit_rate | random_top5_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 1-5 | 121 | 0.1696 | 0.1617 | 0.8099 | 0.9669 | 1.0000 | 2.1750 | 0.0000 | 0.0003 | 0.0001 | 43.4541 | 0.3541 | 0.3137 | 0.8264 | 0.9917 | 1.0000 | 2.1313 | 0.7705 | 0.9711 | 1.0000 |
| 5 | 6-15 | 156 | -0.0086 | -0.0075 | 0.3269 | 0.6346 | 0.8077 | 10.0356 | 5.2615 | 0.0014 | 0.0004 | 43.4541 | 0.2298 | 0.1917 | 0.4744 | 0.7628 | 0.9103 | 7.5450 | 0.3195 | 0.6808 | 0.8532 |
| 5 | 16-28 | 723 | 0.2135 | 0.1632 | 0.4689 | 0.5367 | 0.7787 | 8.6670 | 0.8477 | 0.0057 | 0.0016 | 43.4527 | 0.4708 | 0.3855 | 0.5809 | 0.8188 | 0.9253 | 7.0270 | 0.2352 | 0.5587 | 0.7520 |
| 10 | 1-5 | 121 | 0.1404 | 0.1319 | 0.8017 | 0.9587 | 1.0000 | 2.3036 | 0.0000 | 0.0003 | 0.0001 | 43.4541 | 0.3630 | 0.3210 | 0.8264 | 0.9917 | 1.0000 | 2.3236 | 0.7657 | 0.9690 | 1.0000 |
| 10 | 6-15 | 156 | -0.0135 | -0.0126 | 0.3141 | 0.6154 | 0.7885 | 10.3464 | 5.2858 | 0.0014 | 0.0004 | 43.4541 | 0.2300 | 0.1904 | 0.4679 | 0.7500 | 0.8974 | 7.4996 | 0.3116 | 0.6669 | 0.8414 |
| 10 | 16-28 | 723 | 0.2083 | 0.1594 | 0.4620 | 0.5297 | 0.7635 | 8.6373 | 0.9516 | 0.0057 | 0.0016 | 43.4527 | 0.4600 | 0.3767 | 0.5615 | 0.7925 | 0.8976 | 7.2262 | 0.2274 | 0.5472 | 0.7404 |
| 20 | 1-5 | 121 | 0.1404 | 0.1319 | 0.8017 | 0.9587 | 1.0000 | 2.3036 | 0.0000 | 0.0003 | 0.0001 | 43.4541 | 0.3630 | 0.3210 | 0.8264 | 0.9917 | 1.0000 | 2.3236 | 0.7657 | 0.9690 | 1.0000 |
| 20 | 6-15 | 156 | -0.0145 | -0.0134 | 0.3141 | 0.6154 | 0.7885 | 10.4827 | 5.2858 | 0.0014 | 0.0004 | 43.4541 | 0.2289 | 0.1895 | 0.4679 | 0.7500 | 0.8974 | 7.6360 | 0.3116 | 0.6669 | 0.8414 |
| 20 | 16-28 | 723 | 0.2083 | 0.1594 | 0.4620 | 0.5297 | 0.7635 | 8.6431 | 0.9516 | 0.0057 | 0.0016 | 43.4527 | 0.4600 | 0.3767 | 0.5615 | 0.7925 | 0.8976 | 7.2362 | 0.2272 | 0.5467 | 0.7397 |
| 50 | 1-5 | 121 | 0.1404 | 0.1319 | 0.8017 | 0.9587 | 1.0000 | 2.3036 | 0.0000 | 0.0003 | 0.0001 | 43.4541 | 0.3630 | 0.3210 | 0.8264 | 0.9917 | 1.0000 | 2.3236 | 0.7657 | 0.9690 | 1.0000 |
| 50 | 6-15 | 156 | -0.0145 | -0.0134 | 0.3141 | 0.6154 | 0.7885 | 10.4827 | 5.2858 | 0.0014 | 0.0004 | 43.4541 | 0.2289 | 0.1895 | 0.4679 | 0.7500 | 0.8974 | 7.6360 | 0.3116 | 0.6669 | 0.8414 |
| 50 | 16-28 | 723 | 0.2083 | 0.1594 | 0.4620 | 0.5297 | 0.7635 | 8.6431 | 0.9516 | 0.0057 | 0.0016 | 43.4527 | 0.4600 | 0.3767 | 0.5615 | 0.7925 | 0.8976 | 7.2362 | 0.2272 | 0.5467 | 0.7397 |

## 7. Actor vs Q Critic

Across H≥10, mean Q Spearman is **0.1674** versus Actor **0.4161**; top-3 optimal-set hit is **59.500%** versus Actor **81.000%**. The Actor is closer overall; mean argmax regret is **8.1545** for Q and **6.6946** for Actor. On coupled states Q Spearman is **-0.0093**; on easy states it is **0.2211**, with easy top-3 gain over random of **1.479%**.

## 8. Q Calibration

Calibration uses selected-action Bellman targets from online rollouts, not counterfactual returns. See `q_calibration.csv` and `q_calibration.png`. This is one-step TD calibration and does not establish calibration to an optimal-control value function.

## 9. Limitations

Counterfactual Q_H is evaluation-only and comes from fixed downstream Masked PPO continuation under finite horizons. It is not the optimal-control Q*. State-level diagnostic snapshots are held out, but the counterfactual labels are policy-dependent and may include simulation noise. The existing scalar V(s) and PPO Actor update remain unchanged.

## 10. Decision

**C. ACTION VALUE IS NOT LEARNED RELIABLY**

This decision uses the H≥10 Q-vs-Q_H rank correlation, easy/coupled split, top-3 hit relative to random, Actor comparison, and relative within-state Q spread. The one-seed critic is numerically finite but its action values collapse to a very narrow range and coupled-state ranking is at/below zero, so the formal 10-seed stage was not started. Action-Value Advantage PPO is not implemented.
