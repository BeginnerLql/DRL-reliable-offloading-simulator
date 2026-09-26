# Policy-Centered Action-Advantage Side Learner Diagnosis

Formal PPO reference metadata commit: 7ddef1af40aef6e42c91f692946fda847a465024; source checkout HEAD: 97d0211b0c5df44db9032697aaa3b930284ca791. One seed (trial 0) × 300 episodes × 200 tasks = 60,000 transitions. This is descriptive one-seed evidence, not a formal multi-seed conclusion.

## Scope and regression

The side learner has separate pair-scoring weights and optimizer. It did not select actions or enter PPO losses. An optional observer copied immutable rollout data at the exact production location after raw GAE and actor-used normalization; side updates occurred only after the original PPO episode update. PPO Actor, V critic, losses, hyperparameters, reward, simulator, and masks were not changed.

Eight-episode regression: action mismatch 0; reward max difference 0; delay max difference 0; effective-mask mismatch 0; Actor/V hashes match True/True.
Full 300-episode regression against the existing formal trial-0 and Q-credit trace passed: True; action/mask mismatches 0/0; reward/delay max differences 2.84e-14/8.88e-16; Actor/V hash match True/True.

## PPO advantage targets

Raw GAE is reported for diagnosis. The side learner target is the exact normalized advantage tensor consumed by the PPO Actor loss. Phase statistics pool transitions descriptively; the 60,000 transitions are not independent seeds.

| Phase | Episodes | Raw GAE mean | Raw GAE std | Actor-used mean | Actor-used std | Median | P05 | P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| early (1-100) | 266.13 | 81.04 | 1.2165e-08 | 1 | -0.012441 | -1.6238 | 1.6281 |
| middle (101-200) | 254.1 | 83.235 | 6.263e-09 | 1 | -0.018858 | -1.6037 | 1.6771 |
| late (201-300) | 245.63 | 83.616 | 3.4535e-08 | 1 | -0.011854 | -1.6046 | 1.655 |

Policy-centering residual: mean 1.09e-08; max 1.14e-07. For states with at least two effective actions, median centered spread 0.24298, collapse rate (<1e-6) 0.00%. Single-action support is excluded from the collapse statistic because zero spread is mathematically forced there.

Fixed frozen sample (n=4096): Huber loss 0.41537 at initialization, 0.41238 at 1,000 steps, 0.40978 at 2,000; final selected-action correlation 0.1311, MAE 0.77942, RMSE 0.98533.
Pair-feature wiring checks all pass: True. Different pair features are fed to the shared scorer, pair permutation is equivariant, action indices map to the existing combinations order, and feature gradients are nonzero.

## Held-out ranking against existing Q_H labels

The established 1,000 held-out states and Q_H labels are evaluation-only. Methods share the effective action set and Q_H labels. Tie-aware optimal sets use the existing old-Q diagnostic tolerance max(1e-8, |Q_best| × 1e-10). Coupled uses regret_h20 > 0.01. Load tertiles, four R_req tiers, and safe-set bins 1–5 / 6–15 / 16–28 / empty reuse historical definitions.

| H | Method | Spearman | Kendall | Top1 | Top3 | Top5 | Mean regret |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | Centered Advantage | 0.4914 | 0.4101 | 0.488 | 0.839 | 0.907 | 5.6438 |
| 5 | Old Q critic | 0.1715 | 0.1318 | 0.488 | 0.604 | 0.810 | 8.095 |
| 5 | PPO Actor | 0.4232 | 0.3478 | 0.594 | 0.831 | 0.932 | 6.5154 |
| 10 | Centered Advantage | 0.4826 | 0.4027 | 0.474 | 0.820 | 0.891 | 5.6962 |
| 10 | Old Q critic | 0.1651 | 0.1264 | 0.480 | 0.595 | 0.796 | 8.1376 |
| 10 | PPO Actor | 0.4153 | 0.3411 | 0.579 | 0.810 | 0.910 | 6.6757 |
| 20 | Centered Advantage | 0.4825 | 0.4025 | 0.474 | 0.820 | 0.891 | 5.7049 |
| 20 | Old Q critic | 0.1650 | 0.1263 | 0.480 | 0.595 | 0.796 | 8.163 |
| 20 | PPO Actor | 0.4151 | 0.3409 | 0.579 | 0.810 | 0.910 | 6.7041 |
| 50 | Centered Advantage | 0.4825 | 0.4025 | 0.474 | 0.820 | 0.891 | 5.7049 |
| 50 | Old Q critic | 0.1650 | 0.1263 | 0.480 | 0.595 | 0.796 | 8.163 |
| 50 | PPO Actor | 0.4151 | 0.3409 | 0.579 | 0.810 | 0.910 | 6.7041 |

### Coupled and easy states, H ≥ 10

| Subset | Method | Spearman | Top1 | Top3 | Top5 | Mean regret | States |
|---|---|---:|---:|---:|---:|---:|---:|
| Coupled | Centered Advantage | 0.1289 | 0.236 | 0.500 | 0.620 | 11.458 | 216 |
| Coupled | Old Q critic | -0.0093 | 0.162 | 0.338 | 0.514 | 14.725 | 216 |
| Coupled | PPO Actor | 0.1162 | 0.139 | 0.509 | 0.699 | 14.956 | 216 |
| Easy | Centered Advantage | 0.5908 | 0.540 | 0.908 | 0.966 | 4.116 | 784 |
| Easy | Old Q critic | 0.2184 | 0.568 | 0.666 | 0.874 | 6.3443 | 784 |
| Easy | PPO Actor | 0.5066 | 0.700 | 0.893 | 0.968 | 4.4185 | 784 |

Full subgroup detail is in heldout_advantage_by_load.csv, heldout_advantage_by_requirement.csv, and heldout_advantage_by_safe_set_size.csv.

## GAE versus selected-action Q_H

Q_H labels belong to frozen held-out evaluation replay states; captured GAE targets belong to training rollouts with separate arrival/spatial RNG streams. No shared transition mapping is saved, so a same-state comparison cannot be established.
No counterfactual values were used for training.

## V critic observation only

Across episodes, mean V-to-GAE-return MAE 255.37, correlation -0.2829, explained variance -0.0000. No value parameters or updates were changed.

## Required answers and decision

- Q1: PPO trajectory unchanged in smoke and full trial-0 regression; action, reward, delay, masks, Actor, and V match.
- Q2: actor-used advantage standard deviations early/middle/late: [0.999999998082893, 0.9999999996402816, 1.0000000073819455].
- Q3: fixed-target fit did not clearly converge under the pre-set rule.
- Q4: centering is numerically correct; maximum residual 1.14e-07.
- Q5: pair-specific wiring passed: True.
- Q6: effective-set collapse rate among multi-action states: 0.00%; median centered spread 0.24298.
- Q7/Q8: see ranking table. H≥10 overall Spearman Advantage/Actor/Old-Q = 0.4826/0.4151/0.1650.
- Q9: H20 coupled Spearman Advantage/Old-Q = 0.1287/-0.0094 over 216 states.
- Q10/Q11: the load and safe-set files show all strata; 6–15 is reported explicitly.
- Q12: Q_H labels belong to frozen held-out evaluation replay states; captured GAE targets belong to training rollouts with separate arrival/spatial RNG streams. No shared transition mapping is saved, so a same-state comparison cannot be established.
- Q13 root-cause category: A. ADVANTAGE REPRESENTATION STILL FAILS.

Actor integration: NO. Only category D meets the stated gate; this one-seed diagnostic does not establish multi-seed significance.

## Output files

All experiment artifacts are under diagnostics/results/advantage_side_learner. Q_H is evaluation-only; the formal 10-seed experiment, old Q audit, external baselines, and simulator results were not modified.
