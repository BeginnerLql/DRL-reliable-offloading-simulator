# Q Credit Assignment Audit

Baseline commit: `147247c55485cfbb792fc4629339000802894222`. Scope: 1 seed × 300 episodes (60,000 transitions) plus the existing 1,000 held-out state set. No PPO Actor/V critic/reward/simulator/action-mask/gamma/tau/learning-rate changes; counterfactual values were not used for training.

## 1. Motivation

Across all 60,000 transitions, mean target = reward 60.928 + bootstrap 14.222 = 75.150; mean selected Q is 23.268. In the final (300th) rollout, the previously reported scale reproduces exactly: reward 54.405 + bootstrap 29.885 = target 84.290, selected Q 43.321, TD error 40.970.

## 2. Bellman Target Decomposition

The target is reconstructed as `reward_term + (gamma ** delta_t) * E_pi_old[Q_target(next_state, ·)]`, with terminal rows masked from bootstrapping. Stage means: early target 64.406, middle 74.653, late 86.391. See `bellman_target_stage_summary.csv` and the transition table.

## 3. Bellman Consistency Checks

A reproducible sample of 1000 late transitions was recomputed from the serialized next state, collection-time next mask/probabilities, and the matching target-network snapshot. Max absolute error 1.32e-05; mean 2.44e-06. Snapshot hashes and state/mask/pi alignment all passed: True, True, True, True. Terminal rows were 6; terminal no-bootstrap check True. `gamma ** delta_t` max discrepancy 1.14e-07.

## 4. Reward Event Attribution

The simulator stores resolved task reward through `assign_task_reward(task_id, reward)`, and the Q transition lookup maps that task id back to its originating task/action transition. Existing task reward consists of base reward minus reliability penalty; no global queue/system reward component was found. Instrumentation copied reward assignment time after the original resolution path and the 60,000 action trace, training reward/delay, and eval metrics were checked against the non-instrumented smoke.
Per-transition source attribution: mean CAR 100.000%; mean signed current-task share 100.000%; mean absolute current-task share 100.000%; previous-task absolute share 0.000%; other/global share 0.000%.

## 5. Reward Delay / Credit Lag

Decision-to-first-outcome simulation-time lag P50/P75/P90/P95 = 1.989/2.641/3.207/3.666 seconds. Later decisions before outcome P50/P90/P95 = 1/2/3. The collector assigns the resolved task reward at a later arrival or terminal drain: outcome-to-assignment lag P50/P90/P95 = 1.363/4.518/5.890 seconds. Despite this delayed bookkeeping, actual Q transition contamination matrix has lag-0 absolute contribution share 100.000%, with lags >0 totaling 0.000%.

## 6. Counterfactual Reward Decomposition

The existing deterministic 1,000-state stratified set was replayed over the same evaluated safe/effective actions. Every candidate's recomputed first-interval reward matched the original runner's Q_H=1 (True); 1000 states and 20834 branches were matched to existing Q_H=20/50 rows. Mean within-state spread: interval total 18.1569, current task 17.9194, previous tasks 0.4136, full target 18.1571; median full-target spread 0.000485, with 56.7% of all states at or below 1e-3. Y_full vs Q_H=20 Spearman/top-1 agreement is 0.489/30.4%; removing historical interval reward gives 0.482/29.9% (927 nondegenerate states). Thus historical interval reward is small and its removal does not restore action ranking.

## 7. Bootstrap Action Differentiation

Across next-state branches, mean within-state spread of expected target-Q is 0.000198; mean selected-action Q spread over effective support is 0.006326; mean bootstrap spread is 0.000174. These spreads are negligible next to the roughly 18-point mean interval-reward spread, so the learned target network contributes almost no branch-dependent continuation ranking. Offline target ranking alignment with Q_H=20 is shown above and in `offline_target_comparison.csv`.

## 8. Q Optimization Sanity

Optimizer instrumentation recorded 1200 updates. Mean selected Q shift per optimizer step is 0.036129; mean parameter update norm is 0.009822, and gradients/step losses are finite. A fixed-target supervised fit on a deterministic 4,096-row sample reduced Huber loss from 34.03294 to 21.39038 (62.852% of initial) after 2,000 steps. Loss largely plateaus by 1,000 steps; this rules out a complete update/gradient failure, but the remaining error does not prove the function class can fit the stochastic realized targets exactly.

## 9. V-vs-Q Scale Comparison

| Metric | Value |
|---|---:|
| mean_immediate_reward | 60.928407 |
| mean_effective_discount | 0.825352 |
| rough_fixed_point_Er_over_1_minus_Ediscount | 348.863168 |
| mean_q_prediction | 23.268286 |
| mean_bellman_target | 75.150101 |
| mean_v_value | 22.783050 |
| mean_ppo_gae_return_target | 278.068649 |
| early_mean_v_value | 9.272546 |
| early_mean_ppo_gae_return_target | 275.400775 |
| early_mean_ppo_v_td_delta | 58.925123 |
| middle_mean_v_value | 22.913154 |
| middle_mean_ppo_gae_return_target | 277.009779 |
| middle_mean_ppo_v_td_delta | 56.503893 |
| late_mean_v_value | 36.163449 |
| late_mean_ppo_gae_return_target | 281.795394 |
| late_mean_ppo_v_td_delta | 55.128714 |
| final_episode_mean_v_value | 42.709310 |
| final_episode_mean_ppo_gae_return_target | 239.976229 |
| final_episode_mean_ppo_v_td_delta | 46.442797 |

On the same transitions V(s) also underestimates its multi-step target: all-run V(s) 22.783 vs PPO GAE return 278.069; in episode 300, 42.709 vs 239.976. This shared V/Q gap indicates general critic under-convergence/value fitting limits, rather than a Q-only target-construction error. The `E[r] / (1-E[discount])` value is a rough scale check only; it ignores correlations and state-dependent policy dynamics.

## 10. Root-Cause Decision

**Primary classification: C — BOOTSTRAP ACTION DIFFERENTIATION COLLAPSES.** The transition-level audit rules against B: current-task attribution is 100.0%, historical task attribution is 0.0%, and actual transition reward source lag >0 is zero by the task-id buffer contract. Target construction/discount/terminal/state/policy alignment is consistent (max target recompute error 1.32e-05). Q updates and parameter changes are nonzero; supervised fixed-target loss improves by 37.1%, so there is no complete optimizer/gradient failure, though its plateau and the matching V-critic return gap show a secondary general value-fitting/convergence limitation. Direct held-out evidence for C is the near-zero next-state E_pi[Q_target] branch spread (0.000198) versus interval reward spread (18.157). This is the strongest supported primary explanation for missing action-conditioned continuation ranking, not proof it is the only contributor to the absolute Q-value gap.

## 11. Implications for RL Design

Task/Event-Aligned Credit Assignment is not justified by the actual online Q transition mapping in this run: task rewards are already attached to their originating task/action transition. Outcome resolution is delayed in simulation time, but the task-id keyed pending-reward buffer prevents cross-task reward reassignment. If a future refactor changes to interval rewards, preserve explicit task/action provenance. If bootstrap action spread remains collapsed, investigate action-conditioned representation or a centered/dueling value decomposition in a separate experiment.

## 12. Limitations

The audit covers one seed and 300 episodes, matching the requested auxiliary smoke rather than a multi-seed conclusion. Counterfactual branches use the existing frozen-policy replay protocol and a single final trained Q target network for next-state bootstrap decomposition; this is offline diagnostic only. The code-level provenance and empirical data support task-id assignment semantics for this simulator version.

### Decision summary

- Bellman means all-run: reward 60.928; bootstrap 14.222; target 75.150; Q 23.268.
- Final rollout: reward 54.405; bootstrap 29.885; target 84.290; Q 43.321.
- Current / historical reward contribution: 100.0% / 0.0%.
- Held-out Q_H=1 interval reward replay match: True.
- Supervised fit Huber loss ratio: 62.9%.
- Primary diagnosis: C.
- Task/Event-Aligned Credit Assignment: not indicated by transition reward provenance alone; action-value/bootstrap differentiation warrants follow-up.
