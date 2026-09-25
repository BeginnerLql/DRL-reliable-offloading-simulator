# Long-Horizon Coupling Diagnosis

## Design and validity

Evaluated 1000 sampled decision states from the first evaluation episode of each of the 10 formal seed rows; all states come from the current formal 20×200 evaluation schedule. For each sampled state, every exact production safe action was branched (or the shared max-reliability fallback set when the safe set was empty). Each branch used the same frozen Masked PPO stochastic policy after one forced action, the same arrival/spatial-risk seeds, and the same Torch action RNG seed.

A live SimPy environment cannot be deep-copied because its event queue contains Python generator objects (`TypeError: cannot pickle 'generator' object`). The diagnostic therefore uses deterministic prefix replay from episode start as snapshot/restore: before each intervention it requires an exact digest match for the 35-D observation, CPU waiting/running/resource queues, pending tasks, SimPy event calendar, effective hazards, spatial field/correlation/distance, and RNG states; it also requires identical pre-intervention actions and arrivals. A failed check aborts the run. The original evaluation action/reward/delay/reliability replay is checked against the saved formal files.

Horizons are arrival-decision intervals H=1, 5, 10, 20, 50; interval rewards are discounted by the formal PPO γ=0.9 per decision interval. Outcomes resolved during each interval are assigned to that interval, including outcomes of tasks admitted before the target snapshot. Episode-terminal-shortened horizons record their effective length. Counterfactual Q is a diagnostic finite-horizon upper bound under the fixed downstream PPO policy, not a deployable policy.

## Main results

| H | Effective H mean | Myopic-oracle agreement | Mean regret | Median | P90 | P95 | P(regret≤0.01) | P(regret≤0.1) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 0.986 | 0.4854 | 0.0000 | 0.0000 | 0.0000 | 0.986 | 0.986 |
| 5 | 5.0 | 0.797 | 3.0237 | 0.0000 | 10.6314 | 20.0337 | 0.797 | 0.798 |
| 10 | 9.8 | 0.784 | 3.0342 | 0.0000 | 11.2045 | 19.7204 | 0.784 | 0.785 |
| 20 | 19.0 | 0.784 | 3.0546 | 0.0000 | 11.2045 | 19.9812 | 0.784 | 0.785 |
| 50 | 43.5 | 0.784 | 3.0546 | 0.0000 | 11.2045 | 19.9812 | 0.784 | 0.785 |

### Required questions

**Q1 — How long does one pair action affect queues/latency?** The median threshold crossing of action-induced total-backlog differences is 50%: 3.0 arrivals; 25%: 3.0; 10%: 3.0. The half-life is measured against the Safe Min-Latency branch, with early peak difference (first five arrivals) as the normalization; 1.3% did not cross any threshold by H=50. Queue-memory autocorrelation falls from lag-1 0.462 to lag-5 0.020 and lag-20 -0.011. At H=20, PPO-sampled branches have mean absolute backlog difference 0.000s and mean absolute task-latency difference 0.024s from Safe Min-Latency.
**Q2 — Agreement?** At H=1/5/10/20/50, the myopic action belongs to the H-step oracle optimal set in 98.6%, 79.7%, 78.4%, 78.4%, 78.4% of sampled states, respectively; empty-safe-set cases use the maximum-reliability fallback candidate set. Exact oracle ties are handled as a set with 1e-9 reward tolerance.
**Q3 — Myopic regret?** At H=20, mean regret is 3.0546, P95 is 19.9812, and 78.4%/78.5% of states are within 0.01/0.1 return units. At H=50, mean regret is 3.0546.
**Q4 — Are PPO deviations justified?** For the sampled stochastic action, among states where PPO's decision-time latency estimate exceeds the myopic minimum, the fraction with larger Q_H is H=1: 1.0%, H=5: 14.2%, H=10: 15.3%, H=20: 15.3%, H=50: 15.3%. At H=20 the mean sampled-PPO Q gap is -5.5039 versus Safe Min-Latency. The high-probability/greedy actor action is separately included in `ppo_long_term_justification.csv`.
**Q5 — Critic long-term information?** A deterministic grouped 5-fold linear model predicts Critic V(s) from current observable state/backlog and minimum safe latency with R²=0.122. Correlations between V(s), sampled-PPO branch return, and future queue/backlog/latency appear in `critic_long_term_analysis.csv`; these are associations, not proof of what the network represents.
**Q6 — Where could RL matter?** High-load H=20 mean regret is 3.5618 over 337 states. Requirement and safe-set strata are reported separately in `load_stratified_results.csv`, `requirement_stratified_results.csv`, and `safe_set_size_analysis.csv`; PPO deviation rates by load/requirement are in `ppo_justification_by_stratum.csv`. Regime-specific PPO candidate return advantages meeting the 7-of-10-seed consistency criterion: none found.
**Q7 — Is coupling strong enough to support RL?** B. LONG-TERM COUPLING EXISTS BUT PPO DOES NOT EXPLOIT IT. H-step action branches show measurable long-term differences, but the sampled PPO deviations do not establish a repeatable advantage over the myopic heuristic.

## PPO disagreement and externalities

`ppo_vs_myopic_disagreement.csv` compares Masked PPO sampled/high-probability actions against the Safe Min-Latency action on identical snapshot states, including estimated latency, reliability, rho, and Q_H differences. `action_future_queue_effect.csv` gives paired future queue length/backlog, utilization, wait, latency, and task-reward changes for Safe Min-Latency, its nearest-latency competitor, PPO high-probability, and PPO sampled choices. `oracle_improvement_ceiling.csv` reports the finite-horizon Q_H headroom, plus latency/reward outcomes of the Q_H-optimal action set (these outcome deltas are descriptive and may be negative because the objective is total interval reward). `safe_min_latency_externality.csv` evaluates endpoint reselection and selected-pair backlog change over the next 5/10/20 decisions using the complete formal evaluation traces.

## Queue memory and Q predictability

Queue memory is measured as per-server service-backlog (seconds) autocorrelation at 1/2/5/10/20 task-arrival lags across all 20 formal episodes and 10 seeds. Future backlog/latency correlations are in `queue_autocorrelation.csv`. Linear grouped 5-fold models predict candidate Q_H from decision-time latency, safe-set size, selected-server queues, reliability, rho, and task requirement; R² by H is in `q_predictability.csv`.

## Files

The requested CSVs and figures are in this directory. `counterfactual_state_results.csv` and `replay_checks.csv` provide state/action-level audit data. `replay_manifest.json` files record per-seed sample counts and checkpoint hashes. Oracle reward and latency ceilings compare the best-Q_H action branches with the myopic branch for tasks arriving inside each finite horizon.

## Limits

Only 1000 states are counterfactually branched, stratified across the 10 formal seed rows, requirements, and within-seed max-backlog tertiles. This provides causal action comparisons for the current fixed downstream policy; it is not an optimal continuation-policy oracle, and its action-value ceiling is consequently conditional on the downstream PPO. States come from episode 1 per formal seed to keep external RNG snapshot restoration exact and bounded. Results should not be generalized beyond this simulator configuration.

### H=20 strata (paired state comparisons)

| Stratum | Group | n | Mean safe-set size | Agreement | Mean regret | P90 regret | P95 regret | PPO sampled mean Q gap | PPO positive-gap rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| load | high | 337 | 19.72 | 0.760 | 3.5618 | 12.1397 | 22.1032 | -7.0211 | 0.113 |
| load | low | 345 | 21.90 | 0.800 | 2.9321 | 12.8623 | 19.5908 | -4.2898 | 0.128 |
| load | medium | 318 | 20.72 | 0.792 | 2.6501 | 9.3913 | 18.0721 | -5.2134 | 0.116 |
| requirement | 0.9 | 258 | 27.98 | 0.752 | 4.0170 | 14.8943 | 23.1628 | -5.6187 | 0.155 |
| requirement | 0.99 | 255 | 26.62 | 0.800 | 2.6779 | 9.3464 | 18.1361 | -6.3478 | 0.118 |
| requirement | 0.999 | 245 | 19.84 | 0.788 | 3.2757 | 14.1418 | 22.2162 | -5.7732 | 0.094 |
| requirement | 0.9999 | 242 | 7.96 | 0.798 | 2.2017 | 8.1893 | 15.5119 | -4.2199 | 0.107 |
| safe_set_size | 1-5 | 78 | 2.64 | 0.821 | 2.0839 | 8.8086 | 15.2330 | -2.2295 | 0.064 |
| safe_set_size | 16-28 | 723 | 26.25 | 0.788 | 3.2936 | 12.4221 | 21.8575 | -6.1739 | 0.123 |
| safe_set_size | 6-15 | 156 | 10.28 | 0.686 | 3.2744 | 12.8558 | 16.1475 | -5.5535 | 0.160 |
| safe_set_size | empty | 43 | 0.00 | 1.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |

The H=20 PPO advantage columns are sampled-action Q_H minus Safe Min-Latency Q_H on the same snapshot, and the positive-gap rate is descriptive. Empty-safe-set states use the maximum-reliability fallback action set.
