# Frozen-Policy Value Critic / GAE Target Identifiability Audit



Decision: Primary C, Secondary B; formal-V GO=False; Centered Advantage integration=NO. Validation-selected V3 reduces test MAE from 299.269 to 82.915, while production-consistent normalized H20 GAE Spearman changes 0.1495 → 0.1477, and sign agreement 53.13% → 51.90%.



Source main: 5fc305361404c7fe0f1a8b5f6efb9105007e753c. Trial-0 frozen stochastic Actor, 300 natural episodes / 60,000 decisions. No PPO retraining. One-seed descriptive evidence, not a multi-seed causal conclusion.



## Two implementation facts established before fitting



1. Production masked PPO uses **0.5 × MSE**, with a joint Actor/V gradient-norm cap of 0.5. It does not use Huber. All reported delta=1 Huber fractions are hypothetical; they cannot explain production gradients as Huber saturation.

Source anchors: agents/masked_pair_ppo_agent.py::train_step; core/main_loop.py::_get_ppo_terminal_time; diagnostics/run_long_horizon_coupling.py::_interval_returns; core/env_state.py::get_state.

2. Historical long-horizon Q_H accumulates completion-event interval rewards with gamma**step. Production PPO assigns final task reward back to its originating task transition and discounts by gamma**delta_t. Primary matched comparisons here use the production reward/discount. Legacy interval/step Q_H is retained separately. Historical 21.6% coupled rate and prior ranking numbers therefore do not describe exactly the production objective. No historical artifact was changed.



## Frozen data and terminal integrity



Actor checkpoint SHA256: f8e729db614052e3acd20aab6c12f7e1ef6698668f750eb0d77e9eceae9404d0. Before/after collection and after all side work match. Fixed-state action-distribution hashes match; protected file mismatches: 0.

Split: 180 train / 60 validation / 60 test episodes. Matched indices were frozen before collection and Q_H: 1,000 decisions across the test episodes. Seeds, split IDs, input hashes, index list and optimization gates are persisted. MC maximum recursion residual: 2.27e-13.

After the last arrival, the environment waits for each pending task resolution (first replica result), backfills its own final reward, then stores exactly one terminal transition. Its delta_t ends at the latest task outcome. No terminal bootstrap is used. SimPy subsequently drains remaining non-cancelled replicas. Natural MC uses only that episode’s recorded task-origin rewards; there is no rollout truncation or cross-episode target. The diagnostic original normalized GAE is the value the frozen rollout **would** feed the production actor loss; no Actor update actually occurs.

Observation normalization is the fixed production get_state mapping. Raw state fields and policy-normalized observations, masks, probabilities/logits, physical diagnostics, V/TD/GAE, rewards/delta_t/done and terminal type are saved in per-episode NPZ files. Episode seeds are independent deterministic substreams of one formal seed.



## Horizon-free return scale

| group | mean | std | p05 | p50 | p95 |
|---|---|---|---|---|---|
| train | 337.52 | 109.08 | 162.57 | 334.78 | 521.18 |
| validation | 350.25 | 109.26 | 177.22 | 348.26 | 530.41 |
| test | 342.02 | 109.08 | 172.04 | 336.39 | 528.96 |
| early_position | 382.68 | 106.49 | 218.41 | 376.58 | 568.16 |
| middle_position | 317.05 | 94.394 | 170.86 | 312.09 | 481.8 |
| late_position | 322.88 | 113.68 | 113.57 | 329.65 | 500.45 |
| all | 340.96 | 109.23 | 167.49 | 337.7 | 524.91 |

All reported value errors/calibration are in original return units after inverse transformation; target normalization is TRAIN-only.

## Equal-budget offline value fitting



V1 uses the existing bootstrapped GAE-return target, refreshed once per episode update, not plain TD(0). V2 fits raw horizon-free MC return; V3 fits TRAIN-only standardized MC return and inverse-transforms predictions. Each uses fresh identical initialization, [64,32] tanh, Adam 5e-4, batch64, 2 epochs ×300 train-episode updates =2,400 gradients. The train episodes cycle deterministically. Critic-only clipping uses the same 0.5 limit; this does not reproduce the original combined Actor/V gradient norm. No tuning or test-set checkpoint selection. V4 would be identical to V2 because the production loss is already MSE and is omitted.

Small MLP is [128,64] ReLU with the same budget and standardized target. Ridge alpha=1 uses train-only standardized state columns. They are identifiability diagnostics, not production changes.

MC return removes bootstrap-target dependence and provides a horizon-free **realized** return-to-go fitting target under the frozen policy. It is a stochastic sample, not the true policy value.



### Validation

| arm | mae | rmse | pearson | spearman | r2 | explained_variance |
|---|---|---|---|---|---|---|
| V0 | 307.52 | 326.25 | -0.2254 | -0.21904 | -7.9155 | -7.7285e-07 |
| V1 | 306.92 | 325.67 | -0.23566 | -0.22746 | -7.8842 | -1.0846e-06 |
| V2 | 306.92 | 325.67 | -0.23577 | -0.22749 | -7.8842 | -1.0878e-06 |
| V3 | 84.411 | 106.54 | 0.3039 | 0.2933 | 0.04929 | 0.092352 |
| SmallMLP | 84.105 | 106.13 | 0.31675 | 0.30135 | 0.056449 | 0.099566 |
| Ridge | 83.181 | 104.87 | 0.30443 | 0.29344 | 0.078867 | 0.092438 |

### Test

| arm | mae | rmse | pearson | spearman | r2 | explained_variance |
|---|---|---|---|---|---|---|
| V0 | 299.27 | 318.44 | -0.20993 | -0.19956 | -7.5233 | -7.1876e-07 |
| V1 | 298.66 | 317.87 | -0.21853 | -0.20685 | -7.4927 | -1.0044e-06 |
| V2 | 298.66 | 317.87 | -0.21868 | -0.20692 | -7.4927 | -1.0075e-06 |
| V3 | 82.915 | 105.19 | 0.29545 | 0.28267 | 0.070044 | 0.087244 |
| SmallMLP | 82.642 | 104.74 | 0.3093 | 0.29193 | 0.077924 | 0.09521 |
| Ridge | 82.654 | 104.36 | 0.29468 | 0.28153 | 0.084662 | 0.08627 |



## Hypothetical Huber thresholds / actual optimization

V1/V2 residual thresholds are in raw return units; V3/SmallMLP are in standardized training units. The CSV also includes raw-unit fractions. Prediction-shift magnitudes follow the same native-unit distinction; clipped is the fraction of gradient updates exceeding the shared norm cap.

| arm | unit | hypothetical_huber_saturation_fraction | residual_mean | residual_std |
|---|---|---|---|---|
| V1 | 0 | 1 | -337.59 | 109.09 |
| V1 | 30 | 0.99994 | -330.17 | 109.08 |
| V1 | 150 | 0.99983 | -313.94 | 109.08 |
| V1 | 300 | 0.9995 | -294.07 | 109.08 |
| V2 | 0 | 1 | -337.59 | 109.09 |
| V2 | 30 | 0.99994 | -330.17 | 109.08 |
| V2 | 150 | 0.99983 | -313.94 | 109.08 |
| V2 | 300 | 0.9995 | -294.07 | 109.08 |
| V3 | 0 | 0.30872 | -0.07507 | 1.0085 |
| V3 | 30 | 0.28681 | -0.034168 | 0.96123 |
| V3 | 150 | 0.28589 | -0.13328 | 0.94772 |
| V3 | 300 | 0.28531 | -0.090648 | 0.94755 |
| SmallMLP | 0 | 0.30367 | -0.054288 | 0.99973 |
| SmallMLP | 30 | 0.2865 | -0.012108 | 0.96096 |
| SmallMLP | 150 | 0.28447 | -0.1296 | 0.94493 |
| SmallMLP | 300 | 0.28275 | -0.090676 | 0.94273 |

| arm | gradient_norm_before_clip | parameter_update_norm | prediction_shift | clipped |
|---|---|---|---|---|
| SmallMLP | 0.64229 | 0.013306 | 0.013285 | 0.49542 |
| V1 | 1450.1 | 0.0044998 | 0.018136 | 1 |
| V2 | 1778.7 | 0.0045 | 0.018136 | 1 |
| V3 | 0.71507 | 0.0096999 | 0.012173 | 0.53458 |



## State identifiability and realized-return noise

| method | n | conditional_variance_ratio | mean_neighbor_distance |
|---|---|---|---|
| train-neighbor k20 for test states | 12000 | 0.77061 | 0.87302 |
| test load/demand/R_req coarse bins | 12000 | 0.91743 | nan |

Nearest neighbors and coarse bins approximate states; their residual variance is not an irreducible-noise bound. MC return also depends on remaining natural-episode length; the current observation has no explicit decision index or remaining-episode length. It also excludes the episode-specific spatial hazard realization and effective mask, although those affect the deployed masked distribution. These are possible information limitations, not demonstrated causal explanations; no additional state features are supplied to these fitting arms. Position-stratified return statistics are supplied without adding position to any model.



## Matched-state replay and common randomness

Reconstructed 1000 states, 23065 candidate branches. Aggregate state/mask/probability/backlog/task/snapshot/prefix/arrival/Torch-stream mismatch count: 0. Branches simulate all 200 tasks through natural drain; only the target action is overridden, after consuming its normal categorical draw. Later arrivals, spatial realization and task-index action RNG streams match. Production reliability vectors from the baseline are reused only after hazard/task identity checks; the unforced replay re-evaluates production feasibility.

Each action has one common-random-number continuation, not a many-replicate expectation estimate. H=5/10/20/50 is evaluation only; neither MC/GAE fitting nor arm selection reads Q_H. Centering uses saved masked probabilities. Sign agreement excludes near-zero reference advantages (|A_ref|≤1e-8); denominator and zero-inclusive sensitivity are saved.



### Primary production-consistent normalized GAE alignment

| value_source | horizon | n | nonzero_reference_count | pearson | spearman | sign_agreement | normalized_mae | quartile_agreement |
|---|---|---|---|---|---|---|---|---|
| V0 | 5 | 1000 | 975 | 0.12465 | 0.14873 | 0.53128 | 1.0189 | 0.277 |
| V1 | 5 | 1000 | 975 | 0.12462 | 0.14876 | 0.53128 | 1.0189 | 0.277 |
| V2 | 5 | 1000 | 975 | 0.12462 | 0.14876 | 0.53128 | 1.0189 | 0.277 |
| V3 | 5 | 1000 | 975 | 0.13049 | 0.14568 | 0.51897 | 1.0184 | 0.269 |
| V0 | 10 | 1000 | 975 | 0.12421 | 0.14977 | 0.53128 | 1.0191 | 0.276 |
| V1 | 10 | 1000 | 975 | 0.12418 | 0.14979 | 0.53128 | 1.0191 | 0.276 |
| V2 | 10 | 1000 | 975 | 0.12418 | 0.14979 | 0.53128 | 1.0191 | 0.276 |
| V3 | 10 | 1000 | 975 | 0.13153 | 0.14794 | 0.51897 | 1.0158 | 0.27 |
| V0 | 20 | 1000 | 975 | 0.12415 | 0.14948 | 0.53128 | 1.019 | 0.277 |
| V1 | 20 | 1000 | 975 | 0.12412 | 0.14951 | 0.53128 | 1.019 | 0.277 |
| V2 | 20 | 1000 | 975 | 0.12412 | 0.14951 | 0.53128 | 1.019 | 0.277 |
| V3 | 20 | 1000 | 975 | 0.13179 | 0.14766 | 0.51897 | 1.0159 | 0.271 |
| V0 | 50 | 1000 | 975 | 0.12415 | 0.14945 | 0.53128 | 1.019 | 0.277 |
| V1 | 50 | 1000 | 975 | 0.12412 | 0.14947 | 0.53128 | 1.019 | 0.277 |
| V2 | 50 | 1000 | 975 | 0.12412 | 0.14947 | 0.53128 | 1.019 | 0.277 |
| V3 | 50 | 1000 | 975 | 0.13179 | 0.14764 | 0.51897 | 1.0159 | 0.271 |

### Historical interval/step normalized GAE sensitivity

| value_source | horizon | spearman | sign_agreement |
|---|---|---|---|
| V0 | 5 | 0.1412 | 0.52259 |
| V1 | 5 | 0.14123 | 0.52259 |
| V2 | 5 | 0.14123 | 0.52259 |
| V3 | 5 | 0.13778 | 0.51437 |
| V0 | 10 | 0.14174 | 0.5241 |
| V1 | 10 | 0.14176 | 0.5241 |
| V2 | 10 | 0.14176 | 0.5241 |
| V3 | 10 | 0.13721 | 0.51179 |
| V0 | 20 | 0.14098 | 0.5241 |
| V1 | 20 | 0.14101 | 0.5241 |
| V2 | 20 | 0.14101 | 0.5241 |
| V3 | 20 | 0.13711 | 0.51179 |
| V0 | 50 | 0.1411 | 0.5241 |
| V1 | 50 | 0.14112 | 0.5241 |
| V2 | 50 | 0.14112 | 0.5241 |
| V3 | 50 | 0.13724 | 0.51179 |

### Previously trained centered-advantage learner (frozen)

| horizon | pearson | spearman | sign_agreement |
|---|---|---|---|
| 5 | 0.45576 | 0.45212 | 0.73538 |
| 10 | 0.4532 | 0.45452 | 0.73949 |
| 20 | 0.45327 | 0.45451 | 0.73949 |
| 50 | 0.45327 | 0.45451 | 0.73949 |

This table compares the learner output for the originally selected action to exactly the same reference used for GAE. Per-state within-action ranking and selected greedy regret are separate metrics in frozen_centered_advantage_alignment.csv; they are not directly interchangeable with cross-state GAE correlation.

Raw GAE and all coupling/load/requirement/safe-set strata are in the companion CSVs. Coupled retains myopic H20 regret >0.01 separately for each reward lens. The frozen previous Advantage network is evaluated without parameter updates.



## Required answers

Q1. Actor and original V checkpoints remain frozen; parameter and distribution probes match.

Q2. MC uses gamma**delta_t and a terminal reset; max residual 2.27e-13.

Q3. Original V test MAE=299.27, R²=-7.5233, EV=-7.1876e-07.

Q4. Fresh frozen V1 test MAE=298.66, R²=-7.4927. A change cannot isolate policy tracking from fresh initialization/offline updates/critic-only clipping.

Q5. Raw MC V2 test MAE=298.66, R²=-7.4927.

Q6. Standardized V3 test MAE=82.915, R²=0.070044; raw-versus-standardized scale gate=True.

Q7. Most raw residuals may exceed 1, but active MSE does not saturate like Huber. Gradient clipping/scale is the applicable mechanism to consider.

Q8. MSE-only V4 is redundant with V2, so it is not a distinct experiment.

Q9. Best test R² among V3/Ridge/SmallMLP=0.084662; low-predictability (<0.10) flag=True.

Q10. k20 conditional/global variance ratio=0.77061; coarse-bin ratio=0.91743 (descriptive).

Q11. Exactly 1,000 test decisions matched to their original rollout GAE; mismatch count=0.

Q12. Original normalized GAE H20 Spearman=0.14948, sign=0.53128.

Q13. Validation-selected V3: H20 Spearman=0.14766, sign=0.51897; value gate=True, alignment gate=False.

Q14. Coupled H20 original/best Spearman=0.12766/0.1057; sign=0.58046/0.54023; coupled gate=False.

Q15. The following H20 tables report high-load, all requirements and safe-set strata explicitly.

Q16. Primary=C; Secondary=B. Formal-V modification GO=False; Centered Advantage/Actor integration remains NO.



## H20 key strata



| coupling | value_source | n | nonzero_reference_count | spearman | sign_agreement |
|---|---|---|---|---|---|
| coupled | V0 | 174 | 174 | 0.12766 | 0.58046 |
| coupled | V3 | 174 | 174 | 0.1057 | 0.54023 |
| easy | V0 | 826 | 801 | 0.15222 | 0.5206 |
| easy | V3 | 826 | 801 | 0.15559 | 0.51436 |



| load | value_source | n | nonzero_reference_count | spearman | sign_agreement |
|---|---|---|---|---|---|
| high | V0 | 303 | 295 | 0.21977 | 0.54915 |
| high | V3 | 303 | 295 | 0.22473 | 0.53559 |
| low | V0 | 398 | 389 | 0.12351 | 0.51414 |
| low | V3 | 398 | 389 | 0.10365 | 0.51928 |
| medium | V0 | 299 | 291 | 0.15131 | 0.53608 |
| medium | V3 | 299 | 291 | 0.12029 | 0.50172 |



| requirement | value_source | n | nonzero_reference_count | spearman | sign_agreement |
|---|---|---|---|---|---|
| 0.9 | V0 | 300 | 300 | 0.2448 | 0.58667 |
| 0.9 | V3 | 300 | 300 | 0.21675 | 0.57 |
| 0.99 | V0 | 220 | 220 | -0.022265 | 0.45 |
| 0.99 | V3 | 220 | 220 | 0.039585 | 0.50909 |
| 0.999 | V0 | 220 | 220 | 0.18609 | 0.56818 |
| 0.999 | V3 | 220 | 220 | 0.15569 | 0.54091 |
| 0.9999 | V0 | 260 | 235 | 0.15538 | 0.50213 |
| 0.9999 | V3 | 260 | 235 | 0.16284 | 0.44255 |



| safe_set_bin | value_source | n | nonzero_reference_count | spearman | sign_agreement |
|---|---|---|---|---|---|
| 1-5 | V0 | 55 | 37 | 0.15799 | 0.2973 |
| 1-5 | V3 | 55 | 37 | 0.34657 | 0.2973 |
| 16-28 | V0 | 840 | 840 | 0.14566 | 0.54643 |
| 16-28 | V3 | 840 | 840 | 0.13962 | 0.53214 |
| 6-15 | V0 | 98 | 98 | 0.052413 | 0.4898 |
| 6-15 | V3 | 98 | 98 | 0.098113 | 0.4898 |
| empty | V0 | 7 | 0 | nan | nan |
| empty | V3 | 7 | 0 | nan | nan |



## Decision gates and limitation

The fitted models explain less than 10% of held-out realized-return variance at the fixed budget, supporting limited observed-state predictability (C), not impossibility. Standardization clearly improves optimization (B). However, better absolute-value fit did not improve overall or coupled GAE alignment: the pattern relevant to E is present, and there is no evidence here for D or for immediately changing production PPO. This does not establish the cause of the deployed performance gap to Safe Min-Latency.

```json
{
  "primary": "C",
  "secondary": "B",
  "best_validation_arm": "V3",
  "value_gate": true,
  "overall_gae_gate": false,
  "coupled_gae_gate": false,
  "modify_formal_v": false,
  "integrate_centered_advantage": false,
  "standardization_scale_effect": true,
  "current_state_low_predictability": true,
  "one_seed_only": true,
  "primary_reference": "production task-credit reward + gamma**delta_t",
  "historical_reference": "interval completion reward + gamma**step, sensitivity only",
  "original_huber_hypothesis_rejected": "Formal value loss is MSE; delta1 saturation is hypothetical",
  "v1_inference_limit": "Fresh offline replay and critic-only gradient clipping also differ; improvement alone does not isolate policy nonstationarity causally"
}
```

Classification labels: A = moving-policy value tracking; B = value scale/loss optimization; C = limited state-return identifiability; D = better value fitting improves GAE credit; E = better value fitting does not fix GAE action credit. The C flag means limited prediction in these fixed-budget models, not zero information or proof of irreducible noise.

Multiple diagnostic arms share one formal seed. Neither 60,000 transitions nor 1,000 matched states are independent training seeds. Improvements are descriptive, not statistical proof of a causal mechanism or proof of better deployed PPO.



## Reproduction and regression

Run with the project Conda interpreter from the repository root. The stages are: prepare, collect, fit, branch --workers 4, verify via diagnostics/run_value_gae_identifiability.py; then diagnostics/analyze_value_gae_identifiability.py. Existing collection/branch files are resumed, not silently replaced. Fitting is deterministic for the persisted seeds and software environment.

Only new diagnostic files and this result directory belong to this change. Production sources, formal checkpoints, data spreadsheets and protected historical artifacts are hash-checked. Test results and software versions are recorded in validation_summary.json.
