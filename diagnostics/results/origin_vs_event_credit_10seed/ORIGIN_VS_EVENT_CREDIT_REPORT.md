# Origin-task versus event-interval credit: formal 10-seed study

Both arms use the historical `pair_scoring` Actor, joint clipping, the same PPO hyperparameters, production task reward and masked stochastic deployment. Only training credit differs. Each seed uses 300 × 200 training and 20 × 200 evaluation tasks. Evaluation metrics are task-level.

## Historical regression and reproducibility

The complete trial-0 300-episode gate is in `gate/gate_report.json`. A separate direct 60,000-decision trace gate compares state, decision time, delta_t, action, mask, old log probability and final tensors exactly; see `gate/decision_trace/state_trace_verification.json`. Every Origin trial additionally reproduces its archived Actor/Critic checkpoint, training curve and evaluation actions/rewards. Initial tensors, arrival/spatial streams and input workbooks match between arms.

## Paired performance (Event − Origin)

|Metric|Origin mean ± SD|Event mean ± SD|Delta|95% paired bootstrap CI|W/L/T|
|---|---:|---:|---:|---:|---:|
|mean_reward|61.4291 ± 0.689948|61.4347 ± 0.710411|0.00557032|[-0.0579921, 0.082947]|4/6/0|
|mean_latency|2.01071 ± 0.0102195|2.01019 ± 0.010975|-0.000513318|[-0.00328484, 0.00186914]|3/7/0|
|p95_latency|3.60061 ± 0.0470547|3.60533 ± 0.0504345|0.00472234|[-0.00299325, 0.0133475]|3/5/2|
|overall_rsr|0.972725 ± 0.0115713|0.972725 ± 0.0115713|0|[0, 0]|0/0/10|
|highest_rsr|0.8916 ± 0.0459013|0.8916 ± 0.0459013|0|[0, 0]|0/0/10|
|pair_selection_hhi|0.055509 ± 0.0055785|0.054457 ± 0.00385079|-0.00105204|[-0.00315612, 0.000756586]|7/3/0|
|max_mean_queue|0.0843887 ± 0.0135946|0.0818828 ± 0.0109394|-0.00250588|[-0.00520818, 0.000356788]|7/3/0|

20,000 paired seed bootstrap resamples; seed 2043. Reliability deltas are descriptive.

## Training mechanism

Event return conservation: 3000 episodes, maximum residual 1.48e-12.
Event zero-reward interval fraction: 0.4064. Multiple-event interval fraction: 0.2790. Maximum zero run: 8.
Same-interval rate: 0.4019; shifted-forward rate: 0.5981; median index shift: 1; P90 index shift: 2; max index shift: 13; median resolution delay: 1.98 s.

|Arm|Pre-update Critic MAE|Pearson|Spearman|Explained variance|Pre-update value loss|Raw GAE std|
|---|---:|---:|---:|---:|---:|---:|
|origin|255.42|-0.25716|-0.24214|-2.2974e-05|36033|80.439|
|event|246.6|-0.061459|-0.055025|-4.2736e-06|33808|80.186|

Critic MAE, Pearson and pre-update value loss improve in all 10 paired seeds. Explained variance remains near zero; the relative improvement does not establish a useful value model.
Critic loss here is the pre-update MSE × value-loss coefficient, measured from the actual PPO target before minibatch optimization. It is not the post-update training loss.

## External Safe Min-Latency reference

|Metric|Origin − SafeMin|Event − SafeMin|
|---|---:|---:|
|mean_reward|-4.21183|-4.20626|
|mean_latency|0.165578|0.165065|
|p95_latency|0.463907|0.468629|

SafeMin is an external archived benchmark; it does not participate in either PPO training arm.

## Classification

B — EVENT-INTERVAL CREDIT IMPROVES LEARNING SIGNAL BUT NOT DEPLOYMENT

Event-interval credit was tested for temporal consistency of arrival-to-arrival transitions and all resolution events, including outcomes of previously decided tasks. No expected-Q or prior single-continuation causal claim is assumed.

Go for formal method inclusion: NO. This ablation alone does not justify changing Actor or Critic architecture.
