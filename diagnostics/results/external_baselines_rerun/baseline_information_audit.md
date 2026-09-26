# Baseline Information Audit

The ten formal environment seed rows and saved 300-episode Masked PPO
checkpoints are reused. Within each trial, all new policies use the same task
arrival and spatial-risk streams. The policies' internal queues may diverge.
Tie-breaking uses separate deterministic NumPy RNGs, independent of the
environment generators.

| Policy | Decision-time features | Reliability information |
|---|---|---|
| Masked PPO stochastic | The 35-value observation: per-server nominal/base failure rate, processing frequency, normalized CPU backlog time, uplink rate; task input size, computation demand, and requirement. | Production reliability vector and the same safe/effective mask used during training, evaluated from episode-effective server rates; no direct observation of the spatial field. |
| Max-Reliability | Current task computation demand and episode-effective failure inputs passed through the production Task reliability initializer. | All legal pair reliabilities, without threshold filtering for its choice. |
| Safe Min-Latency | Task input size/computation demand plus current per-server backlog, uplink rate, and processing frequency. Those server/task quantities correspond to the PPO observation (backlog is normalized reversibly using the configured scale). | Same production reliability vector and effective-action-mask helper as Masked PPO. |
| Safe Random | No latency preference; it samples uniformly from the shared safe mask or, if empty, shared maximum-reliability fallback mask. | Same production reliability vector and mask/fallback as Masked PPO. |

## Decision-time latency estimate

For server n at decision time t, the estimator uses B_n(t) from
EnvironmentState.get_server_backlog_time, upload U_i,n = 8 * input MB /
uplink Mbps, and service S_i,n = computation demand / processing frequency.
It computes D_hat_i,n = max(B_n(t), U_i,n) + S_i,n. Existing CPU work drains
while the task uploads. The two replicas upload and execute in parallel, and
the production task resolves at the first replica result, so the estimated
pair latency is min(D_hat_i,j, D_hat_i,k). The production model assumes zero
download time and does not Bernoulli-sample execution failures; reliability is
evaluated analytically. The estimate never reads realized Task_Delay or future
arrivals/queue events, and never steps candidate actions. It is a no-new-arrivals
estimate, so it may be optimistic if future arrivals or already-uploading
replicas join the queue before this task.

## Checkpoint/replay checks

Actor and Critic files are verified against their trial manifest and not
written back. Frozen stochastic Masked PPO replay must match the original 4000
actions, safe masks, task reward, delay, and selected reliability per trial.
