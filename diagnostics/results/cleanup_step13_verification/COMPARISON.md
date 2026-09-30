# Step 13 cleanup verification

Baseline commit: `2884d5b83133295b3e93f934fa9cbe64e6613773`.
Both baseline and current source ran 100 episodes × 200 tasks, using the unchanged PPO configuration and frozen profiles.
The temporary harness calls `Project_main.build_model()` and `MainLoop.EP()`; outputs are redirected to this verification directory.
Only the verification processes set Python random, global NumPy and PyTorch seeds to 2026 and PyTorch CPU threads to 1. Spatial RNG retains MASTER_SEED=2026.
No source or permanent seed configuration changed for this verification. Original experiment files remain unchanged.

## Controlled before/after comparison

- Params, Servers, Tasks, Logs and TaskResults are exactly equal after reading Excel cells.
- All 20,000 transition states, actions, rewards, next states, elapsed times, terminal flags and old log probabilities have the same SHA256.
- Final policy, policy_old and value network tensors have the same SHA256.
- All three diagnostic CSVs are exactly equal; both runs pass the independent reliability audit.
- Excel ZIP bytes are not a reproducibility criterion: workbook timestamps can differ.

## Summary

| Metric | Before cleanup (controlled) | After cleanup (controlled) | Historical Step 13 |
| --- | ---: | ---: | ---: |
| RSR | 0.9047 | 0.9047 | 0.9031 |
| mean_latency_s | 6.96276002337 | 6.96276002337 | 7.07041333894 |
| median_latency_s | 6.49607147379 | 6.49607147379 | 6.57950653296 |
| p95_latency_s | 12.1327112742 | 12.1327112742 | 12.4089685747 |
| mean_episode_reward | -1392.55200467 | -1392.55200467 | -1414.08266779 |

## Historical comparison

The historical run did not fix or save workload and policy RNG state. Its policy choices and workload cannot be replayed exactly.
The historical and current runs have identical Tasks, Servers, all 100 spatial contexts and episode-specific feasibility, but different Logs and TaskResults.
That difference alone is not a cleanup regression; the controlled full-run comparison above isolates the source change and is exactly equal.

Full details: [comparison.json](comparison.json).
Current run diagnostic: [STEP13_REPORT.md](current/diagnostics/STEP13_REPORT.md).
