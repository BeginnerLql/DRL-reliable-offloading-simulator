# DRL-Based Reliable Offloading Simulator

This repository provides a **generic and modular simulator** for reliability-aware
task offloading in **eight unified Edge servers**.
The simulator supports pluggable Deep Reinforcement Learning (DRL) agents
(e.g., **DQN**, **PPO**, **DDPG**) and is not tied to any specific application domain
(e.g., vehicular or RSU-based systems).

All input Excel files are generated automatically, and all experiment outputs are
written to a dedicated results directory.

---

## Project structure

- `Project_main.py`  
  Main entry point for running the simulator.

- `pre_process.py`  
  Standalone launcher for generating input Excel files.

- `post_process.py`  
  Standalone launcher for post-processing and aggregating results.

- `config/`  
  Experiment configuration and centralized paths:
  - `configuration.py` – fixed failure-rate ranges, agent selection, hyperparameters
  - `params.py` – unified parameter object  
  - `paths.py` – single source of truth for project paths (project root, `data`, `results`)

- `core/`  
  Simulation core (environment, state representation, episode loop, tasks, servers).

- `agents/`  
  DRL agents (DQN / PPO / DDPG), implemented as interchangeable modules.

- `tools/`  
  Utility scripts (e.g., generation of input Excel parameter files).

- `io_utils/`  
  Result logging and post-processing utilities.

- `data/`  
  Input Excel files generated during the pre-processing step.

- `results/`  
  Output Excel files generated per model.

---

## Requirements and environment setup

### Python version
- Python **3.9** or **3.10** is recommended.

### Required libraries
All required Python dependencies are listed in `requirements.txt`.

Install dependencies using:

```bash
pip install -r requirements.txt
```

It is recommended to use a virtual environment before installing dependencies.

---

## Execution workflow

### 1) Pre-process (generate input Excel files)

Run this step **only if** input Excel files do not exist or need to be regenerated:

```bash
python pre_process.py
```

This script generates all required Excel files into the `data/` directory.
`PROFILE_SEED = 2026` controls server/task profile generation. NumPy
`SeedSequence` uses fixed stream keys: 0 for servers, 1 for training tasks,
and 2 for evaluation tasks. Within each profile, CPU/rate/failure parameters
or size/demand/reliability-order parameters have separate child streams.
Regeneration preserves the same values and row order independently of global
RNG calls or the order in which profiles are generated. No additional master
seed setting is needed.

The generated workbooks retain their existing column names:

| File | Profile and units |
| --- | --- |
| `data/server_info.xlsx` | Eight servers: `Processing_Frequency` in cycles/s, `Transmission_Rate` in MB/s, `Base_Failure_Rate` in s^-1; existing EUA locations and Server_ID/Site_ID mapping |
| `data/task_parameters.xlsx` | Fixed training set: 200 tasks, `Task_Size` in MB, `Computation_Demand` in CPU cycles |
| `data/task_parameters_evaluation.xlsx` | Independently generated fixed evaluation set with the same schema and distributions |

Task size is continuous uniform on 5–10 MB; computation demand is independently
continuous uniform on 0.5e9–3e9 CPU cycles. Each set has exactly 50 tasks at each
reliability level, with a reproducibly shuffled tier order. There are exactly four
task columns: `Task_ID`, `Task_Size`, `Computation_Demand`, and
`Reliability_Requirement`.

`MainLoop` loads the training file by default and reuses the same attributes and
Task_ID order in every episode. To load evaluation data, pass
`task_parameters_path=EVALUATION_TASK_PATH` (from `config.paths`) to its
constructor. Pass the same path to `save_params_and_logs` when saving that run's
task snapshot. This selects data only; an evaluation execution mode is a later
stage. Future Uniform Random load scans should use the training file; algorithms
being evaluated should share the evaluation file.

`EXPERIMENT_SEED` controls PPO initialization, action sampling and minibatch
shuffling, plus the environment arrival and spatial fields. Arrival and spatial
fields use dedicated environment RNGs, so PPO random-number consumption does
not change either environment trace.

---

### 2) Run the simulation

```bash
python Project_main.py
```

Each experiment writes a new workbook directly to:

```text
results/{model}_{run_id}_{experiment_tag}.xlsx
```

`run_id` is generated once at experiment startup as `YYYYMMDD_HHMMSS`.
Set `EXPERIMENT_TAG = "baseline"` in `config/configuration.py` to label a run.
For example: `results/ppo_20261001_140532_baseline.xlsx`. Existing experiment
files cannot be overwritten: a filename collision raises `FileExistsError`.

Each experiment workbook contains five sheets:

- `Params`: shared configuration and the selected model's hyperparameters,
  including `run_id` and `EXPERIMENT_TAG`; Python internal attributes and other
  models' hyperparameters are excluded.
- `Tasks`: the full task profile snapshot used for this experiment.
- `Servers`: the full server profile snapshot used for this experiment.
- `Logs`: episode-level reward and mean task latency.
- `TaskResults`: task-level outcomes in the existing 20-column schema.

### 3) Summarize saved experiments

```bash
python post_process.py
```

This scans `results/*.xlsx`, skipping Excel temporary files and
`Final_Result_All.xlsx`. Model, run ID and experiment tag are read from each
workbook's `Params` sheet. The output is `results/Final_Result_All.xlsx`:

- `ModelSummary`: one row per experiment, with episode/outcome counts, mean
  episode reward, overall RSR, mean task latency and P95 task latency.
- `EpisodeMetrics`: per-run episode reward, mean latency, task count, RSR and
  P95 task latency.
- `PairSelection`: per-run counts and shares for every unordered server pair,
  including zero-selection pairs.

All three tables include `Model`, `Run_ID`, `Experiment_Tag` and `Source_File`.
RSR is the mean of the saved `requirement_satisfied` booleans; latency is in
seconds. `Final_Result_All.xlsx` is a derived summary and may be overwritten
when regenerated. Individual experiment workbooks are the original records
and must be preserved. With no input experiment workbooks, post-processing
raises `ValueError` without replacing the existing summary.

The historical `results/fixed_rate_results/ppo_results.xlsx` remains unchanged.
New runs do not write there, and post-processing does not scan that directory
or infer metadata for the historical workbook.

---

## Switching DRL agents and experiment setup

The learning algorithm and base failure-rate ranges are controlled via
`config/configuration.py`, which serves as the main experiment configuration file.

Key parameters include:

- `model_summary = "dqn" | "ppo" | "ddpg"`  
  Selects the DRL algorithm used for decision making. The corresponding agent
  implementation is instantiated from the `agents/` directory.

- `SERVER_FAILURE_RATE_RANGE = (0.001, 0.005)` (1/s).
- `NUM_SERVERS = 8`, with 28 unordered pairs of distinct servers.
- `TASK_RELIABILITY_REQUIREMENT_LEVELS = (0.9, 0.99, 0.999, 0.9999)`;
  the generated 200-task profile contains 50 tasks at each level.

Input profiles are loaded once per run. Servers independently sample CPU capacity
uniformly on 1e9–3e9 cycles/s (1–3 GHz), upload rate on 10–30 MB/s, and base failure
rate on 0.001–0.005 s^-1; these values remain fixed after generation. CPU and upload
rate are not coupled. Transmission time is `Task_Size / Transmission_Rate` and
execution time is `Computation_Demand / Processing_Frequency`, both in seconds.
The profile builders sort records, convert field types and build dictionaries;
Python and the underlying libraries report malformed inputs directly.

## PPO/SMDP state representation

The state contains `5N + 1 = 41` features, in this exact group order, with each
server group sorted by Server_ID:

```text
[lambda_active / c_lambda, (L_i / r_n) / c_tx, (C_i / f_n) / c_exe,
 W_run / c_run, W_wait / c_wait, R_i_req]
```

Active failure intensity is read through `get_active_failure_rate`, and each
task-specific upload time uses `Task.get_transmission_rate`. Times are in seconds.
Running work is the remaining CPU service time at the decision's SimPy `env.now`.
Waiting work sums CPU service times of replicas whose uploads have finished and
which have not started CPU execution. Running, uploading and completed replicas
are excluded from waiting work. The current task is observed before its replicas
are launched.

Initial fixed scales are `STATE_FAILURE_RATE_SCALE = 0.005` s^-1,
`STATE_TRANSMISSION_TIME_SCALE_SEC = 1.0` s, and
`STATE_EXECUTION_TIME_SCALE_SEC = 3.0` s. Both CPU work groups share
`BACKLOG_TIME_SCALE_SEC = 4.0` s. Each group is divided directly by its scale;
values may exceed one and are not clipped. The reliability requirement retains
its raw value, such as 0.999, without tier encoding. These initial scales are not
performance-tuned at this stage.

Uploading replicas are not included in the state, so it may not fully describe
future CPU queue changes. This partial observability is accepted at this stage.
Historical 43-feature PPO checkpoints are incompatible with the new input size
and remain preserved; no automatic checkpoint migration is provided.

## Task arrival process

Tasks arrive according to a Poisson process with rate `lambda_a =
TASK_ARRIVAL_RATE`, measured in tasks/s. Therefore, each inter-arrival time is
exponentially distributed:

```text
Delta_T_k ~ Exp(lambda_a)
E[Delta_T_k] = 1 / lambda_a
```

The simulator samples each interval with
`self.arrival_rng.exponential(scale=1.0 / TASK_ARRIVAL_RATE)` and keeps the resulting
floating-point value. With the current `TASK_ARRIVAL_RATE = 0.5` tasks/s, the
mean inter-arrival time is 2.0 seconds.

## Event-driven PPO/SMDP semantics

PPO makes one decision when each task arrives. If task `k` arrives at time
`t_k`, the next decision interval is `delta_t_k = t_(k+1) - t_k`. PPO stores
transitions in task-arrival order:

```text
(s_k, a_k, r_k_interval, s_(k+1), delta_t_k, done_k)
```

`r_k_interval` is the sum of final task outcome rewards that become resolved
during `[t_k, t_(k+1))`. Individual task rewards are `-task_latency`; completion order does not reorder the PPO rollout. The
last arrival closes an explicit terminal interval after all pending replicas
are drained. The drain waits on task-level resolution events instead of using a
computation-demand value as a simulation-time polling timeout. For this final
interval, elapsed time is measured to the actual timestamp of the last resolved
task outcome, rather than to a later polling wake-up time.

For PPO, `gamma_ppo = 0.90` is a per-second discount base. Each transition uses
`gamma_k = gamma_ppo ** delta_t_k`, so a zero-length interval has discount 1.
Advantages use ordered variable-discount GAE:

```text
delta_k = r_k_interval + gamma_k * V(s_(k+1)) * (1 - done_k) - V(s_k)
A_k = delta_k + gamma_k * gae_lambda * (1 - done_k) * A_(k+1)
```

GAE is computed before PPO minibatch shuffling. DQN and DDPG retain their
existing transition and discount behavior.

## Analytical reliability and task completion

Spatial risk uses the Server_ID-sorted Haversine distance matrix in km. The
baseline correlation length is computed from the strict upper-triangle distances:
`ell_0 = median({d_nm: n < m}) / ln(2)`. For eight servers this uses 28 pairs,
and the correlation at the median distance is exactly 0.5. The length is derived
from the current server locations each episode, so topology changes require no
manual length update. The correlation matrix is `Sigma_nm = exp(-d_nm / ell_0)`
with diagonal entries equal to one.

One Gaussian field is sampled per episode and remains fixed until the episode
ends; the next episode draws a new field. Sampling uses `Z = L @ epsilon`, where
`L @ L.T = Sigma` and `epsilon` is standard normal. The existing RNG call remains
supported; `sample_spatial_risk_field(Sigma, standard_normal=epsilon)` also accepts
an explicit vector without consuming RNG state. The same vector with the identity
matrix produces the paired independent field for future comparisons. This change
to the sampling method changes risk traces relative to older runs, while keeping
fixed-seed reproducibility within the new version.

With beta 0.5, effective failure intensity is
`lambda_eff = lambda_0 * exp(beta * Z_phy - beta**2 / 2)`.
The correction term preserves `E[lambda_eff] = lambda_0` over Gaussian risk
realizations. Base failure rates are not overwritten. With spatial risk disabled,
the base failure intensity is used directly; beta zero also recovers base rates.

Decision-time replica reliability is `exp(-lambda_eff * C / f)`. Pair reliability
is `1 - (1 - R_A) * (1 - R_B)` under conditionally independent failures given
the current episode field, and requirement satisfaction compares this snapshot
against the task's raw requirement. Reliability does not mask actions or affect
physical completion: both replicas transmit and queue independently; the first
CPU finish completes the task, and the losing replica continues to completion.

---

## Agent interface contract

The simulation core depends only on a **minimal agent interface**, which ensures
that learning algorithms can be replaced without modifying the environment logic.

- **Discrete-action agents (DQN / PPO):**
  ```text
  select_action(state, epsilon) -> int
  ```

- **Continuous scoring agents (DDPG):**
  ```text
  policy(state) -> score_vector
  ```

The final action selection (e.g., `argmax` over scores) is handled inside the
simulation core.

---

## Adding a new DRL agent

New DRL algorithms can be integrated in an incremental and low-risk manner:

1. Create a new agent implementation inside the `agents/` directory
   (e.g., `agents/a2c_agent.py`).

2. Implement the required action-selection interface expected by the simulation core
   (`select_action` or `policy`, depending on the action space).

3. Register the new agent in the model construction logic
   (e.g., within `build_model()` in `Project_main.py`).

4. Set the corresponding value of `model_summary` in `config/configuration.py`.

This design allows new learning methods to be added without altering the
simulation environment or episode loop.

---

## Notes

- All scripts should be executed from the **project root**.
- Frozen input profiles and saved experiment outputs are tracked in the repository.
  Regenerate or overwrite them only when explicitly required.
- Root-level launcher scripts are provided to avoid Python import issues when running
  utility modules.
