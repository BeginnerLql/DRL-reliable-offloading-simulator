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

---

### 2) Run the simulation

```bash
python Project_main.py
```

Simulation results are automatically written to:

```
results/fixed_rate_results/<model>_results.xlsx
```

---

### 3) Analyze the saved run

```bash
python diagnostics/analyze_spatial_four_tier_run.py
python diagnostics/plot_spatial_four_tier_run.py
```

The diagnostics reconstruct reliability independently and check the saved task
outcomes. The plotting script writes three PNG figures to
`diagnostics/results/spatial_four_tier_run/`.

### 4) Summarize saved model results

```bash
python post_process.py
```

This reads current-schema workbooks from `results/fixed_rate_results/*.xlsx`
and writes a separate `results/Final_Result_All.xlsx`. Input workbooks are
unchanged. The output contains:

- `ModelSummary`: episode/task counts, mean episode reward, mean episode latency,
  overall RSR, mean task latency and P95 task latency for each model.
- `EpisodeMetrics`: episode reward, mean task latency, task count, RSR and P95
  task latency, identified by model and source file.
- `PairSelection`: counts and shares for all unordered server pairs across all
  task outcomes, including pairs with zero selections.

RSR is the mean of the saved `requirement_satisfied` boolean values; latency is
measured in seconds. Multiple model workbooks are summarized in the same tables.

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

Input profiles are loaded once per run. Servers have fixed processing frequencies,
base failure rates and transmission rates (20/24/28/32/36/40/45/50 MB/s).
The profile builders sort records, convert field types and build dictionaries;
Python and the underlying libraries report malformed inputs directly.

## PPO/SMDP state representation

Each server contributes its effective failure intensity, normalized processing
frequency, normalized transmission rate, normalized remaining CPU service time,
and normalized queued CPU service time. Running and waiting backlogs are each
normalized as `B / (B + BACKLOG_TIME_SCALE_SEC)` with scale 4.0 seconds.

The task contributes normalized size, normalized computation demand and its
reliability tier encoded as 0, 1/3, 2/3 or 1. The state is ordered as
`[failure_rates, frequencies, transmission_rates, running_backlogs,
waiting_backlogs, task_size, demand, reliability_tier]`: `5N + 3 = 43` features.
CPU backlogs exclude transmission time.

## Task arrival process

Tasks arrive according to a Poisson process with rate `lambda_a =
TASK_ARRIVAL_RATE`, measured in tasks/s. Therefore, each inter-arrival time is
exponentially distributed:

```text
Delta_T_k ~ Exp(lambda_a)
E[Delta_T_k] = 1 / lambda_a
```

The simulator samples each interval with
`np.random.exponential(scale=1.0 / TASK_ARRIVAL_RATE)` and keeps the resulting
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

Spatial risk samples one Gaussian field per episode with physical correlation
length 0.5 km and beta 0.5. Effective failure intensity is
`lambda_eff = lambda_0 * exp(beta * Z_phy - beta**2 / 2)`.
With spatial risk disabled, the base failure intensity is used directly.

Decision-time replica reliability is `exp(-lambda_eff * C / f)`. Pair reliability
is `1 - (1 - R_A) * (1 - R_B)`, and requirement satisfaction compares this snapshot
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
