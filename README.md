# DRL-Based Reliable Offloading Simulator

This repository provides a **generic and modular simulator** for reliability-aware
task offloading in **distributed Edge/Cloud computing systems**.
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

### 3) Post-process results (optional)

```bash
python post_process.py
```

This step augments result workbooks with additional analysis sheets and may generate
a global aggregated file (e.g., `Final_Result_All.xlsx`) inside the `results/` directory.

---

## Switching DRL agents and experiment setup

The learning algorithm and base failure-rate ranges are controlled via
`config/configuration.py`, which serves as the main experiment configuration file.

Key parameters include:

- `model_summary = "dqn" | "ppo" | "ddpg"`  
  Selects the DRL algorithm used for decision making. The corresponding agent
  implementation is instantiated from the `agents/` directory.

- `EDGE_FAILURE_RATE_RANGE = (0.001, 0.005)` (1/s).
- `CLOUD_FAILURE_RATE_RANGE = (0.0001, 0.001)` (1/s).

Pre-processing generates `data/server_info.xlsx` with one `Servers` sheet and
`data/task_parameters.xlsx`. Each server's base failure rate is sampled uniformly
from its type's range, then remains fixed throughout execution. Episodes load the
same server parameters. Regenerate inputs after changing the ranges.

Primary and backup use their respective server's rate with execution time
`computation_demand / processing_frequency`: failure probability is
`1 - exp(-failure_rate * service_time)`, followed by uniform random sampling.
Queue length does not adjust this rate. The default algorithm remains PPO.
The result workbook's `Servers` sheet records the rates used in the simulation.

## PPO/MDP state representation

Each server contributes three normalized state features:

1. its observable estimated base transient fault arrival rate `lambda_n`;
2. its processing frequency;
3. its current backlog service time.

The backlog time of server `n` is the remaining service time of its currently
executing replica plus the service times of all replicas waiting in its CPU queue:

```text
B_n(t) = R_n(t) + sum(C_q / f_n)
```

It is normalized with the fixed scale `BACKLOG_TIME_SCALE_SEC = 4.0`:

```text
normalized_backlog_time = B_n(t) / (B_n(t) + 4.0)
```

The current task contributes normalized task size and computation demand. The
state is ordered as `[failure_rates, frequencies, backlog_times, task_size, demand]`,
so its dimension remains `3N + 2`. With the current eight servers, PPO receives
26 state features. Backlog time represents CPU service backlog only; network
input/output delay is not included.

## Event-driven PPO/SMDP semantics

PPO makes one decision when each task arrives. If task `k` arrives at time
`t_k`, the next decision interval is `delta_t_k = t_(k+1) - t_k`. PPO stores
transitions in task-arrival order:

```text
(s_k, a_k, r_k_interval, s_(k+1), delta_t_k, done_k)
```

`r_k_interval` is the sum of final task outcome rewards that become resolved
during `[t_k, t_(k+1))`. Individual task rewards still use the unchanged
`calcReward` formula; completion order does not reorder the PPO rollout. The
last arrival closes an explicit terminal interval after all pending replicas
are drained.

For PPO, `gamma_ppo = 0.90` is a per-second discount base. Each transition uses
`gamma_k = gamma_ppo ** delta_t_k`, so a zero-length interval has discount 1.
Advantages use ordered variable-discount GAE:

```text
delta_k = r_k_interval + gamma_k * V(s_(k+1)) * (1 - done_k) - V(s_k)
A_k = delta_k + gamma_k * gae_lambda * (1 - done_k) * A_(k+1)
```

GAE is computed before PPO minibatch shuffling. DQN and DDPG retain their
existing transition and discount behavior.

## Transient server-fault model

Each Edge or Cloud server has a fixed transient fault arrival rate, `lambda_n`,
measured in `1/s`. The ranges are configured by
`EDGE_FAILURE_RATE_RANGE` and `CLOUD_FAILURE_RATE_RANGE` and are sampled once
when `data/server_info.xlsx` is generated.

For a task replica with computation demand `C_i` running on a server with
processing frequency `f_n`, the execution interval is:

```text
t_i,n = C_i / f_n
```

The probability that at least one transient server fault occurs during that
interval is:

```text
P(replica failure) = 1 - exp(-lambda_n * t_i,n)
```

The simulator samples this probability independently for the primary and backup
replicas. A `failure` status means that the current replica execution failed due
to a transient fault. The server does not enter a permanent DOWN state, and the
fault recovery interval is treated as negligible. Therefore later tasks and a
retry on the same server remain allowed. A task-level failure means that all
required replicas failed. Correlated or common-cause faults are not modeled.

---

## Agent interface contract

The simulation core depends only on a **minimal agent interface**, which ensures
that learning algorithms can be replaced without modifying the environment logic.

- **Discrete-action agents (DQN / PPO):**
  ```text
  select_action(state) -> int
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
- Input data (`data/`) and experiment outputs (`results/`) are generated automatically
  and are not expected to be present in the repository.
- Root-level launcher scripts are provided to avoid Python import issues when running
  utility modules.
