# DRL-Based Reliable Offloading Simulator

A SimPy simulator for reliability-aware task offloading with DQN, PPO and DDPG.
The current configuration uses eight Edge servers with different CPU frequencies,
uplink rates and baseline fault rates. Each action selects an unordered pair of
distinct servers; both replicas run in parallel.

See [the code map and cleanup review](docs/CODE_REVIEW.md) for module relationships
and [Context Masked Pair PPO v2](docs/CONTEXT_MASKED_PAIR_PPO.md) for the versioned
actor and reward-credit experiments.

## Layout

| Path | Responsibility |
| --- | --- |
| `Project_main.py` | Build a configured agent, run episodes and export results |
| `pre_process.py` / `post_process.py` | Input generation / Excel aggregation launchers |
| `agents/` | DQN, DDPG, PPO and separate masked/context/auxiliary-critic variants |
| `config/` | Experiment defaults, runtime parameter snapshot and shared paths |
| `core/` | SimPy scheduling, server queues, task lifecycle and spatial hazards |
| `diagnostics/` | Experiment runners, frozen-policy audits, aggregation and reports |
| `io_utils/` | Shared log schema, Excel persistence and post-processing |
| `tests/` | Simulator, agent, diagnostic and data-tool regression tests |
| `tools/` | Data preparation, offline analyses and experiment entry points |
| `data/` | Input workbooks and prepared server topology |
| `results/` / `diagnostics/results/` | Simulation outputs / diagnostic artifacts |

## Setup and execution

Use Python 3.10 or newer and install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. If the input workbooks need regeneration:

```bash
python pre_process.py
```

This writes `data/server_info.xlsx` and `data/task_parameters.xlsx`. Server
coordinates come from `data/eua_melbourne_cbd_site_order.csv`; the generator
expects that prepared topology to exist. Input generation replaces the workbooks,
so retain the original inputs when reproducing a saved experiment.

```bash
python Project_main.py
python post_process.py
```

The simulation writes `results/fixed_rate_results/<model>_results.xlsx`.
Post-processing adds distribution sheets and creates
`results/Final_Result_All.xlsx`.

For the existing tests, install the test runner separately:

```bash
python -m pip install pytest pytest-subtests
python -m pytest -q
```

## Configuration and agents

Edit `config/configuration.py`. `config/params.py` copies those defaults at import
time and derives state/action dimensions; experiment runners can temporarily
patch the runtime snapshot without changing the default configuration.

`model_summary` selects `dqn`, `ppo` or `ddpg`. For the ordinary PPO entry point,
`PPO_ACTOR_MODE` selects `flat` or `pair_scoring` (the current default). The pair
actor applies one shared scorer to symmetric pair features; its input includes
static pair spatial correlation. `agents/networks.py` shares hidden-layer
construction while retaining checkpoint layer names and initialization order.

Masked PPO and Context Masked Pair PPO v2 have separate runners. They are not
selected by setting a new value of `model_summary`. For example:

```bash
python tools/run_context_masked_pair_ppo.py \
  --train-episodes 3 --eval-episodes 3 \
  --output-dir diagnostics/results/my_context_v2_smoke
```

Use a new output directory. This is a functional smoke run, not a multi-seed
performance benchmark. See the v2 document for checkpoint environment checks,
deployment and ablation options.

## Observations and actions

The observation has four server blocks in sorted `Server_ID` order, followed by
three task features:

```text
[base_failure_rates, processing_frequencies, CPU_backlogs, uplink_rates,
 input_data_size, computation_demand, reliability_requirement]
```

The dimension is `4N + 3`, or **35** for eight servers. The action count is
`N(N - 1) / 2`, or **28**, ordered by `combinations(range(1, N + 1), 2)`.

CPU backlog is remaining running service plus all waiting service times. It is
normalized as `B / (B + BACKLOG_TIME_SCALE_SEC)`, with a default scale of 4 seconds.
Replicas still uploading are not part of this physical CPU backlog. Reliability
requirements are encoded through `-log10(1 - R_req)`, then scaled and clipped.
The observation exposes base hazards, not the episode's realized spatial field;
the masked policy separately uses production reliability to construct its support.

## Reliability and task lifecycle

The generated baseline fault rate depends on processing frequency:

```text
lambda_base = LAMBDA_REF * 10 ** (
    FAILURE_RATE_OMEGA * (1 - f / FAILURE_RATE_FMAX)
    / (1 - FAILURE_RATE_FMIN / FAILURE_RATE_FMAX)
)
```

When spatial risk is enabled, each episode samples `Z ~ N(0, R)`, where
`R[j,k] = exp(-distance[j,k] / correlation_length)`, and uses
`lambda_eff[j] = lambda_base[j] * exp(beta_p * Z[j])`. At `Z = 0`, the base rate
is recovered; no lognormal mean correction is applied. Spatial-off runs use the
base rates directly.

For a replica with demand `C` on CPU frequency `f`, the analytical failure
probability is `-expm1(-lambda_eff * C / f)`. Conditional on the episode hazards,
the selected pair uses the product of its two failure probabilities:

```text
joint_failure = primary_failure * backup_failure
execution_reliability = 1 - joint_failure
reliability_satisfied = execution_reliability >= task_requirement
```

Runtime execution does not draw Bernoulli replica failures. Replica completion
status is nominal; task success/failure in reward and summary tables is the
analytical reliability-threshold outcome. The offline Monte Carlo correlation
analyses in `tools/` are separate diagnostics.

Each replica uploads its payload (`8 * MB / Mbps` seconds), requests its server's
CPU at the same priority, and executes. The first completed replica resolves the
task reward. The second continues to completion and releases its CPU. A task is
removed only after reward bookkeeping and both replica completions. Therefore,
task-resolution logs may lack the slower replica's finish timestamp; the separate
`ReplicaCompletions` sheet records actual completions of both replicas.

## Arrivals, rewards and PPO credit

Tasks follow a Poisson arrival process with rate `TASK_ARRIVAL_RATE` (default
0.5 tasks/s). `MainLoop` uses a private NumPy generator for arrivals and another
for spatial risk. PPO minibatch shuffling also has its own generator.

Reward combines the delay-based success/failure term in `MainLoop.calcReward`
with a logarithmic failure-budget violation penalty. The task requirement,
realized pair reliability, reward components and action metadata are logged.

Ordinary PPO and legacy masked PPO collect arrival-ordered transitions:

```text
(s_k, a_k, reward_of_origin_task_k, s_(k+1), delta_t_k, done_k)
```

A reward can resolve before or after its transition shell is stored; task IDs
ensure it is assigned once to the originating task. `delta_t` is time to the next
arrival, or from the final arrival to the latest resolved task outcome. PPO
updates once per episode using `gamma_k = gamma_ppo ** delta_t_k` and ordered GAE
before minibatch shuffling. The simulator still drains the slower replicas.

Context Masked Pair PPO v2 defaults to **event-interval** credit: outcomes are
assigned to the interval in which they actually occurred, including outcomes of
earlier tasks, with within-interval discounting. It also defaults to a context
actor and independent actor/critic gradient clipping. These are intentional
versioned differences; legacy credit and clipping remain available as ablations.
DQN and DDPG retain their own transition and training semantics.

## Agent integration

`MainLoop` calls `select_action(state, epsilon)` for DQN/PPO, or `policy(state)`
and `addNoise(...)` for DDPG scores. The latter are decoded by argmax.

Training interfaces also differ: DQN accepts transition tuples and learns online;
DDPG uses its internal replay buffer and target updates; PPO accepts task-bound
transitions, later reward assignment and episode-end `train_step()`. Specialized
policies can implement `prepare_action(...)` and `record_task_outcome(...)` hooks.
Register new agents in the model builder or provide a dedicated runner that
respects these contracts.
