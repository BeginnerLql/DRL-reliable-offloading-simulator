# State Information Inventory

This inventory is written before training any model for the state sufficiency audit. It distinguishes the vector passed to the Actor network from simulator information used to construct the action distribution.

## A. Information in the current 35-dimensional observation

The current system has eight Edge servers. `EnvironmentState.get_state()` concatenates four ordered per-server blocks and three current-task fields:

| Indices | Feature | Decision-time meaning |
|---|---|---|
| 0–7 | Nominal/base failure rate | Server base hazard, normalized to the configured range. It does not reveal the episode's spatial realization or effective hazard. |
| 8–15 | Processing frequency | Per-server processing frequency, normalized to its configured range. |
| 16–23 | CPU backlog time | Running replica's remaining service plus the service times of replicas already registered in that server's CPU waiting list. It is normalized as `B / (B + BACKLOG_TIME_SCALE_SEC)`. |
| 24–31 | Uplink rate | Per-server uplink Mbps, normalized to the configured range. |
| 32 | Input data size | Current task's input size, normalized to the configured range. |
| 33 | Computation demand | Current task's computation demand, normalized to the configured range. |
| 34 | Reliability requirement | Current task's requirement in the normalized number-of-nines domain. |

Thus the dimension is `4N + 3 = 4*8 + 3 = 35`. The ordering follows `core/env_state.py::get_state()`.

## B. Information used by action selection beyond the Actor MLP input

`ReliabilityMaskedPairPPOAgent.prepare_action()` evaluates the current task against all 28 legal server pairs using the production `production_reliability_vector()` logic. It stores the full pair reliability vector, the safe mask, the effective mask, whether the safe set is empty, the best achievable reliability, and the requirement. The production pair-correlation vector is also available to the pair-scoring Actor architecture.

`select_action()` feeds only the 35-dimensional state to the Actor network. It then applies the saved effective mask to the Actor logits before sampling. Therefore the action distribution is conditional on information that is not an Actor MLP input:

`pi(a | s_35, effective_mask)`.

The effective mask equals the reliability-safe mask when any safe pair exists. With an empty safe set, it is the maximum-reliability fallback set. The safe mask and effective mask can therefore differ on fallback decisions. The full pair reliability vector can vary across episodes because the episode-effective hazards vary spatially.

These are action-selection inputs, not part of the Actor's 35-dimensional feature vector.

## C. Decision-time simulator information omitted from the observation

### In-flight replicas

A selected replica first runs `Task.primary()` or `Task.backup()`, computes its upload duration, and waits for that upload. Only after the upload timeout does it call `EnvironmentState.register_waiting_replica()`. The 35-dimensional backlog is computed from `running_replica` and `waiting_replicas`, so a replica that is uploading has already been committed to a destination but is not yet reflected in CPU backlog.

The environment does not keep a first-class in-flight upload registry. This audit adds a diagnostic-only wrapper around the existing upload-duration calculation and queue-registration callback. It records the upload start, expected duration, destination, computation demand, known CPU service requirement, and the callback time at which the replica enters the CPU waiting list. The wrapper forwards the original return values and SimPy events unchanged. No production source is edited.

At each decision, in-flight counts, total future CPU service workload, and minimum/mean/maximum remaining upload time are computed only from uploads that have started but have not yet registered with the CPU queue.

### Waiting CPU replicas

The simulator stores each waiting replica's task, primary/backup label, and service time. The observation reduces this queue to the sum of service times; it omits queue length, task composition, and individual service times.

### Running replica metadata

The simulator retains the running task, replica label, service time, and service start time. The observation uses only the calculated remaining service time. It omits the task identity and replica metadata.

### Episode-effective spatial reliability context

The episode spatial-risk field, effective per-server failure rates, and correlation context are available in `EnvironmentState`. The Actor observation contains nominal/base hazards only. The production reliability vector and resulting masks incorporate the effective episode context and are available to action selection, but not as inputs to the Actor MLP.

### Outstanding population

At a decision, `EnvironmentState.tasks` contains task objects that have not completed lifecycle cleanup. Per-server waiting and running registries expose CPU replicas. The diagnostic upload registry exposes replicas that have started uploading but have not registered with CPU. Together, the upload registry and CPU registries identify the unfinished replicas for this fixed two-replica execution model; the current 35-dimensional observation does not encode these counts or their composition. `MainLoop.pendingList` is a separate task-outcome bookkeeping list and is not itself a replica-state representation.

## No-future-information rule

All added features are snapshots at the existing decision time. They use only already-started uploads, known destination and task parameters, current CPU registries, current effective reliability context, and the current task. They do not use later arrivals, future random draws, outcomes, delays, rewards, or counterfactual results. The horizon-free MC return remains the unchanged target and is never an input feature.
