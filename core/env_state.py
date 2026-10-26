"""Server CPU metadata, task objects and episode spatial context."""

import numpy as np
from itertools import combinations
from config.params import params, effective_spatial_risk_beta
from core.spatial_risk import (
    build_distance_matrix, build_spatial_correlation_matrix,
    compute_baseline_correlation_length_km, compute_marginal_failure_probability,
    compute_pair_reliability_gh,
)


def predict_server_completion_times(state):
    """Predict completion from observable uploads and CPU work, in server-ID order."""
    n = params.NUM_SERVERS
    fmin, fmax = params.SERVER_PROCESSING_FREQ_RANGE
    rmin, rmax = params.SERVER_TRANSMISSION_RATE_RANGE
    lmin, lmax = params.TASK_SIZE_RANGE
    cmin, cmax = params.TASK_DEMAND_RANGE
    state = np.asarray(state, dtype=np.float64)
    frequency = fmin + state[n:2*n] * (fmax-fmin)
    rate = rmin + state[2*n:3*n] * (rmax-rmin)
    work = state[3*n:4*n] + state[4*n:5*n]
    size = lmin + state[5*n] * (lmax-lmin)
    demand = cmin + state[5*n+1] * (cmax-cmin)
    return np.maximum(size/rate, work) + demand/frequency


def select_latency_greedy_action(state, pairs, action_mask=None):
    """Minimize observable predicted first completion, sum, then action index."""
    predicted = predict_server_completion_times(state)
    indices = (np.flatnonzero(action_mask) if action_mask is not None and np.any(action_mask)
               else range(len(pairs)))
    server_index = {sid: index for index, sid in enumerate(sorted({sid for pair in pairs for sid in pair}))}
    return int(min(indices, key=lambda i: (
        min(predicted[server_index[pairs[i][0]]], predicted[server_index[pairs[i][1]]]),
        predicted[server_index[pairs[i][0]]] + predicted[server_index[pairs[i][1]]], i)))


class EnvironmentState:
    def __init__(self, *, failure_rate_scale=1.0, reliability_model="correlated"):
        if not np.isfinite(failure_rate_scale) or failure_rate_scale < 0:
            raise ValueError("failure_rate_scale must be finite and nonnegative")
        self.failure_rate_scale = float(failure_rate_scale)
        if reliability_model not in ("correlated", "independent"):
            raise ValueError("reliability_model must be correlated or independent")
        self.reliability_model = reliability_model
        self.servers = {}  # Server objects and CPU backlog metadata.
        self.tasks = {}
        self.spatial_risk_server_ids = None
        self.spatial_distance_matrix = None
        self.spatial_correlation_matrix = None
        self.server_pairs = None

    def initialize_spatial_reliability(self):
        """Use the unknown field's covariance; never observe a sampled realization."""
        servers = [info['server_object'] for info in self.servers.values()]
        self.spatial_risk_server_ids, self.spatial_distance_matrix = build_distance_matrix(servers)
        self.spatial_correlation_matrix = build_spatial_correlation_matrix(
            self.spatial_distance_matrix,
            compute_baseline_correlation_length_km(self.spatial_distance_matrix),
        )
        self.server_pairs = list(combinations(self.spatial_risk_server_ids, 2))

    def compute_task_reliabilities(self, task):
        """Cache reliabilities; disabled spatial risk uses beta=0 (base rates).

        Geometry remains available for diagnostics, but cannot affect reliability
        when the latent risk multiplier is identically one.
        """
        if task.reliability_snapshot is None:
            if self.server_pairs is None:
                self.initialize_spatial_reliability()
            ids = self.spatial_risk_server_ids
            servers = [self.get_server_by_id(sid) for sid in ids]
            frequencies = np.array([server.processing_frequency for server in servers])
            rates = np.array([server.base_failure_rate for server in servers]) * self.failure_rate_scale
            pairs = np.array(list(combinations(range(len(ids)), 2)))
            marginals = compute_marginal_failure_probability(
                task.computation_demand, frequencies, rates, effective_spatial_risk_beta(),
            )
            corr, ind = compute_pair_reliability_gh(
                task.computation_demand, frequencies[pairs], rates[pairs],
                self.spatial_correlation_matrix[pairs[:, 0], pairs[:, 1]], effective_spatial_risk_beta(),
            )
            task.reliability_snapshot = (1 - marginals, corr, ind)
        return task.reliability_snapshot

    def get_safe_action_mask(self, task):
        if task.safe_mask is None:
            index = 1 if self.reliability_model == "correlated" else 2
            task.safe_mask = self.compute_task_reliabilities(task)[index] >= task.reliability_requirement
        return task.safe_mask

    def add_server_and_init_environment(self, server_object):
        """Add a server object to the environment state."""
        server_id = server_object.server_id
        self.servers[server_id] = {
            'server_object': server_object,
            'waiting_replicas': [],
            'running_replica': None
        }

    def register_waiting_replica(self, server_id, task, replica_name, service_time):
        """Register a replica waiting for CPU service on a server."""
        server_info = self.servers[server_id]
        server_info['waiting_replicas'].append({
            'task': task,
            'replica': replica_name,
            'service_time': float(service_time),
        })

    def start_replica_execution(
        self, server_id, task, replica_name, service_time, service_start_time
    ):
        """Move a waiting replica to running metadata after CPU acquisition."""
        server_info = self.servers[server_id]
        identity = (task.id, replica_name)
        waiting = server_info['waiting_replicas']
        match = next((item for item in waiting
                      if (item['task'].id, item['replica']) == identity))
        waiting.remove(match)
        server_info['running_replica'] = {
            'task': task,
            'replica': replica_name,
            'service_time': float(service_time),
            'service_start_time': float(service_start_time),
        }

    def complete_replica_execution(self, server_id):
        """Clear running metadata when CPU service completes."""
        self.servers[server_id]['running_replica'] = None

    def get_server_backlog_components(self, server_id, current_time):
        """Return running remaining and queued service times in seconds."""
        server_info = self.servers[server_id]
        running = server_info['running_replica']
        waiting = server_info['waiting_replicas']
        running_backlog = 0.0
        if running is not None:
            elapsed = current_time - running['service_start_time']
            running_backlog = max(running['service_time'] - elapsed, 0.0)
        waiting_backlog = sum(replica['service_time'] for replica in waiting)

        return float(running_backlog), float(waiting_backlog)

    def get_server_by_id(self, server_id):
        """Get a server object by its ID."""
        return self.servers[server_id]['server_object']

    def compute_replica_reliability(self, task, server):
        reliabilities = self.compute_task_reliabilities(task)[0]
        return float(reliabilities[self.spatial_risk_server_ids.index(server.server_id)])

    def compute_pair_reliability(self, task, server_A, server_B):
        reliabilities = self.compute_task_reliabilities(task)[1]
        pair = tuple(sorted((server_A.server_id, server_B.server_id)))
        return float(reliabilities[self.server_pairs.index(pair)])

    def is_reliability_requirement_satisfied(self, task, server_A, server_B):
        """Compare theoretical pair reliability with the task requirement."""
        return bool(
            self.compute_pair_reliability(task, server_A, server_B)
            >= task.reliability_requirement
        )

    def add_task(self, task_object):
        """Add a task object to the environment state."""
        task_id = task_object.id
        self.tasks[task_id] = task_object

    def get_state(self, task):
        """Observe server parameters, separate CPU work and current task attributes."""
        server_ids = sorted(self.servers)
        failure_rates, frequencies, transmission_rates = [], [], []
        running_backlogs, waiting_backlogs = [], []
        for server_id in server_ids:
            server = self.get_server_by_id(server_id)
            running, waiting = self.get_server_backlog_components(server_id, task.env.now)
            failure_rates.append(server.base_failure_rate * self.failure_rate_scale)
            frequencies.append(server.processing_frequency)
            transmission_rates.append(server.transmission_rate)
            running_backlogs.append(running)
            waiting_backlogs.append(waiting)

        frequency_min, frequency_max = params.SERVER_PROCESSING_FREQ_RANGE
        rate_min, rate_max = params.SERVER_TRANSMISSION_RATE_RANGE
        size_min, size_max = params.TASK_SIZE_RANGE
        demand_min, demand_max = params.TASK_DEMAND_RANGE
        if task.deadline is None or not np.isfinite(task.deadline) or task.deadline <= 0:
            raise ValueError("State requires a positive finite task deadline in seconds")
        state = np.concatenate((
            np.asarray(failure_rates) / params.STATE_FAILURE_RATE_SCALE,
            (np.asarray(frequencies) - frequency_min) / (frequency_max - frequency_min),
            (np.asarray(transmission_rates) - rate_min) / (rate_max - rate_min),
            np.asarray(running_backlogs),  # Raw remaining CPU time in seconds.
            np.asarray(waiting_backlogs),  # Raw queued CPU work in seconds.
            [(task.task_size - size_min) / (size_max - size_min),
             (task.computation_demand - demand_min) / (demand_max - demand_min),
             task.deadline],  # Raw relative deadline in seconds.
        ))
        return state.astype(np.float32)
