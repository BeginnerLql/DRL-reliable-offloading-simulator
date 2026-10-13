"""Server CPU metadata, task objects and episode spatial context."""

import numpy as np
from config.params import params


class EnvironmentState:
    def __init__(self):
        self.servers = {}  # Server objects and CPU backlog metadata.
        self.tasks = {}
        self.spatial_risk_server_ids = None
        self.spatial_distance_matrix = None
        self.spatial_correlation_matrix = None
        self.spatial_risk_field = None
        self.effective_failure_rates = None

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

    def set_episode_spatial_risk_context(
        self,
        server_ids,
        distance_matrix,
        correlation_matrix,
        spatial_risk_field,
        effective_failure_rates,
    ):
        """Store one immutable-in-practice episode spatial-risk realization."""
        self.spatial_risk_server_ids = list(server_ids)
        self.spatial_distance_matrix = np.array(distance_matrix, dtype=float, copy=True)
        self.spatial_correlation_matrix = np.array(correlation_matrix, dtype=float, copy=True)
        self.spatial_risk_field = np.array(spatial_risk_field, dtype=float, copy=True)
        self.effective_failure_rates = {
            server_id: float(effective_failure_rates[index])
            for index, server_id in enumerate(server_ids)
        }

    def get_active_failure_rate(self, server_id):
        """Return episode-effective or baseline transient fault arrival rate."""
        server_object = self.get_server_by_id(server_id)
        if self.effective_failure_rates is None:
            return server_object.base_failure_rate
        return self.effective_failure_rates[server_id]

    def compute_replica_reliability(self, task, server):
        """Return exp(-lambda_active * C/f); C/f is CPU execution seconds."""
        failure_rate = self.get_active_failure_rate(server.server_id)
        execution_time = task.computation_demand / server.processing_frequency
        return float(np.exp(-failure_rate * execution_time))

    def compute_pair_reliability(self, task, server_A, server_B):
        """Assume independent replica failures conditional on the episode field."""
        reliability_A = self.compute_replica_reliability(task, server_A)
        reliability_B = self.compute_replica_reliability(task, server_B)
        return float(
            1.0 - (1.0 - reliability_A) * (1.0 - reliability_B)
        )

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
            failure_rates.append(self.get_active_failure_rate(server_id))
            frequencies.append(server.processing_frequency)
            transmission_rates.append(server.transmission_rate)
            running_backlogs.append(running)
            waiting_backlogs.append(waiting)

        frequency_min, frequency_max = params.SERVER_PROCESSING_FREQ_RANGE
        rate_min, rate_max = params.SERVER_TRANSMISSION_RATE_RANGE
        size_min, size_max = params.TASK_SIZE_RANGE
        demand_min, demand_max = params.TASK_DEMAND_RANGE
        state = np.concatenate((
            np.asarray(failure_rates) / params.STATE_FAILURE_RATE_SCALE,
            (np.asarray(frequencies) - frequency_min) / (frequency_max - frequency_min),
            (np.asarray(transmission_rates) - rate_min) / (rate_max - rate_min),
            np.asarray(running_backlogs),  # Raw remaining CPU time in seconds.
            np.asarray(waiting_backlogs),  # Raw queued CPU work in seconds.
            [(task.task_size - size_min) / (size_max - size_min),
             (task.computation_demand - demand_min) / (demand_max - demand_min)],
        ))
        return state.astype(np.float32)
