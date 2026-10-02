"""Server transmission/CPU metadata, task objects and episode spatial context."""

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
            'transmitting_replicas': [],
            'waiting_replicas': [],
            'running_replica': None
        }

    def register_transmitting_replica(
        self, server_id, task, replica_name, tx_finish_time, service_time
    ):
        self.servers[server_id]['transmitting_replicas'].append({
            'task': task,
            'replica': replica_name,
            'tx_finish_time': float(tx_finish_time),
            'service_time': float(service_time),
        })

    def finish_replica_transmission(self, server_id, task, replica_name):
        transmitting = self.servers[server_id]['transmitting_replicas']
        identity = (task.id, replica_name)
        match = next(item for item in transmitting
                     if (item['task'].id, item['replica']) == identity)
        transmitting.remove(match)

    def get_server_transmission_components(self, server_id, current_time):
        """Return next CPU-queue arrival delay and future CPU work in seconds."""
        transmitting = self.servers[server_id]['transmitting_replicas']
        next_remaining = min(
            (max(item['tx_finish_time'] - current_time, 0.0)
             for item in transmitting),
            default=0.0,
        )
        future_workload = sum(item['service_time'] for item in transmitting)
        return float(next_remaining), float(future_workload)

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
        """Compute decision-time reliability over CPU execution only."""
        failure_rate = self.get_active_failure_rate(server.server_id)
        execution_time = task.computation_demand / server.processing_frequency
        return float(np.exp(-failure_rate * execution_time))

    def compute_pair_reliability(self, task, server_A, server_B):
        """Compute pair reliability under independent replica failures."""
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

    def reset(self):
        """Reset the environment state."""
        self.servers = {}
        self.tasks = {}
        self.spatial_risk_server_ids = None
        self.spatial_distance_matrix = None
        self.spatial_correlation_matrix = None
        self.spatial_risk_field = None
        self.effective_failure_rates = None

    def normalize(self, val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    def normalize_reliability_requirement(self, reliability_requirement):
        levels = params.TASK_RELIABILITY_REQUIREMENT_LEVELS
        return levels.index(reliability_requirement) / (len(levels) - 1)

    def get_state(self, task):
        server_ids = sorted(self.servers)

        failure_rates = []
        frequencies = []
        transmission_rates = []
        running_backlogs = []
        waiting_backlogs = []
        next_transmission_times = []
        future_cpu_workloads = []

        frequency_min, frequency_max = params.SERVER_PROCESSING_FREQ_RANGE
        transmission_rate_min = min(params.SERVER_TRANSMISSION_RATES)
        transmission_rate_max = max(params.SERVER_TRANSMISSION_RATES)

        for server_id in server_ids:
            server_info = self.servers[server_id]
            server_object = server_info['server_object']
            failure_rate = float(self.get_active_failure_rate(server_id))
            frequency = float(server_object.processing_frequency)
            transmission_rate = float(server_object.transmission_rate)
            running_backlog, waiting_backlog = self.get_server_backlog_components(
                server_id, task.env.now
            )
            next_transmission_time, future_cpu_workload = (
                self.get_server_transmission_components(server_id, task.env.now)
            )
            failure_rates.append(failure_rate)
            frequencies.append(frequency)
            transmission_rates.append(transmission_rate)
            running_backlogs.append(running_backlog)
            waiting_backlogs.append(waiting_backlog)
            next_transmission_times.append(next_transmission_time)
            future_cpu_workloads.append(future_cpu_workload)

        scale = params.BACKLOG_TIME_SCALE_SEC
        task_size = float(task.task_size)
        task_size_min, task_size_max = params.TASK_SIZE_RANGE
        computation_demand = float(task.computation_demand)
        demand_min, demand_max = params.Low_demand, params.High_demand
        reliability_requirement = float(task.reliability_requirement)
        normalized_reliability_requirement = self.normalize_reliability_requirement(
            reliability_requirement
        )

        normalized_frequencies = self.normalize(
            np.asarray(frequencies), frequency_min, frequency_max
        )
        normalized_transmission_rates = self.normalize(
            np.asarray(transmission_rates),
            transmission_rate_min,
            transmission_rate_max,
        )
        normalized_running_backlogs = np.asarray(running_backlogs) / (
            np.asarray(running_backlogs) + scale
        )
        normalized_waiting_backlogs = np.asarray(waiting_backlogs) / (
            np.asarray(waiting_backlogs) + scale
        )
        normalized_task_size = self.normalize(
            task_size, task_size_min, task_size_max
        )
        normalized_computation_demand = self.normalize(
            computation_demand, demand_min, demand_max
        )
        normalized_next_transmission_times = np.asarray(next_transmission_times) / (
            np.asarray(next_transmission_times) + scale
        )
        normalized_future_cpu_workloads = np.asarray(future_cpu_workloads) / (
            np.asarray(future_cpu_workloads) + scale
        )

        state = np.concatenate((
            np.asarray(failure_rates),
            np.asarray(normalized_frequencies),
            np.asarray(normalized_transmission_rates),
            normalized_running_backlogs,
            normalized_waiting_backlogs,
            np.asarray([
                normalized_task_size,
                normalized_computation_demand,
                normalized_reliability_requirement,
            ]),
            normalized_next_transmission_times,
            normalized_future_cpu_workloads,
        ))
        return state.astype(np.float32)
