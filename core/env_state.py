
"""core.env_state
Environment state container.
"""

import numpy as np
from config.params import params

class EnvironmentState:
    def __init__(self):
        self.servers = {}  # Server objects and CPU backlog metadata.
        self.tasks = {}  # Dictionary to store generated task objects {task_id: task_object}
        self.num_completed_tasks = 0  # Number of completed tasks at all servers
        self.spatial_risk_server_ids = None
        self.spatial_distance_matrix = None
        self.spatial_correlation_matrix = None
        self.spatial_risk_field = None
        self.effective_failure_rates = None

    def add_server_and_init_environment(self, server_object):
        """Add a server object to the environment state."""
        server_id = server_object.server_id  # Extract the server ID from the server object
        #print(f"Adding server with ID {server_id}")
        self.servers[server_id] = {
            'server_object': server_object,
            'waiting_replicas': [],
            'running_replica': None
        }

    def register_waiting_replica(self, server_id, task, selection, service_time):
        """Register a replica waiting for CPU service on a server."""
        server_info = self.servers[server_id]
        identity = (task.id, selection)
        if any((item['task'].id, item['selection']) == identity
               for item in server_info['waiting_replicas']):
            raise RuntimeError(f"Replica {identity} is already waiting on server {server_id}")
        if (server_info['running_replica'] is not None
                and (server_info['running_replica']['task'].id,
                     server_info['running_replica']['selection']) == identity):
            raise RuntimeError(f"Replica {identity} is already running on server {server_id}")
        server_info['waiting_replicas'].append({
            'task': task,
            'selection': selection,
            'service_time': float(service_time),
        })

    def start_replica_execution(self, server_id, task, selection, service_time, service_start_time):
        """Move a waiting replica to running metadata after CPU acquisition."""
        server_info = self.servers[server_id]
        identity = (task.id, selection)
        waiting = server_info['waiting_replicas']
        match = next((item for item in waiting
                      if (item['task'].id, item['selection']) == identity), None)
        if match is None:
            raise RuntimeError(f"Replica {identity} is not registered on server {server_id}")
        waiting.remove(match)
        if server_info['running_replica'] is not None:
            raise RuntimeError(f"Server {server_id} already has a running replica")
        server_info['running_replica'] = {
            'task': task,
            'selection': selection,
            'service_time': float(service_time),
            'service_start_time': float(service_start_time),
        }

    def complete_replica_execution(self, server_id, task, selection):
        """Clear running metadata when CPU service completes."""
        running = self.servers[server_id]['running_replica']
        identity = (task.id, selection)
        if running is None or (running['task'].id, running['selection']) != identity:
            raise RuntimeError(f"Replica {identity} is not running on server {server_id}")
        self.servers[server_id]['running_replica'] = None

    def get_server_backlog_time(self, server_id, current_time=None):
        """Return running remaining service plus waiting service time in seconds."""
        server_info = self.servers[server_id]
        running = server_info['running_replica']
        waiting = server_info['waiting_replicas']
        if current_time is None:
            if running is not None:
                current_time = running['task'].env.now
            elif waiting:
                current_time = waiting[0]['task'].env.now
            else:
                current_time = 0.0

        running_remaining_time = 0.0
        if running is not None:
            elapsed = max(float(current_time) - running['service_start_time'], 0.0)
            running_remaining_time = max(running['service_time'] - elapsed, 0.0)
        waiting_service_time = sum(max(item['service_time'], 0.0)
                                   for item in waiting)
        backlog_time = max(running_remaining_time + waiting_service_time, 0.0)
        assert backlog_time >= -1e-8
        return backlog_time

    def complete_task(self, server_id, task, selection, execute_time):
        """Record a completed replica without changing CPU backlog metadata."""
        self.num_completed_tasks += 1

    def get_server_by_id(self, server_id):
        """Get a server object by its ID."""
        server_info = self.servers.get(server_id)
        if server_info:
            return server_info['server_object']
        else:
            return None

    def set_episode_spatial_risk_context(
        self,
        server_ids,
        distance_matrix,
        correlation_matrix,
        spatial_risk_field,
        effective_failure_rates,
    ):
        """Store one immutable-in-practice episode spatial-risk realization."""
        try:
            ordered_server_ids = list(server_ids)
        except TypeError as exc:
            raise ValueError("server_ids must be a non-empty sequence") from exc
        if not ordered_server_ids:
            raise ValueError("server_ids must not be empty")
        try:
            if len(set(ordered_server_ids)) != len(ordered_server_ids):
                raise ValueError("server_ids must not contain duplicates")
            current_server_ids = set(self.servers)
        except TypeError as exc:
            raise ValueError("server_ids must contain hashable IDs") from exc
        if current_server_ids != set(ordered_server_ids):
            raise ValueError(
                "server_ids must match the current EnvironmentState servers"
            )

        try:
            distance = np.asarray(distance_matrix, dtype=float)
            correlation = np.asarray(correlation_matrix, dtype=float)
            risk_field = np.asarray(spatial_risk_field, dtype=float)
            effective = np.asarray(effective_failure_rates, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("episode spatial-risk context must be numeric") from exc

        expected_matrix_shape = (len(ordered_server_ids), len(ordered_server_ids))
        expected_vector_shape = (len(ordered_server_ids),)
        if distance.shape != expected_matrix_shape:
            raise ValueError("spatial_distance_matrix has an invalid shape")
        if correlation.shape != expected_matrix_shape:
            raise ValueError("spatial_correlation_matrix has an invalid shape")
        if risk_field.shape != expected_vector_shape:
            raise ValueError("spatial_risk_field has an invalid shape")
        if effective.shape != expected_vector_shape:
            raise ValueError("effective_failure_rates has an invalid shape")
        for name, array in (
            ("spatial_distance_matrix", distance),
            ("spatial_correlation_matrix", correlation),
            ("spatial_risk_field", risk_field),
            ("effective_failure_rates", effective),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
        if (effective < 0.0).any():
            raise ValueError("effective_failure_rates must be non-negative")

        self.spatial_risk_server_ids = list(ordered_server_ids)
        self.spatial_distance_matrix = np.array(distance, dtype=float, copy=True)
        self.spatial_correlation_matrix = np.array(
            correlation, dtype=float, copy=True
        )
        self.spatial_risk_field = np.array(risk_field, dtype=float, copy=True)
        self.effective_failure_rates = {
            server_id: float(effective[index])
            for index, server_id in enumerate(ordered_server_ids)
        }

    def set_episode_effective_failure_rates(self, server_ids, effective_failure_rates):
        """Store episode runtime hazards independently of spatial-risk metadata."""
        try:
            ordered_server_ids = list(server_ids)
            effective = np.asarray(effective_failure_rates, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("episode effective failure rates must be numeric") from exc
        if not ordered_server_ids:
            raise ValueError("server_ids must not be empty")
        if len(set(ordered_server_ids)) != len(ordered_server_ids):
            raise ValueError("server_ids must not contain duplicates")
        if set(ordered_server_ids) != set(self.servers):
            raise ValueError("server_ids must match the current EnvironmentState servers")
        if effective.shape != (len(ordered_server_ids),):
            raise ValueError("effective_failure_rates has an invalid shape")
        if not np.isfinite(effective).all() or (effective < 0.0).any():
            raise ValueError("effective_failure_rates must be finite and non-negative")
        self.effective_failure_rates = {
            server_id: float(effective[index])
            for index, server_id in enumerate(ordered_server_ids)
        }
        # This setter is also used for the spatial-off ablation, so do not
        # leave stale spatial metadata attached to the episode.
        self.spatial_risk_server_ids = None
        self.spatial_distance_matrix = None
        self.spatial_correlation_matrix = None
        self.spatial_risk_field = None

    def get_active_failure_rate(self, server_id):
        """Return episode-effective or baseline transient fault arrival rate."""
        server_object = self.get_server_by_id(server_id)
        if server_object is None:
            raise RuntimeError(f"Unknown server_id {server_id}")
        if self.effective_failure_rates is None:
            return server_object.failure_rate
        if server_id not in self.effective_failure_rates:
            raise RuntimeError(
                f"Spatial risk context has no effective failure rate for server_id {server_id}"
            )
        return self.effective_failure_rates[server_id]

    def add_task(self, task_object):
        """Add a task object to the environment state."""
        task_id = task_object.id  # Extract the task ID from the task object
        self.tasks[task_id] = task_object

    def remove_task(self, task_id):
        """Remove a task object from the environment state."""
        if task_id in self.tasks:
            del self.tasks[task_id]
        else:
            print(f"Task with ID {task_id} not found in the task dictionary.")

    def get_task_by_id(self, task_id):
        """Get a task object by its ID."""
        return self.tasks.get(task_id)
   
    def get_min_computation_demand(self):
        """Get the minimum computation demand among all tasks."""
        if not self.tasks:
            print("No tasks available.")
            return None
        
        min_demand = float('inf')  # Initialize min_demand with positive infinity
        
        for task_id, task_obj in self.tasks.items():
            if task_obj.computation_demand < min_demand:
                min_demand = task_obj.computation_demand
        
        return min_demand

    def reset(self):
        """Reset the environment state."""
        self.servers = {}
        self.tasks= {}
        self.num_completed_tasks = 0
        self.spatial_risk_server_ids = None
        self.spatial_distance_matrix = None
        self.spatial_correlation_matrix = None
        self.spatial_risk_field = None
        self.effective_failure_rates = None

    def normalize(self, val, min_val, max_val):
        denominator = max_val - min_val
        if denominator == 0:
            if isinstance(val, np.ndarray):
                return np.zeros_like(val, dtype=np.float32)
            return 0.0
        return (val - min_val) / denominator

    @staticmethod
    def normalize_reliability_requirement(reliability_requirement):
        """Encode R_req in the number-of-nines domain for the policy."""
        try:
            value = float(reliability_requirement)
        except (TypeError, ValueError) as exc:
            raise ValueError("reliability_requirement must be finite and in (0, 1)") from exc
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("reliability_requirement must be finite and in (0, 1)")
        # Raw reliability values cluster near one; map through number of nines
        # / log failure probability before feeding the policy.
        nines = -np.log10(1.0 - value)
        normalized = (nines - 1.0) / 3.0
        return float(np.clip(normalized, 0.0, 1.0))

    def get_state(self, task):
        failure_rates = []
        frequencies = []
        backlog_times = []

        failure_rate_scale = float(params.FAILURE_RATE_SCALE)
        if not np.isfinite(failure_rate_scale) or failure_rate_scale <= 0.0:
            raise ValueError("FAILURE_RATE_SCALE must be a finite positive number")

        for server_id, server_info in self.servers.items():
            server_object = server_info['server_object']
            # The policy observes the known nominal runtime hazard
            # FAILURE_RATE_SCALE * lambda_0, but not the episode-specific
            # spatial realization Z or lambda_eff.
            nominal_failure_rate = failure_rate_scale * server_object.failure_rate
            failure_rates.append(nominal_failure_rate)
            frequencies.append(server_object.processing_frequency)
            backlog_times.append(
                self.get_server_backlog_time(server_id, task.env.now)
            )

        min_raw_failure_rate = min(
            params.EDGE_FAILURE_RATE_RANGE[0],
            params.CLOUD_FAILURE_RATE_RANGE[0]
        )
        max_raw_failure_rate = max(
            params.EDGE_FAILURE_RATE_RANGE[1],
            params.CLOUD_FAILURE_RATE_RANGE[1]
        )
        min_failure_rate = failure_rate_scale * min_raw_failure_rate
        max_failure_rate = failure_rate_scale * max_raw_failure_rate
        normalized_failure_rates = self.normalize(
            np.array(failure_rates), min_failure_rate, max_failure_rate
        )
        normalized_processing_frequencies = self.normalize(
            np.array(frequencies),
            params.EDGE_PROCESSING_FREQ_RANGE[0],
            params.CLOUD_PROCESSING_FREQ_RANGE[1]
        )
        normalized_backlog_times = np.array([
            backlog_time / (backlog_time + params.BACKLOG_TIME_SCALE_SEC)
            for backlog_time in backlog_times
        ], dtype=np.float32)

        normalized_task_size = self.normalize(
            task.task_size, params.TASK_SIZE_RANGE[0], params.TASK_SIZE_RANGE[1]
        )
        normalized_computation_demand = self.normalize(
            task.computation_demand, params.Low_demand, params.High_demand
        )
        normalized_reliability_requirement = self.normalize_reliability_requirement(
            task.reliability_requirement
        )

        normalized_arr = np.concatenate([
            normalized_failure_rates,
            normalized_processing_frequencies,
            normalized_backlog_times,
            [
                normalized_task_size,
                normalized_computation_demand,
                normalized_reliability_requirement,
            ],
        ], dtype=np.float32)
        assert len(normalized_arr) == params.num_states
        return normalized_arr


