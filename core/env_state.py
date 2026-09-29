
"""core.env_state
Environment state container.
"""

import numpy as np
from config.params import params

class EnvironmentState:
    def __init__(self):
        self.servers = {}  # Server objects and CPU backlog metadata.
        self.tasks = {}  # Dictionary to store generated task objects {task_id: task_object}
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

    def register_waiting_replica(self, server_id, task, replica_name, service_time):
        """Register a replica waiting for CPU service on a server."""
        server_info = self.servers[server_id]
        identity = (task.id, replica_name)
        if any((item['task'].id, item['replica']) == identity
               for item in server_info['waiting_replicas']):
            raise RuntimeError(f"Replica {identity} is already waiting on server {server_id}")
        if (server_info['running_replica'] is not None
                and (server_info['running_replica']['task'].id,
                     server_info['running_replica']['replica']) == identity):
            raise RuntimeError(f"Replica {identity} is already running on server {server_id}")
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
                      if (item['task'].id, item['replica']) == identity), None)
        if match is None:
            raise RuntimeError(f"Replica {identity} is not registered on server {server_id}")
        waiting.remove(match)
        if server_info['running_replica'] is not None:
            raise RuntimeError(f"Server {server_id} already has a running replica")
        server_info['running_replica'] = {
            'task': task,
            'replica': replica_name,
            'service_time': float(service_time),
            'service_start_time': float(service_start_time),
        }

    def complete_replica_execution(self, server_id, task, replica_name):
        """Clear running metadata when CPU service completes."""
        running = self.servers[server_id]['running_replica']
        identity = (task.id, replica_name)
        if running is None or (running['task'].id, running['replica']) != identity:
            raise RuntimeError(f"Replica {identity} is not running on server {server_id}")
        self.servers[server_id]['running_replica'] = None

    def get_server_backlog_components(self, server_id, current_time=None):
        """Return running remaining and queued service times in seconds."""
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

        running_backlog = 0.0
        if running is not None:
            elapsed = current_time - running['service_start_time']
            running_backlog = max(running['service_time'] - elapsed, 0.0)
        waiting_backlog = sum(replica['service_time'] for replica in waiting)

        if not np.isfinite(running_backlog) or running_backlog < 0.0:
            raise ValueError("Running CPU backlog must be finite and non-negative")
        if not np.isfinite(waiting_backlog) or waiting_backlog < 0.0:
            raise ValueError("Waiting CPU backlog must be finite and non-negative")
        return float(running_backlog), float(waiting_backlog)

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

    def get_active_failure_rate(self, server_id):
        """Return episode-effective or baseline transient fault arrival rate."""
        server_object = self.get_server_by_id(server_id)
        if server_object is None:
            raise RuntimeError(f"Unknown server_id {server_id}")
        if self.effective_failure_rates is None:
            return server_object.base_failure_rate
        if server_id not in self.effective_failure_rates:
            raise RuntimeError(
                f"Spatial risk context has no effective failure rate for server_id {server_id}"
            )
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
   
    def reset(self):
        """Reset the environment state."""
        self.servers = {}
        self.tasks= {}
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

    def get_state(self, task):
        expected_dimension = 5 * params.NUM_SERVERS + 3
        if params.num_states != expected_dimension:
            raise RuntimeError(
                f"params.num_states ({params.num_states}) does not match "
                f"the 5N+3 state dimension ({expected_dimension})"
            )

        expected_server_ids = list(range(1, params.NUM_SERVERS + 1))
        try:
            server_ids = sorted(self.servers)
        except TypeError as exc:
            raise RuntimeError("Server IDs must be 1..N") from exc
        if len(server_ids) != params.NUM_SERVERS or server_ids != expected_server_ids:
            raise RuntimeError(
                f"Expected {params.NUM_SERVERS} servers with IDs 1..{params.NUM_SERVERS}; "
                f"found IDs {server_ids}"
            )

        failure_rates = []
        frequencies = []
        transmission_rates = []
        running_backlogs = []
        waiting_backlogs = []

        frequency_min, frequency_max = params.SERVER_PROCESSING_FREQ_RANGE
        transmission_rate_min = min(params.SERVER_TRANSMISSION_RATES)
        transmission_rate_max = max(params.SERVER_TRANSMISSION_RATES)

        for server_id in server_ids:
            server_info = self.servers[server_id]
            server_object = server_info['server_object']
            failure_rate = float(self.get_active_failure_rate(server_id))
            frequency = float(server_object.processing_frequency)
            transmission_rate = float(server_object.transmission_rate)
            if not np.isfinite(failure_rate) or failure_rate < 0.0:
                raise ValueError(
                    f"Server {server_id} effective failure rate must be finite and non-negative"
                )
            if (
                not np.isfinite(frequency)
                or not frequency_min <= frequency <= frequency_max
            ):
                raise ValueError(
                    f"Server {server_id} processing_frequency must be finite and in "
                    f"[{frequency_min}, {frequency_max}] MIPS"
                )

            running_backlog, waiting_backlog = self.get_server_backlog_components(
                server_id, task.env.now
            )
            failure_rates.append(failure_rate)
            frequencies.append(frequency)
            transmission_rates.append(transmission_rate)
            running_backlogs.append(running_backlog)
            waiting_backlogs.append(waiting_backlog)

        scale = params.BACKLOG_TIME_SCALE_SEC
        task_size = float(task.task_size)
        task_size_min, task_size_max = params.TASK_SIZE_RANGE
        if (
            not np.isfinite(task_size)
            or not task_size_min <= task_size <= task_size_max
        ):
            raise ValueError(
                f"task.task_size must be finite and in [{task_size_min}, {task_size_max}]"
            )

        computation_demand = float(task.computation_demand)
        demand_min, demand_max = params.Low_demand, params.High_demand
        if (
            not np.isfinite(computation_demand)
            or not demand_min <= computation_demand <= demand_max
        ):
            raise ValueError(
                "task.computation_demand must be finite and in "
                f"[{demand_min}, {demand_max}] MI"
            )

        reliability_requirement = float(task.reliability_requirement)

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

        state = np.concatenate((
            np.asarray(failure_rates),
            np.asarray(normalized_frequencies),
            np.asarray(normalized_transmission_rates),
            normalized_running_backlogs,
            normalized_waiting_backlogs,
            np.asarray([
                normalized_task_size,
                normalized_computation_demand,
                reliability_requirement,
            ]),
        ))
        with np.errstate(over="ignore", invalid="ignore"):
            state = state.astype(np.float32, copy=False)
        if state.shape != (params.num_states,):
            raise RuntimeError(
                f"State shape {state.shape} does not match ({params.num_states},)"
            )
        if not np.isfinite(state).all():
            raise ValueError("State features must be finite when represented as float32")
        return state
