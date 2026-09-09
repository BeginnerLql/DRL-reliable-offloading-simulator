
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

    def normalize(self, val, min_val, max_val):
        denominator = max_val - min_val
        if denominator == 0:
            if isinstance(val, np.ndarray):
                return np.zeros_like(val, dtype=np.float32)
            return 0.0
        return (val - min_val) / denominator

    def get_state(self, task):
        failure_rates = []
        frequencies = []
        backlog_times = []

        for server_id, server_info in self.servers.items():
            server_object = server_info['server_object']
            failure_rates.append(server_object.failure_rate)
            frequencies.append(server_object.processing_frequency)
            backlog_times.append(
                self.get_server_backlog_time(server_id, task.env.now)
            )

        min_failure_rate = min(
            params.EDGE_FAILURE_RATE_RANGE[0],
            params.CLOUD_FAILURE_RATE_RANGE[0]
        )
        max_failure_rate = max(
            params.EDGE_FAILURE_RATE_RANGE[1],
            params.CLOUD_FAILURE_RATE_RANGE[1]
        )
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

        normalized_arr = np.concatenate([
            normalized_failure_rates,
            normalized_processing_frequencies,
            normalized_backlog_times,
            [normalized_task_size, normalized_computation_demand]
        ], dtype=np.float32)
        assert len(normalized_arr) == params.num_states
        return normalized_arr


