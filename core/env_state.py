"""Server CPU metadata, task objects and observable baseline server parameters."""

import numpy as np
from config.params import params


class EnvironmentState:
    def __init__(self):
        self.servers = {}  # Server objects and CPU backlog metadata.
        self.tasks = {}

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
            failure_rates.append(server.base_failure_rate)
            frequencies.append(server.processing_frequency)
            transmission_rates.append(server.transmission_rate)
            running_backlogs.append(running)
            waiting_backlogs.append(waiting)

        frequency_min, frequency_max = params.SERVER_PROCESSING_FREQ_RANGE
        rate_min, rate_max = params.SERVER_TRANSMISSION_RATE_RANGE
        size_min, size_max = params.TASK_SIZE_RANGE
        demand_min, demand_max = params.TASK_DEMAND_RANGE
        state = np.concatenate((
            np.asarray(failure_rates),  # Base failure rates in s^-1, without scaling.
            (np.asarray(frequencies) - frequency_min) / (frequency_max - frequency_min),
            (np.asarray(transmission_rates) - rate_min) / (rate_max - rate_min),
            np.asarray(running_backlogs),  # Raw remaining CPU time in seconds.
            np.asarray(waiting_backlogs),  # Raw queued CPU work in seconds.
            [(task.task_size - size_min) / (size_max - size_min),
             (task.computation_demand - demand_min) / (demand_max - demand_min)],
        ))
        return state.astype(np.float32)
