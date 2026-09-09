
"""core.env_state
Environment state container.
"""

import numpy as np
from config.params import params

class EnvironmentState:
    def __init__(self):
        self.servers = {}  # Server objects and replica-failure statistics.
        self.tasks = {}  # Dictionary to store generated task objects {task_id: task_object}
        self.num_completed_tasks = 0  # Number of completed tasks at all servers

    def add_server_and_init_environment(self, server_object):
        """Add a server object to the environment state."""
        server_id = server_object.server_id  # Extract the server ID from the server object
        #print(f"Adding server with ID {server_id}")
        self.servers[server_id] = {
            'server_object': server_object,
            'tasks_assigned': [],  # List of task objects assigned to this server
            # Legacy empirical statistics for observed primary/backup replica
            # execution failures; these are not server downtime measures.
            'primary_failure_time': 1000000 * server_object.failure_rate,
            'backup_failure_time': 1000000 * server_object.failure_rate,
            'primary_executed_time': 1000000,  # Initialize executed tasks time for the server if it is selected as primary
            'backup_executed_time': 1000000,  # Initialize executed tasks time for the server if it is selected as backup
            'load': 0  # Initialize load for the server (sum of computation demands of tasks assigned to it)
        }

    def assign_task_to_server(self, server_id, task, selection):
        """Assign a task object to a server based on the selection (primary or backup)."""
              
        self.servers[server_id]['tasks_assigned'].append({'task': task, 'selection': selection})

        # Update 'load'
        self.servers[server_id]['load'] += task.computation_demand

    def complete_task(self, server_id, task, selection, execute_time):
        """Set parameters about completed task in the environment state."""
        tasks_assigned = self.servers[server_id]['tasks_assigned']
        for assigned_task in tasks_assigned:
            if assigned_task['task'] == task and assigned_task['selection'] == selection:
                # Update 'load'
                self.servers[server_id]['load'] -= task.computation_demand

                if selection == "primary" :
                    # Update 'primary_executed_tasks'
                    self.servers[server_id]['primary_executed_time'] += execute_time
                    if task.primaryStat == "failure":
                        # Record the observed primary replica execution failure.
                        self.servers[server_id]['primary_failure_time'] += execute_time
                    

                elif selection == "backup": 
                    # Update 'backup_executed_tasks'
                    self.servers[server_id]['backup_executed_time'] += execute_time
                    if task.backupStat == "failure":
                        # Record the observed backup replica execution failure.
                        self.servers[server_id]['backup_failure_time'] += execute_time

                
                self.num_completed_tasks += 1
                break

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
        return (val - min_val) / (max_val - min_val + 1e-8)

    def get_state(self, task):
        primary_failure_rate = []
        backup_failure_rate = []
        frequency = []
        load = []

        for server_id, server_info in self.servers.items():
            server_object = server_info['server_object']

            primary_failure_rate.append(server_info['primary_failure_time'] / (server_info['primary_executed_time'] + 1e-8))
            backup_failure_rate.append(server_info['backup_failure_time'] / (server_info['backup_executed_time'] + 1e-8))
            frequency.append(server_object.processing_frequency)
            load.append(server_info['load'])

        # normalization
        min_failure_rate = min(params.EDGE_FAILURE_RATE_RANGE[0], params.CLOUD_FAILURE_RATE_RANGE[0])
        max_failure_rate = max(params.EDGE_FAILURE_RATE_RANGE[1], params.CLOUD_FAILURE_RATE_RANGE[1])
        norm_primary = self.normalize(np.array(primary_failure_rate), min_failure_rate, max_failure_rate)
        norm_backup = self.normalize(np.array(backup_failure_rate), min_failure_rate, max_failure_rate)
        norm_frequency = self.normalize(np.array(frequency), params.EDGE_PROCESSING_FREQ_RANGE[0], params.CLOUD_PROCESSING_FREQ_RANGE[1])

        max_local_load = max(load) if load else 1
        norm_load = np.array(load, dtype=np.float32) / max_local_load if max_local_load > 0 else np.zeros_like(load, dtype=np.float32)

        norm_task_size = self.normalize(task.task_size, params.TASK_SIZE_RANGE[0], params.TASK_SIZE_RANGE[1])
        norm_demand = self.normalize(task.computation_demand, params.Low_demand, params.High_demand)

        normalized_arr = np.concatenate(
            [norm_primary, norm_backup, norm_frequency, norm_load, [norm_task_size, norm_demand]],
            dtype=np.float32
        )
        return normalized_arr


