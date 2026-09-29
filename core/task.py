"""Runtime task initialized from a run-level task profile."""

import math
import random

from config.params import params


class Task:
    def __init__(self, env, state, task_id, task_profile):
        profile_task_id = task_profile["Task_ID"]
        if profile_task_id != task_id:
            raise ValueError(
                f"Task profile ID {profile_task_id} does not match task_id {task_id}."
            )

        self.env = env
        self.env_state = state
        self.id = task_id
        self.task_size = task_profile["Task_Size"]
        self.computation_demand = task_profile["Computation_Demand"]
        self.reliability_requirement = task_profile["Reliability_Requirement"]

        self.primaryNode = None
        self.backupNode = None
        self.z = None

        self.primaryStarted = None
        self.primaryFinished = None
        self.primaryStat = None
        self.primary_service_time = None
        self.backupStarted = None
        self.backupFinished = None
        self.backupStat = None

        self.resolution_event = self.env.event()

    def execute_task(self, X, Y, Z):

        self.primaryNode=X
        self.backupNode=Y
        self.z = Z
        if self.z == 0:
            self.primaryStarted = self.env.now
            yield self.env.process(self.primary())
            # Retry/failover follows a primary replica execution failure.
            # The fault is transient, so the selected server remains usable.
            if self.primaryStat == "failure":

                yield self.env.timeout(0)
                self.backupStarted = self.env.now
                self.env.process(self.backup())
            
        else: ## z==1
            self.primaryStarted = self.backupStarted = self.env.now
            self.env.process(self.primary())
            self.env.process(self.backup())

    
    def primary(self):
        
        inpDelay , outDelay = self.calc_input_output_delay(self.primaryNode)
              
        yield self.env.timeout(inpDelay)

        
        self.primary_service_time = self.computation_demand / self.primaryNode.processing_frequency
        self.env_state.register_waiting_replica(
            self.primaryNode.server_id, self, "primary", self.primary_service_time
        )
        with self.primaryNode.queue.request(priority=1) as req:
            yield req  # Queueing time in server
            self.env_state.start_replica_execution(
                self.primaryNode.server_id, self, "primary",
                self.primary_service_time, self.env.now
            )
            failure_rate=self.set_failure_rate(self.primaryNode)
            # Simulate execution either success or failed
            yield self.env.timeout(self.primary_service_time)
            self.env_state.complete_replica_execution(
                self.primaryNode.server_id, self, "primary"
            )
            
        # Probability that at least one transient server fault occurs during
        # this primary replica's execution interval.
        fault_prob= 1-math.exp(-failure_rate * self.primary_service_time)
        r=random.uniform(0, 1)
        if(r<fault_prob):
            # This is a primary replica execution failure, not a permanent
            # failure of the selected server.
            self.primaryStat = "failure"
            
        else:
            yield self.env.timeout(outDelay)
            self.primaryStat = "success"

        self.primaryFinished = self.env.now
        
        #print(f"Task {self.id} {'succeeded' if self.primaryStat == 'success' else 'failed'} on primary server {self.primaryNode.server_id}")
        self._signal_resolution_if_ready()

    def backup(self):

        inpDelay , outDelay = self.calc_input_output_delay(self.backupNode)

        # Use PriorityRequest if backupNode is the same as primaryNode.
        # Retry is allowed because the preceding fault is transient and has
        # negligible recovery time in this model.
        if self.backupNode == self.primaryNode: # Retry strategy
            # no inpDelay
            backup_service_time = self.primary_service_time # as primary
            self.env_state.register_waiting_replica(
                self.backupNode.server_id, self, "backup", backup_service_time
            )
            with self.backupNode.queue.request(priority=0) as req: # high priority
                yield req
                self.env_state.start_replica_execution(
                    self.backupNode.server_id, self, "backup",
                    backup_service_time, self.env.now
                )
                failure_rate=self.set_failure_rate(self.backupNode)
                yield self.env.timeout(backup_service_time)
                self.env_state.complete_replica_execution(
                    self.backupNode.server_id, self, "backup"
                )

        else: # recovery block or first result strategy
            yield self.env.timeout(inpDelay)
            backup_service_time = self.computation_demand / self.backupNode.processing_frequency # may differ from primary according to frequency of backup server
            self.env_state.register_waiting_replica(
                self.backupNode.server_id, self, "backup", backup_service_time
            )
            with self.backupNode.queue.request(priority=1) as req:
                yield req
                self.env_state.start_replica_execution(
                    self.backupNode.server_id, self, "backup",
                    backup_service_time, self.env.now
                )
                failure_rate=self.set_failure_rate(self.backupNode)
                yield self.env.timeout(backup_service_time)
                self.env_state.complete_replica_execution(
                    self.backupNode.server_id, self, "backup"
                )

            
        
        
        # Probability that at least one transient server fault occurs during
        # this backup replica's execution interval.
        fault_prob= 1-math.exp(-failure_rate * backup_service_time)
        r=random.uniform(0, 1)
        if(r<fault_prob):
            # This is a backup replica execution failure; the server remains
            # available for later tasks and retries.
            self.backupStat = "failure"
        else:
            yield self.env.timeout(outDelay)
            self.backupStat = "success"
         
        self.backupFinished = self.env.now
        
        #print(f"Task {self.id} {'succeeded' if self.backupStat == 'success' else 'failed'} on backup server {self.backupNode.server_id}")
        self._signal_resolution_if_ready()

    def _is_resolved(self):
        """Match MainLoop.calcReward() final-outcome conditions."""
        if self.z == 0:
            return (
                (
                    self.primaryStat == "success"
                    and self.primaryFinished is not None
                    and self.backupStat is None
                )
                or (
                    self.primaryStat == "failure"
                    and self.backupStat == "success"
                    and self.backupFinished is not None
                )
                or (
                    self.primaryStat == "failure"
                    and self.backupStat == "failure"
                    and self.backupFinished is not None
                )
            )

        # Parallel first-result semantics: one success resolves immediately;
        # two failures require both replica timestamps.
        return (
            (self.primaryStat == "success" and self.primaryFinished is not None)
            or (self.backupStat == "success" and self.backupFinished is not None)
            or (
                self.primaryStat == "failure"
                and self.backupStat == "failure"
                and self.primaryFinished is not None
                and self.backupFinished is not None
            )
        )

    def _signal_resolution_if_ready(self):
        """Signal the task-level resolution event at most once."""
        if self.resolution_event.triggered:
            return
        if self._is_resolved():
            self.resolution_event.succeed(float(self.env.now))

    def calc_input_output_delay(self, server_object):
        if server_object.server_type == "Edge":
            # Calculate input delay for Edge
            
            inpDelay = 0
        else:
            # Calculate input delay for Cloud
            inpDelay = self.task_size / params.rsu_to_cloud_bandwidth


        # Output delay is the same as input delay
        outDelay = inpDelay   
        return inpDelay, outDelay
    
    
    def set_failure_rate(self, server_object):
        """Return this episode's effective transient fault arrival rate (1/s)."""
        return self.env_state.get_active_failure_rate(server_object.server_id)
