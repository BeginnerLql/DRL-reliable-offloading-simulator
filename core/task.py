
"""core.task

Task object used by the SimPy environment.

Change in the modular refactor:
- Excel files live in data/... (config.paths.DATA_DIR)
- So, default params_file is resolved via DATA_DIR.
"""

import os
import pandas as pd
import math
import random

from config.params import params
from config.paths import DATA_DIR


class Task:

    def __init__(self, env, state, id, params_file: str = "task_parameters.xlsx"):
        self.env = env
        self.env_state = state
        self.id = id
        
        # Other attributes
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
        # Resolve params_file:
        # - If an absolute path is passed, use it.
        # - If only a filename is passed, read it from data/.
        resolved = params_file
        if not os.path.isabs(resolved):
            resolved = os.path.join(DATA_DIR, resolved)
        task_info_df = pd.read_excel(resolved)
        task_row = task_info_df.loc[task_info_df['Task_ID'] == self.id]
        self.task_size = task_row['Task_Size'].values[0]
        self.computation_demand = task_row['Computation_Demand'].values[0]
        self.teta = None  

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

                yield self.env.timeout(max(self.teta - (self.primaryFinished - self.primaryStarted), 0))
                self.backupStarted = self.env.now
                self.env.process(self.backup())
            
        else: ## z==1
            self.primaryStarted = self.backupStarted = self.env.now
            self.env.process(self.primary())
            self.env.process(self.backup())

    
    def primary(self):
        
        inpDelay , outDelay = self.calc_input_output_delay(self.primaryNode)
              
        yield self.env.timeout(inpDelay)

        
        Q_time= self.env.now
        with self.primaryNode.queue.request(priority=1) as req:
            yield req  # Queueing time in server
            Q_time= self.env.now - Q_time
            self.env_state.assign_task_to_server(self.primaryNode.server_id, self, "primary")# assign a task to a server as primary run
            # Calculate service time on primaryNode
            self.primary_service_time = self.computation_demand / self.primaryNode.processing_frequency
            #print("service_time", service_time , "for task",self.id,"in server " , self.primaryNode.server_id )
            failure_rate=self.set_failure_rate(self.primaryNode)
            # Simulate execution either success or failed
            yield self.env.timeout(self.primary_service_time)
            
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
        self.env_state.complete_task(self.primaryNode.server_id, self, 'primary', self.primary_service_time)
        
        self.teta= 1.5 * (self.primary_service_time + inpDelay + outDelay + Q_time) 

    def backup(self):

        inpDelay , outDelay = self.calc_input_output_delay(self.backupNode)

        # Use PriorityRequest if backupNode is the same as primaryNode.
        # Retry is allowed because the preceding fault is transient and has
        # negligible recovery time in this model.
        if self.backupNode == self.primaryNode: # Retry strategy
            # no inpDelay
            with self.backupNode.queue.request(priority=0) as req: #high priority
                yield req  
                self.env_state.assign_task_to_server(self.backupNode.server_id, self, "backup") 
                backup_service_time = self.primary_service_time # as primary
                failure_rate=self.set_failure_rate(self.backupNode)
                yield self.env.timeout(backup_service_time)

        else: # recovery block or first result strategy
            yield self.env.timeout(inpDelay)
            with self.backupNode.queue.request(priority=1) as req:
                yield req 
                self.env_state.assign_task_to_server(self.backupNode.server_id, self, "backup") 
                backup_service_time = self.computation_demand / self.backupNode.processing_frequency # may differ from primary according to frequency of backup server
                failure_rate=self.set_failure_rate(self.backupNode)
                yield self.env.timeout(backup_service_time)

            
        
        
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
        self.env_state.complete_task(self.backupNode.server_id, self, "backup", backup_service_time)

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
        """Return the server's observable estimated transient fault arrival rate (1/s)."""
        return server_object.failure_rate
