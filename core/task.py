
"""core.task

Task object used by the SimPy environment.

Change in the modular refactor:
- Excel files live in data/... (config.paths.DATA_DIR)
- So, default params_file is resolved via DATA_DIR.
"""

import os
import pandas as pd
import math

from config.params import params
from config.paths import DATA_DIR


RELIABILITY_REQUIREMENT_LEVELS = (0.9, 0.99, 0.999, 0.9999)


def get_upload_time(input_data_size_mb, uplink_rate_mbps):
    """Return upload time in seconds for MB over an Mbps uplink."""
    try:
        data_size = float(input_data_size_mb)
        rate = float(uplink_rate_mbps)
    except (TypeError, ValueError) as exc:
        raise ValueError("input_data_size_mb and uplink_rate_mbps must be numeric") from exc
    if not math.isfinite(data_size) or data_size < 0.0:
        raise ValueError("input_data_size_mb must be finite and non-negative")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("uplink_rate_mbps must be finite and positive")
    return 8.0 * data_size / rate


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

        # Task-level reliability evaluation, computed once after action selection.
        self.primary_effective_failure_rate = None
        self.backup_effective_failure_rate = None
        self.primary_service_time_for_reliability = None
        self.backup_service_time_for_reliability = None
        self.primary_failure_probability = None
        self.backup_failure_probability = None
        self.joint_failure_probability = None
        self.execution_reliability = None
        self.reliability_satisfied = None
        # Reward diagnostics populated when MainLoop.calcReward resolves this task.
        self.base_reward = None
        self.reliability_violation = None
        self.reliability_penalty = None

        # Task-level event used by the episode drain. This is triggered once
        # when the current primary/backup semantics produce a final outcome;
        # it is not a replica CPU-completion event.
        self.resolution_event = self.env.event()

        # Resolve params_file:
        # - If an absolute path is passed, use it.
        # - If only a filename is passed, read it from data/.
        resolved = params_file
        if not os.path.isabs(resolved):
            resolved = os.path.join(DATA_DIR, resolved)
        task_info_df = pd.read_excel(resolved)
        required_columns = {
            "Task_ID",
            "Computation_Demand",
            "Reliability_Requirement",
        }
        input_column = "Input_Data_Size_MB"
        if input_column not in task_info_df.columns:
            # Read-only compatibility for old temporary fixtures. The formal
            # generated workbook uses Input_Data_Size_MB exclusively.
            if "Task_Size" in task_info_df.columns:
                input_column = "Task_Size"
            else:
                required_columns.add(input_column)
        missing_columns = sorted(required_columns.difference(task_info_df.columns))
        if missing_columns:
            raise ValueError(
                "task_parameters.xlsx is missing required columns: "
                + ", ".join(missing_columns)
            )

        task_row = task_info_df.loc[task_info_df["Task_ID"] == self.id]
        if len(task_row) != 1:
            raise ValueError(
                f"task_parameters.xlsx must contain exactly one row for Task_ID {self.id}"
            )
        self.input_data_size_mb = float(task_row[input_column].values[0])
        if not math.isfinite(self.input_data_size_mb) or self.input_data_size_mb < 0.0:
            raise ValueError("Input_Data_Size_MB must be finite and non-negative")
        self.computation_demand = task_row["Computation_Demand"].values[0]

        reliability_requirement = float(task_row["Reliability_Requirement"].values[0])
        if not math.isfinite(reliability_requirement) or reliability_requirement <= 0.0:
            raise ValueError("Reliability_Requirement must be a finite value greater than 0")
        if not any(
            math.isclose(
                reliability_requirement,
                allowed,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            for allowed in RELIABILITY_REQUIREMENT_LEVELS
        ):
            raise ValueError(
                "Reliability_Requirement must be one of "
                + ", ".join(str(value) for value in RELIABILITY_REQUIREMENT_LEVELS)
            )
        self.reliability_requirement = reliability_requirement
        self.teta = None

    @property
    def task_size(self):
        """Compatibility alias for the persisted input data size."""
        return self.input_data_size_mb

    @task_size.setter
    def task_size(self, value):
        self.input_data_size_mb = float(value)

    def initialize_reliability_evaluation(self, primary_node, backup_node):
        """Compute task reliability once for the selected action."""
        self.primaryNode = primary_node
        self.backupNode = backup_node
        primary_rate = float(self.set_failure_rate(primary_node))
        backup_rate = float(self.set_failure_rate(backup_node))
        primary_service_time = float(self.computation_demand / primary_node.processing_frequency)
        backup_service_time = float(self.computation_demand / backup_node.processing_frequency)
        values = (primary_rate, backup_rate, primary_service_time, backup_service_time)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("reliability inputs must be finite and non-negative")
        self.primary_effective_failure_rate = primary_rate
        self.backup_effective_failure_rate = backup_rate
        self.primary_service_time_for_reliability = primary_service_time
        self.backup_service_time_for_reliability = backup_service_time
        self.primary_failure_probability = -math.expm1(-primary_rate * primary_service_time)
        self.backup_failure_probability = -math.expm1(-backup_rate * backup_service_time)
        # Same-server retry uses the same hazard with conditionally independent
        # transient retry outcomes, so the joint term is p_j * p_k (p^2 for retry).
        self.joint_failure_probability = self.primary_failure_probability * self.backup_failure_probability
        self.execution_reliability = 1.0 - self.joint_failure_probability
        self.reliability_satisfied = self.execution_reliability >= self.reliability_requirement

    def execute_task(self, X, Y, Z):

        self.primaryNode=X
        self.backupNode=Y
        self.z = Z
        if self.z == 0:
            # Sequential mode retains its action semantics, but reliability is
            # assessed analytically and the primary completion resolves it.
            self.primaryStarted = self.env.now
            yield self.env.process(self.primary())
        else: ## z==1
            self.primaryStarted = self.backupStarted = self.env.now
            self.env.process(self.primary())
            self.env.process(self.backup())

    
    def primary(self):
        
        inpDelay , outDelay = self.calc_input_output_delay(self.primaryNode)
              
        yield self.env.timeout(inpDelay)

        
        Q_time= self.env.now
        self.primary_service_time = self.computation_demand / self.primaryNode.processing_frequency
        self.env_state.register_waiting_replica(
            self.primaryNode.server_id, self, "primary", self.primary_service_time
        )
        with self.primaryNode.queue.request(priority=1) as req:
            yield req  # Queueing time in server
            Q_time= self.env.now - Q_time
            self.env_state.start_replica_execution(
                self.primaryNode.server_id, self, "primary",
                self.primary_service_time, self.env.now
            )
            # Reliability was evaluated analytically before execution.
            yield self.env.timeout(self.primary_service_time)
            self.env_state.complete_replica_execution(
                self.primaryNode.server_id, self, "primary"
            )
            
        # Replica completion is nominal; task-level threshold is stored in
        # reliability_satisfied and applied by MainLoop reward bookkeeping.
        yield self.env.timeout(outDelay)
        self.primaryStat = "success"

        self.primaryFinished = self.env.now
        
        #print(f"Task {self.id} {'succeeded' if self.primaryStat == 'success' else 'failed'} on primary server {self.primaryNode.server_id}")
        self.env_state.complete_task(self.primaryNode.server_id, self, 'primary', self.primary_service_time)
        
        self.teta= 1.5 * (self.primary_service_time + inpDelay + outDelay + Q_time)
        self._signal_resolution_if_ready()

    def backup(self):

        inpDelay , outDelay = self.calc_input_output_delay(self.backupNode)

        # Use PriorityRequest if backupNode is the same as primaryNode.
        # Retry is allowed because the preceding fault is transient and has
        # negligible recovery time in this model.
        if self.backupNode == self.primaryNode: # Retry strategy
            # A retry is a new replica and also uploads its input before CPU.
            yield self.env.timeout(inpDelay)
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
                yield self.env.timeout(backup_service_time)
                self.env_state.complete_replica_execution(
                    self.backupNode.server_id, self, "backup"
                )

            
        
        # Replica completion is nominal; task-level threshold is stored in
        # reliability_satisfied and applied by MainLoop reward bookkeeping.
        yield self.env.timeout(outDelay)
        self.backupStat = "success"
         
        self.backupFinished = self.env.now
        
        #print(f"Task {self.id} {'succeeded' if self.backupStat == 'success' else 'failed'} on backup server {self.backupNode.server_id}")
        self.env_state.complete_task(self.backupNode.server_id, self, "backup", backup_service_time)
        self._signal_resolution_if_ready()

    def _is_resolved(self):
        """Return whether the selected execution semantics have a result.

        Runtime replicas are nominally successful; the legacy failure cases are
        retained only so old synthetic event tests remain interpretable.
        """
        if self.z == 0:
            return (
                (self.primaryStat == "success" and self.primaryFinished is not None)
                or (
                    self.primaryStat == "failure"
                    and self.backupStat in {"success", "failure"}
                    and self.backupFinished is not None
                )
            )
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
        # All formal nodes are Edge and each replica uploads its input before
        # requesting CPU. Results are assumed small, so download is zero.
        return get_upload_time(
            self.input_data_size_mb,
            server_object.uplink_rate_mbps,
        ), 0.0
    
    
    def set_failure_rate(self, server_object):
        """Return this episode's effective transient fault arrival rate (1/s)."""
        return self.env_state.get_active_failure_rate(server_object.server_id)
