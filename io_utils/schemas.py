"""Column order for raw MainLoop task-assignment records.

Excel export inserts Final_status after the first 26 columns; diagnostic CSVs
use this raw order directly. Keep legacy names for historical result readers.
"""

TASK_ASSIGNMENT_COLUMNS = [
    "episode",
    "task_id",
    "Primary",
    "Primary_Start",
    "Primary_End",
    "Primary_Status",
    "Backup",
    "Backup_Start",
    "Backup_End",
    "Backup_Status",
    "Z",
    "Reliability_Requirement",
    "Primary_Effective_Failure_Rate",
    "Backup_Effective_Failure_Rate",
    "Primary_Reliability_Service_Time",
    "Backup_Reliability_Service_Time",
    "Primary_Failure_Probability",
    "Backup_Failure_Probability",
    "Joint_Failure_Probability",
    "Execution_Reliability",
    "Reliability_Satisfied",
    "Task_Reward",
    "Task_Delay",
    "Reliability_Margin",
    "Reliability_Shortfall",
    "Reliability_Excess",
    "Base_Reward",
    "Reliability_Violation_Log10",
    "Reliability_Penalty",
    "action_index",
    "server_j",
    "server_k",
]
