# save_parameters_and_logs.py
# - Fixed base failure rates, fixed local paths only
# - No Permutation_Number
# - Reads Excel input files ONLY from data/
# - Writes results per model
# - Creates Excel-native charts (no PNG files)
# - Adds Summary.AVG_Failure (rolling mean over last 40 episodes) + ONLY line chart in Summary

import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference


from config.paths import DATA_DIR, RESULTS_DIR, ensure_dirs


RELIABILITY_DIAGNOSTIC_COLUMNS = [
    "Reliability_Requirement",
    "Task_Count",
    "Success_Count",
    "Failure_Count",
    "Success_Rate",
    "Mean_Task_Reward",
    "Std_Task_Reward",
    "Mean_Task_Delay",
    "Mean_Execution_Reliability",
    "Mean_Reliability_Margin",
    "Median_Reliability_Margin",
    "Mean_Reliability_Excess",
    "Mean_Reliability_Shortfall",
    "Mean_Joint_Failure_Probability",
    "Parallel_Mode_Count",
    "Parallel_Mode_Rate",
]

REPLICA_COMPLETION_COLUMNS = [
    "Episode",
    "Task_ID",
    "Action_Index",
    "Server_ID",
    "Replica_Label",
    "Finish_Time",
    "Service_Time",
]

PAIR_DIAGNOSTIC_COLUMNS = [
    "Reliability_Requirement",
    "Primary",
    "Backup",
    "Z",
    "Selection_Count",
    "Selection_Share_Within_Requirement",
    "Success_Count",
    "Success_Rate",
    "Mean_Task_Reward",
    "Mean_Task_Delay",
    "Mean_Execution_Reliability",
    "Mean_Reliability_Margin",
    "Mean_Joint_Failure_Probability",
]


def build_reliability_diagnostics(task_assignments_df):
    """Aggregate task outcomes by the original, unnormalized R_req value."""
    if task_assignments_df.empty:
        return pd.DataFrame(columns=RELIABILITY_DIAGNOSTIC_COLUMNS)
    rows = []
    grouped = task_assignments_df.groupby("Reliability_Requirement", sort=True, dropna=False)
    for requirement, group in grouped:
        task_count = len(group)
        success_mask = group["Final_status"].eq("success")
        reward = pd.to_numeric(group["Task_Reward"], errors="coerce")
        delay = pd.to_numeric(group["Task_Delay"], errors="coerce")
        execution_reliability = pd.to_numeric(group["Execution_Reliability"], errors="coerce")
        margin = pd.to_numeric(group["Reliability_Margin"], errors="coerce")
        excess = pd.to_numeric(group["Reliability_Excess"], errors="coerce")
        shortfall = pd.to_numeric(group["Reliability_Shortfall"], errors="coerce")
        joint_failure = pd.to_numeric(group["Joint_Failure_Probability"], errors="coerce")
        if "action_index" in group.columns and group["action_index"].notna().any():
            parallel_mask = group["action_index"].notna()
        else:
            parallel_mask = pd.to_numeric(group["Z"], errors="coerce").eq(1)
        rows.append({
            "Reliability_Requirement": requirement,
            "Task_Count": task_count,
            "Success_Count": int(success_mask.sum()),
            "Failure_Count": int((~success_mask).sum()),
            "Success_Rate": float(success_mask.mean()),
            "Mean_Task_Reward": float(reward.mean()),
            "Std_Task_Reward": float(reward.std(ddof=0)),
            "Mean_Task_Delay": float(delay.mean()),
            "Mean_Execution_Reliability": float(execution_reliability.mean()),
            "Mean_Reliability_Margin": float(margin.mean()),
            "Median_Reliability_Margin": float(margin.median()),
            "Mean_Reliability_Excess": float(excess.mean()),
            "Mean_Reliability_Shortfall": float(shortfall.mean()),
            "Mean_Joint_Failure_Probability": float(joint_failure.mean()),
            "Parallel_Mode_Count": int(parallel_mask.sum()),
            "Parallel_Mode_Rate": float(parallel_mask.mean()),
        })
    return pd.DataFrame(rows, columns=RELIABILITY_DIAGNOSTIC_COLUMNS)


def build_pair_diagnostics(task_assignments_df):
    """Aggregate selected primary/backup/Z combinations within each R_req."""
    if task_assignments_df.empty:
        return pd.DataFrame(columns=PAIR_DIAGNOSTIC_COLUMNS)
    rows = []
    requirement_totals = task_assignments_df.groupby(
        "Reliability_Requirement", sort=True, dropna=False
    ).size()
    grouped = task_assignments_df.groupby(
        ["Reliability_Requirement", "Primary", "Backup", "Z"],
        sort=True,
        dropna=False,
    )
    for (requirement, primary, backup, z), group in grouped:
        selection_count = len(group)
        success_mask = group["Final_status"].eq("success")
        rows.append({
            "Reliability_Requirement": requirement,
            "Primary": primary,
            "Backup": backup,
            "Z": z,
            "Selection_Count": selection_count,
            "Selection_Share_Within_Requirement": float(
                selection_count / requirement_totals.loc[requirement]
            ),
            "Success_Count": int(success_mask.sum()),
            "Success_Rate": float(success_mask.mean()),
            "Mean_Task_Reward": float(pd.to_numeric(group["Task_Reward"], errors="coerce").mean()),
            "Mean_Task_Delay": float(pd.to_numeric(group["Task_Delay"], errors="coerce").mean()),
            "Mean_Execution_Reliability": float(pd.to_numeric(group["Execution_Reliability"], errors="coerce").mean()),
            "Mean_Reliability_Margin": float(pd.to_numeric(group["Reliability_Margin"], errors="coerce").mean()),
            "Mean_Joint_Failure_Probability": float(pd.to_numeric(group["Joint_Failure_Probability"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows, columns=PAIR_DIAGNOSTIC_COLUMNS)


def save_params_and_logs(
    params,
    log_data,
    task_Assignments_info,
    episode_spatial_risk_log=None,
    replica_completion_log=None,
):
    # Always write/read relative to project_root, not cwd, not this script's folder.
    ensure_dirs()

    model_name = str(getattr(params, "model_summary", "model")).strip().lower()

    # ---------------------------
    # Results folder + filename
    # ---------------------------
    results_dir = os.path.join(RESULTS_DIR, "fixed_rate_results")
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(results_dir, f"{model_name}_results.xlsx")

    # ---------------------------
    # Load Servers (from data/)
    # ---------------------------
    servers_path = os.path.join(DATA_DIR, "server_info.xlsx")
    server_info = pd.read_excel(servers_path)

    # ---------------------------
    # Load Tasks (from data/)
    # ---------------------------
    task_path = os.path.join(DATA_DIR, "task_parameters.xlsx")
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"File not found: {task_path}")
    task_df = pd.read_excel(task_path)

    # ---------------------------
    # Params dataframe
    # ---------------------------
    params_data = {attr: [value] for attr, value in vars(params).items()}
    df_params = pd.DataFrame(params_data).transpose().reset_index()
    df_params.columns = ["Parameter", "Value"]

    # ---------------------------
    # Logs dataframe
    # expected: (episode, avg_reward, episodic_reward, avg_delay)
    # ---------------------------
    logs_rows = []
    for log in log_data:
        logs_rows.append({
            "Episode": log[0],
            "Avg Reward": log[1] if len(log) > 1 else None,
            "Episode Reward": log[2] if len(log) > 2 else None,
            "Avg Delay": log[3] if len(log) > 3 else None,
        })
    df_logs = pd.DataFrame(logs_rows)

    # ---------------------------
    # Episode spatial-risk audit dataframe (optional)
    # ---------------------------
    df_spatial_risk = None
    if episode_spatial_risk_log is not None and len(episode_spatial_risk_log) > 0:
        df_spatial_risk = pd.DataFrame(episode_spatial_risk_log).rename(
            columns={
                "episode": "Episode",
                "server_id": "Server_ID",
                "spatial_risk_enabled": "Spatial_Risk_Enabled",
                "correlation_length_km": "Correlation_Length_km",
                "beta_p": "Beta_p",
                "spatial_risk_seed": "Spatial_Risk_Seed",
                "base_failure_rate": "Base_Failure_Rate",
                "failure_rate_scale": "Failure_Rate_Scale",
                "scaled_base_failure_rate": "Scaled_Base_Failure_Rate",
                "z_phy": "Z_phy",
                "spatial_hazard_multiplier": "Spatial_Hazard_Multiplier",
                "hazard_multiplier": "Hazard_Multiplier",
                "effective_failure_rate": "Effective_Failure_Rate",
            }
        )
        spatial_risk_columns = [
            "Episode",
            "Server_ID",
            "Spatial_Risk_Enabled",
            "Correlation_Length_km",
            "Beta_p",
            "Spatial_Risk_Seed",
            "Base_Failure_Rate",
            "Failure_Rate_Scale",
            "Scaled_Base_Failure_Rate",
            "Z_phy",
            "Spatial_Hazard_Multiplier",
            "Hazard_Multiplier",
            "Effective_Failure_Rate",
        ]
        missing_spatial_columns = sorted(
            set(spatial_risk_columns).difference(df_spatial_risk.columns)
        )
        if missing_spatial_columns:
            raise ValueError(
                "episode_spatial_risk_log is missing required fields: "
                + ", ".join(missing_spatial_columns)
            )
        df_spatial_risk = df_spatial_risk[spatial_risk_columns]

    # ---------------------------
    # ReplicaCompletions dataframe (optional input, fixed output schema)
    # ---------------------------
    replica_completion_internal_columns = [
        "episode",
        "task_id",
        "action_index",
        "server_id",
        "replica_label",
        "finish_time",
        "service_time",
    ]
    raw_replica_log = (
        [] if replica_completion_log is None else list(replica_completion_log)
    )
    if not raw_replica_log:
        df_replica_completions = pd.DataFrame(
            columns=REPLICA_COMPLETION_COLUMNS
        )
    else:
        df_replica_raw = pd.DataFrame(raw_replica_log)
        missing_replica_columns = sorted(
            set(replica_completion_internal_columns).difference(
                df_replica_raw.columns
            )
        )
        if missing_replica_columns:
            raise ValueError(
                "replica_completion_log is missing required fields: "
                + ", ".join(missing_replica_columns)
            )
        df_replica_raw = df_replica_raw[replica_completion_internal_columns]
        for numeric_column in ("finish_time", "service_time"):
            numeric_values = pd.to_numeric(
                df_replica_raw[numeric_column], errors="coerce"
            )
            if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"replica_completion_log {numeric_column} must be finite"
                )
            if numeric_column == "finish_time" and (numeric_values < 0.0).any():
                raise ValueError(
                    "replica_completion_log finish_time must be non-negative"
                )
            if numeric_column == "service_time" and (numeric_values <= 0.0).any():
                raise ValueError(
                    "replica_completion_log service_time must be positive"
                )
        if (
            pd.to_numeric(df_replica_raw["finish_time"], errors="coerce")
            < pd.to_numeric(df_replica_raw["service_time"], errors="coerce")
        ).any():
            raise ValueError(
                "replica_completion_log finish_time must be >= service_time"
            )
        df_replica_completions = df_replica_raw.rename(
            columns={
                "episode": "Episode",
                "task_id": "Task_ID",
                "action_index": "Action_Index",
                "server_id": "Server_ID",
                "replica_label": "Replica_Label",
                "finish_time": "Finish_Time",
                "service_time": "Service_Time",
            }
        )[REPLICA_COMPLETION_COLUMNS].sort_values(
            ["Episode", "Task_ID", "Finish_Time", "Server_ID"],
            kind="mergesort",
        ).reset_index(drop=True)

    # ---------------------------
    # TaskAssignments dataframe
    # ---------------------------
    task_assignment_columns = [
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
    reward_diagnostic_columns = [
        "Base_Reward",
        "Reliability_Violation_Log10",
        "Reliability_Penalty",
    ]
    normalized_assignment_rows = []
    for assignment in task_Assignments_info:
        row = list(assignment)
        if len(row) == 29:
            # Compatibility with historical callers that predate action metadata.
            row.extend([None, None, None])
        if len(row) != len(task_assignment_columns):
            raise ValueError(
                "TaskAssignments rows must contain either 29 legacy fields or "
                f"{len(task_assignment_columns)} fields"
            )
        normalized_assignment_rows.append(row)
    df_task_Assignments = pd.DataFrame(
        normalized_assignment_rows,
        columns=task_assignment_columns,
    )

    if not df_task_Assignments.empty:
        # Final status is the task-level reliability-threshold outcome.
        df_task_Assignments["Final_status"] = np.where(
            df_task_Assignments["Reliability_Satisfied"].astype(bool),
            "success",
            "failure",
        )
    else:
        df_task_Assignments["Final_status"] = []
    # Preserve the existing 27-column prefix and append new diagnostics at the end.
    df_task_Assignments = df_task_Assignments[
        task_assignment_columns[:26]
        + ["Final_status"]
        + reward_diagnostic_columns
        + ["action_index", "server_j", "server_k"]
    ]

    reliability_diagnostics_df = build_reliability_diagnostics(df_task_Assignments)
    pair_diagnostics_df = build_pair_diagnostics(df_task_Assignments)

    # ---------------------------
    # Summary: counts per episode + AVG_Failure (rolling mean 40)
    # ---------------------------
    if not df_task_Assignments.empty:
        summary_df = df_task_Assignments.groupby(["episode", "Final_status"]).size().unstack(fill_value=0)
    else:
        summary_df = pd.DataFrame()

    if "failure" not in summary_df.columns:
        summary_df["failure"] = 0
    if "success" not in summary_df.columns:
        summary_df["success"] = 0

    summary_df = summary_df.rename(columns={"failure": "Failure", "success": "Success"}).reset_index()

    # sort for rolling mean
    if not summary_df.empty and "episode" in summary_df.columns:
        summary_df = summary_df.sort_values("episode").reset_index(drop=True)

    if not summary_df.empty:
        summary_df["AVG_Failure"] = summary_df["Failure"].rolling(window=40, min_periods=1).mean()
    else:
        summary_df["AVG_Failure"] = []

    # ---------------------------
    # Write Excel
    # ---------------------------
    with pd.ExcelWriter(filename) as writer:
        df_params.to_excel(writer, sheet_name="Params", index=False)
        task_df.to_excel(writer, sheet_name="Tasks", index=False)
        server_info.to_excel(writer, sheet_name="Servers", index=False)
        df_logs.to_excel(writer, sheet_name="Logs", index=False)
        df_task_Assignments.to_excel(writer, sheet_name="TaskAssignments", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        reliability_diagnostics_df.to_excel(writer, sheet_name="ReliabilityDiagnostics", index=False)
        pair_diagnostics_df.to_excel(writer, sheet_name="PairDiagnostics", index=False)
        df_replica_completions.to_excel(
            writer, sheet_name="ReplicaCompletions", index=False
        )
        if df_spatial_risk is not None:
            df_spatial_risk.to_excel(writer, sheet_name="SpatialRisk", index=False)

    # ---------------------------
    # Add Excel-native charts
    # ---------------------------
    wb = load_workbook(filename)

    # Summary: ONLY line chart for AVG_Failure
    if "Summary" in wb.sheetnames:
        ws_sum = wb["Summary"]

        max_row = summary_df.shape[0] + 1  # header included
        # columns: episode | Failure | Success | AVG_Failure
        if max_row >= 2:
            line = LineChart()
            line.title = "Average Failure Over Episodes (Rolling 40)"
            line.x_axis.title = "Episode"
            line.y_axis.title = "AVG_Failure"

            line_data = Reference(ws_sum, min_col=4, min_row=1, max_col=4, max_row=max_row)  # AVG_Failure
            line_cats = Reference(ws_sum, min_col=1, min_row=2, max_row=max_row)             # episode

            line.add_data(line_data, titles_from_data=True)
            line.set_categories(line_cats)
            line.width = 22
            line.height = 10

            # جایگذاری از بالا (چون بارچارت حذف شده، بهتره همون بالا باشه)
            ws_sum.add_chart(line, "F2")

    # Logs charts (Rewards + optional Delay) - unchanged
    if "Logs" in wb.sheetnames:
        ws_logs = wb["Logs"]
        header = [cell.value for cell in ws_logs[1]]

        def col_idx(name):
            try:
                return header.index(name) + 1
            except ValueError:
                return None

        c_episode = col_idx("Episode")
        c_avg_reward = col_idx("Avg Reward")
        c_ep_reward = col_idx("Episode Reward")
        c_avg_delay = col_idx("Avg Delay")

        max_row_logs = ws_logs.max_row

        # Rewards chart
        if c_episode and (c_avg_reward or c_ep_reward) and max_row_logs >= 2:
            rewards_chart = LineChart()
            rewards_chart.title = "Rewards per Episode"
            rewards_chart.y_axis.title = "Reward"
            rewards_chart.x_axis.title = "Episode"

            min_col = min([c for c in [c_avg_reward, c_ep_reward] if c is not None])
            max_col = max([c for c in [c_avg_reward, c_ep_reward] if c is not None])

            data = Reference(ws_logs, min_col=min_col, min_row=1, max_col=max_col, max_row=max_row_logs)
            cats = Reference(ws_logs, min_col=c_episode, min_row=2, max_row=max_row_logs)

            rewards_chart.add_data(data, titles_from_data=True)
            rewards_chart.set_categories(cats)

            ws_logs.add_chart(rewards_chart, "F2")

        # Delay chart
        if c_episode and c_avg_delay and max_row_logs >= 2:
            delay_chart = LineChart()
            delay_chart.title = "Avg Delay per Episode"
            delay_chart.y_axis.title = "Delay"
            delay_chart.x_axis.title = "Episode"

            data = Reference(ws_logs, min_col=c_avg_delay, min_row=1, max_col=c_avg_delay, max_row=max_row_logs)
            cats = Reference(ws_logs, min_col=c_episode, min_row=2, max_row=max_row_logs)

            delay_chart.add_data(data, titles_from_data=True)
            delay_chart.set_categories(cats)

            ws_logs.add_chart(delay_chart, "F20")

    wb.save(filename)
    print("successfully saved logs !")
