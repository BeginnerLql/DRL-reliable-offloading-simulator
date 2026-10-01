"""Write task outcomes after both replicas finish, with fixed-schema logs."""

import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference

from config.paths import DATA_DIR, RESULTS_DIR, ensure_dirs


def save_params_and_logs(params, log_data, task_results):
    ensure_dirs()

    model_name = str(params.model_summary).strip().lower()

    results_dir = os.path.join(RESULTS_DIR, "fixed_rate_results")
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(results_dir, f"{model_name}_results.xlsx")

    servers_path = os.path.join(DATA_DIR, "server_info.xlsx")
    server_info = pd.read_excel(servers_path)

    task_path = os.path.join(DATA_DIR, "task_parameters.xlsx")
    task_df = pd.read_excel(task_path)

    params_data = {attr: [value] for attr, value in vars(params).items()}
    df_params = pd.DataFrame(params_data).transpose().reset_index()
    df_params.columns = ["Parameter", "Value"]

    # Logs dataframe: one raw episode reward and current episode mean task latency.
    df_logs = pd.DataFrame(
        [
            {
                "Episode": episode,
                "Episode Reward": episodic_reward,
                "task_Avg_Delay": task_avg_delay,
            }
            for episode, episodic_reward, task_avg_delay in log_data
        ],
        columns=["Episode", "Episode Reward", "task_Avg_Delay"],
    )

    task_result_columns = [
        "episode",
        "task_id",
        "task_size",
        "computation_demand",
        "reliability_requirement",
        "arrival_time",
        "server_A_id",
        "server_B_id",
        "replica_A_queue_enter_time",
        "replica_A_cpu_start_time",
        "replica_A_finish_time",
        "replica_B_queue_enter_time",
        "replica_B_cpu_start_time",
        "replica_B_finish_time",
        "task_completion_time",
        "task_latency",
        "replica_A_reliability",
        "replica_B_reliability",
        "pair_reliability",
        "requirement_satisfied",
    ]
    df_task_results = pd.DataFrame(task_results, columns=task_result_columns)

    with pd.ExcelWriter(filename) as writer:
        df_params.to_excel(writer, sheet_name="Params", index=False)
        task_df.to_excel(writer, sheet_name="Tasks", index=False)
        server_info.to_excel(writer, sheet_name="Servers", index=False)
        df_logs.to_excel(writer, sheet_name="Logs", index=False)
        df_task_results.to_excel(writer, sheet_name="TaskResults", index=False)

    wb = load_workbook(filename)

    ws_logs = wb["Logs"]
    for column, title, axis_title, anchor in (
        (2, "Rewards per Episode", "Reward", "F2"),
        (3, "task_Avg_Delay per Episode", "task_Avg_Delay (s)", "F20"),
    ):
        chart = LineChart()
        chart.title = title
        chart.y_axis.title = axis_title
        chart.x_axis.title = "Episode"
        data = Reference(
            ws_logs, min_col=column, min_row=1,
            max_col=column, max_row=ws_logs.max_row,
        )
        categories = Reference(ws_logs, min_col=1, min_row=2, max_row=ws_logs.max_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(categories)
        ws_logs.add_chart(chart, anchor)

    wb.save(filename)
    print("successfully saved logs !")
