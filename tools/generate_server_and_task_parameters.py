# generate_server_and_task_parameters.py
# Generates:
# 1) server_info.xlsx -> one sheet: Servers
# 2) task_parameters.xlsx

import os
import pandas as pd
import random
import numpy as np
from scipy.stats import truncnorm

from config.configuration import parameters
from config.paths import DATA_DIR, ensure_dirs


NUM_EDGE_SERVERS = parameters.NUM_EDGE_SERVERS
NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS


def generate_processing_frequencies(number_of_server: int, server_type: str):
    """Generate processing frequencies for edge or cloud servers."""
    if server_type.lower() == "edge":
        return [round(random.uniform(*parameters.EDGE_PROCESSING_FREQ_RANGE), 2) for _ in range(number_of_server)]
    elif server_type.lower() == "cloud":
        return [round(random.uniform(*parameters.CLOUD_PROCESSING_FREQ_RANGE), 2) for _ in range(number_of_server)]
    else:
        raise ValueError("server_type must be 'edge' or 'cloud'")


def generate_server_info(filename: str):
    """Write one sheet of servers with fixed base failure rates (1/s)."""
    server_info = []
    for server_type, count, rate_range in (
        ("Edge", NUM_EDGE_SERVERS, parameters.EDGE_FAILURE_RATE_RANGE),
        ("Cloud", NUM_CLOUD_SERVERS, parameters.CLOUD_FAILURE_RATE_RANGE),
    ):
        frequencies = generate_processing_frequencies(count, server_type)
        for frequency in frequencies:
            failure_rate = random.uniform(*rate_range)
            server_info.append([len(server_info) + 1, server_type, frequency, failure_rate])

    df = pd.DataFrame(server_info, columns=[
        "Server_ID", "Server_Type", "Processing_Frequency", "Failure_Rate"
    ])
    df.to_excel(filename, sheet_name="Servers", index=False)


def generate_task_params(filename: str = "task_parameters.xlsx"):
    """
    Generate task parameters Excel:
      Task_ID, Task_Size, Computation_Demand
    """
    task_info = []
    NUM_TASKS = parameters.taskno
    TASK_SIZE_RANGE = parameters.TASK_SIZE_RANGE

    a, b = parameters.Low_demand, parameters.High_demand
    mu = (a + b) / 2
    sigma = (b - a) / 6
    lower, upper = (a - mu) / sigma, (b - mu) / sigma

    for task_id in range(1, NUM_TASKS + 1):
        # Task_Size (integer)
        task_size = np.random.randint(TASK_SIZE_RANGE[0], TASK_SIZE_RANGE[1] + 1)

        # Computation_Demand (float)
        computation_demand = truncnorm.rvs(lower, upper, loc=mu, scale=sigma)

        task_info.append([task_id, task_size, float(computation_demand)])

    task_df = pd.DataFrame(task_info, columns=["Task_ID", "Task_Size", "Computation_Demand"])
    task_df.to_excel(filename, index=False)


def main():
    """Write all Excel parameter files into data/... (not project root)."""
    ensure_dirs()

    servers_path = os.path.join(DATA_DIR, "server_info.xlsx")
    tasks_path = os.path.join(DATA_DIR, "task_parameters.xlsx")

    generate_server_info(servers_path)
    generate_task_params(tasks_path)
    print("Parameters defined in Excel files!")


if __name__ == "__main__":
    main()
