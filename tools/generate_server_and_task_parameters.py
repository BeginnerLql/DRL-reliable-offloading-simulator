"""Generate unified server and synthetic task parameter workbooks."""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.configuration import parameters
from config.paths import DATA_DIR, ensure_dirs

SERVER_INFO_COLUMNS = [
    "Server_ID",
    "Site_ID",
    "Processing_Frequency",
    "Base_Failure_Rate",
    "Transmission_Rate",
    "Latitude",
    "Longitude",
]
TASK_INFO_COLUMNS = [
    "Task_ID",
    "Task_Size",
    "Computation_Demand",
    "Reliability_Requirement",
]


def _default_topology_path(num_servers: int) -> Path:
    return Path(DATA_DIR) / f"eua_melbourne_cbd_selected_{num_servers}.csv"


def load_eua_topology(
    topology_path: str | Path | None = None,
    num_servers: int | None = None,
) -> pd.DataFrame:
    """Read the selected topology for the configured server count."""
    count = parameters.NUM_SERVERS if num_servers is None else num_servers
    resolved_path = (
        Path(topology_path)
        if topology_path is not None
        else _default_topology_path(count)
    )
    topology_df = pd.read_csv(resolved_path, dtype={"Site_ID": "string"})
    return topology_df.sort_values("Server_ID").reset_index(drop=True)


def generate_processing_frequencies(number_of_servers: int) -> list[float]:
    """Generate frequencies from the unified server configuration."""
    return [
        round(random.uniform(*parameters.SERVER_PROCESSING_FREQ_RANGE), 2)
        for _ in range(number_of_servers)
    ]


def generate_server_info(
    filename: str | Path,
    topology_path: str | Path | None = None,
    num_servers: int | None = None,
) -> pd.DataFrame:
    """Write unified server parameters with selected EUA locations."""
    count = parameters.NUM_SERVERS if num_servers is None else num_servers
    transmission_rates = parameters.SERVER_TRANSMISSION_RATES
    topology_df = load_eua_topology(topology_path, count)
    frequencies = generate_processing_frequencies(count)
    failure_rates = [
        random.uniform(*parameters.SERVER_FAILURE_RATE_RANGE)
        for _ in range(count)
    ]
    server_info = topology_df.copy()
    server_info["Processing_Frequency"] = frequencies
    server_info["Base_Failure_Rate"] = failure_rates
    transmission_rate_by_id = dict(enumerate(transmission_rates, start=1))
    server_info["Transmission_Rate"] = server_info["Server_ID"].map(
        transmission_rate_by_id
    )
    server_info = server_info[SERVER_INFO_COLUMNS]
    server_info.to_excel(filename, sheet_name="Servers", index=False)
    return server_info


def generate_task_params(filename: str | Path = "task_parameters.xlsx") -> pd.DataFrame:
    """Write synthetic task parameters with reliability requirements."""
    task_info = []
    num_tasks = parameters.taskno
    size_range = parameters.TASK_SIZE_RANGE
    reliability_levels = parameters.TASK_RELIABILITY_REQUIREMENT_LEVELS
    if num_tasks % len(reliability_levels):
        raise ValueError("taskno must be divisible by the number of reliability levels.")
    reliability_requirements = np.random.permutation(
        np.repeat(reliability_levels, num_tasks // len(reliability_levels))
    )

    lower_demand, upper_demand = parameters.Low_demand, parameters.High_demand
    mean = (lower_demand + upper_demand) / 2
    standard_deviation = (upper_demand - lower_demand) / 6
    lower_bound = (lower_demand - mean) / standard_deviation
    upper_bound = (upper_demand - mean) / standard_deviation

    for task_id, reliability_requirement in enumerate(reliability_requirements, start=1):
        task_size = np.random.randint(size_range[0], size_range[1] + 1)
        computation_demand = truncnorm.rvs(
            lower_bound,
            upper_bound,
            loc=mean,
            scale=standard_deviation,
        )
        task_info.append([
            task_id,
            task_size,
            float(computation_demand),
            float(reliability_requirement),
        ])

    task_df = pd.DataFrame(task_info, columns=TASK_INFO_COLUMNS)
    task_df.to_excel(filename, index=False)
    return task_df


def main() -> None:
    """Write server and task parameter files into the data directory."""
    random.seed(parameters.PROFILE_SEED)
    np.random.seed(parameters.PROFILE_SEED)
    ensure_dirs()
    generate_server_info(Path(DATA_DIR) / "server_info.xlsx")
    generate_task_params(Path(DATA_DIR) / "task_parameters.xlsx")
    print("Parameters defined in Excel files!")

if __name__ == "__main__":
    main()
