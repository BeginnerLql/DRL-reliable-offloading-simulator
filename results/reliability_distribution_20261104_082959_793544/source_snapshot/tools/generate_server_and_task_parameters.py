"""Generate unified server and synthetic task parameter workbooks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.configuration import parameters
from config.paths import (
    DATA_DIR,
    EVALUATION_TASK_PATH,
    SERVER_INFO_PATH,
    TRAIN_TASK_PATH,
    ensure_dirs,
)

# Workbook units: Processing_Frequency in cycles/s, Transmission_Rate in MB/s,
# Base_Failure_Rate in s^-1; Task_Size in MB, Computation_Demand in CPU cycles.
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
    """Sample CPU capacity in cycles/s from the configured range."""
    return _profile_rng(0, 0).uniform(
        *parameters.SERVER_PROCESSING_FREQ_RANGE, size=number_of_servers
    ).tolist()


def _profile_rng(profile_stream: int, parameter_stream: int) -> np.random.Generator:
    """Stable SeedSequence keys: 0=servers, 1=train tasks, 2=evaluation tasks.

    Each parameter has its own child stream, recreated on every generation call.
    Global RNG calls and generation order therefore cannot alter these profiles.
    """
    return np.random.default_rng(np.random.SeedSequence(
        parameters.PROFILE_SEED, spawn_key=(profile_stream, parameter_stream)
    ))


def generate_server_info(
    filename: str | Path,
    topology_path: str | Path | None = None,
    num_servers: int | None = None,
) -> pd.DataFrame:
    """Write unified server parameters with selected EUA locations."""
    count = parameters.NUM_SERVERS if num_servers is None else num_servers
    topology_df = load_eua_topology(topology_path, count)
    processing_frequencies_cycles_per_second = generate_processing_frequencies(count)
    transmission_rates_mb_per_second = _profile_rng(0, 1).uniform(
        *parameters.SERVER_TRANSMISSION_RATE_RANGE, size=count
    )
    failure_rates_per_second = _profile_rng(0, 2).uniform(
        *parameters.SERVER_FAILURE_RATE_RANGE, size=count
    )
    server_info = topology_df.copy()
    server_info["Processing_Frequency"] = processing_frequencies_cycles_per_second
    server_info["Base_Failure_Rate"] = failure_rates_per_second
    server_info["Transmission_Rate"] = transmission_rates_mb_per_second
    server_info = server_info[SERVER_INFO_COLUMNS]
    server_info.to_excel(filename, sheet_name="Servers", index=False)
    return server_info


def generate_task_params(
    filename: str | Path | None = None,
    *,
    dataset: str = "train",
) -> pd.DataFrame:
    """Write fixed tasks with independent uniform size, demand and reliability."""
    profile_stream = {"train": 1, "evaluation": 2}[dataset]
    if filename is None:
        filename = TRAIN_TASK_PATH if dataset == "train" else EVALUATION_TASK_PATH
    num_tasks = parameters.taskno
    reliability_requirements = _profile_rng(profile_stream, 2).uniform(
        *parameters.TASK_RELIABILITY_REQUIREMENT_RANGE, size=num_tasks
    )

    task_sizes_mb = _profile_rng(profile_stream, 0).uniform(
        *parameters.TASK_SIZE_RANGE, size=num_tasks
    )
    computation_demands_cycles = _profile_rng(profile_stream, 1).uniform(
        *parameters.TASK_DEMAND_RANGE, size=num_tasks
    )
    task_df = pd.DataFrame({
        "Task_ID": np.arange(1, num_tasks + 1),
        "Task_Size": task_sizes_mb,
        "Computation_Demand": computation_demands_cycles,
        "Reliability_Requirement": reliability_requirements,
    }, columns=TASK_INFO_COLUMNS)
    task_df.to_excel(filename, index=False)
    return task_df


def main() -> None:
    """Write server, training-task and evaluation-task workbooks."""
    ensure_dirs()
    generate_server_info(SERVER_INFO_PATH)
    generate_task_params(dataset="train")
    generate_task_params(dataset="evaluation")
    print("Server, training-task and evaluation-task parameters saved to data/.")

if __name__ == "__main__":
    main()
