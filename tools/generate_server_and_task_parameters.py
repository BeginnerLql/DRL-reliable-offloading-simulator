"""Generate fixed server profiles and independent training/evaluation tasks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Read selected EUA locations in Server_ID order."""
    count = parameters.NUM_SERVERS if num_servers is None else num_servers
    resolved_path = (
        Path(topology_path)
        if topology_path is not None
        else _default_topology_path(count)
    )
    topology_df = pd.read_csv(resolved_path, dtype={"Site_ID": "string"})
    return topology_df.sort_values("Server_ID").reset_index(drop=True)


def generate_server_info(
    filename: str | Path,
    topology_path: str | Path | None = None,
    num_servers: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate one frozen server profile; CPU capacity is in cycles/s."""
    count = parameters.NUM_SERVERS if num_servers is None else num_servers
    rng = np.random.default_rng(parameters.PROFILE_SEED) if rng is None else rng
    server_info = load_eua_topology(topology_path, count).copy()
    server_info["Processing_Frequency"] = rng.uniform(
        *parameters.SERVER_PROCESSING_FREQ_RANGE, size=count
    )
    server_info["Base_Failure_Rate"] = rng.uniform(
        *parameters.SERVER_FAILURE_RATE_RANGE, size=count
    )
    server_info["Transmission_Rate"] = rng.uniform(
        *parameters.SERVER_TRANSMISSION_RATE_RANGE, size=count
    )
    server_info = server_info[SERVER_INFO_COLUMNS]
    server_info.to_excel(filename, sheet_name="Servers", index=False)
    return server_info


def generate_task_params(
    filename: str | Path = "task_parameters.xlsx",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate one fixed task set with a balanced, shuffled tier assignment."""
    rng = np.random.default_rng(parameters.PROFILE_SEED) if rng is None else rng
    num_tasks = parameters.taskno
    levels = parameters.TASK_RELIABILITY_REQUIREMENT_LEVELS
    if num_tasks % len(levels):
        raise ValueError("taskno must be divisible by the number of reliability levels.")

    task_df = pd.DataFrame({
        "Task_ID": np.arange(1, num_tasks + 1, dtype=int),
        "Task_Size": rng.uniform(*parameters.TASK_SIZE_RANGE, size=num_tasks),
        "Computation_Demand": rng.uniform(
            *parameters.COMPUTATION_DEMAND_RANGE, size=num_tasks
        ),
        "Reliability_Requirement": rng.permutation(
            np.repeat(levels, num_tasks // len(levels))
        ),
    })[TASK_INFO_COLUMNS]
    task_df.to_excel(filename, index=False)
    return task_df


def main() -> None:
    """Write one server workbook and independent fixed train/eval task workbooks."""
    ensure_dirs()
    server_seed, train_seed, eval_seed = np.random.SeedSequence(
        parameters.PROFILE_SEED
    ).spawn(3)
    generate_server_info(
        Path(DATA_DIR) / "server_info.xlsx",
        rng=np.random.default_rng(server_seed),
    )
    generate_task_params(
        Path(DATA_DIR) / "task_parameters.xlsx",
        rng=np.random.default_rng(train_seed),
    )
    generate_task_params(
        Path(DATA_DIR) / "task_parameters_eval.xlsx",
        rng=np.random.default_rng(eval_seed),
    )
    print("Generated frozen server, training-task and evaluation-task profiles.")


if __name__ == "__main__":
    main()
