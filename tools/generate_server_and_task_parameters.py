"""Generate unified server and synthetic task parameter workbooks."""

import math
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


TOPOLOGY_COLUMNS = ["Server_ID", "Site_ID", "Latitude", "Longitude"]
SERVER_INFO_COLUMNS = [
    "Server_ID",
    "Site_ID",
    "Processing_Frequency",
    "Base_Failure_Rate",
    "Latitude",
    "Longitude",
]
TASK_INFO_COLUMNS = [
    "Task_ID",
    "Task_Size",
    "Computation_Demand",
    "Reliability_Requirement",
]


def _resolve_num_servers(num_servers: int | None) -> int:
    count = parameters.NUM_SERVERS if num_servers is None else num_servers
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 2:
        raise ValueError("num_servers must be an integer greater than or equal to 2.")
    return int(count)


def _default_topology_path(num_servers: int) -> Path:
    return Path(DATA_DIR) / f"eua_melbourne_cbd_selected_{num_servers}.csv"


def _validate_topology(
    topology_df: pd.DataFrame, topology_path: Path, num_servers: int
) -> pd.DataFrame:
    if list(topology_df.columns) != TOPOLOGY_COLUMNS:
        raise ValueError(
            f"Topology file {topology_path} must have exactly these columns in order: "
            + ", ".join(TOPOLOGY_COLUMNS)
        )
    if len(topology_df) != num_servers:
        raise ValueError(
            f"Topology file {topology_path} must contain {num_servers} rows; "
            f"found {len(topology_df)}."
        )

    frame = topology_df.copy()
    server_ids = pd.to_numeric(frame["Server_ID"], errors="coerce")
    if (
        server_ids.isna().any()
        or not server_ids.map(math.isfinite).all()
        or not server_ids.map(lambda value: float(value).is_integer()).all()
    ):
        raise ValueError(f"Topology file {topology_path} contains invalid Server_ID values.")
    frame["Server_ID"] = server_ids.astype(int)
    expected_ids = list(range(1, num_servers + 1))
    if (
        frame["Server_ID"].duplicated().any()
        or sorted(frame["Server_ID"].tolist()) != expected_ids
    ):
        raise ValueError(
            f"Topology file {topology_path} must have unique Server_ID values 1..N."
        )

    frame["Site_ID"] = frame["Site_ID"].astype("string").str.strip()
    if frame["Site_ID"].isna().any() or frame["Site_ID"].eq("").any():
        raise ValueError(f"Topology file {topology_path} contains an empty Site_ID.")
    if frame["Site_ID"].duplicated().any():
        raise ValueError(f"Topology file {topology_path} contains duplicate Site_ID values.")

    for column, lower, upper in (
        ("Latitude", -90.0, 90.0),
        ("Longitude", -180.0, 180.0),
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not frame[column].map(math.isfinite).all():
            raise ValueError(f"Topology file {topology_path} contains invalid {column} values.")
        if not frame[column].between(lower, upper).all():
            raise ValueError(
                f"Topology file {topology_path} contains out-of-range {column} values."
            )

    return frame.sort_values("Server_ID").reset_index(drop=True)[TOPOLOGY_COLUMNS]


def load_eua_topology(
    topology_path: str | Path | None = None,
    num_servers: int | None = None,
) -> pd.DataFrame:
    """Read and validate the selected topology for the configured server count."""
    count = _resolve_num_servers(num_servers)
    resolved_path = (
        Path(topology_path)
        if topology_path is not None
        else _default_topology_path(count)
    )
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Selected EUA topology not found: {resolved_path}. "
            "Run tools/prepare_eua_topology.py with the raw EUA CSV first."
        )
    topology_df = pd.read_csv(resolved_path, dtype={"Site_ID": "string"})
    return _validate_topology(topology_df, resolved_path, count)


def generate_processing_frequencies(number_of_servers: int) -> list[float]:
    """Generate frequencies from the unified server configuration."""
    if (
        isinstance(number_of_servers, bool)
        or not isinstance(number_of_servers, (int, np.integer))
        or number_of_servers < 0
    ):
        raise ValueError("number_of_servers must be a non-negative integer.")
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
    count = _resolve_num_servers(num_servers)
    topology_df = load_eua_topology(topology_path, count)
    frequencies = generate_processing_frequencies(count)
    failure_rates = [
        random.uniform(*parameters.SERVER_FAILURE_RATE_RANGE)
        for _ in range(count)
    ]
    server_info = topology_df.copy()
    server_info["Processing_Frequency"] = frequencies
    server_info["Base_Failure_Rate"] = failure_rates
    server_info = server_info[SERVER_INFO_COLUMNS]
    server_info.to_excel(filename, sheet_name="Servers", index=False)
    return server_info


def generate_task_params(filename: str | Path = "task_parameters.xlsx") -> pd.DataFrame:
    """Write synthetic task parameters and leave reliability requirements unset."""
    task_info = []
    num_tasks = parameters.taskno
    size_range = parameters.TASK_SIZE_RANGE

    lower_demand, upper_demand = parameters.Low_demand, parameters.High_demand
    mean = (lower_demand + upper_demand) / 2
    standard_deviation = (upper_demand - lower_demand) / 6
    lower_bound = (lower_demand - mean) / standard_deviation
    upper_bound = (upper_demand - mean) / standard_deviation

    # Reliability requirement distribution is intentionally unspecified for now.
    for task_id in range(1, num_tasks + 1):
        task_size = np.random.randint(size_range[0], size_range[1] + 1)
        computation_demand = truncnorm.rvs(
            lower_bound,
            upper_bound,
            loc=mean,
            scale=standard_deviation,
        )
        task_info.append([task_id, task_size, float(computation_demand), None])

    task_df = pd.DataFrame(task_info, columns=TASK_INFO_COLUMNS)
    task_df["Reliability_Requirement"] = pd.Series(
        [None] * num_tasks, dtype=object
    )
    task_df.to_excel(filename, index=False)
    return task_df


def main() -> None:
    """Write server and task parameter files into the data directory."""
    ensure_dirs()
    generate_server_info(Path(DATA_DIR) / "server_info.xlsx")
    generate_task_params(Path(DATA_DIR) / "task_parameters.xlsx")
    print("Parameters defined in Excel files!")


if __name__ == "__main__":
    main()
