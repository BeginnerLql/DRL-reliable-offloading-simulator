# generate_server_and_task_parameters.py
# Generates:
# 1) server_info.xlsx -> one sheet: Servers
# 2) task_parameters.xlsx

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config.configuration import parameters
from config.paths import DATA_DIR, ensure_dirs


NUM_EDGE_SERVERS = parameters.NUM_EDGE_SERVERS
NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS
TOPOLOGY_FILENAME = "eua_melbourne_cbd_site_order.csv"
SERVER_INFO_COLUMNS = [
    "Server_ID",
    "Server_Type",  # compatibility column; every formal node is Edge
    "Processing_Frequency",
    "Base_Failure_Rate",
    "Uplink_Rate",
    "Latitude",
    "Longitude",
]
FIXED_PROCESSING_FREQUENCIES = tuple(parameters.FIXED_EDGE_PROCESSING_FREQUENCIES)
FIXED_UPLINK_RATES_MBPS = (18, 30, 22, 36, 16, 28, 40, 24)
RELIABILITY_REQUIREMENT_LEVELS = (0.9, 0.99, 0.999, 0.9999)
RELIABILITY_REQUIREMENT_SEED = 2026


def generate_computation_demands(
    num_tasks: int,
    seed: int = parameters.COMPUTATION_DEMAND_SEED,
) -> list[int]:
    """Return independent discrete-uniform computation demands in MI."""
    if isinstance(num_tasks, (bool, np.bool_)) or not isinstance(num_tasks, (int, np.integer)):
        raise ValueError("num_tasks must be a positive integer")
    if num_tasks <= 0:
        raise ValueError("num_tasks must be a positive integer")

    lower, upper = parameters.COMPUTATION_DEMAND_RANGE_MI
    rng = np.random.default_rng(seed)
    # numpy's integer upper bound is exclusive, so add one to include 50.
    return rng.integers(int(lower), int(upper) + 1, size=int(num_tasks)).astype(int).tolist()


def generate_reliability_requirements(
    num_tasks: int,
    seed: int = RELIABILITY_REQUIREMENT_SEED,
) -> list[float]:
    """Return a reproducibly shuffled, equally balanced requirement list."""
    if isinstance(num_tasks, (bool, np.bool_)) or not isinstance(num_tasks, (int, np.integer)):
        raise ValueError("num_tasks must be a positive integer")
    if num_tasks <= 0 or num_tasks % len(RELIABILITY_REQUIREMENT_LEVELS) != 0:
        raise ValueError(
            "num_tasks must be a positive multiple of "
            f"{len(RELIABILITY_REQUIREMENT_LEVELS)}"
        )

    per_level = num_tasks // len(RELIABILITY_REQUIREMENT_LEVELS)
    requirements = np.repeat(RELIABILITY_REQUIREMENT_LEVELS, per_level).astype(float)
    np.random.default_rng(seed).shuffle(requirements)
    return requirements.tolist()


def generate_processing_frequencies(number_of_server: int, server_type: str = "Edge"):
    """Return the fixed processing frequencies of the formal Edge nodes."""
    if str(server_type).lower() != "edge":
        raise ValueError("the formal server model contains Edge nodes only")
    count = int(number_of_server)
    if count < 0 or count > len(FIXED_PROCESSING_FREQUENCIES):
        raise ValueError("number_of_server exceeds the fixed Edge frequency table")
    return list(FIXED_PROCESSING_FREQUENCIES[:count])


def base_failure_rate_from_frequency(processing_frequency: float) -> float:
    """Compute lambda_base for a normal environment from fixed CPU capacity."""
    frequency = float(processing_frequency)
    f_min = float(parameters.FAILURE_RATE_FMIN)
    f_max = float(parameters.FAILURE_RATE_FMAX)
    exponent = float(parameters.FAILURE_RATE_OMEGA) * (
        1.0 - frequency / f_max
    ) / (1.0 - f_min / f_max)
    return float(parameters.LAMBDA_REF * 10.0 ** exponent)


def _default_topology_path() -> Path:
    return Path(DATA_DIR) / TOPOLOGY_FILENAME


def _validate_topology(topology_df: pd.DataFrame, topology_path: Path) -> pd.DataFrame:
    required_columns = {"Rank", "Latitude", "Longitude"}
    missing_columns = sorted(required_columns.difference(topology_df.columns))
    if missing_columns:
        raise ValueError(
            f"Topology file {topology_path} is missing required columns: "
            + ", ".join(missing_columns)
        )

    topology_df = topology_df.copy()
    topology_df["Rank"] = pd.to_numeric(topology_df["Rank"], errors="coerce")
    if topology_df["Rank"].isna().any() or not topology_df["Rank"].map(lambda value: float(value).is_integer()).all():
        raise ValueError(f"Topology file {topology_path} contains invalid Rank values.")
    topology_df["Rank"] = topology_df["Rank"].astype(int)

    if topology_df["Rank"].duplicated().any():
        raise ValueError(f"Topology file {topology_path} contains duplicate Rank values.")

    expected_ranks = list(range(1, len(topology_df) + 1))
    if topology_df["Rank"].sort_values().tolist() != expected_ranks:
        raise ValueError(
            f"Topology file {topology_path} must have continuous Rank values 1..N."
        )

    for column, lower, upper in (
        ("Latitude", -90.0, 90.0),
        ("Longitude", -180.0, 180.0),
    ):
        topology_df[column] = pd.to_numeric(topology_df[column], errors="coerce")
        if topology_df[column].isna().any() or not topology_df[column].map(math.isfinite).all():
            raise ValueError(
                f"Topology file {topology_path} contains invalid {column} values."
            )
        if not topology_df[column].between(lower, upper).all():
            raise ValueError(
                f"Topology file {topology_path} contains out-of-range {column} values."
            )

    return topology_df.sort_values("Rank").reset_index(drop=True)


def load_eua_topology(topology_path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load and validate the deterministic EUA site ordering."""
    resolved_path = Path(topology_path) if topology_path is not None else _default_topology_path()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"EUA topology file not found: {resolved_path}. "
            "Generate data/eua_melbourne_cbd_site_order.csv first."
        )

    topology_df = pd.read_csv(resolved_path)
    return _validate_topology(topology_df, resolved_path)


def assign_topology_locations(
    topology_df: pd.DataFrame,
    num_edge_servers: int,
    num_cloud_servers: int,
) -> pd.DataFrame:
    """Assign deterministic topology ranks while retaining Edge-first IDs."""
    if num_edge_servers < 0 or num_cloud_servers < 0:
        raise ValueError("Server counts must be non-negative.")

    total_servers = num_edge_servers + num_cloud_servers
    if len(topology_df) < total_servers:
        raise ValueError(
            "Not enough topology sites: "
            f"need {total_servers}, found {len(topology_df)}."
        )

    rank_to_site = topology_df.set_index("Rank")
    cloud_ranks = list(range(1, num_cloud_servers + 1))
    edge_ranks = list(
        range(num_cloud_servers + 1, num_cloud_servers + num_edge_servers + 1)
    )
    used_ranks = cloud_ranks + edge_ranks
    if len(used_ranks) != len(set(used_ranks)):
        raise ValueError("Topology rank assignment contains duplicate ranks.")

    assignments = []
    server_id = 1
    for server_type, ranks in (("Edge", edge_ranks), ("Cloud", cloud_ranks)):
        for rank in ranks:
            site = rank_to_site.loc[rank]
            assignments.append({
                "Server_ID": server_id,
                "Server_Type": server_type,
                "Topology_Rank": rank,
                "Latitude": float(site["Latitude"]),
                "Longitude": float(site["Longitude"]),
            })
            server_id += 1

    return pd.DataFrame(assignments)


def _default_all_edge_assignments(topology_df: pd.DataFrame) -> pd.DataFrame:
    """Return the current eight-node coordinate order as all-Edge records.

    The former six-Edge/two-Cloud layout used topology ranks 3..8 for IDs
    1..6 and ranks 1..2 for IDs 7..8.  Keeping that order preserves the
    existing eight coordinates while removing the old type distinction.
    """
    preferred_ranks = [3, 4, 5, 6, 7, 8, 1, 2]
    if len(topology_df) < len(preferred_ranks):
        raise ValueError(
            f"Not enough topology sites: need 8, found {len(topology_df)}."
        )
    by_rank = topology_df.set_index("Rank")
    rows = []
    for server_id, rank in enumerate(preferred_ranks, start=1):
        site = by_rank.loc[rank]
        rows.append({
            "Server_ID": server_id,
            "Server_Type": "Edge",
            "Topology_Rank": rank,
            "Latitude": float(site["Latitude"]),
            "Longitude": float(site["Longitude"]),
        })
    return pd.DataFrame(rows)


def _existing_coordinates(filename: str | os.PathLike) -> pd.DataFrame | None:
    """Load existing coordinates so migration never moves current nodes."""
    path = Path(filename)
    if not path.exists():
        return None
    try:
        frame = pd.read_excel(path)
    except Exception:
        return None
    required = {"Server_ID", "Latitude", "Longitude"}
    if not required.issubset(frame.columns):
        return None
    frame = frame[list(required)].copy()
    frame["Server_ID"] = pd.to_numeric(frame["Server_ID"], errors="coerce")
    frame["Latitude"] = pd.to_numeric(frame["Latitude"], errors="coerce")
    frame["Longitude"] = pd.to_numeric(frame["Longitude"], errors="coerce")
    if frame.isna().any().any() or frame["Server_ID"].duplicated().any():
        return None
    frame["Server_ID"] = frame["Server_ID"].astype(int)
    if set(frame["Server_ID"]) != set(range(1, NUM_EDGE_SERVERS + 1)):
        return None
    return frame.sort_values("Server_ID").reset_index(drop=True)


def generate_server_info(
    filename: str,
    topology_path: str | os.PathLike | None = None,
):
    """Write the fixed all-Edge server table with preserved coordinates."""
    topology_df = load_eua_topology(topology_path)
    existing = _existing_coordinates(filename)
    if existing is None:
        assignments = _default_all_edge_assignments(topology_df)
    else:
        assignments = existing.assign(Server_Type="Edge")

    frequencies = generate_processing_frequencies(NUM_EDGE_SERVERS, "Edge")
    server_info = []
    for frequency, (_, assignment) in zip(frequencies, assignments.iterrows()):
        server_info.append([
            int(assignment["Server_ID"]),
            "Edge",
            int(frequency),
            base_failure_rate_from_frequency(frequency),
            float(FIXED_UPLINK_RATES_MBPS[int(assignment["Server_ID"]) - 1]),
            float(assignment["Latitude"]),
            float(assignment["Longitude"]),
        ])

    df = pd.DataFrame(server_info, columns=SERVER_INFO_COLUMNS)
    df.to_excel(filename, sheet_name="Servers", index=False)


def generate_task_params(filename: str = "task_parameters.xlsx"):
    """
    Generate task parameters Excel:
      Task_ID, Input_Data_Size_MB, Computation_Demand, Reliability_Requirement
    """
    task_info = []
    NUM_TASKS = parameters.taskno
    INPUT_DATA_SIZE_RANGE_MB = parameters.INPUT_DATA_SIZE_RANGE_MB
    input_data_rng = np.random.default_rng(parameters.INPUT_DATA_SIZE_SEED)
    computation_demands = generate_computation_demands(NUM_TASKS)
    reliability_requirements = generate_reliability_requirements(NUM_TASKS)

    for task_id in range(1, NUM_TASKS + 1):
        # Input payload uses an independent deterministic RNG stream.
        input_data_size_mb = float(input_data_rng.uniform(*INPUT_DATA_SIZE_RANGE_MB))
        computation_demand = computation_demands[task_id - 1]

        task_info.append(
            [
                task_id,
                input_data_size_mb,
                int(computation_demand),
                reliability_requirements[task_id - 1],
            ]
        )

    task_df = pd.DataFrame(
        task_info,
        columns=[
            "Task_ID",
            "Input_Data_Size_MB",
            "Computation_Demand",
            "Reliability_Requirement",
        ],
    )
    task_df.to_excel(filename, index=False)


def main():
    """Write all server and task parameter files into data/."""
    ensure_dirs()

    servers_path = os.path.join(DATA_DIR, "server_info.xlsx")
    tasks_path = os.path.join(DATA_DIR, "task_parameters.xlsx")

    generate_server_info(servers_path)
    generate_task_params(tasks_path)
    print("Parameters defined in Excel files!")


if __name__ == "__main__":
    main()
