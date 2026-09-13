# generate_server_and_task_parameters.py
# Generates:
# 1) server_info.xlsx -> one sheet: Servers
# 2) task_parameters.xlsx

import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from config.configuration import parameters
from config.paths import DATA_DIR, ensure_dirs


NUM_EDGE_SERVERS = parameters.NUM_EDGE_SERVERS
NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS
TOPOLOGY_FILENAME = "eua_melbourne_cbd_site_order.csv"
SERVER_INFO_COLUMNS = [
    "Server_ID",
    "Server_Type",
    "Processing_Frequency",
    "Failure_Rate",
    "Latitude",
    "Longitude",
]
RELIABILITY_REQUIREMENT_LEVELS = (0.9, 0.99, 0.999, 0.9999)
RELIABILITY_REQUIREMENT_SEED = 2026


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


def generate_processing_frequencies(number_of_server: int, server_type: str):
    """Generate processing frequencies for edge or cloud servers."""
    if server_type.lower() == "edge":
        return [round(random.uniform(*parameters.EDGE_PROCESSING_FREQ_RANGE), 2) for _ in range(number_of_server)]
    elif server_type.lower() == "cloud":
        return [round(random.uniform(*parameters.CLOUD_PROCESSING_FREQ_RANGE), 2) for _ in range(number_of_server)]
    else:
        raise ValueError("server_type must be 'edge' or 'cloud'")


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


def generate_server_info(
    filename: str,
    topology_path: str | os.PathLike | None = None,
):
    """Write server parameters with deterministic EUA geographic positions."""
    topology_df = load_eua_topology(topology_path)
    assignments = assign_topology_locations(
        topology_df,
        NUM_EDGE_SERVERS,
        NUM_CLOUD_SERVERS,
    )

    server_info = []
    assignment_offset = 0
    for server_type, count, rate_range in (
        ("Edge", NUM_EDGE_SERVERS, parameters.EDGE_FAILURE_RATE_RANGE),
        ("Cloud", NUM_CLOUD_SERVERS, parameters.CLOUD_FAILURE_RATE_RANGE),
    ):
        frequencies = generate_processing_frequencies(count, server_type)
        type_assignments = assignments.iloc[assignment_offset:assignment_offset + count]
        for frequency, (_, assignment) in zip(frequencies, type_assignments.iterrows()):
            failure_rate = random.uniform(*rate_range)
            server_info.append([
                int(assignment["Server_ID"]),
                server_type,
                frequency,
                failure_rate,
                assignment["Latitude"],
                assignment["Longitude"],
            ])
        assignment_offset += count

    df = pd.DataFrame(server_info, columns=SERVER_INFO_COLUMNS)
    df.to_excel(filename, sheet_name="Servers", index=False)


def generate_task_params(filename: str = "task_parameters.xlsx"):
    """
    Generate task parameters Excel:
      Task_ID, Task_Size, Computation_Demand, Reliability_Requirement
    """
    task_info = []
    NUM_TASKS = parameters.taskno
    TASK_SIZE_RANGE = parameters.TASK_SIZE_RANGE
    reliability_requirements = generate_reliability_requirements(NUM_TASKS)

    a, b = parameters.Low_demand, parameters.High_demand
    mu = (a + b) / 2
    sigma = (b - a) / 6
    lower, upper = (a - mu) / sigma, (b - mu) / sigma

    for task_id in range(1, NUM_TASKS + 1):
        # Task_Size (integer)
        task_size = np.random.randint(TASK_SIZE_RANGE[0], TASK_SIZE_RANGE[1] + 1)

        # Computation_Demand (float)
        computation_demand = truncnorm.rvs(lower, upper, loc=mu, scale=sigma)

        task_info.append(
            [
                task_id,
                task_size,
                float(computation_demand),
                reliability_requirements[task_id - 1],
            ]
        )

    task_df = pd.DataFrame(
        task_info,
        columns=[
            "Task_ID",
            "Task_Size",
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
