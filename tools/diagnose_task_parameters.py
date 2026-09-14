"""Print a compact summary of the formal task-parameter model."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.params import params


def main() -> None:
    path = Path("data") / "task_parameters.xlsx"
    task_df = pd.read_excel(path)
    input_data = task_df["Input_Data_Size_MB"].to_numpy(dtype=float)
    demand = task_df["Computation_Demand"].to_numpy(dtype=float)
    requirements = task_df["Reliability_Requirement"].to_numpy(dtype=float)
    frequencies = np.asarray(params.FIXED_EDGE_PROCESSING_FREQUENCIES, dtype=float)
    print(f"Task count: {len(task_df)}")
    print(
        "Input_Data_Size_MB: "
        f"min={input_data.min():.12g}, mean={input_data.mean():.12g}, "
        f"median={np.median(input_data):.12g}, max={input_data.max():.12g}"
    )
    print(
        "Computation_Demand: "
        f"min={demand.min():.12g}, mean={demand.mean():.12g}, "
        f"median={np.median(demand):.12g}, max={demand.max():.12g}, "
        f"unique_count={len(np.unique(demand))}"
    )
    counts = Counter(int(value) for value in demand)
    print(
        "Computation_Demand distribution: "
        f"integer_check={bool(np.all(demand == np.round(demand)))}, "
        f"count_min={min(counts.values())}, count_max={max(counts.values())}"
    )
    print("Computation_Demand counts:", dict(sorted(counts.items())))
    print("Reliability_Requirement counts:", dict(sorted(Counter(requirements).items())))
    min_service = params.COMPUTATION_DEMAND_RANGE_MI[0] / frequencies.max()
    max_service = params.COMPUTATION_DEMAND_RANGE_MI[1] / frequencies.min()
    print(
        "Theoretical service time: "
        f"min={min_service:.12g}s, max={max_service:.12g}s"
    )
    print(f"num_states={params.num_states}, num_actions={params.num_actions}")


if __name__ == "__main__":
    main()
