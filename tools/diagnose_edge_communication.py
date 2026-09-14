"""Print fixed per-Edge uplink rates and upload times."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from core.task import get_upload_time


def main():
    servers = pd.read_excel(PROJECT_ROOT / "data" / "server_info.xlsx")
    print("Server_ID,Processing_Frequency,Base_Failure_Rate,Uplink_Rate,tx@0.5MB,tx@1MB,tx@2MB")
    for row in servers.itertuples(index=False):
        rates = [get_upload_time(size, row.Uplink_Rate) for size in (0.5, 1.0, 2.0)]
        print(f"{int(row.Server_ID)},{float(row.Processing_Frequency):g},{float(row.Base_Failure_Rate):.12g},{float(row.Uplink_Rate):g}," + ",".join(f"{value:.12g}" for value in rates))
    tasks = pd.read_excel(PROJECT_ROOT / "data" / "task_parameters.xlsx")
    values = tasks["Input_Data_Size_MB"].astype(float)
    print(f"Input_Data_Size_MB: min={values.min():.12g}, mean={values.mean():.12g}, median={values.median():.12g}, max={values.max():.12g}")


if __name__ == "__main__":
    main()
