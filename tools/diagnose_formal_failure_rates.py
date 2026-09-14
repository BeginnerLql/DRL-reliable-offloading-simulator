"""Print one fixed-seed formal base/spatial failure-rate realization."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.params import params
from core.spatial_risk import build_distance_matrix, build_spatial_correlation_matrix, sample_spatial_risk_field, map_spatial_risk_to_effective_failure_rates
from core.server import Server
import simpy


def main():
    frame = pd.read_excel(Path("data/server_info.xlsx"))
    env = simpy.Environment()
    servers = [Server(env, "Edge", int(row.Server_ID), float(row.Processing_Frequency), float(row.Base_Failure_Rate), float(row.Latitude), float(row.Longitude)) for row in frame.itertuples(index=False)]
    _, distances = build_distance_matrix(servers)
    sigma = build_spatial_correlation_matrix(distances, params.SPATIAL_CORRELATION_LENGTH_KM)
    z = sample_spatial_risk_field(sigma, rng=np.random.default_rng(params.SPATIAL_RISK_SEED))
    effective = map_spatial_risk_to_effective_failure_rates(frame["Base_Failure_Rate"].to_numpy(float), z, params.SPATIAL_RISK_BETA_P)
    print("Server_ID,Processing_Frequency,Base_Failure_Rate,Z,Effective_Failure_Rate")
    for row, z_value, effective_rate in zip(frame.itertuples(index=False), z, effective):
        print(f"{int(row.Server_ID)},{float(row.Processing_Frequency):g},{float(row.Base_Failure_Rate):.12g},{z_value:.12g},{effective_rate:.12g}")


if __name__ == "__main__":
    main()
