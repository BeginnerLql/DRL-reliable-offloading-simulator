"""Taskwise safe sets using the same GH integrator as the simulator."""
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from config.params import params
from core.spatial_risk import (GaussHermiteReliability, build_distance_matrix,
                               build_spatial_correlation_matrix,
                               compute_baseline_correlation_length_km)


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    frame = pd.read_excel(OUT / "inputs/server_info.xlsx")
    servers = [SimpleNamespace(server_id=int(row.Server_ID), latitude=row.Latitude,
                               longitude=row.Longitude, frequency=row.Processing_Frequency,
                               failure_rate=row.Base_Failure_Rate, upload_rate=row.Transmission_Rate)
               for row in frame.itertuples()]
    ids, distances = build_distance_matrix(servers)
    assert ids == [server.server_id for server in servers]
    length = compute_baseline_correlation_length_km(distances)
    rho = build_spatial_correlation_matrix(distances, length)
    pd.DataFrame(rho, index=ids, columns=ids).to_csv(OUT / "server_risk_correlations.csv")
    gh = GaussHermiteReliability(params.SPATIAL_RISK_BETA_P, params.SPATIAL_RISK_GH_ORDER)
    cache = {}

    def reliabilities(demand):
        if demand not in cache:
            exposures = [s.failure_rate * demand / s.frequency for s in servers]
            singles = np.array([gh.compute_marginal(a) for a in exposures])
            pairs = [gh.compute_pair(exposures[i], exposures[j], rho[i, j])
                     for i, j in combinations(range(len(servers)), 2)]
            corr = np.array([pair.correlated for pair in pairs])
            independent = np.array([pair.independent for pair in pairs])
            assert np.all(corr <= independent + 1e-12)
            cache[demand] = singles, corr, independent
        return cache[demand]

    directory = OUT / "feasibility"
    directory.mkdir(exist_ok=True)
    summary = []
    for scenario in manifest["scenarios"]:
        tasks = pd.read_excel(scenario["task_path"])
        rows = []
        for task in tasks.itertuples():
            single, corr, independent = reliabilities(task.Computation_Demand)
            threshold = task.Reliability_Requirement
            safe = [s for s, reliability in zip(servers, single) if reliability >= threshold]
            fastest = min(safe, key=lambda s: (task.Task_Size / s.upload_rate +
                                               task.Computation_Demand / s.frequency, s.server_id)) if safe else None
            rows.append(dict(task_id=task.Task_ID, requirement=threshold,
                             computation_demand=task.Computation_Demand,
                             single_safe_count=int((single >= threshold).sum()),
                             corr_pair_safe_count=int((corr >= threshold).sum()),
                             independent_pair_safe_count=int((independent >= threshold).sum()),
                             single_safe_server_ids=",".join(str(s.server_id) for s in safe),
                             fastest_safe_server_by_base_delay=fastest.server_id if fastest else None,
                             best_single_reliability=float(single.max()),
                             best_correlated_reliability=float(corr.max()),
                             best_independent_reliability=float(independent.max())))
        scan = pd.DataFrame(rows)
        scan.to_csv(directory / f"{scenario['id']}.csv", index=False)
        row = dict(scenario_id=scenario["id"], task_count=len(tasks),
                   correlated_pair_ceiling=float(scan.corr_pair_safe_count.gt(0).mean()),
                   independent_pair_ceiling=float(scan.independent_pair_safe_count.gt(0).mean()),
                   single_replica_ceiling=float(scan.single_safe_count.gt(0).mean()))
        for prefix, column in (("single", "single_safe_count"), ("corr_pair", "corr_pair_safe_count"),
                                ("independent_pair", "independent_pair_safe_count")):
            counts = scan[column]
            row.update({f"{prefix}_K0": int(counts.eq(0).sum()),
                        f"{prefix}_K1": int(counts.eq(1).sum()),
                        f"{prefix}_K_ge2": int(counts.ge(2).sum())})
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "feasibility_summary.csv", index=False)
    (OUT / "feasibility_metadata.json").write_text(json.dumps(dict(
        beta=params.SPATIAL_RISK_BETA_P, gh_order=params.SPATIAL_RISK_GH_ORDER,
        correlation_length_km=length, unique_demands=len(cache),
        reliability_exposure="base_failure_rate * CPU_cycles / CPU_frequency; CPU only",
        taskwise_safety_is_static=True), indent=2))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
