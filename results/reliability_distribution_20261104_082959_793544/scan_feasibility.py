"""Independent taskwise GH scan of all three train/evaluation distributions."""
from itertools import combinations
import json
from pathlib import Path
import sys
from types import SimpleNamespace
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))
import numpy as np
import pandas as pd
from config.params import params
from core.spatial_risk import (GaussHermiteReliability, build_distance_matrix,
                               build_spatial_correlation_matrix, compute_baseline_correlation_length_km)


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    frame = pd.read_excel(OUT / "shared_inputs/server_info.xlsx")
    servers = [SimpleNamespace(server_id=int(r.Server_ID), latitude=r.Latitude, longitude=r.Longitude)
               for r in frame.itertuples()]
    ids, distances = build_distance_matrix(servers)
    length = compute_baseline_correlation_length_km(distances)
    correlations = build_spatial_correlation_matrix(distances, length)
    pd.DataFrame(correlations, index=ids, columns=ids).to_csv(OUT / "server_risk_correlations.csv")
    pairs = list(combinations(range(8), 2))
    gh = GaussHermiteReliability(params.SPATIAL_RISK_BETA_P, params.SPATIAL_RISK_GH_ORDER)
    cache = {}
    all_rows, summaries = [], []
    detail = []
    for dataset in ("train", "evaluation"):
        baseline = pd.read_excel(OUT / "shared_inputs" / f"baseline_{dataset}.xlsx")
        for task in baseline.itertuples():
            demand = task.Computation_Demand
            if demand not in cache:
                exposure = frame.Base_Failure_Rate.to_numpy() * demand / frame.Processing_Frequency.to_numpy()
                single = np.array([gh.compute_marginal(a) for a in exposure])
                results = [gh.compute_pair(exposure[i], exposure[j], correlations[i, j]) for i, j in pairs]
                corr = np.array([r.correlated for r in results])
                independent = np.array([r.independent for r in results])
                assert (corr <= independent + 1e-12).all()
                cache[demand] = single, corr, independent
                for (i, j), result in zip(pairs, results):
                    detail.append(dict(dataset=dataset, task_id=task.Task_ID, demand=demand,
                                       server_A=ids[i], server_B=ids[j], rho=correlations[i, j],
                                       marginal_A=result.replica_a, marginal_B=result.replica_b,
                                       Rcorr=result.correlated, Rind=result.independent))
        for group in manifest["groups"]:
            source = OUT / "groups" / group["id"] / "inputs" / ("train200.xlsx" if dataset == "train" else "evaluation200.xlsx")
            tasks = pd.read_excel(source)
            rows = []
            for task in tasks.itertuples():
                single, corr, independent = cache[task.Computation_Demand]
                threshold = task.Reliability_Requirement
                safe = [ids[i] for i in range(8) if single[i] >= threshold]
                base_delays = task.Task_Size / frame.Transmission_Rate.to_numpy() + task.Computation_Demand / frame.Processing_Frequency.to_numpy()
                fastest = min(safe, key=lambda server: (base_delays[ids.index(server)], server)) if safe else None
                row = dict(group_id=group["id"], dataset=dataset, task_id=task.Task_ID,
                           reliability_requirement=threshold, single_safe_count=int((single >= threshold).sum()),
                           corr_pair_safe_count=int((corr >= threshold).sum()),
                           independent_pair_safe_count=int((independent >= threshold).sum()),
                           best_single=float(single.max()), best_corr_pair=float(corr.max()),
                           best_independent_pair=float(independent.max()),
                           single_fallback_server=ids[int(single.argmax())],
                           fastest_safe_server_by_base_delay=fastest)
                rows.append(row)
            all_rows += rows
            scan = pd.DataFrame(rows)
            summary = dict(group_id=group["id"], dataset=dataset,
                           single_ceiling=float(scan.single_safe_count.gt(0).mean()),
                           correlated_pair_ceiling=float(scan.corr_pair_safe_count.gt(0).mean()),
                           independent_pair_ceiling=float(scan.independent_pair_safe_count.gt(0).mean()),
                           mean_single_safe_count=float(scan.single_safe_count.mean()),
                           mean_corr_pair_safe_count=float(scan.corr_pair_safe_count.mean()),
                           mean_independent_pair_safe_count=float(scan.independent_pair_safe_count.mean()))
            for prefix, column in (("single", "single_safe_count"), ("corr_pair", "corr_pair_safe_count"),
                                    ("ind_pair", "independent_pair_safe_count")):
                summary.update({f"{prefix}_K0": int(scan[column].eq(0).sum()),
                                f"{prefix}_K1": int(scan[column].eq(1).sum()),
                                f"{prefix}_K_ge2": int(scan[column].ge(2).sum())})
            summaries.append(summary)
    pd.DataFrame(all_rows).to_csv(OUT / "feasibility_per_task.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "feasibility_summary.csv", index=False)
    pd.DataFrame(detail).to_csv(OUT / "pair_reliability_detail.csv", index=False)
    (OUT / "feasibility_metadata.json").write_text(json.dumps(dict(beta=params.SPATIAL_RISK_BETA_P,
         gh_order=params.SPATIAL_RISK_GH_ORDER, correlation_length_km=length, unique_demands=len(cache),
         failure_exposure="CPU only: base_failure_rate * demand / frequency"), indent=2))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
