"""Diagnose cross-node replica-failure dependence in the existing model.

For a shared latent field, this tool uses

    P(F_j=1, F_k=1) = E_Z[p_j(Z_j) p_k(Z_k)]

rather than drawing Bernoulli failures.  Given ``Z``, transient-fault draws
remain independent (``F_j independent of F_k | Z``), while a correlated latent
field can make the marginal failures dependent when ``beta_p > 0``.  The same
quantity applies to distinct-server primary/backup replicas in both the
parallel ``z=1`` and sequential ``z=0`` interpretations of the quasi-static
episode model.  Queueing, arrival, recovery timing, and reward are not modeled.

The independent reference is a matched-marginal product baseline:

    P_ind(F_j=1, F_k=1) = P(F_j=1) P(F_k=1)

using the same spatial-scenario marginal failure probabilities.  This isolates
cross-node dependence without introducing Monte Carlo differences in the
marginals.  It is a mechanism-isolation reference, not another physical
environment simulation.

This is an offline diagnostic only.  It does not run MainLoop or modify the
simulator runtime.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import DATA_DIR, RESULTS_DIR
from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    sample_spatial_risk_fields,
    validate_correlation_matrix,
)


BETA_VALUES = (0.0, 0.2, 0.5, 0.8)
DEFAULT_SAMPLE_COUNT = 100_000
DEFAULT_SEED = 2026
DEFAULT_CORRELATION_LENGTH_KM = 0.5
JOINT_PROBABILITY_EPSILON = 1e-15
INDEPENDENT_BASELINE_TYPE = "matched_marginal_product"


def _load_servers(server_path: Path):
    required = {
        "Server_ID",
        "Processing_Frequency",
        "Failure_Rate",
        "Latitude",
        "Longitude",
    }
    frame = pd.read_excel(server_path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("server_info.xlsx is missing required columns: " + ", ".join(missing))

    servers = []
    rates_by_id = {}
    frequencies_by_id = {}
    for row_number, row in enumerate(frame.itertuples(index=False), start=2):
        values = row._asdict()
        try:
            raw_id = values["Server_ID"]
            server_id = int(raw_id)
            if float(raw_id) != server_id:
                raise ValueError("Server_ID must be an integer")
            frequency = float(values["Processing_Frequency"])
            failure_rate = float(values["Failure_Rate"])
            latitude = float(values["Latitude"])
            longitude = float(values["Longitude"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid server row {row_number}") from exc
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError(f"Processing_Frequency must be finite and positive on row {row_number}")
        if not math.isfinite(failure_rate) or failure_rate < 0.0:
            raise ValueError(f"Failure_Rate must be finite and non-negative on row {row_number}")
        servers.append(
            SimpleNamespace(
                server_id=server_id,
                latitude=latitude,
                longitude=longitude,
            )
        )
        rates_by_id[server_id] = failure_rate
        frequencies_by_id[server_id] = frequency

    server_ids, distance_matrix = build_distance_matrix(servers)
    base_rates = np.array([rates_by_id[server_id] for server_id in server_ids], dtype=float)
    frequencies = np.array([frequencies_by_id[server_id] for server_id in server_ids], dtype=float)
    return server_ids, base_rates, frequencies, distance_matrix


def _load_tasks(task_path: Path):
    required = {"Task_ID", "Computation_Demand"}
    frame = pd.read_excel(task_path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("task_parameters.xlsx is missing required columns: " + ", ".join(missing))
    task_ids = []
    demands = []
    for row_number, row in enumerate(frame.itertuples(index=False), start=2):
        values = row._asdict()
        try:
            raw_id = values["Task_ID"]
            task_id = int(raw_id)
            if float(raw_id) != task_id:
                raise ValueError("Task_ID must be an integer")
            demand = float(values["Computation_Demand"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid task row {row_number}") from exc
        if not math.isfinite(demand) or demand < 0.0:
            raise ValueError(f"Computation_Demand must be finite and non-negative on row {row_number}")
        task_ids.append(task_id)
        demands.append(demand)
    if not task_ids:
        raise ValueError("task_parameters.xlsx must contain at least one task")
    return np.asarray(task_ids, dtype=int), np.asarray(demands, dtype=float)


def compute_hazard_multipliers(z_samples: object, beta_p: object) -> np.ndarray:
    """Return the vectorized ``exp(beta*Z-beta**2/2)`` multiplier."""
    z = np.asarray(z_samples, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("z_samples must contain only finite values")
    beta = float(beta_p)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta_p must be finite and non-negative")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exponent = beta * z - 0.5 * np.square(np.float64(beta))
    if not np.isfinite(exponent).all():
        raise ValueError("hazard multiplier became non-finite")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        multipliers = np.exp(exponent)
    if not np.isfinite(multipliers).all():
        raise ValueError("hazard multiplier became non-finite")
    return multipliers


def compute_failure_probability_samples(
    effective_failure_rates: np.ndarray,
    service_times: np.ndarray,
) -> np.ndarray:
    """Apply the simulator formula ``1-exp(-lambda_eff*t)`` without Bernoulli draws."""
    rates = np.asarray(effective_failure_rates, dtype=float)
    times = np.asarray(service_times, dtype=float)
    if rates.ndim != 2 or times.ndim != 1 or rates.shape[1] != times.shape[0]:
        raise ValueError("effective rates and service times have incompatible shapes")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        probabilities = -np.expm1(-rates * times[np.newaxis, :])
    if not np.isfinite(probabilities).all():
        raise ValueError("failure probability became non-finite")
    return probabilities


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= JOINT_PROBABILITY_EPSILON:
        return float("nan")
    return float(numerator / denominator)


def _binary_failure_correlation(marginal_j: float, marginal_k: float, joint: float) -> float:
    denominator = math.sqrt(
        max(marginal_j * (1.0 - marginal_j), 0.0)
        * max(marginal_k * (1.0 - marginal_k), 0.0)
    )
    if denominator <= JOINT_PROBABILITY_EPSILON:
        return float("nan")
    return float((joint - marginal_j * marginal_k) / denominator)


def pair_failure_statistics(
    p_j_spatial: object,
    p_k_spatial: object,
) -> dict:
    """Compute pair metrics against the matched-marginal independent baseline."""
    spatial_j = np.asarray(p_j_spatial, dtype=float)
    spatial_k = np.asarray(p_k_spatial, dtype=float)
    if not (
        spatial_j.ndim == spatial_k.ndim == 1
        and spatial_j.shape == spatial_k.shape
    ):
        raise ValueError("pair probability samples must be equal-length vectors")

    marginal_j = float(spatial_j.mean())
    marginal_k = float(spatial_k.mean())
    joint_independent = marginal_j * marginal_k
    # E[XY] = E[X]E[Y] + Cov(X,Y); centering keeps the constant beta=0
    # control exactly equal to the matched-marginal product in floating point.
    covariance = float(
        np.mean((spatial_j - marginal_j) * (spatial_k - marginal_k))
    )
    joint_spatial = joint_independent + covariance
    excess = joint_spatial - joint_independent
    amplification = _safe_ratio(joint_spatial, joint_independent)
    return {
        "independent_baseline_type": INDEPENDENT_BASELINE_TYPE,
        "marginal_j": marginal_j,
        "marginal_k": marginal_k,
        "marginal_j_independent": marginal_j,
        "marginal_k_independent": marginal_k,
        "joint_spatial": joint_spatial,
        "joint_independent": joint_independent,
        "joint_failure_amplification": amplification,
        "excess_joint_failure": excess,
        "relative_joint_underestimation": amplification - 1.0 if math.isfinite(amplification) else float("nan"),
        "reliability_overestimation": excess,
        "binary_failure_correlation": _binary_failure_correlation(marginal_j, marginal_k, joint_spatial),
        "absolute_marginal_difference_j": 0.0,
        "absolute_marginal_difference_k": 0.0,
    }


def theoretical_hazard_correlation(beta_p: float, rho: float) -> float:
    """Return Corr(lambda_eff_j, lambda_eff_k) for beta_p > 0."""
    if beta_p == 0.0:
        return float("nan")
    return float(np.expm1(beta_p**2 * rho) / np.expm1(beta_p**2))


def _empirical_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _finite_max(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")


def run_diagnostic(
    server_path: Path | str = Path(DATA_DIR) / "server_info.xlsx",
    task_path: Path | str = Path(DATA_DIR) / "task_parameters.xlsx",
    output_dir: Path | str = Path(RESULTS_DIR) / "spatial_risk_diagnostics",
    seed: int = DEFAULT_SEED,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    correlation_length_km: float = DEFAULT_CORRELATION_LENGTH_KM,
):
    """Run the spatial-vs-matched-marginal cross-node failure diagnostic."""
    if not isinstance(sample_count, (int, np.integer)) or isinstance(sample_count, (bool, np.bool_)) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if not math.isfinite(float(correlation_length_km)) or correlation_length_km <= 0.0:
        raise ValueError("correlation_length_km must be finite and positive")

    server_ids, base_rates, frequencies, distance_matrix = _load_servers(Path(server_path))
    task_ids, computation_demands = _load_tasks(Path(task_path))
    spatial_correlation = build_spatial_correlation_matrix(distance_matrix, correlation_length_km)
    validate_correlation_matrix(spatial_correlation)

    spatial_z = sample_spatial_risk_fields(
        spatial_correlation,
        int(sample_count),
        rng=np.random.default_rng(seed),
    )
    latent_correlation = np.corrcoef(spatial_z, rowvar=False)
    pair_indices = list(itertools.combinations(range(len(server_ids)), 2))

    detailed_rows = []
    for beta_p in BETA_VALUES:
        spatial_multipliers = compute_hazard_multipliers(spatial_z, beta_p)
        spatial_rates = spatial_multipliers * base_rates[np.newaxis, :]
        spatial_hazard_correlation = {
            pair: _empirical_correlation(spatial_multipliers[:, pair[0]], spatial_multipliers[:, pair[1]])
            for pair in pair_indices
        }
        # Keep probability arrays small: 100000 samples x 2 tasks x N servers
        # avoids retaining a large task-by-sample tensor in memory.
        for start in range(0, len(task_ids), 2):
            task_slice = slice(start, start + 2)
            service_times = computation_demands[task_slice, np.newaxis] / frequencies[np.newaxis, :]
            spatial_probabilities = np.stack(
                [compute_failure_probability_samples(spatial_rates, service_times[index]) for index in range(service_times.shape[0])],
                axis=1,
            )
            for local_index, task_id in enumerate(task_ids[task_slice]):
                for j, k in pair_indices:
                    metrics = pair_failure_statistics(
                        spatial_probabilities[:, local_index, j],
                        spatial_probabilities[:, local_index, k],
                    )
                    rho = float(spatial_correlation[j, k])
                    empirical_hazard = spatial_hazard_correlation[(j, k)]
                    detailed_rows.append({
                        "Beta_p": beta_p,
                        "Independent_Baseline_Type": INDEPENDENT_BASELINE_TYPE,
                        "Task_ID": int(task_id),
                        "Server_J": server_ids[j],
                        "Server_K": server_ids[k],
                        "Distance_km": float(distance_matrix[j, k]),
                        "Rho_phy": rho,
                        "Service_Time_J": float(service_times[local_index, j]),
                        "Service_Time_K": float(service_times[local_index, k]),
                        "Marginal_Failure_J_Spatial": metrics["marginal_j"],
                        "Marginal_Failure_K_Spatial": metrics["marginal_k"],
                        "Marginal_Failure_J_Independent": metrics["marginal_j_independent"],
                        "Marginal_Failure_K_Independent": metrics["marginal_k_independent"],
                        "Joint_Failure_Spatial": metrics["joint_spatial"],
                        "Joint_Failure_Independent": metrics["joint_independent"],
                        "Joint_Failure_Amplification": metrics["joint_failure_amplification"],
                        "Excess_Joint_Failure": metrics["excess_joint_failure"],
                        "Relative_Joint_Underestimation": metrics["relative_joint_underestimation"],
                        "Reliability_Overestimation": metrics["reliability_overestimation"],
                        "Binary_Failure_Correlation": metrics["binary_failure_correlation"],
                        "Absolute_Marginal_Difference_J": metrics["absolute_marginal_difference_j"],
                        "Absolute_Marginal_Difference_K": metrics["absolute_marginal_difference_k"],
                        "Empirical_Latent_Correlation": float(latent_correlation[j, k]),
                        "Theoretical_Latent_Correlation": rho,
                        "Empirical_Hazard_Correlation": empirical_hazard,
                        "Theoretical_Hazard_Correlation": theoretical_hazard_correlation(beta_p, rho),
                    })

    detail_df = pd.DataFrame(detailed_rows)
    summary_rows = []
    for (beta_p, server_j, server_k), group in detail_df.groupby(
        ["Beta_p", "Server_J", "Server_K"], sort=True
    ):
        first = group.iloc[0]
        summary_rows.append({
            "Beta_p": beta_p,
            "Independent_Baseline_Type": INDEPENDENT_BASELINE_TYPE,
            "Server_J": server_j,
            "Server_K": server_k,
            "Distance_km": first["Distance_km"],
            "Rho_phy": first["Rho_phy"],
            "Empirical_Latent_Correlation": first["Empirical_Latent_Correlation"],
            "Theoretical_Latent_Correlation": first["Theoretical_Latent_Correlation"],
            "Empirical_Hazard_Correlation": first["Empirical_Hazard_Correlation"],
            "Theoretical_Hazard_Correlation": first["Theoretical_Hazard_Correlation"],
            "Mean_Marginal_Failure_J_Spatial": group["Marginal_Failure_J_Spatial"].mean(),
            "Mean_Marginal_Failure_K_Spatial": group["Marginal_Failure_K_Spatial"].mean(),
            "Mean_Joint_Failure_Spatial": group["Joint_Failure_Spatial"].mean(),
            "Mean_Joint_Failure_Independent": group["Joint_Failure_Independent"].mean(),
            "Mean_Joint_Failure_Amplification": _finite_mean(group["Joint_Failure_Amplification"]),
            "Mean_Excess_Joint_Failure": group["Excess_Joint_Failure"].mean(),
            "Mean_Relative_Joint_Underestimation": _finite_mean(group["Relative_Joint_Underestimation"]),
            "Mean_Binary_Failure_Correlation": _finite_mean(group["Binary_Failure_Correlation"]),
            "Max_Binary_Failure_Correlation": _finite_max(group["Binary_Failure_Correlation"]),
            "Max_Joint_Failure_Amplification": _finite_max(group["Joint_Failure_Amplification"]),
            "Max_Absolute_Marginal_Difference": max(
                group["Absolute_Marginal_Difference_J"].max(),
                group["Absolute_Marginal_Difference_K"].max(),
            ),
        })
    pair_summary_df = pd.DataFrame(summary_rows)

    global_rows = []
    for beta_p, group in detail_df.groupby("Beta_p", sort=True):
        pair_group = pair_summary_df[pair_summary_df["Beta_p"] == beta_p]
        nearest = pair_group.sort_values(["Distance_km", "Server_J", "Server_K"]).iloc[0]
        farthest = pair_group.sort_values(["Distance_km", "Server_J", "Server_K"]).iloc[-1]
        global_rows.append({
            "Beta_p": beta_p,
            "Independent_Baseline_Type": INDEPENDENT_BASELINE_TYPE,
            "Mean_Joint_Failure_Spatial": group["Joint_Failure_Spatial"].mean(),
            "Mean_Joint_Failure_Independent": group["Joint_Failure_Independent"].mean(),
            "Mean_Joint_Failure_Amplification": _finite_mean(group["Joint_Failure_Amplification"]),
            "Mean_Excess_Joint_Failure": group["Excess_Joint_Failure"].mean(),
            "Mean_Binary_Failure_Correlation": _finite_mean(group["Binary_Failure_Correlation"]),
            "Max_Binary_Failure_Correlation": _finite_max(group["Binary_Failure_Correlation"]),
            "Mean_Absolute_Marginal_Difference": np.mean([
                group["Absolute_Marginal_Difference_J"].mean(),
                group["Absolute_Marginal_Difference_K"].mean(),
            ]),
            "Max_Absolute_Marginal_Difference": max(
                group["Absolute_Marginal_Difference_J"].max(),
                group["Absolute_Marginal_Difference_K"].max(),
            ),
            "Nearest_Pair": f"{int(nearest.Server_J)}-{int(nearest.Server_K)}",
            "Nearest_Pair_Rho": nearest.Rho_phy,
            "Nearest_Pair_Amplification": nearest.Mean_Joint_Failure_Amplification,
            "Farthest_Pair": f"{int(farthest.Server_J)}-{int(farthest.Server_K)}",
            "Farthest_Pair_Rho": farthest.Rho_phy,
            "Farthest_Pair_Amplification": farthest.Mean_Joint_Failure_Amplification,
        })
    global_df = pd.DataFrame(global_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "task_pair_joint_failure.csv"
    pair_path = output_dir / "pair_failure_correlation_summary.csv"
    global_path = output_dir / "pair_failure_global_summary.csv"
    detail_df.to_csv(detail_path, index=False)
    pair_summary_df.to_csv(pair_path, index=False)
    global_df.to_csv(global_path, index=False)

    max_latent_error = float(
        np.max(np.abs(pair_summary_df["Empirical_Latent_Correlation"] - pair_summary_df["Rho_phy"]))
    )
    hazard_rows = pair_summary_df[np.isfinite(pair_summary_df["Theoretical_Hazard_Correlation"])]
    max_hazard_error = float(
        np.max(np.abs(hazard_rows["Empirical_Hazard_Correlation"] - hazard_rows["Theoretical_Hazard_Correlation"]))
    )
    print("beta | mean joint spatial | mean joint independent | amplification | excess | mean binary corr")
    for row in global_df.itertuples(index=False):
        print(
            f"{row.Beta_p:.1f} | {row.Mean_Joint_Failure_Spatial:.8f} | "
            f"{row.Mean_Joint_Failure_Independent:.8f} | "
            f"{row.Mean_Joint_Failure_Amplification:.6f} | "
            f"{row.Mean_Excess_Joint_Failure:.8f} | "
            f"{row.Mean_Binary_Failure_Correlation:.6f}"
        )
    print(f"Servers: {len(server_ids)}; tasks: {len(task_ids)}; distinct pairs: {len(pair_indices)}")
    print(f"ell_phy: {correlation_length_km} km; seed: {seed}; sample_count: {sample_count}")
    print(f"Max latent-correlation error: {max_latent_error:.8f}")
    print(f"Max hazard-correlation error: {max_hazard_error:.8f}")
    print(f"Task-pair output: {detail_path}")
    print(f"Pair summary output: {pair_path}")
    print(f"Global summary output: {global_path}")
    return detail_df, pair_summary_df, global_df


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-input", type=Path, default=Path(DATA_DIR) / "server_info.xlsx")
    parser.add_argument("--task-input", type=Path, default=Path(DATA_DIR) / "task_parameters.xlsx")
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR) / "spatial_risk_diagnostics")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_diagnostic(
        server_path=args.server_input,
        task_path=args.task_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
