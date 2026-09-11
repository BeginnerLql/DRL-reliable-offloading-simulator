"""Diagnose beta_p hazard-multiplier and effective-rate scales.

This tool reads the current server coordinates and base failure rates, builds the
existing physical spatial correlation matrix, samples one common batch of
``Z_phy ~ N(0, R_phy)``, and evaluates beta_p in ``{0.0, 0.2, 0.5, 0.8}``.
It does not build tasks, run MainLoop, train PPO, or draw task failures.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import NormalDist
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


def _load_server_inputs(input_path: Path):
    required_columns = {"Server_ID", "Failure_Rate", "Latitude", "Longitude"}
    server_df = pd.read_excel(input_path)
    missing_columns = sorted(required_columns.difference(server_df.columns))
    if missing_columns:
        raise ValueError(
            "server_info.xlsx is missing required columns: "
            + ", ".join(missing_columns)
        )

    servers = []
    for row_number, row in enumerate(server_df.itertuples(index=False), start=2):
        values = row._asdict()
        try:
            raw_server_id = values["Server_ID"]
            server_id = int(raw_server_id)
            if float(raw_server_id) != server_id:
                raise ValueError("must be an integer")
            base_failure_rate = float(values["Failure_Rate"])
            latitude = float(values["Latitude"])
            longitude = float(values["Longitude"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"invalid server parameter on server_info.xlsx row {row_number}"
            ) from exc
        if not math.isfinite(base_failure_rate) or base_failure_rate < 0.0:
            raise ValueError(
                f"Failure_Rate must be finite and non-negative on row {row_number}"
            )
        servers.append(
            SimpleNamespace(
                server_id=server_id,
                failure_rate=base_failure_rate,
                latitude=latitude,
                longitude=longitude,
            )
        )

    server_ids, distance_matrix = build_distance_matrix(servers)
    rates_by_id = {server.server_id: server.failure_rate for server in servers}
    base_failure_rates = np.array(
        [rates_by_id[server_id] for server_id in server_ids],
        dtype=float,
    )
    return server_ids, base_failure_rates, distance_matrix


def compute_hazard_multiplier(z_samples: object, beta_p: object) -> np.ndarray:
    """Compute ``exp(beta_p * Z_phy - beta_p**2 / 2)`` vectorized."""
    samples = np.asarray(z_samples, dtype=float)
    if not np.isfinite(samples).all():
        raise ValueError("z_samples must contain only finite values")
    try:
        beta = float(beta_p)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta_p must be a finite non-negative scalar") from exc
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta_p must be a finite non-negative scalar")

    # This vectorized diagnostic uses the same mathematical mapping as
    # map_spatial_risk_to_effective_failure_rates().
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exponent = beta * samples - 0.5 * np.square(np.float64(beta))
    if not np.isfinite(exponent).all():
        raise ValueError("hazard multiplier became non-finite")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        multiplier = np.exp(exponent)
    if not np.isfinite(multiplier).all():
        raise ValueError("hazard multiplier became non-finite")
    return np.asarray(multiplier, dtype=float)


def theoretical_multiplier_quantile(beta_p: object, probability: object) -> float:
    """Return the theoretical quantile of ``M=exp(beta*Z-beta**2/2)``."""
    beta = float(beta_p)
    probability = float(probability)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta_p must be a finite non-negative scalar")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    normal_quantile = NormalDist().inv_cdf(probability)
    return float(math.exp(beta * normal_quantile - 0.5 * beta**2))


def _summary_row(beta_p: float, multipliers: np.ndarray, sample_count: int) -> dict:
    flat = np.asarray(multipliers, dtype=float).reshape(-1)
    empirical_p05, empirical_p50, empirical_p95 = np.percentile(flat, [5, 50, 95])
    theoretical_p05 = theoretical_multiplier_quantile(beta_p, 0.05)
    theoretical_p50 = theoretical_multiplier_quantile(beta_p, 0.50)
    theoretical_p95 = theoretical_multiplier_quantile(beta_p, 0.95)
    return {
        "Beta_p": beta_p,
        "Sample_Count": sample_count,
        "Mean_Multiplier": float(flat.mean()),
        "Std_Multiplier": float(flat.std()),
        "P01": float(np.percentile(flat, 1)),
        "P05": float(empirical_p05),
        "P25": float(np.percentile(flat, 25)),
        "P50": float(empirical_p50),
        "P75": float(np.percentile(flat, 75)),
        "P95": float(empirical_p95),
        "P99": float(np.percentile(flat, 99)),
        "Min": float(flat.min()),
        "Max": float(flat.max()),
        "Prob_M_gt_1": float(np.mean(flat > 1.0)),
        "Prob_M_gt_1p5": float(np.mean(flat > 1.5)),
        "Prob_M_gt_2": float(np.mean(flat > 2.0)),
        "Prob_M_gt_3": float(np.mean(flat > 3.0)),
        "Prob_M_lt_0p5": float(np.mean(flat < 0.5)),
        "Empirical_P05": float(empirical_p05),
        "Theoretical_P05": theoretical_p05,
        "P05_Absolute_Error": abs(empirical_p05 - theoretical_p05),
        "Empirical_P50": float(empirical_p50),
        "Theoretical_P50": theoretical_p50,
        "P50_Absolute_Error": abs(empirical_p50 - theoretical_p50),
        "Empirical_P95": float(empirical_p95),
        "Theoretical_P95": theoretical_p95,
        "P95_Absolute_Error": abs(empirical_p95 - theoretical_p95),
    }


def run_diagnostic(
    input_path: Path | str = Path(DATA_DIR) / "server_info.xlsx",
    output_dir: Path | str = Path(RESULTS_DIR) / "spatial_risk_diagnostics",
    seed: int = DEFAULT_SEED,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    correlation_length_km: float = DEFAULT_CORRELATION_LENGTH_KM,
):
    """Run the fixed-ell beta sensitivity diagnostic and write two CSV files."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if sample_count <= 0 or not isinstance(sample_count, (int, np.integer)):
        raise ValueError("sample_count must be a positive integer")
    if not math.isfinite(float(correlation_length_km)) or correlation_length_km <= 0.0:
        raise ValueError("correlation_length_km must be finite and positive")

    server_ids, base_failure_rates, distance_matrix = _load_server_inputs(input_path)
    correlation_matrix = build_spatial_correlation_matrix(
        distance_matrix,
        correlation_length_km,
    )
    validate_correlation_matrix(correlation_matrix)

    # Draw exactly one common realization batch for every beta_p value.
    z_samples = sample_spatial_risk_fields(
        correlation_matrix,
        int(sample_count),
        rng=np.random.default_rng(seed),
    )

    summary_rows = []
    effective_rate_rows = []
    for beta_p in BETA_VALUES:
        multipliers = compute_hazard_multiplier(z_samples, beta_p)
        summary_rows.append(_summary_row(beta_p, multipliers, int(sample_count)))
        effective_rates = base_failure_rates[np.newaxis, :] * multipliers
        p05, p50, p95 = np.percentile(effective_rates, [5, 50, 95], axis=0)
        means = effective_rates.mean(axis=0)
        for index, server_id in enumerate(server_ids):
            effective_rate_rows.append({
                "Beta_p": beta_p,
                "Server_ID": server_id,
                "Base_Failure_Rate": float(base_failure_rates[index]),
                "Mean_Effective_Rate": float(means[index]),
                "P05_Effective_Rate": float(p05[index]),
                "P50_Effective_Rate": float(p50[index]),
                "P95_Effective_Rate": float(p95[index]),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "beta_p_sensitivity_summary.csv"
    rates_path = output_dir / "beta_p_server_effective_rate_quantiles.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(effective_rate_rows).to_csv(rates_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    print("beta | mean | std | p05 | p50 | p95 | P(M>2)")
    for row in summary_df.itertuples(index=False):
        print(
            f"{row.Beta_p:.1f} | {row.Mean_Multiplier:.6f} | "
            f"{row.Std_Multiplier:.6f} | {row.P05:.6f} | "
            f"{row.P50:.6f} | {row.P95:.6f} | {row.Prob_M_gt_2:.6f}"
        )
    print(f"Input: {input_path}")
    print(f"Servers: {len(server_ids)}")
    print(f"ell_phy: {correlation_length_km} km")
    print(f"seed: {seed}; sample_count: {sample_count}")
    print(f"Summary: {summary_path}")
    print(f"Effective-rate quantiles: {rates_path}")
    return summary_df, pd.DataFrame(effective_rate_rows)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DATA_DIR) / "server_info.xlsx",
        help="Path to server_info.xlsx (default: data/server_info.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(RESULTS_DIR) / "spatial_risk_diagnostics",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_diagnostic(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
