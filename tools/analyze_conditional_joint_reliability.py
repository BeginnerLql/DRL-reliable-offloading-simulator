"""Summarize conditional primary/backup reliability under episode risk fields.

This offline diagnostic reuses the simulator's geographic correlation and
effective transient-failure-rate implementation. It samples one complete
``Z_phy ~ N(0, Sigma)`` field per diagnostic episode and keeps that field fixed
for every task and server pair in the episode. For each distinct pair ``j<k``
it then computes, without drawing task failures,

    p_ij(Z_j) = 1 - exp(-lambda_eff_j * C_i / f_j)
    P_JF(j, k | Z) = p_ij(Z_j) * p_ik(Z_k)
    R_exec(j, k | Z) = 1 - P_JF(j, k | Z)

The task workbook is reused in every episode, matching the current simulator,
which loads the same generated task parameters for each episode. This tool
does not run MainLoop, sample Bernoulli failures, evaluate reliability
requirements, or modify PPO/runtime behavior. It writes summaries and a
small deterministic sample, never the full multi-million-row table.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import DATA_DIR, RESULTS_DIR
from core.spatial_risk import (
    build_spatial_correlation_matrix,
    map_spatial_risk_to_effective_failure_rates,
    sample_spatial_risk_fields,
    validate_correlation_matrix,
)
from tools.analyze_pair_failure_correlation import (
    _load_servers,
    _load_tasks,
    compute_failure_probability_samples,
)


DEFAULT_EPISODES = 500
DEFAULT_TASKS_PER_EPISODE = 200
DEFAULT_SEED = 2026
DEFAULT_CORRELATION_LENGTH_KM = 0.5
DEFAULT_BETA_P = 0.8
DEFAULT_SAMPLE_ROWS = 5_000
RELIABILITY_THRESHOLDS = (0.99, 0.999, 0.9999, 0.99999)
QUANTILES = {
    "p01": 0.01,
    "p05": 0.05,
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _finite_non_negative(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _summary_row(metric: str, values: object) -> dict:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{metric} values must be finite and non-empty")
    row = {
        "metric": metric,
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
    }
    row.update(
        {
            name: float(np.quantile(array, quantile))
            for name, quantile in QUANTILES.items()
        }
    )
    row["max"] = float(array.max())
    return row


def calculate_conditional_joint_reliability(
    base_failure_rates: object,
    processing_frequencies: object,
    computation_demands: object,
    correlation_matrix: object,
    *,
    episodes: int,
    beta_p: float,
    seed: int,
) -> dict:
    """Compute exact conditional failure/success probabilities for all samples."""
    episode_count = _positive_integer(episodes, "episodes")
    beta = _finite_non_negative(beta_p, "beta_p")
    validate_correlation_matrix(correlation_matrix)

    rates = np.asarray(base_failure_rates, dtype=float)
    frequencies = np.asarray(processing_frequencies, dtype=float)
    demands = np.asarray(computation_demands, dtype=float)
    correlation = np.asarray(correlation_matrix, dtype=float)
    node_count = correlation.shape[0]
    if rates.shape != (node_count,) or frequencies.shape != (node_count,):
        raise ValueError("server parameter vectors must match the correlation matrix")
    if demands.ndim != 1 or demands.size == 0:
        raise ValueError("computation_demands must be a non-empty vector")
    if not np.isfinite(rates).all() or (rates < 0.0).any():
        raise ValueError("base_failure_rates must be finite and non-negative")
    if not np.isfinite(frequencies).all() or (frequencies <= 0.0).any():
        raise ValueError("processing_frequencies must be finite and positive")
    if not np.isfinite(demands).all() or (demands < 0.0).any():
        raise ValueError("computation_demands must be finite and non-negative")
    if node_count < 2:
        raise ValueError("at least two servers are required")

    spatial_fields = sample_spatial_risk_fields(
        correlation,
        episode_count,
        rng=np.random.default_rng(seed),
    )
    effective_rates = np.vstack(
        [
            map_spatial_risk_to_effective_failure_rates(rates, field, beta)
            for field in spatial_fields
        ]
    )
    service_times = demands[:, np.newaxis] / frequencies[np.newaxis, :]
    failure_probabilities = np.stack(
        [
            compute_failure_probability_samples(effective_rates, task_service_times)
            for task_service_times in service_times
        ],
        axis=1,
    )

    pair_indices = np.asarray(
        list(itertools.combinations(range(node_count), 2)),
        dtype=int,
    )
    pair_j = pair_indices[:, 0]
    pair_k = pair_indices[:, 1]
    joint_failure = (
        failure_probabilities[:, :, pair_j]
        * failure_probabilities[:, :, pair_k]
    )
    joint_success = 1.0 - joint_failure

    for name, values in (
        ("failure_probabilities", failure_probabilities),
        ("joint_failure", joint_failure),
        ("joint_success", joint_success),
    ):
        if (
            not np.isfinite(values).all()
            or (values < 0.0).any()
            or (values > 1.0).any()
        ):
            raise ValueError(f"{name} must lie within [0, 1]")

    return {
        "spatial_fields": spatial_fields,
        "effective_rates": effective_rates,
        "service_times": service_times,
        "failure_probabilities": failure_probabilities,
        "pair_indices": pair_indices,
        "joint_failure": joint_failure,
        "joint_success": joint_success,
    }


def _build_pair_summary(
    server_ids: list,
    distance_matrix: np.ndarray,
    correlation_matrix: np.ndarray,
    pair_indices: np.ndarray,
    joint_failure: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for pair_position, (j, k) in enumerate(pair_indices):
        failure = joint_failure[:, :, pair_position].reshape(-1)
        success = 1.0 - failure
        rows.append(
            {
                "server_j": server_ids[j],
                "server_k": server_ids[k],
                "distance_km": float(distance_matrix[j, k]),
                "rho_jk": float(correlation_matrix[j, k]),
                "count": int(failure.size),
                "joint_failure_mean": float(failure.mean()),
                "joint_failure_p50": float(np.quantile(failure, 0.50)),
                "joint_failure_p90": float(np.quantile(failure, 0.90)),
                "joint_failure_p95": float(np.quantile(failure, 0.95)),
                "joint_failure_p99": float(np.quantile(failure, 0.99)),
                "joint_failure_max": float(failure.max()),
                "joint_success_mean": float(success.mean()),
                "joint_success_p01": float(np.quantile(success, 0.01)),
                "joint_success_p05": float(np.quantile(success, 0.05)),
                "joint_success_p10": float(np.quantile(success, 0.10)),
                "joint_success_p50": float(np.quantile(success, 0.50)),
                "joint_success_min": float(success.min()),
            }
        )
    return pd.DataFrame(rows)


def _build_sample(
    sample_rows: int,
    server_ids: list,
    task_ids: np.ndarray,
    computation_demands: np.ndarray,
    base_failure_rates: np.ndarray,
    processing_frequencies: np.ndarray,
    distance_matrix: np.ndarray,
    correlation_matrix: np.ndarray,
    data: dict,
) -> pd.DataFrame:
    joint_failure = data["joint_failure"]
    row_count = min(_positive_integer(sample_rows, "sample_rows"), joint_failure.size)
    flat_indices = np.linspace(
        0,
        joint_failure.size - 1,
        num=row_count,
        dtype=np.int64,
    )
    episode_indices, task_indices, pair_positions = np.unravel_index(
        flat_indices,
        joint_failure.shape,
    )
    pairs = data["pair_indices"][pair_positions]
    j_indices = pairs[:, 0]
    k_indices = pairs[:, 1]
    failure_probabilities = data["failure_probabilities"]
    p_j = failure_probabilities[episode_indices, task_indices, j_indices]
    p_k = failure_probabilities[episode_indices, task_indices, k_indices]
    sampled_joint_failure = joint_failure[
        episode_indices,
        task_indices,
        pair_positions,
    ]

    return pd.DataFrame(
        {
            "episode": episode_indices + 1,
            "task": task_ids[task_indices],
            "C_i": computation_demands[task_indices],
            "j": np.asarray(server_ids)[j_indices],
            "k": np.asarray(server_ids)[k_indices],
            "f_j": processing_frequencies[j_indices],
            "f_k": processing_frequencies[k_indices],
            "lambda_j_0": base_failure_rates[j_indices],
            "lambda_k_0": base_failure_rates[k_indices],
            "distance_km": distance_matrix[j_indices, k_indices],
            "rho_jk": correlation_matrix[j_indices, k_indices],
            "Z_j": data["spatial_fields"][episode_indices, j_indices],
            "Z_k": data["spatial_fields"][episode_indices, k_indices],
            "lambda_j_eff": data["effective_rates"][episode_indices, j_indices],
            "lambda_k_eff": data["effective_rates"][episode_indices, k_indices],
            "t_ij": data["service_times"][task_indices, j_indices],
            "t_ik": data["service_times"][task_indices, k_indices],
            "p_ij": p_j,
            "p_ik": p_k,
            "joint_failure": sampled_joint_failure,
            "joint_success": 1.0 - sampled_joint_failure,
        }
    )


def _build_task_load_summary(
    computation_demands: np.ndarray,
    joint_failure: np.ndarray,
) -> pd.DataFrame:
    labels = ("0-25%", "25-50%", "50-75%", "75-100%")
    ordered_tasks = np.argsort(computation_demands, kind="stable")
    rows = []
    for label, task_indices in zip(labels, np.array_split(ordered_tasks, 4)):
        failure = joint_failure[:, task_indices, :].reshape(-1)
        success = 1.0 - failure
        rows.append(
            {
                "task_load_quartile": label,
                "task_count": int(len(task_indices)),
                "sample_count": int(failure.size),
                "C_i_min": float(computation_demands[task_indices].min()),
                "C_i_max": float(computation_demands[task_indices].max()),
                "joint_failure_mean": float(failure.mean()),
                "joint_failure_p50": float(np.quantile(failure, 0.50)),
                "joint_failure_p90": float(np.quantile(failure, 0.90)),
                "joint_failure_p99": float(np.quantile(failure, 0.99)),
                "joint_success_mean": float(success.mean()),
                "joint_success_p50": float(np.quantile(success, 0.50)),
                "joint_success_p10": float(np.quantile(success, 0.10)),
                "joint_success_p01": float(np.quantile(success, 0.01)),
            }
        )
    return pd.DataFrame(rows)


def run_diagnostic(
    server_path: Path | str = Path(DATA_DIR) / "server_info.xlsx",
    task_path: Path | str = Path(DATA_DIR) / "task_parameters.xlsx",
    output_dir: Path | str = Path(RESULTS_DIR) / "spatial_risk_diagnostics",
    *,
    episodes: int = DEFAULT_EPISODES,
    tasks_per_episode: int = DEFAULT_TASKS_PER_EPISODE,
    correlation_length_km: float = DEFAULT_CORRELATION_LENGTH_KM,
    beta_p: float = DEFAULT_BETA_P,
    seed: int = DEFAULT_SEED,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> dict:
    """Run the fixed-parameter conditional reliability distribution diagnostic."""
    episode_count = _positive_integer(episodes, "episodes")
    task_count = _positive_integer(tasks_per_episode, "tasks_per_episode")
    sample_count = _positive_integer(sample_rows, "sample_rows")
    correlation_length = float(correlation_length_km)
    if not math.isfinite(correlation_length) or correlation_length <= 0.0:
        raise ValueError("correlation_length_km must be finite and positive")

    server_ids, base_rates, frequencies, distance_matrix = _load_servers(
        Path(server_path)
    )
    task_ids, computation_demands = _load_tasks(Path(task_path))
    if len(task_ids) < task_count:
        raise ValueError(
            f"task workbook contains {len(task_ids)} tasks; {task_count} are required"
        )
    task_ids = task_ids[:task_count]
    computation_demands = computation_demands[:task_count]

    correlation_matrix = build_spatial_correlation_matrix(
        distance_matrix,
        correlation_length,
    )
    validate_correlation_matrix(correlation_matrix)
    data = calculate_conditional_joint_reliability(
        base_rates,
        frequencies,
        computation_demands,
        correlation_matrix,
        episodes=episode_count,
        beta_p=beta_p,
        seed=seed,
    )

    joint_failure = data["joint_failure"]
    joint_success = data["joint_success"]
    summary = pd.DataFrame(
        [
            _summary_row("joint_failure_probability", joint_failure),
            _summary_row("joint_success_probability", joint_success),
        ]
    )
    pair_summary = _build_pair_summary(
        server_ids,
        distance_matrix,
        correlation_matrix,
        data["pair_indices"],
        joint_failure,
    )
    samples = _build_sample(
        sample_count,
        server_ids,
        task_ids,
        computation_demands,
        base_rates,
        frequencies,
        distance_matrix,
        correlation_matrix,
        data,
    )
    task_load_summary = _build_task_load_summary(computation_demands, joint_failure)
    threshold_coverage = {
        threshold: float(np.mean(joint_success >= threshold))
        for threshold in RELIABILITY_THRESHOLDS
    }

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": output_directory / "conditional_joint_reliability_summary.csv",
        "by_pair": output_directory / "conditional_joint_reliability_by_pair.csv",
        "samples": output_directory / "conditional_joint_reliability_samples.csv",
        "by_task_load": output_directory / "conditional_joint_reliability_by_task_load.csv",
    }
    summary.to_csv(outputs["summary"], index=False)
    pair_summary.to_csv(outputs["by_pair"], index=False)
    samples.to_csv(outputs["samples"], index=False)
    task_load_summary.to_csv(outputs["by_task_load"], index=False)

    print(
        f"episodes={episode_count}; tasks_per_episode={task_count}; "
        f"servers={len(server_ids)}; distinct_pairs={len(data['pair_indices'])}"
    )
    print(
        f"ell={correlation_length:g} km; beta={float(beta_p):g}; "
        f"seed={seed}; conditional_samples={joint_failure.size}"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.metric}: mean={row.mean:.12g}; min={row.min:.12g}; "
            f"p01={row.p01:.12g}; p50={row.p50:.12g}; "
            f"p99={row.p99:.12g}; max={row.max:.12g}"
        )
    for threshold, coverage in threshold_coverage.items():
        print(f"P(joint_success >= {threshold:g}) = {coverage:.8%}")
    for name, path in outputs.items():
        print(f"{name}: {path}")

    return {
        "summary": summary,
        "by_pair": pair_summary,
        "samples": samples,
        "by_task_load": task_load_summary,
        "threshold_coverage": threshold_coverage,
        "data": data,
        "outputs": outputs,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-input", type=Path, default=Path(DATA_DIR) / "server_info.xlsx"
    )
    parser.add_argument(
        "--task-input", type=Path, default=Path(DATA_DIR) / "task_parameters.xlsx"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(RESULTS_DIR) / "spatial_risk_diagnostics",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--tasks-per-episode", type=int, default=DEFAULT_TASKS_PER_EPISODE
    )
    parser.add_argument(
        "--correlation-length-km",
        type=float,
        default=DEFAULT_CORRELATION_LENGTH_KM,
    )
    parser.add_argument("--beta-p", type=float, default=DEFAULT_BETA_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_diagnostic(
        server_path=args.server_input,
        task_path=args.task_input,
        output_dir=args.output_dir,
        episodes=args.episodes,
        tasks_per_episode=args.tasks_per_episode,
        correlation_length_km=args.correlation_length_km,
        beta_p=args.beta_p,
        seed=args.seed,
        sample_rows=args.sample_rows,
    )


if __name__ == "__main__":
    main()
