"""Offline sensitivity of conditional joint reliability to failure-rate scale.

This tool reuses ``tools.analyze_conditional_joint_reliability``.  It samples
one matched set of episode-level spatial risk fields and evaluates the exact
conditional probabilities for every requested base failure-rate scale.  It
never samples task failures and never changes simulator configuration or
runtime behavior.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import DATA_DIR, RESULTS_DIR
from core.spatial_risk import build_spatial_correlation_matrix, validate_correlation_matrix
from tools.analyze_conditional_joint_reliability import (
    DEFAULT_BETA_P,
    DEFAULT_CORRELATION_LENGTH_KM,
    DEFAULT_EPISODES,
    DEFAULT_SEED,
    DEFAULT_TASKS_PER_EPISODE,
    _finite_positive,
    _load_servers,
    _load_tasks,
    calculate_conditional_joint_reliability,
)

DEFAULT_FAILURE_RATE_SCALES = (1.0, 3.0, 5.0, 10.0, 15.0)
DEFAULT_RELIABILITY_REQUIREMENTS = (0.9, 0.99, 0.999, 0.9999)
SUMMARY_QUANTILES = {
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


def _normalize_scales(scales: object) -> tuple[float, ...]:
    try:
        values = tuple(_finite_positive(value, "failure_rate_scale") for value in scales)
    except TypeError as exc:
        raise ValueError("failure_rate_scales must be an iterable") from exc
    if not values:
        raise ValueError("failure_rate_scales must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("failure_rate_scales must contain unique values")
    return values


def _normalize_requirements(requirements: object) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in requirements)
    except (TypeError, ValueError) as exc:
        raise ValueError("reliability_requirements must be numeric") from exc
    if not values:
        raise ValueError("reliability_requirements must not be empty")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("reliability_requirements must lie within [0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("reliability_requirements must contain unique values")
    return tuple(sorted(values))


def _quantile_values(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, quantile))
        for name, quantile in SUMMARY_QUANTILES.items()
    }


def calculate_failure_rate_scale_sensitivity(
    base_failure_rates: object,
    processing_frequencies: object,
    computation_demands: object,
    correlation_matrix: object,
    *,
    failure_rate_scales: object = DEFAULT_FAILURE_RATE_SCALES,
    episodes: int = DEFAULT_EPISODES,
    beta_p: float = DEFAULT_BETA_P,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Evaluate all scales on one matched set of spatial fields."""
    scales = _normalize_scales(failure_rate_scales)
    episode_count = _positive_integer(episodes, "episodes")
    validate_correlation_matrix(correlation_matrix)

    base_data = calculate_conditional_joint_reliability(
        base_failure_rates,
        processing_frequencies,
        computation_demands,
        correlation_matrix,
        episodes=episode_count,
        beta_p=beta_p,
        seed=seed,
        failure_rate_scale=1.0,
    )
    matched_fields = base_data["spatial_fields"]
    data_by_scale = {}
    for scale in scales:
        if scale == 1.0:
            data_by_scale[scale] = base_data
            continue
        data_by_scale[scale] = calculate_conditional_joint_reliability(
            base_failure_rates,
            processing_frequencies,
            computation_demands,
            correlation_matrix,
            episodes=episode_count,
            beta_p=beta_p,
            seed=seed,
            failure_rate_scale=scale,
            spatial_fields=matched_fields,
        )

    return {
        "scales": scales,
        "spatial_fields": matched_fields,
        "data_by_scale": data_by_scale,
    }


def _build_coverage(
    data_by_scale: dict[float, dict],
    requirements: tuple[float, ...],
) -> pd.DataFrame:
    rows = []
    for scale, data in data_by_scale.items():
        success = data["joint_success"].reshape(-1)
        total = int(success.size)
        for requirement in requirements:
            success_samples = int(np.count_nonzero(success >= requirement))
            rows.append(
                {
                    "failure_rate_scale": scale,
                    "reliability_requirement": requirement,
                    "total_samples": total,
                    "success_samples": success_samples,
                    "success_rate": success_samples / total,
                    "failure_rate": (total - success_samples) / total,
                }
            )
    return pd.DataFrame(rows)


def _build_summary(data_by_scale: dict[float, dict]) -> pd.DataFrame:
    rows = []
    for scale, data in data_by_scale.items():
        failure = data["joint_failure"].reshape(-1)
        success = data["joint_success"].reshape(-1)
        row = {
            "failure_rate_scale": scale,
            "joint_failure_mean": float(failure.mean()),
            "joint_failure_max": float(failure.max()),
            "joint_success_mean": float(success.mean()),
            "joint_success_min": float(success.min()),
            "joint_success_max": float(success.max()),
        }
        row.update({f"joint_failure_{key}": value for key, value in _quantile_values(failure).items()})
        row.update({f"joint_success_{key}": value for key, value in _quantile_values(success).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _build_threshold_matrix(coverage: pd.DataFrame) -> pd.DataFrame:
    matrix = coverage.pivot(
        index="failure_rate_scale",
        columns="reliability_requirement",
        values="success_rate",
    ).reset_index()
    matrix.columns = [
        "failure_rate_scale"
        if column == "failure_rate_scale"
        else f"R_{column:g}"
        for column in matrix.columns
    ]
    return matrix


def run_diagnostic(
    server_path: Path | str = Path(DATA_DIR) / "server_info.xlsx",
    task_path: Path | str = Path(DATA_DIR) / "task_parameters.xlsx",
    output_dir: Path | str = Path(RESULTS_DIR) / "spatial_risk_diagnostics",
    *,
    failure_rate_scales: object = DEFAULT_FAILURE_RATE_SCALES,
    reliability_requirements: object = DEFAULT_RELIABILITY_REQUIREMENTS,
    episodes: int = DEFAULT_EPISODES,
    tasks_per_episode: int = DEFAULT_TASKS_PER_EPISODE,
    correlation_length_km: float = DEFAULT_CORRELATION_LENGTH_KM,
    beta_p: float = DEFAULT_BETA_P,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run matched-environment failure-rate scaling sensitivity analysis."""
    scales = _normalize_scales(failure_rate_scales)
    requirements = _normalize_requirements(reliability_requirements)
    episode_count = _positive_integer(episodes, "episodes")
    task_count = _positive_integer(tasks_per_episode, "tasks_per_episode")
    correlation_length = _finite_positive(correlation_length_km, "correlation_length_km")

    server_ids, base_rates, frequencies, distance_matrix = _load_servers(Path(server_path))
    task_ids, computation_demands = _load_tasks(Path(task_path))
    if len(task_ids) < task_count:
        raise ValueError(
            f"task workbook contains {len(task_ids)} tasks; {task_count} are required"
        )
    task_ids = task_ids[:task_count]
    computation_demands = computation_demands[:task_count]
    correlation_matrix = build_spatial_correlation_matrix(
        distance_matrix, correlation_length
    )
    validate_correlation_matrix(correlation_matrix)

    result = calculate_failure_rate_scale_sensitivity(
        base_rates,
        frequencies,
        computation_demands,
        correlation_matrix,
        failure_rate_scales=scales,
        episodes=episode_count,
        beta_p=beta_p,
        seed=seed,
    )
    coverage = _build_coverage(result["data_by_scale"], requirements)
    summary = _build_summary(result["data_by_scale"])
    threshold_matrix = _build_threshold_matrix(coverage)

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "coverage": output_directory / "failure_rate_scale_reliability_coverage.csv",
        "summary": output_directory / "failure_rate_scale_joint_reliability_summary.csv",
        "threshold_matrix": output_directory / "failure_rate_scale_threshold_matrix.csv",
    }
    coverage.to_csv(outputs["coverage"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    threshold_matrix.to_csv(outputs["threshold_matrix"], index=False)

    print(
        f"episodes={episode_count}; tasks_per_episode={task_count}; "
        f"servers={len(server_ids)}; distinct_pairs={len(result['data_by_scale'][scales[0]]['pair_indices'])}"
    )
    print(
        f"ell={correlation_length:g} km; beta={float(beta_p):g}; seed={seed}; "
        f"matched_fields={result['spatial_fields'].shape[0]}; scales={list(scales)}"
    )
    print(coverage.to_string(index=False))
    for name, path in outputs.items():
        print(f"{name}: {path}")

    return {
        "server_ids": server_ids,
        "task_ids": task_ids,
        "computation_demands": computation_demands,
        "distance_matrix": distance_matrix,
        "correlation_matrix": correlation_matrix,
        "scales": scales,
        "requirements": requirements,
        "spatial_fields": result["spatial_fields"],
        "data_by_scale": result["data_by_scale"],
        "coverage": coverage,
        "summary": summary,
        "threshold_matrix": threshold_matrix,
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
    parser.add_argument("--failure-rate-scales", type=float, nargs="+", default=DEFAULT_FAILURE_RATE_SCALES)
    parser.add_argument("--reliability-requirements", type=float, nargs="+", default=DEFAULT_RELIABILITY_REQUIREMENTS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--tasks-per-episode", type=int, default=DEFAULT_TASKS_PER_EPISODE)
    parser.add_argument("--correlation-length-km", type=float, default=DEFAULT_CORRELATION_LENGTH_KM)
    parser.add_argument("--beta-p", type=float, default=DEFAULT_BETA_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_diagnostic(
        server_path=args.server_input,
        task_path=args.task_input,
        output_dir=args.output_dir,
        failure_rate_scales=args.failure_rate_scales,
        reliability_requirements=args.reliability_requirements,
        episodes=args.episodes,
        tasks_per_episode=args.tasks_per_episode,
        correlation_length_km=args.correlation_length_km,
        beta_p=args.beta_p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
