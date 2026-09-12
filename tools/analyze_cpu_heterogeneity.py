"""Analyze CPU/service-time heterogeneity as an offline sensitivity study.

All server failure rates are fixed at the original mean.  Original CPU
capacities are contracted toward their fixed mean as

    f_j(gamma) = mean(f) + gamma * (f_j - mean(f)).

For every C0-C3 scenario this tool reuses the existing Step 2E Monte Carlo
diagnostic, Step 3A backup-selection logic, and Step 3A-1 ranking logic.  It
does not run or modify the simulator, PPO, reward, or production failure model.
Temporary server workbooks isolate the study from data/server_info.xlsx.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import DATA_DIR, RESULTS_DIR
from tools.analyze_backup_ranking_competition import rank_backup_candidates
from tools.analyze_correlation_aware_backup_selection import (
    build_primary_candidates,
    compare_backup_selections,
)
from tools.analyze_pair_failure_correlation import (
    BETA_VALUES,
    DEFAULT_CORRELATION_LENGTH_KM,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    run_diagnostic as run_pair_failure_diagnostic,
)


CPU_HETEROGENEITY_SCENARIOS = (
    (0.0, "C0"),
    (1.0 / 3.0, "C1"),
    (2.0 / 3.0, "C2"),
    (1.0, "C3"),
)
RATIO_EPSILON = 1e-15
DEFAULT_SERVER_PATH = Path(DATA_DIR) / "server_info.xlsx"
DEFAULT_TASK_PATH = Path(DATA_DIR) / "task_parameters.xlsx"
DEFAULT_OUTPUT_DIR = Path(RESULTS_DIR) / "spatial_risk_diagnostics"
DEFAULT_H0_COMPARISON_PATH = (
    DEFAULT_OUTPUT_DIR / "heterogeneity_backup_selection_comparison.csv"
)
DEFAULT_H0_RANKING_PATH = (
    DEFAULT_OUTPUT_DIR / "heterogeneity_ranking_competition.csv"
)

COMPARISON_COLUMNS = (
    "Gamma",
    "CPU_Heterogeneity_Scenario",
    "Beta_p",
    "Task_ID",
    "Primary_Server",
    "CPU_Mean",
    "CPU_Std",
    "CPU_CV",
    "Primary_CPU",
    "Primary_Marginal_Failure",
    "Independent_Selected_Backup",
    "Independent_Backup_CPU",
    "Independent_Backup_Marginal_Failure",
    "Independent_Selected_Rho",
    "Independent_Selected_Distance_km",
    "Independent_Actual_Joint_Risk",
    "Correlation_Aware_Selected_Backup",
    "Correlation_Aware_Backup_CPU",
    "Correlation_Aware_Backup_Marginal_Failure",
    "Correlation_Aware_Selected_Rho",
    "Correlation_Aware_Selected_Distance_km",
    "Correlation_Aware_Actual_Joint_Risk",
    "Selection_Changed",
    "Absolute_Risk_Reduction",
    "Relative_Risk_Reduction",
)

RANKING_COLUMNS = (
    "Gamma",
    "CPU_Heterogeneity_Scenario",
    "Beta_p",
    "Task_ID",
    "Primary_Server",
    "CPU_Mean",
    "CPU_Std",
    "CPU_CV",
    "Best_Backup",
    "Second_Backup",
    "Best_Marginal",
    "Second_Marginal",
    "Marginal_Ratio",
    "Best_Rho",
    "Second_Rho",
    "Best_Amplification",
    "Second_Amplification",
    "Observed_Amplification_Ratio",
    "Required_Amplification_Ratio",
    "Reversal_Margin",
    "Actual_Reversal",
)


def _validate_original_cpus(original_cpus: object) -> np.ndarray:
    cpus = np.asarray(original_cpus, dtype=float)
    if cpus.ndim != 1 or cpus.size == 0:
        raise ValueError("original CPU capacities must be a non-empty vector")
    if not np.isfinite(cpus).all():
        raise ValueError("original CPU capacities must be finite")
    if (cpus <= 0.0).any():
        raise ValueError("original CPU capacities must be strictly positive")
    return cpus


def contract_cpu_capacities(original_cpus: object, gamma: object) -> np.ndarray:
    """Contract positive CPU capacities toward their mean."""
    cpus = _validate_original_cpus(original_cpus)
    try:
        gamma_value = float(gamma)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("gamma must be finite and within [0, 1]") from exc
    if not math.isfinite(gamma_value) or not 0.0 <= gamma_value <= 1.0:
        raise ValueError("gamma must be finite and within [0, 1]")

    mean_cpu = float(cpus.mean())
    if gamma_value == 0.0:
        return np.full_like(cpus, mean_cpu)
    if gamma_value == 1.0:
        return cpus.copy()
    return mean_cpu + gamma_value * (cpus - mean_cpu)


def cpu_statistics(cpus: object) -> tuple[float, float, float]:
    """Return population mean, population standard deviation, and CV."""
    values = _validate_original_cpus(cpus)
    mean_cpu = float(values.mean())
    std_cpu = float(values.std(ddof=0))
    return mean_cpu, std_cpu, float(std_cpu / mean_cpu)


def build_cpu_scenarios(
    server_ids: object,
    original_cpus: object,
    fixed_failure_rate: object,
) -> pd.DataFrame:
    """Build the deterministic C0-C3 CPU parameter table."""
    ids = np.asarray(server_ids)
    cpus = _validate_original_cpus(original_cpus)
    if ids.ndim != 1 or ids.size != cpus.size:
        raise ValueError("server_ids and original_cpus must have equal length")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("server_ids must be unique")
    try:
        fixed_rate = float(fixed_failure_rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fixed failure rate must be finite and non-negative") from exc
    if not math.isfinite(fixed_rate) or fixed_rate < 0.0:
        raise ValueError("fixed failure rate must be finite and non-negative")

    original_mean, original_std, original_cv = cpu_statistics(cpus)
    rows = []
    for gamma, scenario in CPU_HETEROGENEITY_SCENARIOS:
        scenario_cpus = contract_cpu_capacities(cpus, gamma)
        mean_cpu, std_cpu, cv_cpu = cpu_statistics(scenario_cpus)
        if not math.isclose(mean_cpu, original_mean, rel_tol=1e-14, abs_tol=1e-14):
            raise RuntimeError(f"{scenario} changed mean CPU capacity")
        if not math.isclose(std_cpu, gamma * original_std, rel_tol=1e-12, abs_tol=1e-14):
            raise RuntimeError(f"{scenario} CPU standard deviation scaling failed")
        if not math.isclose(cv_cpu, gamma * original_cv, rel_tol=1e-12, abs_tol=1e-14):
            raise RuntimeError(f"{scenario} CPU CV scaling failed")
        for server_id, original_cpu, scenario_cpu in zip(ids, cpus, scenario_cpus):
            rows.append({
                "Gamma": gamma,
                "CPU_Heterogeneity_Scenario": scenario,
                "Server_ID": int(server_id),
                "Original_CPU": original_cpu,
                "Scenario_CPU": scenario_cpu,
                "CPU_Mean": mean_cpu,
                "CPU_Std": std_cpu,
                "CPU_CV": cv_cpu,
                "Fixed_Baseline_Failure_Rate": fixed_rate,
            })
    return pd.DataFrame(rows)


def _scenario_metadata(cpu_table: pd.DataFrame, gamma: float):
    rows = cpu_table[cpu_table["Gamma"] == gamma]
    if rows.empty:
        raise ValueError(f"missing CPU scenario for gamma={gamma}")
    return (
        str(rows["CPU_Heterogeneity_Scenario"].iloc[0]),
        dict(zip(rows["Server_ID"].astype(int), rows["Scenario_CPU"].astype(float))),
        float(rows["CPU_Mean"].iloc[0]),
        float(rows["CPU_Std"].iloc[0]),
        float(rows["CPU_CV"].iloc[0]),
    )


def enforce_homogeneous_marginals(pair_detail: pd.DataFrame) -> pd.DataFrame:
    """Remove finite-sample marginal noise when all server inputs are equal.

    Under C0 every server has the same failure rate and service time for a
    fixed task, so its marginal failure distribution is identical.  Step 2E
    estimates each server marginal from a different Monte Carlo column.  This
    function pools those statistically equivalent estimates while preserving
    each pair's estimated covariance, making the exact C0 symmetry and
    deterministic Server_ID tie-break explicit.
    """
    result = pair_detail.copy()
    marginal_columns = (
        "Marginal_Failure_J_Spatial",
        "Marginal_Failure_K_Spatial",
    )
    for (beta_p, task_id), indices in result.groupby(
        ["Beta_p", "Task_ID"], sort=False
    ).groups.items():
        group = result.loc[indices]
        values_by_server: dict[int, list[float]] = {}
        for row in group.itertuples():
            values_by_server.setdefault(int(row.Server_J), []).append(
                float(row.Marginal_Failure_J_Spatial)
            )
            values_by_server.setdefault(int(row.Server_K), []).append(
                float(row.Marginal_Failure_K_Spatial)
            )
        server_estimates = []
        for server_id, estimates in values_by_server.items():
            if not np.allclose(estimates, estimates[0], rtol=0.0, atol=1e-15):
                raise RuntimeError(
                    f"inconsistent marginal estimate for server {server_id}, "
                    f"beta={beta_p}, task={task_id}"
                )
            server_estimates.append(estimates[0])
        common_marginal = float(np.mean(server_estimates))

        old_product = (
            group["Marginal_Failure_J_Spatial"].to_numpy(dtype=float)
            * group["Marginal_Failure_K_Spatial"].to_numpy(dtype=float)
        )
        covariance = group["Joint_Failure_Spatial"].to_numpy(dtype=float) - old_product
        new_independent = common_marginal * common_marginal
        new_joint = new_independent + covariance
        result.loc[indices, list(marginal_columns)] = common_marginal
        for column in (
            "Marginal_Failure_J_Independent",
            "Marginal_Failure_K_Independent",
        ):
            if column in result.columns:
                result.loc[indices, column] = common_marginal
        result.loc[indices, "Joint_Failure_Independent"] = new_independent
        result.loc[indices, "Joint_Failure_Spatial"] = new_joint
        if "Joint_Failure_Amplification" in result.columns:
            result.loc[indices, "Joint_Failure_Amplification"] = np.divide(
                new_joint,
                new_independent,
                out=np.full(new_joint.shape, np.nan),
                where=abs(new_independent) > RATIO_EPSILON,
            )
        if "Excess_Joint_Failure" in result.columns:
            result.loc[indices, "Excess_Joint_Failure"] = covariance
        if "Relative_Joint_Underestimation" in result.columns:
            result.loc[indices, "Relative_Joint_Underestimation"] = np.divide(
                covariance,
                new_independent,
                out=np.full(covariance.shape, np.nan),
                where=abs(new_independent) > RATIO_EPSILON,
            )
        if "Reliability_Overestimation" in result.columns:
            result.loc[indices, "Reliability_Overestimation"] = covariance
    return result


def validate_homogeneous_marginals(pair_detail: pd.DataFrame) -> None:
    """Require equal C0 marginals and service times for each task and beta."""
    candidates = build_primary_candidates(pair_detail)
    for key, group in candidates.groupby(["Beta_p", "Task_ID"], sort=False):
        marginals = group["Backup_Marginal_Failure"].to_numpy(dtype=float)
        if not np.allclose(marginals, marginals[0], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"C0 marginal equality failed for beta/task={key}")
    for _, group in pair_detail.groupby("Task_ID", sort=False):
        service_times = np.concatenate([
            group["Service_Time_J"].to_numpy(dtype=float),
            group["Service_Time_K"].to_numpy(dtype=float),
        ])
        if not np.allclose(service_times, service_times[0], rtol=0.0, atol=1e-14):
            raise RuntimeError("C0 service-time equality failed")


def decorate_selection_comparison(
    comparison: pd.DataFrame,
    gamma: float,
    scenario: str,
    cpu_by_server: dict[int, float],
    cpu_mean: float,
    cpu_std: float,
    cpu_cv: float,
) -> pd.DataFrame:
    result = comparison.copy().rename(columns={
        "Independent_Selected_Rho_phy": "Independent_Selected_Rho",
        "Correlation_Aware_Selected_Rho_phy": "Correlation_Aware_Selected_Rho",
    })
    result["Gamma"] = gamma
    result["CPU_Heterogeneity_Scenario"] = scenario
    result["CPU_Mean"] = cpu_mean
    result["CPU_Std"] = cpu_std
    result["CPU_CV"] = cpu_cv
    result["Primary_CPU"] = result["Primary_Server"].map(cpu_by_server)
    result["Independent_Backup_CPU"] = result["Independent_Selected_Backup"].map(
        cpu_by_server
    )
    result["Correlation_Aware_Backup_CPU"] = result[
        "Correlation_Aware_Selected_Backup"
    ].map(cpu_by_server)
    if result[[
        "Primary_CPU",
        "Independent_Backup_CPU",
        "Correlation_Aware_Backup_CPU",
    ]].isna().any().any():
        raise ValueError("selection contains an unknown server ID")
    return result.loc[:, COMPARISON_COLUMNS]


def decorate_ranking_competition(
    ranking: pd.DataFrame,
    comparison: pd.DataFrame,
    gamma: float,
    scenario: str,
    cpu_mean: float,
    cpu_std: float,
    cpu_cv: float,
) -> pd.DataFrame:
    keys = ["Beta_p", "Task_ID", "Primary_Server"]
    actual = comparison[keys + ["Selection_Changed"]].rename(
        columns={"Selection_Changed": "Actual_Reversal"}
    )
    result = ranking.merge(actual, on=keys, validate="one_to_one").rename(columns={
        "Marginal_Gap_Ratio_1_2": "Marginal_Ratio",
        "Required_Amplification_Ratio_For_Reversal": "Required_Amplification_Ratio",
    })
    result["Gamma"] = gamma
    result["CPU_Heterogeneity_Scenario"] = scenario
    result["CPU_Mean"] = cpu_mean
    result["CPU_Std"] = cpu_std
    result["CPU_CV"] = cpu_cv
    return result.loc[:, RANKING_COLUMNS]


def _finite_statistic(values: pd.Series, operation: str) -> float:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    return float(getattr(finite, operation)()) if not finite.empty else float("nan")


def summarize_sensitivity(comparison: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["Gamma", "CPU_Heterogeneity_Scenario", "Beta_p"]
    for key, group in comparison.groupby(keys, sort=True):
        gamma, scenario, beta_p = key
        ranked = ranking[
            (ranking["Gamma"] == gamma)
            & (ranking["CPU_Heterogeneity_Scenario"] == scenario)
            & (ranking["Beta_p"] == beta_p)
        ]
        if len(ranked) != len(group):
            raise RuntimeError("comparison and ranking case counts differ")
        changes = int(group["Selection_Changed"].sum())
        reversals = int(ranked["Actual_Reversal"].sum())
        rows.append({
            "Gamma": gamma,
            "CPU_Heterogeneity_Scenario": scenario,
            "Beta_p": beta_p,
            "CPU_Mean": group["CPU_Mean"].iloc[0],
            "CPU_Std": group["CPU_Std"].iloc[0],
            "CPU_CV": group["CPU_CV"].iloc[0],
            "Case_Count": len(group),
            "Selection_Change_Count": changes,
            "Selection_Change_Rate": changes / len(group),
            "Mean_Independent_Actual_Joint_Risk": group["Independent_Actual_Joint_Risk"].mean(),
            "Mean_Correlation_Aware_Actual_Joint_Risk": group["Correlation_Aware_Actual_Joint_Risk"].mean(),
            "Mean_Absolute_Risk_Reduction": group["Absolute_Risk_Reduction"].mean(),
            "Mean_Relative_Risk_Reduction": _finite_statistic(group["Relative_Risk_Reduction"], "mean"),
            "Median_Relative_Risk_Reduction": _finite_statistic(group["Relative_Risk_Reduction"], "median"),
            "Max_Relative_Risk_Reduction": _finite_statistic(group["Relative_Risk_Reduction"], "max"),
            "Fraction_Positive_Risk_Reduction": (group["Absolute_Risk_Reduction"] > 0.0).mean(),
            "Mean_Independent_Selected_Rho": group["Independent_Selected_Rho"].mean(),
            "Mean_Correlation_Aware_Selected_Rho": group["Correlation_Aware_Selected_Rho"].mean(),
            "Mean_Independent_Selected_Distance": group["Independent_Selected_Distance_km"].mean(),
            "Mean_Correlation_Aware_Selected_Distance": group["Correlation_Aware_Selected_Distance_km"].mean(),
            "Mean_Marginal_Ratio": _finite_statistic(ranked["Marginal_Ratio"], "mean"),
            "Median_Marginal_Ratio": _finite_statistic(ranked["Marginal_Ratio"], "median"),
            "Min_Marginal_Ratio": _finite_statistic(ranked["Marginal_Ratio"], "min"),
            "Mean_Observed_Amplification_Ratio": _finite_statistic(ranked["Observed_Amplification_Ratio"], "mean"),
            "Max_Observed_Amplification_Ratio": _finite_statistic(ranked["Observed_Amplification_Ratio"], "max"),
            "Mean_Reversal_Margin": _finite_statistic(ranked["Reversal_Margin"], "mean"),
            "Max_Reversal_Margin": _finite_statistic(ranked["Reversal_Margin"], "max"),
            "Reversal_Count": reversals,
            "Reversal_Rate": reversals / len(ranked),
        })
    return pd.DataFrame(rows)


def validate_beta_zero_controls(comparison: pd.DataFrame, ranking: pd.DataFrame) -> None:
    beta_zero = comparison[comparison["Beta_p"] == 0.0]
    if beta_zero["Selection_Changed"].any():
        raise RuntimeError("beta=0 must not change backup selection")
    if not (beta_zero["Absolute_Risk_Reduction"] == 0.0).all():
        raise RuntimeError("beta=0 risk reduction must be exactly zero")
    if ranking.loc[ranking["Beta_p"] == 0.0, "Actual_Reversal"].any():
        raise RuntimeError("beta=0 must not contain a reversal")


def validate_c3_against_h0(
    c3_comparison: pd.DataFrame,
    c3_ranking: pd.DataFrame,
    h0_comparison: pd.DataFrame,
    h0_ranking: pd.DataFrame,
) -> None:
    """Require C3 to reproduce Step 3A-2 alpha=0/H0."""
    h0_comparison = h0_comparison[h0_comparison["Alpha"] == 0.0]
    h0_ranking = h0_ranking[h0_ranking["Alpha"] == 0.0]
    keys = ["Beta_p", "Task_ID", "Primary_Server"]
    selection_columns = (
        "Independent_Selected_Backup",
        "Correlation_Aware_Selected_Backup",
        "Selection_Changed",
    )
    risk_columns = (
        "Independent_Actual_Joint_Risk",
        "Correlation_Aware_Actual_Joint_Risk",
    )
    merged = c3_comparison.merge(
        h0_comparison,
        on=keys,
        how="outer",
        suffixes=("_C3", "_H0"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("C3 and Step 3A-2 H0 contain different cases")
    for column in selection_columns:
        if not (merged[f"{column}_C3"] == merged[f"{column}_H0"]).all():
            raise RuntimeError(f"C3 does not reproduce H0 {column}")
    for column in risk_columns:
        if not np.allclose(merged[f"{column}_C3"], merged[f"{column}_H0"], rtol=1e-12, atol=1e-15):
            raise RuntimeError(f"C3 does not reproduce H0 {column}")

    ranked = c3_ranking.merge(
        h0_ranking[keys + ["Actual_Reversal"]],
        on=keys,
        how="outer",
        suffixes=("_C3", "_H0"),
        validate="one_to_one",
        indicator=True,
    )
    if not ranked["_merge"].eq("both").all():
        raise RuntimeError("C3 and H0 ranking cases differ")
    if not (ranked["Actual_Reversal_C3"] == ranked["Actual_Reversal_H0"]).all():
        raise RuntimeError("C3 does not reproduce H0 reversal results")


def _load_server_parameters(server_path: Path | str) -> pd.DataFrame:
    path = Path(server_path)
    if not path.is_file():
        raise FileNotFoundError(f"server parameter file not found: {path}")
    frame = pd.read_excel(path)
    required = {"Server_ID", "Processing_Frequency", "Failure_Rate"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("server_info.xlsx is missing required columns: " + ", ".join(missing))
    if frame["Server_ID"].duplicated().any():
        raise ValueError("Server_ID must be unique")
    _validate_original_cpus(frame["Processing_Frequency"].to_numpy(dtype=float))
    rates = frame["Failure_Rate"].to_numpy(dtype=float)
    if not np.isfinite(rates).all() or (rates < 0.0).any():
        raise ValueError("failure rates must be finite and non-negative")
    return frame


def _first_reversal_scenario(summary: pd.DataFrame):
    rows = summary[summary["Reversal_Count"] > 0]
    if rows.empty:
        return None
    return rows.sort_values(["Gamma", "Beta_p"], ascending=[False, True], kind="mergesort").iloc[0]


def _print_results(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    output_paths: tuple[Path, Path, Path, Path],
) -> None:
    print(
        "gamma | scenario | beta | CPU CV | selection change | mean relative "
        "reduction | max relative reduction | max reversal margin | reversal rate"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.Gamma:.6f} | {row.CPU_Heterogeneity_Scenario} | {row.Beta_p:.1f} | "
            f"{row.CPU_CV:.6f} | {row.Selection_Change_Rate:.6f} | "
            f"{row.Mean_Relative_Risk_Reduction:.6f} | "
            f"{row.Max_Relative_Risk_Reduction:.6f} | "
            f"{row.Max_Reversal_Margin:.6f} | {row.Reversal_Rate:.6f}"
        )
    first = _first_reversal_scenario(summary)
    if first is None:
        print("\nNo actual reversal occurred in any gamma-beta scenario.")
    else:
        print(
            "\nFirst reversal while reducing CPU heterogeneity: "
            f"{first['CPU_Heterogeneity_Scenario']}, "
            f"gamma={first['Gamma']:.6f}, beta={first['Beta_p']:.1f}"
        )

    print("\nbeta=0.8 C0-C3 summary:")
    columns = [
        "Gamma", "CPU_Heterogeneity_Scenario", "CPU_CV",
        "Selection_Change_Rate", "Mean_Relative_Risk_Reduction",
        "Max_Relative_Risk_Reduction", "Mean_Independent_Selected_Rho",
        "Mean_Correlation_Aware_Selected_Rho", "Mean_Reversal_Margin",
        "Max_Reversal_Margin", "Reversal_Rate",
    ]
    print(summary[summary["Beta_p"] == 0.8][columns].to_string(index=False))

    c0 = comparison[(comparison["Gamma"] == 0.0) & (comparison["Beta_p"] == 0.8)].copy()
    c0 = c0.sort_values(
        ["Relative_Risk_Reduction", "Task_ID", "Primary_Server"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(10)
    c0_columns = [
        "Task_ID", "Primary_Server", "Independent_Selected_Backup",
        "Correlation_Aware_Selected_Backup", "Independent_Backup_Marginal_Failure",
        "Independent_Selected_Rho", "Correlation_Aware_Selected_Rho",
        "Independent_Selected_Distance_km", "Correlation_Aware_Selected_Distance_km",
        "Independent_Actual_Joint_Risk", "Correlation_Aware_Actual_Joint_Risk",
        "Relative_Risk_Reduction",
    ]
    print("\nC0 beta=0.8 top 10 relative-risk reductions:")
    print(c0[c0_columns].to_string(index=False, float_format=lambda value: f"{value:.10g}"))
    for label, path in zip(("CPU parameters", "Selection", "Ranking", "Summary"), output_paths):
        print(f"{label} output: {path}")


def run_diagnostic(
    server_path: Path | str = DEFAULT_SERVER_PATH,
    task_path: Path | str = DEFAULT_TASK_PATH,
    h0_comparison_path: Path | str = DEFAULT_H0_COMPARISON_PATH,
    h0_ranking_path: Path | str = DEFAULT_H0_RANKING_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    correlation_length_km: float = DEFAULT_CORRELATION_LENGTH_KM,
):
    """Run C0-C3 through the existing offline dependence diagnostics."""
    server_frame = _load_server_parameters(server_path)
    server_ids = server_frame["Server_ID"].to_numpy(dtype=int)
    original_cpus = server_frame["Processing_Frequency"].to_numpy(dtype=float)
    fixed_rate = float(server_frame["Failure_Rate"].to_numpy(dtype=float).mean())
    cpu_table = build_cpu_scenarios(server_ids, original_cpus, fixed_rate)

    comparisons = []
    rankings = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        for gamma, _ in CPU_HETEROGENEITY_SCENARIOS:
            scenario, cpu_by_server, cpu_mean, cpu_std, cpu_cv = _scenario_metadata(cpu_table, gamma)
            scenario_servers = server_frame.copy()
            scenario_servers["Failure_Rate"] = fixed_rate
            scenario_servers["Processing_Frequency"] = scenario_servers["Server_ID"].map(cpu_by_server)
            scenario_server_path = temporary_root / f"{scenario}_server_info.xlsx"
            scenario_output_dir = temporary_root / scenario
            scenario_servers.to_excel(scenario_server_path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                pair_detail, _, _ = run_pair_failure_diagnostic(
                    server_path=scenario_server_path,
                    task_path=task_path,
                    output_dir=scenario_output_dir,
                    seed=seed,
                    sample_count=sample_count,
                    correlation_length_km=correlation_length_km,
                )
            if set(pair_detail["Beta_p"].unique()) != set(BETA_VALUES):
                raise RuntimeError(f"{scenario} produced unexpected beta values")
            if gamma == 0.0:
                pair_detail = enforce_homogeneous_marginals(pair_detail)
                validate_homogeneous_marginals(pair_detail)

            base_comparison = compare_backup_selections(pair_detail)
            base_ranking = rank_backup_candidates(pair_detail)
            comparisons.append(decorate_selection_comparison(
                base_comparison, gamma, scenario, cpu_by_server,
                cpu_mean, cpu_std, cpu_cv,
            ))
            rankings.append(decorate_ranking_competition(
                base_ranking, base_comparison, gamma, scenario,
                cpu_mean, cpu_std, cpu_cv,
            ))

    comparison = pd.concat(comparisons, ignore_index=True)
    ranking = pd.concat(rankings, ignore_index=True)
    summary = summarize_sensitivity(comparison, ranking)
    validate_beta_zero_controls(comparison, ranking)
    validate_c3_against_h0(
        comparison[comparison["Gamma"] == 1.0],
        ranking[ranking["Gamma"] == 1.0],
        pd.read_csv(h0_comparison_path),
        pd.read_csv(h0_ranking_path),
    )

    for frame in (comparison, ranking, summary, cpu_table):
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise RuntimeError("diagnostic output contains infinity")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    parameter_path = output_directory / "cpu_heterogeneity_parameters.csv"
    comparison_path = output_directory / "cpu_heterogeneity_backup_selection.csv"
    ranking_path = output_directory / "cpu_heterogeneity_ranking_competition.csv"
    summary_path = output_directory / "cpu_heterogeneity_sensitivity_summary.csv"
    cpu_table.to_csv(parameter_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    summary.to_csv(summary_path, index=False)
    _print_results(
        comparison,
        summary,
        (parameter_path, comparison_path, ranking_path, summary_path),
    )
    return comparison, ranking, summary, cpu_table


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-input", type=Path, default=DEFAULT_SERVER_PATH)
    parser.add_argument("--task-input", type=Path, default=DEFAULT_TASK_PATH)
    parser.add_argument("--h0-comparison-input", type=Path, default=DEFAULT_H0_COMPARISON_PATH)
    parser.add_argument("--h0-ranking-input", type=Path, default=DEFAULT_H0_RANKING_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_diagnostic(
        server_path=args.server_input,
        task_path=args.task_input,
        h0_comparison_path=args.h0_comparison_input,
        h0_ranking_path=args.h0_ranking_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
