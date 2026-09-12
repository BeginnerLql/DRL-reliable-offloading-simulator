"""Analyze baseline failure-rate heterogeneity as an offline sensitivity study.

The original eight server failure rates are contracted toward their fixed mean:

    lambda_j(alpha) = mean(lambda) + alpha * (lambda_j - mean(lambda)).

Only alpha changes.  For every scenario this tool reuses the existing Step 2E
Monte Carlo diagnostic, Step 3A backup selection, and Step 3A-1 ranking logic.
Server CPU frequencies, locations, tasks, spatial correlation, beta values,
sample count, and random seed remain unchanged.  Temporary scenario workbooks
are used; data/server_info.xlsx and all existing diagnostic CSVs are read-only.
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
from tools.analyze_backup_ranking_competition import (
    rank_backup_candidates,
)
from tools.analyze_correlation_aware_backup_selection import (
    compare_backup_selections,
)
from tools.analyze_pair_failure_correlation import (
    BETA_VALUES,
    DEFAULT_CORRELATION_LENGTH_KM,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    run_diagnostic as run_pair_failure_diagnostic,
)


HETEROGENEITY_SCENARIOS = (
    (0.0, "H0"),
    (1.0 / 3.0, "H1"),
    (2.0 / 3.0, "H2"),
    (1.0, "H3"),
)
RATIO_EPSILON = 1e-15
DEFAULT_SERVER_PATH = Path(DATA_DIR) / "server_info.xlsx"
DEFAULT_TASK_PATH = Path(DATA_DIR) / "task_parameters.xlsx"
DEFAULT_OUTPUT_DIR = Path(RESULTS_DIR) / "spatial_risk_diagnostics"
DEFAULT_STEP3A_PATH = DEFAULT_OUTPUT_DIR / "backup_selection_comparison.csv"

COMPARISON_COLUMNS = (
    "Alpha",
    "Heterogeneity_Scenario",
    "Beta_p",
    "Task_ID",
    "Primary_Server",
    "Lambda_Mean",
    "Lambda_Std",
    "Lambda_CV",
    "Primary_Baseline_Lambda",
    "Primary_Marginal_Failure",
    "Independent_Selected_Backup",
    "Independent_Backup_Lambda",
    "Independent_Backup_Marginal_Failure",
    "Independent_Selected_Rho_phy",
    "Independent_Selected_Distance_km",
    "Independent_Actual_Joint_Risk",
    "Correlation_Aware_Selected_Backup",
    "Correlation_Aware_Backup_Lambda",
    "Correlation_Aware_Backup_Marginal_Failure",
    "Correlation_Aware_Selected_Rho_phy",
    "Correlation_Aware_Selected_Distance_km",
    "Correlation_Aware_Actual_Joint_Risk",
    "Selection_Changed",
    "Absolute_Risk_Reduction",
    "Relative_Risk_Reduction",
)

RANKING_COLUMNS = (
    "Alpha",
    "Heterogeneity_Scenario",
    "Beta_p",
    "Task_ID",
    "Primary_Server",
    "Best_Backup",
    "Second_Backup",
    "Best_Marginal",
    "Second_Marginal",
    "Marginal_Ratio",
    "Best_Rho",
    "Second_Rho",
    "Best_Amplification",
    "Second_Amplification",
    "Required_Amplification_Ratio",
    "Observed_Amplification_Ratio",
    "Reversal_Margin",
    "Actual_Reversal",
)


def _validate_original_rates(original_rates: object) -> np.ndarray:
    rates = np.asarray(original_rates, dtype=float)
    if rates.ndim != 1 or rates.size == 0:
        raise ValueError("original failure rates must be a non-empty vector")
    if not np.isfinite(rates).all():
        raise ValueError("original failure rates must be finite")
    if (rates < 0.0).any():
        raise ValueError("original failure rates must be non-negative")
    return rates


def contract_failure_rates(
    original_rates: object,
    alpha: object,
) -> np.ndarray:
    """Contract rates toward their mean while preserving that mean."""
    rates = _validate_original_rates(original_rates)
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("alpha must be finite and within [0, 1]") from exc
    if not math.isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
        raise ValueError("alpha must be finite and within [0, 1]")

    mean_rate = float(rates.mean())
    if alpha_value == 0.0:
        return np.full_like(rates, mean_rate)
    if alpha_value == 1.0:
        return rates.copy()
    return mean_rate + alpha_value * (rates - mean_rate)


def failure_rate_statistics(rates: object) -> tuple[float, float, float]:
    """Return population mean, population standard deviation, and CV."""
    values = _validate_original_rates(rates)
    mean_rate = float(values.mean())
    std_rate = float(values.std(ddof=0))
    cv_rate = (
        float(std_rate / mean_rate)
        if abs(mean_rate) > RATIO_EPSILON
        else float("nan")
    )
    return mean_rate, std_rate, cv_rate


def build_failure_rate_scenarios(
    server_ids: object,
    original_rates: object,
) -> pd.DataFrame:
    """Build the deterministic H0-H3 server-rate parameter table."""
    ids = np.asarray(server_ids)
    rates = _validate_original_rates(original_rates)
    if ids.ndim != 1 or ids.size != rates.size:
        raise ValueError("server_ids and original_rates must have equal length")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("server_ids must be unique")

    original_mean, original_std, original_cv = failure_rate_statistics(rates)
    rows = []
    for alpha, scenario in HETEROGENEITY_SCENARIOS:
        scenario_rates = contract_failure_rates(rates, alpha)
        mean_rate, std_rate, cv_rate = failure_rate_statistics(scenario_rates)
        if not math.isclose(mean_rate, original_mean, rel_tol=1e-14, abs_tol=1e-18):
            raise RuntimeError(f"{scenario} changed the mean failure rate")
        if not math.isclose(std_rate, alpha * original_std, rel_tol=1e-12, abs_tol=1e-18):
            raise RuntimeError(f"{scenario} standard deviation scaling failed")
        expected_cv = alpha * original_cv
        if not math.isclose(cv_rate, expected_cv, rel_tol=1e-12, abs_tol=1e-18):
            raise RuntimeError(f"{scenario} CV scaling failed")
        for server_id, original_rate, scenario_rate in zip(
            ids,
            rates,
            scenario_rates,
        ):
            rows.append({
                "Alpha": alpha,
                "Heterogeneity_Scenario": scenario,
                "Server_ID": int(server_id),
                "Original_Failure_Rate": original_rate,
                "Scenario_Failure_Rate": scenario_rate,
                "Lambda_Mean": mean_rate,
                "Lambda_Std": std_rate,
                "Lambda_CV": cv_rate,
            })
    return pd.DataFrame(rows)


def _scenario_metadata(
    failure_rate_table: pd.DataFrame,
    alpha: float,
) -> tuple[str, dict[int, float], float, float, float]:
    scenario_rows = failure_rate_table[
        failure_rate_table["Alpha"] == alpha
    ]
    if scenario_rows.empty:
        raise ValueError(f"missing failure-rate scenario for alpha={alpha}")
    scenario = str(scenario_rows["Heterogeneity_Scenario"].iloc[0])
    rate_by_server = dict(zip(
        scenario_rows["Server_ID"].astype(int),
        scenario_rows["Scenario_Failure_Rate"].astype(float),
    ))
    return (
        scenario,
        rate_by_server,
        float(scenario_rows["Lambda_Mean"].iloc[0]),
        float(scenario_rows["Lambda_Std"].iloc[0]),
        float(scenario_rows["Lambda_CV"].iloc[0]),
    )


def decorate_selection_comparison(
    comparison: pd.DataFrame,
    alpha: float,
    scenario: str,
    rate_by_server: dict[int, float],
    lambda_mean: float,
    lambda_std: float,
    lambda_cv: float,
) -> pd.DataFrame:
    """Attach scenario parameters and selected-server rates."""
    result = comparison.copy()
    result["Alpha"] = alpha
    result["Heterogeneity_Scenario"] = scenario
    result["Lambda_Mean"] = lambda_mean
    result["Lambda_Std"] = lambda_std
    result["Lambda_CV"] = lambda_cv
    result["Primary_Baseline_Lambda"] = result["Primary_Server"].map(
        rate_by_server
    )
    result["Independent_Backup_Lambda"] = result[
        "Independent_Selected_Backup"
    ].map(rate_by_server)
    result["Correlation_Aware_Backup_Lambda"] = result[
        "Correlation_Aware_Selected_Backup"
    ].map(rate_by_server)
    if result[[
        "Primary_Baseline_Lambda",
        "Independent_Backup_Lambda",
        "Correlation_Aware_Backup_Lambda",
    ]].isna().any().any():
        raise ValueError("selection contains an unknown server ID")
    return result.loc[:, COMPARISON_COLUMNS]


def decorate_ranking_competition(
    ranking: pd.DataFrame,
    comparison: pd.DataFrame,
    alpha: float,
    scenario: str,
) -> pd.DataFrame:
    """Attach scenario identity and actual selection reversal."""
    keys = ["Beta_p", "Task_ID", "Primary_Server"]
    actual_reversal = comparison[keys + ["Selection_Changed"]].rename(
        columns={"Selection_Changed": "Actual_Reversal"}
    )
    result = ranking.merge(
        actual_reversal,
        on=keys,
        how="inner",
        validate="one_to_one",
    ).rename(columns={
        "Marginal_Gap_Ratio_1_2": "Marginal_Ratio",
        "Required_Amplification_Ratio_For_Reversal": (
            "Required_Amplification_Ratio"
        ),
    })
    result["Alpha"] = alpha
    result["Heterogeneity_Scenario"] = scenario
    return result.loc[:, RANKING_COLUMNS]


def _finite_statistic(values: pd.Series, operation: str) -> float:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite.empty:
        return float("nan")
    return float(getattr(finite, operation)())


def summarize_sensitivity(
    comparison: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate selection and reversal metrics for all alpha-beta scenarios."""
    rows = []
    group_keys = ["Alpha", "Heterogeneity_Scenario", "Beta_p"]
    for key, comparison_group in comparison.groupby(group_keys, sort=True):
        alpha, scenario, beta_p = key
        ranking_group = ranking[
            (ranking["Alpha"] == alpha)
            & (ranking["Heterogeneity_Scenario"] == scenario)
            & (ranking["Beta_p"] == beta_p)
        ]
        if len(ranking_group) != len(comparison_group):
            raise RuntimeError("comparison and ranking case counts differ")

        change_count = int(comparison_group["Selection_Changed"].sum())
        reversal_count = int(ranking_group["Actual_Reversal"].sum())
        rows.append({
            "Alpha": alpha,
            "Heterogeneity_Scenario": scenario,
            "Beta_p": beta_p,
            "Lambda_Mean": comparison_group["Lambda_Mean"].iloc[0],
            "Lambda_Std": comparison_group["Lambda_Std"].iloc[0],
            "Lambda_CV": comparison_group["Lambda_CV"].iloc[0],
            "Case_Count": int(len(comparison_group)),
            "Selection_Change_Count": change_count,
            "Selection_Change_Rate": change_count / len(comparison_group),
            "Mean_Independent_Actual_Joint_Risk": comparison_group[
                "Independent_Actual_Joint_Risk"
            ].mean(),
            "Mean_Correlation_Aware_Actual_Joint_Risk": comparison_group[
                "Correlation_Aware_Actual_Joint_Risk"
            ].mean(),
            "Mean_Absolute_Risk_Reduction": comparison_group[
                "Absolute_Risk_Reduction"
            ].mean(),
            "Mean_Relative_Risk_Reduction": _finite_statistic(
                comparison_group["Relative_Risk_Reduction"],
                "mean",
            ),
            "Median_Relative_Risk_Reduction": _finite_statistic(
                comparison_group["Relative_Risk_Reduction"],
                "median",
            ),
            "Max_Relative_Risk_Reduction": _finite_statistic(
                comparison_group["Relative_Risk_Reduction"],
                "max",
            ),
            "Fraction_Positive_Risk_Reduction": (
                comparison_group["Absolute_Risk_Reduction"] > 0.0
            ).mean(),
            "Mean_Independent_Selected_Rho": comparison_group[
                "Independent_Selected_Rho_phy"
            ].mean(),
            "Mean_Correlation_Aware_Selected_Rho": comparison_group[
                "Correlation_Aware_Selected_Rho_phy"
            ].mean(),
            "Mean_Independent_Selected_Distance_km": comparison_group[
                "Independent_Selected_Distance_km"
            ].mean(),
            "Mean_Correlation_Aware_Selected_Distance_km": comparison_group[
                "Correlation_Aware_Selected_Distance_km"
            ].mean(),
            "Mean_Marginal_Ratio": _finite_statistic(
                ranking_group["Marginal_Ratio"],
                "mean",
            ),
            "Median_Marginal_Ratio": _finite_statistic(
                ranking_group["Marginal_Ratio"],
                "median",
            ),
            "Min_Marginal_Ratio": _finite_statistic(
                ranking_group["Marginal_Ratio"],
                "min",
            ),
            "Mean_Observed_Amplification_Ratio": _finite_statistic(
                ranking_group["Observed_Amplification_Ratio"],
                "mean",
            ),
            "Max_Observed_Amplification_Ratio": _finite_statistic(
                ranking_group["Observed_Amplification_Ratio"],
                "max",
            ),
            "Mean_Reversal_Margin": _finite_statistic(
                ranking_group["Reversal_Margin"],
                "mean",
            ),
            "Max_Reversal_Margin": _finite_statistic(
                ranking_group["Reversal_Margin"],
                "max",
            ),
            "Reversal_Count": reversal_count,
            "Reversal_Rate": reversal_count / len(ranking_group),
        })
    return pd.DataFrame(rows)


def validate_beta_zero_controls(
    comparison: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    """Require independence controls to hold for every alpha."""
    beta_zero_comparison = comparison[comparison["Beta_p"] == 0.0]
    beta_zero_ranking = ranking[ranking["Beta_p"] == 0.0]
    if beta_zero_comparison["Selection_Changed"].any():
        raise RuntimeError("beta=0 must not change backup selection")
    if not (
        beta_zero_comparison["Absolute_Risk_Reduction"] == 0.0
    ).all():
        raise RuntimeError("beta=0 risk reduction must be exactly zero")
    if beta_zero_ranking["Actual_Reversal"].any():
        raise RuntimeError("beta=0 must not contain a reversal")


def validate_h3_against_step3a(
    h3_comparison: pd.DataFrame,
    step3a_comparison: pd.DataFrame,
) -> None:
    """Require alpha=1 selections and risks to reproduce existing Step 3A."""
    keys = ["Beta_p", "Task_ID", "Primary_Server"]
    required = set(keys).union({
        "Independent_Selected_Backup",
        "Correlation_Aware_Selected_Backup",
        "Independent_Actual_Joint_Risk",
        "Correlation_Aware_Actual_Joint_Risk",
    })
    missing = sorted(required.difference(step3a_comparison.columns))
    if missing:
        raise ValueError(
            "backup_selection_comparison.csv is missing required columns: "
            + ", ".join(missing)
        )

    h3 = h3_comparison[list(required)].copy()
    existing = step3a_comparison[list(required)].copy()
    merged = h3.merge(
        existing,
        on=keys,
        how="outer",
        suffixes=("_H3", "_Step3A"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("H3 and Step 3A contain different cases")
    for column in (
        "Independent_Selected_Backup",
        "Correlation_Aware_Selected_Backup",
    ):
        if not (
            merged[f"{column}_H3"] == merged[f"{column}_Step3A"]
        ).all():
            raise RuntimeError(f"H3 does not reproduce Step 3A {column}")
    for column in (
        "Independent_Actual_Joint_Risk",
        "Correlation_Aware_Actual_Joint_Risk",
    ):
        if not np.allclose(
            merged[f"{column}_H3"],
            merged[f"{column}_Step3A"],
            rtol=1e-12,
            atol=1e-15,
        ):
            raise RuntimeError(f"H3 does not reproduce Step 3A {column}")


def _load_server_parameters(server_path: Path | str) -> pd.DataFrame:
    path = Path(server_path)
    if not path.is_file():
        raise FileNotFoundError(f"server parameter file not found: {path}")
    frame = pd.read_excel(path)
    required = {"Server_ID", "Failure_Rate"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "server_info.xlsx is missing required columns: "
            + ", ".join(missing)
        )
    if frame["Server_ID"].duplicated().any():
        raise ValueError("Server_ID must be unique")
    _validate_original_rates(frame["Failure_Rate"].to_numpy(dtype=float))
    return frame


def _first_reversal_scenario(summary: pd.DataFrame):
    reversal_rows = summary[summary["Reversal_Count"] > 0].copy()
    if reversal_rows.empty:
        return None
    return reversal_rows.sort_values(
        ["Alpha", "Beta_p"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]


def _print_results(
    comparison: pd.DataFrame,
    ranking: pd.DataFrame,
    summary: pd.DataFrame,
    output_paths: tuple[Path, Path, Path],
) -> None:
    print(
        "alpha | scenario | beta | lambda CV | selection change | "
        "mean relative reduction | max relative reduction | "
        "max reversal margin | reversal rate"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.Alpha:.6f} | {row.Heterogeneity_Scenario} | "
            f"{row.Beta_p:.1f} | {row.Lambda_CV:.6f} | "
            f"{row.Selection_Change_Rate:.6f} | "
            f"{row.Mean_Relative_Risk_Reduction:.6f} | "
            f"{row.Max_Relative_Risk_Reduction:.6f} | "
            f"{row.Max_Reversal_Margin:.6f} | "
            f"{row.Reversal_Rate:.6f}"
        )

    first = _first_reversal_scenario(summary)
    if first is None:
        print("\nNo actual reversal occurred in any alpha-beta scenario.")
    else:
        alpha = float(first["Alpha"])
        beta_p = float(first["Beta_p"])
        scenario = str(first["Heterogeneity_Scenario"])
        selected = comparison[
            (comparison["Alpha"] == alpha)
            & (comparison["Beta_p"] == beta_p)
            & comparison["Selection_Changed"]
        ].copy()
        ranked = ranking[
            (ranking["Alpha"] == alpha)
            & (ranking["Beta_p"] == beta_p)
        ][[
            "Task_ID",
            "Primary_Server",
            "Best_Marginal",
            "Best_Rho",
            "Best_Amplification",
            "Reversal_Margin",
        ]]
        selected = selected.merge(
            ranked,
            on=["Task_ID", "Primary_Server"],
            how="left",
            validate="one_to_one",
        )
        alternative_independent_risk = (
            selected["Primary_Marginal_Failure"]
            * selected["Correlation_Aware_Backup_Marginal_Failure"]
        )
        selected["Alternative_Amplification"] = np.divide(
            selected["Correlation_Aware_Actual_Joint_Risk"],
            alternative_independent_risk,
            out=np.full(len(selected), np.nan),
            where=np.abs(alternative_independent_risk) > RATIO_EPSILON,
        )
        selected = selected.rename(columns={
            "Correlation_Aware_Backup_Marginal_Failure": (
                "Alternative_Marginal"
            ),
            "Correlation_Aware_Selected_Rho_phy": "Alternative_Rho",
        })
        columns = [
            "Task_ID",
            "Primary_Server",
            "Independent_Selected_Backup",
            "Correlation_Aware_Selected_Backup",
            "Best_Marginal",
            "Alternative_Marginal",
            "Best_Rho",
            "Alternative_Rho",
            "Best_Amplification",
            "Alternative_Amplification",
            "Reversal_Margin",
            "Independent_Actual_Joint_Risk",
            "Correlation_Aware_Actual_Joint_Risk",
            "Relative_Risk_Reduction",
        ]
        top_cases = selected.sort_values(
            ["Relative_Risk_Reduction", "Task_ID", "Primary_Server"],
            ascending=[False, True, True],
            kind="mergesort",
        ).head(10)
        print(
            f"\nFirst reversal while reducing heterogeneity: "
            f"{scenario}, alpha={alpha:.6f}, beta={beta_p:.1f}"
        )
        print(
            top_cases[columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.10g}",
            )
        )

    comparison_path, ranking_path, summary_path = output_paths
    print(f"Comparison output: {comparison_path}")
    print(f"Ranking output: {ranking_path}")
    print(f"Summary output: {summary_path}")


def run_diagnostic(
    server_path: Path | str = DEFAULT_SERVER_PATH,
    task_path: Path | str = DEFAULT_TASK_PATH,
    step3a_path: Path | str = DEFAULT_STEP3A_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    correlation_length_km: float = DEFAULT_CORRELATION_LENGTH_KM,
):
    """Run H0-H3 through the existing offline dependence diagnostics."""
    server_frame = _load_server_parameters(server_path)
    server_ids = server_frame["Server_ID"].to_numpy(dtype=int)
    original_rates = server_frame["Failure_Rate"].to_numpy(dtype=float)
    failure_rate_table = build_failure_rate_scenarios(
        server_ids,
        original_rates,
    )

    comparisons = []
    rankings = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        for alpha, _ in HETEROGENEITY_SCENARIOS:
            (
                scenario,
                rate_by_server,
                lambda_mean,
                lambda_std,
                lambda_cv,
            ) = _scenario_metadata(failure_rate_table, alpha)

            scenario_servers = server_frame.copy()
            scenario_servers["Failure_Rate"] = scenario_servers[
                "Server_ID"
            ].map(rate_by_server)
            scenario_server_path = (
                temporary_root / f"{scenario}_server_info.xlsx"
            )
            scenario_output_dir = temporary_root / scenario
            scenario_servers.to_excel(scenario_server_path, index=False)

            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
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

            base_comparison = compare_backup_selections(pair_detail)
            base_ranking = rank_backup_candidates(pair_detail)
            comparisons.append(decorate_selection_comparison(
                base_comparison,
                alpha,
                scenario,
                rate_by_server,
                lambda_mean,
                lambda_std,
                lambda_cv,
            ))
            rankings.append(decorate_ranking_competition(
                base_ranking,
                base_comparison,
                alpha,
                scenario,
            ))

    comparison = pd.concat(comparisons, ignore_index=True)
    ranking = pd.concat(rankings, ignore_index=True)
    summary = summarize_sensitivity(comparison, ranking)
    validate_beta_zero_controls(comparison, ranking)

    h3_comparison = comparison[comparison["Alpha"] == 1.0]
    validate_h3_against_step3a(
        h3_comparison,
        pd.read_csv(step3a_path),
    )
    if not (
        summary.loc[summary["Alpha"] == 1.0, "Selection_Change_Count"] == 0
    ).all():
        raise RuntimeError("H3 must reproduce zero Step 3A selection changes")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison_path = (
        output_directory / "heterogeneity_backup_selection_comparison.csv"
    )
    ranking_path = (
        output_directory / "heterogeneity_ranking_competition.csv"
    )
    summary_path = (
        output_directory / "heterogeneity_sensitivity_summary.csv"
    )
    failure_rate_path = (
        output_directory / "heterogeneity_failure_rates.csv"
    )
    comparison.to_csv(comparison_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    summary.to_csv(summary_path, index=False)
    failure_rate_table.to_csv(failure_rate_path, index=False)
    _print_results(
        comparison,
        ranking,
        summary,
        (comparison_path, ranking_path, summary_path),
    )
    print(f"Failure-rate parameters: {failure_rate_path}")
    return comparison, ranking, summary, failure_rate_table


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-input",
        type=Path,
        default=DEFAULT_SERVER_PATH,
    )
    parser.add_argument(
        "--task-input",
        type=Path,
        default=DEFAULT_TASK_PATH,
    )
    parser.add_argument(
        "--step3a-input",
        type=Path,
        default=DEFAULT_STEP3A_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_diagnostic(
        server_path=args.server_input,
        task_path=args.task_input,
        step3a_path=args.step3a_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
