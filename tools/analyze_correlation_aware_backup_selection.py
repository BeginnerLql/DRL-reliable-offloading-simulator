"""Compare independent-aware and correlation-aware backup selection offline.

For each beta, task, and fixed primary server, the independent-aware method
selects the distinct backup with the lowest marginal failure probability.  The
correlation-aware oracle selects the distinct backup with the lowest spatial
joint-failure probability.  Both selections are evaluated with the same
Joint_Failure_Spatial values, so the comparison measures selection quality
under correlated risk.

This is a mechanism-isolation diagnostic.  The correlation-aware result is an
offline oracle/reference and does not imply that PPO observes exact Monte Carlo
joint-failure probabilities.  The tool only reads Step 2E output and does not
run or modify the simulator.
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

from config.paths import RESULTS_DIR


MATCHED_BASELINE_TYPE = "matched_marginal_product"
RELATIVE_RISK_EPSILON = 1e-15
DEFAULT_INPUT_PATH = (
    Path(RESULTS_DIR)
    / "spatial_risk_diagnostics"
    / "task_pair_joint_failure.csv"
)
DEFAULT_OUTPUT_DIR = Path(RESULTS_DIR) / "spatial_risk_diagnostics"

REQUIRED_COLUMNS = (
    "Beta_p",
    "Independent_Baseline_Type",
    "Task_ID",
    "Server_J",
    "Server_K",
    "Distance_km",
    "Rho_phy",
    "Marginal_Failure_J_Spatial",
    "Marginal_Failure_K_Spatial",
    "Joint_Failure_Spatial",
    "Joint_Failure_Independent",
)

NUMERIC_COLUMNS = tuple(
    column for column in REQUIRED_COLUMNS
    if column != "Independent_Baseline_Type"
)

COMPARISON_COLUMNS = (
    "Beta_p",
    "Task_ID",
    "Primary_Server",
    "Primary_Marginal_Failure",
    "Independent_Selected_Backup",
    "Independent_Backup_Marginal_Failure",
    "Independent_Selected_Distance_km",
    "Independent_Selected_Rho_phy",
    "Independent_Assumed_Joint_Risk",
    "Independent_Actual_Joint_Risk",
    "Correlation_Aware_Selected_Backup",
    "Correlation_Aware_Backup_Marginal_Failure",
    "Correlation_Aware_Selected_Distance_km",
    "Correlation_Aware_Selected_Rho_phy",
    "Correlation_Aware_Actual_Joint_Risk",
    "Selection_Changed",
    "Absolute_Risk_Reduction",
    "Relative_Risk_Reduction",
    "Independent_Assumption_Underestimation",
    "Independent_Assumption_Underestimation_Ratio",
)


def validate_pair_failure_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the matched-marginal Step 2E input table."""
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(
            "task_pair_joint_failure.csv is missing required columns: "
            + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("task_pair_joint_failure.csv must not be empty")
    if not frame["Independent_Baseline_Type"].eq(MATCHED_BASELINE_TYPE).all():
        raise ValueError(
            "Independent_Baseline_Type must be matched_marginal_product for all rows"
        )

    validated = frame.copy()
    for column in NUMERIC_COLUMNS:
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
        if not np.isfinite(validated[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite values")

    for column in ("Task_ID", "Server_J", "Server_K"):
        values = validated[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain integer values")
        validated[column] = values.astype(int)

    if not (validated["Server_J"] < validated["Server_K"]).all():
        raise ValueError(
            "each pair must contain distinct servers ordered as Server_J < Server_K"
        )

    duplicate_keys = ["Beta_p", "Task_ID", "Server_J", "Server_K"]
    if validated.duplicated(duplicate_keys).any():
        raise ValueError(
            "task_pair_joint_failure.csv contains duplicate beta/task/server pairs"
        )

    probability_columns = (
        "Marginal_Failure_J_Spatial",
        "Marginal_Failure_K_Spatial",
        "Joint_Failure_Spatial",
        "Joint_Failure_Independent",
    )
    for column in probability_columns:
        if ((validated[column] < 0.0) | (validated[column] > 1.0)).any():
            raise ValueError(f"{column} must be within [0, 1]")

    return validated


def load_pair_failure_data(input_path: Path | str) -> pd.DataFrame:
    """Read and validate the Step 2E task-pair table."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"pair-failure diagnostic input not found: {path}")
    return validate_pair_failure_data(pd.read_csv(path))


def build_primary_candidates(pair_frame: pd.DataFrame) -> pd.DataFrame:
    """Orient every unordered pair once for each possible fixed primary."""
    frame = validate_pair_failure_data(pair_frame)

    primary_is_j = pd.DataFrame({
        "Beta_p": frame["Beta_p"],
        "Task_ID": frame["Task_ID"],
        "Primary_Server": frame["Server_J"],
        "Primary_Marginal_Failure": frame["Marginal_Failure_J_Spatial"],
        "Backup_Server": frame["Server_K"],
        "Backup_Marginal_Failure": frame["Marginal_Failure_K_Spatial"],
        "Distance_km": frame["Distance_km"],
        "Rho_phy": frame["Rho_phy"],
        "Joint_Failure_Spatial": frame["Joint_Failure_Spatial"],
        "Joint_Failure_Independent": frame["Joint_Failure_Independent"],
    })
    primary_is_k = pd.DataFrame({
        "Beta_p": frame["Beta_p"],
        "Task_ID": frame["Task_ID"],
        "Primary_Server": frame["Server_K"],
        "Primary_Marginal_Failure": frame["Marginal_Failure_K_Spatial"],
        "Backup_Server": frame["Server_J"],
        "Backup_Marginal_Failure": frame["Marginal_Failure_J_Spatial"],
        "Distance_km": frame["Distance_km"],
        "Rho_phy": frame["Rho_phy"],
        "Joint_Failure_Spatial": frame["Joint_Failure_Spatial"],
        "Joint_Failure_Independent": frame["Joint_Failure_Independent"],
    })

    candidates = pd.concat(
        [primary_is_j, primary_is_k],
        ignore_index=True,
    ).sort_values(
        ["Beta_p", "Task_ID", "Primary_Server", "Backup_Server"],
        kind="mergesort",
        ignore_index=True,
    )

    if (candidates["Primary_Server"] == candidates["Backup_Server"]).any():
        raise ValueError("backup server must differ from primary server")

    server_ids = sorted(
        set(frame["Server_J"].tolist()) | set(frame["Server_K"].tolist())
    )
    expected_servers = set(server_ids)
    for (beta_p, task_id), group in candidates.groupby(
        ["Beta_p", "Task_ID"],
        sort=False,
    ):
        if set(group["Primary_Server"]) != expected_servers:
            raise ValueError(
                f"incomplete primary-server set for beta={beta_p}, task={task_id}"
            )
        for primary_server, primary_group in group.groupby(
            "Primary_Server",
            sort=False,
        ):
            expected_backups = expected_servers.difference({primary_server})
            if (
                len(primary_group) != len(expected_backups)
                or set(primary_group["Backup_Server"]) != expected_backups
            ):
                raise ValueError(
                    "incomplete or duplicate backup candidates for "
                    f"beta={beta_p}, task={task_id}, primary={primary_server}"
                )
            if primary_group["Primary_Marginal_Failure"].nunique() != 1:
                raise ValueError(
                    "primary marginal failure must be identical across candidates "
                    f"for beta={beta_p}, task={task_id}, primary={primary_server}"
                )

    return candidates


def _safe_relative(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    numerator_values = numerator.to_numpy(dtype=float)
    denominator_values = denominator.to_numpy(dtype=float)
    result = np.full(numerator_values.shape, np.nan, dtype=float)
    valid = np.abs(denominator_values) > RELATIVE_RISK_EPSILON
    np.divide(
        numerator_values,
        denominator_values,
        out=result,
        where=valid,
    )
    return result


def _select_first(
    candidates: pd.DataFrame,
    criterion: str,
) -> pd.DataFrame:
    keys = ["Beta_p", "Task_ID", "Primary_Server"]
    return (
        candidates.sort_values(
            keys + [criterion, "Backup_Server"],
            kind="mergesort",
        )
        .drop_duplicates(keys, keep="first")
        .sort_values(keys, kind="mergesort", ignore_index=True)
    )


def compare_backup_selections(
    pair_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Select backups by each criterion and evaluate both with spatial risk."""
    candidates = build_primary_candidates(pair_frame)
    keys = ["Beta_p", "Task_ID", "Primary_Server"]

    independent = _select_first(
        candidates,
        "Backup_Marginal_Failure",
    )[[
        *keys,
        "Primary_Marginal_Failure",
        "Backup_Server",
        "Backup_Marginal_Failure",
        "Distance_km",
        "Rho_phy",
        "Joint_Failure_Spatial",
    ]].rename(columns={
        "Backup_Server": "Independent_Selected_Backup",
        "Backup_Marginal_Failure": "Independent_Backup_Marginal_Failure",
        "Distance_km": "Independent_Selected_Distance_km",
        "Rho_phy": "Independent_Selected_Rho_phy",
        "Joint_Failure_Spatial": "Independent_Actual_Joint_Risk",
    })

    correlation_aware = _select_first(
        candidates,
        "Joint_Failure_Spatial",
    )[[
        *keys,
        "Backup_Server",
        "Backup_Marginal_Failure",
        "Distance_km",
        "Rho_phy",
        "Joint_Failure_Spatial",
    ]].rename(columns={
        "Backup_Server": "Correlation_Aware_Selected_Backup",
        "Backup_Marginal_Failure": "Correlation_Aware_Backup_Marginal_Failure",
        "Distance_km": "Correlation_Aware_Selected_Distance_km",
        "Rho_phy": "Correlation_Aware_Selected_Rho_phy",
        "Joint_Failure_Spatial": "Correlation_Aware_Actual_Joint_Risk",
    })

    comparison = independent.merge(
        correlation_aware,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    comparison["Independent_Assumed_Joint_Risk"] = (
        comparison["Primary_Marginal_Failure"]
        * comparison["Independent_Backup_Marginal_Failure"]
    )
    comparison["Selection_Changed"] = (
        comparison["Independent_Selected_Backup"]
        != comparison["Correlation_Aware_Selected_Backup"]
    )
    comparison["Absolute_Risk_Reduction"] = (
        comparison["Independent_Actual_Joint_Risk"]
        - comparison["Correlation_Aware_Actual_Joint_Risk"]
    )
    comparison["Relative_Risk_Reduction"] = _safe_relative(
        comparison["Absolute_Risk_Reduction"],
        comparison["Independent_Actual_Joint_Risk"],
    )
    comparison["Independent_Assumption_Underestimation"] = (
        comparison["Independent_Actual_Joint_Risk"]
        - comparison["Independent_Assumed_Joint_Risk"]
    )
    comparison["Independent_Assumption_Underestimation_Ratio"] = (
        _safe_relative(
            comparison["Independent_Actual_Joint_Risk"],
            comparison["Independent_Assumed_Joint_Risk"],
        )
        - 1.0
    )

    if (
        comparison["Correlation_Aware_Actual_Joint_Risk"]
        > comparison["Independent_Actual_Joint_Risk"]
    ).any():
        raise RuntimeError(
            "correlation-aware selection cannot exceed independent-selection risk"
        )

    beta_zero = comparison[comparison["Beta_p"] == 0.0]
    if not beta_zero.empty:
        if beta_zero["Selection_Changed"].any():
            raise ValueError("beta=0 control changed backup selection")
        if not (beta_zero["Absolute_Risk_Reduction"] == 0.0).all():
            raise ValueError("beta=0 control must have exactly zero risk reduction")

    return comparison.loc[:, COMPARISON_COLUMNS].sort_values(
        ["Beta_p", "Task_ID", "Primary_Server"],
        kind="mergesort",
        ignore_index=True,
    )


def _finite_statistic(values: pd.Series, operation: str) -> float:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite.empty:
        return float("nan")
    return float(getattr(finite, operation)())


def summarize_backup_selections(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selection changes and risk reductions for each beta."""
    summary_rows = []
    for beta_p, group in comparison.groupby("Beta_p", sort=True):
        change_count = int(group["Selection_Changed"].sum())
        summary_rows.append({
            "Beta_p": beta_p,
            "Case_Count": int(len(group)),
            "Selection_Change_Count": change_count,
            "Selection_Change_Rate": change_count / len(group),
            "Mean_Independent_Actual_Joint_Risk": group[
                "Independent_Actual_Joint_Risk"
            ].mean(),
            "Mean_Correlation_Aware_Actual_Joint_Risk": group[
                "Correlation_Aware_Actual_Joint_Risk"
            ].mean(),
            "Mean_Absolute_Risk_Reduction": group[
                "Absolute_Risk_Reduction"
            ].mean(),
            "Median_Absolute_Risk_Reduction": group[
                "Absolute_Risk_Reduction"
            ].median(),
            "Max_Absolute_Risk_Reduction": group[
                "Absolute_Risk_Reduction"
            ].max(),
            "Mean_Relative_Risk_Reduction": _finite_statistic(
                group["Relative_Risk_Reduction"],
                "mean",
            ),
            "Median_Relative_Risk_Reduction": _finite_statistic(
                group["Relative_Risk_Reduction"],
                "median",
            ),
            "Max_Relative_Risk_Reduction": _finite_statistic(
                group["Relative_Risk_Reduction"],
                "max",
            ),
            "Mean_Independent_Selected_Rho": group[
                "Independent_Selected_Rho_phy"
            ].mean(),
            "Mean_Correlation_Aware_Selected_Rho": group[
                "Correlation_Aware_Selected_Rho_phy"
            ].mean(),
            "Mean_Independent_Selected_Distance_km": group[
                "Independent_Selected_Distance_km"
            ].mean(),
            "Mean_Correlation_Aware_Selected_Distance_km": group[
                "Correlation_Aware_Selected_Distance_km"
            ].mean(),
            "Mean_Independent_Assumption_Underestimation_Ratio": (
                _finite_statistic(
                    group["Independent_Assumption_Underestimation_Ratio"],
                    "mean",
                )
            ),
            "Fraction_Cases_With_Positive_Risk_Reduction": (
                group["Absolute_Risk_Reduction"] > 0.0
            ).mean(),
        })
    return pd.DataFrame(summary_rows)


def _print_results(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    comparison_path: Path,
    summary_path: Path,
) -> None:
    print(
        "beta | cases | change rate | mean relative reduction | "
        "max relative reduction | independent rho | correlation-aware rho | "
        "independent distance | correlation-aware distance"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.Beta_p:.1f} | {row.Case_Count:d} | "
            f"{row.Selection_Change_Rate:.6f} | "
            f"{row.Mean_Relative_Risk_Reduction:.6f} | "
            f"{row.Max_Relative_Risk_Reduction:.6f} | "
            f"{row.Mean_Independent_Selected_Rho:.6f} | "
            f"{row.Mean_Correlation_Aware_Selected_Rho:.6f} | "
            f"{row.Mean_Independent_Selected_Distance_km:.6f} | "
            f"{row.Mean_Correlation_Aware_Selected_Distance_km:.6f}"
        )

    case_columns = [
        "Task_ID",
        "Primary_Server",
        "Independent_Selected_Backup",
        "Independent_Backup_Marginal_Failure",
        "Independent_Selected_Rho_phy",
        "Independent_Actual_Joint_Risk",
        "Correlation_Aware_Selected_Backup",
        "Correlation_Aware_Backup_Marginal_Failure",
        "Correlation_Aware_Selected_Rho_phy",
        "Correlation_Aware_Actual_Joint_Risk",
        "Relative_Risk_Reduction",
    ]
    for beta_p in (0.5, 0.8):
        top_cases = (
            comparison[comparison["Beta_p"] == beta_p]
            .sort_values(
                ["Relative_Risk_Reduction", "Task_ID", "Primary_Server"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .head(10)
        )
        print(f"\nTop 10 relative-risk reductions for beta={beta_p:.1f}:")
        print(top_cases[case_columns].to_string(index=False))

    print(f"Comparison output: {comparison_path}")
    print(f"Summary output: {summary_path}")


def run_diagnostic(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
):
    """Run the offline backup-selection comparison and write both CSV files."""
    pair_frame = load_pair_failure_data(input_path)
    comparison = compare_backup_selections(pair_frame)
    summary = summarize_backup_selections(comparison)

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison_path = output_directory / "backup_selection_comparison.csv"
    summary_path = output_directory / "backup_selection_summary.csv"
    comparison.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    _print_results(comparison, summary, comparison_path, summary_path)
    return comparison, summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Step 2E task_pair_joint_failure.csv path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the comparison and summary CSV files",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_diagnostic(
        input_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
