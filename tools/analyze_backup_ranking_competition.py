"""Explain competition between marginally ranked backup candidates offline.

For every beta, task, and fixed primary server, this diagnostic ranks distinct
backup candidates by marginal failure probability, inspects the best three,
and compares their spatial joint-risk amplification.  The reversal margin is

    (best amplification / second amplification)
    / (second marginal / best marginal).

A value above one means correlation amplification is strong enough for the
second-best marginal candidate to beat the best marginal candidate in spatial
joint risk.  This tool reads existing Step 2E and Step 3A outputs only.  It does
not run or modify the simulator, PPO, reward, failure model, or server model.
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
from tools.analyze_correlation_aware_backup_selection import (
    build_primary_candidates,
    load_pair_failure_data,
)


RATIO_EPSILON = 1e-15
DEFAULT_DIAGNOSTIC_DIR = Path(RESULTS_DIR) / "spatial_risk_diagnostics"
DEFAULT_PAIR_INPUT = DEFAULT_DIAGNOSTIC_DIR / "task_pair_joint_failure.csv"
DEFAULT_SELECTION_INPUT = (
    DEFAULT_DIAGNOSTIC_DIR / "backup_selection_comparison.csv"
)

DETAIL_COLUMNS = (
    "Beta_p",
    "Task_ID",
    "Primary_Server",
    "Best_Backup",
    "Second_Backup",
    "Third_Backup",
    "Best_Marginal",
    "Second_Marginal",
    "Third_Marginal",
    "Marginal_Gap_1_2",
    "Marginal_Gap_Ratio_1_2",
    "Best_Rho",
    "Second_Rho",
    "Third_Rho",
    "Best_Distance_km",
    "Second_Distance_km",
    "Third_Distance_km",
    "Best_Joint_Risk",
    "Second_Joint_Risk",
    "Third_Joint_Risk",
    "Best_Amplification",
    "Second_Amplification",
    "Third_Amplification",
    "Joint_Risk_Gap_1_2",
    "Joint_Risk_Ratio_1_2",
    "Required_Amplification_Ratio_For_Reversal",
    "Observed_Amplification_Ratio",
    "Reversal_Margin",
    "Near_Tie_5pct",
    "Near_Tie_10pct",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or abs(denominator) <= RATIO_EPSILON
    ):
        return float("nan")
    return numerator / denominator


def rank_backup_candidates(pair_frame: pd.DataFrame) -> pd.DataFrame:
    """Rank the seven distinct candidates and calculate reversal diagnostics."""
    candidates = build_primary_candidates(pair_frame)
    keys = ["Beta_p", "Task_ID", "Primary_Server"]
    candidates = candidates.sort_values(
        keys + ["Backup_Marginal_Failure", "Backup_Server"],
        kind="mergesort",
        ignore_index=True,
    )

    rows = []
    for (beta_p, task_id, primary_server), group in candidates.groupby(
        keys,
        sort=True,
    ):
        if len(group) < 3:
            raise ValueError(
                "at least three distinct backup candidates are required for "
                f"beta={beta_p}, task={task_id}, primary={primary_server}"
            )
        best, second, third = (
            group.iloc[0],
            group.iloc[1],
            group.iloc[2],
        )

        best_amplification = _safe_ratio(
            best["Joint_Failure_Spatial"],
            best["Joint_Failure_Independent"],
        )
        second_amplification = _safe_ratio(
            second["Joint_Failure_Spatial"],
            second["Joint_Failure_Independent"],
        )
        third_amplification = _safe_ratio(
            third["Joint_Failure_Spatial"],
            third["Joint_Failure_Independent"],
        )
        marginal_ratio = _safe_ratio(
            second["Backup_Marginal_Failure"],
            best["Backup_Marginal_Failure"],
        )
        joint_risk_ratio = _safe_ratio(
            second["Joint_Failure_Spatial"],
            best["Joint_Failure_Spatial"],
        )
        observed_amplification_ratio = _safe_ratio(
            best_amplification,
            second_amplification,
        )
        reversal_margin = _safe_ratio(
            observed_amplification_ratio,
            marginal_ratio,
        )

        rows.append({
            "Beta_p": beta_p,
            "Task_ID": int(task_id),
            "Primary_Server": int(primary_server),
            "Best_Backup": int(best["Backup_Server"]),
            "Second_Backup": int(second["Backup_Server"]),
            "Third_Backup": int(third["Backup_Server"]),
            "Best_Marginal": best["Backup_Marginal_Failure"],
            "Second_Marginal": second["Backup_Marginal_Failure"],
            "Third_Marginal": third["Backup_Marginal_Failure"],
            "Marginal_Gap_1_2": (
                second["Backup_Marginal_Failure"]
                - best["Backup_Marginal_Failure"]
            ),
            "Marginal_Gap_Ratio_1_2": marginal_ratio,
            "Best_Rho": best["Rho_phy"],
            "Second_Rho": second["Rho_phy"],
            "Third_Rho": third["Rho_phy"],
            "Best_Distance_km": best["Distance_km"],
            "Second_Distance_km": second["Distance_km"],
            "Third_Distance_km": third["Distance_km"],
            "Best_Joint_Risk": best["Joint_Failure_Spatial"],
            "Second_Joint_Risk": second["Joint_Failure_Spatial"],
            "Third_Joint_Risk": third["Joint_Failure_Spatial"],
            "Best_Amplification": best_amplification,
            "Second_Amplification": second_amplification,
            "Third_Amplification": third_amplification,
            "Joint_Risk_Gap_1_2": (
                second["Joint_Failure_Spatial"]
                - best["Joint_Failure_Spatial"]
            ),
            "Joint_Risk_Ratio_1_2": joint_risk_ratio,
            "Required_Amplification_Ratio_For_Reversal": marginal_ratio,
            "Observed_Amplification_Ratio": observed_amplification_ratio,
            "Reversal_Margin": reversal_margin,
            "Near_Tie_5pct": (
                math.isfinite(joint_risk_ratio)
                and joint_risk_ratio <= 1.05
            ),
            "Near_Tie_10pct": (
                math.isfinite(joint_risk_ratio)
                and joint_risk_ratio <= 1.10
            ),
        })

    ranking = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    backup_columns = ["Best_Backup", "Second_Backup", "Third_Backup"]
    for column in backup_columns:
        if (ranking[column] == ranking["Primary_Server"]).any():
            raise RuntimeError("ranked backup must differ from primary server")
    return ranking


def _finite_statistic(values: pd.Series, operation: str) -> float:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite.empty:
        return float("nan")
    return float(getattr(finite, operation)())


def summarize_ranking_competition(ranking: pd.DataFrame) -> pd.DataFrame:
    """Aggregate marginal gaps, joint-risk gaps, and reversal proximity."""
    rows = []
    for beta_p, group in ranking.groupby("Beta_p", sort=True):
        near_five_count = int(group["Near_Tie_5pct"].sum())
        near_ten_count = int(group["Near_Tie_10pct"].sum())
        rows.append({
            "Beta_p": beta_p,
            "Case_Count": int(len(group)),
            "Mean_Marginal_Gap_Ratio_1_2": _finite_statistic(
                group["Marginal_Gap_Ratio_1_2"],
                "mean",
            ),
            "Median_Marginal_Gap_Ratio_1_2": _finite_statistic(
                group["Marginal_Gap_Ratio_1_2"],
                "median",
            ),
            "Min_Marginal_Gap_Ratio_1_2": _finite_statistic(
                group["Marginal_Gap_Ratio_1_2"],
                "min",
            ),
            "Max_Marginal_Gap_Ratio_1_2": _finite_statistic(
                group["Marginal_Gap_Ratio_1_2"],
                "max",
            ),
            "Mean_Joint_Risk_Ratio_1_2": _finite_statistic(
                group["Joint_Risk_Ratio_1_2"],
                "mean",
            ),
            "Median_Joint_Risk_Ratio_1_2": _finite_statistic(
                group["Joint_Risk_Ratio_1_2"],
                "median",
            ),
            "Min_Joint_Risk_Ratio_1_2": _finite_statistic(
                group["Joint_Risk_Ratio_1_2"],
                "min",
            ),
            "Mean_Required_Amplification_Ratio": _finite_statistic(
                group["Required_Amplification_Ratio_For_Reversal"],
                "mean",
            ),
            "Mean_Observed_Amplification_Ratio": _finite_statistic(
                group["Observed_Amplification_Ratio"],
                "mean",
            ),
            "Max_Observed_Amplification_Ratio": _finite_statistic(
                group["Observed_Amplification_Ratio"],
                "max",
            ),
            "Mean_Reversal_Margin": _finite_statistic(
                group["Reversal_Margin"],
                "mean",
            ),
            "Max_Reversal_Margin": _finite_statistic(
                group["Reversal_Margin"],
                "max",
            ),
            "Near_Tie_5pct_Count": near_five_count,
            "Near_Tie_5pct_Rate": near_five_count / len(group),
            "Near_Tie_10pct_Count": near_ten_count,
            "Near_Tie_10pct_Rate": near_ten_count / len(group),
        })
    return pd.DataFrame(rows)


def cross_validate_best_backups(
    ranking: pd.DataFrame,
    comparison_path: Path | str,
) -> None:
    """Confirm marginally best backups match the Step 3A selections."""
    path = Path(comparison_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"backup-selection comparison input not found: {path}"
        )
    comparison = pd.read_csv(path)
    required = {
        "Beta_p",
        "Task_ID",
        "Primary_Server",
        "Independent_Selected_Backup",
    }
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise ValueError(
            "backup_selection_comparison.csv is missing required columns: "
            + ", ".join(missing)
        )
    keys = ["Beta_p", "Task_ID", "Primary_Server"]
    if comparison.duplicated(keys).any():
        raise ValueError(
            "backup_selection_comparison.csv contains duplicate cases"
        )

    validation = ranking[keys + ["Best_Backup"]].merge(
        comparison[keys + ["Independent_Selected_Backup"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not validation["_merge"].eq("both").all():
        raise ValueError(
            "ranking and backup-selection comparison contain different cases"
        )
    if not (
        validation["Best_Backup"]
        == validation["Independent_Selected_Backup"]
    ).all():
        raise ValueError(
            "Best_Backup does not match Independent_Selected_Backup"
        )


def _print_results(
    ranking: pd.DataFrame,
    summary: pd.DataFrame,
    detail_path: Path,
    summary_path: Path,
) -> None:
    print(
        "beta | cases | mean marginal ratio | max observed amp ratio | "
        "max reversal margin | near 5% | near 10%"
    )
    for row in summary.itertuples(index=False):
        print(
            f"{row.Beta_p:.1f} | {row.Case_Count:d} | "
            f"{row.Mean_Marginal_Gap_Ratio_1_2:.6f} | "
            f"{row.Max_Observed_Amplification_Ratio:.6f} | "
            f"{row.Max_Reversal_Margin:.6f} | "
            f"{row.Near_Tie_5pct_Rate:.6f} | "
            f"{row.Near_Tie_10pct_Rate:.6f}"
        )

    columns = [
        "Task_ID",
        "Primary_Server",
        "Best_Backup",
        "Second_Backup",
        "Best_Marginal",
        "Second_Marginal",
        "Marginal_Gap_Ratio_1_2",
        "Best_Rho",
        "Second_Rho",
        "Best_Amplification",
        "Second_Amplification",
        "Required_Amplification_Ratio_For_Reversal",
        "Observed_Amplification_Ratio",
        "Reversal_Margin",
        "Joint_Risk_Ratio_1_2",
    ]
    for beta_p in (0.2, 0.5, 0.8):
        top_cases = (
            ranking[ranking["Beta_p"] == beta_p]
            .sort_values(
                ["Reversal_Margin", "Task_ID", "Primary_Server"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .head(10)
        )
        print(f"\nTop 10 reversal margins for beta={beta_p:.1f}:")
        print(
            top_cases[columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.10g}",
            )
        )

    print(f"Ranking output: {detail_path}")
    print(f"Summary output: {summary_path}")


def run_diagnostic(
    pair_input: Path | str = DEFAULT_PAIR_INPUT,
    selection_input: Path | str | None = DEFAULT_SELECTION_INPUT,
    output_dir: Path | str = DEFAULT_DIAGNOSTIC_DIR,
):
    """Run the offline ranking-competition analysis and write both CSVs."""
    pair_frame = load_pair_failure_data(pair_input)
    ranking = rank_backup_candidates(pair_frame)
    if selection_input is not None:
        cross_validate_best_backups(ranking, selection_input)
    summary = summarize_ranking_competition(ranking)

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    detail_path = output_directory / "backup_ranking_competition.csv"
    summary_path = (
        output_directory / "backup_ranking_competition_summary.csv"
    )
    ranking.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    _print_results(ranking, summary, detail_path, summary_path)
    return ranking, summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-input",
        type=Path,
        default=DEFAULT_PAIR_INPUT,
        help="Step 2E task_pair_joint_failure.csv path",
    )
    parser.add_argument(
        "--selection-input",
        type=Path,
        default=DEFAULT_SELECTION_INPUT,
        help="Step 3A backup_selection_comparison.csv path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_DIR,
        help="directory for ranking detail and summary CSV files",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_diagnostic(
        pair_input=args.pair_input,
        selection_input=args.selection_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
