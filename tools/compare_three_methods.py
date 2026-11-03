"""Audit matched evaluations and compare PPO, correlated and independent greedy."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("Corr-PPO", "Corr-delayGreedy", "Indep-delayGreedy")
COMMON_SETTINGS = (
    "total_episodes", "taskno", "TASK_ARRIVAL_RATE", "EXPERIMENT_SEED",
    "NUM_SERVERS", "SPATIAL_RISK_ENABLED", "SPATIAL_RISK_BETA_P",
    "SPATIAL_RISK_GH_ORDER", "RELIABILITY_ACTION_MASK_ENABLED",
)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(indep_report, corr_comparisons, output):
    indep = pd.read_csv(indep_report / "comparison.csv").set_index("arrival_rate")
    corr = pd.concat([pd.read_csv(path) for path in corr_comparisons]).set_index("arrival_rate")
    if not indep.index.is_unique or not corr.index.is_unique:
        raise ValueError("Duplicate arrival rates in source comparisons")
    output.mkdir(parents=True, exist_ok=False)
    summaries, episode_tables, differences, audits = [], [], [], []
    for rate, entry in indep.sort_index().iterrows():
        directories = (
            resolve(entry.ppo_source), resolve(corr.loc[rate].baseline_directory),
            resolve(entry.baseline_directory),
        )
        reference_tasks, reference_settings, reference_hashes = None, None, None
        per_method = {}
        for method, directory in zip(METHODS, directories):
            settings = json.loads((directory / "parameters.json").read_text())
            status = json.loads((directory / "status.json").read_text())
            assert status["status"] == "completed", directory
            assert status["last_completed_episode"] == settings["total_episodes"]
            expected_name = "ppo" if method == "Corr-PPO" else method.lower()
            assert settings["model_summary"].strip().lower() == expected_name
            common = {key: settings[key] for key in COMMON_SETTINGS}
            assert common["TASK_ARRIVAL_RATE"] == rate
            hashes = {name: digest(directory / name)
                      for name in ("server_info.xlsx", "task_parameters.xlsx")}
            tasks = pd.read_csv(directory / "task_results.csv")
            episodes, task_count = settings["total_episodes"], settings["taskno"]
            assert len(tasks) == episodes * task_count
            assert not tasks.duplicated(["episode", "task_id"]).any()
            assert tasks.groupby("episode").size().tolist() == [task_count] * episodes
            assert sorted(tasks.episode.unique()) == list(range(1, episodes + 1))
            assert (tasks.task_delay >= 0).all()
            np.testing.assert_allclose(tasks.task_completion_time,
                                       np.minimum(tasks.replica_A_finish_time, tasks.replica_B_finish_time))
            np.testing.assert_allclose(tasks.task_delay,
                                       tasks.task_completion_time - tasks.arrival_time)
            np.testing.assert_allclose(tasks.pair_reliability_independent,
                                       1 - (1 - tasks.replica_A_reliability) * (1 - tasks.replica_B_reliability))
            for model in ("correlated", "independent"):
                np.testing.assert_array_equal(tasks[f"requirement_satisfied_{model}"],
                                              tasks[f"pair_reliability_{model}"] >= tasks.reliability_requirement)
            mask_model = "independent" if method == "Indep-delayGreedy" else "correlated"
            np.testing.assert_array_equal(tasks.feasible_action_count > 0,
                                          tasks[f"requirement_satisfied_{mask_model}"])
            np.testing.assert_array_equal(tasks.mask_fallback_used, tasks.feasible_action_count == 0)
            matched = tasks[["episode", "task_id", "task_size", "computation_demand",
                             "reliability_requirement", "arrival_time"]]
            if reference_tasks is None:
                reference_tasks, reference_settings, reference_hashes = matched, common, hashes
            else:
                assert common == reference_settings, (method, common, reference_settings)
                assert hashes == reference_hashes, (method, directory)
                np.testing.assert_allclose(matched, reference_tasks, rtol=1e-12, atol=1e-12)
            per_episode = tasks.groupby("episode").agg(
                mean_delay_seconds=("task_delay", "mean"),
                rsr_correlated=("requirement_satisfied_correlated", "mean"),
                rsr_independent=("requirement_satisfied_independent", "mean"),
                fallback_rate=("mask_fallback_used", "mean"),
            )
            per_method[method] = per_episode
            episode_tables.append(per_episode.assign(arrival_rate=rate, method=method).reset_index())
            summaries.append(dict(
                arrival_rate=rate, method=method, episodes=episodes, tasks=len(tasks),
                mean_delay_seconds=float(tasks.task_delay.mean()),
                p95_delay_seconds=float(tasks.task_delay.quantile(.95)),
                episode_mean_delay_std_seconds=float(per_episode.mean_delay_seconds.std()),
                mean_upload_delay_seconds=float(tasks.task_upload_delay.mean()),
                mean_queue_delay_seconds=float(tasks.task_queue_delay.mean()),
                mean_execution_delay_seconds=float(tasks.task_execution_delay.mean()),
                rsr_correlated=float(tasks.requirement_satisfied_correlated.mean()),
                rsr_independent=float(tasks.requirement_satisfied_independent.mean()),
                mask_fallback_rate=float(tasks.mask_fallback_used.mean()),
                mean_feasible_actions=float(tasks.feasible_action_count.mean()),
                independent_only_satisfied_rate=float((tasks.requirement_satisfied_independent
                                                       & ~tasks.requirement_satisfied_correlated).mean()),
                source_directory=str(directory),
            ))
            audits.append(dict(arrival_rate=rate, method=method, settings=common,
                               input_sha256=hashes, paired_arrivals_verified=True,
                               source_directory=str(directory),
                               task_results_sha256=digest(directory / "task_results.csv")))
        for left, right in ((METHODS[2], METHODS[0]), (METHODS[2], METHODS[1]), (METHODS[0], METHODS[1])):
            delta = per_method[left].mean_delay_seconds - per_method[right].mean_delay_seconds
            half_width = float(t.ppf(.975, len(delta) - 1) * delta.std() / np.sqrt(len(delta)))
            differences.append(dict(
                arrival_rate=rate, left_method=left, right_method=right,
                paired_episodes=len(delta), mean_delay_difference_seconds=float(delta.mean()),
                ci95_lower_seconds=float(delta.mean() - half_width),
                ci95_upper_seconds=float(delta.mean() + half_width),
                left_delay_reduction_vs_right_percent=float(
                    -delta.mean() / per_method[right].mean_delay_seconds.mean() * 100),
                correlated_rsr_difference_percentage_points=float(
                    (per_method[left].rsr_correlated - per_method[right].rsr_correlated).mean() * 100),
            ))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    wide = summary.pivot(index="arrival_rate", columns="method", values=[
        "mean_delay_seconds", "p95_delay_seconds", "rsr_correlated", "rsr_independent", "mask_fallback_rate",
    ])
    wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
    wide.to_csv(output / "comparison_wide.csv")
    pd.concat(episode_tables, ignore_index=True).to_csv(output / "paired_episode_metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "paired_differences.csv", index=False)
    (output / "audit.json").write_text(json.dumps(dict(
        runs=audits, independent_report=str(indep_report),
        reliability_metric="Fraction of tasks whose pair reliability meets its task requirement; not sampled successes.",
        ci_scope="Paired Student-t 95% intervals over episode mean delays; fixed server/task profiles and one trained PPO checkpoint.",
    ), indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
    colors, markers = ("#2563eb", "#d97706", "#15803d"), ("o", "s", "^")
    panels = (("mean_delay_seconds", "Mean task delay (s)", 1),
              ("p95_delay_seconds", "P95 task delay (s)", 1),
              ("rsr_correlated", "Reliability requirement met (%)\nEvaluated with correlated reliability", 100))
    for axis, (metric, title, scale) in zip(axes, panels):
        for method, color, marker in zip(METHODS, colors, markers):
            frame = summary[summary.method == method].sort_values("arrival_rate")
            axis.plot(frame.arrival_rate, frame[metric] * scale, label=method,
                      color=color, marker=marker, linewidth=1.8, markersize=5)
        axis.set(title=title, xlabel="Arrival rate (tasks/s)", xticks=sorted(indep.index))
        axis.grid(alpha=.2)
    axes[2].set_ylim(0, 100)
    axes[0].legend(frameon=False)
    fig.suptitle("500 episodes × 200 tasks per rate; matched arrivals, seed 2027")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    print(summary[["arrival_rate", "method", "mean_delay_seconds", "p95_delay_seconds",
                   "rsr_correlated", "rsr_independent", "mask_fallback_rate"]].to_string(index=False))
    print(f"Comparison saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indep-report", type=Path, required=True)
    parser.add_argument("--corr-comparisons", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.indep_report.resolve(), [path.resolve() for path in args.corr_comparisons], args.output.resolve())
