"""Plot the key results of the completed Step 13 PPO diagnostic run."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from analyze_spatial_four_tier_run import OUTPUT_DIR, RESULT_WORKBOOK, reconstruct_run


TIERS = (0.9, 0.99, 0.999, 0.9999)
LABELS = ("0.9", "0.99", "0.999", "0.9999")
COLORS = {
    "all": "#348A9A",
    "partial": "#E9A23B",
    "zero": "#C95758",
    "observed": "#245A81",
    "conditional": "#8D65A6",
    "ceiling": "#526771",
}


def save_figure(figure, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_feasibility_and_rsr(tasks, outcomes, safe_counts):
    requirements = tasks["Reliability_Requirement"].to_numpy(dtype=float)
    all_safe, partial, zero = [], [], []
    rsr, conditional, ceiling = [], [], []
    for tier in TIERS:
        counts = safe_counts[:, requirements == tier].ravel()
        results = outcomes.loc[outcomes["reliability_requirement"] == tier]
        met = results["requirement_satisfied"].astype(bool)
        n = len(counts)
        zero_count = int(np.count_nonzero(counts == 0))
        all_safe.append(np.count_nonzero(counts == 28) / n)
        partial.append(np.count_nonzero((counts > 0) & (counts < 28)) / n)
        zero.append(zero_count / n)
        rsr.append(float(met.mean()))
        conditional.append(float(met.sum() / (n - zero_count)))
        ceiling.append(1 - zero_count / n)

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), layout="constrained")
    figure.suptitle("Step 13 | Reliability feasibility and chosen-action success", fontsize=17, weight="bold")
    x = np.arange(len(TIERS))

    ax = axes[0]
    ax.bar(x, all_safe, color=COLORS["all"], label="All 28 safe")
    ax.bar(x, partial, bottom=all_safe, color=COLORS["partial"], label="1–27 safe")
    ax.bar(x, zero, bottom=np.array(all_safe) + partial, color=COLORS["zero"], label="0 safe")
    for index in range(len(TIERS)):
        if partial[index] > 0.04:
            ax.text(index, all_safe[index] + partial[index] / 2, f"{partial[index]:.1%}",
                    ha="center", va="center", fontsize=10, weight="bold", color="#263238")
        if all_safe[index] > 0.08:
            ax.text(index, all_safe[index] / 2, f"{all_safe[index]:.1%}",
                    ha="center", va="center", fontsize=10, weight="bold", color="white")
    ax.annotate("0-safe: 0.8%", (3, 0.996), xytext=(2.45, 1.075),
                arrowprops={"arrowstyle": "-", "color": COLORS["zero"]},
                color=COLORS["zero"], fontsize=9)
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 1.12), ylabel="Share of task–episode instances",
           xlabel="Raw reliability requirement", title="A  Feasible server-pair counts")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    ax = axes[1]
    width = 0.23
    for offset, values, color, label in (
        (-width, rsr, COLORS["observed"], "Observed RSR"),
        (0, conditional, COLORS["conditional"], "Conditional RSR"),
        (width, ceiling, COLORS["ceiling"], "Feasibility ceiling"),
    ):
        bars = ax.bar(x + offset, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.1%}",
                    ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 1.17), ylabel="Share satisfied or feasible",
           xlabel="Raw reliability requirement", title="B  PPO reliability outcomes")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    figure.text(0.5, -0.015,
                "100 episodes × 200 tasks; 5,000 outcomes per tier. Conditional RSR excludes instances with no feasible pair.",
                ha="center", fontsize=10, color="#526771")
    save_figure(figure, "STEP13_FEASIBILITY_RSR")


def plot_performance(outcomes, logs, episodes, pairs):
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10), layout="constrained")
    figure.suptitle("Step 13 | PPO performance and diagnostic relationships", fontsize=17, weight="bold")

    ax = axes[0, 0]
    latency = [outcomes.loc[outcomes["reliability_requirement"] == tier, "task_latency"].to_numpy()
               for tier in TIERS]
    boxes = ax.boxplot(latency, tick_labels=LABELS, whis=(5, 95), showfliers=False,
                       patch_artist=True, widths=0.55, medianprops={"color": "#203844", "linewidth": 1.8})
    for box in boxes["boxes"]:
        box.set(facecolor="#9FCBD1", edgecolor="#348A9A")
    ax.scatter(np.arange(1, 5), [values.mean() for values in latency],
               marker="D", s=35, color=COLORS["observed"], label="Mean", zorder=3)
    ax.set(xlabel="Raw reliability requirement", ylabel="Task latency (s)",
           title="A  Latency by tier (whiskers: P5–P95)")
    overall_latency = outcomes["task_latency"]
    ax.text(0.03, 0.96,
            f"Overall: mean {overall_latency.mean():.2f} s · "
            f"median {overall_latency.median():.2f} s · "
            f"P95 {overall_latency.quantile(0.95):.2f} s",
            transform=ax.transAxes, va="top", fontsize=9, color="#526771",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    ax.legend(loc="upper right", frameon=False)

    ax = axes[0, 1]
    reward = logs["Episode Reward"].to_numpy(dtype=float)
    index = logs["Episode"].to_numpy(dtype=int)
    ax.plot(index, reward, linewidth=1.0, alpha=0.4, color="#8899A0", label="Per episode")
    ax.plot(index, pd.Series(reward).rolling(10, min_periods=10).mean(),
            linewidth=2.2, color=COLORS["observed"], label="10-episode mean")
    ax.axhline(reward.mean(), color=COLORS["zero"], linestyle="--", linewidth=1.2,
               label=f"Overall mean {reward.mean():.0f}")
    ax.set(xlabel="Episode", ylabel="Episode Reward", title="B  Reward over the full run")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax = axes[1, 0]
    severity = episodes["Mean_Lambda_Ratio"].to_numpy(dtype=float)
    rsr = episodes["RSR"].to_numpy(dtype=float)
    ax.scatter(severity, rsr, s=29, alpha=0.75, color=COLORS["conditional"], edgecolor="white", linewidth=0.4)
    slope, intercept = np.polyfit(severity, rsr, 1)
    line_x = np.linspace(severity.min(), severity.max(), 100)
    ax.plot(line_x, slope * line_x + intercept, color=COLORS["zero"], linewidth=1.8)
    correlation = np.corrcoef(severity, rsr)[0, 1]
    ax.text(0.04, 0.08, f"Pearson r = {correlation:.3f}\n100 episode observations",
            transform=ax.transAxes, fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    ax.set(xlabel="Episode mean λ_eff / base λ", ylabel="Episode RSR",
           title="C  Spatial severity and reliability")
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax = axes[1, 1]
    shares = pairs["Share"].to_numpy(dtype=float) * 100
    ranks = np.arange(1, len(shares) + 1)
    colors = [COLORS["observed"] if rank <= 5 else "#A7B8C0" for rank in ranks]
    ax.bar(ranks, shares, color=colors, width=0.8)
    ax.axhline(100 / len(shares), color=COLORS["zero"], linestyle="--", linewidth=1.2,
               label="Uniform share (3.57%)")
    hhi = float(np.square(pairs["Share"]).sum())
    ax.text(0.97, 0.96, f"Top pair {pairs.iloc[0]['Pair']}: {shares[0]:.2f}%\nHHI = {hhi:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10)
    ax.set(xlabel="Server pair ranked by selection frequency", ylabel="Selection share (%)",
           xlim=(0, 29), title="D  Action concentration across 28 pairs")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save_figure(figure, "STEP13_PERFORMANCE")


def plot_spatial_correlation(correlation):
    figure, ax = plt.subplots(figsize=(7.5, 6.7), layout="constrained")
    image = ax.imshow(correlation, cmap="YlGnBu", vmin=0, vmax=1)
    for row in range(correlation.shape[0]):
        for col in range(correlation.shape[1]):
            value = correlation[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if value > 0.55 else "#203844")
    ax.set(xticks=np.arange(8), yticks=np.arange(8),
           xticklabels=np.arange(1, 9), yticklabels=np.arange(1, 9),
           xlabel="Server ID", ylabel="Server ID",
           title="Spatial latent-risk correlation | length 0.5 km")
    figure.colorbar(image, ax=ax, shrink=0.85, label="Correlation exp(−distance / 0.5 km)")
    save_figure(figure, "STEP13_SPATIAL_CORRELATION")


def main():
    _, tasks, _, correlation, _, _, safe_counts = reconstruct_run()
    outcomes = pd.read_excel(RESULT_WORKBOOK, sheet_name="TaskResults")
    logs = pd.read_excel(RESULT_WORKBOOK, sheet_name="Logs").sort_values("Episode")
    episodes = pd.read_csv(OUTPUT_DIR / "spatial_episode_summary.csv")
    pairs = pd.read_csv(OUTPUT_DIR / "pair_selection_summary.csv")
    plot_feasibility_and_rsr(tasks, outcomes, safe_counts)
    plot_performance(outcomes, logs, episodes, pairs)
    plot_spatial_correlation(correlation)
    print("Saved three Step 13 figures as PNG in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
