"""Plot verified sweep summaries without changing experiment data."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/msyh.ttc")
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname="/mnt/c/Windows/Fonts/msyh.ttc").get_name(),
    "font.size": 11, "axes.unicode_minus": False, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlepad": 14, "axes.labelpad": 8,
})
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(exist_ok=True)
METHODS = ("Corr-PPO", "Ind-PPO", "Corr-delayGreedy", "Indep-delayGreedy")
COLORS = ("#0072B2", "#009E73", "#D55E00", "#AA4499")
MARKERS = ("o", "^", "s", "D")


def draw(axis, metric):
    file = "mean_delay_comparison.csv" if metric == "delay" else "rsr_correlated_comparison.csv"
    frame = pd.read_csv(ROOT / file).set_index("arrival_rate")
    for method, color, marker in zip(METHODS, COLORS, MARKERS):
        values = frame[method] if metric == "delay" else frame[method] * 100
        axis.plot(frame.index, values, label=method, color=color, marker=marker,
                  linewidth=2.1, markersize=7 if method == "Corr-PPO" else 5,
                  markerfacecolor="white", markeredgewidth=1.5,
                  linestyle="--" if method == "Corr-delayGreedy" else "-")
    axis.set_xticks(frame.index)
    axis.set_xlabel("任务到达率（任务/秒）")
    axis.grid(axis="y", color="#d1d5db", linewidth=.7, alpha=.7)
    axis.set_axisbelow(True)
    if metric == "delay":
        axis.set(title="平均完成时延", ylabel="平均任务完成时延（秒）", ylim=(0, 28))
        axis.legend(loc="upper left", frameon=False, fontsize=10)
    else:
        axis.set(title="相关可靠性要求达标率", ylabel="相关模型 RSR（%）", ylim=(60, 100))
        axis.text(2.95, 93.1, "Corr-PPO 与 Corr-delayGreedy 重合：91.50%",
                  ha="right", va="bottom", fontsize=9, color="#4b5563")
        axis.legend(loc="lower left", frameon=False, fontsize=10)


def save(figure, name):
    for extension in ("png", "pdf", "svg"):
        with (OUTPUT / f"{name}.{extension}").open("xb") as output:
            figure.savefig(output, format=extension, dpi=220, bbox_inches="tight", facecolor="white")


figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained")
draw(axes[0], "delay")
draw(axes[1], "rsr")
figure.suptitle("四种方法在不同任务到达率下的表现", fontsize=15)
figure.supxlabel("原评估集 · 每个到达率 500 轮 × 200 个任务 · 到达流种子 2027", fontsize=10, color="#4b5563")
save(figure, "four_method_comparison")
plt.close(figure)

for metric, name in (("delay", "mean_delay"), ("rsr", "correlated_rsr")):
    figure, axis = plt.subplots(figsize=(8, 5.5), layout="constrained")
    draw(axis, metric)
    save(figure, name)
    plt.close(figure)
print(OUTPUT / "four_method_comparison.png")
