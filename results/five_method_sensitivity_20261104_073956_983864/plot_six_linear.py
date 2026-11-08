"""Export six separate linear-axis figures from the original sensitivity results."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/five_method_linear_mpl")
import hashlib
import json
from pathlib import Path
import sys
import zipfile
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, AutoMinorLocator
from analyze_and_plot import METHODS, LABELS, COLORS, MARKERS, STYLES, GROUP_LABELS, setup_plots

AXES = {
    ("arrival_rate", "mean_delay_seconds"): (0, 72, list(range(0, 71, 10))),
    ("arrival_rate", "success_rate"): (60, 102, list(range(60, 101, 5))),
    ("demand_multiplier", "mean_delay_seconds"): (0, 600, list(range(0, 601, 100))),
    ("demand_multiplier", "success_rate"): (10, 102, list(range(10, 101, 10))),
    ("task_count", "mean_delay_seconds"): (0, 140, list(range(0, 141, 20))),
    ("task_count", "success_rate"): (60, 102, list(range(60, 101, 5))),
}
TITLES = {"arrival_rate": "不同任务到达率", "demand_multiplier": "不同计算需求倍率", "task_count": "不同任务数量"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protected = list(OUT.glob("*.csv")) + [OUT / "manifest.json", OUT / "training_registry.json", OUT / "validation.json"]
    protected += list((OUT / "inputs").iterdir()) + list((OUT / "checkpoints").glob("*.pt"))
    before = {str(f.relative_to(OUT)): digest(f) for f in protected if f.is_file()}
    summary = pd.read_csv(OUT / "summary.csv")
    manifest = json.loads((OUT / "manifest.json").read_text())
    baseline = summary[summary.scenario_id.eq("baseline")].set_index("method")
    np.testing.assert_allclose(baseline.loc[METHODS].mean_delay_seconds,
                               [6.637330408478415, 6.136310464102536, 14.601750459858689,
                                14.603460119822094, 56.24320047701445], rtol=1e-13)
    np.testing.assert_allclose(baseline.loc[METHODS].success_rate, [.995, .96621, .995, .97086, .655])
    setup_plots()
    directory = OUT / "figures_linear"
    directory.mkdir(exist_ok=True)
    records, outputs = [], []
    with PdfPages(directory / "six_linear_figures.pdf") as combined:
        for group, points in manifest["groups"].items():
            for metric in ("mean_delay_seconds", "success_rate"):
                fig, ax = plt.subplots(figsize=(8.6, 5.5), layout="constrained")
                lower, upper, ticks = AXES[group, metric]
                x = np.array([p["value"] for p in points])
                factor = 100 if metric == "success_rate" else 1
                for method in METHODS:
                    frame = summary[summary.method.eq(method)].set_index("scenario_id")
                    values = np.array([frame.loc[p["scenario_id"], metric] for p in points]) * factor
                    assert ((values >= lower) & (values <= upper)).all(), (group, metric, method)
                    line, = ax.plot(x, values, color=COLORS[method], marker=MARKERS[method],
                                    linestyle=STYLES[method], linewidth=1.8, label=LABELS[method],
                                    markersize=7 if method == "corr-delaygreedy" else 5,
                                    markerfacecolor="white" if method in METHODS[:2] else COLORS[method],
                                    zorder=5 if method in METHODS[:2] else 3)
                    np.testing.assert_array_equal(line.get_xdata(), x)
                    np.testing.assert_array_equal(line.get_ydata(), values)
                ax.set_xscale("linear")
                ax.set_yscale("linear")
                ax.set_ylim(lower, upper)
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.set_xticks(x)
                ax.set_xlabel(GROUP_LABELS[group])
                ax.set_ylabel("平均时延（秒）" if metric == "mean_delay_seconds" else "任务成功率（可靠性满足率，%）")
                ax.set_title(TITLES[group] + "下的" + ("平均时延" if metric == "mean_delay_seconds" else "任务成功率"), fontsize=13)
                ax.grid(which="major", alpha=.22)
                location = ("upper left" if metric == "mean_delay_seconds" else
                            "lower left" if group == "demand_multiplier" else "center right")
                ax.legend(loc=location, ncols=2, fontsize=9, framealpha=.92)
                assert ax.get_yscale() == ax.get_xscale() == "linear"
                assert len(ax.lines) == 5
                stem = f"{group}_{metric}"
                for extension in ("png", "pdf", "svg"):
                    target = directory / f"{stem}.{extension}"
                    fig.savefig(target, dpi=240)
                    outputs.append(target)
                combined.savefig(fig)
                plt.close(fig)
                records.append(dict(group=group, metric=metric, xscale="linear", yscale="linear",
                                    ymin=lower, ymax=upper, yticks=ticks, methods=5,
                                    all_points_visible=True, filename=stem))
    assert before == {relative: digest(OUT / relative) for relative in before}
    with zipfile.ZipFile(directory / "six_linear_figures.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for target in outputs:
            archive.write(target, target.name)
    (directory / "validation.json").write_text(json.dumps(dict(
        source_experiment="five_method_sensitivity_20261104_073956_983864",
        reliability_requirement_range=[.92, .999], figure_count=6, all_axes_linear=True,
        original_results_and_checkpoints_unchanged=True, source_sha256=before,
        figures=records), indent=2), encoding="utf-8")
    (directory / "README.md").write_text(
        "# 原敏感性实验的六张线性坐标图\n\n"
        "数据源为上一级 summary.csv，可靠性要求 U[0.92, 0.999]，每点评估 500 轮，两个 PPO 各训练 8000 轮。\n\n"
        "到达率、计算需求倍率、任务数量三组各输出平均时延和任务成功率，全部使用线性坐标。"
        "双副本成功率沿用相关可靠性口径，RASOUL 使用单副本 GH 边际可靠性。"
        "纵轴范围保留全部五种方法的所有数据点，时延从 0 起，成功率按实际范围收窄。\n\n"
        "每图均有 PNG、PDF、SVG；six_linear_figures.pdf 包含六个独立页面，"
        "six_linear_figures.zip 打包六图的三种格式。\n", encoding="utf-8")
    print("Exported six linear-axis figures; original data and checkpoint hashes unchanged.")


if __name__ == "__main__":
    main()
