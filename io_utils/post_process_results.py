"""Summarize current result workbooks without modifying experiment files."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import RESULTS_DIR


def process_all_results(root_dir: str):
    root = Path(root_dir)
    summaries = []
    episode_tables = []
    pair_tables = []

    for path in sorted((root / "fixed_rate_results").glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        model = path.stem.removesuffix("_results")
        with pd.ExcelFile(path) as workbook:
            logs = pd.read_excel(workbook, sheet_name="Logs")
            outcomes = pd.read_excel(workbook, sheet_name="TaskResults")
            servers = pd.read_excel(workbook, sheet_name="Servers")

        summaries.append({
            "Model": model,
            "Source_File": path.name,
            "Episodes": len(logs),
            "Tasks": len(outcomes),
            "Mean_Episode_Reward": logs["Episode Reward"].mean(),
            "Mean_Episode_Latency_s": logs["task_Avg_Delay"].mean(),
            "Overall_RSR": outcomes["requirement_satisfied"].mean(),
            "Mean_Task_Latency_s": outcomes["task_latency"].mean(),
            "P95_Task_Latency_s": outcomes["task_latency"].quantile(0.95),
        })

        episode_metrics = outcomes.groupby("episode").agg(
            Tasks=("task_id", "size"),
            RSR=("requirement_satisfied", "mean"),
            P95_Task_Latency_s=("task_latency", lambda values: values.quantile(0.95)),
        )
        episodes = logs[["Episode", "Episode Reward", "task_Avg_Delay"]].merge(
            episode_metrics, left_on="Episode", right_index=True,
        ).sort_values("Episode")
        episodes.insert(0, "Model", model)
        episodes.insert(1, "Source_File", path.name)
        episode_tables.append(episodes)

        # Pair order does not distinguish the two replicas.
        selected_pairs = np.sort(
            outcomes[["server_A_id", "server_B_id"]].to_numpy(dtype=int), axis=1,
        )
        counts = pd.DataFrame(selected_pairs, columns=["server_A_id", "server_B_id"])
        counts = counts.value_counts(sort=False)
        all_pairs = pd.MultiIndex.from_tuples(
            combinations(sorted(servers["Server_ID"].astype(int)), 2),
            names=["server_A_id", "server_B_id"],
        )
        pairs = counts.reindex(all_pairs, fill_value=0).rename("Selections").reset_index()
        pairs["Share"] = pairs["Selections"] / len(outcomes)
        pairs.insert(0, "Model", model)
        pairs.insert(1, "Source_File", path.name)
        pair_tables.append(pairs)
        print(f"Read {path.name}: {len(logs)} episodes, {len(outcomes)} task outcomes")

    output_path = root / "Final_Result_All.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="ModelSummary", index=False)
        pd.concat(episode_tables, ignore_index=True).to_excel(
            writer, sheet_name="EpisodeMetrics", index=False,
        )
        pd.concat(pair_tables, ignore_index=True).to_excel(
            writer, sheet_name="PairSelection", index=False,
        )
    print(f"Saved {output_path}")


def main():
    process_all_results(RESULTS_DIR)


if __name__ == "__main__":
    main()
