"""Run the ten deterministic held-out counterfactual replay trials in parallel."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

import pandas as pd
from config.params import params
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/user/anaconda3/envs/DRL-reliable-offloading-simulator/bin/python"
OUT = ROOT / "diagnostics/results/q_credit_audit"
PER_TRIAL = OUT / "counterfactual_by_trial"


def run_trial(trial_id):
    destination = PER_TRIAL / f"trial_{int(trial_id):03d}"
    destination.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, str(ROOT / "diagnostics/run_q_credit_counterfactual.py"),
               "--trial-id", str(int(trial_id)), "--output-dir", str(destination)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"trial {trial_id} failed:\n{result.stdout}\n{result.stderr}")
    return int(trial_id), result.stdout.strip()


def main():
    PER_TRIAL.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(run_trial, trial): trial for trial in range(10)}
        for future in as_completed(futures):
            trial, message = future.result()
            print(f"Completed held-out trial {trial + 1}/10\n{message}", flush=True)
            results.append(trial)
    frames = []
    boot = []
    for trial in sorted(results):
        folder = PER_TRIAL / f"trial_{trial:03d}"
        frames.append(pd.read_csv(folder / "counterfactual_reward_decomposition.csv"))
        boot.append(pd.read_csv(folder / "bootstrap_action_spread.csv"))
    combined = pd.concat(frames, ignore_index=True)
    combined_boot = pd.concat(boot, ignore_index=True)
    terminal = combined.next_effective_mask_size.eq(0)
    combined.loc[terminal, "discount_term"] = np.power(params.gamma_ppo, combined.loc[terminal, "interval_duration_seconds"].clip(lower=0.0))
    combined.loc[terminal, "bootstrap_term"] = 0.0
    combined.loc[terminal, "y_full"] = combined.loc[terminal, "interval_reward_total"]
    combined.loc[terminal, "y_current"] = combined.loc[terminal, "r_current_task"]
    combined.loc[terminal, "y_no_history"] = combined.loc[terminal, "interval_reward_total"] - combined.loc[terminal, "r_previous_tasks"]
    if combined.state_id.nunique() != 1000:
        raise RuntimeError(f"Expected 1,000 held-out states; got {combined.state_id.nunique()}")
    if combined.reward_sum_matches_existing_q_h1.astype(bool).sum() != len(combined):
        raise RuntimeError("At least one H=1 interval reward failed the source runner replay check")
    ref = pd.concat([
        pd.read_csv(ROOT / f"diagnostics/results/long_horizon_coupling/runs/trial_{trial:03d}/candidate_returns.csv")
        for trial in range(10)
    ], ignore_index=True)[["state_id", "action_index", "q_h20", "q_h50"]]
    if len(combined.merge(ref, on=["state_id", "action_index"], how="left", validate="one_to_one")) != len(combined):
        raise RuntimeError("Held-out state/action alignment with prior horizon sweep failed")
    combined.to_csv(OUT / "counterfactual_reward_decomposition.csv", index=False)
    combined_boot.to_csv(OUT / "bootstrap_action_spread.csv", index=False)
    pd.DataFrame([{
        "held_out_state_count": int(combined.state_id.nunique()),
        "state_action_branch_count": int(len(combined)),
        "q_h1_exact_replay_match_count": int(combined.reward_sum_matches_existing_q_h1.sum()),
        "q_h20_q_h50_alignment_count": int(len(combined)),
        "trials_completed": len(results), "counterfactual_targets_used_for_training": False,
    }]).to_json(OUT / "counterfactual_replay_status.json", orient="records", indent=2)
    print(f"Merged {len(combined)} branches across {combined.state_id.nunique()} held-out states.", flush=True)


if __name__ == "__main__":
    main()
