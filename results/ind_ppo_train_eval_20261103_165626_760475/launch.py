"""Train a fresh Ind-PPO and freeze-evaluate it using original task profiles."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import Project_main
from config.params import params
from config.paths import EVALUATION_TASK_PATH, TRAIN_TASK_PATH, SERVER_INFO_PATH
from tools import run_experiments as runner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with (OUT / "execution_plan.json").open("x", encoding="utf-8") as output:
        protected = [Path(EVALUATION_TASK_PATH), Path(TRAIN_TASK_PATH), Path(SERVER_INFO_PATH),
                     ROOT / "results/ppo_20261031_180331_baseline.pt", ROOT / "config/configuration.py"]
        hashes = {str(path): digest(path) for path in protected}
        json.dump(dict(
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            method="ind-ppo", train_episodes=params.total_episodes, tasks_per_episode=params.taskno,
            train_seed=params.EXPERIMENT_SEED, train_arrival_rate=params.TASK_ARRIVAL_RATE,
            evaluation_episodes=500, evaluation_seed=2027, evaluation_arrival_rate=2.0,
            evaluation_task_path=str(EVALUATION_TASK_PATH), training_task_path=str(TRAIN_TASK_PATH),
            independent_training_from_scratch=True, prior_checkpoint_loaded=False,
            spatial_risk_enabled=params.SPATIAL_RISK_ENABLED,
            reliability_action_mask_enabled=params.RELIABILITY_ACTION_MASK_ENABLED,
            hyperparameters={key: value for key, value in vars(params).items() if key.endswith("_ppo")},
            protected_sha256=hashes, unit_tests_passed=33,
        ), output, indent=2)
    assert params.total_episodes == 8000 and params.taskno == 200
    assert params.SPATIAL_RISK_ENABLED and params.RELIABILITY_ACTION_MASK_ENABLED
    assert params.TASK_ARRIVAL_RATE == 2
    torch.set_num_threads(1)
    logger = importlib.import_module("io_utils.long_run_logger")
    writer = importlib.import_module("io_utils.save_parameters_and_logs")
    with patch.object(runner, "RESULTS_DIR", OUT), patch.object(Project_main, "RESULTS_DIR", OUT), \
            patch.object(logger, "RESULTS_DIR", OUT), patch.object(writer, "RESULTS_DIR", OUT):
        report = runner.main(["train-eval", "--ppo-agent", "ind-ppo", "--episodes", "500",
                              "--rates", "2", "--seed", "2027", "--device", "cpu",
                              "--task-path", str(EVALUATION_TASK_PATH)])
    assert all(digest(path) == value for path, value in hashes.items())
    state = json.loads((report / "status.json").read_text())
    assert state["status"] == "completed"
    with (OUT / "completed.json").open("x", encoding="utf-8") as output:
        json.dump(dict(status="completed", report_directory=str(report),
                       evaluation_directory=state["ppo_runs"]["2.0"],
                       protected_files_unchanged=True), output, indent=2)
    print("TRAINING_AND_EVALUATION_COMPLETED", OUT, flush=True)


if __name__ == "__main__":
    main()
