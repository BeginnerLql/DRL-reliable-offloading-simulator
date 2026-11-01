"""Evaluate the existing arrival-2 checkpoint at lower arrival rates."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from results.run_arrival2_8000_eval500 import (
    assert_identical, build_model, frozen_evaluation, params,
    LongRunLogger, save_params_and_logs, EVALUATION_TASK_PATH,
)

OUTPUT = ROOT / "results"
CHECKPOINT = OUTPUT / "ppo_20261031_180331_baseline.pt"
STAMP = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
STATUS = OUTPUT / f"frozen_arrival_sweep_{STAMP}.json"
state = {"phase": "running", "checkpoint": str(CHECKPOINT), "evaluation_seed": 2027,
         "episodes_per_rate": 500, "summaries": []}


def save_status():
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def main():
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    saved_state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    save_status()
    for rate in (0.5, 1.0):
        np.random.seed(2027)
        torch.manual_seed(2027)
        model = build_model()
        model.load_model(CHECKPOINT)
        settings = {key: value for key, value in vars(params).items() if not key.startswith("_")}
        settings.update(total_episodes=500, TASK_ARRIVAL_RATE=rate, EXPERIMENT_SEED=2027,
                        EXPERIMENT_TAG=f"arrival{rate:g}_frozen_eval500", EVALUATION_ONLY=True,
                        SOURCE_CHECKPOINT=str(CHECKPOINT), SOURCE_TRAINING_ARRIVAL_RATE=2.0,
                        ACTION_SELECTION="sample masked categorical policy")
        eval_params = SimpleNamespace(**settings)
        run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        with LongRunLogger(eval_params, model, run_id) as logger:
            state.update(current_arrival_rate=rate, current_directory=str(logger.run_dir))
            save_status()
            (logger.run_dir / "configuration.py").rename(logger.run_dir / "source_training_configuration.py")
            shutil.copy2(EVALUATION_TASK_PATH, logger.run_dir / "task_parameters.xlsx")
            shutil.copy2(EVALUATION_TASK_PATH, logger.run_dir / Path(EVALUATION_TASK_PATH).name)
            # Apply the requested arrival rate only inside this evaluation.
            with patch.object(params, "TASK_ARRIVAL_RATE", rate):
                loop = frozen_evaluation(model, 500, logger)
        assert_identical(saved_state, torch.load(logger.run_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=True))
        save_params_and_logs(eval_params, loop.log_data, loop.task_results, run_id=run_id,
                             task_parameters_path=EVALUATION_TASK_PATH,
                             task_results_csv_path=logger.task_results_csv_path)
        tasks = pd.read_csv(logger.task_results_csv_path)
        logs = pd.read_csv(logger.run_dir / "logs.csv")
        assert len(tasks) == 100000 and len(logs) == 500
        met = tasks.requirement_satisfied_correlated.astype(bool)
        summary = {
            "arrival_rate": rate, "episodes": 500, "task_executions": len(tasks),
            "mean_delay_seconds": float(tasks.task_delay.mean()),
            "mean_delay_met_seconds": float(tasks.loc[met, "task_delay"].mean()),
            "mean_delay_unmet_seconds": float(tasks.loc[~met, "task_delay"].mean()),
            "mean_upload_delay_seconds": float(tasks.task_upload_delay.mean()),
            "mean_queue_delay_seconds": float(tasks.task_queue_delay.mean()),
            "mean_execution_delay_seconds": float(tasks.task_execution_delay.mean()),
            "p95_task_delay_seconds": float(tasks.task_delay.quantile(.95)),
            "rsr_correlated": float(met.mean()),
            "mask_fallback_rate": float(tasks.mask_fallback_used.mean()),
            "weights_and_optimizer_unchanged": True, "source_checkpoint_sha256": digest,
            "directory": str(logger.run_dir), "workbook": str(logger.run_dir) + ".xlsx",
        }
        (logger.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state["summaries"].append(summary)
        save_status()
        print(json.dumps(summary, indent=2), flush=True)
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == digest
    state["phase"] = "completed"
    save_status()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        state.update(phase="failed", error=repr(error))
        save_status()
        raise
