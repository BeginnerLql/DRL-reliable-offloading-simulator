"""One-off experiment orchestration; reuse the existing training and logging code."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from Project_main import build_model, run_simulation
from config.params import params
from config.paths import EVALUATION_TASK_PATH, RESULTS_DIR
from core.main_loop import MainLoop
from io_utils.long_run_logger import LongRunLogger
from io_utils.save_parameters_and_logs import save_params_and_logs

OUTPUT = Path(RESULTS_DIR)
STAMP = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
STATUS = OUTPUT / f"arrival2_8000_eval500_{STAMP}.json"
state = {"pid": os.getpid(), "training_episodes": 8000, "evaluation_episodes": 500,
         "arrival_rate": 2.0, "evaluation_seed": 2027}


def write_status(phase, **values):
    state.update(phase=phase, **values)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def model_state(model):
    return {
        "policy": model.policy_net.state_dict(),
        "sampling_policy": model.policy_old.state_dict(),
        "value": model.value_net.state_dict(),
        "actor_optimizer": model.optimizer_policy.state_dict(),
        "critic_optimizer": model.optimizer_value.state_dict(),
    }


def assert_identical(before, after):
    if isinstance(before, torch.Tensor):
        assert torch.equal(before, after), "Frozen model/optimizer tensor changed"
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            assert_identical(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after)
        for left, right in zip(before, after):
            assert_identical(left, right)
    else:
        assert before == after


def frozen_evaluation(model, episodes, callback=None):
    for network in (model.policy_net, model.policy_old, model.value_net):
        network.eval()
        network.requires_grad_(False)
    before = copy.deepcopy(model_state(model))
    loop = MainLoop(
        model, episodes, params.taskno, params.num_states, params.num_actions,
        task_parameters_path=EVALUATION_TASK_PATH, episode_end_callback=callback,
    )
    loop.arrival_rng = np.random.default_rng(2027)
    # MainLoop already calls train_step once per episode. During evaluation,
    # replace that call with buffer cleanup, without changing simulator code.
    with patch.object(model, "train_step", side_effect=model.clear_rollout) as disabled_update:
        with torch.inference_mode():
            loop.EP()
        assert disabled_update.call_count == episodes
    assert_identical(before, model_state(model))
    assert not model.states
    return loop


def main():
    assert params.model_summary == "ppo"
    assert params.total_episodes == 8000 and params.TASK_ARRIVAL_RATE == 2.0
    write_status("training")
    existing = set(OUTPUT.glob("ppo_*.pt"))
    run_simulation()
    checkpoints = set(OUTPUT.glob("ppo_*.pt")) - existing
    assert len(checkpoints) == 1, f"Expected one new final checkpoint: {checkpoints}"
    checkpoint = checkpoints.pop()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training_dir = checkpoint.with_suffix("")
    training_status = json.loads((training_dir / "status.json").read_text())
    assert training_status["last_completed_episode"] == 8000
    assert training_status["checkpoint_episode"] == 8000

    np.random.seed(2027)
    torch.manual_seed(2027)
    model = build_model()
    model.load_model(checkpoint)
    eval_params = SimpleNamespace(**{
        key: value for key, value in vars(params).items() if not key.startswith("_")
    })
    eval_params.total_episodes = 500
    eval_params.EXPERIMENT_SEED = 2027
    eval_params.EXPERIMENT_TAG = "arrival2_frozen_eval500"
    eval_params.EVALUATION_ONLY = True
    eval_params.SOURCE_CHECKPOINT = str(checkpoint)
    eval_params.ACTION_SELECTION = "sample masked categorical policy"
    eval_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    with LongRunLogger(eval_params, model, eval_id) as logger:
        write_status("evaluating", training_checkpoint=str(checkpoint),
                     training_directory=str(training_dir), evaluation_directory=str(logger.run_dir))
        # The existing logger copies the default training profile. Replace that
        # snapshot with the evaluation input actually used by MainLoop.
        shutil.copy2(EVALUATION_TASK_PATH, logger.run_dir / "task_parameters.xlsx")
        shutil.copy2(EVALUATION_TASK_PATH, logger.run_dir / Path(EVALUATION_TASK_PATH).name)
        loop = frozen_evaluation(model, 500, logger)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    save_params_and_logs(
        eval_params, loop.log_data, loop.task_results, run_id=eval_id,
        task_parameters_path=EVALUATION_TASK_PATH,
        task_results_csv_path=logger.task_results_csv_path,
    )
    logs = pd.read_csv(logger.run_dir / "logs.csv")
    tasks = pd.read_csv(logger.task_results_csv_path)
    assert len(logs) == 500 and len(tasks) == 100000
    delay = logs["task_Avg_Delay"]
    summary = {
        "evaluation_episodes": len(logs), "evaluation_tasks": len(tasks),
        "mean_delay_seconds": float(delay.mean()),
        "episode_mean_delay_std_seconds": float(delay.std(ddof=1)),
        "mean_upload_delay_seconds": float(tasks.task_upload_delay.mean()),
        "mean_queue_delay_seconds": float(tasks.task_queue_delay.mean()),
        "mean_execution_delay_seconds": float(tasks.task_execution_delay.mean()),
        "rsr_correlated": float(logs.RSR_Correlated.mean()),
        "rsr_independent": float(logs.RSR_Independent.mean()),
        "mask_fallback_rate": float(logs.Mask_Fallback_Rate.mean()),
        "weights_and_optimizer_unchanged": True,
        "source_checkpoint_sha256": digest,
    }
    (logger.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_status("completed", summary=summary)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write_status("failed", error=repr(error))
        raise
