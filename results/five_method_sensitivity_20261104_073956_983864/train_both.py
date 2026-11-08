"""Train both PPO variants once on the new baseline using isolated processes."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import torch
import Project_main
from config.params import params
from tools.run_experiments import temporary_params


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_one(method):
    result_path = OUT / f"training_{method}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert digest(result["checkpoint"]) == result["checkpoint_sha256"]
        return result
    torch.set_num_threads(1)
    logger = importlib.import_module("io_utils.long_run_logger")
    writer = importlib.import_module("io_utils.save_parameters_and_logs")
    with (OUT / f"training_{method}.log").open("x", buffering=1) as log, \
            redirect_stdout(log), redirect_stderr(log), \
            temporary_params(model_summary=method, EXPERIMENT_TAG="profile2029_rel092_099"), \
            patch.object(logger, "RESULTS_DIR", OUT / "training"), \
            patch.object(writer, "RESULTS_DIR", OUT / "training"), \
            patch.object(Project_main, "RESULTS_DIR", OUT / "checkpoints"):
        assert params.total_episodes == 8000 and params.taskno == 200
        assert params.TASK_ARRIVAL_RATE == 2.0 and params.EXPERIMENT_SEED == 2026
        checkpoint = Project_main.run_simulation(device="cpu")
        training_dir = OUT / "training" / checkpoint.stem
        status = json.loads((training_dir / "status.json").read_text())
        assert status["status"] == "completed" and status["last_completed_episode"] == 8000
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for network in ("policy_net", "policy_old", "value_net"):
            assert all(torch.isfinite(value).all() for value in state[network].values())
        for optimizer in ("opt_policy", "opt_value"):
            assert {int(value["step"]) for value in state[optimizer]["state"].values()} == {64000}
        result = dict(method=method, status="completed", training_episodes=8000, training_tasks=1600000,
                      checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                      training_directory=str(training_dir), elapsed_seconds=status["elapsed_seconds"],
                      from_scratch=True, actor_steps=64000, critic_steps=64000)
        with result_path.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        return result


def main():
    for name in ("training", "checkpoints"):
        (OUT / name).mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(train_one, method) for method in ("ppo", "ind-ppo")]
        results = [future.result() for future in as_completed(futures)]
    with (OUT / "training_registry.json").open("x", encoding="utf-8") as output:
        json.dump({row["method"]: row for row in results}, output, indent=2)
    print("Both PPO variants completed 8000 episodes.", flush=True)


if __name__ == "__main__":
    main()
