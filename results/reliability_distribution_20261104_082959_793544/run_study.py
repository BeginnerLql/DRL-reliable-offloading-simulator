"""Paired reliability-distribution experiments; all algorithm code stays unchanged."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from config.configuration import parameters
from config.params import params
from core.main_loop import MainLoop
from Project_main import build_model
from io_utils.long_run_logger import LongRunLogger
from tools.generate_server_and_task_parameters import generate_task_params, _profile_rng
from tools.run_experiments import temporary_params

METHODS = ("ppo", "ind-ppo", "corr-delaygreedy", "indep-delaygreedy", "rasoul-delay")
GROUPS = (
    dict(id="rel090_099", lower=.9, upper=.99, label="U[0.9, 0.99]"),
    dict(id="rel099_0999", lower=.99, upper=.999, label="U[0.99, 0.999]"),
    dict(id="rel0999_09999", lower=.999, upper=.9999, label="U[0.999, 0.9999]"),
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def verify_protected():
    for relative, expected in json.loads((OUT / "protected_repository_files.json").read_text()).items():
        assert digest(ROOT / relative) == expected, relative


def prepare():
    assert not (OUT / "manifest.json").exists(), "Do not overwrite existing experiment inputs"
    verify_protected()
    previous = ROOT / "results/five_method_sensitivity_20261104_073956_983864"
    shared = OUT / "shared_inputs"
    shared.mkdir()
    shutil.copy2(ROOT / "data/server_info.xlsx", shared / "server_info.xlsx")
    for dataset, source in (("train", "task_parameters.xlsx"), ("evaluation", "task_parameters_evaluation.xlsx")):
        shutil.copy2(ROOT / "data" / source, shared / f"baseline_{dataset}.xlsx")
    # Retain exactly the previous baseline's per-episode arrival samples.
    arrivals = np.load(previous / "inputs/evaluation_unit_interarrivals.npy")[:, :200]
    assert arrivals.shape == (500, 200)
    np.save(shared / "evaluation_unit_interarrivals.npy", arrivals)
    quantiles = {}
    row_checks = []
    for dataset, stream in (("train", 1), ("evaluation", 2)):
        quantiles[dataset] = _profile_rng(stream, 2).random(200)
        np.save(shared / f"{dataset}_reliability_quantiles.npy", quantiles[dataset])
    for group in GROUPS:
        inputs = OUT / "groups" / group["id"] / "inputs"
        inputs.mkdir(parents=True)
        shutil.copy2(shared / "server_info.xlsx", inputs / "server_info.xlsx")
        shutil.copy2(shared / "evaluation_unit_interarrivals.npy", inputs / "evaluation_unit_interarrivals.npy")
        for dataset in ("train", "evaluation"):
            target = inputs / ("train200.xlsx" if dataset == "train" else "evaluation200.xlsx")
            with patch.object(parameters, "TASK_RELIABILITY_REQUIREMENT_RANGE", (group["lower"], group["upper"])):
                generate_task_params(target, dataset=dataset)
            tasks = pd.read_excel(target)
            original = pd.read_excel(shared / f"baseline_{dataset}.xlsx")
            pd.testing.assert_frame_equal(tasks.drop(columns="Reliability_Requirement"),
                                          original.drop(columns="Reliability_Requirement"), check_exact=True)
            expected = group["lower"] + (group["upper"] - group["lower"]) * quantiles[dataset]
            np.testing.assert_allclose(tasks.Reliability_Requirement, expected, rtol=0, atol=2e-16)
            assert tasks.Reliability_Requirement.between(group["lower"], group["upper"]).all()
            assert tasks.Task_ID.tolist() == list(range(1, 201))
            for row in tasks.itertuples():
                row_checks.append(dict(group_id=group["id"], dataset=dataset, task_id=row.Task_ID,
                                       unchanged_other_columns=True, reliability_requirement=row.Reliability_Requirement))
    pd.DataFrame(row_checks).to_csv(OUT / "input_row_verification.csv", index=False)
    # Freeze the previously verified evaluation harness into this experiment.
    shutil.copy2(previous / "evaluate_all.py", OUT / "evaluation_harness.py")
    write_json(OUT / "manifest.json", dict(
        groups=list(GROUPS), methods=list(METHODS), train_episodes=8000, evaluation_episodes=500,
        task_count=200, arrival_rate=2., demand_multiplier=1., profile_seed=2029,
        training_seed=2026, evaluation_seed=2027, server_profiles_fixed=True,
        only_reliability_requirement_changes=True, shared_uniform_quantiles=True,
        training_and_evaluation_streams_independent=True, replicas={m: 1 if m == "rasoul-delay" else 2 for m in METHODS},
        arrival_protocol="Reuse previous baseline's PCG64(2027) 500x600 exponential samples, first 200 per episode / 2",
        input_sha256={str(f.relative_to(OUT)): digest(f) for f in OUT.glob("**/inputs/*") if f.is_file()},
        shared_input_sha256={str(f.relative_to(OUT)): digest(f) for f in shared.iterdir()},
        algorithm_changed=False, hyperparameters_changed=False, official_workbooks_overwritten=False))
    verify_protected()
    print("Prepared three paired distributions; 1,200 task rows verified.", flush=True)


def train_one(group, method, smoke=False):
    torch.set_num_threads(1)
    group_root = OUT / "groups" / group["id"]
    base = OUT / "smoke_training" / group["id"] if smoke else group_root
    base.mkdir(parents=True, exist_ok=True)
    result_path = base / f"training_{method}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result["smoke"] == smoke and digest(result["checkpoint"]) == result["checkpoint_sha256"]
        return result
    inputs = group_root / "inputs"
    task_path = inputs / "train200.xlsx"
    server_path = inputs / "server_info.xlsx"
    episodes = 2 if smoke else 8000
    logger_module = importlib.import_module("io_utils.long_run_logger")
    writer_module = importlib.import_module("io_utils.save_parameters_and_logs")
    core_module = importlib.import_module("core.main_loop")
    with (base / f"training_{method}.log").open("x", buffering=1) as stream, \
            redirect_stdout(stream), redirect_stderr(stream), \
            temporary_params(model_summary=method, total_episodes=episodes, taskno=200,
                             TASK_ARRIVAL_RATE=2., EXPERIMENT_SEED=2026,
                             TASK_RELIABILITY_REQUIREMENT_RANGE=(group["lower"], group["upper"]),
                             EXPERIMENT_TAG=f"{group['id']}_seed2026"), \
            patch.object(core_module, "SERVER_INFO_PATH", server_path), \
            patch.object(logger_module, "SERVER_INFO_PATH", server_path), \
            patch.object(logger_module, "TRAIN_TASK_PATH", task_path), \
            patch.object(logger_module, "RESULTS_DIR", base / "training"), \
            patch.object(writer_module, "SERVER_INFO_PATH", server_path), \
            patch.object(writer_module, "RESULTS_DIR", base / "training"):
        np.random.seed(2026)
        torch.manual_seed(2026)
        model = build_model(device="cpu")
        initial_policy_hash = hashlib.sha256(b"".join(
            value.detach().numpy().tobytes() for value in model.policy_net.state_dict().values())).hexdigest()
        settings = {k: v for k, v in vars(params).items() if not k.startswith("_")}
        settings.update(SMOKE_ONLY=smoke, GROUP=group, TRAIN_FROM_SCRATCH=True,
                        TRAINING_TASK_PATH=str(task_path), SERVER_PATH=str(server_path),
                        INITIAL_POLICY_SHA256=initial_policy_hash)
        recorded = SimpleNamespace(**settings)
        with LongRunLogger(recorded, model, "train") as logger:
            (logger.run_dir / "configuration.py").rename(logger.run_dir / "source_configuration.py")
            assert digest(logger.run_dir / "server_info.xlsx") == digest(server_path)
            # The original source file is retained; parameters.json is the active run configuration.
            shutil.copy2(task_path, logger.run_dir / "task_parameters.xlsx")
            loop = MainLoop(model, episodes, 200, params.num_states, params.num_actions,
                            task_parameters_path=task_path, episode_end_callback=logger)
            assert len(loop.task_profiles) == 200
            loaded = np.array([p["Reliability_Requirement"] for p in loop.task_profiles.values()])
            np.testing.assert_array_equal(loaded, pd.read_excel(task_path).Reliability_Requirement)
            loop.EP()
            checkpoint_dir = base / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint = checkpoint_dir / f"{logger.run_dir.name}.pt"
            model.save_model(checkpoint)
            writer_module.save_params_and_logs(recorded, loop.log_data, loop.task_results, "train",
                                              task_parameters_path=task_path,
                                              task_results_csv_path=logger.task_results_csv_path)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for network in ("policy_net", "policy_old", "value_net"):
            assert all(torch.isfinite(v).all() for v in state[network].values())
        expected_steps = 8 * episodes
        for optimizer in ("opt_policy", "opt_value"):
            assert {int(v["step"]) for v in state[optimizer]["state"].values()} == {expected_steps}
        logs = pd.read_csv(logger.run_dir / "logs.csv")
        assert len(logs) == episodes and logs.Episode.tolist() == list(range(1, episodes + 1))
        status = json.loads((logger.run_dir / "status.json").read_text())
        assert status["status"] == "completed" and status["last_completed_episode"] == episodes
        write_json(result_path, dict(group_id=group["id"], method=method, smoke=smoke,
                   training_episodes=episodes, training_tasks=episodes * 200, from_scratch=True,
                   checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                   initial_policy_sha256=initial_policy_hash, training_directory=str(logger.run_dir),
                   training_task_sha256=digest(task_path), server_sha256=digest(server_path),
                   actor_steps=expected_steps, critic_steps=expected_steps,
                   elapsed_seconds=status["elapsed_seconds"]))
    return json.loads(result_path.read_text())


def evaluate_one(group, method, smoke=False):
    group_root = OUT / "groups" / group["id"]
    spec = importlib.util.spec_from_file_location("distribution_eval_harness", OUT / "evaluation_harness.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.OUT = group_root
    scenario = dict(id="baseline", arrival_rate=2., demand_multiplier=1., task_count=200,
                    task_path=str(group_root / "inputs/evaluation200.xlsx"))
    logger_module = importlib.import_module("io_utils.long_run_logger")
    class EvaluationLogger(LongRunLogger):
        def __enter__(self):
            super().__enter__()
            shutil.copy2(group_root / "inputs/train200.xlsx", self.run_dir / "task_parameters.xlsx")
            return self

    harness.LongRunLogger = EvaluationLogger
    with temporary_params(TASK_RELIABILITY_REQUIREMENT_RANGE=(group["lower"], group["upper"])), \
            patch.object(logger_module, "TRAIN_TASK_PATH", group_root / "inputs/train200.xlsx"):
        return dict(harness.evaluate_job(method, scenario, 2 if smoke else 500, smoke), group_id=group["id"])


def smoke_test():
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn")) as pool:
        training = [pool.submit(train_one, group, method, True) for group in GROUPS for method in METHODS[:2]]
        train_results = [f.result() for f in as_completed(training)]
        testing = [pool.submit(evaluate_one, group, method, True) for group in GROUPS for method in METHODS]
        eval_results = [f.result() for f in as_completed(testing)]
    assert len({r["initial_policy_sha256"] for r in train_results}) == 1
    for group in GROUPS:
        rows = [r for r in eval_results if r["group_id"] == group["id"]]
        assert len({r["paired_input_sha256"] for r in rows}) == 1
    write_json(OUT / "smoke_validation.json", dict(passed=True, fresh_training_checks=6,
               evaluations=15, all_three_distributions=True, all_five_methods=True,
               paired_inputs_verified=True, same_initial_networks=True))
    verify_protected()
    print("Smoke passed: 6 fresh training checks and 15 evaluations.", flush=True)


def run():
    assert json.loads((OUT / "smoke_validation.json").read_text())["passed"]
    verify_protected()
    train_results, eval_results = [], []
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn")) as train_pool, \
            ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn")) as eval_pool:
        training = [train_pool.submit(train_one, group, method) for group in GROUPS for method in METHODS[:2]]
        testing = [eval_pool.submit(evaluate_one, group, method) for group in GROUPS for method in METHODS[2:]]
        for future in as_completed(training):
            result = future.result()
            train_results.append(result)
            print(f"Training {len(train_results)}/6 completed: {result['group_id']} {result['method']}", flush=True)
        for group in GROUPS:
            results = {r["method"]: r for r in train_results if r["group_id"] == group["id"]}
            assert len(results) == 2
            write_json(OUT / "groups" / group["id"] / "training_registry.json", results)
            testing += [eval_pool.submit(evaluate_one, group, method) for method in METHODS[:2]]
        write_json(OUT / "training_registry.json", train_results)
        for future in as_completed(testing):
            result = future.result()
            eval_results.append(result)
            print(f"Evaluation {len(eval_results)}/15: {result['group_id']} {result['method']} "
                  f"delay={result['mean_delay_seconds']:.4f} RSR={result['success_rate']:.4f}", flush=True)
    assert len(train_results) == 6 and len(eval_results) == 15
    verify_protected()
    write_json(OUT / "simulation_complete.json", dict(completed=True, training_runs=6,
               evaluation_runs=15, training_episodes_per_run=8000, evaluation_episodes_per_run=500,
               total_training_tasks=9600000, total_evaluation_tasks=1500000,
               repository_inputs_and_algorithm_unchanged=True))
    print("All six training runs and fifteen evaluations completed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "run"))
    action = parser.parse_args().action
    {"prepare": prepare, "smoke": smoke_test, "run": run}[action]()
