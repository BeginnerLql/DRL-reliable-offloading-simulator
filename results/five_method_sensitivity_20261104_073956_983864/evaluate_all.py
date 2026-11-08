"""Isolated frozen evaluations with shared Poisson arrival templates."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from Project_main import build_model
from config.params import params
from core.main_loop import MainLoop
from io_utils.long_run_logger import LongRunLogger
from tools.run_experiments import temporary_params, model_state, assert_identical

METHODS = ("ppo", "ind-ppo", "corr-delaygreedy", "indep-delaygreedy", "rasoul-delay")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_job(method, scenario, episodes=500, smoke=False):
    base = OUT / ("smoke" if smoke else "evaluations") / scenario["id"] / method
    base.mkdir(parents=True, exist_ok=True)
    result_path = base / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result["episodes"] == episodes and result["smoke"] == smoke
        return result
    torch.set_num_threads(1)
    is_ppo = method in ("ppo", "ind-ppo")
    source = None
    if is_ppo and not smoke:
        source = json.loads((OUT / "training_registry.json").read_text())[method]
        assert digest(source["checkpoint"]) == source["checkpoint_sha256"]
    logger_module = importlib.import_module("io_utils.long_run_logger")
    writer_module = importlib.import_module("io_utils.save_parameters_and_logs")
    core_module = importlib.import_module("core.main_loop")
    task_path = Path(scenario["task_path"])
    server_path = OUT / "inputs/server_info.xlsx"
    task_count = int(scenario["task_count"])
    rate = float(scenario["arrival_rate"])
    unit_arrivals = np.load(OUT / "inputs/evaluation_unit_interarrivals.npy")[:episodes, :task_count]
    with (base / "run.log").open("x", buffering=1) as log, redirect_stdout(log), redirect_stderr(log), \
            temporary_params(model_summary=method, total_episodes=episodes, taskno=task_count,
                             TASK_ARRIVAL_RATE=rate, EXPERIMENT_SEED=2027, EXPERIMENT_TAG=scenario["id"]), \
            patch.object(logger_module, "RESULTS_DIR", base), patch.object(writer_module, "RESULTS_DIR", base), \
            patch.object(core_module, "SERVER_INFO_PATH", server_path), \
            patch.object(logger_module, "SERVER_INFO_PATH", server_path), \
            patch.object(writer_module, "SERVER_INFO_PATH", server_path):
        np.random.seed(2027)
        torch.manual_seed(2027)
        model = build_model(device="cpu")
        settings = {key: value for key, value in vars(params).items() if not key.startswith("_")}
        settings.update(EVALUATION_ONLY=True, SMOKE_ONLY=smoke, SCENARIO=scenario,
                        TRAINING_DEMAND_NORMALIZATION_RANGE=params.TASK_DEMAND_RANGE,
                        REPLICA_COUNT=1 if method == "rasoul-delay" else 2,
                        ARRIVAL_TEMPLATE_SHA256=digest(OUT / "inputs/evaluation_unit_interarrivals.npy"))
        if source is not None:
            model.load_model(source["checkpoint"])
            settings.update(SOURCE_CHECKPOINT=source["checkpoint"], SOURCE_CHECKPOINT_SHA256=source["checkpoint_sha256"])
        frozen = None
        if is_ppo:
            for network in (model.policy_net, model.policy_old, model.value_net):
                network.eval()
                network.requires_grad_(False)
            frozen = copy.deepcopy(model_state(model))
        recorded = SimpleNamespace(**settings)
        rasoul_stats = []
        with LongRunLogger(recorded, model, "eval") as logger:
            (logger.run_dir / "configuration.py").rename(logger.run_dir / "source_configuration.py")
            (logger.run_dir / "task_parameters.xlsx").rename(logger.run_dir / "source_training_task_parameters.xlsx")
            import shutil
            shutil.copy2(task_path, logger.run_dir / "task_parameters.xlsx")

            def finish(loop):
                if method == "rasoul-delay":
                    assert sum(model.completed_counts.values()) == task_count
                    assert all(task.rasoul_feedback_processed for task in loop.env_state.tasks.values())
                    for server, count in model.completed_counts.items():
                        rasoul_stats.append(dict(episode=loop.this_episode, server_id=server,
                                                  completed_feedback=count, discrepancy_sum=model.discrepancy_sums[server],
                                                  mean_discrepancy=model.discrepancy_sums[server] / count if count else None))
                logger(loop)

            loop = MainLoop(model, episodes, task_count, params.num_states, params.num_actions,
                            task_parameters_path=task_path, episode_end_callback=finish)
            # The same exponential samples are used by all methods and nested N prefixes.
            loop._sample_interarrival_time = lambda: float(unit_arrivals[loop.this_episode - 1, loop.taskCounter - 1] / rate)
            with ExitStack() as stack:
                if is_ppo:
                    stack.enter_context(patch.object(model, "train_step", model.clear_rollout))
                    stack.enter_context(torch.inference_mode())
                loop.EP()
        assert not loop.pendingList
        if is_ppo:
            assert_identical(frozen, model_state(model))
            assert not model.states
        tasks = pd.read_csv(logger.task_results_csv_path)
        assert len(tasks) == episodes * task_count
        assert not tasks.duplicated(["episode", "task_id"]).any()
        assert tasks.groupby("episode").size().tolist() == [task_count] * episodes
        assert tasks.task_delay.ge(0).all() and np.isfinite(tasks.task_delay).all()
        np.testing.assert_allclose(tasks.arrival_time, np.cumsum(unit_arrivals / rate, axis=1).ravel(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(tasks.task_delay, tasks.task_completion_time - tasks.arrival_time)
        np.testing.assert_allclose(tasks.task_delay, tasks.task_upload_delay + tasks.task_queue_delay + tasks.task_execution_delay)
        if method == "rasoul-delay":
            assert tasks.server_B_id.isna().all() and tasks.replica_B_reliability.isna().all()
            assert tasks.pair_reliability_correlated.isna().all()
            np.testing.assert_allclose(tasks.task_reliability, tasks.replica_A_reliability)
            np.testing.assert_array_equal(tasks.requirement_satisfied, tasks.task_reliability >= tasks.reliability_requirement)
            assert sum(row["completed_feedback"] for row in rasoul_stats) == episodes * task_count
            pd.DataFrame(rasoul_stats).to_csv(base / "episode_server_learning.csv", index=False)
        else:
            assert tasks.server_A_id.ne(tasks.server_B_id).all()
            np.testing.assert_allclose(tasks.task_completion_time,
                                       np.minimum(tasks.replica_A_finish_time, tasks.replica_B_finish_time))
            np.testing.assert_allclose(tasks.pair_reliability_independent,
                                       1 - (1 - tasks.replica_A_reliability) * (1 - tasks.replica_B_reliability))
        mask_model = "independent" if method in ("ind-ppo", "indep-delaygreedy") else "correlated"
        np.testing.assert_array_equal(tasks.feasible_action_count > 0, tasks[f"requirement_satisfied_{mask_model}"])
        np.testing.assert_array_equal(tasks.mask_fallback_used, tasks.feasible_action_count == 0)
        writer_module.save_params_and_logs(recorded, loop.log_data, loop.task_results, "eval",
                                          task_parameters_path=task_path,
                                          task_results_csv_path=logger.task_results_csv_path)
        pd.DataFrame(loop.log_data, columns=LongRunLogger.LOG_COLUMNS).to_csv(base / "episode_metrics.csv", index=False)
        matched = tasks[["episode", "task_id", "task_size", "computation_demand", "reliability_requirement", "arrival_time"]]
        result = dict(method=method, scenario_id=scenario["id"], **{key: scenario[key] for key in
                      ("arrival_rate", "demand_multiplier", "task_count")}, episodes=episodes,
                      tasks=len(tasks), smoke=smoke, mean_delay_seconds=float(tasks.task_delay.mean()),
                      p95_delay_seconds=float(tasks.task_delay.quantile(.95)),
                      mean_upload_seconds=float(tasks.task_upload_delay.mean()),
                      mean_queue_seconds=float(tasks.task_queue_delay.mean()),
                      mean_execution_seconds=float(tasks.task_execution_delay.mean()),
                      success_rate=float(tasks.requirement_satisfied_correlated.mean()),
                      rsr_independent=float(tasks.requirement_satisfied_independent.mean()),
                      fallback_rate=float(tasks.mask_fallback_used.mean()),
                      paired_input_sha256=hashlib.sha256(np.ascontiguousarray(matched.to_numpy(dtype=float)).tobytes()).hexdigest(),
                      server_sha256=digest(server_path), task_sha256=digest(task_path),
                      source_checkpoint_sha256=source["checkpoint_sha256"] if source else None,
                      frozen_weights_and_optimizer_verified=is_ppo,
                      run_directory=str(logger.run_dir), task_results_sha256=digest(logger.task_results_csv_path))
        if source is not None:
            assert digest(source["checkpoint"]) == source["checkpoint_sha256"]
        with result_path.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text())
    scenarios = manifest["scenarios"]
    if args.smoke:
        scenarios = [row for row in scenarios if row["id"] in ("demand_multiplier_0.5", "task_count_300")]
    jobs = [(method, scenario) for scenario in scenarios for method in args.methods]
    done = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_job, method, scenario, 2 if args.smoke else 500, args.smoke)
                   for method, scenario in jobs]
        for future in as_completed(futures):
            result = future.result()
            done.append(result)
            print(f"{len(done)}/{len(jobs)} {result['scenario_id']} {result['method']} "
                  f"delay={result['mean_delay_seconds']:.4f} RSR={result['success_rate']:.4f}", flush=True)
    if args.smoke:
        for scenario in scenarios:
            rows = [row for row in done if row["scenario_id"] == scenario["id"]]
            assert len({row["paired_input_sha256"] for row in rows}) == 1
        with (OUT / "harness_smoke.json").open("x") as output:
            json.dump(dict(passed=True, runs=len(done), all_five_methods=True, arrival_prefix_matching=True), output, indent=2)


if __name__ == "__main__":
    main()
