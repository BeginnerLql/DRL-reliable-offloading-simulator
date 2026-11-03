"""Unified training, frozen PPO evaluation and paired greedy comparison."""

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from Project_main import build_model, run_simulation
from config.params import params
from config.paths import EVALUATION_TASK_PATH, RESULTS_DIR, SERVER_INFO_PATH
from core.main_loop import MainLoop
from io_utils.long_run_logger import LongRunLogger
from io_utils.save_parameters_and_logs import save_params_and_logs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def temporary_params(**settings):
    original = {key: getattr(params, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(params, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(params, key, value)


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


def frozen_evaluation(model, episodes, *, seed, task_path, callback=None):
    for network in (model.policy_net, model.policy_old, model.value_net):
        network.eval()
        network.requires_grad_(False)
    before = copy.deepcopy(model_state(model))
    loop = MainLoop(model, episodes, params.taskno, params.num_states, params.num_actions,
                    task_parameters_path=task_path, episode_end_callback=callback)
    loop.arrival_rng = np.random.default_rng(seed)
    # Keep simulator bookkeeping while replacing its episode-end optimizer step.
    original_train_step = model.train_step
    model.train_step = model.clear_rollout
    try:
        with torch.inference_mode():
            loop.EP()
    finally:
        model.train_step = original_train_step
    assert_identical(before, model_state(model))
    assert not model.states
    return loop


def evaluate(agent, rate, *, episodes, seed, device, checkpoint, task_path, run_id):
    task_path = Path(task_path).resolve()
    task_count = len(pd.read_excel(task_path))
    if not task_count:
        raise ValueError("Evaluation task set is empty.")
    settings = dict(model_summary=agent, total_episodes=episodes, taskno=task_count,
                    TASK_ARRIVAL_RATE=rate, EXPERIMENT_SEED=seed,
                    EXPERIMENT_TAG=f"arrival{rate:g}_eval{episodes}")
    with temporary_params(**settings):
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = build_model(device=device)
        recorded = {key: value for key, value in vars(params).items() if not key.startswith("_")}
        recorded.update(EVALUATION_ONLY=True, INFERENCE_DEVICE=device,
                        ACTION_SELECTION=("sample masked categorical policy" if agent == "ppo"
                                          else "deterministic safe-set greedy"))
        if agent == "ppo":
            checkpoint = Path(checkpoint).resolve()
            source_hash = digest(checkpoint)
            model.load_model(checkpoint)
            recorded.update(SOURCE_CHECKPOINT=str(checkpoint), SOURCE_CHECKPOINT_SHA256=source_hash)
        eval_params = SimpleNamespace(**recorded)
        with LongRunLogger(eval_params, model, run_id) as logger:
            (logger.run_dir / "configuration.py").rename(logger.run_dir / "source_configuration.py")
            shutil.copy2(task_path, logger.run_dir / "task_parameters.xlsx")
            if task_path.name != "task_parameters.xlsx":
                shutil.copy2(task_path, logger.run_dir / task_path.name)
            if agent == "ppo":
                loop = frozen_evaluation(model, episodes, seed=seed, task_path=task_path,
                                         callback=logger)
            else:
                loop = MainLoop(model, episodes, task_count, params.num_states, params.num_actions,
                                task_parameters_path=task_path, episode_end_callback=logger)
                loop.EP()
        assert not loop.pendingList
        tasks = pd.read_csv(logger.task_results_csv_path)
        assert len(tasks) == episodes * task_count and len(loop.log_data) == episodes
        save_params_and_logs(eval_params, loop.log_data, loop.task_results, run_id,
                             task_parameters_path=task_path,
                             task_results_csv_path=logger.task_results_csv_path)
        summary = {
            "arrival_rate": rate, "episodes": episodes, "task_executions": len(tasks),
            "mean_delay_seconds": float(tasks.task_delay.mean()),
            "episode_mean_delay_std_seconds": (float(tasks.groupby("episode").task_delay.mean().std())
                                                if episodes > 1 else None),
            "p95_task_delay_seconds": float(tasks.task_delay.quantile(.95)),
            "mean_upload_delay_seconds": float(tasks.task_upload_delay.mean()),
            "mean_queue_delay_seconds": float(tasks.task_queue_delay.mean()),
            "mean_execution_delay_seconds": float(tasks.task_execution_delay.mean()),
            "rsr_correlated": float(tasks.requirement_satisfied_correlated.mean()),
            "rsr_independent": float(tasks.requirement_satisfied_independent.mean()),
            "mask_fallback_rate": float(tasks.mask_fallback_used.mean()),
            "inference_device": device,
        }
        if agent == "ppo":
            assert digest(checkpoint) == source_hash
            assert_identical(torch.load(checkpoint, map_location="cpu", weights_only=True),
                             torch.load(logger.run_dir / "checkpoint_latest.pt",
                                        map_location="cpu", weights_only=True))
            summary.update(weights_and_optimizer_unchanged=True, source_checkpoint_sha256=source_hash)
        write_json(logger.run_dir / "summary.json", summary)
        return logger.run_dir


def validate_reference(directory, rate, *, episodes, seed, task_path):
    saved = json.loads((directory / "parameters.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    if status["status"] != "completed" or status["last_completed_episode"] < episodes:
        raise ValueError(f"PPO evaluation is incomplete or too short: {directory}")
    if len(pd.read_excel(task_path)) != saved["taskno"]:
        raise ValueError(f"PPO reference uses a different task count: {directory}")
    expected = dict(model_summary="ppo", TASK_ARRIVAL_RATE=rate, EXPERIMENT_SEED=seed)
    for key in ("NUM_SERVERS", "SPATIAL_RISK_ENABLED", "SPATIAL_RISK_BETA_P",
                "SPATIAL_RISK_GH_ORDER", "RELIABILITY_ACTION_MASK_ENABLED"):
        expected[key] = getattr(params, key)
    for key, value in expected.items():
        if saved[key] != value:
            raise ValueError(f"PPO reference differs in {key}: {directory}")
    for source, snapshot in ((SERVER_INFO_PATH, "server_info.xlsx"),
                             (task_path, "task_parameters.xlsx")):
        if digest(source) != digest(directory / snapshot):
            raise ValueError(f"PPO reference input differs: {source}")


def compare_runs(ppo_dir, baseline_dir, *, rate, episodes, report_dir):
    settings = json.loads((baseline_dir / "parameters.json").read_text())
    baseline = pd.read_csv(baseline_dir / "task_results.csv")
    ppo = pd.read_csv(ppo_dir / "task_results.csv", nrows=len(baseline))
    assert len(ppo) == len(baseline)
    for key in ("episode", "task_id", "task_size", "computation_demand",
                "reliability_requirement", "arrival_time"):
        np.testing.assert_allclose(baseline[key], ppo[key], rtol=1e-12, atol=1e-12)
    independent_baseline = str(settings["model_summary"]).lower() == "indep-delaygreedy"
    if not independent_baseline:
        for key in ("feasible_action_count", "mask_fallback_used", "requirement_satisfied_correlated"):
            np.testing.assert_array_equal(baseline[key], ppo[key])
    satisfied = baseline[
        "requirement_satisfied_independent" if independent_baseline else "requirement_satisfied_correlated"
    ]
    assert ((baseline.feasible_action_count > 0) == satisfied).all()
    np.testing.assert_array_equal(baseline.mask_fallback_used, baseline.feasible_action_count == 0)
    paired = pd.DataFrame({
        "baseline_mean_delay_seconds": baseline.groupby("episode").task_delay.mean(),
        "ppo_mean_delay_seconds": ppo.groupby("episode").task_delay.mean(),
    })
    assert len(paired) == episodes and paired.index.tolist() == list(range(1, episodes + 1))
    assert not paired.isna().any().any()
    paired["baseline_minus_ppo_seconds"] = paired.baseline_mean_delay_seconds - paired.ppo_mean_delay_seconds
    paired.to_csv(report_dir / f"arrival{rate:g}_paired_episodes.csv")
    baseline_mean, ppo_mean = baseline.task_delay.mean(), ppo.task_delay.mean()
    return {
        "arrival_rate": rate, "baseline_episodes": episodes, "baseline_tasks": len(baseline),
        "baseline_model": settings["model_summary"],
        "baseline_mean_delay_seconds": float(baseline_mean),
        "baseline_p95_delay_seconds": float(baseline.task_delay.quantile(.95)),
        "baseline_mean_queue_delay_seconds": float(baseline.task_queue_delay.mean()),
        "baseline_rsr_correlated": float(baseline.requirement_satisfied_correlated.mean()),
        "baseline_rsr_independent": float(baseline.requirement_satisfied_independent.mean()),
        "baseline_fallback_rate": float(baseline.mask_fallback_used.mean()),
        "ppo_mean_delay_seconds": float(ppo_mean),
        "paired_mean_baseline_minus_ppo_seconds": float(paired.baseline_minus_ppo_seconds.mean()),
        "paired_difference_std_seconds": (float(paired.baseline_minus_ppo_seconds.std())
                                          if episodes > 1 else None),
        "ppo_delay_reduction_vs_baseline_percent": float((baseline_mean - ppo_mean) / baseline_mean * 100),
        "paired_arrivals_verified": True,
        "evaluation_seed": settings["EXPERIMENT_SEED"],
        "server_sha256": digest(baseline_dir / "server_info.xlsx"),
        "evaluation_tasks_sha256": digest(baseline_dir / "task_parameters.xlsx"),
        "baseline_workbook": str(baseline_dir) + ".xlsx",
        "baseline_directory": str(baseline_dir), "ppo_source": str(ppo_dir),
    }


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train-eval", "ppo", "greedy", "compare"))
    parser.add_argument("--rates", nargs="+", type=positive_float, default=[params.TASK_ARRIVAL_RATE])
    parser.add_argument("--episodes", type=int, default=500, help="evaluation episodes per arrival rate")
    parser.add_argument("--seed", type=int, default=2027, help="evaluation seed, not a training seed")
    parser.add_argument("--checkpoint", type=Path, help="PPO checkpoint for ppo/compare")
    parser.add_argument("--ppo-runs", type=Path, help="greedy mode: JSON mapping rates to existing PPO directories")
    parser.add_argument("--greedy-agent", choices=("corr-delaygreedy", "indep-delaygreedy"),
                        default="corr-delaygreedy", help="greedy/compare baseline reliability model")
    parser.add_argument("--task-path", type=Path, default=Path(EVALUATION_TASK_PATH))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--merge", type=Path, nargs="*", default=[], help="existing comparison CSVs to merge")
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.seed < 0 or len(set(args.rates)) != len(args.rates):
        parser.error("episodes must be positive, seed nonnegative, and rates unique")
    if args.mode in ("ppo", "compare") and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("ppo/compare requires --checkpoint pointing to an existing file")
    if args.mode == "greedy" and (args.ppo_runs is None or not args.ppo_runs.is_file()):
        parser.error("greedy requires --ppo-runs pointing to an existing JSON file")
    if args.mode == "train-eval" and str(params.model_summary).lower() != "ppo":
        parser.error("train-eval requires model_summary='ppo' in configuration.py")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or auto")
    if not args.task_path.is_file():
        parser.error("evaluation task file does not exist")
    if args.mode in ("greedy", "compare") and not params.RELIABILITY_ACTION_MASK_ENABLED:
        parser.error("paired comparison requires PPO reliability masking enabled")
    if (args.mode in ("greedy", "compare") and args.greedy_agent == "corr-delaygreedy"
            and not params.SPATIAL_RISK_ENABLED):
        parser.error("Corr-delayGreedy requires SPATIAL_RISK_ENABLED=True")
    if args.merge and args.mode not in ("greedy", "compare"):
        parser.error("--merge requires greedy/compare mode")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    ppo_runs = {}
    if args.mode == "greedy":
        registry = json.loads(args.ppo_runs.read_text())
        registry = {float(rate): Path(path) for rate, path in registry.items()}
        for rate in args.rates:
            if rate not in registry:
                raise ValueError(f"No PPO reference for arrival rate {rate:g}")
            directory = registry[rate]
            if not directory.is_absolute():
                directory = ROOT / directory
            validate_reference(directory, rate, episodes=args.episodes, seed=args.seed, task_path=args.task_path)
            ppo_runs[rate] = directory
        hashes = {json.loads((path / "summary.json").read_text())["source_checkpoint_sha256"]
                  for path in ppo_runs.values()}
        if len(hashes) != 1:
            raise ValueError("PPO references come from different trained checkpoints.")
    previous = [pd.read_csv(path) for path in args.merge]
    if previous:
        rates = [rate for frame in previous for rate in frame.arrival_rate] + args.rates
        if len(set(rates)) != len(rates):
            raise ValueError("Merged comparisons contain duplicate arrival rates.")

    run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    report_dir = Path(RESULTS_DIR) / f"experiments_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", mode=args.mode, episodes_per_rate=args.episodes,
                 evaluation_seed=args.seed, rates=args.rates, device=device, ppo_runs={})
    write_json(report_dir / "arguments.json", vars(args))
    try:
        checkpoint = args.checkpoint
        if args.mode == "train-eval":
            state.update(phase="training")
            write_json(report_dir / "status.json", state)
            checkpoint = run_simulation(device=device)
        comparisons = []
        for rate in args.rates:
            if args.mode != "greedy":
                state.update(phase="ppo_evaluation", current_arrival_rate=rate)
                write_json(report_dir / "status.json", state)
                ppo_runs[rate] = evaluate("ppo", rate, episodes=args.episodes, seed=args.seed,
                                          device=device, checkpoint=checkpoint,
                                          task_path=args.task_path, run_id=run_id)
            state["ppo_runs"][rate] = str(ppo_runs[rate])
            write_json(report_dir / "ppo_runs.json", state["ppo_runs"])
            if args.mode in ("greedy", "compare"):
                state.update(phase="greedy_evaluation", current_arrival_rate=rate)
                write_json(report_dir / "status.json", state)
                baseline = evaluate(args.greedy_agent, rate, episodes=args.episodes, seed=args.seed,
                                    device=device, checkpoint=None, task_path=args.task_path, run_id=run_id)
                comparisons.append(compare_runs(ppo_runs[rate], baseline, rate=rate,
                                                episodes=args.episodes, report_dir=report_dir))
                pd.DataFrame(comparisons).to_csv(report_dir / "comparison.csv", index=False)
        if comparisons:
            combined = pd.concat(previous + [pd.DataFrame(comparisons)], ignore_index=True)
            combined.sort_values("arrival_rate").to_csv(report_dir / "comparison_all_rates.csv", index=False)
        state.update(status="completed", phase="completed")
        write_json(report_dir / "summary.json", dict(state, comparisons=comparisons))
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(report_dir / "status.json", state)
    print(f"Results: {report_dir}")
    return report_dir


if __name__ == "__main__":
    main()
