"""Command-line entry point for paired Flat/Pair PPO experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.paired_ppo_experiment import DEFAULT_MASTER_SEED, run_experiment


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--num-probes", type=int, default=100)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--eval-policy", choices=("greedy",), default="greedy",
        help="Evaluation policy. The formal runner currently uses deterministic greedy evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="diagnostics/paired_ppo_experiment/formal_10seed",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_experiment(
        num_trials=args.num_trials,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        tasks=args.tasks,
        num_probes=args.num_probes,
        master_seed=args.master_seed,
        bootstrap_samples=args.bootstrap_samples,
        output_dir=Path(args.output_dir),
    )
    print(f"paired experiment output: {result['output_dir']}")
    print(f"successful runs: {result['successful_runs']}")
    print(f"failed runs: {result['failed_runs']}")
    return 0 if result["failed_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
