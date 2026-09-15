"""Command line entry point for PPO convergence-length diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure direct script invocation can import repository packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ppo_convergence_diagnostic import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_MASTER_SEED,
    run_convergence_diagnostic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--num-probes", type=int, default=50)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="diagnostics/ppo_convergence/checkpoint_100ep_2trial")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_convergence_diagnostic(
        num_trials=args.num_trials,
        checkpoints=args.checkpoints,
        eval_episodes=args.eval_episodes,
        tasks=args.tasks,
        num_probes=args.num_probes,
        master_seed=args.master_seed,
        bootstrap_samples=args.bootstrap_samples,
        output_dir=args.output_dir,
    )
    print(f"Output: {result['output_dir']}")
    print(f"Runs: {result['metadata']['successful_runs']}")
    print(f"Checkpoints: {args.checkpoints}")


if __name__ == "__main__":
    main()
