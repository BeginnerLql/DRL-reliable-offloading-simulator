import copy
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import Project_main
from config.params import params
from config.paths import EVALUATION_TASK_PATH
from tools import run_experiments as runner


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))
        logger = importlib.import_module("io_utils.long_run_logger")
        writer = importlib.import_module("io_utils.save_parameters_and_logs")
        for module in (runner, Project_main, logger, writer):
            self.stack.enter_context(patch.object(module, "RESULTS_DIR", self.root))
        torch.set_num_threads(1)
        self.tasks = self.root / "evaluation.xlsx"
        frame = pd.read_excel(EVALUATION_TASK_PATH).head(12)
        frame.loc[0, "Reliability_Requirement"] = 0.999999
        frame.to_excel(self.tasks, index=False)
        self.checkpoint = self.root / "source.pt"
        with runner.temporary_params(model_summary="ppo"):
            Project_main.build_model(device="cpu").save_model(self.checkpoint)

    def args(self, mode, *extra):
        return [mode, "--episodes", "2", "--rates", "1.5", "--seed", "123", "--task-path", str(self.tasks),
                "--device", "cpu", *extra]

    def test_compare_and_reuse_only_greedy_preserve_checkpoint_and_pairing(self):
        checkpoint_hash = runner.digest(self.checkpoint)
        original_rate, original_seed = params.TASK_ARRIVAL_RATE, params.EXPERIMENT_SEED
        report = runner.main(self.args("compare", "--checkpoint", str(self.checkpoint)))
        comparison = pd.read_csv(report / "comparison.csv")
        self.assertEqual(comparison.baseline_tasks.tolist(), [24])
        self.assertTrue(comparison.paired_arrivals_verified.all())
        self.assertEqual(runner.digest(self.checkpoint), checkpoint_hash)
        self.assertEqual(params.TASK_ARRIVAL_RATE, original_rate)
        self.assertEqual(params.EXPERIMENT_SEED, original_seed)
        self.assertEqual(json.loads((report / "status.json").read_text())["status"], "completed")
        ppo_directory = Path(comparison.ppo_source.iloc[0])
        before = {path: runner.digest(path) for path in ppo_directory.iterdir() if path.is_file()}
        reuse = runner.main(self.args("greedy", "--ppo-runs", str(report / "ppo_runs.json")))
        repeated = pd.read_csv(reuse / "comparison.csv")
        np.testing.assert_allclose(repeated.baseline_mean_delay_seconds,
                                   comparison.baseline_mean_delay_seconds, rtol=0, atol=0)
        for path, content_hash in before.items():
            self.assertEqual(runner.digest(path), content_hash)

        paired = pd.read_csv(reuse / "arrival1.5_paired_episodes.csv")
        self.assertEqual(paired.episode.tolist(), [1, 2])
        self.assertAlmostEqual(paired.baseline_minus_ppo_seconds.mean(),
                               repeated.paired_mean_baseline_minus_ppo_seconds.iloc[0])
        merged = runner.main(self.args("compare", "--checkpoint", str(self.checkpoint),
                                      "--rates", "2.5", "--merge", str(report / "comparison.csv")))
        self.assertEqual(pd.read_csv(merged / "comparison_all_rates.csv").arrival_rate.tolist(),
                         [1.5, 2.5])
        with self.assertRaisesRegex(ValueError, "duplicate arrival rates"):
            runner.main(self.args("compare", "--checkpoint", str(self.checkpoint),
                                  "--merge", str(report / "comparison.csv")))
        # A different arrival seed must be rejected before another evaluation starts.
        with self.assertRaisesRegex(ValueError, "EXPERIMENT_SEED"):
            runner.main(self.args("greedy", "--ppo-runs", str(report / "ppo_runs.json"),
                                  "--seed", "2028"))

    def test_frozen_evaluation_never_calls_training_and_restores_method(self):
        with runner.temporary_params(model_summary="ppo", taskno=12):
            model = Project_main.build_model(device="cpu")
            before = copy.deepcopy(runner.model_state(model))
            original_train_step = model.train_step
            loop = runner.frozen_evaluation(model, 2, seed=123, task_path=self.tasks)
            self.assertEqual(model.train_step, original_train_step)
            runner.assert_identical(before, runner.model_state(model))
            self.assertEqual(len(loop.task_results), 24)
            self.assertFalse(model.states)

    def test_training_then_evaluation_in_temporary_directory(self):
        with runner.temporary_params(model_summary="ppo", total_episodes=1, taskno=12):
            report = runner.main(self.args("train-eval"))
        state = json.loads((report / "status.json").read_text())
        self.assertEqual(state["status"], "completed")
        evaluation = Path(state["ppo_runs"]["1.5"])
        saved = json.loads((evaluation / "parameters.json").read_text())
        self.assertEqual(saved["taskno"], 12)
        self.assertEqual(saved["total_episodes"], 2)
        self.assertTrue(Path(saved["SOURCE_CHECKPOINT"]).is_file())
        self.assertEqual(len(pd.read_csv(evaluation / "task_results.csv")), 24)

    def test_validation_and_temporary_params_restore_after_failure(self):
        original = params.TASK_ARRIVAL_RATE
        with self.assertRaises(RuntimeError):
            with runner.temporary_params(TASK_ARRIVAL_RATE=99):
                raise RuntimeError("test")
        self.assertEqual(params.TASK_ARRIVAL_RATE, original)
        for arguments in (["compare"], ["ppo", "--checkpoint", str(self.checkpoint),
                                       "--rates", "nan"], ["greedy"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                runner.parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
