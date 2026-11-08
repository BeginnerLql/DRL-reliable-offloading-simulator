import copy
import importlib
import io
import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
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

    def test_independent_greedy_comparison_and_reuse(self):
        report = runner.main(self.args("compare", "--checkpoint", str(self.checkpoint),
                                       "--greedy-agent", "indep-delaygreedy"))
        comparison = pd.read_csv(report / "comparison.csv")
        self.assertEqual(comparison.baseline_model.tolist(), ["indep-delaygreedy"])
        self.assertTrue(comparison.paired_arrivals_verified.all())
        baseline_dir = Path(comparison.baseline_directory.iloc[0])
        baseline = pd.read_csv(baseline_dir / "task_results.csv")
        np.testing.assert_array_equal(baseline.feasible_action_count > 0,
                                      baseline.requirement_satisfied_independent)
        self.assertAlmostEqual(comparison.baseline_rsr_independent.iloc[0],
                               baseline.requirement_satisfied_independent.mean())
        reuse = runner.main(self.args("greedy", "--ppo-runs", str(report / "ppo_runs.json"),
                                      "--greedy-agent", "indep-delaygreedy"))
        repeated = pd.read_csv(reuse / "comparison.csv")
        self.assertEqual(repeated.baseline_model.tolist(), ["indep-delaygreedy"])
        self.assertEqual(repeated.baseline_mean_delay_seconds.iloc[0],
                         comparison.baseline_mean_delay_seconds.iloc[0])

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

    def test_independent_ppo_training_frozen_evaluation_and_reference_reuse(self):
        with runner.temporary_params(total_episodes=2, taskno=12):
            report = runner.main(self.args("train-eval", "--ppo-agent", "ind-ppo"))
        state = json.loads((report / "status.json").read_text())
        self.assertEqual(state["status"], "completed")
        directory = Path(state["ppo_runs"]["1.5"])
        saved = json.loads((directory / "parameters.json").read_text())
        self.assertEqual(saved["model_summary"], "ind-ppo")
        self.assertTrue(Path(saved["SOURCE_CHECKPOINT"]).name.startswith("ind-ppo_"))
        summary = json.loads((directory / "summary.json").read_text())
        self.assertTrue(summary["weights_and_optimizer_unchanged"])
        workbook_params = pd.read_excel(str(directory) + ".xlsx", sheet_name="Params")
        self.assertIn("actor_lr_ppo", workbook_params.Parameter.tolist())
        self.assertIn("hidden_layers_ppo", workbook_params.Parameter.tolist())
        reuse = runner.main(self.args("greedy", "--ppo-agent", "ind-ppo",
                                      "--greedy-agent", "indep-delaygreedy",
                                      "--ppo-runs", str(report / "ppo_runs.json")))
        comparison = pd.read_csv(reuse / "comparison.csv")
        self.assertEqual(comparison.ppo_model.tolist(), ["ind-ppo"])
        self.assertTrue(comparison.paired_arrivals_verified.all())
        # Its own logged checkpoint and frozen state are both preserved.
        runner.assert_identical(torch.load(saved["SOURCE_CHECKPOINT"], map_location="cpu", weights_only=True),
                                torch.load(directory / "checkpoint_latest.pt", map_location="cpu", weights_only=True))

    def test_rasoul_cli_logs_and_single_dual_post_processing(self):
        from io_utils.post_process_results import process_all_results

        old_actions = params.num_actions
        report = runner.main(self.args("rasoul"))
        self.assertEqual(params.num_actions, old_actions)
        state = json.loads((report / "status.json").read_text())
        self.assertEqual(state["status"], "completed")
        directory = Path(state["rasoul_runs"]["1.5"])
        saved = json.loads((directory / "parameters.json").read_text())
        self.assertEqual(saved["num_actions"], 8)
        self.assertEqual(saved["REPLICA_COUNT"], 1)
        tasks = pd.read_csv(directory / "task_results.csv")
        self.assertEqual(len(tasks), 24)
        self.assertTrue(tasks.server_B_id.isna().all())
        self.assertTrue(tasks.pair_reliability_correlated.isna().all())
        self.assertTrue(tasks.pair_reliability_independent.isna().all())
        self.assertTrue(tasks.replica_count.eq(1).all())
        learning = json.loads((directory / "learning_statistics.json").read_text())
        self.assertEqual(learning["decisions"], 12)
        self.assertEqual(sum(learning["completed_counts"].values()), 12)
        aggregation = self.root / "aggregation"
        aggregation.mkdir()
        shutil.copy2(str(directory) + ".xlsx", aggregation / "single.xlsx")
        dual_dir = runner.evaluate("corr-delaygreedy", 1.5, episodes=1, seed=123, device="cpu",
                                   checkpoint=None, task_path=self.tasks, run_id="postprocess_dual")
        shutil.copy2(str(dual_dir) + ".xlsx", aggregation / "dual.xlsx")
        process_all_results(aggregation)
        workbook = aggregation / "Final_Result_All.xlsx"
        self.assertEqual(len(pd.read_excel(workbook, sheet_name="ServerSelection")), 8)
        self.assertEqual(len(pd.read_excel(workbook, sheet_name="PairSelection")), 28)
        self.assertEqual(len(pd.read_excel(workbook, sheet_name="ModelSummary")), 2)
        original_hash = runner.digest(workbook)
        with self.assertRaises(FileExistsError):
            process_all_results(aggregation)
        self.assertEqual(runner.digest(workbook), original_hash)

    def test_rasoul_main_entrypoint_writes_single_replica_workbook(self):
        with runner.temporary_params(model_summary="RASOUL-Delay", total_episodes=2, taskno=12):
            self.assertIsNone(Project_main.run_simulation(device="cpu"))
        workbook, = self.root.glob("rasoul-delay_*.xlsx")
        tasks = pd.read_excel(workbook, sheet_name="TaskResults")
        self.assertEqual(len(tasks), 24)
        self.assertTrue(tasks.replica_count.eq(1).all())
        self.assertTrue(tasks.server_B_id.isna().all())
        self.assertTrue(tasks.pair_reliability_correlated.isna().all())
        np.testing.assert_array_equal(tasks.task_reliability, tasks.replica_A_reliability)

    def test_rasoul_name_variants_preserve_metrics_and_learning_output(self):
        original = params.model_summary
        for index, name in enumerate(("RASOUL-Delay", " RASOUL-DELAY ", "rasoul-delay")):
            with self.subTest(name=name):
                with runner.temporary_params(model_summary=name):
                    self.assertEqual(params.model_summary, "rasoul-delay")
                    self.assertEqual(params.num_actions, 8)
                self.assertEqual(params.model_summary, original)
                directory = runner.evaluate(name, 1.5, episodes=1, seed=123, device="cpu",
                                            checkpoint=None, task_path=self.tasks, run_id=f"name{index}")
                self.assertTrue(directory.name.startswith("rasoul-delay_"))
                self.assertTrue(Path(str(directory) + ".xlsx").is_file())
                saved = json.loads((directory / "parameters.json").read_text())
                self.assertEqual(saved["model_summary"], "rasoul-delay")
                self.assertEqual(saved["REPLICA_COUNT"], 1)
                self.assertEqual(saved["ACTION_SELECTION"], "single-replica delay LCB with completed feedback")
                summary = json.loads((directory / "summary.json").read_text())
                self.assertIn("rsr_single", summary)
                learning = json.loads((directory / "learning_statistics.json").read_text())
                self.assertEqual(sum(learning["completed_counts"].values()), 12)

    def test_streaming_logger_and_workbook_normalize_raw_model_name(self):
        logger_module = importlib.import_module("io_utils.long_run_logger")
        writer = importlib.import_module("io_utils.save_parameters_and_logs")
        with runner.temporary_params(model_summary="rasoul-delay", total_episodes=1, taskno=12):
            raw = SimpleNamespace(**{key: value for key, value in vars(params).items()
                                     if not key.startswith("_")})
            raw.model_summary = " RASOUL-Delay "
            model = Project_main.build_model(device="cpu")
            with logger_module.LongRunLogger(raw, model, "raw_name") as logger:
                loop = runner.MainLoop(model, 1, 12, params.num_states, params.num_actions,
                                       task_parameters_path=self.tasks, episode_end_callback=logger)
                loop.EP()
            writer.save_params_and_logs(raw, loop.log_data, loop.task_results, "raw_name",
                                        task_parameters_path=self.tasks,
                                        task_results_csv_path=logger.task_results_csv_path)
        self.assertTrue(logger.run_dir.name.startswith("rasoul-delay_"))
        self.assertEqual(json.loads((logger.run_dir / "parameters.json").read_text())["model_summary"],
                         "rasoul-delay")
        workbook = Path(str(logger.run_dir) + ".xlsx")
        saved = pd.read_excel(workbook, sheet_name="Params").set_index("Parameter")["Value"]
        self.assertEqual(saved["model_summary"], "rasoul-delay")
        self.assertEqual(raw.model_summary, " RASOUL-Delay ")

    def test_relocated_canonical_and_legacy_result_directories(self):
        from io_utils.post_process_results import process_all_results
        from openpyxl import load_workbook

        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                directory = runner.evaluate("RASOUL-Delay", 1.5, episodes=1, seed=123, device="cpu",
                                            checkpoint=None, task_path=self.tasks, run_id=f"move{legacy}")
                workbook = Path(str(directory) + ".xlsx")
                if legacy:
                    old_directory = directory.with_name(directory.name.replace("rasoul-delay", "RASOUL-Delay", 1))
                    directory.rename(old_directory)
                    directory = old_directory
                    book = load_workbook(workbook)
                    book["TaskResults"]["A2"] = str(directory / "task_results.csv")
                    book.save(workbook)
                    book.close()
                moved = self.root / f"moved{legacy}"
                moved.mkdir()
                shutil.move(directory, moved / directory.name)
                shutil.move(workbook, moved / workbook.name)
                process_all_results(moved)
                summary = pd.read_excel(moved / "Final_Result_All.xlsx", sheet_name="ModelSummary")
                self.assertEqual(summary.Task_Outcomes.tolist(), [12])

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
