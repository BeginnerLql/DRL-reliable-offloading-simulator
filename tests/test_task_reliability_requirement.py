"""Tests for task-level reliability requirements."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from config.configuration import parameters
from core.task import Task
from tools.analyze_pair_failure_correlation import _load_tasks
from tools.generate_server_and_task_parameters import (
    generate_reliability_requirements,
    generate_task_params,
)


class TaskReliabilityRequirementTests(unittest.TestCase):
    def test_requirements_are_balanced_shuffled_and_reproducible(self):
        first = generate_reliability_requirements(200, seed=2026)
        second = generate_reliability_requirements(200, seed=2026)
        self.assertEqual(len(first), 200)
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(first),
            Counter({0.9: 50, 0.99: 50, 0.999: 50, 0.9999: 50}),
        )
        self.assertNotEqual(set(first[:50]), {0.9})

    def test_generator_adds_requirement_column_without_changing_task_columns(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            output_path = Path(temp_directory) / "task_parameters.xlsx"
            generate_task_params(output_path)
            task_df = pd.read_excel(output_path)

        self.assertEqual(
            list(task_df.columns),
            [
                "Task_ID",
                "Task_Size",
                "Computation_Demand",
                "Reliability_Requirement",
            ],
        )
        self.assertEqual(len(task_df), 200)
        self.assertEqual(
            Counter(task_df["Reliability_Requirement"].tolist()),
            Counter({0.9: 50, 0.99: 50, 0.999: 50, 0.9999: 50}),
        )
        self.assertTrue(np.isfinite(task_df["Task_Size"]).all())
        self.assertTrue(np.isfinite(task_df["Computation_Demand"]).all())

    def test_task_reads_requirement_and_diagnostic_loader_ignores_extra_column(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            task_path = Path(temp_directory) / "task_parameters.xlsx"
            task_df = pd.DataFrame(
                {
                    "Task_ID": [1, 2],
                    "Task_Size": [10, 20],
                    "Computation_Demand": [30.0, 40.0],
                    "Reliability_Requirement": [0.999, 0.9999],
                }
            )
            task_df.to_excel(task_path, index=False)

            env = simpy.Environment()
            task = Task(env, object(), 1, params_file=task_path)
            self.assertEqual(task.task_size, 10)
            self.assertEqual(task.computation_demand, 30.0)
            self.assertAlmostEqual(task.reliability_requirement, 0.999)

            task_ids, demands = _load_tasks(task_path)
            self.assertEqual(task_ids.tolist(), [1, 2])
            self.assertTrue(np.array_equal(demands, np.array([30.0, 40.0])))

    def test_invalid_requirement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            task_path = Path(temp_directory) / "task_parameters.xlsx"
            pd.DataFrame(
                {
                    "Task_ID": [1],
                    "Task_Size": [10],
                    "Computation_Demand": [30.0],
                    "Reliability_Requirement": [0.95],
                }
            ).to_excel(task_path, index=False)
            with self.assertRaisesRegex(ValueError, "Reliability_Requirement"):
                Task(simpy.Environment(), object(), 1, params_file=task_path)


if __name__ == "__main__":
    unittest.main()
