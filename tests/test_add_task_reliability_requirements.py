"""Tests for safe task-workbook reliability migration."""

import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tools.add_task_reliability_requirements import migrate_task_parameters
from tools.generate_server_and_task_parameters import generate_reliability_requirements


class AddTaskReliabilityRequirementsTests(unittest.TestCase):
    @staticmethod
    def _old_tasks():
        return pd.DataFrame(
            {
                "Task_ID": np.arange(1, 201),
                "Task_Size": np.arange(100, 300),
                "Computation_Demand": np.linspace(1.25, 99.75, 200),
            }
        )

    def test_migration_preserves_existing_columns_and_adds_only_requirement(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            task_path = root / "task_parameters.xlsx"
            server_path = root / "server_info.xlsx"
            original = self._old_tasks()
            original.to_excel(task_path, index=False)
            original = pd.read_excel(task_path)
            server_path.write_bytes(b"server workbook sentinel")
            server_digest = hashlib.sha256(server_path.read_bytes()).digest()

            migrate_task_parameters(task_path, seed=2026)
            migrated = pd.read_excel(task_path)

            self.assertEqual(
                list(migrated.columns),
                ["Task_ID", "Task_Size", "Computation_Demand", "Reliability_Requirement"],
            )
            pd.testing.assert_frame_equal(
                migrated.loc[:, ["Task_ID", "Task_Size", "Computation_Demand"]],
                original,
                check_dtype=False,
                check_exact=True,
            )
            self.assertEqual(
                Counter(migrated["Reliability_Requirement"]),
                Counter({0.9: 50, 0.99: 50, 0.999: 50, 0.9999: 50}),
            )
            self.assertEqual(server_path.read_bytes(), b"server workbook sentinel")
            self.assertEqual(hashlib.sha256(server_path.read_bytes()).digest(), server_digest)

    def test_existing_requirement_is_deterministically_refreshed(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            task_path = Path(temp_directory) / "task_parameters.xlsx"
            original = self._old_tasks()
            original["Reliability_Requirement"] = 0.9
            original.to_excel(task_path, index=False)

            first = migrate_task_parameters(task_path, seed=2026)
            second = migrate_task_parameters(task_path, seed=2026)
            expected = generate_reliability_requirements(200, seed=2026)

            self.assertEqual(first["Reliability_Requirement"].tolist(), expected)
            self.assertEqual(second["Reliability_Requirement"].tolist(), expected)
            self.assertTrue(
                np.array_equal(
                    first[["Task_ID", "Task_Size", "Computation_Demand"]].to_numpy(),
                    second[["Task_ID", "Task_Size", "Computation_Demand"]].to_numpy(),
                )
            )

    def test_missing_or_unexpected_columns_are_rejected_without_replacement(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            missing_path = root / "missing.xlsx"
            missing = self._old_tasks().drop(columns="Computation_Demand")
            missing.to_excel(missing_path, index=False)
            with self.assertRaisesRegex(ValueError, "Computation_Demand"):
                migrate_task_parameters(missing_path)
            pd.testing.assert_frame_equal(pd.read_excel(missing_path), missing)

            unexpected_path = root / "unexpected.xlsx"
            unexpected = self._old_tasks()
            unexpected["Unrelated"] = 1
            unexpected.to_excel(unexpected_path, index=False)
            with self.assertRaisesRegex(ValueError, "Unrelated"):
                migrate_task_parameters(unexpected_path)
            pd.testing.assert_frame_equal(pd.read_excel(unexpected_path), unexpected)


if __name__ == "__main__":
    unittest.main()
