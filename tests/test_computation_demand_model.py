import tempfile
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from config.params import params
from core.env_state import EnvironmentState
from tools.generate_server_and_task_parameters import (
    generate_computation_demands,
    generate_reliability_requirements,
    generate_task_params,
)


class ComputationDemandModelTests(unittest.TestCase):
    def test_formal_range_and_seed(self):
        self.assertEqual(params.COMPUTATION_DEMAND_RANGE_MI, (5, 50))
        self.assertEqual(params.COMPUTATION_DEMAND_SEED, 2030)
        self.assertEqual(params.Low_demand, 5)
        self.assertEqual(params.High_demand, 50)

    def test_discrete_uniform_generator_is_reproducible_and_integer_valued(self):
        first = generate_computation_demands(200)
        second = generate_computation_demands(200)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 200)
        self.assertTrue(all(isinstance(value, int) for value in first))
        self.assertTrue(all(5 <= value <= 50 for value in first))
        self.assertGreater(len(set(first)), 1)
        self.assertAlmostEqual(np.mean(first), 27.5, delta=5.0)

    def test_computation_rng_is_independent_of_input_rng(self):
        baseline = generate_computation_demands(200)
        input_rng = np.random.default_rng(params.INPUT_DATA_SIZE_SEED)
        input_rng.uniform(*params.INPUT_DATA_SIZE_RANGE_MB, size=200)
        changed_global = generate_computation_demands(200)
        self.assertEqual(baseline, changed_global)

    def test_generated_workbook_keeps_independent_input_and_requirement_streams(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            output_path = Path(temp_directory) / "task_parameters.xlsx"
            generate_task_params(output_path)
            task_df = pd.read_excel(output_path)

        self.assertEqual(
            list(task_df.columns),
            [
                "Task_ID",
                "Input_Data_Size_MB",
                "Computation_Demand",
                "Reliability_Requirement",
            ],
        )
        expected_input = np.random.default_rng(params.INPUT_DATA_SIZE_SEED).uniform(
            *params.INPUT_DATA_SIZE_RANGE_MB,
            size=params.taskno,
        )
        np.testing.assert_allclose(task_df["Input_Data_Size_MB"], expected_input)
        self.assertEqual(
            task_df["Computation_Demand"].tolist(),
            generate_computation_demands(params.taskno),
        )
        self.assertEqual(
            Counter(task_df["Reliability_Requirement"].tolist()),
            Counter({0.9: 50, 0.99: 50, 0.999: 50, 0.9999: 50}),
        )

    def test_current_workbook_matches_formal_task_model(self):
        task_df = pd.read_excel(Path("data") / "task_parameters.xlsx")
        demand = task_df["Computation_Demand"].to_numpy(dtype=float)
        self.assertEqual(len(task_df), 200)
        self.assertTrue(np.isfinite(demand).all())
        self.assertTrue(np.all(demand >= 5))
        self.assertTrue(np.all(demand <= 50))
        self.assertTrue(np.all(demand == np.round(demand)))
        self.assertTrue(task_df["Input_Data_Size_MB"].between(0.5, 2.0).all())

    def test_state_normalization_uses_five_to_fifty(self):
        self.assertEqual(EnvironmentState().normalize(5, *params.COMPUTATION_DEMAND_RANGE_MI), 0.0)
        self.assertEqual(EnvironmentState().normalize(50, *params.COMPUTATION_DEMAND_RANGE_MI), 1.0)
        self.assertTrue(0.0 <= EnvironmentState().normalize(27, 5, 50) <= 1.0)

    def test_service_time_theoretical_range(self):
        frequencies = params.FIXED_EDGE_PROCESSING_FREQUENCIES
        service_times = [demand / frequency for demand in (5, 50) for frequency in frequencies]
        self.assertAlmostEqual(min(service_times), 5 / 20)
        self.assertAlmostEqual(max(service_times), 50 / 10)

    def test_dimensions_remain_unchanged(self):
        self.assertEqual(params.num_states, 35)
        self.assertEqual(params.num_actions, 28)


if __name__ == "__main__":
    unittest.main()
