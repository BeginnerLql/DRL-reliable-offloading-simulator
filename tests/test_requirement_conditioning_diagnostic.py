"""Focused checks for the offline reliability-requirement diagnostic."""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics.requirement_conditioning_diagnostic import input_trace


class RequirementInputTraceTest(unittest.TestCase):
    def test_all_four_requirements_reach_every_pair_feature(self):
        servers = pd.DataFrame({
            "Server_ID": list(range(1, 9)),
            "Base_Failure_Rate": [0.001] * 8,
            "Processing_Frequency": [12.5] * 8,
            "Uplink_Rate": [10.0] * 8,
        }).set_index("Server_ID")
        tasks = pd.DataFrame({"Input_Data_Size_MB": [2.0],
                              "Computation_Demand": [50.0]})
        pairs = list(itertools.combinations(range(1, 9), 2))
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "r_req_input_trace.txt"
            input_trace(servers, tasks, pairs, np.zeros(28), trace)
            content = trace.read_text(encoding="utf-8")
            for level in ("0.9", "0.99", "0.999", "0.9999"):
                self.assertIn(f"R_req={level}:", content)
            self.assertIn("all_28_pair_features[-1]=0.333333343", content)
            self.assertIn("Only the final observation coordinate changes", content)


if __name__ == "__main__":
    unittest.main()
