import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from tools.prepare_eua_topology import (
    TopologyValidationError,
    haversine_km,
    prepare_topology,
)


class EUATopologyPreparationTests(unittest.TestCase):
    @staticmethod
    def _input_frame():
        rows = []
        for index in range(20):
            rows.append({
                "SITE_ID": 100 + index,
                "LATITUDE": -37.8100 + (index % 5) * 0.001,
                "LONGITUDE": 144.9500 + (index // 5) * 0.001,
                "NAME": f"Site {index}",
                "STATE": "VIC",
                "LICENSING_AREA_ID": "area",
                "POSTCODE": 3000,
                "SITE_PRECISION": "Within 10 meters" if index != 0 else " Within 10 meters ",
                "ELEVATION": 10,
                "HCIS_L2": "hcis",
            })
        rows.append({
            "SITE_ID": 999,
            "LATITUDE": -37.8,
            "LONGITUDE": 144.9,
            "NAME": "Low precision site",
            "STATE": "VIC",
            "LICENSING_AREA_ID": "area",
            "POSTCODE": 3000,
            "SITE_PRECISION": "Within 100 meters",
            "ELEVATION": 10,
            "HCIS_L2": "hcis",
        })
        return pd.DataFrame(rows)

    def test_haversine_same_point_is_zero(self):
        self.assertAlmostEqual(
            haversine_km(-37.81, 144.95, -37.81, 144.95),
            0.0,
            places=12,
        )

    def test_filter_order_and_output_are_deterministic_and_nested(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "site-optus-melbCBD.csv"
            output_one = root / "topology-one.csv"
            output_two = root / "topology-two.csv"
            self._input_frame().to_csv(input_path, index=False)

            with self.assertWarnsRegex(UserWarning, "Expected 99"):
                first = prepare_topology(input_path, output_one)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                second = prepare_topology(input_path, output_two)

            self.assertEqual(output_one.read_bytes(), output_two.read_bytes())
            self.assertEqual(str(first.iloc[0]["Site_ID"]), "107")
            self.assertEqual(list(first.columns), [
                "Rank", "Site_ID", "Latitude", "Longitude", "Name", "Site_Precision"
            ])
            self.assertEqual(first["Rank"].tolist(), list(range(1, 21)))
            self.assertEqual(first["Site_ID"].nunique(), 20)
            self.assertTrue(
                (first["Site_Precision"] == "Within 10 meters").all()
            )
            self.assertNotIn("999", first["Site_ID"].astype(str).tolist())

            first_six = set(first["Site_ID"].iloc[:6])
            first_ten = set(first["Site_ID"].iloc[:10])
            first_twenty = set(first["Site_ID"].iloc[:20])
            self.assertTrue(first_six < first_ten < first_twenty)
            self.assertGreaterEqual(first["Latitude"].min(), -37.82)

            # The ordering cannot depend on input row order when a tie occurs.
            shuffled_path = root / "shuffled.csv"
            self._input_frame().iloc[::-1].to_csv(shuffled_path, index=False)
            shuffled_output = root / "topology-shuffled.csv"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                prepare_topology(shuffled_path, shuffled_output)
            self.assertEqual(output_one.read_bytes(), shuffled_output.read_bytes())

    def test_duplicate_filtered_site_id_is_rejected(self):
        frame = self._input_frame()
        frame.loc[1, "SITE_ID"] = frame.loc[0, "SITE_ID"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "duplicate.csv"
            output_path = Path(temporary_directory) / "output.csv"
            frame.to_csv(input_path, index=False)
            with self.assertRaises(TopologyValidationError):
                prepare_topology(input_path, output_path)

    def test_output_path_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "sites.csv"
            self._input_frame().to_csv(input_path, index=False)
            with self.assertRaises(ValueError):
                prepare_topology(input_path, input_path)


if __name__ == "__main__":
    unittest.main()
