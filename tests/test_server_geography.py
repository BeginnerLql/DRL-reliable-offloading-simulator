import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import simpy

from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.server import Server


class ServerGeographyTests(unittest.TestCase):
    def test_server_stores_valid_coordinates_as_float(self):
        server = Server(
            simpy.Environment(),
            "Edge",
            1,
            10.0,
            0.001,
            "-37.814395",
            "144.963537",
        )

        self.assertEqual(server.latitude, -37.814395)
        self.assertEqual(server.longitude, 144.963537)
        self.assertIsInstance(server.latitude, float)
        self.assertIsInstance(server.longitude, float)

    def test_invalid_latitude_is_rejected(self):
        for latitude in (-90.000001, 90.000001):
            with self.subTest(latitude=latitude):
                with self.assertRaisesRegex(ValueError, "latitude"):
                    Server(simpy.Environment(), "Edge", 1, 10, 0.001, latitude, 0)

    def test_invalid_longitude_is_rejected(self):
        for longitude in (-180.000001, 180.000001):
            with self.subTest(longitude=longitude):
                with self.assertRaisesRegex(ValueError, "longitude"):
                    Server(simpy.Environment(), "Edge", 1, 10, 0.001, 0, longitude)

    def test_non_finite_coordinates_are_rejected(self):
        for latitude, longitude in ((float("nan"), 0), (float("inf"), 0), (0, float("-inf"))):
            with self.subTest(latitude=latitude, longitude=longitude):
                with self.assertRaisesRegex(ValueError, "must be a finite number"):
                    Server(
                        simpy.Environment(),
                        "Edge",
                        1,
                        10,
                        0.001,
                        latitude,
                        longitude,
                    )

    @staticmethod
    def _server_info_frame():
        return pd.DataFrame({
            "Server_ID": [1, 2, 3],
            "Server_Type": ["Edge", "Edge", "Cloud"],
            "Processing_Frequency": [10.0, 11.0, 40.0],
            "Failure_Rate": [0.001, 0.002, 0.0005],
            "Uplink_Rate": [18.0, 30.0, 22.0],
            "Latitude": [-37.81517, -37.813175, -37.814395],
            "Longitude": [144.97476, 144.952919, 144.963537],
        })

    def test_main_loop_set_servers_loads_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self._server_info_frame().to_excel(
                data_dir / "server_info.xlsx",
                index=False,
            )

            loop = MainLoop.__new__(MainLoop)
            loop.env = simpy.Environment()
            loop.env_state = EnvironmentState()
            with patch("core.main_loop.DATA_DIR", str(data_dir)):
                loop.setServers()

        self.assertEqual(set(loop.env_state.servers), {1, 2, 3})
        self.assertEqual(len(loop.env_state.servers), 3)
        for server_id, expected in self._server_info_frame().set_index("Server_ID").iterrows():
            server = loop.env_state.servers[server_id]["server_object"]
            self.assertEqual(server.server_id, server_id)
            self.assertAlmostEqual(server.latitude, expected["Latitude"])
            self.assertAlmostEqual(server.longitude, expected["Longitude"])

    def test_main_loop_rejects_missing_coordinate_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            incomplete = self._server_info_frame().drop(columns=["Longitude"])
            incomplete.to_excel(data_dir / "server_info.xlsx", index=False)

            loop = MainLoop.__new__(MainLoop)
            loop.env = simpy.Environment()
            loop.env_state = EnvironmentState()
            with patch("core.main_loop.DATA_DIR", str(data_dir)):
                with self.assertRaisesRegex(ValueError, "Longitude"):
                    loop.setServers()


if __name__ == "__main__":
    unittest.main()
