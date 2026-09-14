import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.generate_server_and_task_parameters import (
    SERVER_INFO_COLUMNS,
    assign_topology_locations,
    generate_server_info,
    load_eua_topology,
)


class EUAServerLocationTests(unittest.TestCase):
    @staticmethod
    def _topology_frame(count=20):
        return pd.DataFrame({
            "Rank": list(range(1, count + 1)),
            "Site_ID": [f"site-{rank}" for rank in range(1, count + 1)],
            "Latitude": [-37.8000 - rank * 0.001 for rank in range(1, count + 1)],
            "Longitude": [144.9000 + rank * 0.001 for rank in range(1, count + 1)],
            "Name": [f"Site {rank}" for rank in range(1, count + 1)],
            "Site_Precision": ["Within 10 meters"] * count,
        })

    def test_six_edge_two_cloud_assignment(self):
        assignments = assign_topology_locations(self._topology_frame(), 6, 2)

        self.assertEqual(assignments["Server_ID"].tolist(), list(range(1, 9)))
        self.assertEqual(assignments["Server_ID"].nunique(), 8)
        self.assertEqual(
            assignments.loc[assignments["Server_Type"] == "Cloud", "Topology_Rank"].tolist(),
            [1, 2],
        )
        self.assertEqual(
            assignments.loc[assignments["Server_Type"] == "Edge", "Topology_Rank"].tolist(),
            [3, 4, 5, 6, 7, 8],
        )
        self.assertEqual(assignments["Topology_Rank"].nunique(), 8)

    def test_expanding_edge_count_preserves_existing_positions(self):
        topology = self._topology_frame()
        six_edge = assign_topology_locations(topology, 6, 2)
        ten_edge = assign_topology_locations(topology, 10, 2)

        six_edge_positions = six_edge[six_edge["Server_Type"] == "Edge"].set_index("Topology_Rank")
        ten_edge_positions = ten_edge[ten_edge["Topology_Rank"].isin(range(3, 9))].set_index("Topology_Rank")
        pd.testing.assert_frame_equal(
            six_edge_positions[["Latitude", "Longitude"]],
            ten_edge_positions[["Latitude", "Longitude"]],
        )

        six_cloud_positions = six_edge[six_edge["Server_Type"] == "Cloud"].set_index("Topology_Rank")
        ten_cloud_positions = ten_edge[ten_edge["Server_Type"] == "Cloud"].set_index("Topology_Rank")
        pd.testing.assert_frame_equal(
            six_cloud_positions[["Latitude", "Longitude"]],
            ten_cloud_positions[["Latitude", "Longitude"]],
        )

    def test_generated_server_info_has_fixed_all_edge_nodes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            topology_path = root / "topology.csv"
            server_path = root / "server_info.xlsx"
            self._topology_frame().to_csv(topology_path, index=False)

            generate_server_info(server_path, topology_path=topology_path)
            server_info = pd.read_excel(server_path)

        self.assertEqual(list(server_info.columns), SERVER_INFO_COLUMNS)
        self.assertNotIn("Site_ID", server_info.columns)
        self.assertEqual(server_info["Server_ID"].tolist(), list(range(1, 9)))
        self.assertEqual(server_info["Server_ID"].nunique(), 8)
        self.assertEqual(server_info["Server_Type"].tolist(), ["Edge"] * 8)
        self.assertEqual(server_info["Processing_Frequency"].tolist(), [10, 11, 12, 14, 15, 17, 18, 20])
        self.assertTrue((server_info["Base_Failure_Rate"].diff().dropna() < 0).all())

        expected_ranks = list(range(3, 9)) + [1, 2]
        for row, rank in zip(server_info.itertuples(index=False), expected_ranks):
            site = self._topology_frame().iloc[rank - 1]
            self.assertAlmostEqual(row.Latitude, site.Latitude)
            self.assertAlmostEqual(row.Longitude, site.Longitude)

    def test_topology_validation_rejects_duplicate_rank_and_insufficient_sites(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            duplicate_path = root / "duplicate.csv"
            duplicate = self._topology_frame()
            duplicate.loc[1, "Rank"] = duplicate.loc[0, "Rank"]
            duplicate.to_csv(duplicate_path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate Rank"):
                load_eua_topology(duplicate_path)

            topology = self._topology_frame(4)
            with self.assertRaisesRegex(ValueError, "Not enough topology sites"):
                assign_topology_locations(topology, 3, 2)


if __name__ == "__main__":
    unittest.main()
