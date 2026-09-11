import unittest

import numpy as np
import simpy

from core.server import Server
from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    haversine_distance_km,
    sample_spatial_risk_field,
    sample_spatial_risk_fields,
    validate_correlation_matrix,
)


class SpatialRiskTests(unittest.TestCase):
    @staticmethod
    def _server(server_id, latitude, longitude):
        return Server(
            simpy.Environment(),
            "Edge",
            server_id,
            10.0,
            0.001,
            latitude,
            longitude,
        )

    def test_haversine_same_point_is_zero(self):
        distance = haversine_distance_km(
            -37.814395,
            144.963537,
            -37.814395,
            144.963537,
        )
        self.assertAlmostEqual(distance, 0.0, places=12)

    def test_haversine_is_symmetric_and_matches_melbourne_scale(self):
        point_a = (-37.814395, 144.963537)
        point_b = (-37.820910, 144.955155)
        distance_ab = haversine_distance_km(*point_a, *point_b)
        distance_ba = haversine_distance_km(*point_b, *point_a)

        self.assertGreater(distance_ab, 0.0)
        self.assertAlmostEqual(distance_ab, distance_ba, places=12)
        self.assertAlmostEqual(distance_ab, 1.033, delta=0.02)

    def test_distance_matrix_sorts_and_maps_server_ids_explicitly(self):
        server_9 = self._server(9, -37.814395, 144.963537)
        server_2 = self._server(2, -37.820910, 144.955155)
        server_5 = self._server(5, -37.812390, 144.971200)

        server_ids, distance_matrix = build_distance_matrix(
            [server_9, server_2, server_5]
        )

        self.assertEqual(server_ids, [2, 5, 9])
        self.assertEqual(distance_matrix.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(distance_matrix)))
        self.assertTrue(np.all(distance_matrix >= 0.0))
        self.assertTrue(np.allclose(distance_matrix, distance_matrix.T))
        self.assertTrue(np.allclose(np.diag(distance_matrix), 0.0))
        self.assertAlmostEqual(
            distance_matrix[0, 1],
            haversine_distance_km(
                server_2.latitude,
                server_2.longitude,
                server_5.latitude,
                server_5.longitude,
            ),
        )
        self.assertAlmostEqual(
            distance_matrix[0, 2],
            haversine_distance_km(
                server_2.latitude,
                server_2.longitude,
                server_9.latitude,
                server_9.longitude,
            ),
        )

    def test_distance_matrix_rejects_empty_and_duplicate_servers(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_distance_matrix([])

        duplicate_a = self._server(2, -37.814395, 144.963537)
        duplicate_b = self._server(2, -37.820910, 144.955155)
        with self.assertRaisesRegex(ValueError, "duplicate server_id"):
            build_distance_matrix([duplicate_a, duplicate_b])

    def test_correlation_length_must_be_finite_and_positive(self):
        distance_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
        for correlation_length in (0.0, -0.5, float("nan"), float("inf")):
            with self.subTest(correlation_length=correlation_length):
                with self.assertRaisesRegex(ValueError, "correlation_length_km"):
                    build_spatial_correlation_matrix(
                        distance_matrix,
                        correlation_length,
                    )

    def test_spatial_correlation_matrix_uses_exponential_kernel(self):
        distance_matrix = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.25],
            [1.0, 0.25, 0.0],
        ])
        correlation_matrix = build_spatial_correlation_matrix(
            distance_matrix,
            0.5,
        )

        expected = np.exp(-distance_matrix / 0.5)
        self.assertTrue(np.allclose(correlation_matrix, expected))
        self.assertTrue(np.allclose(correlation_matrix, correlation_matrix.T))
        self.assertTrue(np.allclose(np.diag(correlation_matrix), 1.0))
        self.assertTrue((correlation_matrix > 0.0).all())
        self.assertTrue((correlation_matrix <= 1.0).all())
        self.assertLess(correlation_matrix[0, 2], correlation_matrix[0, 1])

    def test_correlation_length_has_exp_minus_one_meaning(self):
        distance_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
        correlation_matrix = build_spatial_correlation_matrix(
            distance_matrix,
            0.5,
        )
        self.assertAlmostEqual(correlation_matrix[0, 1], np.exp(-1.0))
        self.assertAlmostEqual(correlation_matrix[0, 1], 0.367879, places=5)

    def test_spatial_correlation_from_geographic_servers_is_psd(self):
        servers = [
            self._server(3, -37.814395, 144.963537),
            self._server(1, -37.820910, 144.955155),
            self._server(2, -37.812390, 144.971200),
        ]
        _, distance_matrix = build_distance_matrix(servers)
        correlation_matrix = build_spatial_correlation_matrix(distance_matrix, 0.5)
        self.assertTrue(validate_correlation_matrix(correlation_matrix))

    def test_invalid_correlation_matrix_reports_negative_eigenvalue(self):
        invalid_matrix = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
        ])
        with self.assertRaisesRegex(ValueError, "minimum eigenvalue"):
            validate_correlation_matrix(invalid_matrix)

    def test_invalid_correlation_matrix_shape_and_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "square"):
            validate_correlation_matrix(np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_correlation_matrix(np.array([[1.0, np.nan], [np.nan, 1.0]]))
        with self.assertRaisesRegex(ValueError, r"within \[-1, 1\]"):
            validate_correlation_matrix(np.array([[1.0, 1.1], [1.1, 1.0]]))
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_correlation_matrix(np.array([[1.0, 0.2], [0.3, 1.0]]))


    def test_single_spatial_risk_field_has_expected_shape_and_is_finite(self):
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        sample = sample_spatial_risk_field(
            correlation_matrix,
            rng=np.random.default_rng(2026),
        )
        self.assertEqual(sample.shape, (2,))
        self.assertTrue(np.isfinite(sample).all())

    def test_spatial_risk_sampling_is_reproducible_with_same_seed(self):
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        first = sample_spatial_risk_field(
            correlation_matrix,
            rng=np.random.default_rng(7),
        )
        second = sample_spatial_risk_field(
            correlation_matrix,
            rng=np.random.default_rng(7),
        )
        self.assertTrue(np.array_equal(first, second))

    def test_spatial_risk_sampling_changes_with_different_seeds(self):
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        first = sample_spatial_risk_field(
            correlation_matrix,
            rng=np.random.default_rng(7),
        )
        second = sample_spatial_risk_field(
            correlation_matrix,
            rng=np.random.default_rng(8),
        )
        self.assertFalse(np.array_equal(first, second))

    def test_spatial_risk_batch_has_expected_shape_and_is_finite(self):
        correlation_matrix = np.array([
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])
        samples = sample_spatial_risk_fields(
            correlation_matrix,
            1000,
            rng=np.random.default_rng(2026),
        )
        self.assertEqual(samples.shape, (1000, 3))
        self.assertTrue(np.isfinite(samples).all())

    def test_spatial_risk_batch_matches_target_statistics(self):
        correlation_matrix = np.array([
            [1.0, 0.45, 0.20],
            [0.45, 1.0, 0.35],
            [0.20, 0.35, 1.0],
        ])
        samples = sample_spatial_risk_fields(
            correlation_matrix,
            50_000,
            rng=np.random.default_rng(2026),
        )

        self.assertTrue(np.all(np.abs(samples.mean(axis=0)) < 0.03))
        self.assertTrue(np.all(np.abs(samples.var(axis=0, ddof=1) - 1.0) < 0.05))
        empirical_correlation = np.corrcoef(samples, rowvar=False)
        self.assertTrue(
            np.allclose(empirical_correlation, correlation_matrix, atol=0.03)
        )

    def test_independent_spatial_risk_components_have_near_zero_correlation(self):
        samples = sample_spatial_risk_fields(
            np.eye(3),
            50_000,
            rng=np.random.default_rng(2026),
        )
        empirical_correlation = np.corrcoef(samples, rowvar=False)
        off_diagonal = empirical_correlation - np.eye(3)
        self.assertLess(np.max(np.abs(off_diagonal)), 0.03)

    def test_strongly_correlated_spatial_risk_components_match_target(self):
        correlation_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
        samples = sample_spatial_risk_fields(
            correlation_matrix,
            50_000,
            rng=np.random.default_rng(2026),
        )
        empirical_correlation = np.corrcoef(samples, rowvar=False)
        self.assertAlmostEqual(empirical_correlation[0, 1], 0.9, delta=0.03)

    def test_spatial_risk_sampler_reuses_correlation_validation(self):
        invalid_matrix = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
        ])
        with self.assertRaisesRegex(ValueError, "minimum eigenvalue"):
            sample_spatial_risk_field(invalid_matrix, rng=np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "minimum eigenvalue"):
            sample_spatial_risk_fields(
                invalid_matrix,
                10,
                rng=np.random.default_rng(1),
            )

    def test_spatial_risk_batch_requires_positive_integer_count(self):
        correlation_matrix = np.eye(2)
        for num_samples in (0, -1, 1.5, float("nan")):
            with self.subTest(num_samples=num_samples):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    sample_spatial_risk_fields(
                        correlation_matrix,
                        num_samples,
                        rng=np.random.default_rng(1),
                    )


if __name__ == "__main__":
    unittest.main()
