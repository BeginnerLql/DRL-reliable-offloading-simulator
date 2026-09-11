"""Pure geographic distance and latent physical risk-correlation utilities.

The exponential kernel implemented here is

    R_jk = exp(-d_jk / ell_phy)

where ``d_jk`` is the Haversine distance in kilometres and ``ell_phy`` is a
positive correlation length in kilometres.  This represents latent physical
environmental risk correlation; it is not a failure-event Pearson
correlation, a failure probability, or a joint failure probability.

This module only computes matrices.  It does not sample a spatial risk field,
modify Server.failure_rate, or mutate simulator state.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EARTH_RADIUS_KM = 6371.0088
_DISTANCE_TOLERANCE_KM = 1e-10


def _coordinate(value: object, name: str, minimum: float, maximum: float) -> float:
    try:
        coordinate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(coordinate):
        raise ValueError(f"{name} must be a finite number")
    if not minimum <= coordinate <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return coordinate


def haversine_distance_km(
    lat1: object,
    lon1: object,
    lat2: object,
    lon2: object,
) -> float:
    """Return the Haversine great-circle distance between two degree coordinates."""
    lat1_rad = math.radians(_coordinate(lat1, "latitude", -90.0, 90.0))
    lon1_rad = math.radians(_coordinate(lon1, "longitude", -180.0, 180.0))
    lat2_rad = math.radians(_coordinate(lat2, "latitude", -90.0, 90.0))
    lon2_rad = math.radians(_coordinate(lon2, "longitude", -180.0, 180.0))

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2.0) ** 2
    )
    a = min(1.0, max(0.0, a))
    distance = 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("Haversine distance must be finite and non-negative")
    return float(distance)


def build_distance_matrix(servers: Iterable[object]) -> tuple[list[object], np.ndarray]:
    """Build a Server_ID-sorted Haversine distance matrix.

    The returned ``server_ids`` list defines the row/column mapping explicitly;
    matrix index zero is not assumed to represent Server_ID 1.
    """
    server_list = list(servers)
    if not server_list:
        raise ValueError("servers must not be empty")

    server_ids = [server.server_id for server in server_list]
    if len(set(server_ids)) != len(server_ids):
        raise ValueError("duplicate server_id values are not allowed")

    try:
        ordered_servers = sorted(server_list, key=lambda server: server.server_id)
    except TypeError as exc:
        raise ValueError("server_id values must be mutually orderable") from exc

    ordered_ids = [server.server_id for server in ordered_servers]
    count = len(ordered_servers)
    distance_matrix = np.zeros((count, count), dtype=float)

    for row_index, server_a in enumerate(ordered_servers):
        for column_index in range(row_index + 1, count):
            server_b = ordered_servers[column_index]
            distance = haversine_distance_km(
                server_a.latitude,
                server_a.longitude,
                server_b.latitude,
                server_b.longitude,
            )
            distance_matrix[row_index, column_index] = distance
            distance_matrix[column_index, row_index] = distance

    if distance_matrix.shape != (count, count):
        raise ValueError("distance matrix has an invalid shape")
    if not np.isfinite(distance_matrix).all() or (distance_matrix < 0.0).any():
        raise ValueError("distance matrix must be finite and non-negative")
    return ordered_ids, distance_matrix


def _validated_distance_matrix(distance_matrix: object) -> np.ndarray:
    try:
        matrix = np.asarray(distance_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance_matrix must be numeric") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be a two-dimensional square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("distance_matrix must contain only finite values")
    if (matrix < 0.0).any():
        raise ValueError("distance_matrix must be non-negative")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=_DISTANCE_TOLERANCE_KM):
        raise ValueError("distance_matrix must be symmetric")
    if not np.allclose(
        np.diag(matrix),
        0.0,
        rtol=0.0,
        atol=_DISTANCE_TOLERANCE_KM,
    ):
        raise ValueError("distance_matrix diagonal must be approximately zero")
    return matrix


def build_spatial_correlation_matrix(
    distance_matrix: object,
    correlation_length_km: object,
) -> np.ndarray:
    """Build ``exp(-D / correlation_length_km)`` from a valid distance matrix."""
    try:
        correlation_length = float(correlation_length_km)
    except (TypeError, ValueError) as exc:
        raise ValueError("correlation_length_km must be a finite positive number") from exc
    if not math.isfinite(correlation_length) or correlation_length <= 0.0:
        raise ValueError("correlation_length_km must be a finite positive number")

    matrix = _validated_distance_matrix(distance_matrix)
    correlation_matrix = np.exp(-matrix / correlation_length)
    if not np.isfinite(correlation_matrix).all():
        raise ValueError("spatial correlation matrix must contain only finite values")
    return correlation_matrix


def validate_correlation_matrix(
    correlation_matrix: object,
    tolerance: float = 1e-10,
) -> bool:
    """Validate shape, bounds, symmetry, unit diagonal, and positive semidefiniteness."""
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be a finite non-negative number") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative number")

    try:
        matrix = np.asarray(correlation_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("correlation_matrix must be numeric") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation_matrix must be a two-dimensional square matrix")
    if matrix.shape[0] == 0:
        raise ValueError("correlation_matrix must not be empty")
    if not np.isfinite(matrix).all():
        raise ValueError("correlation_matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError("correlation_matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("correlation_matrix diagonal must be approximately one")
    if (matrix < -1.0 - tolerance).any() or (matrix > 1.0 + tolerance).any():
        raise ValueError("correlation_matrix entries must lie within [-1, 1]")

    symmetric_matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric_matrix)
    minimum_eigenvalue = float(eigenvalues.min())
    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            "correlation_matrix is not positive semidefinite; "
            f"minimum eigenvalue = {minimum_eigenvalue}"
        )
    return True
