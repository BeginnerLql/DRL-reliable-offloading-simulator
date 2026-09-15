"""Pure geographic distance and latent physical risk-correlation utilities.

The exponential kernel implemented here is

    R_jk = exp(-d_jk / ell_phy)

where ``d_jk`` is the Haversine distance in kilometres and ``ell_phy`` is a
positive correlation length in kilometres.  This represents latent physical
environmental risk correlation; it is not a failure-event Pearson
correlation, a failure probability, or a joint failure probability.

This module only computes matrices or samples explicitly requested latent risk
fields.  It does not modify ``Server.failure_rate`` or mutate simulator state.
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


def extract_pair_correlations(
    server_ids: Iterable[object],
    correlation_matrix: object,
    action_pairs: Iterable[tuple[object, object]],
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Return spatial-risk correlations in the supplied action-pair order.

    ``server_ids`` defines the row/column mapping of the matrix; action pairs
    use the same external server IDs as the simulator.  This helper is pure
    and does not infer or alter any runtime failure rates.
    """
    ids = list(server_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("server_ids must be non-empty and unique")
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
    if matrix.ndim != 2 or matrix.shape != (len(ids), len(ids)):
        raise ValueError(
            "correlation_matrix must be square with one row per server_id"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("correlation_matrix must contain only finite values")
    if (matrix < -tolerance).any() or (matrix > 1.0 + tolerance).any():
        raise ValueError("correlation_matrix entries must lie in [0, 1]")

    id_to_index = {server_id: index for index, server_id in enumerate(ids)}
    pairs = list(action_pairs)
    correlations = []
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each action pair must contain exactly two server IDs")
        server_j, server_k = pair
        if server_j == server_k:
            raise ValueError("action pairs must contain two distinct server IDs")
        if server_j not in id_to_index or server_k not in id_to_index:
            raise ValueError("action pair contains an unknown server ID")
        value = float(matrix[id_to_index[server_j], id_to_index[server_k]])
        if not math.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
            raise ValueError("pair spatial-risk correlations must lie in [0, 1]")
        correlations.append(float(np.clip(value, 0.0, 1.0)))

    result = np.asarray(correlations, dtype=float)
    if result.ndim != 1 or result.shape[0] != len(pairs):
        raise ValueError("extracted pair correlations have an invalid shape")
    return result


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


def _sampling_inputs(
    correlation_matrix: object,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.random.Generator]:
    """Validate sampling inputs without repairing the covariance matrix."""
    validate_correlation_matrix(correlation_matrix)
    matrix = np.asarray(correlation_matrix, dtype=float)
    generator = np.random.default_rng() if rng is None else rng
    if not hasattr(generator, "multivariate_normal"):
        raise ValueError("rng must provide a multivariate_normal method")
    return matrix, generator


def _validate_num_samples(num_samples: object) -> int:
    if isinstance(num_samples, (bool, np.bool_)) or not isinstance(
        num_samples,
        (int, np.integer),
    ):
        raise ValueError("num_samples must be a positive integer")
    count = int(num_samples)
    if count <= 0:
        raise ValueError("num_samples must be a positive integer")
    return count


def sample_spatial_risk_field(
    correlation_matrix: object,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample one latent standardized physical environmental risk field.

    The returned ``Z_phy`` follows ``N(0, R)``: each component has expected
    value zero and variance one, and the component correlation is ``R``.  The
    field is a latent physical environmental risk field, not a failure field
    and not a failure-event or failure-probability sample.
    """
    matrix, generator = _sampling_inputs(correlation_matrix, rng)
    dimension = matrix.shape[0]
    sample = generator.multivariate_normal(
        mean=np.zeros(dimension, dtype=float),
        cov=matrix,
        check_valid="raise",
    )
    sample = np.asarray(sample, dtype=float)
    if sample.shape != (dimension,):
        raise ValueError("sampled spatial risk field must have shape (N,)")
    if not np.isfinite(sample).all():
        raise ValueError("sampled spatial risk field must be finite")
    return sample


def sample_spatial_risk_fields(
    correlation_matrix: object,
    num_samples: object,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a batch of latent standardized physical environmental risk fields.

    The returned array has shape ``(num_samples, N)``.  Every row is a draw
    from ``Z_phy ~ N(0, R)`` with expected value zero, variance one, and
    component correlation ``R``.  These are latent physical environmental
    risk fields, not failure fields or failure-event samples.
    """
    count = _validate_num_samples(num_samples)
    matrix, generator = _sampling_inputs(correlation_matrix, rng)
    dimension = matrix.shape[0]
    samples = generator.multivariate_normal(
        mean=np.zeros(dimension, dtype=float),
        cov=matrix,
        size=count,
        check_valid="raise",
    )
    samples = np.asarray(samples, dtype=float)
    expected_shape = (count, dimension)
    if samples.shape != expected_shape:
        raise ValueError(
            "sampled spatial risk fields must have shape "
            f"{expected_shape}"
        )
    if not np.isfinite(samples).all():
        raise ValueError("sampled spatial risk fields must be finite")
    return samples


def _validate_failure_rate_vector(base_failure_rates: object) -> np.ndarray:
    try:
        rates = np.asarray(base_failure_rates, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("base_failure_rates must be a numeric one-dimensional vector") from exc
    if rates.ndim != 1:
        raise ValueError("base_failure_rates must be one-dimensional")
    if rates.size == 0:
        raise ValueError("base_failure_rates must not be empty")
    if not np.isfinite(rates).all():
        raise ValueError("base_failure_rates must contain only finite values")
    if (rates < 0.0).any():
        raise ValueError("base_failure_rates must be non-negative")
    return rates


def _validate_spatial_risk_vector(spatial_risk_field: object) -> np.ndarray:
    try:
        risk = np.asarray(spatial_risk_field, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("spatial_risk_field must be a numeric one-dimensional vector") from exc
    if risk.ndim != 1:
        raise ValueError("spatial_risk_field must be one-dimensional")
    if risk.size == 0:
        raise ValueError("spatial_risk_field must not be empty")
    if not np.isfinite(risk).all():
        raise ValueError("spatial_risk_field must contain only finite values")
    return risk


def _validate_beta_p(beta_p: object) -> float:
    try:
        beta_array = np.asarray(beta_p, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta_p must be a finite non-negative scalar") from exc
    if beta_array.ndim != 0:
        raise ValueError("beta_p must be a scalar")
    beta = float(beta_array)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta_p must be a finite non-negative scalar")
    return beta


def map_spatial_risk_to_effective_failure_rates(
    base_failure_rates: object,
    spatial_risk_field: object,
    beta_p: object,
) -> np.ndarray:
    """Map a latent physical risk field to effective transient failure rates.

    For each node, this implements

    ``lambda_eff_j = lambda_0_j * exp(beta_p * Z_phy_j)``.

    ``lambda_0_j`` is the normal-environment rate in ``1/s`` and ``Z_phy``
    is a latent standardized physical environmental risk field.  For
    ``beta_p > 0``, larger ``Z_phy_j`` produces a larger effective rate, while
    smaller values produce a smaller rate.  Since ``lambda_0_j`` is defined at
    normal environment, ``Z_phy_j = 0`` exactly recovers ``lambda_0_j``; no
    lognormal mean correction is applied.

    This is a pure numerical mapping.  It does not read or modify ``Server``,
    task, or simulator state.
    """
    base_rates = _validate_failure_rate_vector(base_failure_rates)
    risk_field = _validate_spatial_risk_vector(spatial_risk_field)
    if base_rates.shape != risk_field.shape:
        raise ValueError(
            "base_failure_rates and spatial_risk_field must have identical shapes"
        )
    beta = _validate_beta_p(beta_p)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exponent = np.float64(beta) * risk_field
    if not np.isfinite(exponent).all():
        raise ValueError("effective failure rate became non-finite")

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        effective_rates = base_rates * np.exp(exponent)
    if (
        not np.isfinite(effective_rates).all()
        or (effective_rates < 0.0).any()
    ):
        raise ValueError("effective failure rate became non-finite")
    return np.asarray(effective_rates, dtype=float)
