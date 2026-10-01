"""Pure geographic distance and latent physical risk-correlation utilities.

The exponential kernel implemented here is

    R_jk = exp(-d_jk / ell_phy)

where ``d_jk`` is the Haversine distance in kilometres and ``ell_phy`` is a
positive correlation length in kilometres.  This represents latent physical
environmental risk correlation; it is not a failure-event Pearson
correlation, a failure probability, or a joint failure probability.

This module only computes matrices or samples explicitly requested latent risk
fields.  It does not modify ``Server.base_failure_rate`` or mutate simulator
state.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(
    lat1: object,
    lon1: object,
    lat2: object,
    lon2: object,
) -> float:
    """Return the Haversine great-circle distance between two degree coordinates."""
    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))

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
    return float(distance)


def build_distance_matrix(servers: Iterable[object]) -> tuple[list[object], np.ndarray]:
    """Build a Server_ID-sorted Haversine distance matrix.

    The returned ``server_ids`` list defines the row/column mapping explicitly;
    matrix index zero is not assumed to represent Server_ID 1.
    """
    ordered_servers = sorted(servers, key=lambda server: server.server_id)
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

    return ordered_ids, distance_matrix


def build_spatial_correlation_matrix(
    distance_matrix: object,
    correlation_length_km: object,
) -> np.ndarray:
    """Build ``exp(-D / correlation_length_km)`` from a valid distance matrix."""
    correlation_length = float(correlation_length_km)
    if correlation_length <= 0:
        raise ValueError("correlation_length_km must be positive")
    matrix = np.asarray(distance_matrix, dtype=float)
    return np.exp(-matrix / correlation_length)


def sample_spatial_risk_field(
    correlation_matrix: object,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one latent standardized physical environmental risk field.

    The returned ``Z_phy`` follows ``N(0, R)``: each component has expected
    value zero and variance one, and the component correlation is ``R``.  The
    field is a latent physical environmental risk field, not a failure field
    and not a failure-event or failure-probability sample.
    """
    matrix = np.asarray(correlation_matrix, dtype=float)
    dimension = matrix.shape[0]
    return rng.multivariate_normal(
        mean=np.zeros(dimension, dtype=float),
        cov=matrix,
        check_valid="raise",
    )


def map_spatial_risk_to_effective_failure_rates(
    base_failure_rates: object,
    spatial_risk_field: object,
    beta_p: object,
) -> np.ndarray:
    """Map a latent physical risk field to effective transient failure rates.

    For each node, this implements

    ``lambda_eff_j = lambda_0_j * exp(beta_p * Z_phy_j - beta_p**2 / 2)``.

    ``lambda_0_j`` is the base transient fault arrival rate in ``1/s`` and
    ``Z_phy`` is a latent standardized physical environmental risk field.  For
    ``beta_p > 0``, larger ``Z_phy_j`` produces a larger effective rate, while
    smaller values produce a smaller rate.  The ``-beta_p**2 / 2`` term gives
    ``E[lambda_eff_j] = lambda_0_j`` when ``Z_phy_j ~ N(0, 1)``; it does not
    preserve the marginal task-failure probability.  In particular, ``Z_phy_j
    = 0`` does not generally recover ``lambda_0_j`` when ``beta_p > 0``.

    This is a pure numerical mapping.  It does not read or modify ``Server``,
    task, or simulator state.
    """
    base_rates = np.asarray(base_failure_rates, dtype=float)
    risk = np.asarray(spatial_risk_field, dtype=float)
    beta = float(beta_p)
    return base_rates * np.exp(beta * risk - 0.5 * beta ** 2)
