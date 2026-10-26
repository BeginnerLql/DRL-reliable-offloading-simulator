"""Pure geographic distance and latent physical risk-correlation utilities.

The exponential kernel implemented here is

    R_jk = exp(-d_jk / ell_phy)

where ``d_jk`` is the Haversine distance in kilometres and ``ell_phy`` is a
nonnegative correlation length in kilometres (zero means independent nodes).
This represents latent physical environmental risk correlation, not failure-event Pearson
correlation, a failure probability, or a joint failure probability.

Gauss-Hermite integration below marginalizes the unknown field, assuming
conditionally independent node failures during CPU execution only. No other
common-cause failures are added. Nothing here mutates simulator state.
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
    """Build ``exp(-D / ell)``; ell=0 gives independent node risks."""
    correlation_length = float(correlation_length_km)
    if not np.isfinite(correlation_length) or correlation_length < 0:
        raise ValueError("correlation_length_km must be finite and nonnegative")
    matrix = np.asarray(distance_matrix, dtype=float)
    if correlation_length == 0:
        return np.eye(matrix.shape[0], dtype=float)
    return np.exp(-matrix / correlation_length)


def compute_baseline_correlation_length_km(distance_matrix: object) -> float:
    """Return median distinct-pair distance / ln(2), in kilometres.

    Only the strict upper triangle is used, so diagonal zeros and duplicate
    symmetric entries do not contribute. Correlation at this median is 0.5.
    """
    matrix = np.asarray(distance_matrix, dtype=float)
    pair_distances = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return float(np.median(pair_distances) / np.log(2.0))


def sample_spatial_risk_field(
    correlation_matrix: object,
    rng: np.random.Generator | None = None,
    *,
    standard_normal: object | None = None,
) -> np.ndarray:
    """Sample one latent standardized physical environmental risk field.

    The returned ``Z_phy`` follows ``N(0, R)``: each component has expected
    value zero and variance one, and the component correlation is ``R``.  The
    field is a latent physical environmental risk field, not a failure field
    and not a failure-event or failure-probability sample.

    Supply an RNG or an explicit standard-normal vector epsilon. The result is
    L @ epsilon, where L is the lower Cholesky factor of the correlation matrix.
    Reusing epsilon with the identity matrix gives the paired independent field.
    When epsilon is supplied, the RNG is not consumed.
    """
    matrix = np.asarray(correlation_matrix, dtype=float)
    factor = np.linalg.cholesky(matrix)
    epsilon = (
        rng.standard_normal(matrix.shape[0])
        if standard_normal is None
        else np.asarray(standard_normal, dtype=float)
    )
    return factor @ epsilon


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


def _normal_gh_rule(quadrature_order: int, beta_p: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert physicists' Hermite nodes/weights to standard-normal expectation."""
    if isinstance(quadrature_order, (bool, np.bool_)) or not isinstance(
        quadrature_order, (int, np.integer)
    ) or quadrature_order < 1:
        raise ValueError("quadrature_order must be a positive integer")
    if not np.isfinite(beta_p) or beta_p < 0:
        raise ValueError("beta_p must be finite and nonnegative")
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def _cpu_failure_exposure(computation_demand, processing_frequency, base_failure_rate):
    """Return lambda_0 * C/f with NumPy broadcasting; units cancel."""
    demand, frequency, rate = np.broadcast_arrays(
        np.asarray(computation_demand, dtype=float),
        np.asarray(processing_frequency, dtype=float),
        np.asarray(base_failure_rate, dtype=float),
    )
    if (not all(np.all(np.isfinite(value)) for value in (demand, frequency, rate))
            or np.any(demand < 0) or np.any(frequency <= 0) or np.any(rate < 0)):
        raise ValueError("C and lambda_0 must be finite and nonnegative; f must be positive")
    return rate * (demand / frequency)


def _conditional_failure_probability(exposure, risk, beta_p):
    multiplier = map_spatial_risk_to_effective_failure_rates(1.0, risk, beta_p)
    return -np.expm1(-exposure * multiplier)


def compute_marginal_failure_probability(
    computation_demand,
    processing_frequency,
    base_failure_rate,
    beta_p: float = 0.5,
    *,
    quadrature_order: int = 32,
) -> np.ndarray:
    """Return qbar=E[1-exp(-lambda(Z)*C/f)] using one-dimensional GH.

    Inputs broadcast; demand is in cycles, frequency in cycles/s, rate in 1/s.
    Upload and queue time do not contribute to the failure exposure.
    """
    nodes, weights = _normal_gh_rule(quadrature_order, beta_p)
    exposure = _cpu_failure_exposure(
        computation_demand, processing_frequency, base_failure_rate,
    )
    failures = _conditional_failure_probability(exposure[..., None], nodes, beta_p)
    return np.average(failures, weights=weights, axis=-1)


def compute_pair_reliability_gh(
    computation_demand,
    processing_frequencies,
    base_failure_rates,
    risk_correlation,
    beta_p: float = 0.5,
    *,
    quadrature_order: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_corr, R_ind), marginalizing an unknown bivariate Gaussian field.

    Frequencies and base rates have a final axis of length two, in (m,n) order.
    Demand and rho broadcast over their leading axes. For T tasks and P pairs,
    use demand (T,1), frequencies/rates (P,2), rho (P,); outputs are (T,P).

    Set Z_m=U and Z_n=rho*U+sqrt(1-rho**2)*V for independent standard normals
    U,V. Tensor GH computes 1-E[q_m*q_n]; R_ind=1-qbar_m*qbar_n uses the SAME
    one-node marginals. Rho is latent risk correlation in [0,1], not failure
    correlation. Rho=1 is supported, including colocated distinct servers.
    """
    nodes, weights = _normal_gh_rule(quadrature_order, beta_p)
    frequencies = np.asarray(processing_frequencies, dtype=float)
    rates = np.asarray(base_failure_rates, dtype=float)
    rho = np.asarray(risk_correlation, dtype=float)
    if frequencies.ndim == 0 or rates.ndim == 0 or (
        frequencies.shape[-1] != 2 or rates.shape[-1] != 2
    ):
        raise ValueError("processing_frequencies and base_failure_rates must end in two nodes")
    if not np.all(np.isfinite(rho)) or np.any((rho < 0) | (rho > 1)):
        raise ValueError("risk_correlation must be finite and in [0,1]")
    exposure = _cpu_failure_exposure(
        np.asarray(computation_demand, dtype=float)[..., None], frequencies, rates,
    )
    exposure, rho = np.broadcast_arrays(exposure, rho[..., None])
    rho = rho[..., 0]
    failures = _conditional_failure_probability(exposure[..., None], nodes, beta_p)
    marginals = np.average(failures, weights=weights, axis=-1)
    independent = 1.0 - marginals[..., 0] * marginals[..., 1]
    risk_n = (rho[..., None, None] * nodes[:, None]
              + np.sqrt(1.0 - rho**2)[..., None, None] * nodes[None, :])
    failures_n = _conditional_failure_probability(
        exposure[..., 1, None, None], risk_n, beta_p,
    )
    joint_weights = weights[:, None] * weights[None, :]
    # Normalize the finite-precision weight sum, keeping saturated probabilities valid.
    joint_failure = np.sum(
        failures[..., 0, :, None] * failures_n * joint_weights,
        axis=(-2, -1),
    ) / joint_weights.sum()
    return 1.0 - joint_failure, independent
