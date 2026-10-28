"""Geographic correlation and reliability integrated over unknown Gaussian risk.

The exponential kernel implemented here is

    R_jk = exp(-d_jk / ell_phy)

where ``d_jk`` is the Haversine distance in kilometres and ``ell_phy`` is a
positive correlation length in kilometres.  This represents latent physical
environmental risk correlation; it is not a failure-event Pearson
correlation, a failure probability, or a joint failure probability.

Reliability uses Gauss-Hermite quadrature, without sampling a risk realization
or assigning episode-effective failure rates to servers. Replica failures are
independent conditional on the latent field; integrating it out retains their
spatial dependence. The independent comparison retains the same marginals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


def compute_baseline_correlation_length_km(distance_matrix: object) -> float:
    """Return median distinct-pair distance / ln(2), in kilometres.

    Only the strict upper triangle is used, so diagonal zeros and duplicate
    symmetric entries do not contribute. Correlation at this median is 0.5.
    """
    matrix = np.asarray(distance_matrix, dtype=float)
    pair_distances = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return float(np.median(pair_distances) / np.log(2.0))


@dataclass(frozen=True)
class PairReliability:
    """Marginal replica reliabilities and two pair reliability comparisons."""

    replica_a: float
    replica_b: float
    correlated: float
    independent: float


class GaussHermiteReliability:
    """Integrate CPU reliability over an unobserved standardized Gaussian field.

    For exposure ``a_j = lambda_0_j * C / f_j``, conditional reliability is
    ``r_j(z) = exp(-a_j * exp(beta_p*z - beta_p**2/2))``.
    Marginal reliability is E[r_j(Z)], with Z ~ N(0, 1). Pair reliability is
    ``1 - E[(1-r_a(Z_a)) * (1-r_b(Z_b))]`` for correlation rho. Independence
    sets rho=0, retaining both marginal risk distributions and beta_p.

    NumPy's Hermite weights integrate exp(-x**2); standard normal expectations
    therefore use nodes sqrt(2)*x and weights w/sqrt(pi). Nodes are fixed
    integration points, not sampled risk fields. Only CPU execution time is
    exposed to failures, as in the original reliability model.
    """

    def __init__(self, beta_p: float, order: int = 32):
        self.beta_p = float(beta_p)
        if not np.isfinite(self.beta_p) or self.beta_p < 0:
            raise ValueError("beta_p must be finite and nonnegative")
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("GH order must be a positive integer")
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        self.nodes = np.sqrt(2.0) * nodes
        self.weights = weights / np.sqrt(np.pi)
        self.pair_weights = np.outer(self.weights, self.weights)

    def _conditional_failure(self, exposure, z):
        multiplier = np.exp(self.beta_p * z - 0.5 * self.beta_p ** 2)
        # expm1 avoids subtracting two nearly equal numbers for small exposure.
        return -np.expm1(-exposure * multiplier)

    def compute_pair(self, exposure_a: float, exposure_b: float, rho: float) -> PairReliability:
        """Return reliabilities for dimensionless exposures and latent correlation.

        A symmetric transform of independent standard normal quadrature axes
        gives Z_a=c*X+s*Y, Z_b=c*X-s*Y, where c=sqrt((1+rho)/2),
        s=sqrt((1-rho)/2). Both variances are one and covariance is rho.
        This also preserves replica exchange symmetry at finite GH order.
        """
        exposure_a, exposure_b, rho = float(exposure_a), float(exposure_b), float(rho)
        if not all(np.isfinite(a) and a >= 0 for a in (exposure_a, exposure_b)):
            raise ValueError("exposures must be finite and nonnegative")
        if not np.isfinite(rho) or not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")

        failure_a = self._conditional_failure(exposure_a, self.nodes)
        failure_b = self._conditional_failure(exposure_b, self.nodes)
        mean_failure_a = float(self.weights @ failure_a)
        mean_failure_b = float(self.weights @ failure_b)
        independent = 1.0 - mean_failure_a * mean_failure_b

        if rho == 0.0:
            correlated = independent
        elif abs(rho) == 1.0:
            # A singular pair distribution is a one-dimensional expectation.
            joint_failure = self.weights @ (
                failure_a * self._conditional_failure(exposure_b, rho * self.nodes)
            )
            correlated = float(1.0 - joint_failure)
        else:
            shared = np.sqrt((1.0 + rho) / 2.0) * self.nodes[:, None]
            difference = np.sqrt((1.0 - rho) / 2.0) * self.nodes[None, :]
            joint_failure = np.sum(
                self.pair_weights
                * self._conditional_failure(exposure_a, shared + difference)
                * self._conditional_failure(exposure_b, shared - difference)
            )
            correlated = float(1.0 - joint_failure)

        return PairReliability(
            replica_a=1.0 - mean_failure_a,
            replica_b=1.0 - mean_failure_b,
            correlated=correlated,
            independent=independent,
        )
