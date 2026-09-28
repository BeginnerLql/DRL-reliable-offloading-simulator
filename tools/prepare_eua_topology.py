"""Select a deterministic distance-balanced topology from EUA sites."""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.configuration import parameters


EARTH_RADIUS_KM = 6371.0088
EXPECTED_HIGH_PRECISION_COUNT = 99
TOPOLOGY_COLUMNS = ["Server_ID", "Site_ID", "Latitude", "Longitude"]
CANDIDATE_COLUMNS = ["Site_ID", "Latitude", "Longitude"]


class TopologyValidationError(ValueError):
    """Raised when EUA site records cannot form a valid topology."""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between coordinates in kilometres."""
    lat1_rad, lon1_rad = math.radians(float(lat1)), math.radians(float(lon1))
    lat2_rad, lon2_rad = math.radians(float(lat2)), math.radians(float(lon2))
    delta_lat, delta_lon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    hav = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2.0) ** 2
    )
    hav = min(1.0, max(0.0, hav))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(hav))


def _site_id_sort_key(site_id: object) -> tuple:
    """Sort numeric-looking IDs numerically, with stable textual ties."""
    text = str(site_id).strip()
    try:
        return (0, float(text), text)
    except (TypeError, ValueError):
        return (1, text)


def _normalized_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    column_names = {
        "SITE_ID": "Site_ID",
        "LATITUDE": "Latitude",
        "LONGITUDE": "Longitude",
    }
    frame = candidates.rename(columns=column_names).copy()
    required = {"Site_ID", "Latitude", "Longitude"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise TopologyValidationError(
            "Candidate sites are missing required columns: " + ", ".join(missing)
        )

    frame = frame[["Site_ID", "Latitude", "Longitude"]]
    frame["Site_ID"] = frame["Site_ID"].astype("string").str.strip()
    if frame["Site_ID"].isna().any() or frame["Site_ID"].eq("").any():
        raise TopologyValidationError("Candidate sites contain an empty Site_ID.")
    if frame["Site_ID"].duplicated().any():
        duplicates = frame.loc[frame["Site_ID"].duplicated(keep=False), "Site_ID"]
        values = sorted(set(duplicates.tolist()), key=_site_id_sort_key)
        raise TopologyValidationError(
            "Candidate sites contain duplicate Site_ID values: "
            + ", ".join(values)
        )

    for column, lower, upper in (
        ("Latitude", -90.0, 90.0),
        ("Longitude", -180.0, 180.0),
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not frame[column].map(math.isfinite).all():
            raise TopologyValidationError(
                f"Candidate sites contain a non-numeric or non-finite {column}."
            )
        if not frame[column].between(lower, upper).all():
            raise TopologyValidationError(
                f"Candidate sites contain out-of-range {column} values."
            )

    frame = frame.sort_values(
        "Site_ID", key=lambda values: values.map(_site_id_sort_key)
    ).reset_index(drop=True)
    return frame


def _validate_candidate_pool(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Validate a cleaned candidate pool before topology selection."""
    missing = sorted(set(CANDIDATE_COLUMNS).difference(candidate_df.columns))
    if missing:
        raise TopologyValidationError(
            "Candidate CSV is missing required columns: " + ", ".join(missing)
        )
    return _normalized_candidates(candidate_df)


def _distance_profile(candidates: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Build the K-by-K near/mid/far matrix using all candidate pairs."""
    coordinates = candidates[["Latitude", "Longitude"]].to_numpy(dtype=float)
    candidate_count = len(coordinates)
    distances = np.zeros((candidate_count, candidate_count), dtype=float)
    pair_distances = []
    for left in range(candidate_count):
        for right in range(left + 1, candidate_count):
            distance = haversine_km(*coordinates[left], *coordinates[right])
            distances[left, right] = distances[right, left] = distance
            pair_distances.append(distance)

    q33, q67 = np.quantile(pair_distances, [1.0 / 3.0, 2.0 / 3.0])
    bins = np.zeros_like(distances, dtype=np.int8)
    bins[distances > q33] = 1
    bins[distances > q67] = 2
    return bins, float(q33), float(q67)


def _pair_bin_counts(indices: Sequence[int], bins: np.ndarray) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    for offset, left in enumerate(indices):
        for right in indices[offset + 1 :]:
            counts[int(bins[left, right])] += 1
    return tuple(counts)


def _objective(indices: Sequence[int], bins: np.ndarray) -> tuple[float, tuple[int, int, int]]:
    counts = _pair_bin_counts(indices, bins)
    pair_count = len(indices) * (len(indices) - 1) // 2
    if sum(counts) != pair_count:
        raise RuntimeError("Distance-bin counts do not match the selected pair count.")
    target = pair_count / 3.0
    score = sum((count - target) ** 2 for count in counts)
    return score, counts


def _selected_site_key(indices: Sequence[int], candidates: pd.DataFrame) -> tuple:
    ids = candidates.iloc[list(indices)]["Site_ID"].tolist()
    return tuple(sorted((_site_id_sort_key(site_id) for site_id in ids)))


def _select_balanced_topology(
    candidates: pd.DataFrame, num_servers: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = _normalized_candidates(candidates)
    K = len(frame)
    if isinstance(num_servers, bool) or not isinstance(num_servers, (int, np.integer)):
        raise ValueError("num_servers must be an integer.")
    N = int(num_servers)
    if not 2 <= N <= K:
        raise ValueError(f"num_servers must satisfy 2 <= N <= K; got N={N}, K={K}.")

    bins, q33, q67 = _distance_profile(frame)
    coordinates = frame[["Latitude", "Longitude"]].to_numpy(dtype=float)
    centroid_lat = math.fsum(coordinates[:, 0]) / K
    centroid_lon = math.fsum(coordinates[:, 1]) / K
    center_distances = [
        haversine_km(latitude, longitude, centroid_lat, centroid_lon)
        for latitude, longitude in coordinates
    ]
    first = min(
        range(K),
        key=lambda index: (
            center_distances[index],
            _site_id_sort_key(frame.iloc[index]["Site_ID"]),
        ),
    )
    selected = [first]

    while len(selected) < N:
        remaining = [index for index in range(K) if index not in selected]
        selected.append(
            min(
                remaining,
                key=lambda candidate: (
                    _objective([*selected, candidate], bins)[0],
                    _site_id_sort_key(frame.iloc[candidate]["Site_ID"]),
                ),
            )
        )

    current_score, _ = _objective(selected, bins)
    while True:
        selected_set = set(selected)
        unselected = [index for index in range(K) if index not in selected_set]
        best = None
        for removed in selected:
            retained = [index for index in selected if index != removed]
            for added in unselected:
                proposal = [*retained, added]
                score, _ = _objective(proposal, bins)
                if score >= current_score:
                    continue
                key = _selected_site_key(proposal, frame)
                candidate = (score, key, removed, added)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None:
            break
        current_score, _, removed, added = best
        selected.remove(removed)
        selected.append(added)

    selected = sorted(selected, key=lambda index: _site_id_sort_key(frame.iloc[index]["Site_ID"]))
    selected_frame = frame.iloc[selected].reset_index(drop=True)
    selected_frame.insert(0, "Server_ID", range(1, N + 1))
    selected_frame = selected_frame[TOPOLOGY_COLUMNS]
    score, counts = _objective(selected, bins)
    selected_distances = [
        haversine_km(
            coordinates[left, 0], coordinates[left, 1],
            coordinates[right, 0], coordinates[right, 1],
        )
        for offset, left in enumerate(selected)
        for right in selected[offset + 1 :]
    ]
    summary = {
        "candidate_count": K,
        "selected_count": N,
        "q33": q33,
        "q67": q67,
        "bin_counts": counts,
        "pair_count": N * (N - 1) // 2,
        "objective": score,
        "minimum_distance": min(selected_distances),
        "maximum_distance": max(selected_distances),
    }
    return selected_frame, summary


def select_balanced_topology(candidates: pd.DataFrame, num_servers: int) -> pd.DataFrame:
    """Select N sites with balanced global near, mid, and far pair distances."""
    selected, _ = _select_balanced_topology(candidates, num_servers)
    return selected


def prepare_topology(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    num_servers: int = parameters.NUM_SERVERS,
) -> pd.DataFrame:
    """Select a topology from the cleaned candidate pool and write its CSV."""
    input_path = Path(input_path) if input_path is not None else _default_input_path()
    output_path = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(num_servers)
    )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from the candidate CSV path.")

    candidate_df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        dtype={"Site_ID": "string"},
    )
    candidates = _validate_candidate_pool(candidate_df)
    K = len(candidates)
    if K != EXPECTED_HIGH_PRECISION_COUNT:
        warnings.warn(
            "Expected about 99 candidate sites, "
            f"but found {K}; using the actual candidate count.",
            UserWarning,
            stacklevel=2,
        )

    topology_df, summary = _select_balanced_topology(candidates, num_servers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topology_df.to_csv(output_path, index=False, lineterminator="\n")

    print(f"Candidate site count K: {summary['candidate_count']}")
    print(f"Selected server count N: {summary['selected_count']}")
    print(f"Global Q33: {summary['q33']:.6f} km")
    print(f"Global Q67: {summary['q67']:.6f} km")
    near, mid, far = summary["bin_counts"]
    print(f"Selected near/mid/far pairs: {near}/{mid}/{far}")
    print(f"Total selected pair count: {summary['pair_count']}")
    print(f"Final J_N: {summary['objective']:.6f}")
    print(f"Minimum selected pair distance: {summary['minimum_distance']:.6f} km")
    print(f"Maximum selected pair distance: {summary['maximum_distance']:.6f} km")
    print(f"Input candidate path: {input_path}")
    print(f"Output path: {output_path}")
    return topology_df


def _default_input_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "eua_melbourne_cbd_candidates.csv"
    )


def _default_output_path(num_servers: int) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / f"eua_melbourne_cbd_selected_{num_servers}.csv"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a deterministic distance-balanced topology from EUA candidates."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Candidate CSV path (default: data/eua_melbourne_cbd_candidates.csv).",
    )
    parser.add_argument(
        "--num-servers", type=int, default=parameters.NUM_SERVERS,
        help=f"Selected server count (default: {parameters.NUM_SERVERS}).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: data/eua_melbourne_cbd_selected_{N}.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_topology(args.input, args.output, args.num_servers)


if __name__ == "__main__":
    main()
