"""Prepare a deterministic geographic ordering of EUA Melbourne CBD sites.

The EUA dataset provides real base-station geographic locations.  This tool
currently keeps only records whose ``SITE_PRECISION`` is ``Within 10 meters``
and preserves nearby/co-located sites without merging them.  The retained
sites are ordered by a centroid-nearest seed followed by farthest-point
sampling using Haversine distance.  The ordering supports nested experiments:
``S_N`` is the first ``N`` sites in the generated order.

This step prepares geographic topology only.  It does not generate servers,
model failure correlation, or modify the simulator's execution/training logic.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Sequence

import pandas as pd


EARTH_RADIUS_KM = 6371.0088
PRECISION_FILTER = "Within 10 meters"
EXPECTED_HIGH_PRECISION_COUNT = 99
DISTANCE_TIE_TOLERANCE_KM = 1e-9
REQUIRED_INPUT_COLUMNS = {
    "SITE_ID",
    "LATITUDE",
    "LONGITUDE",
    "NAME",
    "SITE_PRECISION",
}
OUTPUT_COLUMNS = [
    "Rank",
    "Site_ID",
    "Latitude",
    "Longitude",
    "Name",
    "Site_Precision",
]


class TopologyValidationError(ValueError):
    """Raised when the filtered EUA site records are invalid."""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two coordinates in kilometres."""
    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    hav = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2.0) ** 2
    )
    # Floating-point roundoff can move hav just outside [0, 1].
    hav = min(1.0, max(0.0, hav))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(hav))


def _site_id_sort_key(site_id: object) -> tuple:
    """Build a deterministic ascending key, numeric when IDs are numeric."""
    text = str(site_id).strip()
    try:
        return (0, float(text), text)
    except (TypeError, ValueError):
        return (1, text)


def _validate_and_filter_sites(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Filter high-precision sites and validate fields used by topology."""
    cleaned_columns = [str(column).strip() for column in raw_df.columns]
    raw_df = raw_df.copy()
    raw_df.columns = cleaned_columns

    missing_columns = sorted(REQUIRED_INPUT_COLUMNS.difference(raw_df.columns))
    if missing_columns:
        raise TopologyValidationError(
            "Input CSV is missing required columns: " + ", ".join(missing_columns)
        )

    original_count = len(raw_df)
    precision = raw_df["SITE_PRECISION"].astype("string").str.strip()
    selected = raw_df.loc[precision == PRECISION_FILTER].copy()

    selected["SITE_ID"] = selected["SITE_ID"].astype("string").str.strip()
    if selected["SITE_ID"].isna().any() or (selected["SITE_ID"] == "").any():
        raise TopologyValidationError(
            "Filtered sites contain an empty SITE_ID."
        )

    for column in ("LATITUDE", "LONGITUDE"):
        if selected[column].isna().any():
            raise TopologyValidationError(
                f"Filtered sites contain an empty {column}."
            )
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        if selected[column].isna().any() or not selected[column].map(math.isfinite).all():
            raise TopologyValidationError(
                f"Filtered sites contain a non-numeric or non-finite {column}."
            )

    if selected["SITE_ID"].duplicated().any():
        duplicates = selected.loc[
            selected["SITE_ID"].duplicated(keep=False), "SITE_ID"
        ].tolist()
        raise TopologyValidationError(
            "Filtered sites contain duplicate SITE_ID values: "
            + ", ".join(map(str, sorted(set(duplicates), key=_site_id_sort_key)))
        )

    selected["SITE_PRECISION"] = PRECISION_FILTER
    return selected, original_count


def _distance_to_centroid(row: pd.Series, centroid_lat: float, centroid_lon: float) -> float:
    return haversine_km(
        row["LATITUDE"], row["LONGITUDE"], centroid_lat, centroid_lon
    )


def farthest_point_order(candidates: pd.DataFrame) -> pd.DataFrame:
    """Order candidates by centroid seed followed by farthest-point sampling."""
    if candidates.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    records = candidates.reset_index(drop=True).to_dict("records")
    centroid_lat = math.fsum(float(record["LATITUDE"]) for record in records) / len(records)
    centroid_lon = math.fsum(float(record["LONGITUDE"]) for record in records) / len(records)

    centroid_distances = [
        _distance_to_centroid(
            pd.Series(record), centroid_lat, centroid_lon
        )
        for record in records
    ]
    nearest_centroid_distance = min(centroid_distances)
    nearest_centroid_indices = [
        index
        for index, distance in enumerate(centroid_distances)
        if math.isclose(
            distance,
            nearest_centroid_distance,
            rel_tol=0.0,
            abs_tol=DISTANCE_TIE_TOLERANCE_KM,
        )
    ]
    first_index = min(
        nearest_centroid_indices,
        key=lambda index: _site_id_sort_key(records[index]["SITE_ID"]),
    )

    selected_indices = [first_index]
    remaining_indices = set(range(len(records))) - {first_index}

    while remaining_indices:
        def min_distance_to_selected(index: int) -> float:
            point = records[index]
            return min(
                haversine_km(
                    point["LATITUDE"],
                    point["LONGITUDE"],
                    records[selected_index]["LATITUDE"],
                    records[selected_index]["LONGITUDE"],
                )
                for selected_index in selected_indices
            )

        distances = {
            index: min_distance_to_selected(index)
            for index in remaining_indices
        }
        farthest_distance = max(distances.values())
        farthest_indices = [
            index
            for index, distance in distances.items()
            if math.isclose(
                distance,
                farthest_distance,
                rel_tol=0.0,
                abs_tol=DISTANCE_TIE_TOLERANCE_KM,
            )
        ]
        next_index = min(
            farthest_indices,
            key=lambda index: _site_id_sort_key(records[index]["SITE_ID"]),
        )
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)

    ordered = pd.DataFrame([records[index] for index in selected_indices])
    ordered = ordered.rename(columns={
        "SITE_ID": "Site_ID",
        "LATITUDE": "Latitude",
        "LONGITUDE": "Longitude",
        "NAME": "Name",
        "SITE_PRECISION": "Site_Precision",
    })
    ordered.insert(0, "Rank", range(1, len(ordered) + 1))
    return ordered[OUTPUT_COLUMNS]


def pairwise_distance_extrema(topology_df: pd.DataFrame) -> tuple[float, float]:
    """Return minimum and maximum pairwise Haversine distances in kilometres."""
    coordinates = list(zip(topology_df["Latitude"], topology_df["Longitude"]))
    distances = [
        haversine_km(lat1, lon1, lat2, lon2)
        for index, (lat1, lon1) in enumerate(coordinates)
        for lat2, lon2 in coordinates[index + 1 :]
    ]
    if not distances:
        return 0.0, 0.0
    return min(distances), max(distances)


def prepare_topology(input_path: Path | str, output_path: Path | str) -> pd.DataFrame:
    """Read, validate, order, and write the filtered EUA topology."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from the input CSV path.")
    raw_df = pd.read_csv(input_path, encoding="utf-8-sig")
    candidates, original_count = _validate_and_filter_sites(raw_df)

    high_precision_count = len(candidates)
    if high_precision_count != EXPECTED_HIGH_PRECISION_COUNT:
        warnings.warn(
            "Expected 99 Within-10m sites, "
            f"but found {high_precision_count}; writing the actual records.",
            UserWarning,
            stacklevel=2,
        )

    topology_df = farthest_point_order(candidates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topology_df.to_csv(output_path, index=False, lineterminator="\n")

    min_distance, max_distance = pairwise_distance_extrema(topology_df)
    first = topology_df.iloc[0] if not topology_df.empty else None
    first_site = first["Site_ID"] if first is not None else "None"
    print(f"Original sites: {original_count}")
    print(f"Within-10m sites: {high_precision_count}")
    print(f"Selected sites: {len(topology_df)}")
    print(f"First selected SITE_ID: {first_site}")
    if first is not None:
        print(
            "First selected coordinates: "
            f"{float(first['Latitude']):.8f}, {float(first['Longitude']):.8f}"
        )
    print(f"Minimum pairwise distance: {min_distance:.6f} km")
    print(f"Maximum pairwise distance: {max_distance:.6f} km")
    print(f"Output: {output_path}")
    return topology_df


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "eua_melbourne_cbd_site_order.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic EUA Melbourne CBD site ordering."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the EUA site-optus-melbCBD.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="Output CSV path (default: data/eua_melbourne_cbd_site_order.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_topology(args.input, args.output)


if __name__ == "__main__":
    main()
