#!/usr/bin/env -S uv run --with pandas
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from data_utils import find_data_file

EARTH_RADIUS_M = 6_371_000.0


def _great_circle_distance_argmin(
    stop_lat_deg: float,
    stop_lon_deg: float,
    intersection_lats_rad,
    intersection_lons_rad,
) -> int:
    stop_lat_rad = np.radians(float(stop_lat_deg))
    stop_lon_rad = np.radians(float(stop_lon_deg))

    dlat = intersection_lats_rad - stop_lat_rad
    dlon = intersection_lons_rad - stop_lon_rad
    a = (np.sin(dlat / 2.0) ** 2) + (
        np.cos(stop_lat_rad) * np.cos(intersection_lats_rad) * (np.sin(dlon / 2.0) ** 2)
    )
    return int(np.argmin(a))


def snap_stops_to_intersections(stops: pd.DataFrame, intersections: pd.DataFrame) -> pd.DataFrame:
    required_stop_cols = {"stop_lat", "stop_lon"}
    required_intersection_cols = {"Latitude", "Longitude"}

    missing_stops = required_stop_cols - set(stops.columns)
    missing_intersections = required_intersection_cols - set(intersections.columns)

    if missing_stops:
        raise ValueError(f"Stops file missing required columns: {sorted(missing_stops)}")
    if missing_intersections:
        raise ValueError(
            f"Intersections file missing required columns: {sorted(missing_intersections)}"
        )

    out = stops.copy()
    out["stop_lat"] = pd.to_numeric(out["stop_lat"], errors="coerce")
    out["stop_lon"] = pd.to_numeric(out["stop_lon"], errors="coerce")

    intersections = intersections.copy()
    intersections["Latitude"] = pd.to_numeric(intersections["Latitude"], errors="coerce")
    intersections["Longitude"] = pd.to_numeric(intersections["Longitude"], errors="coerce")
    intersections = intersections.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    if intersections.empty:
        raise ValueError("No valid intersection coordinates available.")

    ints_lats_rad = np.radians(intersections["Latitude"].to_numpy())
    ints_lons_rad = np.radians(intersections["Longitude"].to_numpy())

    snapped_lats: list[float | None] = []
    snapped_lons: list[float | None] = []
    snapped_dist_m: list[float | None] = []

    total = len(out)
    for idx, row in enumerate(out.itertuples(index=False), start=1):
        lat = getattr(row, "stop_lat")
        lon = getattr(row, "stop_lon")

        if pd.isna(lat) or pd.isna(lon):
            snapped_lats.append(None)
            snapped_lons.append(None)
            snapped_dist_m.append(None)
            continue

        nearest_idx = _great_circle_distance_argmin(lat, lon, ints_lats_rad, ints_lons_rad)
        nearest_lat = float(intersections.at[nearest_idx, "Latitude"])
        nearest_lon = float(intersections.at[nearest_idx, "Longitude"])

        dlat = np.radians(nearest_lat - float(lat))
        dlon = np.radians(nearest_lon - float(lon))
        lat1 = np.radians(float(lat))
        lat2 = np.radians(nearest_lat)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

        snapped_lats.append(nearest_lat)
        snapped_lons.append(nearest_lon)
        snapped_dist_m.append(float(EARTH_RADIUS_M * c))

        if idx % 500 == 0 or idx == total:
            print(f"Snapped stops: {idx}/{total}")

    out["stop_lat"] = snapped_lats
    out["stop_lon"] = snapped_lons
    out["snapped_distance_m"] = snapped_dist_m
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snap stop coordinates in stops.txt to nearest intersections from intersections.csv"
    )
    parser.add_argument(
        "--stops",
        default="stops.txt",
        help="Input stops file name or path (default: stops.txt)",
    )
    parser.add_argument(
        "--intersections",
        default="intersections.csv",
        help="Input intersections CSV name or path (default: intersections.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to overwriting the stops file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stops_arg = pathlib.Path(args.stops)
    intersections_arg = pathlib.Path(args.intersections)

    stops_path = stops_arg if stops_arg.exists() else find_data_file(args.stops)
    intersections_path = (
        intersections_arg if intersections_arg.exists() else find_data_file(args.intersections)
    )
    output_path = pathlib.Path(args.output) if args.output else stops_path

    stops = pd.read_csv(stops_path)
    intersections = pd.read_csv(intersections_path)

    print(f"Loaded {len(stops)} stops from {stops_path}")
    print(f"Loaded {len(intersections)} intersections from {intersections_path}")

    snapped = snap_stops_to_intersections(stops, intersections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapped.to_csv(output_path, index=False)
    print(f"Wrote snapped stops file: {output_path.resolve()}")


if __name__ == "__main__":
    main()
