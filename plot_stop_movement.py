#!/usr/bin/env -S uv run --with pandas --with folium --with numpy
from __future__ import annotations

import argparse
import pathlib

import folium
import numpy as np
import pandas as pd

from data_utils import find_data_file

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def _normalize(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    return out


def prepare_joined(original: pd.DataFrame, snapped: pd.DataFrame) -> pd.DataFrame:
    if "stop_lat" not in original.columns or "stop_lon" not in original.columns:
        raise ValueError("Original stops file must contain stop_lat and stop_lon")
    if "stop_lat" not in snapped.columns or "stop_lon" not in snapped.columns:
        raise ValueError("Snapped stops file must contain stop_lat and stop_lon")

    original = _normalize(original, "stop_lat", "stop_lon")
    snapped = _normalize(snapped, "stop_lat", "stop_lon")

    if "stop_id" in original.columns and "stop_id" in snapped.columns:
        joined = original[["stop_id", "stop_name", "stop_lat", "stop_lon"]].rename(
            columns={"stop_lat": "orig_lat", "stop_lon": "orig_lon"}
        ).merge(
            snapped[["stop_id", "stop_lat", "stop_lon"]].rename(
                columns={"stop_lat": "snap_lat", "stop_lon": "snap_lon"}
            ),
            on="stop_id",
            how="inner",
        )
    else:
        if len(original) != len(snapped):
            raise ValueError("Without stop_id, original and snapped files must have the same row count")
        joined = pd.DataFrame(
            {
                "stop_id": np.arange(len(original)),
                "stop_name": original.get("stop_name", pd.Series([""] * len(original))),
                "orig_lat": original["stop_lat"],
                "orig_lon": original["stop_lon"],
                "snap_lat": snapped["stop_lat"],
                "snap_lon": snapped["stop_lon"],
            }
        )

    joined = joined.dropna(subset=["orig_lat", "orig_lon", "snap_lat", "snap_lon"]).copy()
    joined["distance_m"] = haversine_m(
        joined["orig_lat"].to_numpy(),
        joined["orig_lon"].to_numpy(),
        joined["snap_lat"].to_numpy(),
        joined["snap_lon"].to_numpy(),
    )
    return joined


def build_map(joined: pd.DataFrame, output: pathlib.Path) -> None:
    if joined.empty:
        raise ValueError("No rows with valid original and snapped coordinates to plot")

    center_lat = float((joined["orig_lat"].mean() + joined["snap_lat"].mean()) / 2.0)
    center_lon = float((joined["orig_lon"].mean() + joined["snap_lon"].mean()) / 2.0)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    movement_layer = folium.FeatureGroup(name="Movement Lines", show=True)
    original_layer = folium.FeatureGroup(name="Original Stops", show=True)
    snapped_layer = folium.FeatureGroup(name="Snapped Stops", show=True)

    for row in joined.itertuples(index=False):
        tooltip = (
            f"stop_id={row.stop_id}"
            f" | {row.stop_name if isinstance(row.stop_name, str) else ''}"
            f" | moved={row.distance_m:.1f} m"
        )

        folium.PolyLine(
            locations=[[row.orig_lat, row.orig_lon], [row.snap_lat, row.snap_lon]],
            color="#4d4d4d",
            weight=1,
            opacity=0.55,
            tooltip=tooltip,
        ).add_to(movement_layer)

        folium.CircleMarker(
            location=[row.orig_lat, row.orig_lon],
            radius=2,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.9,
            weight=1,
            tooltip=tooltip,
        ).add_to(original_layer)

        folium.CircleMarker(
            location=[row.snap_lat, row.snap_lon],
            radius=2,
            color="#d62728",
            fill=True,
            fill_opacity=0.9,
            weight=1,
            tooltip=tooltip,
        ).add_to(snapped_layer)

    movement_layer.add_to(m)
    original_layer.add_to(m)
    snapped_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot original vs snapped stops and connecting movement lines"
    )
    parser.add_argument(
        "--original",
        default="data/sfmta_data/stops.txt",
        help="Original stops file path (default: data/sfmta_data/stops.txt)",
    )
    parser.add_argument(
        "--snapped",
        required=True,
        help="Snapped stops file path",
    )
    parser.add_argument(
        "--output",
        default="sf_stops_movement_map.html",
        help="Output HTML map path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    original_arg = pathlib.Path(args.original)
    snapped_arg = pathlib.Path(args.snapped)

    original_path = original_arg if original_arg.exists() else find_data_file(args.original)
    snapped_path = snapped_arg if snapped_arg.exists() else find_data_file(args.snapped)

    original = pd.read_csv(original_path)
    snapped = pd.read_csv(snapped_path)

    joined = prepare_joined(original, snapped)

    output = pathlib.Path(args.output)
    build_map(joined, output)

    print(f"Plotted {len(joined)} stops")
    print(f"Mean move distance: {joined['distance_m'].mean():.2f} m")
    print(f"Median move distance: {joined['distance_m'].median():.2f} m")
    print(f"Max move distance: {joined['distance_m'].max():.2f} m")
    print(f"Wrote movement map: {output.resolve()}")


if __name__ == "__main__":
    main()
