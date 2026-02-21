#!/usr/bin/env -S uv run --with pandas --with matplotlib
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import find_data_file

EPQS_BASE_URL = "https://epqs.nationalmap.gov/v1/json"


def fetch_altitude_meters(lon: float, lat: float, timeout: float = 10.0, retries: int = 3) -> float | None:
    params = urllib.parse.urlencode(
        {
            "x": lon,
            "y": lat,
            "units": "Meters",
            "wkid": 4326,
        }
    )
    url = f"{EPQS_BASE_URL}?{params}"

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            value = payload.get("value")
            if value is None:
                return None
            return float(value)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            if attempt == retries:
                return None
            time.sleep(0.35 * attempt)
    return None


def enrich_altitudes(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    if "stop_lon" not in df.columns or "stop_lat" not in df.columns:
        raise ValueError("Input stops file must include stop_lon and stop_lat columns")

    df = df.copy()
    coord_key = list(zip(df["stop_lon"].round(6), df["stop_lat"].round(6)))
    unique_coords = sorted(set(coord_key))

    altitude_by_coord: dict[tuple[float, float], float | None] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(fetch_altitude_meters, lon, lat): (lon, lat) for lon, lat in unique_coords
        }
        total = len(future_map)
        for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            lon, lat = future_map[future]
            altitude_by_coord[(lon, lat)] = future.result()
            if idx % 250 == 0 or idx == total:
                print(f"Fetched altitudes: {idx}/{total}")

    df["elevation_m"] = [altitude_by_coord[key] for key in coord_key]
    return df


def plot_3d(df: pd.DataFrame, output_path: pathlib.Path) -> None:
    plot_df = df.dropna(subset=["stop_lon", "stop_lat", "elevation_m"])
    if plot_df.empty:
        raise ValueError("No valid rows to plot. elevation_m is empty.")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        plot_df["stop_lon"],
        plot_df["stop_lat"],
        plot_df["elevation_m"],
        c=plot_df["elevation_m"],
        cmap="terrain",
        s=7,
        alpha=0.85,
    )

    ax.set_title("San Francisco Bus Stops by Elevation")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Elevation (m)")
    fig.colorbar(scatter, ax=ax, pad=0.1, label="Elevation (m)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch stop elevations via USGS EPQS and plot SF bus stops in 3D"
    )
    parser.add_argument(
        "--input",
        default="stops.txt",
        help="Input stops CSV/GTFS file name or path (default: stops.txt)",
    )
    parser.add_argument(
        "--output-csv",
        default="data/sfmta_data/stops_with_altitude.csv",
        help="Output CSV with elevation_m column",
    )
    parser.add_argument(
        "--output-plot",
        default="sf_bus_stops_3d.png",
        help="Output image path for 3D plot",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent EPQS requests (default: 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_arg = pathlib.Path(args.input)
    input_path = input_arg if input_arg.exists() else find_data_file(args.input)

    stops = pd.read_csv(input_path)
    print(f"Loaded {len(stops)} stops from {input_path}")

    enriched = enrich_altitudes(stops, workers=max(1, args.workers))

    output_csv = pathlib.Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_csv, index=False)
    print(f"Wrote altitude-enriched stops CSV: {output_csv.resolve()}")

    output_plot = pathlib.Path(args.output_plot)
    plot_3d(enriched, output_plot)
    print(f"Wrote 3D bus-stop plot: {output_plot.resolve()}")

    missing = int(enriched["elevation_m"].isna().sum())
    if missing:
        print(f"Warning: altitude lookup failed for {missing} stops")


if __name__ == "__main__":
    main()
