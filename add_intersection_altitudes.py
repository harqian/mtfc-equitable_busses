#!/usr/bin/env -S uv run --with pandas
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from data_utils import find_data_file

EPQS_BASE_URL = "https://epqs.nationalmap.gov/v1/json"


def fetch_altitude_meters(lon: float, lat: float, timeout: float = 10.0, retries: int = 3) -> float | None:
    params = urllib.parse.urlencode({"x": lon, "y": lat, "units": "Meters", "wkid": 4326})
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
    required_cols = {"Longitude", "Latitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input intersections file is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")
    out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")

    coord_key: list[tuple[float | None, float | None]] = []
    unique_coords: set[tuple[float, float]] = set()
    for lon, lat in zip(out["Longitude"], out["Latitude"]):
        if pd.isna(lon) or pd.isna(lat):
            coord_key.append((None, None))
            continue
        rounded = (round(float(lon), 6), round(float(lat), 6))
        coord_key.append(rounded)
        unique_coords.add(rounded)

    altitude_by_coord: dict[tuple[float, float], float | None] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(fetch_altitude_meters, lon, lat): (lon, lat) for lon, lat in sorted(unique_coords)
        }
        total = len(future_map)
        for idx, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            lon, lat = future_map[future]
            altitude_by_coord[(lon, lat)] = future.result()
            if idx % 250 == 0 or idx == total:
                print(f"Fetched altitudes: {idx}/{total}")

    out["elevation_m"] = [
        altitude_by_coord.get((lon, lat)) if lon is not None and lat is not None else None
        for lon, lat in coord_key
    ]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch intersection elevations via USGS EPQS and write intersections.csv with elevation_m"
    )
    parser.add_argument(
        "--input",
        default="intersections.csv",
        help="Input intersections CSV name or path (default: intersections.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to same as --input (in-place update).",
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
    output_path = pathlib.Path(args.output) if args.output else input_path

    intersections = pd.read_csv(input_path)
    print(f"Loaded {len(intersections)} intersections from {input_path}")

    enriched = enrich_altitudes(intersections, workers=max(1, args.workers))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    print(f"Wrote altitude-enriched intersections CSV: {output_path.resolve()}")

    missing = int(enriched["elevation_m"].isna().sum())
    if missing:
        print(f"Warning: altitude lookup failed for {missing} intersections")


if __name__ == "__main__":
    main()
