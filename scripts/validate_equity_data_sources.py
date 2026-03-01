#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd

from model import build_sf_equity_data_bundle, load_census_tract_geometries


@dataclass(frozen=True)
class ValidationSummary:
    sf_epc_tracts: int
    sf_population_rows: int
    population_join_matches: int
    population_join_unmatched: int
    sf_tract_geometry_rows: int
    mtc_population_field: str
    stop_rows: int
    unique_stops: int
    stops_with_tract_match: int
    stops_with_epc_match: int
    epc_stop_count: int
    non_epc_stop_count: int
    unique_routes_touching_epc: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and validate live tract-level equity datasets against local route-stop and "
            "population files."
        )
    )
    parser.add_argument(
        "--population-csv",
        default="data/sf_population_density_tracts_2019_2023.csv",
        help="Local tract population CSV with geoid and population columns.",
    )
    parser.add_argument(
        "--route-stops-csv",
        default="data/simplified_bus_route_stops.csv",
        help="Local route-stop CSV with stop_lat and stop_lon columns.",
    )
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_population_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"geoid": str})
    _require_columns(df, {"geoid", "population"}, f"population CSV {path}")
    df["geoid"] = df["geoid"].astype(str)
    df["population"] = pd.to_numeric(df["population"], errors="raise")
    return df


def load_stop_points(path: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    stops = pd.read_csv(path, dtype={"route_id": str, "stop_id": str})
    _require_columns(stops, {"route_id", "stop_id", "stop_lat", "stop_lon"}, f"route stops CSV {path}")
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()

    unique_stops = (
        stops.loc[:, ["stop_id", "stop_lat", "stop_lon"]]
        .drop_duplicates(subset=["stop_id"])
        .reset_index(drop=True)
    )
    stop_points = gpd.GeoDataFrame(
        unique_stops,
        geometry=gpd.points_from_xy(unique_stops["stop_lon"], unique_stops["stop_lat"]),
        crs="EPSG:4326",
    )
    return stops, stop_points


def validate_sources(population_csv: str, route_stops_csv: str) -> ValidationSummary:
    equity_bundle = build_sf_equity_data_bundle()
    sf_epc = equity_bundle.sf_epc_tracts.copy()

    population = load_population_table(population_csv)
    population_join = sf_epc.merge(population, on="geoid", how="left", indicator=True)
    population_join_matches = int((population_join["_merge"] == "both").sum())
    if population_join_matches == 0:
        raise ValueError("No SF EPC tracts matched the local population CSV by GEOID")

    tracts = load_census_tract_geometries()
    sf_tracts = tracts.loc[tracts["COUNTYFP"] == "075", ["GEOID", "geometry"]].copy()
    if sf_tracts.crs != "EPSG:4326":
        sf_tracts = sf_tracts.to_crs("EPSG:4326")

    stops, stop_points = load_stop_points(route_stops_csv)
    stop_tract_join = gpd.sjoin(
        stop_points,
        sf_tracts,
        how="left",
        predicate="within",
    )
    stops_with_tract_match = int(stop_tract_join["GEOID"].notna().sum())
    if stops_with_tract_match == 0:
        raise ValueError("No bus stops were assigned to a San Francisco census tract")

    stop_epc_join = stop_tract_join.merge(
        sf_epc.loc[:, ["geoid", "epc_2050", "epc_class"]],
        left_on="GEOID",
        right_on="geoid",
        how="left",
    )
    stops_with_epc_match = int(stop_epc_join["epc_2050"].notna().sum())
    if stops_with_epc_match == 0:
        raise ValueError("No tract-matched bus stops joined to the live EPC dataset")

    epc_stop_ids = set(stop_epc_join.loc[stop_epc_join["epc_2050"] == 1, "stop_id"].astype(str))
    route_epc = stops.loc[stops["stop_id"].astype(str).isin(epc_stop_ids), ["route_id", "stop_id"]].drop_duplicates()

    return ValidationSummary(
        sf_epc_tracts=int(len(sf_epc)),
        sf_population_rows=int(len(population)),
        population_join_matches=population_join_matches,
        population_join_unmatched=int(len(sf_epc) - population_join_matches),
        sf_tract_geometry_rows=int(len(sf_tracts)),
        mtc_population_field=equity_bundle.population_field,
        stop_rows=int(len(stops)),
        unique_stops=int(len(stop_points)),
        stops_with_tract_match=stops_with_tract_match,
        stops_with_epc_match=stops_with_epc_match,
        epc_stop_count=int(len(epc_stop_ids)),
        non_epc_stop_count=int(stop_epc_join["stop_id"].nunique() - len(epc_stop_ids)),
        unique_routes_touching_epc=int(route_epc["route_id"].nunique()),
    )


def main() -> int:
    args = parse_args()
    try:
        summary = validate_sources(
            population_csv=args.population_csv,
            route_stops_csv=args.route_stops_csv,
        )
    except Exception as exc:  # pragma: no cover - direct CLI failure path
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    print("Equity data validation succeeded")
    print(f"- live SF EPC tracts: {summary.sf_epc_tracts}")
    print(f"- MTC population field used for equity weighting: {summary.mtc_population_field}")
    print(f"- local SF population rows: {summary.sf_population_rows}")
    print(f"- EPC tracts matched to local population: {summary.population_join_matches}")
    print(f"- EPC tracts missing from local population: {summary.population_join_unmatched}")
    print(f"- SF tract geometries from Census TIGER: {summary.sf_tract_geometry_rows}")
    print(f"- route-stop rows with coordinates: {summary.stop_rows}")
    print(f"- unique bus stops tested: {summary.unique_stops}")
    print(f"- stops matched to SF census tracts: {summary.stops_with_tract_match}")
    print(f"- tract-matched stops joined to EPC labels: {summary.stops_with_epc_match}")
    print(f"- unique stops inside EPC 2050 tracts: {summary.epc_stop_count}")
    print(f"- unique stops outside EPC 2050 tracts: {summary.non_epc_stop_count}")
    print(f"- unique routes touching EPC 2050 tracts: {summary.unique_routes_touching_epc}")
    if summary.population_join_unmatched > 0:
        print(
            "- warning: the live EPC feed is usable, but the local population CSV does not cover every "
            "live SF EPC tract GEOID"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
