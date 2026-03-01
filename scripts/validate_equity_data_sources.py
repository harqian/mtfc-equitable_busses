#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import build_sf_equity_data_bundle, load_census_tract_geometries, load_epc_tracts_geojson


@dataclass(frozen=True)
class ValidationSummary:
    live_sf_epc_tracts: int
    final_sf_tracts: int
    final_epc_tracts: int
    final_non_epc_tracts: int
    sf_population_rows: int
    sf_population_matches_geometry: int
    sf_population_missing_geometry: int
    sf_tract_geometry_rows: int
    live_mtc_population_field: str
    live_epc_matches_baseline: int
    live_epc_unmatched_baseline: int
    bundle_population_note: str
    bundle_unmatched_epc_note: str
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
    live_epc = load_epc_tracts_geojson()
    live_sf_epc = live_epc.loc[live_epc["geoid"].astype(str).str.startswith("06075")].copy()
    equity_bundle = build_sf_equity_data_bundle()
    sf_tracts_with_equity = equity_bundle.sf_epc_tracts.copy()

    population = load_population_table(population_csv)
    tracts = load_census_tract_geometries()
    sf_tracts = tracts.loc[tracts["COUNTYFP"] == "075", ["GEOID", "geometry"]].copy()
    if sf_tracts.crs != "EPSG:4326":
        sf_tracts = sf_tracts.to_crs("EPSG:4326")

    population_join = sf_tracts.merge(
        population,
        left_on="GEOID",
        right_on="geoid",
        how="left",
        indicator=True,
    )
    sf_population_matches_geometry = int((population_join["_merge"] == "both").sum())
    if sf_population_matches_geometry == 0:
        raise ValueError("No San Francisco census tracts matched the local population CSV by GEOID")

    live_epc_matches_baseline = int(live_sf_epc["geoid"].isin(sf_tracts_with_equity["geoid"]).sum())
    if live_epc_matches_baseline == 0:
        raise ValueError("No live San Francisco EPC GEOIDs matched the final tract baseline")

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
        sf_tracts_with_equity.loc[:, ["geoid", "epc_2050", "epc_class"]],
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
        live_sf_epc_tracts=int(len(live_sf_epc)),
        final_sf_tracts=int(len(sf_tracts_with_equity)),
        final_epc_tracts=int((sf_tracts_with_equity["epc_2050"] == 1).sum()),
        final_non_epc_tracts=int((sf_tracts_with_equity["epc_2050"] == 0).sum()),
        sf_population_rows=int(len(population)),
        sf_population_matches_geometry=sf_population_matches_geometry,
        sf_population_missing_geometry=int(len(sf_tracts) - sf_population_matches_geometry),
        sf_tract_geometry_rows=int(len(sf_tracts)),
        live_mtc_population_field=next(
            (
                note.split("'")[1]
                for note in equity_bundle.notes
                if "live MTC EPC field" in note and "'" in note
            ),
            "unknown",
        ),
        live_epc_matches_baseline=live_epc_matches_baseline,
        live_epc_unmatched_baseline=int(len(live_sf_epc) - live_epc_matches_baseline),
        bundle_population_note=equity_bundle.notes[0],
        bundle_unmatched_epc_note=equity_bundle.notes[-1],
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
    print(f"- live SF EPC tracts from MTC feed: {summary.live_sf_epc_tracts}")
    print(f"- final SF tracts in equity bundle: {summary.final_sf_tracts}")
    print(f"- final EPC tracts in equity bundle: {summary.final_epc_tracts}")
    print(f"- final non-EPC tracts in equity bundle: {summary.final_non_epc_tracts}")
    print(f"- live MTC population field available in EPC feed: {summary.live_mtc_population_field}")
    print(f"- population source behavior: {summary.bundle_population_note}")
    print(f"- local SF population rows: {summary.sf_population_rows}")
    print(f"- SF tract geometries matched to local population: {summary.sf_population_matches_geometry}")
    print(f"- SF tract geometries missing local population: {summary.sf_population_missing_geometry}")
    print(f"- SF tract geometries from Census TIGER: {summary.sf_tract_geometry_rows}")
    print(f"- live SF EPC GEOIDs matched into final baseline: {summary.live_epc_matches_baseline}")
    print(f"- live SF EPC GEOIDs excluded from final baseline: {summary.live_epc_unmatched_baseline}")
    print(f"- EPC overlay note: {summary.bundle_unmatched_epc_note}")
    print(f"- route-stop rows with coordinates: {summary.stop_rows}")
    print(f"- unique bus stops tested: {summary.unique_stops}")
    print(f"- stops matched to SF census tracts: {summary.stops_with_tract_match}")
    print(f"- tract-matched stops joined to EPC labels: {summary.stops_with_epc_match}")
    print(f"- unique stops inside EPC 2050 tracts: {summary.epc_stop_count}")
    print(f"- unique stops outside EPC 2050 tracts: {summary.non_epc_stop_count}")
    print(f"- unique routes touching EPC 2050 tracts: {summary.unique_routes_touching_epc}")
    if summary.live_epc_unmatched_baseline > 0:
        print(
            "- warning: the live EPC feed is usable, but some live EPC GEOIDs do not align with the "
            "current SF tract geometry/population baseline"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
