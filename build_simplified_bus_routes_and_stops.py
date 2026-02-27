#!/usr/bin/env -S uv run --with pandas
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from data_utils import find_data_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build simplified bus-only route/stop CSVs from snapped stops and GTFS files, "
            "including weekday/weekend planned trips and typical buses per route."
        )
    )
    parser.add_argument("--snapped-stops", default="data/snapped_stops.txt")
    parser.add_argument("--intersections", default="data/intersections.csv")
    parser.add_argument("--routes", default="routes.txt")
    parser.add_argument("--trips", default="trips.txt")
    parser.add_argument("--stop-times", default="stop_times.txt")
    parser.add_argument(
        "--line-summary",
        default="data/sfmta_data/line_weekday_weekend_summary.csv",
    )
    parser.add_argument(
        "--route-daily",
        default="data/sfmta_data/route_daily_service.csv",
    )
    parser.add_argument(
        "--daily-summary",
        default="data/sfmta_data/daily_service_summary.csv",
    )
    parser.add_argument(
        "--out-route-stops",
        default="data/simplified_bus_route_stops.csv",
        help="Output CSV with one row per route-stop.",
    )
    parser.add_argument(
        "--out-routes",
        default="data/simplified_bus_routes.csv",
        help="Output CSV with one row per route and aggregated ordered stops.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> pathlib.Path:
    as_path = pathlib.Path(value)
    if as_path.exists():
        return as_path
    return find_data_file(value)


def mode_int(values: pd.Series) -> int:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(int)
    if clean.empty:
        return 0
    modes = clean.mode()
    return int(modes.iloc[0]) if not modes.empty else 0


def add_elevation_to_snapped_stops(snapped: pd.DataFrame, intersections: pd.DataFrame) -> pd.DataFrame:
    out = snapped.copy()
    out["stop_lat"] = pd.to_numeric(out["stop_lat"], errors="coerce")
    out["stop_lon"] = pd.to_numeric(out["stop_lon"], errors="coerce")
    out["elevation_m"] = pd.NA

    ints = intersections.copy()
    ints["Latitude"] = pd.to_numeric(ints["Latitude"], errors="coerce")
    ints["Longitude"] = pd.to_numeric(ints["Longitude"], errors="coerce")
    ints["elevation_m"] = pd.to_numeric(ints["elevation_m"], errors="coerce")
    ints = ints.dropna(subset=["Latitude", "Longitude", "elevation_m"])

    for precision in (8, 7, 6):
        missing_mask = out["elevation_m"].isna()
        if not missing_mask.any():
            break

        lookup = (
            ints.assign(
                lat_r=ints["Latitude"].round(precision),
                lon_r=ints["Longitude"].round(precision),
            )
            .drop_duplicates(subset=["lat_r", "lon_r"], keep="first")
            .set_index(["lat_r", "lon_r"])["elevation_m"]
        )

        keys = list(
            zip(
                out.loc[missing_mask, "stop_lat"].round(precision),
                out.loc[missing_mask, "stop_lon"].round(precision),
            )
        )
        mapped = pd.Series(keys).map(lookup)
        out.loc[missing_mask, "elevation_m"] = mapped.values

    return out


def build_route_stop_table(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    snapped_stops_with_elev: pd.DataFrame,
) -> pd.DataFrame:
    bus_routes = routes.loc[routes["route_type"].astype(str).str.strip() == "3"].copy()
    bus_route_ids = set(bus_routes["route_id"].astype(str))

    bus_trips = trips.loc[
        trips["route_id"].astype(str).isin(bus_route_ids),
        ["trip_id", "route_id", "direction_id", "shape_id"],
    ].copy()
    bus_trips["direction_id"] = bus_trips["direction_id"].fillna("0").astype(str)
    bus_trips["shape_id"] = bus_trips["shape_id"].fillna("").astype(str)

    merged = bus_trips.merge(stop_times[["trip_id", "stop_id", "stop_sequence"]], on="trip_id", how="inner")
    merged["stop_sequence"] = pd.to_numeric(merged["stop_sequence"], errors="coerce")
    merged = merged.dropna(subset=["stop_sequence"])

    trip_stats = (
        merged.groupby(["route_id", "direction_id", "trip_id", "shape_id"], as_index=False)
        .agg(stop_count=("stop_id", "nunique"), max_stop_sequence=("stop_sequence", "max"))
        .sort_values(
            ["route_id", "direction_id", "stop_count", "max_stop_sequence", "trip_id"],
            ascending=[True, True, False, False, True],
        )
    )

    shape_rank = (
        trip_stats.groupby(["route_id", "direction_id", "shape_id"], as_index=False)
        .agg(shape_trip_count=("trip_id", "nunique"), shape_stop_count_mean=("stop_count", "mean"))
        .sort_values(
            ["route_id", "direction_id", "shape_trip_count", "shape_stop_count_mean", "shape_id"],
            ascending=[True, True, False, False, True],
        )
    )
    canonical_shapes = (
        shape_rank.drop_duplicates(subset=["route_id", "direction_id"], keep="first")
        .loc[:, ["route_id", "direction_id", "shape_id"]]
        .rename(columns={"shape_id": "canonical_shape_id"})
    )

    canonical_trips = (
        trip_stats.merge(
            canonical_shapes,
            left_on=["route_id", "direction_id", "shape_id"],
            right_on=["route_id", "direction_id", "canonical_shape_id"],
            how="inner",
        )
        .sort_values(
            ["route_id", "direction_id", "stop_count", "max_stop_sequence", "trip_id"],
            ascending=[True, True, False, False, True],
        )
        .drop_duplicates(subset=["route_id", "direction_id"], keep="first")
        .loc[:, ["route_id", "direction_id", "trip_id"]]
        .rename(columns={"trip_id": "canonical_trip_id"})
    )

    canonical_stops = merged.merge(
        canonical_trips,
        left_on=["route_id", "direction_id", "trip_id"],
        right_on=["route_id", "direction_id", "canonical_trip_id"],
        how="inner",
    )

    route_stops = (
        canonical_stops.groupby(["route_id", "direction_id", "stop_id"], as_index=False)
        .agg(route_stop_order=("stop_sequence", "min"))
        .sort_values(["route_id", "direction_id", "route_stop_order", "stop_id"])
        .reset_index(drop=True)
    )

    route_stops = route_stops.merge(canonical_trips, on=["route_id", "direction_id"], how="left")

    route_stops = route_stops.merge(
        bus_routes[["route_id", "route_short_name", "route_long_name"]],
        on="route_id",
        how="left",
    )

    route_stops = route_stops.merge(
        snapped_stops_with_elev[
            ["stop_id", "stop_name", "stop_lat", "stop_lon", "elevation_m", "snapped_distance_m"]
        ],
        on="stop_id",
        how="left",
    )

    route_stops = route_stops.sort_values(
        ["route_id", "direction_id", "route_stop_order", "stop_id"]
    ).reset_index(drop=True)

    return route_stops[
        [
            "route_id",
            "direction_id",
            "canonical_trip_id",
            "route_short_name",
            "route_long_name",
            "stop_id",
            "stop_name",
            "route_stop_order",
            "stop_lat",
            "stop_lon",
            "elevation_m",
            "snapped_distance_m",
        ]
    ]


def build_route_service_metrics(
    routes: pd.DataFrame,
    line_summary: pd.DataFrame,
    route_daily: pd.DataFrame,
) -> pd.DataFrame:
    bus_routes = routes.loc[routes["route_type"].astype(str).str.strip() == "3"].copy()
    bus_routes = bus_routes[["route_id", "route_short_name", "route_long_name"]]

    line_bus = line_summary.copy()
    line_bus = line_bus.loc[line_bus["route_id"].astype(str).isin(set(bus_routes["route_id"].astype(str)))]
    line_bus = line_bus[
        [
            "route_id",
            "weekday_typical_runs",
            "weekend_typical_runs",
        ]
    ].rename(
        columns={
            "weekday_typical_runs": "weekday_planned_trips",
            "weekend_typical_runs": "weekend_planned_trips",
        }
    )

    per_day = route_daily.copy()
    per_day["date"] = pd.to_datetime(per_day["date"], errors="coerce")
    per_day = per_day.dropna(subset=["date"])
    per_day["day_type"] = per_day["date"].dt.weekday.map(lambda d: "weekend" if d >= 5 else "weekday")
    per_day = per_day.loc[per_day["route_id"].astype(str).isin(set(bus_routes["route_id"].astype(str)))]

    buses_typical = (
        per_day.groupby(["route_id", "day_type"], as_index=False)["route_estimated_buses"]
        .agg(mode_int)
        .rename(columns={"route_estimated_buses": "typical_buses"})
    )
    buses_typical = buses_typical.pivot(index="route_id", columns="day_type", values="typical_buses").reset_index()
    buses_typical = buses_typical.rename(
        columns={"weekday": "weekday_typical_buses", "weekend": "weekend_typical_buses"}
    )

    out = bus_routes.merge(line_bus, on="route_id", how="left").merge(buses_typical, on="route_id", how="left")
    for col in [
        "weekday_planned_trips",
        "weekend_planned_trips",
        "weekday_typical_buses",
        "weekend_typical_buses",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    return out


def build_routes_aggregates(route_stops: pd.DataFrame) -> pd.DataFrame:
    ordered = route_stops.sort_values(["route_id", "direction_id", "route_stop_order", "stop_id"]).copy()

    def join_values(s: pd.Series) -> str:
        return "|".join([str(v) for v in s if pd.notna(v)])

    def join_locations(df: pd.DataFrame) -> str:
        vals: list[str] = []
        for row in df.itertuples(index=False):
            if pd.notna(row.stop_lat) and pd.notna(row.stop_lon) and pd.notna(row.elevation_m):
                vals.append(f"{float(row.stop_lat):.8f},{float(row.stop_lon):.8f},{float(row.elevation_m):.3f}")
        return "|".join(vals)

    agg_base = (
        ordered.groupby("route_id", as_index=False)
        .agg(
            stop_count=("stop_id", "nunique"),
            ordered_stop_ids=("stop_id", join_values),
            ordered_stop_names=("stop_name", join_values),
        )
        .reset_index(drop=True)
    )

    loc_rows = []
    for route_id, group in ordered.groupby("route_id", sort=False):
        loc_rows.append({"route_id": route_id, "ordered_stop_locations_lat_lon_elevation": join_locations(group)})
    loc_df = pd.DataFrame(loc_rows)

    return agg_base.merge(loc_df, on="route_id", how="left")


def main() -> None:
    args = parse_args()

    snapped_path = resolve_path(args.snapped_stops)
    intersections_path = resolve_path(args.intersections)
    routes_path = resolve_path(args.routes)
    trips_path = resolve_path(args.trips)
    stop_times_path = resolve_path(args.stop_times)
    line_summary_path = resolve_path(args.line_summary)
    route_daily_path = resolve_path(args.route_daily)
    daily_summary_path = resolve_path(args.daily_summary)

    snapped = pd.read_csv(snapped_path, dtype={"stop_id": str})
    intersections = pd.read_csv(intersections_path)
    routes = pd.read_csv(routes_path, dtype={"route_id": str, "route_type": str})
    trips = pd.read_csv(trips_path, dtype={"trip_id": str, "route_id": str})
    stop_times = pd.read_csv(
        stop_times_path,
        usecols=["trip_id", "stop_id", "stop_sequence"],
        dtype={"trip_id": str, "stop_id": str},
    )
    line_summary = pd.read_csv(line_summary_path, dtype={"route_id": str})
    route_daily = pd.read_csv(route_daily_path, dtype={"route_id": str})
    daily_summary = pd.read_csv(daily_summary_path)

    snapped_with_elev = add_elevation_to_snapped_stops(snapped, intersections)
    route_stops = build_route_stop_table(routes, trips, stop_times, snapped_with_elev)
    route_service = build_route_service_metrics(routes, line_summary, route_daily)
    route_agg = build_routes_aggregates(route_stops)

    simplified_routes = route_service.merge(route_agg, on="route_id", how="left")
    simplified_routes = simplified_routes.sort_values(["route_id"]).reset_index(drop=True)

    out_route_stops = pathlib.Path(args.out_route_stops)
    out_routes = pathlib.Path(args.out_routes)
    out_route_stops.parent.mkdir(parents=True, exist_ok=True)
    out_routes.parent.mkdir(parents=True, exist_ok=True)

    route_stops.to_csv(out_route_stops, index=False)
    simplified_routes.to_csv(out_routes, index=False)

    weekday_mode_system_buses = mode_int(
        daily_summary.loc[
            pd.to_datetime(daily_summary["date"]).dt.weekday < 5, "estimated_buses_systemwide"
        ]
    )
    weekend_mode_system_buses = mode_int(
        daily_summary.loc[
            pd.to_datetime(daily_summary["date"]).dt.weekday >= 5, "estimated_buses_systemwide"
        ]
    )

    print(f"Wrote route-stop table: {out_route_stops.resolve()} ({len(route_stops)} rows)")
    print(f"Wrote route summary table: {out_routes.resolve()} ({len(simplified_routes)} rows)")
    print(
        "Systemwide typical buses from daily_service_summary.csv "
        f"(weekday/weekend): {weekday_mode_system_buses}/{weekend_mode_system_buses}"
    )


if __name__ == "__main__":
    main()
