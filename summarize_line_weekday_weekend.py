#!/usr/bin/env -S uv run --with pandas
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from data_utils import find_data_file
from summarize_sf_daily_bus_service import build_active_service_table, compute_daily_outputs

ROUTE_TYPE_LABELS = {
    "0": "Tram/Streetcar",
    "3": "Bus",
    "5": "Cable Tram",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-line weekday/weekend scheduled run counts. "
            "Includes GTFS vehicle type and notes bus model availability."
        )
    )
    parser.add_argument("--routes", default="routes.txt", help="GTFS routes file")
    parser.add_argument("--trips", default="trips.txt", help="GTFS trips file")
    parser.add_argument("--calendar", default="calendar.txt", help="GTFS calendar file")
    parser.add_argument(
        "--calendar-dates", default="calendar_dates.txt", help="GTFS calendar_dates file"
    )
    parser.add_argument(
        "--out",
        default="data/sfmta_data/line_weekday_weekend_summary.csv",
        help="Output CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    routes = pd.read_csv(find_data_file(args.routes), dtype={"route_id": str, "route_type": str})
    trips = pd.read_csv(find_data_file(args.trips), dtype={"route_id": str, "service_id": str})
    calendar = pd.read_csv(find_data_file(args.calendar), dtype={"service_id": str})
    calendar_dates = pd.read_csv(find_data_file(args.calendar_dates), dtype={"service_id": str})

    active_services = build_active_service_table(calendar, calendar_dates)
    route_daily, _ = compute_daily_outputs(routes, trips, active_services)

    route_daily["date"] = pd.to_datetime(route_daily["date"])
    route_daily["is_weekend"] = route_daily["date"].dt.weekday >= 5
    route_daily["day_type"] = route_daily["is_weekend"].map({False: "weekday", True: "weekend"})

    summary = (
        route_daily.groupby(
            ["route_id", "route_short_name", "route_long_name", "route_type", "day_type"],
            as_index=False,
        )["scheduled_trips"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    typical = (
        route_daily.groupby(
            ["route_id", "route_short_name", "route_long_name", "route_type", "day_type"],
            as_index=False,
        )["scheduled_trips"]
        .agg(lambda s: int(s.mode().iloc[0]) if not s.mode().empty else 0)
        .rename(columns={"scheduled_trips": "typical_runs"})
    )

    pivot = summary.pivot_table(
        index=["route_id", "route_short_name", "route_long_name", "route_type"],
        columns="day_type",
        values=["mean", "min", "max"],
        fill_value=0,
    )

    pivot.columns = [f"{day}_{stat}_runs" for stat, day in pivot.columns]
    pivot = pivot.reset_index()

    typical_pivot = typical.pivot_table(
        index=["route_id", "route_short_name", "route_long_name", "route_type"],
        columns="day_type",
        values="typical_runs",
        fill_value=0,
    ).reset_index()
    typical_pivot = typical_pivot.rename(
        columns={"weekday": "weekday_typical_runs", "weekend": "weekend_typical_runs"}
    )
    pivot = pivot.merge(
        typical_pivot,
        on=["route_id", "route_short_name", "route_long_name", "route_type"],
        how="left",
    )

    for col in [
        "weekday_typical_runs",
        "weekend_typical_runs",
        "weekday_mean_runs",
        "weekend_mean_runs",
        "weekday_min_runs",
        "weekday_max_runs",
        "weekend_min_runs",
        "weekend_max_runs",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["weekday_mean_runs"] = pivot["weekday_mean_runs"].round(2)
    pivot["weekend_mean_runs"] = pivot["weekend_mean_runs"].round(2)
    for col in [
        "weekday_typical_runs",
        "weekend_typical_runs",
        "weekday_min_runs",
        "weekday_max_runs",
        "weekend_min_runs",
        "weekend_max_runs",
    ]:
        pivot[col] = pivot[col].astype(int)

    pivot["vehicle_type"] = pivot["route_type"].map(ROUTE_TYPE_LABELS).fillna("Unknown")
    pivot["bus_model"] = "Not available in GTFS static files in this directory"

    output = pivot[
        [
            "route_id",
            "route_short_name",
            "route_long_name",
            "route_type",
            "vehicle_type",
            "bus_model",
            "weekday_typical_runs",
            "weekday_mean_runs",
            "weekday_min_runs",
            "weekday_max_runs",
            "weekend_typical_runs",
            "weekend_mean_runs",
            "weekend_min_runs",
            "weekend_max_runs",
        ]
    ].sort_values("route_id")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    print(f"Wrote line weekday/weekend summary: {out_path.resolve()} ({len(output)} rows)")


if __name__ == "__main__":
    main()
