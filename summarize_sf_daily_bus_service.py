#!/usr/bin/env -S uv run --with pandas
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from data_utils import find_data_file


def parse_date_yyyymmdd(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, format="%Y%m%d")


def build_active_service_table(calendar: pd.DataFrame, calendar_dates: pd.DataFrame) -> pd.DataFrame:
    weekday_col_by_idx = {
        0: "monday",
        1: "tuesday",
        2: "wednesday",
        3: "thursday",
        4: "friday",
        5: "saturday",
        6: "sunday",
    }

    cal = calendar.copy()
    cal["start_date"] = cal["start_date"].astype(str).map(parse_date_yyyymmdd)
    cal["end_date"] = cal["end_date"].astype(str).map(parse_date_yyyymmdd)

    min_date = cal["start_date"].min()
    max_date = cal["end_date"].max()
    dates = pd.date_range(min_date, max_date, freq="D")
    all_dates = pd.DataFrame({"date": dates})

    base_rows = []
    for row in cal.itertuples(index=False):
        row_dates = dates[(dates >= row.start_date) & (dates <= row.end_date)]
        if row_dates.empty:
            continue

        day_col = weekday_col_by_idx[row_dates[0].weekday()]
        allowed = []
        for dt in row_dates:
            day_col = weekday_col_by_idx[dt.weekday()]
            if int(getattr(row, day_col)) == 1:
                allowed.append(dt)

        if allowed:
            base_rows.append(
                pd.DataFrame(
                    {
                        "date": allowed,
                        "service_id": row.service_id,
                        "active": 1,
                    }
                )
            )

    if base_rows:
        base = pd.concat(base_rows, ignore_index=True)
    else:
        base = pd.DataFrame(columns=["date", "service_id", "active"])

    cd = calendar_dates.copy()
    if not cd.empty:
        cd["date"] = cd["date"].astype(str).map(parse_date_yyyymmdd)
        adds = cd.loc[cd["exception_type"] == 1, ["date", "service_id"]].copy()
        adds["active"] = 1
        removes = cd.loc[cd["exception_type"] == 2, ["date", "service_id"]].copy()
        removes["active"] = 0
    else:
        adds = pd.DataFrame(columns=["date", "service_id", "active"])
        removes = pd.DataFrame(columns=["date", "service_id", "active"])

    active = pd.concat([base, adds], ignore_index=True)
    if not removes.empty:
        remove_pairs = set(zip(removes["date"], removes["service_id"]))
        active = active[
            ~active.apply(lambda r: (r["date"], r["service_id"]) in remove_pairs, axis=1)
        ]

    active = active.drop_duplicates(subset=["date", "service_id"], keep="last")
    active = active.sort_values(["date", "service_id"]).reset_index(drop=True)

    if active.empty:
        return pd.DataFrame(columns=["date", "service_id"])

    return active[["date", "service_id"]]


def compute_daily_outputs(
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    active_services: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trips = trips.copy()
    trips["block_id"] = trips["block_id"].fillna("").astype(str).str.strip()

    trips_by_service_route = (
        trips.groupby(["service_id", "route_id"], as_index=False)
        .agg(
            scheduled_trips=("trip_id", "nunique"),
            route_estimated_buses=("block_id", lambda s: s[s != ""].nunique()),
            route_trips_missing_block=("block_id", lambda s: int((s == "").sum())),
        )
    )

    route_daily = active_services.merge(trips_by_service_route, on="service_id", how="left")
    route_daily = route_daily.dropna(subset=["route_id"])

    route_daily = route_daily.merge(
        routes[["route_id", "route_short_name", "route_long_name", "route_type"]],
        on="route_id",
        how="left",
    )

    route_daily = (
        route_daily.groupby(
            ["date", "route_id", "route_short_name", "route_long_name", "route_type"],
            as_index=False,
        )
        .agg(
            scheduled_trips=("scheduled_trips", "sum"),
            route_estimated_buses=("route_estimated_buses", "sum"),
            route_trips_missing_block=("route_trips_missing_block", "sum"),
        )
        .sort_values(["date", "route_id"])
        .reset_index(drop=True)
    )

    daily_rows = []
    trips_min = trips[["service_id", "trip_id", "block_id"]].copy()
    for dt, group in active_services.groupby("date", sort=True):
        service_ids = group["service_id"].tolist()
        day_trips = trips_min[trips_min["service_id"].isin(service_ids)]
        unique_blocks = day_trips.loc[day_trips["block_id"] != "", "block_id"].nunique()
        missing_blocks = int((day_trips["block_id"] == "").sum())
        daily_rows.append(
            {
                "date": dt,
                "active_service_ids": len(service_ids),
                "scheduled_trips_total": day_trips["trip_id"].nunique(),
                "estimated_buses_systemwide": int(unique_blocks),
                "trips_missing_block_id": missing_blocks,
            }
        )

    daily_summary = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    return route_daily, daily_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize scheduled trips per route per day and estimated daily buses"
    )
    parser.add_argument("--routes", default="routes.txt", help="GTFS routes file")
    parser.add_argument("--trips", default="trips.txt", help="GTFS trips file")
    parser.add_argument("--calendar", default="calendar.txt", help="GTFS calendar file")
    parser.add_argument(
        "--calendar-dates", default="calendar_dates.txt", help="GTFS calendar_dates file"
    )
    parser.add_argument(
        "--route-daily-out",
        default="data/sfmta_data/route_daily_service.csv",
        help="Output CSV for per-route daily stats",
    )
    parser.add_argument(
        "--daily-summary-out",
        default="data/sfmta_data/daily_service_summary.csv",
        help="Output CSV for whole-system daily stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    routes = pd.read_csv(find_data_file(args.routes), dtype={"route_id": str})
    trips = pd.read_csv(find_data_file(args.trips), dtype={"route_id": str, "service_id": str})
    calendar = pd.read_csv(find_data_file(args.calendar), dtype={"service_id": str})
    calendar_dates = pd.read_csv(find_data_file(args.calendar_dates), dtype={"service_id": str})

    active_services = build_active_service_table(calendar, calendar_dates)
    route_daily, daily_summary = compute_daily_outputs(routes, trips, active_services)

    route_out = pathlib.Path(args.route_daily_out)
    route_out.parent.mkdir(parents=True, exist_ok=True)
    route_daily.to_csv(route_out, index=False)

    daily_out = pathlib.Path(args.daily_summary_out)
    daily_out.parent.mkdir(parents=True, exist_ok=True)
    daily_summary.to_csv(daily_out, index=False)

    print(f"Wrote per-route daily service: {route_out.resolve()} ({len(route_daily)} rows)")
    print(f"Wrote daily summary: {daily_out.resolve()} ({len(daily_summary)} rows)")


if __name__ == "__main__":
    main()
