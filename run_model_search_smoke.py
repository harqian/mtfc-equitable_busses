#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

from model import SearchConfig, load_cost_parameters, load_route_fleet_domain, run_route_fleet_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short TS/SA route fleet search smoke test.")
    parser.add_argument("--routes-csv", default="data/simplified_bus_routes.csv")
    parser.add_argument("--route-stops-csv", default="data/simplified_bus_route_stops.csv")
    parser.add_argument("--cost-parameters-json", default="data/cost_parameters.json")
    parser.add_argument("--iterations", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp-init", type=float, default=0.8)
    parser.add_argument("--temp-factor", type=float, default=0.98)
    parser.add_argument("--tenure-init", type=float, default=3.0)
    parser.add_argument("--tenure-factor", type=float, default=1.2)
    parser.add_argument("--nbhd-add", type=int, default=24)
    parser.add_argument("--nbhd-drop", type=int, default=24)
    parser.add_argument("--nbhd-swap", type=int, default=60)
    parser.add_argument("--attractive-max", type=int, default=20)
    parser.add_argument("--nonimp-in-max", type=int, default=8)
    parser.add_argument("--nonimp-out-max", type=int, default=30)
    parser.add_argument("--average-operating-speed", type=float)
    parser.add_argument("--service-days-per-year", type=float)
    parser.add_argument("--show-top", type=int, default=15, help="How many route deltas to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cost_parameters = load_cost_parameters(args.cost_parameters_json)
    assumptions = cost_parameters.estimation_assumptions
    if args.average_operating_speed is not None:
        assumptions = replace(
            assumptions,
            average_operating_speed_mph=replace(
                assumptions.average_operating_speed_mph,
                value=float(args.average_operating_speed),
            ),
        )
    if args.service_days_per_year is not None:
        assumptions = replace(
            assumptions,
            weekday_service_days_per_year=replace(
                assumptions.weekday_service_days_per_year,
                value=float(args.service_days_per_year),
            ),
        )
    cost_parameters = replace(cost_parameters, estimation_assumptions=assumptions)

    domain = load_route_fleet_domain(args.routes_csv, route_stops_path=args.route_stops_csv)

    config = SearchConfig(
        max_iterations=args.iterations,
        temp_init=args.temp_init,
        temp_factor=args.temp_factor,
        tenure_init=args.tenure_init,
        tenure_factor=args.tenure_factor,
        nonimp_in_max=args.nonimp_in_max,
        nonimp_out_max=args.nonimp_out_max,
        nbhd_add_lim=args.nbhd_add,
        nbhd_drop_lim=args.nbhd_drop,
        nbhd_swap_lim=args.nbhd_swap,
        attractive_max=args.attractive_max,
    )

    result = run_route_fleet_search(
        domain=domain,
        config=config,
        seed=args.seed,
        cost_parameters=cost_parameters,
    )
    baseline = result.initial_cost_breakdown
    optimized = result.best_cost_breakdown

    event_counts = Counter(event["event_case"] for event in result.events)

    print("Search complete")
    print(f"- routes: {len(domain.route_ids)}")
    print(f"- iterations: {result.iterations_completed}")
    print(f"- baseline annual cost: ${baseline.annual_total_cost:,.2f}")
    print(f"- optimized annual cost: ${optimized.annual_total_cost:,.2f}")
    print(f"- annual savings: ${baseline.annual_total_cost - optimized.annual_total_cost:,.2f}")
    print(f"- baseline budget slack: ${baseline.annual_budget_slack:,.2f}")
    print(f"- optimized budget slack: ${optimized.annual_budget_slack:,.2f}")
    print(f"- accepted improving moves: {result.accepted_improving_moves}")
    print(f"- accepted non-improving moves: {result.accepted_nonimproving_moves}")
    print("- event counts:")
    for key in sorted(event_counts):
        print(f"  - {key}: {event_counts[key]}")

    changes = result.route_cost_delta_table.loc[result.route_cost_delta_table["delta_fleet"] != 0].copy()
    print(f"- routes changed: {len(changes)}")
    if len(changes) > 0:
        show = changes.head(max(1, args.show_top))
        cols = [
            "route_id",
            "baseline_fleet",
            "optimized_fleet",
            "delta_fleet",
            "delta_annual_vehicle_miles",
            "delta_labor_cost",
            "delta_maintenance_cost",
            "delta_energy_cost",
            "delta_total_cost",
        ]
        print("\nTop route-level annual cost deltas:")
        print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
