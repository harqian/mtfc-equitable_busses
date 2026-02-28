#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

from model import SearchConfig, load_parameters, load_route_fleet_domain, run_route_fleet_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short TS/SA route fleet search smoke test.")
    parser.add_argument("--routes-csv", default="data/simplified_bus_routes.csv")
    parser.add_argument("--route-stops-csv", default="data/simplified_bus_route_stops.csv")
    parser.add_argument("--parameters-json", default="data/parameters.json")
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
    parameters = load_parameters(args.parameters_json)
    assumptions = parameters.estimation_assumptions
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
    parameters = replace(parameters, estimation_assumptions=assumptions)

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
        parameters=parameters,
    )
    baseline = result.initial_cost_breakdown
    optimized = result.best_cost_breakdown
    baseline_emissions = result.initial_emissions_breakdown
    optimized_emissions = result.best_emissions_breakdown
    baseline_objective = result.initial_objective_breakdown
    optimized_objective = result.best_objective_breakdown

    event_counts = Counter(event["event_case"] for event in result.events)

    print("Search complete")
    print(f"- routes: {len(domain.route_ids)}")
    print(f"- iterations: {result.iterations_completed}")
    print(f"- initial combined objective: {result.initial_objective:,.4f}")
    print(f"- best combined objective: {result.best_objective:,.4f}")
    print(f"- baseline annual cost: ${baseline.annual_total_cost:,.2f}")
    print(f"- optimized annual cost: ${optimized.annual_total_cost:,.2f}")
    print(f"- annual savings: ${baseline.annual_total_cost - optimized.annual_total_cost:,.2f}")
    print(f"- baseline annual net emissions: {baseline_emissions.baseline_total_emissions_grams:,.2f} g")
    print(f"- optimized annual net emissions: {optimized_emissions.candidate_total_emissions_grams:,.2f} g")
    print(
        f"- annual net-emissions delta: "
        f"{optimized_emissions.absolute_delta_emissions_grams:,.2f} g "
        f"({optimized_emissions.percent_delta_emissions:,.2f}%)"
    )
    print(f"- baseline budget slack: ${baseline.annual_budget_slack:,.2f}")
    print(f"- optimized budget slack: ${optimized.annual_budget_slack:,.2f}")
    print(f"- accepted improving moves: {result.accepted_improving_moves}")
    print(f"- accepted non-improving moves: {result.accepted_nonimproving_moves}")
    print("- pillar summary:")
    print(
        "  - cost: "
        f"baseline=${baseline_objective.cost.baseline_value:,.2f}, "
        f"optimized=${optimized_objective.cost.current_value:,.2f}, "
        f"delta=${optimized_objective.cost.absolute_delta:,.2f}, "
        f"pct={optimized_objective.cost.percent_delta:,.2f}%, "
        f"coefficient={optimized_objective.cost.coefficient:,.4f}, "
        f"contribution={optimized_objective.cost.weighted_contribution:,.4f}"
    )
    print(
        "  - emissions: "
        f"baseline={baseline_objective.emissions.baseline_value:,.2f} g, "
        f"optimized={optimized_objective.emissions.current_value:,.2f} g, "
        f"delta={optimized_objective.emissions.absolute_delta:,.2f} g, "
        f"pct={optimized_objective.emissions.percent_delta:,.2f}%, "
        f"coefficient={optimized_objective.emissions.coefficient:,.4f}, "
        f"contribution={optimized_objective.emissions.weighted_contribution:,.4f}"
    )
    print("- event counts:")
    for key in sorted(event_counts):
        print(f"  - {key}: {event_counts[key]}")

    changes = result.route_cost_delta_table.merge(
        result.route_emissions_delta_table,
        on=["route_id", "baseline_fleet", "optimized_fleet", "delta_fleet"],
        how="outer",
    )
    changes = changes.loc[changes["delta_fleet"] != 0].copy()
    changes = changes.sort_values(
        ["delta_total_cost", "delta_net_emissions_grams"],
        key=lambda series: series.abs(),
        ascending=False,
    )
    print(f"- routes changed: {len(changes)}")
    if len(changes) > 0:
        show = changes.head(max(1, args.show_top))
        cols = [
            "route_id",
            "baseline_fleet",
            "optimized_fleet",
            "delta_fleet",
            "delta_annual_vehicle_miles",
            "delta_total_cost",
            "delta_riders",
            "delta_bus_emissions_grams",
            "delta_rider_emissions_avoided_grams",
            "delta_net_emissions_grams",
        ]
        print("\nTop route-level cost and emissions deltas:")
        print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
