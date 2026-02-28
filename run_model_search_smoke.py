#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
from collections import Counter

from model import SearchConfig, load_route_fleet_domain, run_route_fleet_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short TS/SA route fleet search smoke test.")
    parser.add_argument("--routes-csv", default="data/simplified_bus_routes.csv")
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
    parser.add_argument("--deviation-weight", type=float, default=1.0)
    parser.add_argument("--service-weight", type=float, default=0.25)
    parser.add_argument("--show-top", type=int, default=15, help="How many route deltas to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain = load_route_fleet_domain(args.routes_csv)

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
        deviation_weight=args.deviation_weight,
        service_weight=args.service_weight,
    )

    result = run_route_fleet_search(domain=domain, config=config, seed=args.seed)

    event_counts = Counter(event["event_case"] for event in result.events)

    print("Search complete")
    print(f"- routes: {len(domain.route_ids)}")
    print(f"- iterations: {result.iterations_completed}")
    print(f"- initial objective: {result.initial_objective:.6f}")
    print(f"- best objective: {result.best_objective:.6f}")
    print(f"- accepted improving moves: {result.accepted_improving_moves}")
    print(f"- accepted non-improving moves: {result.accepted_nonimproving_moves}")
    print("- event counts:")
    for key in sorted(event_counts):
        print(f"  - {key}: {event_counts[key]}")

    changes = result.best_route_table.loc[result.best_route_table["delta_fleet"] != 0].copy()
    changes = changes.sort_values("delta_fleet", key=lambda s: s.abs(), ascending=False)

    print(f"- routes changed: {len(changes)}")
    if len(changes) > 0:
        show = changes.head(max(1, args.show_top))
        cols = ["route_id", "route_short_name", "weekday_typical_buses", "optimized_fleet", "delta_fleet"]
        print("\nTop route deltas:")
        print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
