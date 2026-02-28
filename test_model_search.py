import math
import tempfile
import unittest
import json
from pathlib import Path

from model import (
    SearchConfig,
    apply_move,
    canonical_move_key,
    compute_cost_breakdown,
    compute_sa_acceptance_probability,
    decay_tenures,
    has_global_fleet_conservation,
    is_feasible_solution,
    is_within_route_bounds,
    load_cost_parameters,
    load_route_fleet_domain,
    make_add_move,
    make_drop_move,
    make_swap_move,
    objective_function,
    run_route_fleet_search,
)


CSV_FIXTURE = """route_id,route_short_name,route_long_name,weekday_typical_buses,weekday_planned_trips,stop_count
R3,3,Three,2,40,20
R1,1,One,5,120,60
R2,2,Two,3,80,40
"""

SYNTHETIC_SEARCH_FIXTURE = """route_id,route_short_name,route_long_name,weekday_typical_buses,weekday_planned_trips,stop_count
A,A,Alpha,4,220,90
B,B,Beta,4,140,55
C,C,Gamma,4,30,10
D,D,Delta,4,20,8
"""

PHASE2_ROUTE_FIXTURE = """route_id,route_short_name,route_long_name,weekday_typical_buses,weekday_planned_trips,stop_count
R1,1,One,1,10,2
R2,2,Two,2,4,2
"""

PHASE2_ROUTE_STOPS_FIXTURE = """route_id,direction_id,route_stop_order,stop_lat,stop_lon
R1,0,1,0.0,0.0
R1,0,2,0.0,1.0
R1,1,1,0.0,1.0
R1,1,2,0.0,0.0
R2,0,1,0.0,0.0
R2,0,2,1.0,0.0
"""


class TestRouteFleetPrimitives(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tempdir.name) / "simplified_bus_routes.csv"
        self.csv_path.write_text(CSV_FIXTURE, encoding="utf-8")
        self.cost_params_path = Path(self.tempdir.name) / "cost_parameters.json"
        self.cost_params_path.write_text(
            json.dumps(
                {
                    "reporting_constants": {
                        "annual_fare_revenue": {"value": 1.0, "label": "fare"},
                        "annual_advertising_revenue": {"value": 2.0, "label": "ads"},
                        "annual_external_subsidies": {"value": 3.0, "label": "subsidy"},
                        "annual_budget_ceiling": {"value": 4.0, "label": "budget"},
                    },
                    "operating_cost_parameters": {
                        "labor_cost_per_vehicle_hour": {"value": 5.0, "label": "labor"},
                        "maintenance_cost_per_vehicle_mile": {"value": 6.0, "label": "maintenance"},
                        "energy_cost_per_vehicle_mile": {"value": 7.0, "label": "energy"},
                        "annualized_capital_cost_per_vehicle": {"value": 1000.0, "label": "capital"},
                    },
                    "estimation_assumptions": {
                        "average_operating_speed_mph": {"value": 8.0, "label": "speed"},
                        "deadhead_multiplier": {"value": 1.1, "label": "deadhead"},
                        "dwell_recovery_multiplier": {"value": 1.2, "label": "dwell"},
                        "weekday_service_days_per_year": {"value": 260.0, "label": "days"},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.domain = load_route_fleet_domain(self.csv_path)
        self.cost_params = load_cost_parameters(self.cost_params_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_loader_creates_stable_route_order_and_bounds(self) -> None:
        self.assertEqual(self.domain.route_ids, ("R1", "R2", "R3"))
        self.assertEqual(self.domain.baseline, (5, 3, 2))
        self.assertEqual(self.domain.lower_bounds, (0, 0, 0))
        self.assertEqual(self.domain.upper_bounds, (15, 13, 12))

    def test_cost_parameter_loader_validates_schema(self) -> None:
        loaded = load_cost_parameters(self.cost_params_path)
        self.assertEqual(loaded.reporting_constants.annual_fare_revenue.value, 1.0)
        self.assertEqual(loaded.operating_cost_parameters.annualized_capital_cost_per_vehicle.label, "capital")

        bad_path = Path(self.tempdir.name) / "bad_cost_parameters.json"
        bad_path.write_text(
            json.dumps({"reporting_constants": {}, "operating_cost_parameters": {}}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            load_cost_parameters(bad_path)

    def test_objective_is_finite_and_deterministic_for_valid_vector(self) -> None:
        y = (5, 3, 2)
        obj1 = objective_function(y, domain=self.domain)
        obj2 = objective_function(y, domain=self.domain)
        self.assertTrue(math.isfinite(obj1))
        self.assertEqual(obj1, obj2)
        self.assertEqual(obj1, 0.0)

    def test_objective_rejects_invalid_vectors(self) -> None:
        with self.assertRaises(ValueError):
            objective_function((5, 3), domain=self.domain)

        with self.assertRaises(ValueError):
            objective_function((5, 3, 2.5), domain=self.domain)

        with self.assertRaises(ValueError):
            objective_function((100, 3, 2), domain=self.domain)

    def test_cost_breakdown_is_structured_for_baseline_and_expansion(self) -> None:
        baseline_breakdown = compute_cost_breakdown(
            self.domain.baseline,
            domain=self.domain,
            cost_parameters=self.cost_params,
        )
        expanded_breakdown = compute_cost_breakdown(
            (6, 3, 2),
            domain=self.domain,
            cost_parameters=self.cost_params,
        )

        self.assertEqual(baseline_breakdown.objective_cost, 0.0)
        self.assertEqual(expanded_breakdown.objective_cost, 1000.0)
        self.assertEqual(len(baseline_breakdown.route_breakdowns), len(self.domain.route_ids))
        self.assertEqual(expanded_breakdown.net_new_fleet, 1)
        self.assertTrue(all(route.annual_total_cost == 0.0 for route in baseline_breakdown.route_breakdowns))

    def test_feasibility_checks_bounds_and_fleet_conservation(self) -> None:
        baseline = self.domain.baseline
        self.assertTrue(is_within_route_bounds(baseline, self.domain))
        self.assertTrue(has_global_fleet_conservation(baseline, self.domain))
        self.assertTrue(is_feasible_solution(baseline, self.domain))

        out_of_bounds = (16, 3, 2)
        self.assertFalse(is_within_route_bounds(out_of_bounds, self.domain))
        self.assertFalse(is_feasible_solution(out_of_bounds, self.domain))

        not_conserved = (6, 3, 2)
        self.assertTrue(is_within_route_bounds(not_conserved, self.domain))
        self.assertFalse(has_global_fleet_conservation(not_conserved, self.domain))
        self.assertFalse(is_feasible_solution(not_conserved, self.domain))
        self.assertTrue(is_feasible_solution(not_conserved, self.domain, require_global_fleet_conservation=False))

    def test_move_application_preserves_expected_invariants(self) -> None:
        baseline = self.domain.baseline

        add_move = make_add_move(0)
        after_add = apply_move(baseline, add_move, domain=self.domain)
        self.assertEqual(after_add, (6, 3, 2))
        self.assertEqual(sum(after_add), sum(baseline) + 1)

        drop_move = make_drop_move(1)
        after_drop = apply_move(baseline, drop_move, domain=self.domain)
        self.assertEqual(after_drop, (5, 2, 2))
        self.assertEqual(sum(after_drop), sum(baseline) - 1)

        swap_move = make_swap_move(0, 2)
        after_swap = apply_move(baseline, swap_move, domain=self.domain)
        self.assertEqual(after_swap, (6, 3, 1))
        self.assertEqual(sum(after_swap), sum(baseline))

        self.assertEqual(canonical_move_key(add_move), ("ADD", 0, -1))
        self.assertEqual(canonical_move_key(drop_move), ("DROP", -1, 1))
        self.assertEqual(canonical_move_key(swap_move), ("SWAP", 0, 2))


class TestSearchLoopBehavior(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tempdir.name) / "synthetic_routes.csv"
        self.csv_path.write_text(SYNTHETIC_SEARCH_FIXTURE, encoding="utf-8")
        self.domain = load_route_fleet_domain(self.csv_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_sa_probability_and_tenure_decay_helpers(self) -> None:
        self.assertEqual(compute_sa_acceptance_probability(delta=-1.0, temperature=1.0), 1.0)
        self.assertEqual(compute_sa_acceptance_probability(delta=1.0, temperature=0.0), 0.0)
        self.assertLess(compute_sa_acceptance_probability(delta=1.0, temperature=0.5), 1.0)

        add = [3.0, 0.2]
        drop = [1.0, 0.0]
        decay_tenures(add, drop)
        self.assertEqual(add, [2.0, 0.0])
        self.assertEqual(drop, [0.0, 0.0])

    def test_search_run_returns_valid_structured_result(self) -> None:
        config = SearchConfig(
            max_iterations=40,
            temp_init=0.8,
            temp_factor=0.97,
            tenure_init=2.0,
            tenure_factor=1.2,
            nonimp_in_max=4,
            nonimp_out_max=12,
            nbhd_add_lim=8,
            nbhd_drop_lim=8,
            nbhd_swap_lim=20,
            attractive_max=6,
        )
        result = run_route_fleet_search(domain=self.domain, config=config, seed=7)

        self.assertEqual(len(result.best_vector), len(self.domain.route_ids))
        self.assertTrue(math.isfinite(result.best_objective))
        self.assertLessEqual(result.best_objective, result.initial_objective + 1e-9)
        self.assertEqual(result.iterations_completed, 40)
        self.assertEqual(len(result.events), 40)
        self.assertEqual(result.initial_cost_breakdown.objective_cost, result.initial_objective)
        self.assertEqual(result.best_cost_breakdown.objective_cost, result.best_objective)
        self.assertEqual(len(result.best_cost_breakdown.route_breakdowns), len(self.domain.route_ids))
        self.assertEqual(result.best_budget_slack, result.best_cost_breakdown.annual_budget_slack)
        self.assertEqual(result.initial_budget_slack, result.initial_cost_breakdown.annual_budget_slack)
        self.assertAlmostEqual(
            result.annual_cost_delta_vs_baseline,
            result.best_cost_breakdown.annual_total_cost - result.initial_cost_breakdown.annual_total_cost,
        )
        self.assertFalse(result.route_cost_delta_table.empty)

    def test_seeded_runs_are_deterministic(self) -> None:
        config = SearchConfig(max_iterations=25, nbhd_add_lim=6, nbhd_drop_lim=6, nbhd_swap_lim=12)
        result1 = run_route_fleet_search(domain=self.domain, config=config, seed=123)
        result2 = run_route_fleet_search(domain=self.domain, config=config, seed=123)

        self.assertEqual(result1.best_vector, result2.best_vector)
        self.assertEqual(result1.best_objective, result2.best_objective)
        self.assertEqual(result1.events, result2.events)

    def test_all_search_event_vectors_are_feasible(self) -> None:
        config = SearchConfig(max_iterations=35, nbhd_add_lim=7, nbhd_drop_lim=7, nbhd_swap_lim=15)
        result = run_route_fleet_search(domain=self.domain, config=config, seed=11)

        self.assertTrue(
            all(
                is_feasible_solution(event["current_vector"], domain=self.domain)
                for event in result.events
            )
        )


class TestPhaseTwoRouteDrivers(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.routes_path = Path(self.tempdir.name) / "routes.csv"
        self.route_stops_path = Path(self.tempdir.name) / "route_stops.csv"
        self.cost_params_path = Path(self.tempdir.name) / "cost_parameters.json"
        self.routes_path.write_text(PHASE2_ROUTE_FIXTURE, encoding="utf-8")
        self.route_stops_path.write_text(PHASE2_ROUTE_STOPS_FIXTURE, encoding="utf-8")
        self.cost_params_path.write_text(
            json.dumps(
                {
                    "reporting_constants": {
                        "annual_fare_revenue": {"value": 1.0, "label": "fare"},
                        "annual_advertising_revenue": {"value": 2.0, "label": "ads"},
                        "annual_external_subsidies": {"value": 3.0, "label": "subsidy"},
                        "annual_budget_ceiling": {"value": 4.0, "label": "budget"},
                    },
                    "operating_cost_parameters": {
                        "labor_cost_per_vehicle_hour": {"value": 10.0, "label": "labor"},
                        "maintenance_cost_per_vehicle_mile": {"value": 2.0, "label": "maintenance"},
                        "energy_cost_per_vehicle_mile": {"value": 1.0, "label": "energy"},
                        "annualized_capital_cost_per_vehicle": {"value": 500.0, "label": "capital"},
                    },
                    "estimation_assumptions": {
                        "average_operating_speed_mph": {"value": 10.0, "label": "speed"},
                        "deadhead_multiplier": {"value": 1.0, "label": "deadhead"},
                        "dwell_recovery_multiplier": {"value": 1.0, "label": "dwell"},
                        "weekday_service_days_per_year": {"value": 1.0, "label": "days"},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.cost_params = load_cost_parameters(self.cost_params_path)
        self.domain = load_route_fleet_domain(self.routes_path, route_stops_path=self.route_stops_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_distance_estimation_is_deterministic_for_fixed_stops(self) -> None:
        domain2 = load_route_fleet_domain(self.routes_path, route_stops_path=self.route_stops_path)

        self.assertEqual(self.domain.route_driver_estimates, domain2.route_driver_estimates)
        r1 = self.domain.route_driver_estimates[0]
        self.assertAlmostEqual(r1.one_way_distance_miles, 69.093, places=2)
        self.assertAlmostEqual(r1.round_trip_distance_miles, 138.186, places=2)

    def test_vehicle_miles_and_hours_are_finite_and_non_negative(self) -> None:
        breakdown = compute_cost_breakdown(self.domain.baseline, domain=self.domain, cost_parameters=self.cost_params)

        self.assertTrue(math.isfinite(breakdown.annual_total_cost))
        for route in breakdown.route_breakdowns:
            self.assertGreaterEqual(route.baseline_annual_vehicle_miles, 0.0)
            self.assertGreaterEqual(route.candidate_annual_vehicle_miles, 0.0)
            self.assertGreaterEqual(route.baseline_annual_vehicle_hours, 0.0)
            self.assertGreaterEqual(route.candidate_annual_vehicle_hours, 0.0)
            self.assertTrue(math.isfinite(route.candidate_annual_vehicle_miles))
            self.assertTrue(math.isfinite(route.candidate_annual_vehicle_hours))

    def test_candidate_vectors_change_costs_monotonically(self) -> None:
        baseline = compute_cost_breakdown((1, 2), domain=self.domain, cost_parameters=self.cost_params)
        expanded = compute_cost_breakdown((2, 2), domain=self.domain, cost_parameters=self.cost_params)
        reduced = compute_cost_breakdown((0, 2), domain=self.domain, cost_parameters=self.cost_params)

        base_r1 = baseline.route_breakdowns[0]
        exp_r1 = expanded.route_breakdowns[0]
        red_r1 = reduced.route_breakdowns[0]

        self.assertGreater(exp_r1.candidate_annual_vehicle_miles, base_r1.candidate_annual_vehicle_miles)
        self.assertGreater(exp_r1.annual_total_cost, base_r1.annual_total_cost)
        self.assertEqual(red_r1.candidate_annual_vehicle_miles, 0.0)
        self.assertEqual(red_r1.candidate_annual_vehicle_hours, 0.0)
        self.assertLess(reduced.annual_total_cost, baseline.annual_total_cost)


class TestPhaseThreeObjectiveAndReporting(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.routes_path = Path(self.tempdir.name) / "routes.csv"
        self.route_stops_path = Path(self.tempdir.name) / "route_stops.csv"
        self.cost_params_path = Path(self.tempdir.name) / "cost_parameters.json"
        self.alt_cost_params_path = Path(self.tempdir.name) / "alt_cost_parameters.json"
        self.routes_path.write_text(PHASE2_ROUTE_FIXTURE, encoding="utf-8")
        self.route_stops_path.write_text(PHASE2_ROUTE_STOPS_FIXTURE, encoding="utf-8")

        base_payload = {
            "reporting_constants": {
                "annual_fare_revenue": {"value": 100.0, "label": "fare"},
                "annual_advertising_revenue": {"value": 20.0, "label": "ads"},
                "annual_external_subsidies": {"value": 30.0, "label": "subsidy"},
                "annual_budget_ceiling": {"value": 10000.0, "label": "budget"},
            },
            "operating_cost_parameters": {
                "labor_cost_per_vehicle_hour": {"value": 10.0, "label": "labor"},
                "maintenance_cost_per_vehicle_mile": {"value": 2.0, "label": "maintenance"},
                "energy_cost_per_vehicle_mile": {"value": 1.0, "label": "energy"},
                "annualized_capital_cost_per_vehicle": {"value": 500.0, "label": "capital"},
            },
            "estimation_assumptions": {
                "average_operating_speed_mph": {"value": 10.0, "label": "speed"},
                "deadhead_multiplier": {"value": 1.0, "label": "deadhead"},
                "dwell_recovery_multiplier": {"value": 1.0, "label": "dwell"},
                "weekday_service_days_per_year": {"value": 1.0, "label": "days"},
            },
        }
        alt_payload = json.loads(json.dumps(base_payload))
        alt_payload["reporting_constants"]["annual_fare_revenue"]["value"] = 999999.0
        alt_payload["reporting_constants"]["annual_advertising_revenue"]["value"] = 888888.0
        alt_payload["reporting_constants"]["annual_external_subsidies"]["value"] = 777777.0

        self.cost_params_path.write_text(json.dumps(base_payload), encoding="utf-8")
        self.alt_cost_params_path.write_text(json.dumps(alt_payload), encoding="utf-8")
        self.cost_params = load_cost_parameters(self.cost_params_path)
        self.alt_cost_params = load_cost_parameters(self.alt_cost_params_path)
        self.domain = load_route_fleet_domain(self.routes_path, route_stops_path=self.route_stops_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_reporting_constants_do_not_change_objective_ranking(self) -> None:
        baseline = compute_cost_breakdown((1, 2), domain=self.domain, cost_parameters=self.cost_params)
        baseline_alt = compute_cost_breakdown((1, 2), domain=self.domain, cost_parameters=self.alt_cost_params)

        self.assertEqual(baseline.objective_cost, baseline_alt.objective_cost)
        self.assertNotEqual(baseline.annual_total_revenue, baseline_alt.annual_total_revenue)

    def test_global_fleet_conservation_keeps_capital_cost_at_zero_for_swaps(self) -> None:
        swapped = compute_cost_breakdown((0, 3), domain=self.domain, cost_parameters=self.cost_params)

        self.assertEqual(sum((0, 3)), sum(self.domain.baseline))
        self.assertEqual(swapped.net_new_fleet, 0)
        self.assertEqual(swapped.annual_capital_cost, 0.0)

    def test_search_result_reports_budget_and_route_deltas(self) -> None:
        result = run_route_fleet_search(
            domain=self.domain,
            config=SearchConfig(max_iterations=10, nbhd_add_lim=4, nbhd_drop_lim=4, nbhd_swap_lim=6),
            seed=5,
        )

        self.assertIn("delta_total_cost", result.route_cost_delta_table.columns)
        self.assertAlmostEqual(
            result.route_cost_delta_table["delta_total_cost"].sum(),
            result.annual_cost_delta_vs_baseline,
        )
        self.assertAlmostEqual(
            result.best_budget_slack,
            result.best_cost_breakdown.reporting_constants.annual_budget_ceiling.value
            - result.best_cost_breakdown.annual_total_cost,
        )

    def test_search_uses_supplied_cost_parameters_for_objective(self) -> None:
        fast_params_payload = {
            "reporting_constants": {
                "annual_fare_revenue": {"value": 100.0, "label": "fare"},
                "annual_advertising_revenue": {"value": 20.0, "label": "ads"},
                "annual_external_subsidies": {"value": 30.0, "label": "subsidy"},
                "annual_budget_ceiling": {"value": 10000.0, "label": "budget"},
            },
            "operating_cost_parameters": {
                "labor_cost_per_vehicle_hour": {"value": 10.0, "label": "labor"},
                "maintenance_cost_per_vehicle_mile": {"value": 2.0, "label": "maintenance"},
                "energy_cost_per_vehicle_mile": {"value": 1.0, "label": "energy"},
                "annualized_capital_cost_per_vehicle": {"value": 500.0, "label": "capital"},
            },
            "estimation_assumptions": {
                "average_operating_speed_mph": {"value": 100.0, "label": "speed"},
                "deadhead_multiplier": {"value": 1.0, "label": "deadhead"},
                "dwell_recovery_multiplier": {"value": 1.0, "label": "dwell"},
                "weekday_service_days_per_year": {"value": 1.0, "label": "days"},
            },
        }
        fast_path = Path(self.tempdir.name) / "fast_cost_parameters.json"
        fast_path.write_text(json.dumps(fast_params_payload), encoding="utf-8")
        fast_params = load_cost_parameters(fast_path)

        default_result = run_route_fleet_search(
            domain=self.domain,
            config=SearchConfig(max_iterations=5, nbhd_add_lim=4, nbhd_drop_lim=4, nbhd_swap_lim=6),
            seed=3,
            cost_parameters=self.cost_params,
        )
        fast_result = run_route_fleet_search(
            domain=self.domain,
            config=SearchConfig(max_iterations=5, nbhd_add_lim=4, nbhd_drop_lim=4, nbhd_swap_lim=6),
            seed=3,
            cost_parameters=fast_params,
        )

        self.assertNotEqual(default_result.initial_objective, fast_result.initial_objective)
        self.assertEqual(default_result.initial_objective, default_result.initial_cost_breakdown.objective_cost)
        self.assertEqual(fast_result.initial_objective, fast_result.initial_cost_breakdown.objective_cost)


if __name__ == "__main__":
    unittest.main()
