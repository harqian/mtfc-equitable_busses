import math
import tempfile
import unittest
from pathlib import Path

from model import (
    SearchConfig,
    apply_move,
    canonical_move_key,
    compute_sa_acceptance_probability,
    decay_tenures,
    has_global_fleet_conservation,
    is_feasible_solution,
    is_within_route_bounds,
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


class TestRouteFleetPrimitives(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tempdir.name) / "simplified_bus_routes.csv"
        self.csv_path.write_text(CSV_FIXTURE, encoding="utf-8")
        self.domain = load_route_fleet_domain(self.csv_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_loader_creates_stable_route_order_and_bounds(self) -> None:
        self.assertEqual(self.domain.route_ids, ("R1", "R2", "R3"))
        self.assertEqual(self.domain.baseline, (5, 3, 2))
        self.assertEqual(self.domain.lower_bounds, (0, 0, 0))
        self.assertEqual(self.domain.upper_bounds, (15, 13, 12))

    def test_objective_is_finite_and_deterministic_for_valid_vector(self) -> None:
        y = (5, 3, 2)
        obj1 = objective_function(y, domain=self.domain)
        obj2 = objective_function(y, domain=self.domain)
        self.assertTrue(math.isfinite(obj1))
        self.assertEqual(obj1, obj2)

    def test_objective_rejects_invalid_vectors(self) -> None:
        with self.assertRaises(ValueError):
            objective_function((5, 3), domain=self.domain)

        with self.assertRaises(ValueError):
            objective_function((5, 3, 2.5), domain=self.domain)

        with self.assertRaises(ValueError):
            objective_function((100, 3, 2), domain=self.domain)

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
            deviation_weight=0.05,
            service_weight=1.0,
        )
        result = run_route_fleet_search(domain=self.domain, config=config, seed=7)

        self.assertEqual(len(result.best_vector), len(self.domain.route_ids))
        self.assertTrue(math.isfinite(result.best_objective))
        self.assertLessEqual(result.best_objective, result.initial_objective + 1e-9)
        self.assertEqual(result.iterations_completed, 40)
        self.assertEqual(len(result.events), 40)

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


if __name__ == "__main__":
    unittest.main()
