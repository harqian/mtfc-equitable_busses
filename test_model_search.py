import math
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

import geopandas as gpd
from shapely.geometry import Polygon

from data_utils import find_data_file as resolve_data_file
from model import (
    SearchConfig,
    _load_weekday_ridership,
    apply_move,
    assign_stops_to_sf_tracts,
    build_sf_equity_data_bundle,
    canonical_move_key,
    compute_cost_breakdown,
    compute_equity_breakdown,
    compute_emissions_breakdown,
    compute_tract_service_access_summaries,
    compute_objective_breakdown,
    compute_sa_acceptance_probability,
    decay_tenures,
    has_global_fleet_conservation,
    is_feasible_solution,
    is_within_route_bounds,
    load_epc_tracts_geojson,
    load_parameters,
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

PHASE2_ROUTE_STOPS_FIXTURE = """route_id,direction_id,route_stop_order,stop_lat,stop_lon,elevation_m
R1,0,1,0.0,0.0,0.0
R1,0,2,0.0,1.0,1609.344
R1,1,1,0.0,1.0,1609.344
R1,1,2,0.0,0.0,0.0
R2,0,1,0.0,0.0,
R2,0,2,1.0,0.0,100.0
"""

PHASE3_RIDERSHIP_FIXTURE = """Month,Route,Service Category,Service Day of the Week,Average Daily Boardings
January 2024,1 One,Local,Weekday,"1,000"
February 2024,1 One,Local,Weekday,"1,500"
January 2024,2 Two,Local,Weekday,"200"
February 2024,2 Two,Local,Weekday,"250"
January 2024,1 One,Local,Saturday,"700"
"""


class TestRouteFleetPrimitives(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tempdir.name) / "simplified_bus_routes.csv"
        self.csv_path.write_text(CSV_FIXTURE, encoding="utf-8")
        self.parameters_path = Path(self.tempdir.name) / "parameters.json"
        self.parameters_path.write_text(
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
                    "emissions_parameters": {
                        "car_emissions_grams_per_mile": {"value": 353.802111, "label": "car"},
                        "car_ownership_probability": {"value": 0.7, "label": "ownership"},
                        "bus_base_emissions_grams_per_mile": {"value": 2830.0, "label": "bus base"},
                        "bus_climb_penalty_grams_per_mile": {"value": 5660.0, "label": "bus climb"},
                    },
                    "objective_weights": {
                        "cost_percent_change_coefficient": {"value": 1.0, "label": "cost weight"},
                        "emissions_percent_change_coefficient": {"value": 1.0, "label": "emissions weight"},
                        "equity_percent_change_coefficient": {"value": 0.0, "label": "equity weight"},
                    },
                    "equity_parameters": {
                        "service_intensity_coefficient": {"value": 1.0, "label": "service intensity"},
                        "waiting_time_coefficient": {"value": 1.0, "label": "waiting time"},
                    },
                    "ridership_assumptions": {
                        "route_average_trip_fraction": {"value": 0.5, "label": "trip fraction"},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.domain = load_route_fleet_domain(self.csv_path)
        self.parameters = load_parameters(self.parameters_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_loader_creates_stable_route_order_and_bounds(self) -> None:
        self.assertEqual(self.domain.route_ids, ("R1", "R2", "R3"))
        self.assertEqual(self.domain.baseline, (5, 3, 2))
        self.assertEqual(self.domain.lower_bounds, (0, 0, 0))
        self.assertEqual(self.domain.upper_bounds, (15, 13, 12))

    def test_cost_parameter_loader_validates_schema(self) -> None:
        loaded = load_parameters(self.parameters_path)
        self.assertEqual(loaded.reporting_constants.annual_fare_revenue.value, 1.0)
        self.assertEqual(loaded.operating_cost_parameters.annualized_capital_cost_per_vehicle.label, "capital")
        self.assertEqual(loaded.emissions_parameters.car_emissions_grams_per_mile.value, 353.802111)
        self.assertEqual(loaded.objective_weights.cost_percent_change_coefficient.value, 1.0)
        self.assertEqual(loaded.objective_weights.equity_percent_change_coefficient.value, 0.0)
        self.assertEqual(loaded.equity_parameters.service_intensity_coefficient.value, 1.0)

        bad_path = Path(self.tempdir.name) / "bad_parameters.json"
        bad_path.write_text(
            json.dumps({"reporting_constants": {}, "operating_cost_parameters": {}}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            load_parameters(bad_path)

    def test_objective_is_finite_and_deterministic_for_valid_vector(self) -> None:
        y = (5, 3, 2)
        obj1 = objective_function(y, domain=self.domain, parameters=self.parameters)
        obj2 = objective_function(y, domain=self.domain, parameters=self.parameters)
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
            parameters=self.parameters,
        )
        expanded_breakdown = compute_cost_breakdown(
            (6, 3, 2),
            domain=self.domain,
            parameters=self.parameters,
        )

        self.assertEqual(baseline_breakdown.objective_cost, 0.0)
        self.assertEqual(expanded_breakdown.objective_cost, 1000.0)
        self.assertEqual(len(baseline_breakdown.route_breakdowns), len(self.domain.route_ids))
        self.assertEqual(expanded_breakdown.net_new_fleet, 1)
        self.assertTrue(all(route.annual_total_cost == 0.0 for route in baseline_breakdown.route_breakdowns))

    def test_emissions_and_objective_breakdowns_are_structured(self) -> None:
        baseline_emissions = compute_emissions_breakdown(
            self.domain.baseline,
            domain=self.domain,
            parameters=self.parameters,
        )
        expanded_objective = compute_objective_breakdown(
            (6, 3, 2),
            domain=self.domain,
            parameters=self.parameters,
        )

        self.assertEqual(len(baseline_emissions.route_breakdowns), len(self.domain.route_ids))
        self.assertEqual(baseline_emissions.candidate_total_emissions_grams, 0.0)
        self.assertEqual(expanded_objective.cost.baseline_value, 0.0)
        self.assertGreater(expanded_objective.cost.current_value, 0.0)
        self.assertGreater(expanded_objective.total_combined_objective, 0.0)

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
        self.assertEqual(result.initial_objective_breakdown.total_combined_objective, result.initial_objective)
        self.assertEqual(result.best_objective_breakdown.total_combined_objective, result.best_objective)
        self.assertEqual(len(result.best_cost_breakdown.route_breakdowns), len(self.domain.route_ids))
        self.assertEqual(len(result.best_emissions_breakdown.route_breakdowns), len(self.domain.route_ids))
        self.assertEqual(len(result.best_equity_breakdown.tract_breakdowns), len(self.domain.equity_tracts))
        self.assertEqual(result.best_budget_slack, result.best_cost_breakdown.annual_budget_slack)
        self.assertEqual(result.initial_budget_slack, result.initial_cost_breakdown.annual_budget_slack)
        self.assertAlmostEqual(
            result.annual_cost_delta_vs_baseline,
            result.best_cost_breakdown.annual_total_cost - result.initial_cost_breakdown.annual_total_cost,
        )
        self.assertFalse(result.route_cost_delta_table.empty)
        self.assertFalse(result.route_emissions_delta_table.empty)
        self.assertIn("delta_utility", result.tract_equity_delta_table.columns)

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


class TestPhaseOneEquityData(unittest.TestCase):
    def _mock_epc_feature_collection(self) -> dict[str, object]:
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "geoid": "06075010100",
                        "epc_2050": 1,
                        "epc_class": "Tier 1",
                        "tot_pop": 1200,
                    },
                    "geometry": {"type": "Point", "coordinates": [-122.4, 37.7]},
                },
                {
                    "type": "Feature",
                    "properties": {
                        "geoid": "06075010200",
                        "epc_2050": 0,
                        "epc_class": "Not EPC",
                        "tot_pop": 800,
                    },
                    "geometry": {"type": "Point", "coordinates": [-122.5, 37.8]},
                },
            ],
        }

    def _mock_epc_frame(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "geoid": ["06075010100", "06075010200", "06001400100"],
                "epc_2050": [1, 0, 1],
                "epc_class": ["Tier 1", "Not EPC", "Tier 2"],
                "tot_pop": [1200, 800, 500],
            },
            geometry=gpd.points_from_xy([-122.4, -122.5, -122.2], [37.7, 37.8, 37.9]),
            crs="EPSG:4326",
        )

    def _mock_census_frame(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "GEOID": ["06075010200", "06075010100", "06001400100"],
                "COUNTYFP": ["075", "075", "001"],
            },
            geometry=gpd.points_from_xy([-122.5, -122.4, -122.2], [37.8, 37.7, 37.9]),
            crs="EPSG:4326",
        )

    @mock.patch("model.load_sf_tract_population_table")
    @mock.patch("model._download_to_path")
    @mock.patch("model.gpd.read_file")
    def test_equity_loader_returns_sf_rows_with_population_and_is_deterministic(
        self,
        read_file: mock.Mock,
        _download: mock.Mock,
        load_population: mock.Mock,
    ) -> None:
        _download.side_effect = lambda _url, out_path, attempts=3: out_path.write_text("{}", encoding="utf-8")
        read_file.side_effect = lambda path: (
            self._mock_epc_frame() if str(path).endswith("mtc_epc.geojson") else self._mock_census_frame()
        )
        load_population.return_value = self._mock_epc_frame().loc[:, ["geoid", "tot_pop"]].rename(
            columns={"tot_pop": "population"}
        )

        first = build_sf_equity_data_bundle()
        second = build_sf_equity_data_bundle()

        self.assertEqual(first.population_field, "population")
        self.assertEqual(list(first.sf_epc_tracts["geoid"]), ["06075010100", "06075010200"])
        self.assertEqual(list(first.sf_epc_tracts["epc_2050"]), [1, 0])
        self.assertEqual(list(first.sf_epc_tracts["tract_population"]), [1200, 800])
        self.assertEqual(first.sf_epc_tracts[["geoid", "epc_2050", "epc_class", "tract_population"]].to_dict("records"),
                         second.sf_epc_tracts[["geoid", "epc_2050", "epc_class", "tract_population"]].to_dict("records"))

    @mock.patch("model._download_to_path")
    @mock.patch("model.gpd.read_file")
    def test_load_epc_tracts_geojson_parses_feature_collection_when_driver_rejects_payload(
        self,
        read_file: mock.Mock,
        download: mock.Mock,
    ) -> None:
        def write_feature_collection(_url: str, out_path: Path, attempts: int = 3) -> None:
            out_path.write_text(json.dumps(self._mock_epc_feature_collection()), encoding="utf-8")

        download.side_effect = write_feature_collection
        read_file.side_effect = RuntimeError("unsupported format")

        gdf = load_epc_tracts_geojson()

        self.assertEqual(list(gdf["geoid"]), ["06075010100", "06075010200"])
        self.assertEqual(list(gdf["epc_2050"]), [1, 0])
        self.assertEqual(list(gdf["tot_pop"]), [1200, 800])

    @mock.patch("model.gpd.read_file")
    @mock.patch("model._download_to_path")
    def test_load_epc_tracts_geojson_falls_back_to_cached_payload_when_live_download_fails(
        self,
        download: mock.Mock,
        read_file: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            cache_path = Path(tempdir) / "mtc_epc.geojson"
            cache_path.write_text(json.dumps(self._mock_epc_feature_collection()), encoding="utf-8")
            download.side_effect = ConnectionRefusedError("upstream unavailable")
            read_file.side_effect = RuntimeError("unsupported format")

            with mock.patch("model._equity_cache_path", return_value=cache_path):
                gdf = load_epc_tracts_geojson()

        self.assertEqual(list(gdf["geoid"]), ["06075010100", "06075010200"])
        self.assertEqual(list(gdf["epc_class"]), ["Tier 1", "Not EPC"])


class TestPhaseTwoRouteDrivers(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.routes_path = Path(self.tempdir.name) / "routes.csv"
        self.route_stops_path = Path(self.tempdir.name) / "route_stops.csv"
        self.parameters_path = Path(self.tempdir.name) / "parameters.json"
        self.routes_path.write_text(PHASE2_ROUTE_FIXTURE, encoding="utf-8")
        self.route_stops_path.write_text(PHASE2_ROUTE_STOPS_FIXTURE, encoding="utf-8")
        self.parameters_path.write_text(
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
                    "emissions_parameters": {
                        "car_emissions_grams_per_mile": {"value": 353.802111, "label": "car"},
                        "car_ownership_probability": {"value": 0.7, "label": "ownership"},
                        "bus_base_emissions_grams_per_mile": {"value": 2830.0, "label": "bus base"},
                        "bus_climb_penalty_grams_per_mile": {"value": 5660.0, "label": "bus climb"},
                    },
                    "objective_weights": {
                        "cost_percent_change_coefficient": {"value": 1.0, "label": "cost weight"},
                        "emissions_percent_change_coefficient": {"value": 1.0, "label": "emissions weight"},
                        "equity_percent_change_coefficient": {"value": 0.0, "label": "equity weight"},
                    },
                    "equity_parameters": {
                        "service_intensity_coefficient": {"value": 1.0, "label": "service intensity"},
                        "waiting_time_coefficient": {"value": 1.0, "label": "waiting time"},
                    },
                    "ridership_assumptions": {
                        "route_average_trip_fraction": {"value": 0.5, "label": "trip fraction"},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.synthetic_sf_tracts = gpd.GeoDataFrame(
            {
                "geoid": ["06075000100", "06075000200"],
                "epc_2050": [1, 0],
                "epc_class": ["Tier 1", "Not EPC"],
                "tract_population": [1000, 800],
            },
            geometry=[
                Polygon([(-0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (-0.5, 0.5)]),
                Polygon([(-0.5, 0.5), (0.5, 0.5), (0.5, 1.5), (-0.5, 1.5)]),
            ],
            crs="EPSG:4326",
        )
        self.parameters = load_parameters(self.parameters_path)
        self.domain = load_route_fleet_domain(
            self.routes_path,
            route_stops_path=self.route_stops_path,
            equity_data=build_sf_equity_data_bundle(
                epc_tracts=self.synthetic_sf_tracts.loc[:, ["geoid", "epc_2050", "epc_class", "tract_population", "geometry"]]
                .rename(columns={"tract_population": "tot_pop"}),
                census_tracts=gpd.GeoDataFrame(
                    {
                        "GEOID": ["06075000100", "06075000200"],
                        "COUNTYFP": ["075", "075"],
                    },
                    geometry=self.synthetic_sf_tracts.geometry,
                    crs="EPSG:4326",
                ),
            ),
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_distance_estimation_is_deterministic_for_fixed_stops(self) -> None:
        domain2 = load_route_fleet_domain(
            self.routes_path,
            route_stops_path=self.route_stops_path,
            equity_data=build_sf_equity_data_bundle(
                epc_tracts=self.synthetic_sf_tracts.loc[:, ["geoid", "epc_2050", "epc_class", "tract_population", "geometry"]]
                .rename(columns={"tract_population": "tot_pop"}),
                census_tracts=gpd.GeoDataFrame(
                    {
                        "GEOID": ["06075000100", "06075000200"],
                        "COUNTYFP": ["075", "075"],
                    },
                    geometry=self.synthetic_sf_tracts.geometry,
                    crs="EPSG:4326",
                ),
            ),
        )

        self.assertEqual(self.domain.route_driver_estimates, domain2.route_driver_estimates)
        r1 = self.domain.route_driver_estimates[0]
        self.assertAlmostEqual(r1.horizontal_one_way_distance_miles, 69.093, places=2)
        self.assertAlmostEqual(r1.horizontal_round_trip_distance_miles, 138.186, places=2)
        self.assertAlmostEqual(r1.one_way_distance_miles, r1.horizontal_one_way_distance_miles, places=6)
        self.assertAlmostEqual(r1.round_trip_distance_miles, r1.horizontal_round_trip_distance_miles, places=6)
        self.assertGreater(r1.three_d_one_way_distance_miles, r1.horizontal_one_way_distance_miles)
        self.assertGreater(r1.three_d_round_trip_distance_miles, r1.horizontal_round_trip_distance_miles)
        self.assertAlmostEqual(r1.uphill_gain_one_way_miles, 0.5, places=3)
        self.assertAlmostEqual(r1.uphill_gain_round_trip_miles, 1.0, places=3)
        self.assertGreater(r1.uphill_gain_round_trip_feet, 0.0)
        self.assertNotEqual(r1.vehicle_type_category, "unknown")

    def test_vehicle_miles_and_hours_are_finite_and_non_negative(self) -> None:
        breakdown = compute_cost_breakdown(self.domain.baseline, domain=self.domain, parameters=self.parameters)

        self.assertTrue(math.isfinite(breakdown.annual_total_cost))
        for route_driver, route in zip(self.domain.route_driver_estimates, breakdown.route_breakdowns):
            self.assertGreaterEqual(route.baseline_annual_vehicle_miles, 0.0)
            self.assertGreaterEqual(route.candidate_annual_vehicle_miles, 0.0)
            self.assertGreaterEqual(route.baseline_annual_vehicle_hours, 0.0)
            self.assertGreaterEqual(route.candidate_annual_vehicle_hours, 0.0)
            self.assertTrue(math.isfinite(route.candidate_annual_vehicle_miles))
            self.assertTrue(math.isfinite(route.candidate_annual_vehicle_hours))
            self.assertGreaterEqual(route_driver.horizontal_one_way_distance_miles, 0.0)
            self.assertGreaterEqual(route_driver.three_d_one_way_distance_miles, route_driver.horizontal_one_way_distance_miles)
            self.assertGreaterEqual(route_driver.uphill_gain_one_way_miles, 0.0)

    def test_missing_elevation_rows_do_not_crash_and_emit_notes(self) -> None:
        r2 = self.domain.route_driver_estimates[1]

        self.assertAlmostEqual(r2.three_d_one_way_distance_miles, r2.horizontal_one_way_distance_miles, places=6)
        self.assertEqual(r2.uphill_gain_round_trip_miles, 0.0)
        self.assertTrue(any("Missing stop elevations were treated as zero elevation change" in note for note in r2.notes))
        self.assertEqual(r2.vehicle_type_source, "peak_vehicles_by_route.csv:VehicleType_2025")

    def test_candidate_vectors_change_costs_monotonically(self) -> None:
        baseline = compute_cost_breakdown((1, 2), domain=self.domain, parameters=self.parameters)
        expanded = compute_cost_breakdown((2, 2), domain=self.domain, parameters=self.parameters)
        reduced = compute_cost_breakdown((0, 2), domain=self.domain, parameters=self.parameters)

        base_r1 = baseline.route_breakdowns[0]
        exp_r1 = expanded.route_breakdowns[0]
        red_r1 = reduced.route_breakdowns[0]

        self.assertGreater(exp_r1.candidate_annual_vehicle_miles, base_r1.candidate_annual_vehicle_miles)
        self.assertGreater(exp_r1.annual_total_cost, base_r1.annual_total_cost)
        self.assertEqual(red_r1.candidate_annual_vehicle_miles, 0.0)
        self.assertEqual(red_r1.candidate_annual_vehicle_hours, 0.0)
        self.assertLess(reduced.annual_total_cost, baseline.annual_total_cost)

    def test_stop_assignments_map_most_points_to_sf_tracts(self) -> None:
        route_stops = assign_stops_to_sf_tracts(
            self.domain.stop_tract_assignments.drop(columns=["geoid"]),
            self.synthetic_sf_tracts,
        )

        self.assertGreaterEqual(int(route_stops["geoid"].notna().sum()), 5)
        self.assertIn("06075000100", set(route_stops["geoid"].dropna().astype(str)))
        self.assertIn("06075000200", set(route_stops["geoid"].dropna().astype(str)))

    def test_route_tract_summaries_are_deterministic_and_finite(self) -> None:
        coverage1 = self.domain.route_tract_coverage
        coverage2 = load_route_fleet_domain(
            self.routes_path,
            route_stops_path=self.route_stops_path,
            equity_data=build_sf_equity_data_bundle(
                epc_tracts=self.synthetic_sf_tracts.loc[:, ["geoid", "epc_2050", "epc_class", "tract_population", "geometry"]]
                .rename(columns={"tract_population": "tot_pop"}),
                census_tracts=gpd.GeoDataFrame(
                    {
                        "GEOID": ["06075000100", "06075000200"],
                        "COUNTYFP": ["075", "075"],
                    },
                    geometry=self.synthetic_sf_tracts.geometry,
                    crs="EPSG:4326",
                ),
            ),
        ).route_tract_coverage

        self.assertEqual(coverage1, coverage2)
        self.assertEqual(coverage1[0].route_id, "R1")
        self.assertTrue(coverage1[0].touches_epc_tract)
        self.assertEqual(dict(coverage1[0].stop_counts_by_tract)["06075000100"], 4)
        self.assertTrue(all(len(summary.tract_geoids) >= 0 for summary in coverage1))

    @mock.patch("model.get_default_equity_data")
    @mock.patch("model.find_data_file")
    def test_explicit_default_route_stops_path_still_builds_equity_cache(
        self,
        find_data_file_mock: mock.Mock,
        get_default_equity_data_mock: mock.Mock,
    ) -> None:
        def fake_find_data_file(name: str) -> Path:
            if name == "simplified_bus_route_stops.csv":
                return self.route_stops_path
            if name == "simplified_bus_routes.csv":
                return self.routes_path
            return resolve_data_file(name)

        find_data_file_mock.side_effect = fake_find_data_file
        get_default_equity_data_mock.return_value = build_sf_equity_data_bundle(
            epc_tracts=self.synthetic_sf_tracts.loc[
                :, ["geoid", "epc_2050", "epc_class", "tract_population", "geometry"]
            ].rename(columns={"tract_population": "tot_pop"}),
            census_tracts=gpd.GeoDataFrame(
                {
                    "GEOID": ["06075000100", "06075000200"],
                    "COUNTYFP": ["075", "075"],
                },
                geometry=self.synthetic_sf_tracts.geometry,
                crs="EPSG:4326",
            ),
            population_table=self.synthetic_sf_tracts.loc[:, ["geoid", "tract_population"]].rename(
                columns={"tract_population": "population"}
            ),
        )

        domain = load_route_fleet_domain(self.routes_path, route_stops_path=self.route_stops_path)

        self.assertEqual(len(domain.equity_tracts), 2)
        self.assertTrue(any(len(summary.tract_geoids) > 0 for summary in domain.route_tract_coverage))
        self.assertTrue(any(domain.route_metadata["touches_epc_tract"]))

    def test_candidate_tract_service_changes_monotonically_with_fleet(self) -> None:
        baseline = {row.geoid: row for row in compute_tract_service_access_summaries((1, 2), domain=self.domain)}
        expanded = {row.geoid: row for row in compute_tract_service_access_summaries((2, 2), domain=self.domain)}
        reduced = {row.geoid: row for row in compute_tract_service_access_summaries((0, 2), domain=self.domain)}

        self.assertGreater(
            expanded["06075000100"].candidate_service_intensity,
            baseline["06075000100"].candidate_service_intensity,
        )
        self.assertLess(
            reduced["06075000100"].candidate_service_intensity,
            baseline["06075000100"].candidate_service_intensity,
        )


class TestPhaseThreeObjectiveAndReporting(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.routes_path = Path(self.tempdir.name) / "routes.csv"
        self.route_stops_path = Path(self.tempdir.name) / "route_stops.csv"
        self.parameters_path = Path(self.tempdir.name) / "parameters.json"
        self.alt_parameters_path = Path(self.tempdir.name) / "alt_parameters.json"
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
            "emissions_parameters": {
                "car_emissions_grams_per_mile": {"value": 353.802111, "label": "car"},
                "car_ownership_probability": {"value": 0.7, "label": "ownership"},
                "bus_base_emissions_grams_per_mile": {"value": 2830.0, "label": "bus base"},
                "bus_climb_penalty_grams_per_mile": {"value": 5660.0, "label": "bus climb"},
            },
            "objective_weights": {
                "cost_percent_change_coefficient": {"value": 2.0, "label": "cost weight"},
                "emissions_percent_change_coefficient": {"value": 1.0, "label": "emissions weight"},
                "equity_percent_change_coefficient": {"value": 0.0, "label": "equity weight"},
            },
            "equity_parameters": {
                "service_intensity_coefficient": {"value": 1.0, "label": "service intensity"},
                "waiting_time_coefficient": {"value": 1.0, "label": "waiting time"},
            },
            "ridership_assumptions": {
                "route_average_trip_fraction": {"value": 0.5, "label": "trip fraction"},
            },
        }
        alt_payload = json.loads(json.dumps(base_payload))
        alt_payload["reporting_constants"]["annual_fare_revenue"]["value"] = 999999.0
        alt_payload["reporting_constants"]["annual_advertising_revenue"]["value"] = 888888.0
        alt_payload["reporting_constants"]["annual_external_subsidies"]["value"] = 777777.0

        self.parameters_path.write_text(json.dumps(base_payload), encoding="utf-8")
        self.alt_parameters_path.write_text(json.dumps(alt_payload), encoding="utf-8")
        self.synthetic_sf_tracts = gpd.GeoDataFrame(
            {
                "geoid": ["06075000100", "06075000200"],
                "epc_2050": [1, 0],
                "epc_class": ["Tier 1", "Not EPC"],
                "tract_population": [1000, 800],
            },
            geometry=[
                Polygon([(-0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (-0.5, 0.5)]),
                Polygon([(-0.5, 0.5), (0.5, 0.5), (0.5, 1.5), (-0.5, 1.5)]),
            ],
            crs="EPSG:4326",
        )
        self.parameters = load_parameters(self.parameters_path)
        self.alt_parameters = load_parameters(self.alt_parameters_path)
        self.domain = load_route_fleet_domain(
            self.routes_path,
            route_stops_path=self.route_stops_path,
            equity_data=build_sf_equity_data_bundle(
                epc_tracts=self.synthetic_sf_tracts.loc[
                    :, ["geoid", "epc_2050", "epc_class", "tract_population", "geometry"]
                ].rename(columns={"tract_population": "tot_pop"}),
                census_tracts=gpd.GeoDataFrame(
                    {
                        "GEOID": ["06075000100", "06075000200"],
                        "COUNTYFP": ["075", "075"],
                    },
                    geometry=self.synthetic_sf_tracts.geometry,
                    crs="EPSG:4326",
                ),
            ),
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_reporting_constants_do_not_change_objective_ranking(self) -> None:
        baseline = compute_cost_breakdown((1, 2), domain=self.domain, parameters=self.parameters)
        baseline_alt = compute_cost_breakdown((1, 2), domain=self.domain, parameters=self.alt_parameters)

        self.assertEqual(baseline.objective_cost, baseline_alt.objective_cost)
        self.assertNotEqual(baseline.annual_total_revenue, baseline_alt.annual_total_revenue)

    def test_global_fleet_conservation_keeps_capital_cost_at_zero_for_swaps(self) -> None:
        swapped = compute_cost_breakdown((0, 3), domain=self.domain, parameters=self.parameters)

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
        self.assertIn("delta_net_emissions_grams", result.route_emissions_delta_table.columns)
        self.assertAlmostEqual(
            result.route_cost_delta_table["delta_total_cost"].sum(),
            result.annual_cost_delta_vs_baseline,
        )
        self.assertAlmostEqual(
            result.best_budget_slack,
            result.best_cost_breakdown.reporting_constants.annual_budget_ceiling.value
            - result.best_cost_breakdown.annual_total_cost,
        )
        self.assertAlmostEqual(
            result.best_objective_breakdown.total_combined_objective,
            result.best_objective_breakdown.cost.weighted_contribution
            + result.best_objective_breakdown.emissions.weighted_contribution
            + result.best_objective_breakdown.equity.weighted_contribution,
        )
        self.assertIsNotNone(result.initial_equity_breakdown)
        self.assertIsNotNone(result.best_equity_breakdown)
        self.assertIn("delta_utility", result.tract_equity_delta_table.columns)

    def test_negative_baseline_emissions_keep_percent_delta_sign_aligned_with_worse_outcomes(self) -> None:
        baseline = compute_objective_breakdown(
            (1, 2),
            domain=self.domain,
            parameters=self.parameters,
            weekday_ridership={"1": 10000.0, "2": 10000.0},
        )
        worsened = compute_objective_breakdown(
            (0, 2),
            domain=self.domain,
            parameters=self.parameters,
            weekday_ridership={"1": 10000.0, "2": 10000.0},
        )

        self.assertLess(baseline.emissions.baseline_value, 0.0)
        self.assertGreater(worsened.emissions.absolute_delta, 0.0)
        self.assertGreater(worsened.emissions.percent_delta, 0.0)

    def test_search_uses_supplied_parameters_for_objective(self) -> None:
        fast_parameters_payload = {
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
            "emissions_parameters": {
                "car_emissions_grams_per_mile": {"value": 353.802111, "label": "car"},
                "car_ownership_probability": {"value": 0.7, "label": "ownership"},
                "bus_base_emissions_grams_per_mile": {"value": 2830.0, "label": "bus base"},
                "bus_climb_penalty_grams_per_mile": {"value": 5660.0, "label": "bus climb"},
            },
            "objective_weights": {
                "cost_percent_change_coefficient": {"value": 1.0, "label": "cost weight"},
                "emissions_percent_change_coefficient": {"value": 1.0, "label": "emissions weight"},
                "equity_percent_change_coefficient": {"value": 0.0, "label": "equity weight"},
            },
            "equity_parameters": {
                "service_intensity_coefficient": {"value": 1.0, "label": "service intensity"},
                "waiting_time_coefficient": {"value": 1.0, "label": "waiting time"},
            },
            "ridership_assumptions": {
                "route_average_trip_fraction": {"value": 0.5, "label": "trip fraction"},
            },
        }
        fast_path = Path(self.tempdir.name) / "fast_parameters.json"
        fast_path.write_text(json.dumps(fast_parameters_payload), encoding="utf-8")
        fast_parameters = load_parameters(fast_path)

        default_result = run_route_fleet_search(
            domain=self.domain,
            config=SearchConfig(max_iterations=5, nbhd_add_lim=4, nbhd_drop_lim=4, nbhd_swap_lim=6),
            seed=3,
            parameters=self.parameters,
        )
        fast_result = run_route_fleet_search(
            domain=self.domain,
            config=SearchConfig(max_iterations=5, nbhd_add_lim=4, nbhd_drop_lim=4, nbhd_swap_lim=6),
            seed=3,
            parameters=fast_parameters,
        )

        default_candidate_objective = objective_function(
            (0, 3),
            domain=self.domain,
            parameters=self.parameters,
        )
        fast_candidate_objective = objective_function(
            (0, 3),
            domain=self.domain,
            parameters=fast_parameters,
        )

        self.assertNotEqual(default_candidate_objective, fast_candidate_objective)
        self.assertEqual(
            default_result.initial_objective,
            default_result.initial_objective_breakdown.total_combined_objective,
        )
        self.assertEqual(
            fast_result.initial_objective,
            fast_result.initial_objective_breakdown.total_combined_objective,
        )

    def test_equity_tract_utilities_are_finite(self) -> None:
        breakdown = compute_equity_breakdown((1, 2), domain=self.domain, parameters=self.parameters)

        self.assertTrue(math.isfinite(breakdown.current_population_gap))
        self.assertTrue(math.isfinite(breakdown.current_area_gap))
        self.assertGreaterEqual(len(breakdown.tract_breakdowns), 2)
        for tract in breakdown.tract_breakdowns:
            self.assertTrue(math.isfinite(tract.baseline_utility))
            self.assertTrue(math.isfinite(tract.candidate_utility))
            self.assertTrue(math.isfinite(tract.baseline_waiting_time_proxy))
            self.assertTrue(math.isfinite(tract.candidate_waiting_time_proxy))

    def test_equity_gaps_aggregate_exactly_from_tract_rows(self) -> None:
        breakdown = compute_equity_breakdown((1, 2), domain=self.domain, parameters=self.parameters)
        epc = [tract for tract in breakdown.tract_breakdowns if tract.epc_2050 == 1]
        non_epc = [tract for tract in breakdown.tract_breakdowns if tract.epc_2050 == 0]

        baseline_epc_weighted = sum(tract.tract_population * tract.baseline_utility for tract in epc) / sum(
            tract.tract_population for tract in epc
        )
        baseline_non_epc_weighted = sum(
            tract.tract_population * tract.baseline_utility for tract in non_epc
        ) / sum(tract.tract_population for tract in non_epc)
        current_epc_weighted = sum(tract.tract_population * tract.candidate_utility for tract in epc) / sum(
            tract.tract_population for tract in epc
        )
        current_non_epc_weighted = sum(
            tract.tract_population * tract.candidate_utility for tract in non_epc
        ) / sum(tract.tract_population for tract in non_epc)

        self.assertAlmostEqual(
            breakdown.baseline_population_gap,
            abs(baseline_non_epc_weighted - baseline_epc_weighted),
        )
        self.assertAlmostEqual(
            breakdown.current_population_gap,
            abs(current_non_epc_weighted - current_epc_weighted),
        )
        self.assertAlmostEqual(
            breakdown.baseline_area_gap,
            abs(
                sum(tract.baseline_utility for tract in non_epc) / len(non_epc)
                - sum(tract.baseline_utility for tract in epc) / len(epc)
            ),
        )
        self.assertAlmostEqual(
            breakdown.current_area_gap,
            abs(
                sum(tract.candidate_utility for tract in non_epc) / len(non_epc)
                - sum(tract.candidate_utility for tract in epc) / len(epc)
            ),
        )

    def test_epc_heavy_route_changes_move_equity_gap_in_expected_direction(self) -> None:
        baseline = compute_equity_breakdown((1, 2), domain=self.domain, parameters=self.parameters)
        expanded_epc_route = compute_equity_breakdown((2, 2), domain=self.domain, parameters=self.parameters)
        reduced_epc_route = compute_equity_breakdown((0, 2), domain=self.domain, parameters=self.parameters)

        self.assertGreater(
            expanded_epc_route.current_population_gap,
            baseline.current_population_gap,
        )
        self.assertLess(
            reduced_epc_route.current_population_gap,
            baseline.current_population_gap,
        )


class TestPhaseThreeEmissionsCalculations(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.routes_path = Path(self.tempdir.name) / "routes.csv"
        self.route_stops_path = Path(self.tempdir.name) / "route_stops.csv"
        self.parameters_path = Path(self.tempdir.name) / "parameters.json"
        self.ridership_path = Path(self.tempdir.name) / "ridership.csv"
        self.routes_path.write_text(PHASE2_ROUTE_FIXTURE, encoding="utf-8")
        self.route_stops_path.write_text(PHASE2_ROUTE_STOPS_FIXTURE, encoding="utf-8")
        self.ridership_path.write_text(PHASE3_RIDERSHIP_FIXTURE, encoding="utf-8")
        self.parameters_path.write_text(
            json.dumps(
                {
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
                        "weekday_service_days_per_year": {"value": 10.0, "label": "days"},
                    },
                    "emissions_parameters": {
                        "car_emissions_grams_per_mile": {"value": 100.0, "label": "car"},
                        "car_ownership_probability": {"value": 0.5, "label": "ownership"},
                        "bus_base_emissions_grams_per_mile": {"value": 1000.0, "label": "bus base"},
                        "bus_climb_penalty_grams_per_mile": {"value": 500.0, "label": "bus climb"},
                    },
                    "objective_weights": {
                        "cost_percent_change_coefficient": {"value": 1.0, "label": "cost weight"},
                        "emissions_percent_change_coefficient": {"value": 1.0, "label": "emissions weight"},
                        "equity_percent_change_coefficient": {"value": 0.0, "label": "equity weight"},
                    },
                    "equity_parameters": {
                        "service_intensity_coefficient": {"value": 1.0, "label": "service intensity"},
                        "waiting_time_coefficient": {"value": 1.0, "label": "waiting time"},
                    },
                    "ridership_assumptions": {
                        "route_average_trip_fraction": {"value": 0.5, "label": "trip fraction"},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.parameters = load_parameters(self.parameters_path)
        self.domain = load_route_fleet_domain(self.routes_path, route_stops_path=self.route_stops_path)
        self.weekday_ridership = _load_weekday_ridership(self.ridership_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_weekday_ridership_loader_uses_latest_values(self) -> None:
        self.assertEqual(self.weekday_ridership["1"], 1500.0)
        self.assertEqual(self.weekday_ridership["2"], 250.0)

    def test_baseline_and_candidate_emissions_are_finite(self) -> None:
        breakdown = compute_emissions_breakdown(
            self.domain.baseline,
            domain=self.domain,
            parameters=self.parameters,
            weekday_ridership=self.weekday_ridership,
        )

        self.assertTrue(math.isfinite(breakdown.baseline_total_emissions_grams))
        self.assertTrue(math.isfinite(breakdown.candidate_total_emissions_grams))
        for route in breakdown.route_breakdowns:
            self.assertTrue(math.isfinite(route.baseline_net_emissions_grams))
            self.assertTrue(math.isfinite(route.candidate_net_emissions_grams))

    def test_increasing_service_increases_bus_emissions_and_avoided_rider_emissions(self) -> None:
        baseline = compute_emissions_breakdown(
            (1, 2),
            domain=self.domain,
            parameters=self.parameters,
            weekday_ridership=self.weekday_ridership,
        )
        expanded = compute_emissions_breakdown(
            (2, 2),
            domain=self.domain,
            parameters=self.parameters,
            weekday_ridership=self.weekday_ridership,
        )

        base_r1 = baseline.route_breakdowns[0]
        exp_r1 = expanded.route_breakdowns[0]

        self.assertGreater(exp_r1.candidate_bus_emissions_grams, base_r1.candidate_bus_emissions_grams)
        self.assertGreater(exp_r1.candidate_rider_emissions_avoided_grams, base_r1.candidate_rider_emissions_avoided_grams)
        self.assertGreater(exp_r1.candidate_riders, base_r1.candidate_riders)

    def test_net_emissions_aggregate_exactly_from_routes(self) -> None:
        breakdown = compute_emissions_breakdown(
            (2, 1),
            domain=self.domain,
            parameters=self.parameters,
            weekday_ridership=self.weekday_ridership,
        )

        self.assertAlmostEqual(
            breakdown.baseline_total_emissions_grams,
            sum(route.baseline_net_emissions_grams for route in breakdown.route_breakdowns),
        )
        self.assertAlmostEqual(
            breakdown.candidate_total_emissions_grams,
            sum(route.candidate_net_emissions_grams for route in breakdown.route_breakdowns),
        )
        self.assertAlmostEqual(
            breakdown.absolute_delta_emissions_grams,
            breakdown.candidate_total_emissions_grams - breakdown.baseline_total_emissions_grams,
        )


if __name__ == "__main__":
    unittest.main()
