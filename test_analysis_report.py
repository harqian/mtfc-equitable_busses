import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from model import EquityDataBundle, SearchConfig, load_parameters, load_route_fleet_domain
from scripts.generate_model_analysis_report import (
    DEFAULT_ITERATION_LADDER,
    FIGURE_FILENAMES,
    WeightScenario,
    build_route_geometries,
    classify_sacrifice,
    generate_analysis_artifacts,
    generate_figures_from_artifacts,
    parse_iteration_ladder,
    safe_ratio,
)


ROUTES_FIXTURE = """route_id,route_short_name,route_long_name,weekday_typical_buses,weekday_planned_trips,stop_count
R1,1,One,2,20,4
R2,2,Two,2,15,4
"""

ROUTE_STOPS_FIXTURE = """route_id,direction_id,route_stop_order,stop_lat,stop_lon,elevation_m
R1,0,1,37.7700,-122.4300,0
R1,0,2,37.7700,-122.4200,10
R1,1,1,37.7700,-122.4200,10
R1,1,2,37.7700,-122.4300,0
R2,0,1,37.7800,-122.4100,5
R2,0,2,37.7850,-122.4050,15
R2,1,1,37.7850,-122.4050,15
R2,1,2,37.7800,-122.4100,5
"""

PARAMETERS_FIXTURE = {
    "reporting_constants": {
        "annual_fare_revenue": {"value": 100.0, "label": "fare"},
        "annual_advertising_revenue": {"value": 10.0, "label": "ads"},
        "annual_external_subsidies": {"value": 20.0, "label": "subsidy"},
        "annual_budget_ceiling": {"value": 10000.0, "label": "budget"},
    },
    "operating_cost_parameters": {
        "labor_cost_per_vehicle_hour": {"value": 5.0, "label": "labor"},
        "maintenance_cost_per_vehicle_mile": {"value": 1.0, "label": "maintenance"},
        "energy_cost_per_vehicle_mile": {"value": 1.0, "label": "energy"},
        "annualized_capital_cost_per_vehicle": {"value": 100.0, "label": "capital"},
    },
    "estimation_assumptions": {
        "average_operating_speed_mph": {"value": 12.0, "label": "speed"},
        "deadhead_multiplier": {"value": 1.0, "label": "deadhead"},
        "dwell_recovery_multiplier": {"value": 1.0, "label": "dwell"},
        "weekday_service_days_per_year": {"value": 250.0, "label": "days"},
    },
    "emissions_parameters": {
        "car_emissions_grams_per_mile": {"value": 350.0, "label": "car"},
        "car_ownership_probability": {"value": 0.5, "label": "ownership"},
        "bus_base_emissions_grams_per_mile": {"value": 1000.0, "label": "bus"},
        "bus_climb_penalty_grams_per_mile": {"value": 10.0, "label": "climb"},
    },
    "objective_weights": {
        "cost_percent_change_coefficient": {"value": 1.0, "label": "cost"},
        "emissions_percent_change_coefficient": {"value": 1.0, "label": "emissions"},
        "equity_percent_change_coefficient": {"value": 1.0, "label": "equity"},
    },
    "equity_parameters": {
        "service_intensity_coefficient": {"value": 1.0, "label": "service"},
        "waiting_time_coefficient": {"value": 1.0, "label": "wait"},
    },
    "ridership_assumptions": {
        "route_average_trip_fraction": {"value": 0.5, "label": "trip"},
    },
}


class TestGenerateModelAnalysisReport(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.tempdir.name)
        self.routes_path = self.temp_path / "routes.csv"
        self.route_stops_path = self.temp_path / "route_stops.csv"
        self.parameters_path = self.temp_path / "parameters.json"
        self.output_path = self.temp_path / "report_output"

        self.routes_path.write_text(ROUTES_FIXTURE, encoding="utf-8")
        self.route_stops_path.write_text(ROUTE_STOPS_FIXTURE, encoding="utf-8")
        self.parameters_path.write_text(json.dumps(PARAMETERS_FIXTURE), encoding="utf-8")

        tracts = gpd.GeoDataFrame(
            {
                "geoid": ["06075000100", "06075000200"],
                "epc_2050": [1, 0],
                "epc_class": ["Tier 1", "Not EPC"],
                "tract_population": [1000.0, 800.0],
            },
            geometry=[
                Polygon([(-122.44, 37.76), (-122.41, 37.76), (-122.41, 37.78), (-122.44, 37.78)]),
                Polygon([(-122.42, 37.775), (-122.39, 37.775), (-122.39, 37.79), (-122.42, 37.79)]),
            ],
            crs="EPSG:4326",
        )
        equity_data = EquityDataBundle(
            sf_epc_tracts=tracts,
            population_field="tract_population",
            notes=("synthetic",),
        )
        self.domain = load_route_fleet_domain(
            self.routes_path,
            route_stops_path=self.route_stops_path,
            equity_data=equity_data,
        )
        self.parameters = load_parameters(self.parameters_path)
        self.scenarios = [
            WeightScenario(
                scenario_id="balanced",
                run_label="Balanced",
                cost_weight=1.0,
                emissions_weight=1.0,
                equity_weight=1.0,
            ),
            WeightScenario(
                scenario_id="equity_focus",
                run_label="Equity Focus",
                cost_weight=0.5,
                emissions_weight=0.5,
                equity_weight=2.0,
            ),
        ]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _generate_artifacts(self, *, iteration_ladder=(1, 3), output_path: Path | None = None) -> dict[str, Path]:
        out_dir = self.output_path if output_path is None else output_path
        ridership = {"R1": 400.0, "R2": 250.0, "1": 400.0, "2": 250.0}
        with mock.patch("model.get_default_weekday_ridership", return_value=ridership):
            return generate_analysis_artifacts(
                domain=self.domain,
                base_parameters=self.parameters,
                output_dir=out_dir,
                scenarios=self.scenarios,
                base_config=SearchConfig(
                    max_iterations=DEFAULT_ITERATION_LADDER[-1],
                    temp_init=0.8,
                    temp_factor=0.97,
                    tenure_init=2.0,
                    tenure_factor=1.2,
                    nonimp_in_max=4,
                    nonimp_out_max=10,
                    nbhd_add_lim=4,
                    nbhd_drop_lim=4,
                    nbhd_swap_lim=6,
                    attractive_max=4,
                ),
                seed=9,
                routes_csv=self.routes_path,
                route_stops_csv=self.route_stops_path,
                parameters_json=self.parameters_path,
                scenarios_json=self.temp_path / "scenarios.json",
                iteration_ladder=iteration_ladder,
            )

    def test_generate_analysis_artifacts_writes_expected_outputs(self) -> None:
        paths = self._generate_artifacts()

        self.assertTrue(paths["figures_dir"].is_dir())
        for key in (
            "run_manifest",
            "scenario_summary",
            "strategy_tradeoffs",
            "iteration_events",
            "objective_history",
            "route_deltas",
            "tract_deltas",
        ):
            self.assertTrue(paths[key].exists(), key)
            self.assertGreater(paths[key].stat().st_size, 0, key)

        manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["iteration_ladder"], [1, 3])
        self.assertEqual(manifest["scenario_count"], 2)
        self.assertEqual(len(manifest["runs"]), 4)
        self.assertIn("selected_best_run_id", manifest)

        scenario_summary = pd.read_csv(paths["scenario_summary"])
        strategy_tradeoffs = pd.read_csv(paths["strategy_tradeoffs"])
        iteration_events = pd.read_csv(paths["iteration_events"])
        objective_history = pd.read_csv(paths["objective_history"])
        route_deltas = pd.read_csv(paths["route_deltas"])
        tract_deltas = pd.read_csv(paths["tract_deltas"])

        self.assertEqual(len(scenario_summary), 4)
        self.assertEqual(len(strategy_tradeoffs), 4)
        self.assertEqual(sorted(strategy_tradeoffs["iterations"].tolist()), [1, 1, 3, 3])
        self.assertEqual(len(iteration_events), 8)
        self.assertEqual(len(objective_history), 8)
        for column in (
            "annual_cost_delta",
            "annual_net_emissions_delta_grams",
            "equity_population_gap_delta",
            "cost_delta_per_gram_net_emissions_change_status",
            "cost_delta_per_equity_gap_change_status",
            "net_emissions_delta_per_equity_gap_change_status",
            "is_sacrifice_case",
        ):
            self.assertIn(column, strategy_tradeoffs.columns)
        self.assertIn("delta_total_cost", route_deltas.columns)
        self.assertIn("delta_net_emissions_grams", route_deltas.columns)
        self.assertIn("touches_epc_tract", route_deltas.columns)
        self.assertIn("delta_utility", tract_deltas.columns)

    def test_generate_analysis_artifacts_respects_custom_iteration_ladder(self) -> None:
        custom_ladder = parse_iteration_ladder("1,3,7")
        paths = self._generate_artifacts(iteration_ladder=custom_ladder, output_path=self.temp_path / "custom_output")
        manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
        scenario_summary = pd.read_csv(paths["scenario_summary"])
        self.assertEqual(manifest["iteration_ladder"], [1, 3, 7])
        self.assertEqual(sorted(scenario_summary["iterations"].unique().tolist()), [1, 3, 7])

    def test_derived_metric_helpers_flag_sacrifice_and_zero_denominators(self) -> None:
        row = pd.Series(
            {
                "cost_percent_delta": 2.0,
                "emissions_percent_delta": -4.0,
                "equity_percent_delta": -1.0,
            }
        )
        flags = classify_sacrifice(row)
        self.assertTrue(flags["sacrifice_cost_for_emissions_equity"])
        self.assertTrue(flags["is_sacrifice_case"])

        ratio, status = safe_ratio(10.0, 0.0)
        self.assertIsNone(ratio)
        self.assertEqual(status, "undefined_zero_denominator")

    def test_build_route_geometries_produces_non_empty_route_features(self) -> None:
        geoms = build_route_geometries(self.route_stops_path)
        self.assertFalse(geoms.empty)
        self.assertEqual(sorted(geoms["route_id"].unique().tolist()), ["R1", "R2"])
        self.assertTrue(all(geom.length > 0 for geom in geoms.geometry))

    def test_generate_figures_from_saved_artifacts_creates_expected_pngs(self) -> None:
        paths = self._generate_artifacts(iteration_ladder=(1, 2), output_path=self.temp_path / "fig_output")
        figures = generate_figures_from_artifacts(paths["run_manifest"].parent, route_stops_csv=self.route_stops_path)
        self.assertEqual(sorted(path.name for path in figures), sorted(FIGURE_FILENAMES))
        for figure in figures:
            self.assertTrue(figure.exists(), figure.name)
            self.assertGreater(figure.stat().st_size, 0, figure.name)


if __name__ == "__main__":
    unittest.main()
