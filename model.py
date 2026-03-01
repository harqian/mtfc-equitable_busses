import hashlib
import json
import math
import pathlib
import random
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Optional, Sequence

import folium
import geopandas as gpd
import pandas as pd

from data_utils import find_data_file


MTC_EPC_GEOJSON_URL = (
    "https://hub.arcgis.com/api/download/v1/items/"
    "28a03a46fe9c4df0a29746d6f8c633c8/geojson?redirect=true&layers=0"
)
CENSUS_TRACTS_ZIP_URL = "https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_06_tract.zip"
SF_COUNTY_FIPS = "075"
SF_TRACT_GEOID_PREFIX = "06075"
_EQUITY_POPULATION_FIELDS = (
    "tot_pop",
    "totpop",
    "population",
    "total_population",
)


@dataclass(frozen=True)
class ParameterValue:
    value: float
    label: str


@dataclass(frozen=True)
class ReportingConstants:
    annual_fare_revenue: ParameterValue
    annual_advertising_revenue: ParameterValue
    annual_external_subsidies: ParameterValue
    annual_budget_ceiling: ParameterValue


@dataclass(frozen=True)
class OperatingCostParameters:
    labor_cost_per_vehicle_hour: ParameterValue
    maintenance_cost_per_vehicle_mile: ParameterValue
    energy_cost_per_vehicle_mile: ParameterValue
    annualized_capital_cost_per_vehicle: ParameterValue


@dataclass(frozen=True)
class EstimationAssumptions:
    average_operating_speed_mph: ParameterValue
    deadhead_multiplier: ParameterValue
    dwell_recovery_multiplier: ParameterValue
    weekday_service_days_per_year: ParameterValue


@dataclass(frozen=True)
class CostParameters:
    reporting_constants: ReportingConstants
    operating_cost_parameters: OperatingCostParameters
    estimation_assumptions: EstimationAssumptions
    emissions_parameters: "EmissionsParameters"
    objective_weights: "ObjectiveWeights"
    equity_parameters: "EquityParameters"
    ridership_assumptions: "RidershipAssumptions"


@dataclass(frozen=True)
class EmissionsParameters:
    car_emissions_grams_per_mile: ParameterValue
    car_ownership_probability: ParameterValue
    bus_base_emissions_grams_per_mile: ParameterValue
    bus_climb_penalty_grams_per_mile: ParameterValue


@dataclass(frozen=True)
class ObjectiveWeights:
    cost_percent_change_coefficient: ParameterValue
    emissions_percent_change_coefficient: ParameterValue
    equity_percent_change_coefficient: ParameterValue


@dataclass(frozen=True)
class EquityParameters:
    service_intensity_coefficient: ParameterValue
    waiting_time_coefficient: ParameterValue


@dataclass(frozen=True)
class RidershipAssumptions:
    route_average_trip_fraction: ParameterValue


@dataclass(frozen=True)
class EquityTractMetadata:
    geoid: str
    epc_2050: int
    epc_class: str
    population: float


@dataclass(frozen=True)
class TractServiceAccessSummary:
    geoid: str
    route_count: int
    stop_count: int
    baseline_service_intensity: float
    candidate_service_intensity: float


@dataclass(frozen=True)
class RouteTractCoverageSummary:
    route_id: str
    tract_geoids: tuple[str, ...]
    stop_counts_by_tract: tuple[tuple[str, int], ...]
    touches_epc_tract: bool


@dataclass(frozen=True)
class EquityDataBundle:
    sf_epc_tracts: gpd.GeoDataFrame
    population_field: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RouteFleetDomain:
    route_ids: tuple[str, ...]
    baseline: tuple[int, ...]
    lower_bounds: tuple[int, ...]
    upper_bounds: tuple[int, ...]
    service_weights: tuple[float, ...]
    route_driver_estimates: tuple["RouteCostDriverEstimate", ...]
    route_metadata: pd.DataFrame
    stop_tract_assignments: pd.DataFrame
    route_tract_coverage: tuple["RouteTractCoverageSummary", ...]
    equity_tracts: tuple[EquityTractMetadata, ...]


@dataclass(frozen=True)
class Move:
    move_type: str
    add_index: Optional[int] = None
    drop_index: Optional[int] = None


@dataclass(frozen=True)
class SearchConfig:
    max_iterations: int = 100
    temp_init: float = 1.0
    temp_factor: float = 0.98
    tenure_init: float = 3.0
    tenure_factor: float = 1.25
    nonimp_in_max: int = 8
    nonimp_out_max: int = 30
    nbhd_add_lim: int = 25
    nbhd_drop_lim: int = 25
    nbhd_swap_lim: int = 60
    attractive_max: int = 20
    step: int = 1
    require_global_fleet_conservation: bool = True


@dataclass(frozen=True)
class SearchResult:
    best_vector: tuple[int, ...]
    best_objective: float
    initial_vector: tuple[int, ...]
    initial_objective: float
    iterations_completed: int
    accepted_improving_moves: int
    accepted_nonimproving_moves: int
    events: tuple[dict, ...]
    best_route_table: pd.DataFrame
    initial_cost_breakdown: "SystemCostBreakdown"
    best_cost_breakdown: "SystemCostBreakdown"
    initial_emissions_breakdown: "SystemEmissionsBreakdown"
    best_emissions_breakdown: "SystemEmissionsBreakdown"
    initial_equity_breakdown: "SystemEquityBreakdown"
    best_equity_breakdown: "SystemEquityBreakdown"
    initial_objective_breakdown: "ObjectiveBreakdown"
    best_objective_breakdown: "ObjectiveBreakdown"
    annual_cost_delta_vs_baseline: float
    initial_budget_slack: float
    best_budget_slack: float
    route_cost_delta_table: pd.DataFrame
    route_emissions_delta_table: pd.DataFrame
    tract_equity_delta_table: pd.DataFrame


@dataclass(frozen=True)
class CandidateNeighbor:
    move: Move
    vector: tuple[int, ...]
    objective: float


@dataclass(frozen=True)
class RouteCostBreakdown:
    route_id: str
    baseline_fleet: int
    candidate_fleet: int
    fleet_delta: int
    service_scale: float
    one_way_distance_miles: float
    round_trip_distance_miles: float
    baseline_annual_vehicle_miles: float
    candidate_annual_vehicle_miles: float
    baseline_annual_vehicle_hours: float
    candidate_annual_vehicle_hours: float
    annual_labor_cost: float
    annual_maintenance_cost: float
    annual_energy_cost: float
    annual_capital_cost: float
    annual_total_cost: float
    drivers_estimated: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SystemCostBreakdown:
    objective_cost: float
    annual_labor_cost: float
    annual_maintenance_cost: float
    annual_energy_cost: float
    annual_capital_cost: float
    annual_total_cost: float
    annual_total_revenue: float
    annual_budget_slack: float
    baseline_total_fleet: int
    candidate_total_fleet: int
    net_new_fleet: int
    route_breakdowns: tuple[RouteCostBreakdown, ...]
    reporting_constants: ReportingConstants
    estimation_assumptions: EstimationAssumptions
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RouteEmissionsBreakdown:
    route_id: str
    baseline_fleet: int
    candidate_fleet: int
    fleet_delta: int
    service_scale: float
    baseline_riders: float
    candidate_riders: float
    average_rider_trip_distance_miles: float
    baseline_bus_emissions_grams: float
    candidate_bus_emissions_grams: float
    baseline_rider_emissions_avoided_grams: float
    candidate_rider_emissions_avoided_grams: float
    baseline_net_emissions_grams: float
    candidate_net_emissions_grams: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SystemEmissionsBreakdown:
    baseline_total_emissions_grams: float
    candidate_total_emissions_grams: float
    absolute_delta_emissions_grams: float
    percent_delta_emissions: float
    route_breakdowns: tuple[RouteEmissionsBreakdown, ...]
    emissions_parameters: EmissionsParameters
    ridership_assumptions: RidershipAssumptions
    notes: tuple[str, ...]


@dataclass(frozen=True)
class EquityTractBreakdown:
    geoid: str
    epc_2050: int
    epc_class: str
    tract_population: float
    route_count_touching_tract: int
    baseline_service_intensity: float
    candidate_service_intensity: float
    baseline_waiting_time_proxy: float
    candidate_waiting_time_proxy: float
    baseline_utility: float
    candidate_utility: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SystemEquityBreakdown:
    baseline_population_gap: float
    current_population_gap: float
    absolute_population_gap_delta: float
    percent_population_gap_delta: float
    baseline_area_gap: float
    current_area_gap: float
    absolute_area_gap_delta: float
    percent_area_gap_delta: float
    baseline_epc_weighted_mean_utility: float
    current_epc_weighted_mean_utility: float
    baseline_non_epc_weighted_mean_utility: float
    current_non_epc_weighted_mean_utility: float
    baseline_epc_mean_utility: float
    current_epc_mean_utility: float
    baseline_non_epc_mean_utility: float
    current_non_epc_mean_utility: float
    tract_breakdowns: tuple[EquityTractBreakdown, ...]
    equity_parameters: EquityParameters
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ObjectivePillarBreakdown:
    baseline_value: float
    current_value: float
    absolute_delta: float
    percent_delta: float
    coefficient: float
    weighted_contribution: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ObjectiveBreakdown:
    cost: ObjectivePillarBreakdown
    emissions: ObjectivePillarBreakdown
    equity: ObjectivePillarBreakdown
    total_combined_objective: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RouteCostDriverEstimate:
    route_id: str
    baseline_fleet: int
    weekday_planned_trips: float
    direction_count: int
    one_way_distance_miles: float
    round_trip_distance_miles: float
    horizontal_one_way_distance_miles: float
    horizontal_round_trip_distance_miles: float
    three_d_one_way_distance_miles: float
    three_d_round_trip_distance_miles: float
    uphill_gain_one_way_miles: float
    uphill_gain_round_trip_miles: float
    uphill_gain_one_way_feet: float
    uphill_gain_round_trip_feet: float
    vehicle_type_category: str
    vehicle_type_label: str
    vehicle_type_source: str
    drivers_estimated: bool
    notes: tuple[str, ...]


_DEFAULT_DOMAIN: RouteFleetDomain | None = None
_DEFAULT_PARAMETERS: CostParameters | None = None
_DEFAULT_WEEKDAY_RIDERSHIP: dict[str, float] | None = None
_DEFAULT_EQUITY_DATA: EquityDataBundle | None = None


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _download_to_path(url: str, out_path: Path, attempts: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mtfc-equity-loader/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = response.read()
            out_path.write_bytes(payload)
            if out_path.stat().st_size == 0:
                raise ValueError(f"Downloaded zero bytes from {url}")
            return
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            if attempt < attempts:
                sleep(float(attempt))
    assert last_error is not None
    raise last_error


def _resolve_equity_population_field(columns: Sequence[object]) -> str:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in _EQUITY_POPULATION_FIELDS:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "MTC EPC GeoJSON is missing a supported tract population field; "
        f"tried {_EQUITY_POPULATION_FIELDS}"
    )


def load_epc_tracts_geojson() -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory(prefix="equity-data-") as tempdir:
        geojson_path = Path(tempdir) / "mtc_epc.geojson"
        _download_to_path(MTC_EPC_GEOJSON_URL, geojson_path)
        try:
            gdf = gpd.read_file(geojson_path)
        except Exception:
            payload = json.loads(geojson_path.read_text(encoding="utf-8"))
            gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    _require_columns(gdf, {"geoid", "epc_2050", "epc_class", "geometry"}, "MTC EPC GeoJSON")
    population_field = _resolve_equity_population_field(gdf.columns)
    gdf["geoid"] = gdf["geoid"].astype(str)
    gdf["epc_2050"] = pd.to_numeric(gdf["epc_2050"], errors="coerce").fillna(0).astype(int)
    gdf["epc_class"] = gdf["epc_class"].fillna("").astype(str)
    gdf[population_field] = pd.to_numeric(gdf[population_field], errors="coerce")
    return gdf


def load_census_tract_geometries() -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory(prefix="equity-data-") as tempdir:
        zip_path = Path(tempdir) / "tl_2024_06_tract.zip"
        _download_to_path(CENSUS_TRACTS_ZIP_URL, zip_path)
        gdf = gpd.read_file(f"zip://{zip_path}")
    _require_columns(gdf, {"GEOID", "COUNTYFP", "geometry"}, "Census tract TIGER shapefile")
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    gdf["COUNTYFP"] = gdf["COUNTYFP"].astype(str)
    return gdf


def build_sf_equity_data_bundle(
    epc_tracts: gpd.GeoDataFrame | None = None,
    census_tracts: gpd.GeoDataFrame | None = None,
) -> EquityDataBundle:
    epc = load_epc_tracts_geojson() if epc_tracts is None else epc_tracts.copy()
    tracts = load_census_tract_geometries() if census_tracts is None else census_tracts.copy()

    population_field = _resolve_equity_population_field(epc.columns)
    sf_epc = epc.loc[epc["geoid"].astype(str).str.startswith(SF_TRACT_GEOID_PREFIX)].copy()
    if sf_epc.empty:
        raise ValueError("MTC EPC feed returned no San Francisco census tracts")

    sf_tracts = tracts.loc[tracts["COUNTYFP"].astype(str) == SF_COUNTY_FIPS, ["GEOID", "geometry"]].copy()
    if sf_tracts.empty:
        raise ValueError("Census TIGER feed returned no San Francisco tract geometries")
    if sf_tracts.crs != "EPSG:4326":
        sf_tracts = sf_tracts.to_crs("EPSG:4326")

    merged = sf_epc.merge(
        sf_tracts.rename(columns={"GEOID": "geoid"}),
        on="geoid",
        how="inner",
        suffixes=("_epc", ""),
    )
    if merged.empty:
        raise ValueError("No San Francisco EPC rows matched San Francisco tract geometries by GEOID")

    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=sf_tracts.crs)
    merged = merged.loc[:, ["geoid", "epc_2050", "epc_class", population_field, "geometry"]].copy()
    merged = merged.rename(columns={population_field: "tract_population"})
    merged["tract_population"] = pd.to_numeric(merged["tract_population"], errors="coerce")
    merged = merged.dropna(subset=["tract_population"]).sort_values(["geoid"]).reset_index(drop=True)
    if merged.empty:
        raise ValueError("No San Francisco EPC tracts retained a usable tract population field")

    notes = (
        f"Population weighting uses the live MTC EPC field '{population_field}'.",
        "San Francisco tract geometry comes from the California TIGER tract feed filtered to county FIPS 075.",
    )
    return EquityDataBundle(
        sf_epc_tracts=merged,
        population_field=population_field,
        notes=notes,
    )


def get_default_equity_data() -> EquityDataBundle:
    global _DEFAULT_EQUITY_DATA
    if _DEFAULT_EQUITY_DATA is None:
        _DEFAULT_EQUITY_DATA = build_sf_equity_data_bundle()
    return _DEFAULT_EQUITY_DATA


def _load_parameter_value(block_name: str, field_name: str, raw: object) -> ParameterValue:
    if not isinstance(raw, dict):
        raise ValueError(f"{block_name}.{field_name} must be an object with value and label")

    value = raw.get("value")
    label = raw.get("label")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{block_name}.{field_name}.value must be numeric")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{block_name}.{field_name}.label must be a non-empty string")

    return ParameterValue(value=float(value), label=label.strip())


def _require_mapping(raw: object, block_name: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{block_name} must be a JSON object")
    return raw


def load_parameters(data_path: str | pathlib.Path | None = None) -> CostParameters:
    if data_path is None:
        json_path = find_data_file("parameters.json")
    else:
        json_path = pathlib.Path(data_path)

    with json_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    root = _require_mapping(raw, "root")
    expected_blocks = {
        "reporting_constants",
        "operating_cost_parameters",
        "estimation_assumptions",
        "emissions_parameters",
        "objective_weights",
        "equity_parameters",
        "ridership_assumptions",
    }
    missing_blocks = expected_blocks.difference(root)
    if missing_blocks:
        raise ValueError(f"Missing required blocks in {json_path}: {sorted(missing_blocks)}")

    reporting_raw = _require_mapping(root["reporting_constants"], "reporting_constants")
    operating_raw = _require_mapping(root["operating_cost_parameters"], "operating_cost_parameters")
    estimation_raw = _require_mapping(root["estimation_assumptions"], "estimation_assumptions")
    emissions_raw = _require_mapping(root["emissions_parameters"], "emissions_parameters")
    objective_raw = _require_mapping(root["objective_weights"], "objective_weights")
    equity_raw = _require_mapping(root["equity_parameters"], "equity_parameters")
    ridership_raw = _require_mapping(root["ridership_assumptions"], "ridership_assumptions")

    reporting = ReportingConstants(
        annual_fare_revenue=_load_parameter_value(
            "reporting_constants", "annual_fare_revenue", reporting_raw.get("annual_fare_revenue")
        ),
        annual_advertising_revenue=_load_parameter_value(
            "reporting_constants",
            "annual_advertising_revenue",
            reporting_raw.get("annual_advertising_revenue"),
        ),
        annual_external_subsidies=_load_parameter_value(
            "reporting_constants",
            "annual_external_subsidies",
            reporting_raw.get("annual_external_subsidies"),
        ),
        annual_budget_ceiling=_load_parameter_value(
            "reporting_constants", "annual_budget_ceiling", reporting_raw.get("annual_budget_ceiling")
        ),
    )

    operating = OperatingCostParameters(
        labor_cost_per_vehicle_hour=_load_parameter_value(
            "operating_cost_parameters",
            "labor_cost_per_vehicle_hour",
            operating_raw.get("labor_cost_per_vehicle_hour"),
        ),
        maintenance_cost_per_vehicle_mile=_load_parameter_value(
            "operating_cost_parameters",
            "maintenance_cost_per_vehicle_mile",
            operating_raw.get("maintenance_cost_per_vehicle_mile"),
        ),
        energy_cost_per_vehicle_mile=_load_parameter_value(
            "operating_cost_parameters",
            "energy_cost_per_vehicle_mile",
            operating_raw.get("energy_cost_per_vehicle_mile"),
        ),
        annualized_capital_cost_per_vehicle=_load_parameter_value(
            "operating_cost_parameters",
            "annualized_capital_cost_per_vehicle",
            operating_raw.get("annualized_capital_cost_per_vehicle"),
        ),
    )

    assumptions = EstimationAssumptions(
        average_operating_speed_mph=_load_parameter_value(
            "estimation_assumptions",
            "average_operating_speed_mph",
            estimation_raw.get("average_operating_speed_mph"),
        ),
        deadhead_multiplier=_load_parameter_value(
            "estimation_assumptions", "deadhead_multiplier", estimation_raw.get("deadhead_multiplier")
        ),
        dwell_recovery_multiplier=_load_parameter_value(
            "estimation_assumptions",
            "dwell_recovery_multiplier",
            estimation_raw.get("dwell_recovery_multiplier"),
        ),
        weekday_service_days_per_year=_load_parameter_value(
            "estimation_assumptions",
            "weekday_service_days_per_year",
            estimation_raw.get("weekday_service_days_per_year"),
        ),
    )

    emissions = EmissionsParameters(
        car_emissions_grams_per_mile=_load_parameter_value(
            "emissions_parameters",
            "car_emissions_grams_per_mile",
            emissions_raw.get("car_emissions_grams_per_mile"),
        ),
        car_ownership_probability=_load_parameter_value(
            "emissions_parameters",
            "car_ownership_probability",
            emissions_raw.get("car_ownership_probability"),
        ),
        bus_base_emissions_grams_per_mile=_load_parameter_value(
            "emissions_parameters",
            "bus_base_emissions_grams_per_mile",
            emissions_raw.get("bus_base_emissions_grams_per_mile"),
        ),
        bus_climb_penalty_grams_per_mile=_load_parameter_value(
            "emissions_parameters",
            "bus_climb_penalty_grams_per_mile",
            emissions_raw.get("bus_climb_penalty_grams_per_mile"),
        ),
    )

    objective_weights = ObjectiveWeights(
        cost_percent_change_coefficient=_load_parameter_value(
            "objective_weights",
            "cost_percent_change_coefficient",
            objective_raw.get("cost_percent_change_coefficient"),
        ),
        emissions_percent_change_coefficient=_load_parameter_value(
            "objective_weights",
            "emissions_percent_change_coefficient",
            objective_raw.get("emissions_percent_change_coefficient"),
        ),
        equity_percent_change_coefficient=_load_parameter_value(
            "objective_weights",
            "equity_percent_change_coefficient",
            objective_raw.get("equity_percent_change_coefficient"),
        ),
    )

    equity_parameters = EquityParameters(
        service_intensity_coefficient=_load_parameter_value(
            "equity_parameters",
            "service_intensity_coefficient",
            equity_raw.get("service_intensity_coefficient"),
        ),
        waiting_time_coefficient=_load_parameter_value(
            "equity_parameters",
            "waiting_time_coefficient",
            equity_raw.get("waiting_time_coefficient"),
        ),
    )

    ridership_assumptions = RidershipAssumptions(
        route_average_trip_fraction=_load_parameter_value(
            "ridership_assumptions",
            "route_average_trip_fraction",
            ridership_raw.get("route_average_trip_fraction"),
        ),
    )

    return CostParameters(
        reporting_constants=reporting,
        operating_cost_parameters=operating,
        estimation_assumptions=assumptions,
        emissions_parameters=emissions,
        objective_weights=objective_weights,
        equity_parameters=equity_parameters,
        ridership_assumptions=ridership_assumptions,
    )


def load_cost_parameters(data_path: str | pathlib.Path | None = None) -> CostParameters:
    return load_parameters(data_path)


def load_model_parameters(data_path: str | pathlib.Path | None = None) -> CostParameters:
    return load_parameters(data_path)



def _validate_move(move: Move, n_routes: int) -> None:
    if move.move_type not in {"ADD", "DROP", "SWAP"}:
        raise ValueError(f"Unsupported move type: {move.move_type}")

    if move.move_type == "ADD":
        if move.add_index is None or move.drop_index is not None:
            raise ValueError("ADD move requires add_index and no drop_index")
    elif move.move_type == "DROP":
        if move.drop_index is None or move.add_index is not None:
            raise ValueError("DROP move requires drop_index and no add_index")
    else:
        if move.add_index is None or move.drop_index is None:
            raise ValueError("SWAP move requires add_index and drop_index")
        if move.add_index == move.drop_index:
            raise ValueError("SWAP move cannot use the same route for add/drop")

    for idx in (move.add_index, move.drop_index):
        if idx is not None and (idx < 0 or idx >= n_routes):
            raise IndexError(f"Route index out of range: {idx}")


def _coerce_solution_vector(y: Sequence[int], expected_len: int) -> tuple[int, ...]:
    if len(y) != expected_len:
        raise ValueError(f"Solution length {len(y)} does not match expected {expected_len}")

    out: list[int] = []
    for idx, value in enumerate(y):
        if isinstance(value, bool):
            raise ValueError(f"Route {idx} has boolean value {value}; expected integer fleet")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Route {idx} has non-numeric value {value!r}") from exc
        if not numeric.is_integer():
            raise ValueError(f"Route {idx} has non-integer fleet value {value!r}")
        out.append(int(numeric))

    return tuple(out)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.7613
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return float(2.0 * radius_miles * math.asin(math.sqrt(a)))


def _load_route_stops_table(route_stops_path: str | pathlib.Path | None = None) -> pd.DataFrame:
    if route_stops_path is None:
        csv_path = find_data_file("simplified_bus_route_stops.csv")
    else:
        csv_path = pathlib.Path(route_stops_path)

    route_stops = pd.read_csv(csv_path, dtype={"route_id": str})
    required = {"route_id", "direction_id", "route_stop_order", "stop_lat", "stop_lon"}
    missing = required.difference(route_stops.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    selected_columns = sorted(required)
    if "stop_id" in route_stops.columns:
        selected_columns.append("stop_id")
    if "elevation_m" in route_stops.columns:
        selected_columns.append("elevation_m")

    out = route_stops.loc[:, selected_columns].copy()
    out["direction_id"] = out["direction_id"].fillna("0").astype(str)
    out["route_stop_order"] = pd.to_numeric(out["route_stop_order"], errors="coerce")
    out["stop_lat"] = pd.to_numeric(out["stop_lat"], errors="coerce")
    out["stop_lon"] = pd.to_numeric(out["stop_lon"], errors="coerce")
    if "elevation_m" in out.columns:
        out["elevation_m"] = pd.to_numeric(out["elevation_m"], errors="coerce")
    else:
        out["elevation_m"] = float("nan")
    out = out.dropna(subset=["route_id", "route_stop_order", "stop_lat", "stop_lon"])
    out["route_id"] = out["route_id"].astype(str)
    if "stop_id" in out.columns:
        out["stop_id"] = out["stop_id"].astype(str)
    else:
        out["stop_id"] = out.index.astype(str)
    return out.sort_values(["route_id", "direction_id", "route_stop_order"]).reset_index(drop=True)


def assign_stops_to_sf_tracts(
    route_stops: pd.DataFrame,
    sf_tracts: gpd.GeoDataFrame,
) -> pd.DataFrame:
    _require_columns(route_stops, {"route_id", "stop_id", "stop_lat", "stop_lon"}, "route stops table")
    _require_columns(sf_tracts, {"geoid", "geometry"}, "San Francisco tract geometry")

    unique_stops = (
        route_stops.loc[:, ["stop_id", "stop_lat", "stop_lon"]]
        .drop_duplicates(subset=["stop_id"])
        .reset_index(drop=True)
    )
    stop_points = gpd.GeoDataFrame(
        unique_stops,
        geometry=gpd.points_from_xy(unique_stops["stop_lon"], unique_stops["stop_lat"]),
        crs="EPSG:4326",
    )
    working_tracts = sf_tracts.loc[:, ["geoid", "geometry"]].copy()
    if working_tracts.crs != "EPSG:4326":
        working_tracts = working_tracts.to_crs("EPSG:4326")

    stop_tract_join = gpd.sjoin(stop_points, working_tracts, how="left", predicate="within")
    assignments = route_stops.copy()
    assignments = assignments.merge(
        stop_tract_join.loc[:, ["stop_id", "geoid"]],
        on="stop_id",
        how="left",
    )
    return assignments


def _build_route_tract_coverage(
    route_stops_with_tracts: pd.DataFrame,
    route_ids: Sequence[str],
    sf_epc_tracts: gpd.GeoDataFrame,
) -> tuple[RouteTractCoverageSummary, ...]:
    epc_lookup = {
        str(row.geoid): int(row.epc_2050)
        for row in sf_epc_tracts.loc[:, ["geoid", "epc_2050"]].itertuples(index=False)
    }
    summaries: list[RouteTractCoverageSummary] = []
    for route_id in route_ids:
        route_rows = route_stops_with_tracts.loc[
            (route_stops_with_tracts["route_id"].astype(str) == str(route_id))
            & route_stops_with_tracts["geoid"].notna()
        ].copy()
        tract_counts = route_rows["geoid"].astype(str).value_counts().sort_index()
        tract_geoids = tuple(tract_counts.index.tolist())
        stop_counts_by_tract = tuple((str(geoid), int(count)) for geoid, count in tract_counts.items())
        summaries.append(
            RouteTractCoverageSummary(
                route_id=str(route_id),
                tract_geoids=tract_geoids,
                stop_counts_by_tract=stop_counts_by_tract,
                touches_epc_tract=any(epc_lookup.get(str(geoid), 0) == 1 for geoid in tract_geoids),
            )
        )
    return tuple(summaries)


def compute_tract_service_access_summaries(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
) -> tuple[TractServiceAccessSummary, ...]:
    active_domain = domain or get_default_domain()
    solution = _coerce_solution_vector(y, len(active_domain.route_ids))
    route_coverage_lookup = {summary.route_id: summary for summary in active_domain.route_tract_coverage}
    route_driver_lookup = {driver.route_id: driver for driver in active_domain.route_driver_estimates}

    tract_service: dict[str, dict[str, object]] = {}
    for route_id, candidate_fleet in zip(active_domain.route_ids, solution):
        coverage = route_coverage_lookup.get(route_id)
        if coverage is None or not coverage.stop_counts_by_tract:
            continue
        driver = route_driver_lookup[route_id]
        total_stops = sum(count for _, count in coverage.stop_counts_by_tract)
        if total_stops <= 0:
            continue
        service_scale, _scale_notes = _service_scale(candidate_fleet, driver.baseline_fleet)
        baseline_route_service = max(float(driver.weekday_planned_trips), 0.0)
        candidate_route_service = baseline_route_service * float(service_scale)

        for geoid, stop_count in coverage.stop_counts_by_tract:
            tract_stats = tract_service.setdefault(
                str(geoid),
                {
                    "routes": set(),
                    "stop_count": 0,
                    "baseline_service_intensity": 0.0,
                    "candidate_service_intensity": 0.0,
                },
            )
            share = float(stop_count) / float(total_stops)
            routes = tract_stats["routes"]
            assert isinstance(routes, set)
            routes.add(route_id)
            tract_stats["stop_count"] = int(tract_stats["stop_count"]) + int(stop_count)
            tract_stats["baseline_service_intensity"] = (
                float(tract_stats["baseline_service_intensity"]) + baseline_route_service * share
            )
            tract_stats["candidate_service_intensity"] = (
                float(tract_stats["candidate_service_intensity"]) + candidate_route_service * share
            )

    summaries: list[TractServiceAccessSummary] = []
    for geoid in sorted(tract_service):
        tract_stats = tract_service[geoid]
        routes = tract_stats["routes"]
        assert isinstance(routes, set)
        summaries.append(
            TractServiceAccessSummary(
                geoid=str(geoid),
                route_count=len(routes),
                stop_count=int(tract_stats["stop_count"]),
                baseline_service_intensity=float(tract_stats["baseline_service_intensity"]),
                candidate_service_intensity=float(tract_stats["candidate_service_intensity"]),
            )
        )
    return tuple(summaries)


def _waiting_time_proxy(service_intensity: float) -> float:
    return float(1.0 / (1.0 + max(service_intensity, 0.0)))


def _tract_utility(
    service_intensity: float,
    waiting_time_proxy: float,
    equity_parameters: EquityParameters,
) -> float:
    return float(
        equity_parameters.service_intensity_coefficient.value * service_intensity
        - equity_parameters.waiting_time_coefficient.value * waiting_time_proxy
    )


def _weighted_mean(pairs: Sequence[tuple[float, float]]) -> float:
    total_weight = sum(max(weight, 0.0) for weight, _value in pairs)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(max(weight, 0.0) * value for weight, value in pairs) / total_weight)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _feet_from_miles(distance_miles: float) -> float:
    return float(distance_miles * 5280.0)


def _miles_from_feet(distance_feet: float) -> float:
    return float(distance_feet / 5280.0)


def _normalize_route_key(value: object) -> str:
    text = str(value).strip().upper()
    compact = "".join(ch for ch in text if ch.isalnum())
    if compact.startswith("R") and len(compact) > 1 and compact[1:].isdigit():
        return compact[1:]
    return compact


def _normalize_vehicle_type_label(raw: object) -> tuple[str, str]:
    text = str(raw).strip()
    normalized = text.upper().replace("-", " ")
    normalized = " ".join(normalized.split())
    label_map = {
        "M STD": "motor_standard",
        "M ARTIC": "motor_articulated",
        "T STD": "trolley_standard",
        "T ARTIC": "trolley_articulated",
    }
    category = label_map.get(normalized, "unknown")
    return category, text


def _select_vehicle_type_label(raw: object) -> tuple[str, str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "unknown", "Unknown"

    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        return "unknown", "Unknown"

    ranked_parts: list[tuple[int, str, str]] = []
    rank_map = {
        "motor_articulated": 4,
        "motor_standard": 3,
        "trolley_articulated": 2,
        "trolley_standard": 1,
        "unknown": 0,
    }
    for part in parts:
        category, label = _normalize_vehicle_type_label(part)
        ranked_parts.append((rank_map.get(category, 0), category, label))

    ranked_parts.sort(key=lambda item: (-item[0], item[2]))
    _, category, label = ranked_parts[0]
    return category, label


def _load_peak_vehicle_assignments(
    data_path: str | pathlib.Path | None = None,
) -> dict[str, tuple[str, str, str]]:
    if data_path is None:
        csv_path = find_data_file("peak_vehicles_by_route.csv")
    else:
        csv_path = pathlib.Path(data_path)

    try:
        peak = pd.read_csv(csv_path, dtype=str)
    except FileNotFoundError:
        return {}

    required = {"Route"}
    if not required.issubset(peak.columns):
        return {}

    vehicle_columns = [
        "VehicleType_2025",
        "VehicleType_2020",
        "VehicleType_Existing_2017",
        "VehicleType_2030",
    ]

    assignments: dict[str, tuple[str, str, str]] = {}
    for row in peak.itertuples(index=False):
        route_label = getattr(row, "Route", None)
        if route_label is None:
            continue

        selected_value = None
        selected_source = "peak_vehicles_by_route.csv"
        for col in vehicle_columns:
            if hasattr(row, col):
                value = getattr(row, col)
                if isinstance(value, str) and value.strip():
                    selected_value = value.strip()
                    selected_source = f"peak_vehicles_by_route.csv:{col}"
                    break

        category, label = _select_vehicle_type_label(selected_value)
        assignments[_normalize_route_key(route_label)] = (category, label, selected_source)

    return assignments


def _resolve_route_vehicle_type(
    route_id: str,
    route_short_name: str,
    peak_assignments: dict[str, tuple[str, str, str]],
) -> tuple[str, str, str, tuple[str, ...]]:
    keys_to_try = [_normalize_route_key(route_id), _normalize_route_key(route_short_name)]
    for key in keys_to_try:
        if key and key in peak_assignments:
            category, label, source = peak_assignments[key]
            return category, label, source, ()

    fallback_notes = ("No peak-file vehicle match was found; using systemwide unknown-bus fallback metadata.",)
    return "unknown", "Unknown bus", "system_fallback", fallback_notes


def _compute_route_direction_metrics(route_stops: pd.DataFrame) -> dict[str, dict[str, dict[str, float | bool]]]:
    metrics_by_route: dict[str, dict[str, dict[str, float | bool]]] = {}
    for (route_id, direction_id), group in route_stops.groupby(["route_id", "direction_id"], sort=True):
        ordered = group.sort_values(["route_stop_order"]).reset_index(drop=True)
        if len(ordered) < 2:
            metrics_by_route.setdefault(route_id, {})[direction_id] = {
                "horizontal_distance_miles": 0.0,
                "three_d_distance_miles": 0.0,
                "uphill_gain_miles": 0.0,
                "uphill_gain_feet": 0.0,
                "missing_elevation_segments": 0.0,
                "segments": 0.0,
            }
            continue

        horizontal_distance_miles = 0.0
        three_d_distance_miles = 0.0
        uphill_gain_feet = 0.0
        missing_elevation_segments = 0
        for left, right in zip(ordered.itertuples(index=False), ordered.iloc[1:].itertuples(index=False)):
            segment_horizontal_miles = _haversine_miles(
                float(left.stop_lat),
                float(left.stop_lon),
                float(right.stop_lat),
                float(right.stop_lon),
            )
            horizontal_distance_miles += segment_horizontal_miles

            if pd.isna(left.elevation_m) or pd.isna(right.elevation_m):
                vertical_miles = 0.0
                segment_uphill_feet = 0.0
                missing_elevation_segments += 1
            else:
                delta_meters = float(right.elevation_m) - float(left.elevation_m)
                delta_feet = delta_meters * 3.280839895
                vertical_miles = _miles_from_feet(abs(delta_feet))
                segment_uphill_feet = max(delta_feet, 0.0)
                uphill_gain_feet += segment_uphill_feet

            three_d_distance_miles += math.sqrt(segment_horizontal_miles**2 + vertical_miles**2)

        metrics_by_route.setdefault(route_id, {})[direction_id] = {
            "horizontal_distance_miles": float(horizontal_distance_miles),
            "three_d_distance_miles": float(three_d_distance_miles),
            "uphill_gain_miles": _miles_from_feet(uphill_gain_feet),
            "uphill_gain_feet": float(uphill_gain_feet),
            "missing_elevation_segments": float(missing_elevation_segments),
            "segments": float(max(len(ordered) - 1, 0)),
        }
    return metrics_by_route


def _build_route_driver_estimates(
    table: pd.DataFrame,
    route_stops_path: str | pathlib.Path | None = None,
) -> tuple[RouteCostDriverEstimate, ...]:
    direction_metrics: dict[str, dict[str, dict[str, float | bool]]] = {}
    try:
        if route_stops_path is not None:
            route_stops = _load_route_stops_table(route_stops_path)
            direction_metrics = _compute_route_direction_metrics(route_stops)
        elif table.attrs.get("source_is_default", False):
            route_stops = _load_route_stops_table()
            direction_metrics = _compute_route_direction_metrics(route_stops)
    except FileNotFoundError:
        direction_metrics = {}

    peak_assignments = _load_peak_vehicle_assignments()

    estimates: list[RouteCostDriverEstimate] = []
    for row in table.itertuples(index=False):
        per_direction = direction_metrics.get(str(row.route_id), {})
        directional_metrics = [per_direction[key] for key in sorted(per_direction)]
        direction_count = len(directional_metrics)
        vehicle_type_category, vehicle_type_label, vehicle_type_source, vehicle_notes = _resolve_route_vehicle_type(
            route_id=str(row.route_id),
            route_short_name=str(row.route_short_name),
            peak_assignments=peak_assignments,
        )

        notes: list[str] = [
            "Stop-to-stop haversine path length is an approximation, not GTFS shape ground truth.",
            "Weekday planned trips are treated as one-way revenue trips.",
        ]
        if direction_count == 0:
            one_way_distance_miles = 0.0
            round_trip_distance_miles = 0.0
            horizontal_one_way_distance_miles = 0.0
            horizontal_round_trip_distance_miles = 0.0
            three_d_one_way_distance_miles = 0.0
            three_d_round_trip_distance_miles = 0.0
            uphill_gain_one_way_miles = 0.0
            uphill_gain_round_trip_miles = 0.0
            uphill_gain_one_way_feet = 0.0
            uphill_gain_round_trip_feet = 0.0
            drivers_estimated = False
            notes.append("No route-stop geometry was available for this route.")
        elif direction_count == 1:
            horizontal_one_way_distance_miles = float(directional_metrics[0]["horizontal_distance_miles"])
            horizontal_round_trip_distance_miles = horizontal_one_way_distance_miles * 2.0
            three_d_one_way_distance_miles = float(directional_metrics[0]["three_d_distance_miles"])
            three_d_round_trip_distance_miles = three_d_one_way_distance_miles * 2.0
            uphill_gain_one_way_miles = float(directional_metrics[0]["uphill_gain_miles"])
            uphill_gain_round_trip_miles = uphill_gain_one_way_miles * 2.0
            uphill_gain_one_way_feet = float(directional_metrics[0]["uphill_gain_feet"])
            uphill_gain_round_trip_feet = uphill_gain_one_way_feet * 2.0
            one_way_distance_miles = horizontal_one_way_distance_miles
            round_trip_distance_miles = horizontal_round_trip_distance_miles
            drivers_estimated = True
            notes.append("Only one direction was available; round-trip distance doubles the observed direction.")
        else:
            horizontal_one_way_distance_miles = sum(
                float(metric["horizontal_distance_miles"]) for metric in directional_metrics
            ) / float(direction_count)
            horizontal_round_trip_distance_miles = sum(
                float(metric["horizontal_distance_miles"]) for metric in directional_metrics
            )
            three_d_one_way_distance_miles = sum(
                float(metric["three_d_distance_miles"]) for metric in directional_metrics
            ) / float(direction_count)
            three_d_round_trip_distance_miles = sum(
                float(metric["three_d_distance_miles"]) for metric in directional_metrics
            )
            uphill_gain_one_way_miles = sum(
                float(metric["uphill_gain_miles"]) for metric in directional_metrics
            ) / float(direction_count)
            uphill_gain_round_trip_miles = sum(float(metric["uphill_gain_miles"]) for metric in directional_metrics)
            uphill_gain_one_way_feet = sum(
                float(metric["uphill_gain_feet"]) for metric in directional_metrics
            ) / float(direction_count)
            uphill_gain_round_trip_feet = sum(float(metric["uphill_gain_feet"]) for metric in directional_metrics)
            one_way_distance_miles = horizontal_one_way_distance_miles
            round_trip_distance_miles = horizontal_round_trip_distance_miles
            drivers_estimated = True
            notes.append("Round-trip distance sums observed directional paths; one-way distance is their mean.")
        notes.extend(vehicle_notes)

        if direction_count > 0:
            missing_elevation_segments = int(
                sum(float(metric["missing_elevation_segments"]) for metric in directional_metrics)
            )
            total_segments = int(sum(float(metric["segments"]) for metric in directional_metrics))
            if missing_elevation_segments > 0:
                notes.append(
                    "Missing stop elevations were treated as zero elevation change for "
                    f"{missing_elevation_segments} of {total_segments} route segments."
                )
            notes.append(
                "Terrain metrics preserve horizontal distance, 3D distance, and uphill-only gain per direction."
            )

        estimates.append(
            RouteCostDriverEstimate(
                route_id=str(row.route_id),
                baseline_fleet=int(row.weekday_typical_buses),
                weekday_planned_trips=float(row.weekday_planned_trips),
                direction_count=direction_count,
                one_way_distance_miles=float(one_way_distance_miles),
                round_trip_distance_miles=float(round_trip_distance_miles),
                horizontal_one_way_distance_miles=float(horizontal_one_way_distance_miles),
                horizontal_round_trip_distance_miles=float(horizontal_round_trip_distance_miles),
                three_d_one_way_distance_miles=float(three_d_one_way_distance_miles),
                three_d_round_trip_distance_miles=float(three_d_round_trip_distance_miles),
                uphill_gain_one_way_miles=float(uphill_gain_one_way_miles),
                uphill_gain_round_trip_miles=float(uphill_gain_round_trip_miles),
                uphill_gain_one_way_feet=float(uphill_gain_one_way_feet),
                uphill_gain_round_trip_feet=float(uphill_gain_round_trip_feet),
                vehicle_type_category=vehicle_type_category,
                vehicle_type_label=vehicle_type_label,
                vehicle_type_source=vehicle_type_source,
                drivers_estimated=drivers_estimated,
                notes=tuple(notes),
            )
        )

    return tuple(estimates)


def load_route_fleet_domain(
    data_path: str | pathlib.Path | None = None,
    route_stops_path: str | pathlib.Path | None = None,
    equity_data: EquityDataBundle | None = None,
) -> RouteFleetDomain:
    use_default_route_stops = data_path is None and route_stops_path is None
    if data_path is None:
        csv_path = find_data_file("simplified_bus_routes.csv")
    else:
        csv_path = pathlib.Path(data_path)

    routes = pd.read_csv(csv_path, dtype={"route_id": str})
    required = {
        "route_id",
        "route_short_name",
        "route_long_name",
        "weekday_typical_buses",
        "weekday_planned_trips",
        "stop_count",
    }
    missing = required.difference(routes.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    table = routes.loc[:, sorted(required)].copy()
    table["weekday_typical_buses"] = pd.to_numeric(table["weekday_typical_buses"], errors="coerce")
    table["weekday_planned_trips"] = pd.to_numeric(table["weekday_planned_trips"], errors="coerce")
    table["stop_count"] = pd.to_numeric(table["stop_count"], errors="coerce")
    table = table.dropna(subset=["route_id", "weekday_typical_buses", "weekday_planned_trips", "stop_count"])
    table["route_id"] = table["route_id"].astype(str)

    table["weekday_typical_buses"] = table["weekday_typical_buses"].round().astype(int)
    table["weekday_planned_trips"] = table["weekday_planned_trips"].astype(float)
    table["stop_count"] = table["stop_count"].astype(float)

    table = table.sort_values(["route_id"]).drop_duplicates(subset=["route_id"], keep="first").reset_index(drop=True)
    table["route_index"] = table.index
    table.attrs["source_is_default"] = use_default_route_stops

    baseline = table["weekday_typical_buses"].astype(int)
    lower = (baseline - 10).clip(lower=0)
    upper = baseline + 10

    service_raw = table["weekday_planned_trips"].clip(lower=0) + table["stop_count"].clip(lower=0)
    denom = float(service_raw.max()) if len(service_raw) > 0 else 0.0
    if denom <= 0.0:
        service = pd.Series([1.0] * len(table), index=table.index, dtype=float)
    else:
        service = 1.0 + (service_raw / denom)

    stop_tract_assignments = pd.DataFrame(
        columns=["route_id", "direction_id", "route_stop_order", "stop_lat", "stop_lon", "stop_id", "geoid"]
    )
    route_tract_coverage: tuple[RouteTractCoverageSummary, ...] = tuple(
        RouteTractCoverageSummary(
            route_id=str(route_id),
            tract_geoids=(),
            stop_counts_by_tract=(),
            touches_epc_tract=False,
        )
        for route_id in table["route_id"].tolist()
    )
    equity_tracts: tuple[EquityTractMetadata, ...] = ()

    metadata = table.loc[
        :,
        [
            "route_index",
            "route_id",
            "route_short_name",
            "route_long_name",
            "weekday_typical_buses",
            "weekday_planned_trips",
            "stop_count",
        ],
    ].copy()
    metadata["lower_bound"] = lower.values
    metadata["upper_bound"] = upper.values
    route_driver_estimates = _build_route_driver_estimates(table, route_stops_path=route_stops_path)

    driver_lookup = {driver.route_id: driver for driver in route_driver_estimates}
    metadata["one_way_distance_miles"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].one_way_distance_miles
    )
    metadata["round_trip_distance_miles"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].round_trip_distance_miles
    )
    metadata["horizontal_one_way_distance_miles"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].horizontal_one_way_distance_miles
    )
    metadata["horizontal_round_trip_distance_miles"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].horizontal_round_trip_distance_miles
    )
    metadata["three_d_one_way_distance_miles"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].three_d_one_way_distance_miles
    )
    metadata["three_d_round_trip_distance_miles"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].three_d_round_trip_distance_miles
    )
    metadata["uphill_gain_one_way_miles"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].uphill_gain_one_way_miles
    )
    metadata["uphill_gain_round_trip_miles"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].uphill_gain_round_trip_miles
    )
    metadata["uphill_gain_one_way_feet"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].uphill_gain_one_way_feet
    )
    metadata["uphill_gain_round_trip_feet"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].uphill_gain_round_trip_feet
    )
    metadata["direction_count"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].direction_count
    )
    metadata["vehicle_type_category"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].vehicle_type_category
    )
    metadata["vehicle_type_label"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].vehicle_type_label
    )
    metadata["vehicle_type_source"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].vehicle_type_source
    )

    should_build_equity_cache = route_stops_path is not None or use_default_route_stops
    active_equity_data: EquityDataBundle | None = equity_data
    if active_equity_data is None and use_default_route_stops:
        active_equity_data = get_default_equity_data()
    if should_build_equity_cache and active_equity_data is not None:
        route_stops = _load_route_stops_table(route_stops_path)
        stop_tract_assignments = assign_stops_to_sf_tracts(route_stops, active_equity_data.sf_epc_tracts)
        route_tract_coverage = _build_route_tract_coverage(
            stop_tract_assignments,
            route_ids=table["route_id"].tolist(),
            sf_epc_tracts=active_equity_data.sf_epc_tracts,
        )
        coverage_lookup = {summary.route_id: summary for summary in route_tract_coverage}
        metadata["touches_epc_tract"] = metadata["route_id"].map(
            lambda route_id: coverage_lookup[str(route_id)].touches_epc_tract
        )
        metadata["served_tract_count"] = metadata["route_id"].map(
            lambda route_id: len(coverage_lookup[str(route_id)].tract_geoids)
        )
        equity_tracts = tuple(
            EquityTractMetadata(
                geoid=str(row.geoid),
                epc_2050=int(row.epc_2050),
                epc_class=str(row.epc_class),
                population=float(row.tract_population),
            )
            for row in active_equity_data.sf_epc_tracts.loc[
                :, ["geoid", "epc_2050", "epc_class", "tract_population"]
            ].sort_values(["geoid"]).itertuples(index=False)
        )
    else:
        metadata["touches_epc_tract"] = False
        metadata["served_tract_count"] = 0

    return RouteFleetDomain(
        route_ids=tuple(table["route_id"].tolist()),
        baseline=tuple(int(v) for v in baseline.tolist()),
        lower_bounds=tuple(int(v) for v in lower.tolist()),
        upper_bounds=tuple(int(v) for v in upper.tolist()),
        service_weights=tuple(float(v) for v in service.tolist()),
        route_driver_estimates=route_driver_estimates,
        route_metadata=metadata,
        stop_tract_assignments=stop_tract_assignments,
        route_tract_coverage=route_tract_coverage,
        equity_tracts=equity_tracts,
    )


def get_default_domain() -> RouteFleetDomain:
    global _DEFAULT_DOMAIN
    if _DEFAULT_DOMAIN is None:
        _DEFAULT_DOMAIN = load_route_fleet_domain()
    return _DEFAULT_DOMAIN


def get_default_parameters() -> CostParameters:
    global _DEFAULT_PARAMETERS
    if _DEFAULT_PARAMETERS is None:
        _DEFAULT_PARAMETERS = load_parameters()
    return _DEFAULT_PARAMETERS


def get_default_cost_parameters() -> CostParameters:
    return get_default_parameters()


def _parse_month_label_to_ordinal(value: object) -> int:
    month_text = str(value).strip()
    try:
        ts = pd.to_datetime(month_text, format="%B %Y")
    except (TypeError, ValueError):
        return -1
    return int(ts.year * 12 + ts.month)


def _normalize_ridership_route_label(value: object) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    first_token = text.split(" ", 1)[0]
    return "".join(ch for ch in first_token if ch.isalnum())


def _load_weekday_ridership(
    ridership_path: str | pathlib.Path | None = None,
) -> dict[str, float]:
    if ridership_path is None:
        csv_path = pathlib.Path("/Users/hq/Downloads/RidershipbyRouteTableDownload.csv")
    else:
        csv_path = pathlib.Path(ridership_path)

    ridership = pd.read_csv(csv_path, dtype=str)
    required = {"Month", "Route", "Service Day of the Week", "Average Daily Boardings"}
    missing = required.difference(ridership.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    weekday = ridership.loc[ridership["Service Day of the Week"].astype(str).str.strip() == "Weekday"].copy()
    weekday["route_key"] = weekday["Route"].map(_normalize_ridership_route_label)
    weekday["month_ordinal"] = weekday["Month"].map(_parse_month_label_to_ordinal)
    weekday["average_daily_boardings"] = (
        weekday["Average Daily Boardings"].astype(str).str.replace(",", "", regex=False)
    )
    weekday["average_daily_boardings"] = pd.to_numeric(weekday["average_daily_boardings"], errors="coerce")
    weekday = weekday.dropna(subset=["route_key", "month_ordinal", "average_daily_boardings"])
    weekday = weekday.loc[weekday["route_key"] != ""].copy()
    weekday = weekday.sort_values(["route_key", "month_ordinal"]).drop_duplicates(
        subset=["route_key"],
        keep="last",
    )
    return {
        str(row.route_key): float(row.average_daily_boardings)
        for row in weekday.itertuples(index=False)
    }


def get_default_weekday_ridership() -> dict[str, float]:
    global _DEFAULT_WEEKDAY_RIDERSHIP
    if _DEFAULT_WEEKDAY_RIDERSHIP is None:
        _DEFAULT_WEEKDAY_RIDERSHIP = _load_weekday_ridership()
    return _DEFAULT_WEEKDAY_RIDERSHIP


def _service_scale(candidate_fleet: int, baseline_fleet: int) -> tuple[float, tuple[str, ...]]:
    if baseline_fleet > 0:
        return float(candidate_fleet) / float(baseline_fleet), ()
    if candidate_fleet <= 0:
        return 0.0, ("Baseline fleet is zero; candidate service scale is clamped to zero.",)
    return float(candidate_fleet), (
        "Baseline fleet is zero; candidate fleet count is used as the service scale proxy.",
    )


def _annual_service_from_driver(
    driver: RouteCostDriverEstimate,
    params: CostParameters,
    service_scale: float,
) -> tuple[float, float, float, float]:
    deadhead_multiplier = params.estimation_assumptions.deadhead_multiplier.value
    service_days = params.estimation_assumptions.weekday_service_days_per_year.value
    operating_speed = params.estimation_assumptions.average_operating_speed_mph.value
    dwell_multiplier = params.estimation_assumptions.dwell_recovery_multiplier.value

    baseline_annual_miles = (
        driver.one_way_distance_miles
        * driver.weekday_planned_trips
        * service_days
        * deadhead_multiplier
    )
    candidate_annual_miles = baseline_annual_miles * service_scale

    if operating_speed <= 0:
        raise ValueError("average_operating_speed_mph must be greater than zero")

    baseline_annual_hours = (baseline_annual_miles / operating_speed) * dwell_multiplier
    candidate_annual_hours = (candidate_annual_miles / operating_speed) * dwell_multiplier
    return (
        float(baseline_annual_miles),
        float(baseline_annual_hours),
        float(candidate_annual_miles),
        float(candidate_annual_hours),
    )


def compute_cost_breakdown(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
    parameters: CostParameters | None = None,
    cost_parameters: CostParameters | None = None,
) -> SystemCostBreakdown:
    active_domain = domain or get_default_domain()
    params = parameters or cost_parameters or get_default_parameters()
    solution = _coerce_solution_vector(y, len(active_domain.route_ids))

    if not is_within_route_bounds(solution, active_domain):
        raise ValueError("Solution violates per-route bounds")

    baseline_total_fleet = sum(active_domain.baseline)
    candidate_total_fleet = sum(solution)
    net_new_fleet = max(candidate_total_fleet - baseline_total_fleet, 0)
    capital_unit_cost = params.operating_cost_parameters.annualized_capital_cost_per_vehicle.value
    annual_capital_cost = float(net_new_fleet * capital_unit_cost)

    labor_total = 0.0
    maintenance_total = 0.0
    energy_total = 0.0
    route_breakdowns: list[RouteCostBreakdown] = []
    for driver, candidate_fleet in zip(active_domain.route_driver_estimates, solution):
        service_scale, scale_notes = _service_scale(candidate_fleet, driver.baseline_fleet)
        (
            baseline_annual_vehicle_miles,
            baseline_annual_vehicle_hours,
            candidate_annual_vehicle_miles,
            candidate_annual_vehicle_hours,
        ) = _annual_service_from_driver(driver, params, service_scale)

        annual_labor_cost = (
            candidate_annual_vehicle_hours
            * params.operating_cost_parameters.labor_cost_per_vehicle_hour.value
        )
        annual_maintenance_cost = (
            candidate_annual_vehicle_miles
            * params.operating_cost_parameters.maintenance_cost_per_vehicle_mile.value
        )
        annual_energy_cost = (
            candidate_annual_vehicle_miles
            * params.operating_cost_parameters.energy_cost_per_vehicle_mile.value
        )
        annual_total_cost = annual_labor_cost + annual_maintenance_cost + annual_energy_cost

        labor_total += annual_labor_cost
        maintenance_total += annual_maintenance_cost
        energy_total += annual_energy_cost

        route_breakdowns.append(
            RouteCostBreakdown(
                route_id=driver.route_id,
                baseline_fleet=int(driver.baseline_fleet),
                candidate_fleet=int(candidate_fleet),
                fleet_delta=int(candidate_fleet - driver.baseline_fleet),
                service_scale=float(service_scale),
                one_way_distance_miles=float(driver.one_way_distance_miles),
                round_trip_distance_miles=float(driver.round_trip_distance_miles),
                baseline_annual_vehicle_miles=float(baseline_annual_vehicle_miles),
                candidate_annual_vehicle_miles=float(candidate_annual_vehicle_miles),
                baseline_annual_vehicle_hours=float(baseline_annual_vehicle_hours),
                candidate_annual_vehicle_hours=float(candidate_annual_vehicle_hours),
                annual_labor_cost=float(annual_labor_cost),
                annual_maintenance_cost=float(annual_maintenance_cost),
                annual_energy_cost=float(annual_energy_cost),
                annual_capital_cost=0.0,
                annual_total_cost=float(annual_total_cost),
                drivers_estimated=driver.drivers_estimated,
                notes=driver.notes + scale_notes,
            )
        )

    notes = (
        "Route mileage uses stop-to-stop haversine geometry derived from local route-stop sequences.",
        "Annual service uses weekday planned trips only and scales with candidate fleet relative to baseline.",
        "Reporting constants are loaded for reporting only and do not affect candidate ranking.",
    )

    annual_total_cost = labor_total + maintenance_total + energy_total + annual_capital_cost
    annual_total_revenue = (
        params.reporting_constants.annual_fare_revenue.value
        + params.reporting_constants.annual_advertising_revenue.value
        + params.reporting_constants.annual_external_subsidies.value
    )
    annual_budget_slack = params.reporting_constants.annual_budget_ceiling.value - annual_total_cost

    return SystemCostBreakdown(
        objective_cost=float(annual_total_cost),
        annual_labor_cost=float(labor_total),
        annual_maintenance_cost=float(maintenance_total),
        annual_energy_cost=float(energy_total),
        annual_capital_cost=annual_capital_cost,
        annual_total_cost=float(annual_total_cost),
        annual_total_revenue=float(annual_total_revenue),
        annual_budget_slack=float(annual_budget_slack),
        baseline_total_fleet=baseline_total_fleet,
        candidate_total_fleet=candidate_total_fleet,
        net_new_fleet=net_new_fleet,
        route_breakdowns=tuple(route_breakdowns),
        reporting_constants=params.reporting_constants,
        estimation_assumptions=params.estimation_assumptions,
        notes=notes,
    )


def cost_objective(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
    parameters: CostParameters | None = None,
    cost_parameters: CostParameters | None = None,
) -> float:
    return float(
        compute_cost_breakdown(y, domain=domain, parameters=parameters, cost_parameters=cost_parameters).objective_cost
    )


def compute_emissions_breakdown(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
    parameters: CostParameters | None = None,
    cost_parameters: CostParameters | None = None,
    weekday_ridership: dict[str, float] | None = None,
) -> SystemEmissionsBreakdown:
    active_domain = domain or get_default_domain()
    params = parameters or cost_parameters or get_default_parameters()
    solution = _coerce_solution_vector(y, len(active_domain.route_ids))
    ridership_lookup = weekday_ridership or get_default_weekday_ridership()

    if not is_within_route_bounds(solution, active_domain):
        raise ValueError("Solution violates per-route bounds")

    route_breakdowns: list[RouteEmissionsBreakdown] = []
    baseline_total = 0.0
    candidate_total = 0.0
    for driver, candidate_fleet in zip(active_domain.route_driver_estimates, solution):
        service_scale, scale_notes = _service_scale(candidate_fleet, driver.baseline_fleet)
        ridership_key = _normalize_route_key(driver.route_id)
        baseline_riders = float(ridership_lookup.get(ridership_key, 0.0))
        candidate_riders = baseline_riders * service_scale
        average_rider_trip_distance_miles = (
            driver.one_way_distance_miles * params.ridership_assumptions.route_average_trip_fraction.value
        )
        (
            baseline_annual_vehicle_miles,
            _baseline_annual_vehicle_hours,
            candidate_annual_vehicle_miles,
            _candidate_annual_vehicle_hours,
        ) = _annual_service_from_driver(driver, params, service_scale)

        uphill_ratio = (
            driver.uphill_gain_round_trip_miles / driver.three_d_round_trip_distance_miles
            if driver.three_d_round_trip_distance_miles > 0.0
            else 0.0
        )
        bus_emissions_factor = (
            params.emissions_parameters.bus_base_emissions_grams_per_mile.value
            + params.emissions_parameters.bus_climb_penalty_grams_per_mile.value * uphill_ratio
        )
        baseline_bus_emissions_grams = baseline_annual_vehicle_miles * bus_emissions_factor
        candidate_bus_emissions_grams = candidate_annual_vehicle_miles * bus_emissions_factor

        annualization_factor = params.estimation_assumptions.weekday_service_days_per_year.value
        rider_emissions_factor = (
            average_rider_trip_distance_miles
            * params.emissions_parameters.car_ownership_probability.value
            * params.emissions_parameters.car_emissions_grams_per_mile.value
            * annualization_factor
        )
        baseline_rider_emissions_avoided_grams = baseline_riders * rider_emissions_factor
        candidate_rider_emissions_avoided_grams = candidate_riders * rider_emissions_factor
        baseline_net_emissions_grams = baseline_bus_emissions_grams - baseline_rider_emissions_avoided_grams
        candidate_net_emissions_grams = candidate_bus_emissions_grams - candidate_rider_emissions_avoided_grams

        notes = driver.notes + scale_notes
        if baseline_riders <= 0.0:
            notes = notes + (
                "No weekday ridership match was found for this route; baseline rider demand defaults to zero.",
            )
        notes = notes + (
            "Ridership uses the latest available weekday average daily boardings from the local ridership file.",
            "Average rider trip distance is proxied as one-way route distance multiplied by route_average_trip_fraction.",
            "Bus emissions are annualized weekday service mileage times a base factor plus uphill-distance penalty.",
            "Rider emissions avoided are annualized weekday riders times average trip distance and car-mode assumptions.",
        )

        baseline_total += baseline_net_emissions_grams
        candidate_total += candidate_net_emissions_grams
        route_breakdowns.append(
            RouteEmissionsBreakdown(
                route_id=driver.route_id,
                baseline_fleet=int(driver.baseline_fleet),
                candidate_fleet=int(candidate_fleet),
                fleet_delta=int(candidate_fleet - driver.baseline_fleet),
                service_scale=float(service_scale),
                baseline_riders=float(baseline_riders),
                candidate_riders=float(candidate_riders),
                average_rider_trip_distance_miles=float(average_rider_trip_distance_miles),
                baseline_bus_emissions_grams=float(baseline_bus_emissions_grams),
                candidate_bus_emissions_grams=float(candidate_bus_emissions_grams),
                baseline_rider_emissions_avoided_grams=float(baseline_rider_emissions_avoided_grams),
                candidate_rider_emissions_avoided_grams=float(candidate_rider_emissions_avoided_grams),
                baseline_net_emissions_grams=float(baseline_net_emissions_grams),
                candidate_net_emissions_grams=float(candidate_net_emissions_grams),
                notes=notes,
            )
        )

    absolute_delta = candidate_total - baseline_total
    percent_delta, percent_notes = _percent_change_vs_baseline(candidate_total, baseline_total)
    notes = (
        "Net emissions are defined as annual bus operating emissions minus annual rider car emissions avoided.",
        "Weekday ridership is annualized with the same weekday service-day assumption used by the cost model.",
    ) + percent_notes
    return SystemEmissionsBreakdown(
        baseline_total_emissions_grams=float(baseline_total),
        candidate_total_emissions_grams=float(candidate_total),
        absolute_delta_emissions_grams=float(absolute_delta),
        percent_delta_emissions=float(percent_delta),
        route_breakdowns=tuple(route_breakdowns),
        emissions_parameters=params.emissions_parameters,
        ridership_assumptions=params.ridership_assumptions,
        notes=notes,
    )


def compute_equity_breakdown(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
    parameters: CostParameters | None = None,
    cost_parameters: CostParameters | None = None,
) -> SystemEquityBreakdown:
    active_domain = domain or get_default_domain()
    params = parameters or cost_parameters or get_default_parameters()
    solution = _coerce_solution_vector(y, len(active_domain.route_ids))

    if not is_within_route_bounds(solution, active_domain):
        raise ValueError("Solution violates per-route bounds")

    baseline_service_lookup = {
        summary.geoid: summary
        for summary in compute_tract_service_access_summaries(active_domain.baseline, domain=active_domain)
    }
    candidate_service_lookup = {
        summary.geoid: summary
        for summary in compute_tract_service_access_summaries(solution, domain=active_domain)
    }

    tract_breakdowns: list[EquityTractBreakdown] = []
    for tract in active_domain.equity_tracts:
        baseline_service = baseline_service_lookup.get(tract.geoid)
        candidate_service = candidate_service_lookup.get(tract.geoid)
        baseline_service_intensity = (
            baseline_service.baseline_service_intensity if baseline_service is not None else 0.0
        )
        candidate_service_intensity = (
            candidate_service.candidate_service_intensity if candidate_service is not None else 0.0
        )
        route_count = max(
            baseline_service.route_count if baseline_service is not None else 0,
            candidate_service.route_count if candidate_service is not None else 0,
        )
        baseline_waiting = _waiting_time_proxy(baseline_service_intensity)
        candidate_waiting = _waiting_time_proxy(candidate_service_intensity)
        tract_breakdowns.append(
            EquityTractBreakdown(
                geoid=tract.geoid,
                epc_2050=int(tract.epc_2050),
                epc_class=tract.epc_class,
                tract_population=float(tract.population),
                route_count_touching_tract=int(route_count),
                baseline_service_intensity=float(baseline_service_intensity),
                candidate_service_intensity=float(candidate_service_intensity),
                baseline_waiting_time_proxy=float(baseline_waiting),
                candidate_waiting_time_proxy=float(candidate_waiting),
                baseline_utility=_tract_utility(
                    baseline_service_intensity,
                    baseline_waiting,
                    params.equity_parameters,
                ),
                candidate_utility=_tract_utility(
                    candidate_service_intensity,
                    candidate_waiting,
                    params.equity_parameters,
                ),
                notes=(
                    "Tract utility uses allocated weekday trips as service intensity and 1/(1+service) as the waiting-time proxy.",
                ),
            )
        )

    epc_tracts = [tract for tract in tract_breakdowns if tract.epc_2050 == 1]
    non_epc_tracts = [tract for tract in tract_breakdowns if tract.epc_2050 == 0]

    baseline_epc_weighted_mean = _weighted_mean(
        [(tract.tract_population, tract.baseline_utility) for tract in epc_tracts]
    )
    current_epc_weighted_mean = _weighted_mean(
        [(tract.tract_population, tract.candidate_utility) for tract in epc_tracts]
    )
    baseline_non_epc_weighted_mean = _weighted_mean(
        [(tract.tract_population, tract.baseline_utility) for tract in non_epc_tracts]
    )
    current_non_epc_weighted_mean = _weighted_mean(
        [(tract.tract_population, tract.candidate_utility) for tract in non_epc_tracts]
    )

    baseline_epc_mean = _mean([tract.baseline_utility for tract in epc_tracts])
    current_epc_mean = _mean([tract.candidate_utility for tract in epc_tracts])
    baseline_non_epc_mean = _mean([tract.baseline_utility for tract in non_epc_tracts])
    current_non_epc_mean = _mean([tract.candidate_utility for tract in non_epc_tracts])

    baseline_population_gap = abs(baseline_non_epc_weighted_mean - baseline_epc_weighted_mean)
    current_population_gap = abs(current_non_epc_weighted_mean - current_epc_weighted_mean)
    baseline_area_gap = abs(baseline_non_epc_mean - baseline_epc_mean)
    current_area_gap = abs(current_non_epc_mean - current_epc_mean)
    population_gap_percent_delta, population_gap_notes = _percent_change_vs_baseline(
        current_value=current_population_gap,
        baseline_value=baseline_population_gap,
    )
    area_gap_percent_delta, area_gap_notes = _percent_change_vs_baseline(
        current_value=current_area_gap,
        baseline_value=baseline_area_gap,
    )

    notes = (
        "Population equity uses the absolute gap between non-EPC and EPC population-weighted mean tract utility.",
        "Area equity uses the absolute gap between non-EPC and EPC unweighted mean tract utility.",
    ) + population_gap_notes + area_gap_notes
    return SystemEquityBreakdown(
        baseline_population_gap=float(baseline_population_gap),
        current_population_gap=float(current_population_gap),
        absolute_population_gap_delta=float(current_population_gap - baseline_population_gap),
        percent_population_gap_delta=float(population_gap_percent_delta),
        baseline_area_gap=float(baseline_area_gap),
        current_area_gap=float(current_area_gap),
        absolute_area_gap_delta=float(current_area_gap - baseline_area_gap),
        percent_area_gap_delta=float(area_gap_percent_delta),
        baseline_epc_weighted_mean_utility=float(baseline_epc_weighted_mean),
        current_epc_weighted_mean_utility=float(current_epc_weighted_mean),
        baseline_non_epc_weighted_mean_utility=float(baseline_non_epc_weighted_mean),
        current_non_epc_weighted_mean_utility=float(current_non_epc_weighted_mean),
        baseline_epc_mean_utility=float(baseline_epc_mean),
        current_epc_mean_utility=float(current_epc_mean),
        baseline_non_epc_mean_utility=float(baseline_non_epc_mean),
        current_non_epc_mean_utility=float(current_non_epc_mean),
        tract_breakdowns=tuple(tract_breakdowns),
        equity_parameters=params.equity_parameters,
        notes=notes,
    )


def _percent_change_vs_baseline(current_value: float, baseline_value: float) -> tuple[float, tuple[str, ...]]:
    if math.isclose(baseline_value, 0.0, abs_tol=1e-12):
        if math.isclose(current_value, 0.0, abs_tol=1e-12):
            return 0.0, ("Baseline value is zero; percent delta is pinned to zero because current is also zero.",)
        return current_value - baseline_value, (
            "Baseline value is zero; percent delta falls back to absolute delta to avoid division by zero.",
        )
    denominator = abs(baseline_value)
    percent_delta = ((current_value - baseline_value) / denominator) * 100.0
    notes: tuple[str, ...] = ()
    if baseline_value < 0.0:
        notes = (
            "Percent delta uses the baseline magnitude so sign remains aligned with the lower-is-better objective when the baseline is negative.",
        )
    return float(percent_delta), notes


def compute_objective_breakdown(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
    parameters: CostParameters | None = None,
    cost_parameters: CostParameters | None = None,
    weekday_ridership: dict[str, float] | None = None,
) -> ObjectiveBreakdown:
    active_domain = domain or get_default_domain()
    params = parameters or cost_parameters or get_default_parameters()

    baseline_cost = compute_cost_breakdown(active_domain.baseline, domain=active_domain, parameters=params)
    candidate_cost = compute_cost_breakdown(y, domain=active_domain, parameters=params)
    baseline_emissions = compute_emissions_breakdown(
        active_domain.baseline,
        domain=active_domain,
        parameters=params,
        weekday_ridership=weekday_ridership,
    )
    candidate_emissions = compute_emissions_breakdown(
        y,
        domain=active_domain,
        parameters=params,
        weekday_ridership=weekday_ridership,
    )
    baseline_equity = compute_equity_breakdown(
        active_domain.baseline,
        domain=active_domain,
        parameters=params,
    )
    candidate_equity = compute_equity_breakdown(
        y,
        domain=active_domain,
        parameters=params,
    )

    cost_percent_delta, cost_notes = _percent_change_vs_baseline(
        current_value=candidate_cost.annual_total_cost,
        baseline_value=baseline_cost.annual_total_cost,
    )
    emissions_percent_delta, emissions_notes = _percent_change_vs_baseline(
        current_value=candidate_emissions.candidate_total_emissions_grams,
        baseline_value=baseline_emissions.candidate_total_emissions_grams,
    )
    equity_percent_delta, equity_notes = _percent_change_vs_baseline(
        current_value=candidate_equity.current_population_gap,
        baseline_value=baseline_equity.current_population_gap,
    )

    cost_contribution = cost_percent_delta * params.objective_weights.cost_percent_change_coefficient.value
    emissions_contribution = (
        emissions_percent_delta * params.objective_weights.emissions_percent_change_coefficient.value
    )
    equity_contribution = (
        equity_percent_delta * params.objective_weights.equity_percent_change_coefficient.value
    )

    notes = (
        "Combined objective is the sum of per-pillar percent deltas multiplied by explicit coefficients.",
    )
    return ObjectiveBreakdown(
        cost=ObjectivePillarBreakdown(
            baseline_value=float(baseline_cost.annual_total_cost),
            current_value=float(candidate_cost.annual_total_cost),
            absolute_delta=float(candidate_cost.annual_total_cost - baseline_cost.annual_total_cost),
            percent_delta=float(cost_percent_delta),
            coefficient=float(params.objective_weights.cost_percent_change_coefficient.value),
            weighted_contribution=float(cost_contribution),
            notes=cost_notes,
        ),
        emissions=ObjectivePillarBreakdown(
            baseline_value=float(baseline_emissions.candidate_total_emissions_grams),
            current_value=float(candidate_emissions.candidate_total_emissions_grams),
            absolute_delta=float(
                candidate_emissions.candidate_total_emissions_grams
                - baseline_emissions.candidate_total_emissions_grams
            ),
            percent_delta=float(emissions_percent_delta),
            coefficient=float(params.objective_weights.emissions_percent_change_coefficient.value),
            weighted_contribution=float(emissions_contribution),
            notes=emissions_notes,
        ),
        equity=ObjectivePillarBreakdown(
            baseline_value=float(baseline_equity.current_population_gap),
            current_value=float(candidate_equity.current_population_gap),
            absolute_delta=float(
                candidate_equity.current_population_gap - baseline_equity.current_population_gap
            ),
            percent_delta=float(equity_percent_delta),
            coefficient=float(params.objective_weights.equity_percent_change_coefficient.value),
            weighted_contribution=float(equity_contribution),
            notes=equity_notes,
        ),
        total_combined_objective=float(cost_contribution + emissions_contribution + equity_contribution),
        notes=notes,
    )


def objective_function(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
    parameters: CostParameters | None = None,
    cost_parameters: CostParameters | None = None,
) -> float:
    breakdown = compute_objective_breakdown(
        y,
        domain=domain,
        parameters=parameters,
        cost_parameters=cost_parameters,
    )
    return float(breakdown.total_combined_objective)


def is_within_route_bounds(y: Sequence[int], domain: RouteFleetDomain | None = None) -> bool:
    active_domain = domain or get_default_domain()
    try:
        solution = _coerce_solution_vector(y, len(active_domain.route_ids))
    except ValueError:
        return False

    for i, val in enumerate(solution):
        if val < active_domain.lower_bounds[i] or val > active_domain.upper_bounds[i]:
            return False
    return True


def has_global_fleet_conservation(y: Sequence[int], domain: RouteFleetDomain | None = None) -> bool:
    active_domain = domain or get_default_domain()
    try:
        solution = _coerce_solution_vector(y, len(active_domain.route_ids))
    except ValueError:
        return False
    return sum(solution) == sum(active_domain.baseline)


def is_feasible_solution(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
    require_global_fleet_conservation: bool = True,
) -> bool:
    active_domain = domain or get_default_domain()
    if not is_within_route_bounds(y, active_domain):
        return False
    if require_global_fleet_conservation and not has_global_fleet_conservation(y, active_domain):
        return False
    return True


def make_add_move(route_index: int) -> Move:
    return Move(move_type="ADD", add_index=route_index)


def make_drop_move(route_index: int) -> Move:
    return Move(move_type="DROP", drop_index=route_index)


def make_swap_move(add_index: int, drop_index: int) -> Move:
    return Move(move_type="SWAP", add_index=add_index, drop_index=drop_index)


def canonical_move_key(move: Move) -> tuple[str, int, int]:
    add_index = -1 if move.add_index is None else move.add_index
    drop_index = -1 if move.drop_index is None else move.drop_index
    return (move.move_type, add_index, drop_index)


def apply_move(
    y: Sequence[int],
    move: Move,
    step: int = 1,
    domain: RouteFleetDomain | None = None,
) -> tuple[int, ...]:
    if step <= 0:
        raise ValueError("Step must be a positive integer")

    active_domain = domain or get_default_domain()
    base = list(_coerce_solution_vector(y, len(active_domain.route_ids)))
    _validate_move(move, len(base))

    if move.move_type == "ADD":
        assert move.add_index is not None
        base[move.add_index] += step
    elif move.move_type == "DROP":
        assert move.drop_index is not None
        base[move.drop_index] -= step
    else:
        assert move.add_index is not None
        assert move.drop_index is not None
        base[move.add_index] += step
        base[move.drop_index] -= step

    return tuple(base)


def compute_sa_acceptance_probability(delta: float, temperature: float) -> float:
    if delta <= 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return float(math.exp(-delta / temperature))


def should_accept_nonimproving(delta: float, temperature: float, rng: random.Random) -> bool:
    prob = compute_sa_acceptance_probability(delta=delta, temperature=temperature)
    return rng.random() < prob


def decay_tenures(add_tenure: list[float], drop_tenure: list[float]) -> None:
    for i in range(len(add_tenure)):
        add_tenure[i] = max(add_tenure[i] - 1.0, 0.0)
        drop_tenure[i] = max(drop_tenure[i] - 1.0, 0.0)


def _evaluate_candidate(
    vector: tuple[int, ...],
    domain: RouteFleetDomain,
    config: SearchConfig,
    parameters: CostParameters | None = None,
    cost_parameters: CostParameters | None = None,
) -> float:
    return objective_function(
        vector,
        domain=domain,
        parameters=parameters,
        cost_parameters=cost_parameters,
    )


def _is_move_tabu(move: Move, add_tenure: Sequence[float], drop_tenure: Sequence[float]) -> bool:
    if move.move_type == "ADD":
        assert move.add_index is not None
        return add_tenure[move.add_index] > 0
    if move.move_type == "DROP":
        assert move.drop_index is not None
        return drop_tenure[move.drop_index] > 0
    assert move.add_index is not None and move.drop_index is not None
    return (add_tenure[move.add_index] > 0) or (drop_tenure[move.drop_index] > 0)


def _first_pass_candidates(
    current: tuple[int, ...],
    domain: RouteFleetDomain,
    config: SearchConfig,
    parameters: CostParameters | None,
    add_tenure: Sequence[float],
    drop_tenure: Sequence[float],
    obj_best: float,
    rng: random.Random,
) -> tuple[list[CandidateNeighbor], list[CandidateNeighbor]]:
    n_routes = len(current)
    add_indices = list(range(n_routes))
    drop_indices = list(range(n_routes))
    rng.shuffle(add_indices)
    rng.shuffle(drop_indices)

    add_candidates: list[CandidateNeighbor] = []
    for idx in add_indices:
        if len(add_candidates) >= config.nbhd_add_lim:
            break
        if current[idx] + config.step > domain.upper_bounds[idx]:
            continue
        move = make_add_move(idx)
        candidate = apply_move(current, move, step=config.step, domain=domain)
        obj = _evaluate_candidate(candidate, domain, config, parameters=parameters)
        if _is_move_tabu(move, add_tenure, drop_tenure) and not (obj < obj_best):
            continue
        add_candidates.append(CandidateNeighbor(move=move, vector=candidate, objective=obj))

    drop_candidates: list[CandidateNeighbor] = []
    for idx in drop_indices:
        if len(drop_candidates) >= config.nbhd_drop_lim:
            break
        if current[idx] - config.step < domain.lower_bounds[idx]:
            continue
        move = make_drop_move(idx)
        candidate = apply_move(current, move, step=config.step, domain=domain)
        obj = _evaluate_candidate(candidate, domain, config, parameters=parameters)
        if _is_move_tabu(move, add_tenure, drop_tenure) and not (obj < obj_best):
            continue
        drop_candidates.append(CandidateNeighbor(move=move, vector=candidate, objective=obj))

    return add_candidates, drop_candidates


def select_neighborhood_candidates(
    current: tuple[int, ...],
    domain: RouteFleetDomain,
    config: SearchConfig,
    parameters: CostParameters | None,
    add_tenure: Sequence[float],
    drop_tenure: Sequence[float],
    obj_best: float,
    rng: random.Random,
) -> tuple[CandidateNeighbor | None, CandidateNeighbor | None]:
    add_moves1, drop_moves1 = _first_pass_candidates(
        current=current,
        domain=domain,
        config=config,
        parameters=parameters,
        add_tenure=add_tenure,
        drop_tenure=drop_tenure,
        obj_best=obj_best,
        rng=rng,
    )

    add_moves2 = sorted(add_moves1, key=lambda n: n.objective)[: config.nbhd_add_lim]
    drop_moves2 = sorted(drop_moves1, key=lambda n: n.objective)[: config.nbhd_drop_lim]

    final_moves: list[CandidateNeighbor] = []

    if not config.require_global_fleet_conservation:
        for cand in add_moves2 + drop_moves2:
            if is_feasible_solution(
                cand.vector,
                domain=domain,
                require_global_fleet_conservation=config.require_global_fleet_conservation,
            ):
                final_moves.append(cand)

    swaps_added = 0
    add_loop = 0
    limit = min(len(add_moves2), len(drop_moves2))
    for add_move in add_moves2:
        if swaps_added >= config.nbhd_swap_lim or add_loop > limit:
            break
        drop_loop = 0
        for drop_move in drop_moves2:
            if drop_loop > add_loop:
                break
            drop_loop += 1

            assert add_move.move.add_index is not None
            assert drop_move.move.drop_index is not None
            add_idx = add_move.move.add_index
            drop_idx = drop_move.move.drop_index
            if add_idx == drop_idx:
                continue

            move = make_swap_move(add_idx, drop_idx)
            candidate = apply_move(current, move, step=config.step, domain=domain)
            if not is_feasible_solution(
                candidate,
                domain=domain,
                require_global_fleet_conservation=config.require_global_fleet_conservation,
            ):
                continue

            obj = _evaluate_candidate(candidate, domain, config, parameters=parameters)
            if _is_move_tabu(move, add_tenure, drop_tenure) and not (obj < obj_best):
                continue

            final_moves.append(CandidateNeighbor(move=move, vector=candidate, objective=obj))
            swaps_added += 1
            if swaps_added >= config.nbhd_swap_lim:
                break
        add_loop += 1

    if not final_moves:
        return None, None

    final_moves.sort(key=lambda n: n.objective)
    best = final_moves[0]
    second = None
    if len(final_moves) > 1:
        second = final_moves[1]
    return best, second


def _apply_inverse_tabu(move: Move, add_tenure: list[float], drop_tenure: list[float], tenure: float) -> None:
    if move.add_index is not None:
        drop_tenure[move.add_index] = tenure
    if move.drop_index is not None:
        add_tenure[move.drop_index] = tenure


def _route_comparison_table(domain: RouteFleetDomain, best_vector: tuple[int, ...]) -> pd.DataFrame:
    out = domain.route_metadata.copy()
    out = out.sort_values("route_index").reset_index(drop=True)
    out["optimized_fleet"] = list(best_vector)
    out["delta_fleet"] = out["optimized_fleet"] - out["weekday_typical_buses"]
    return out


def _route_cost_delta_table(
    baseline: SystemCostBreakdown,
    candidate: SystemCostBreakdown,
) -> pd.DataFrame:
    baseline_rows = {
        route.route_id: route
        for route in baseline.route_breakdowns
    }
    candidate_rows = {
        route.route_id: route
        for route in candidate.route_breakdowns
    }

    rows: list[dict[str, float | int | str]] = []
    for route_id in sorted(candidate_rows):
        base = baseline_rows[route_id]
        cand = candidate_rows[route_id]
        rows.append(
            {
                "route_id": route_id,
                "baseline_fleet": base.baseline_fleet,
                "optimized_fleet": cand.candidate_fleet,
                "delta_fleet": cand.candidate_fleet - base.baseline_fleet,
                "baseline_annual_vehicle_miles": base.candidate_annual_vehicle_miles,
                "optimized_annual_vehicle_miles": cand.candidate_annual_vehicle_miles,
                "delta_annual_vehicle_miles": cand.candidate_annual_vehicle_miles - base.candidate_annual_vehicle_miles,
                "baseline_annual_vehicle_hours": base.candidate_annual_vehicle_hours,
                "optimized_annual_vehicle_hours": cand.candidate_annual_vehicle_hours,
                "delta_annual_vehicle_hours": cand.candidate_annual_vehicle_hours - base.candidate_annual_vehicle_hours,
                "baseline_labor_cost": base.annual_labor_cost,
                "optimized_labor_cost": cand.annual_labor_cost,
                "delta_labor_cost": cand.annual_labor_cost - base.annual_labor_cost,
                "baseline_maintenance_cost": base.annual_maintenance_cost,
                "optimized_maintenance_cost": cand.annual_maintenance_cost,
                "delta_maintenance_cost": cand.annual_maintenance_cost - base.annual_maintenance_cost,
                "baseline_energy_cost": base.annual_energy_cost,
                "optimized_energy_cost": cand.annual_energy_cost,
                "delta_energy_cost": cand.annual_energy_cost - base.annual_energy_cost,
                "baseline_total_cost": base.annual_total_cost,
                "optimized_total_cost": cand.annual_total_cost,
                "delta_total_cost": cand.annual_total_cost - base.annual_total_cost,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("delta_total_cost", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out


def _route_emissions_delta_table(
    baseline: SystemEmissionsBreakdown,
    candidate: SystemEmissionsBreakdown,
) -> pd.DataFrame:
    baseline_rows = {route.route_id: route for route in baseline.route_breakdowns}
    candidate_rows = {route.route_id: route for route in candidate.route_breakdowns}

    rows: list[dict[str, float | int | str]] = []
    for route_id in sorted(candidate_rows):
        base = baseline_rows[route_id]
        cand = candidate_rows[route_id]
        rows.append(
            {
                "route_id": route_id,
                "baseline_fleet": base.baseline_fleet,
                "optimized_fleet": cand.candidate_fleet,
                "delta_fleet": cand.candidate_fleet - base.baseline_fleet,
                "baseline_riders": base.candidate_riders,
                "optimized_riders": cand.candidate_riders,
                "delta_riders": cand.candidate_riders - base.candidate_riders,
                "baseline_bus_emissions_grams": base.candidate_bus_emissions_grams,
                "optimized_bus_emissions_grams": cand.candidate_bus_emissions_grams,
                "delta_bus_emissions_grams": cand.candidate_bus_emissions_grams - base.candidate_bus_emissions_grams,
                "baseline_rider_emissions_avoided_grams": base.candidate_rider_emissions_avoided_grams,
                "optimized_rider_emissions_avoided_grams": cand.candidate_rider_emissions_avoided_grams,
                "delta_rider_emissions_avoided_grams": (
                    cand.candidate_rider_emissions_avoided_grams - base.candidate_rider_emissions_avoided_grams
                ),
                "baseline_net_emissions_grams": base.candidate_net_emissions_grams,
                "optimized_net_emissions_grams": cand.candidate_net_emissions_grams,
                "delta_net_emissions_grams": cand.candidate_net_emissions_grams - base.candidate_net_emissions_grams,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            "delta_net_emissions_grams",
            key=lambda s: s.abs(),
            ascending=False,
        ).reset_index(drop=True)
    return out


def _tract_equity_delta_table(
    baseline: SystemEquityBreakdown,
    candidate: SystemEquityBreakdown,
) -> pd.DataFrame:
    columns = [
        "geoid",
        "epc_2050",
        "epc_class",
        "tract_population",
        "route_count_touching_tract",
        "baseline_service_intensity",
        "optimized_service_intensity",
        "delta_service_intensity",
        "baseline_utility",
        "optimized_utility",
        "delta_utility",
    ]
    baseline_rows = {tract.geoid: tract for tract in baseline.tract_breakdowns}
    candidate_rows = {tract.geoid: tract for tract in candidate.tract_breakdowns}

    rows: list[dict[str, float | int | str]] = []
    for geoid in sorted(candidate_rows):
        base = baseline_rows[geoid]
        cand = candidate_rows[geoid]
        rows.append(
            {
                "geoid": geoid,
                "epc_2050": cand.epc_2050,
                "epc_class": cand.epc_class,
                "tract_population": cand.tract_population,
                "route_count_touching_tract": cand.route_count_touching_tract,
                "baseline_service_intensity": base.candidate_service_intensity,
                "optimized_service_intensity": cand.candidate_service_intensity,
                "delta_service_intensity": cand.candidate_service_intensity - base.candidate_service_intensity,
                "baseline_utility": base.candidate_utility,
                "optimized_utility": cand.candidate_utility,
                "delta_utility": cand.candidate_utility - base.candidate_utility,
            }
        )

    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values("delta_utility", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out


def run_route_fleet_search(
    domain: RouteFleetDomain | None = None,
    config: SearchConfig | None = None,
    seed: int = 0,
    initial_solution: Sequence[int] | None = None,
    parameters: CostParameters | None = None,
    cost_parameters: CostParameters | None = None,
) -> SearchResult:
    active_domain = domain or get_default_domain()
    cfg = config or SearchConfig()
    active_parameters = parameters or cost_parameters or get_default_parameters()
    rng = random.Random(seed)

    if cfg.max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")
    if cfg.step <= 0:
        raise ValueError("step must be > 0")

    if initial_solution is None:
        sol_current = tuple(active_domain.baseline)
    else:
        sol_current = _coerce_solution_vector(initial_solution, len(active_domain.route_ids))

    if not is_feasible_solution(
        sol_current,
        domain=active_domain,
        require_global_fleet_conservation=cfg.require_global_fleet_conservation,
    ):
        raise ValueError("Initial solution is infeasible")

    n_routes = len(sol_current)
    add_tenure = [0.0] * n_routes
    drop_tenure = [0.0] * n_routes
    tenure = float(cfg.tenure_init)
    temperature = float(cfg.temp_init)

    obj_current = _evaluate_candidate(sol_current, active_domain, cfg, parameters=active_parameters)
    sol_best = sol_current
    obj_best = obj_current

    nonimp_in = 0
    nonimp_out = 0
    accepted_improving = 0
    accepted_nonimproving = 0
    attractive_solutions: list[tuple[tuple[int, ...], float]] = []
    events: list[dict] = []

    for iteration in range(1, cfg.max_iterations + 1):
        best, second = select_neighborhood_candidates(
            current=sol_current,
            domain=active_domain,
            config=cfg,
            parameters=active_parameters,
            add_tenure=add_tenure,
            drop_tenure=drop_tenure,
            obj_best=obj_best,
            rng=rng,
        )

        event_case = "no_move"
        accepted = False
        jumped = False
        sa_prob = None
        selected_move_key: tuple[str, int, int] | None = None

        if best is not None:
            selected_move_key = canonical_move_key(best.move)
            if best.objective < obj_current:
                event_case = "improvement"
                accepted = True
                accepted_improving += 1
                nonimp_in = 0
                nonimp_out = 0
                tenure = float(cfg.tenure_init)
                sol_current = best.vector
                obj_current = best.objective
                _apply_inverse_tabu(best.move, add_tenure, drop_tenure, tenure)

                if obj_current < obj_best:
                    sol_best = sol_current
                    obj_best = obj_current
            else:
                nonimp_in += 1
                nonimp_out += 1
                delta = best.objective - obj_current
                sa_prob = compute_sa_acceptance_probability(delta, temperature)

                if should_accept_nonimproving(delta, temperature, rng):
                    event_case = "nonimp_accept"
                    accepted = True
                    accepted_nonimproving += 1
                    nonimp_in = 0
                    tenure *= cfg.tenure_factor
                    sol_current = best.vector
                    obj_current = best.objective
                    _apply_inverse_tabu(best.move, add_tenure, drop_tenure, tenure)
                    if second is not None:
                        attractive_solutions.append((second.vector, second.objective))
                else:
                    event_case = "nonimp_reject"
                    if best is not None:
                        attractive_solutions.append((best.vector, best.objective))

        if len(attractive_solutions) > cfg.attractive_max:
            remove_idx = rng.randrange(len(attractive_solutions))
            attractive_solutions.pop(remove_idx)

        if nonimp_in > cfg.nonimp_in_max and attractive_solutions:
            nonimp_in = 0
            nonimp_out += 1
            tenure *= cfg.tenure_factor
            jump_idx = rng.randrange(len(attractive_solutions))
            sol_current, obj_current = attractive_solutions.pop(jump_idx)
            jumped = True
            event_case = f"{event_case}+jump"

        if nonimp_out > cfg.nonimp_out_max:
            tenure = float(cfg.tenure_init)

        decay_tenures(add_tenure, drop_tenure)
        temperature *= cfg.temp_factor

        feasible_current = is_feasible_solution(
            sol_current,
            domain=active_domain,
            require_global_fleet_conservation=cfg.require_global_fleet_conservation,
        )
        if not feasible_current:
            raise RuntimeError("Search produced infeasible current solution")

        events.append(
            {
                "iteration": iteration,
                "event_case": event_case,
                "accepted": accepted,
                "jumped": jumped,
                "move": selected_move_key,
                "obj_current": float(obj_current),
                "obj_best": float(obj_best),
                "sa_prob": None if sa_prob is None else float(sa_prob),
                "nonimp_in": nonimp_in,
                "nonimp_out": nonimp_out,
                "tenure": float(tenure),
                "temperature": float(temperature),
                "current_vector": sol_current,
                "feasible_current": feasible_current,
            }
        )

    initial_breakdown = compute_cost_breakdown(
        active_domain.baseline,
        domain=active_domain,
        parameters=active_parameters,
    )
    initial_emissions_breakdown = compute_emissions_breakdown(
        active_domain.baseline,
        domain=active_domain,
        parameters=active_parameters,
    )
    initial_equity_breakdown = compute_equity_breakdown(
        active_domain.baseline,
        domain=active_domain,
        parameters=active_parameters,
    )
    initial_objective_breakdown = compute_objective_breakdown(
        active_domain.baseline,
        domain=active_domain,
        parameters=active_parameters,
    )
    best_breakdown = compute_cost_breakdown(
        sol_best,
        domain=active_domain,
        parameters=active_parameters,
    )
    best_emissions_breakdown = compute_emissions_breakdown(
        sol_best,
        domain=active_domain,
        parameters=active_parameters,
    )
    best_equity_breakdown = compute_equity_breakdown(
        sol_best,
        domain=active_domain,
        parameters=active_parameters,
    )
    best_objective_breakdown = compute_objective_breakdown(
        sol_best,
        domain=active_domain,
        parameters=active_parameters,
    )
    route_cost_delta_table = _route_cost_delta_table(initial_breakdown, best_breakdown)
    route_emissions_delta_table = _route_emissions_delta_table(
        initial_emissions_breakdown,
        best_emissions_breakdown,
    )
    tract_equity_delta_table = _tract_equity_delta_table(
        initial_equity_breakdown,
        best_equity_breakdown,
    )

    return SearchResult(
        best_vector=sol_best,
        best_objective=float(best_objective_breakdown.total_combined_objective),
        initial_vector=tuple(active_domain.baseline),
        initial_objective=float(initial_objective_breakdown.total_combined_objective),
        iterations_completed=cfg.max_iterations,
        accepted_improving_moves=accepted_improving,
        accepted_nonimproving_moves=accepted_nonimproving,
        events=tuple(events),
        best_route_table=_route_comparison_table(active_domain, sol_best),
        initial_cost_breakdown=initial_breakdown,
        best_cost_breakdown=best_breakdown,
        initial_emissions_breakdown=initial_emissions_breakdown,
        best_emissions_breakdown=best_emissions_breakdown,
        initial_equity_breakdown=initial_equity_breakdown,
        best_equity_breakdown=best_equity_breakdown,
        initial_objective_breakdown=initial_objective_breakdown,
        best_objective_breakdown=best_objective_breakdown,
        annual_cost_delta_vs_baseline=float(best_breakdown.annual_total_cost - initial_breakdown.annual_total_cost),
        initial_budget_slack=float(initial_breakdown.annual_budget_slack),
        best_budget_slack=float(best_breakdown.annual_budget_slack),
        route_cost_delta_table=route_cost_delta_table,
        route_emissions_delta_table=route_emissions_delta_table,
        tract_equity_delta_table=tract_equity_delta_table,
    )


def _route_color(route_id: str) -> str:
    digest = hashlib.md5(route_id.encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


def render_intersections_and_routes_map() -> pathlib.Path:
    intersections = pd.read_csv(find_data_file("intersections.csv"))
    route_stops = pd.read_csv(find_data_file("simplified_bus_route_stops.csv"), dtype={"route_id": str})

    center_lat = float(pd.to_numeric(intersections["Latitude"], errors="coerce").mean())
    center_lon = float(pd.to_numeric(intersections["Longitude"], errors="coerce").mean())

    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    intersections_layer = folium.FeatureGroup(name="intersections", show=True)
    for row in intersections.itertuples(index=False):
        folium.CircleMarker(
            location=[row.Latitude, row.Longitude],
            radius=1,
            fill=True,
            tooltip=f"{row.Street_Name_1} x {row.Street_Name_2} (zip {row.Zip_Code})",
        ).add_to(intersections_layer)
    intersections_layer.add_to(map_obj)

    route_stops["stop_lat"] = pd.to_numeric(route_stops["stop_lat"], errors="coerce")
    route_stops["stop_lon"] = pd.to_numeric(route_stops["stop_lon"], errors="coerce")
    route_stops["route_stop_order"] = pd.to_numeric(route_stops["route_stop_order"], errors="coerce")
    route_stops["direction_id"] = route_stops["direction_id"].fillna("0").astype(str)
    route_stops = route_stops.dropna(subset=["stop_lat", "stop_lon", "route_stop_order"])

    bus_lines_layer = folium.FeatureGroup(name="bus_lines", show=True)
    for (route_id, direction_id), group in route_stops.groupby(["route_id", "direction_id"], sort=True):
        ordered = group.sort_values(["route_stop_order", "stop_id"])
        coords = list(zip(ordered["stop_lat"], ordered["stop_lon"]))
        if len(coords) < 2:
            continue

        short_name = str(ordered["route_short_name"].iloc[0])
        long_name = str(ordered["route_long_name"].iloc[0])
        folium.PolyLine(
            locations=coords,
            color=_route_color(f"{route_id}-{direction_id}"),
            weight=3,
            opacity=0.8,
            tooltip=f"{short_name} dir {direction_id}: {long_name}",
        ).add_to(bus_lines_layer)
    bus_lines_layer.add_to(map_obj)

    folium.LayerControl(collapsed=False).add_to(map_obj)

    output = pathlib.Path("sf_intersections_and_bus_lines.html")
    map_obj.save(output)
    return output


if __name__ == "__main__":
    render_intersections_and_routes_map()
