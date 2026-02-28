import hashlib
import json
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import folium
import pandas as pd

from data_utils import find_data_file


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


@dataclass(frozen=True)
class RouteFleetDomain:
    route_ids: tuple[str, ...]
    baseline: tuple[int, ...]
    lower_bounds: tuple[int, ...]
    upper_bounds: tuple[int, ...]
    service_weights: tuple[float, ...]
    route_driver_estimates: tuple["RouteCostDriverEstimate", ...]
    route_metadata: pd.DataFrame


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
    annual_cost_delta_vs_baseline: float
    initial_budget_slack: float
    best_budget_slack: float
    route_cost_delta_table: pd.DataFrame


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
class RouteCostDriverEstimate:
    route_id: str
    baseline_fleet: int
    weekday_planned_trips: float
    direction_count: int
    one_way_distance_miles: float
    round_trip_distance_miles: float
    drivers_estimated: bool
    notes: tuple[str, ...]


_DEFAULT_DOMAIN: RouteFleetDomain | None = None
_DEFAULT_COST_PARAMETERS: CostParameters | None = None


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


def load_cost_parameters(data_path: str | pathlib.Path | None = None) -> CostParameters:
    if data_path is None:
        json_path = find_data_file("cost_parameters.json")
    else:
        json_path = pathlib.Path(data_path)

    with json_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    root = _require_mapping(raw, "root")
    expected_blocks = {"reporting_constants", "operating_cost_parameters", "estimation_assumptions"}
    missing_blocks = expected_blocks.difference(root)
    if missing_blocks:
        raise ValueError(f"Missing required blocks in {json_path}: {sorted(missing_blocks)}")

    reporting_raw = _require_mapping(root["reporting_constants"], "reporting_constants")
    operating_raw = _require_mapping(root["operating_cost_parameters"], "operating_cost_parameters")
    estimation_raw = _require_mapping(root["estimation_assumptions"], "estimation_assumptions")

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

    return CostParameters(
        reporting_constants=reporting,
        operating_cost_parameters=operating,
        estimation_assumptions=assumptions,
    )



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

    out = route_stops.loc[:, sorted(required)].copy()
    out["direction_id"] = out["direction_id"].fillna("0").astype(str)
    out["route_stop_order"] = pd.to_numeric(out["route_stop_order"], errors="coerce")
    out["stop_lat"] = pd.to_numeric(out["stop_lat"], errors="coerce")
    out["stop_lon"] = pd.to_numeric(out["stop_lon"], errors="coerce")
    out = out.dropna(subset=["route_id", "route_stop_order", "stop_lat", "stop_lon"])
    out["route_id"] = out["route_id"].astype(str)
    return out.sort_values(["route_id", "direction_id", "route_stop_order"]).reset_index(drop=True)


def _compute_route_direction_lengths(route_stops: pd.DataFrame) -> dict[str, dict[str, float]]:
    lengths_by_route: dict[str, dict[str, float]] = {}
    for (route_id, direction_id), group in route_stops.groupby(["route_id", "direction_id"], sort=True):
        ordered = group.sort_values(["route_stop_order"]).reset_index(drop=True)
        coords = list(zip(ordered["stop_lat"], ordered["stop_lon"]))
        if len(coords) < 2:
            lengths_by_route.setdefault(route_id, {})[direction_id] = 0.0
            continue

        length_miles = 0.0
        for (lat1, lon1), (lat2, lon2) in zip(coords, coords[1:]):
            length_miles += _haversine_miles(float(lat1), float(lon1), float(lat2), float(lon2))
        lengths_by_route.setdefault(route_id, {})[direction_id] = float(length_miles)
    return lengths_by_route


def _build_route_driver_estimates(
    table: pd.DataFrame,
    route_stops_path: str | pathlib.Path | None = None,
) -> tuple[RouteCostDriverEstimate, ...]:
    direction_lengths: dict[str, dict[str, float]] = {}
    try:
        if route_stops_path is not None:
            route_stops = _load_route_stops_table(route_stops_path)
            direction_lengths = _compute_route_direction_lengths(route_stops)
        elif table.attrs.get("source_is_default", False):
            route_stops = _load_route_stops_table()
            direction_lengths = _compute_route_direction_lengths(route_stops)
    except FileNotFoundError:
        direction_lengths = {}

    estimates: list[RouteCostDriverEstimate] = []
    for row in table.itertuples(index=False):
        per_direction = direction_lengths.get(str(row.route_id), {})
        directional_lengths = [float(per_direction[key]) for key in sorted(per_direction)]
        direction_count = len(directional_lengths)

        notes: list[str] = [
            "Stop-to-stop haversine path length is an approximation, not GTFS shape ground truth.",
            "Weekday planned trips are treated as one-way revenue trips.",
        ]
        if direction_count == 0:
            one_way_distance_miles = 0.0
            round_trip_distance_miles = 0.0
            drivers_estimated = False
            notes.append("No route-stop geometry was available for this route.")
        elif direction_count == 1:
            one_way_distance_miles = directional_lengths[0]
            round_trip_distance_miles = directional_lengths[0] * 2.0
            drivers_estimated = True
            notes.append("Only one direction was available; round-trip distance doubles the observed direction.")
        else:
            one_way_distance_miles = sum(directional_lengths) / float(direction_count)
            round_trip_distance_miles = sum(directional_lengths)
            drivers_estimated = True
            notes.append("Round-trip distance sums observed directional paths; one-way distance is their mean.")

        estimates.append(
            RouteCostDriverEstimate(
                route_id=str(row.route_id),
                baseline_fleet=int(row.weekday_typical_buses),
                weekday_planned_trips=float(row.weekday_planned_trips),
                direction_count=direction_count,
                one_way_distance_miles=float(one_way_distance_miles),
                round_trip_distance_miles=float(round_trip_distance_miles),
                drivers_estimated=drivers_estimated,
                notes=tuple(notes),
            )
        )

    return tuple(estimates)


def load_route_fleet_domain(
    data_path: str | pathlib.Path | None = None,
    route_stops_path: str | pathlib.Path | None = None,
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
    metadata["direction_count"] = metadata["route_id"].map(
        lambda route_id: driver_lookup[str(route_id)].direction_count
    )

    return RouteFleetDomain(
        route_ids=tuple(table["route_id"].tolist()),
        baseline=tuple(int(v) for v in baseline.tolist()),
        lower_bounds=tuple(int(v) for v in lower.tolist()),
        upper_bounds=tuple(int(v) for v in upper.tolist()),
        service_weights=tuple(float(v) for v in service.tolist()),
        route_driver_estimates=route_driver_estimates,
        route_metadata=metadata,
    )


def get_default_domain() -> RouteFleetDomain:
    global _DEFAULT_DOMAIN
    if _DEFAULT_DOMAIN is None:
        _DEFAULT_DOMAIN = load_route_fleet_domain()
    return _DEFAULT_DOMAIN


def get_default_cost_parameters() -> CostParameters:
    global _DEFAULT_COST_PARAMETERS
    if _DEFAULT_COST_PARAMETERS is None:
        _DEFAULT_COST_PARAMETERS = load_cost_parameters()
    return _DEFAULT_COST_PARAMETERS


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
    cost_parameters: CostParameters | None = None,
) -> SystemCostBreakdown:
    active_domain = domain or get_default_domain()
    params = cost_parameters or get_default_cost_parameters()
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
    cost_parameters: CostParameters | None = None,
) -> float:
    return float(compute_cost_breakdown(y, domain=domain, cost_parameters=cost_parameters).objective_cost)


def objective_function(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
    cost_parameters: CostParameters | None = None,
) -> float:
    return cost_objective(y, domain=domain, cost_parameters=cost_parameters)


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
    cost_parameters: CostParameters | None = None,
) -> float:
    return objective_function(vector, domain=domain, cost_parameters=cost_parameters)


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
    cost_parameters: CostParameters | None,
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
        obj = _evaluate_candidate(candidate, domain, config, cost_parameters=cost_parameters)
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
        obj = _evaluate_candidate(candidate, domain, config, cost_parameters=cost_parameters)
        if _is_move_tabu(move, add_tenure, drop_tenure) and not (obj < obj_best):
            continue
        drop_candidates.append(CandidateNeighbor(move=move, vector=candidate, objective=obj))

    return add_candidates, drop_candidates


def select_neighborhood_candidates(
    current: tuple[int, ...],
    domain: RouteFleetDomain,
    config: SearchConfig,
    cost_parameters: CostParameters | None,
    add_tenure: Sequence[float],
    drop_tenure: Sequence[float],
    obj_best: float,
    rng: random.Random,
) -> tuple[CandidateNeighbor | None, CandidateNeighbor | None]:
    add_moves1, drop_moves1 = _first_pass_candidates(
        current=current,
        domain=domain,
        config=config,
        cost_parameters=cost_parameters,
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

            obj = _evaluate_candidate(candidate, domain, config, cost_parameters=cost_parameters)
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


def run_route_fleet_search(
    domain: RouteFleetDomain | None = None,
    config: SearchConfig | None = None,
    seed: int = 0,
    initial_solution: Sequence[int] | None = None,
    cost_parameters: CostParameters | None = None,
) -> SearchResult:
    active_domain = domain or get_default_domain()
    cfg = config or SearchConfig()
    active_cost_parameters = cost_parameters or get_default_cost_parameters()
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

    obj_current = _evaluate_candidate(sol_current, active_domain, cfg, cost_parameters=active_cost_parameters)
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
            cost_parameters=active_cost_parameters,
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
        cost_parameters=active_cost_parameters,
    )
    best_breakdown = compute_cost_breakdown(
        sol_best,
        domain=active_domain,
        cost_parameters=active_cost_parameters,
    )
    route_cost_delta_table = _route_cost_delta_table(initial_breakdown, best_breakdown)

    return SearchResult(
        best_vector=sol_best,
        best_objective=float(obj_best),
        initial_vector=tuple(active_domain.baseline),
        initial_objective=float(initial_breakdown.objective_cost),
        iterations_completed=cfg.max_iterations,
        accepted_improving_moves=accepted_improving,
        accepted_nonimproving_moves=accepted_nonimproving,
        events=tuple(events),
        best_route_table=_route_comparison_table(active_domain, sol_best),
        initial_cost_breakdown=initial_breakdown,
        best_cost_breakdown=best_breakdown,
        annual_cost_delta_vs_baseline=float(best_breakdown.annual_total_cost - initial_breakdown.annual_total_cost),
        initial_budget_slack=float(initial_breakdown.annual_budget_slack),
        best_budget_slack=float(best_breakdown.annual_budget_slack),
        route_cost_delta_table=route_cost_delta_table,
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
