import hashlib
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import folium
import pandas as pd

from data_utils import find_data_file


@dataclass(frozen=True)
class RouteFleetDomain:
    route_ids: tuple[str, ...]
    baseline: tuple[int, ...]
    lower_bounds: tuple[int, ...]
    upper_bounds: tuple[int, ...]
    service_weights: tuple[float, ...]
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
    deviation_weight: float = 1.0
    service_weight: float = 0.25


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


@dataclass(frozen=True)
class CandidateNeighbor:
    move: Move
    vector: tuple[int, ...]
    objective: float


_DEFAULT_DOMAIN: RouteFleetDomain | None = None


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


def load_route_fleet_domain(data_path: str | pathlib.Path | None = None) -> RouteFleetDomain:
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

    return RouteFleetDomain(
        route_ids=tuple(table["route_id"].tolist()),
        baseline=tuple(int(v) for v in baseline.tolist()),
        lower_bounds=tuple(int(v) for v in lower.tolist()),
        upper_bounds=tuple(int(v) for v in upper.tolist()),
        service_weights=tuple(float(v) for v in service.tolist()),
        route_metadata=metadata,
    )


def get_default_domain() -> RouteFleetDomain:
    global _DEFAULT_DOMAIN
    if _DEFAULT_DOMAIN is None:
        _DEFAULT_DOMAIN = load_route_fleet_domain()
    return _DEFAULT_DOMAIN


def objective_function(
    y: Sequence[int],
    domain: RouteFleetDomain | None = None,
    deviation_weight: float = .0,
    service_weight: float = 0.25,
) -> float:
    if deviation_weight < 0 or service_weight < 0:
        raise ValueError("Objective weights must be non-negative")

    active_domain = domain or get_default_domain()
    solution = _coerce_solution_vector(y, len(active_domain.route_ids))

    if not is_within_route_bounds(solution, active_domain):
        raise ValueError("Solution violates per-route bounds")

    deviation_term = 0.0
    service_term = 0.0
    for i, val in enumerate(solution):
        base = active_domain.baseline[i]
        route_weight = active_domain.service_weights[i]
        deviation_term += abs(val - base) * route_weight
        service_term += val * route_weight

    obj = (deviation_weight * deviation_term) - (service_weight * service_term)
    return float(obj)


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
) -> float:
    return objective_function(
        vector,
        domain=domain,
        deviation_weight=config.deviation_weight,
        service_weight=config.service_weight,
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
        obj = _evaluate_candidate(candidate, domain, config)
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
        obj = _evaluate_candidate(candidate, domain, config)
        if _is_move_tabu(move, add_tenure, drop_tenure) and not (obj < obj_best):
            continue
        drop_candidates.append(CandidateNeighbor(move=move, vector=candidate, objective=obj))

    return add_candidates, drop_candidates


def select_neighborhood_candidates(
    current: tuple[int, ...],
    domain: RouteFleetDomain,
    config: SearchConfig,
    add_tenure: Sequence[float],
    drop_tenure: Sequence[float],
    obj_best: float,
    rng: random.Random,
) -> tuple[CandidateNeighbor | None, CandidateNeighbor | None]:
    add_moves1, drop_moves1 = _first_pass_candidates(
        current=current,
        domain=domain,
        config=config,
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

            obj = _evaluate_candidate(candidate, domain, config)
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


def run_route_fleet_search(
    domain: RouteFleetDomain | None = None,
    config: SearchConfig | None = None,
    seed: int = 0,
    initial_solution: Sequence[int] | None = None,
) -> SearchResult:
    active_domain = domain or get_default_domain()
    cfg = config or SearchConfig()
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

    obj_current = _evaluate_candidate(sol_current, active_domain, cfg)
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

    return SearchResult(
        best_vector=sol_best,
        best_objective=float(obj_best),
        initial_vector=tuple(active_domain.baseline),
        initial_objective=float(_evaluate_candidate(active_domain.baseline, active_domain, cfg)),
        iterations_completed=cfg.max_iterations,
        accepted_improving_moves=accepted_improving,
        accepted_nonimproving_moves=accepted_nonimproving,
        events=tuple(events),
        best_route_table=_route_comparison_table(active_domain, sol_best),
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
