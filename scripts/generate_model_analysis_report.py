#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle
from shapely.geometry import LineString

from model import (
    CostParameters,
    ObjectiveWeights,
    ParameterValue,
    RouteFleetDomain,
    SearchConfig,
    SearchResult,
    load_parameters,
    load_route_fleet_domain,
    run_route_fleet_search,
)

FULL_ITERATION_LADDER = (1, 4, 16, 64, 256)
DEFAULT_ITERATION_LADDER = (1, 4, 16, 64)
DEFAULT_SCENARIOS_PATH = Path("data/model_analysis_weight_scenarios.json")
PILLAR_COLORS = {
    "cost": "#0f766e",
    "emissions": "#ea580c",
    "equity": "#2563eb",
}
FIGURE_FILENAMES = (
    "objective_vs_iteration.png",
    "pillar_outcomes_by_iteration.png",
    "strategy_tradeoffs.png",
    "sacrifice_cases.png",
    "cost_benefit_summary.png",
    "research_workflow.png",
    "model_components.png",
    "baseline_route_map.png",
    "optimized_route_map.png",
    "route_delta_map.png",
)


@dataclass(frozen=True)
class WeightScenario:
    scenario_id: str
    run_label: str
    cost_weight: float
    emissions_weight: float
    equity_weight: float


@dataclass(frozen=True)
class RunContext:
    scenario_id: str
    run_label: str
    cost_weight: float
    emissions_weight: float
    equity_weight: float
    seed: int
    iterations: int
    run_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate optimizer sweep artifacts and figures.")
    parser.add_argument("--routes-csv", default="data/simplified_bus_routes.csv")
    parser.add_argument("--route-stops-csv", default="data/simplified_bus_route_stops.csv")
    parser.add_argument("--parameters-json", default="data/parameters.json")
    parser.add_argument("--scenarios-json", default=str(DEFAULT_SCENARIOS_PATH))
    parser.add_argument("--output-dir", help="Optional deterministic output directory.")
    parser.add_argument(
        "--from-artifacts-dir",
        help="Regenerate figures from an existing artifact directory without rerunning the search.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--iteration-ladder",
        default="1,4,16,64",
        help="Comma-separated iteration budgets to run. Default: 1,4,16,64",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        help="Restrict the sweep to one or more scenario IDs from the scenarios JSON file.",
    )
    parser.add_argument("--skip-figures", action="store_true", help="Write artifacts only.")
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
    return parser.parse_args()


def parse_iteration_ladder(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid iteration value '{item}' in --iteration-ladder") from exc
        if value <= 0:
            raise ValueError("--iteration-ladder values must be positive integers")
        values.append(value)
    if not values:
        raise ValueError("--iteration-ladder must contain at least one positive integer")
    return tuple(values)


def load_weight_scenarios(
    scenarios_path: str | Path = DEFAULT_SCENARIOS_PATH,
    *,
    scenario_ids: Sequence[str] | None = None,
) -> list[WeightScenario]:
    path = Path(scenarios_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must contain a non-empty JSON array of weight scenarios")

    scenarios: list[WeightScenario] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario entry {index} in {path} must be an object")
        scenario_id = str(item.get("scenario_id", "")).strip()
        run_label = str(item.get("run_label") or scenario_id).strip()
        if not scenario_id:
            raise ValueError(f"Scenario entry {index} in {path} is missing scenario_id")
        if scenario_id in seen_ids:
            raise ValueError(f"Scenario IDs must be unique; duplicate '{scenario_id}' found in {path}")
        scenarios.append(
            WeightScenario(
                scenario_id=scenario_id,
                run_label=run_label,
                cost_weight=float(item["cost_weight"]),
                emissions_weight=float(item["emissions_weight"]),
                equity_weight=float(item["equity_weight"]),
            )
        )
        seen_ids.add(scenario_id)

    if scenario_ids:
        selected = set(scenario_ids)
        missing = sorted(selected.difference(seen_ids))
        if missing:
            raise ValueError(f"Requested scenario IDs not found in {path}: {missing}")
        scenarios = [scenario for scenario in scenarios if scenario.scenario_id in selected]

    if not scenarios:
        raise ValueError("Scenario selection produced an empty sweep")
    return scenarios


def build_search_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        max_iterations=DEFAULT_ITERATION_LADDER[-1],
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


def apply_weight_scenario(base_parameters: CostParameters, scenario: WeightScenario) -> CostParameters:
    objective_weights = ObjectiveWeights(
        cost_percent_change_coefficient=ParameterValue(
            value=float(scenario.cost_weight),
            label=base_parameters.objective_weights.cost_percent_change_coefficient.label,
        ),
        emissions_percent_change_coefficient=ParameterValue(
            value=float(scenario.emissions_weight),
            label=base_parameters.objective_weights.emissions_percent_change_coefficient.label,
        ),
        equity_percent_change_coefficient=ParameterValue(
            value=float(scenario.equity_weight),
            label=base_parameters.objective_weights.equity_percent_change_coefficient.label,
        ),
    )
    return replace(base_parameters, objective_weights=objective_weights)


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / f"model_analysis_report_{timestamp}"


def _run_context(scenario: WeightScenario, seed: int, iterations: int) -> RunContext:
    return RunContext(
        scenario_id=scenario.scenario_id,
        run_label=scenario.run_label,
        cost_weight=float(scenario.cost_weight),
        emissions_weight=float(scenario.emissions_weight),
        equity_weight=float(scenario.equity_weight),
        seed=int(seed),
        iterations=int(iterations),
        run_id=f"{scenario.scenario_id}__iter_{int(iterations):03d}__seed_{int(seed)}",
    )


def _context_columns(context: RunContext) -> dict[str, Any]:
    return asdict(context)


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def safe_ratio(numerator: float, denominator: float, *, tol: float = 1e-9) -> tuple[float | None, str]:
    if denominator is None or not math.isfinite(float(denominator)):
        return None, "undefined_non_finite_denominator"
    if abs(float(denominator)) <= tol:
        return None, "undefined_zero_denominator"
    return float(numerator) / float(denominator), "ok"


def classify_sacrifice(row: pd.Series, tol: float = 1e-6) -> dict[str, bool]:
    improved = {
        "cost": float(row["cost_percent_delta"]) < -tol,
        "emissions": float(row["emissions_percent_delta"]) < -tol,
        "equity": float(row["equity_percent_delta"]) < -tol,
    }
    worsened = {
        "cost": float(row["cost_percent_delta"]) > tol,
        "emissions": float(row["emissions_percent_delta"]) > tol,
        "equity": float(row["equity_percent_delta"]) > tol,
    }
    sacrifice_cost = worsened["cost"] and improved["emissions"] and improved["equity"]
    sacrifice_emissions = worsened["emissions"] and improved["cost"] and improved["equity"]
    sacrifice_equity = worsened["equity"] and improved["cost"] and improved["emissions"]
    return {
        "sacrifice_cost_for_emissions_equity": sacrifice_cost,
        "sacrifice_emissions_for_cost_equity": sacrifice_emissions,
        "sacrifice_equity_for_cost_emissions": sacrifice_equity,
        "is_sacrifice_case": sacrifice_cost or sacrifice_emissions or sacrifice_equity,
    }


def flatten_iteration_events(result: SearchResult, context: RunContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in result.events:
        move = event.get("move")
        rows.append(
            {
                **_context_columns(context),
                "iteration": int(event["iteration"]),
                "event_case": str(event["event_case"]),
                "accepted": bool(event["accepted"]),
                "jumped": bool(event["jumped"]),
                "move_type": None if move is None else move[0],
                "move_add_index": None if move is None else int(move[1]),
                "move_drop_index": None if move is None else int(move[2]),
                "obj_current": float(event["obj_current"]),
                "obj_best": float(event["obj_best"]),
                "sa_prob": None if event["sa_prob"] is None else float(event["sa_prob"]),
                "nonimp_in": int(event["nonimp_in"]),
                "nonimp_out": int(event["nonimp_out"]),
                "tenure": float(event["tenure"]),
                "temperature": float(event["temperature"]),
                "current_vector": json.dumps(list(event["current_vector"])),
                "feasible_current": bool(event["feasible_current"]),
            }
        )
    return rows


def flatten_objective_history(result: SearchResult, context: RunContext) -> list[dict[str, Any]]:
    return [
        {
            **_context_columns(context),
            "iteration": int(event["iteration"]),
            "current_objective": float(event["obj_current"]),
            "best_objective": float(event["obj_best"]),
            "accepted": bool(event["accepted"]),
            "event_case": str(event["event_case"]),
        }
        for event in result.events
    ]


def flatten_route_deltas(result: SearchResult, context: RunContext, domain: RouteFleetDomain) -> pd.DataFrame:
    route_delta = result.route_cost_delta_table.merge(
        result.route_emissions_delta_table,
        on=["route_id", "baseline_fleet", "optimized_fleet", "delta_fleet"],
        how="outer",
    )
    metadata_columns = [
        "route_id",
        "route_short_name",
        "route_long_name",
        "route_index",
        "weekday_typical_buses",
        "weekday_planned_trips",
        "stop_count",
        "touches_epc_tract",
        "served_tract_count",
    ]
    route_delta = route_delta.merge(domain.route_metadata.loc[:, metadata_columns], on="route_id", how="left")
    for key, value in _context_columns(context).items():
        route_delta[key] = value
    return route_delta


def flatten_tract_deltas(result: SearchResult, context: RunContext) -> pd.DataFrame:
    tract_delta = result.tract_equity_delta_table.copy()
    for key, value in _context_columns(context).items():
        tract_delta[key] = value
    return tract_delta


def summarize_run(result: SearchResult, context: RunContext, route_delta: pd.DataFrame) -> dict[str, Any]:
    obj = result.best_objective_breakdown
    event_counts = pd.Series([event["event_case"] for event in result.events]).value_counts()
    route_changes = route_delta.loc[route_delta["delta_fleet"] != 0].copy()
    ratio_cost_emissions, ratio_cost_emissions_status = safe_ratio(
        result.best_cost_breakdown.annual_total_cost - result.initial_cost_breakdown.annual_total_cost,
        result.best_emissions_breakdown.absolute_delta_emissions_grams,
    )
    ratio_cost_equity, ratio_cost_equity_status = safe_ratio(
        result.best_cost_breakdown.annual_total_cost - result.initial_cost_breakdown.annual_total_cost,
        result.best_equity_breakdown.absolute_population_gap_delta,
    )
    ratio_emissions_equity, ratio_emissions_equity_status = safe_ratio(
        result.best_emissions_breakdown.absolute_delta_emissions_grams,
        result.best_equity_breakdown.absolute_population_gap_delta,
    )
    summary = {
        **_context_columns(context),
        "initial_objective": float(result.initial_objective),
        "best_objective": float(result.best_objective),
        "objective_improvement": float(result.initial_objective - result.best_objective),
        "cost_weighted_contribution": float(obj.cost.weighted_contribution),
        "emissions_weighted_contribution": float(obj.emissions.weighted_contribution),
        "equity_weighted_contribution": float(obj.equity.weighted_contribution),
        "cost_coefficient": float(obj.cost.coefficient),
        "emissions_coefficient": float(obj.emissions.coefficient),
        "equity_coefficient": float(obj.equity.coefficient),
        "annual_cost_delta": float(result.best_cost_breakdown.annual_total_cost - result.initial_cost_breakdown.annual_total_cost),
        "annual_net_emissions_delta_grams": float(result.best_emissions_breakdown.absolute_delta_emissions_grams),
        "equity_population_gap_delta": float(result.best_equity_breakdown.absolute_population_gap_delta),
        "cost_percent_delta": float(obj.cost.percent_delta),
        "emissions_percent_delta": float(obj.emissions.percent_delta),
        "equity_percent_delta": float(obj.equity.percent_delta),
        "accepted_improving_moves": int(result.accepted_improving_moves),
        "accepted_nonimproving_moves": int(result.accepted_nonimproving_moves),
        "event_count": int(len(result.events)),
        "improvement_event_count": int(event_counts.get("improvement", 0)),
        "nonimp_accept_event_count": int(event_counts.get("nonimp_accept", 0)),
        "nonimp_reject_event_count": int(event_counts.get("nonimp_reject", 0)),
        "route_count_changed": int(len(route_changes)),
        "epc_touching_route_count_changed": int(
            route_changes.loc[route_changes["touches_epc_tract"].fillna(False)].route_id.nunique()
        ),
        "route_delta_rows": int(len(route_delta)),
        "tract_delta_rows": int(len(result.tract_equity_delta_table)),
        "cost_delta_per_gram_net_emissions_change": ratio_cost_emissions,
        "cost_delta_per_gram_net_emissions_change_status": ratio_cost_emissions_status,
        "cost_delta_per_equity_gap_change": ratio_cost_equity,
        "cost_delta_per_equity_gap_change_status": ratio_cost_equity_status,
        "net_emissions_delta_per_equity_gap_change": ratio_emissions_equity,
        "net_emissions_delta_per_equity_gap_change_status": ratio_emissions_equity_status,
    }
    summary.update(classify_sacrifice(pd.Series(summary)))
    return summary


def enrich_strategy_tradeoffs(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    out["run_rank_by_objective"] = out["best_objective"].rank(method="dense", ascending=True).astype(int)
    out["run_rank_by_cost"] = out["annual_cost_delta"].rank(method="dense", ascending=True).astype(int)
    out["run_rank_by_emissions"] = out["annual_net_emissions_delta_grams"].rank(method="dense", ascending=True).astype(int)
    out["run_rank_by_equity"] = out["equity_population_gap_delta"].rank(method="dense", ascending=True).astype(int)
    return out.sort_values(["best_objective", "scenario_id", "iterations"]).reset_index(drop=True)


def _sort_frame(frame: pd.DataFrame, sort_columns: Sequence[str]) -> pd.DataFrame:
    present = [column for column in sort_columns if column in frame.columns]
    if not present or frame.empty:
        return frame
    return frame.sort_values(present).reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path, sort_columns: Sequence[str]) -> None:
    _sort_frame(frame, sort_columns).to_csv(path, index=False)


def load_artifacts(artifacts_dir: str | Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    artifact_root = Path(artifacts_dir)
    manifest = json.loads((artifact_root / "run_manifest.json").read_text(encoding="utf-8"))
    frames = {
        "scenario_summary": pd.read_csv(artifact_root / "scenario_summary.csv"),
        "strategy_tradeoffs": pd.read_csv(artifact_root / "strategy_tradeoffs.csv"),
        "iteration_events": pd.read_csv(artifact_root / "iteration_events.csv"),
        "objective_history": pd.read_csv(artifact_root / "objective_history.csv"),
        "route_deltas": pd.read_csv(artifact_root / "route_deltas.csv", dtype={"route_id": str}),
        "tract_deltas": pd.read_csv(artifact_root / "tract_deltas.csv", dtype={"geoid": str}),
    }
    for frame_name in ("scenario_summary", "strategy_tradeoffs"):
        for column in (
            "sacrifice_cost_for_emissions_equity",
            "sacrifice_emissions_for_cost_equity",
            "sacrifice_equity_for_cost_emissions",
            "is_sacrifice_case",
        ):
            if column in frames[frame_name].columns:
                frames[frame_name][column] = frames[frame_name][column].astype(str).str.lower().map(
                    {"true": True, "false": False}
                ).fillna(False)
    return manifest, frames


def generate_analysis_artifacts(
    *,
    domain: RouteFleetDomain,
    base_parameters: CostParameters,
    output_dir: str | Path,
    scenarios: Sequence[WeightScenario],
    base_config: SearchConfig,
    seed: int,
    routes_csv: str | Path,
    route_stops_csv: str | Path,
    parameters_json: str | Path,
    scenarios_json: str | Path,
    iteration_ladder: Sequence[int] = DEFAULT_ITERATION_LADDER,
) -> dict[str, Path]:
    normalized_ladder = tuple(int(value) for value in iteration_ladder)
    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(scenarios) * len(normalized_ladder)
    run_index = 0

    scenario_summary_rows: list[dict[str, Any]] = []
    iteration_event_rows: list[dict[str, Any]] = []
    objective_history_rows: list[dict[str, Any]] = []
    route_frames: list[pd.DataFrame] = []
    tract_frames: list[pd.DataFrame] = []
    manifest_runs: list[dict[str, Any]] = []

    for scenario in scenarios:
        scenario_parameters = apply_weight_scenario(base_parameters, scenario)
        for iterations in normalized_ladder:
            run_index += 1
            config = replace(base_config, max_iterations=int(iterations))
            context = _run_context(scenario, seed=seed, iterations=int(iterations))
            log_progress(
                "Running "
                f"{run_index}/{total_runs}: scenario={scenario.scenario_id} "
                f"label={scenario.run_label!r} iterations={iterations} seed={seed}"
            )
            result = run_route_fleet_search(
                domain=domain,
                config=config,
                seed=seed,
                parameters=scenario_parameters,
            )
            route_delta = flatten_route_deltas(result, context, domain)
            tract_delta = flatten_tract_deltas(result, context)
            scenario_summary_rows.append(summarize_run(result, context, route_delta))
            iteration_event_rows.extend(flatten_iteration_events(result, context))
            objective_history_rows.extend(flatten_objective_history(result, context))
            route_frames.append(route_delta)
            tract_frames.append(tract_delta)
            manifest_runs.append(
                {
                    **_context_columns(context),
                    "initial_objective": float(result.initial_objective),
                    "best_objective": float(result.best_objective),
                    "route_delta_rows": int(len(route_delta)),
                    "tract_delta_rows": int(len(tract_delta)),
                    "event_rows": int(len(result.events)),
                }
            )

    scenario_summary = pd.DataFrame(scenario_summary_rows)
    strategy_tradeoffs = enrich_strategy_tradeoffs(scenario_summary)
    iteration_events = pd.DataFrame(iteration_event_rows)
    objective_history = pd.DataFrame(objective_history_rows)
    route_deltas = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    tract_deltas = pd.concat(tract_frames, ignore_index=True) if tract_frames else pd.DataFrame()

    paths = {
        "scenario_summary": output_path / "scenario_summary.csv",
        "strategy_tradeoffs": output_path / "strategy_tradeoffs.csv",
        "iteration_events": output_path / "iteration_events.csv",
        "objective_history": output_path / "objective_history.csv",
        "route_deltas": output_path / "route_deltas.csv",
        "tract_deltas": output_path / "tract_deltas.csv",
        "run_manifest": output_path / "run_manifest.json",
        "figures_dir": figures_dir,
    }

    common_sort = ("scenario_id", "iterations", "seed")
    _write_csv(scenario_summary, paths["scenario_summary"], common_sort)
    _write_csv(strategy_tradeoffs, paths["strategy_tradeoffs"], ("run_rank_by_objective", *common_sort))
    _write_csv(iteration_events, paths["iteration_events"], (*common_sort, "iteration"))
    _write_csv(objective_history, paths["objective_history"], (*common_sort, "iteration"))
    _write_csv(route_deltas, paths["route_deltas"], (*common_sort, "route_id"))
    _write_csv(tract_deltas, paths["tract_deltas"], (*common_sort, "geoid"))

    best_run_id = None
    if not strategy_tradeoffs.empty:
        best_run_id = str(strategy_tradeoffs.iloc[0]["run_id"])

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_dir": str(output_path.resolve()),
        "figures_dir": str(figures_dir.resolve()),
        "seed": int(seed),
        "iteration_ladder": [int(value) for value in normalized_ladder],
        "default_iteration_ladder": [int(value) for value in DEFAULT_ITERATION_LADDER],
        "full_iteration_ladder": [int(value) for value in FULL_ITERATION_LADDER],
        "scenario_count": len(scenarios),
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "selected_best_run_id": best_run_id,
        "source_files": {
            "routes_csv": str(Path(routes_csv).resolve()),
            "route_stops_csv": str(Path(route_stops_csv).resolve()),
            "parameters_json": str(Path(parameters_json).resolve()),
            "scenarios_json": str(Path(scenarios_json).resolve()),
        },
        "artifact_files": {name: str(path.resolve()) for name, path in paths.items() if name != "figures_dir"},
        "runs": manifest_runs,
    }
    paths["run_manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def _figure_path(figures_dir: Path, filename: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / filename


def _set_sparse_iteration_ticks(ax: plt.Axes, iteration_values: Sequence[int]) -> None:
    ticks = sorted(int(value) for value in set(iteration_values))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])


def plot_objective_vs_iteration(strategy_tradeoffs: pd.DataFrame, figures_dir: Path) -> Path:
    path = _figure_path(figures_dir, "objective_vs_iteration.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    if strategy_tradeoffs.empty:
        ax.text(0.5, 0.5, "No strategies available", ha="center", va="center")
    else:
        for scenario_id, group in strategy_tradeoffs.groupby("scenario_id", sort=True):
            ordered = group.sort_values("iterations")
            ax.plot(
                ordered["iterations"],
                ordered["best_objective"],
                marker="o",
                linewidth=2.2,
                label=str(ordered.iloc[0]["run_label"]),
            )
        best_row = strategy_tradeoffs.sort_values("best_objective").iloc[0]
        ax.scatter([best_row["iterations"]], [best_row["best_objective"]], color="black", zorder=5)
        ax.annotate(
            f"Best: {best_row['run_label']} @ {int(best_row['iterations'])}",
            (best_row["iterations"], best_row["best_objective"]),
            xytext=(8, -18),
            textcoords="offset points",
        )
        _set_sparse_iteration_ticks(ax, strategy_tradeoffs["iterations"].tolist())
        ax.legend(frameon=False)
    ax.set_title("Combined Objective vs Iteration Budget")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Best Combined Objective")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pillar_outcomes(strategy_tradeoffs: pd.DataFrame, figures_dir: Path) -> Path:
    path = _figure_path(figures_dir, "pillar_outcomes_by_iteration.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
    panels = (
        ("annual_cost_delta", "Annual Cost Delta", PILLAR_COLORS["cost"]),
        ("annual_net_emissions_delta_grams", "Net Emissions Delta (g)", PILLAR_COLORS["emissions"]),
        ("equity_population_gap_delta", "Equity Gap Delta", PILLAR_COLORS["equity"]),
    )
    if strategy_tradeoffs.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No strategies available", ha="center", va="center")
            ax.axis("off")
    else:
        for ax, (column, title, color) in zip(axes, panels):
            for scenario_id, group in strategy_tradeoffs.groupby("scenario_id", sort=True):
                ordered = group.sort_values("iterations")
                ax.plot(
                    ordered["iterations"],
                    ordered[column],
                    marker="o",
                    linewidth=2.0,
                    color=color,
                    alpha=0.85,
                    label=str(ordered.iloc[0]["run_label"]),
                )
            ax.axhline(0.0, color="#334155", linewidth=0.9, alpha=0.5)
            ax.set_title(title)
            ax.grid(alpha=0.22)
            _set_sparse_iteration_ticks(ax, strategy_tradeoffs["iterations"].tolist())
        axes[0].legend(frameon=False, loc="best")
    for ax in axes:
        ax.set_xlabel("Iterations")
    fig.suptitle("Per-Pillar Outcomes by Iteration Budget", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_strategy_tradeoffs(strategy_tradeoffs: pd.DataFrame, figures_dir: Path) -> Path:
    path = _figure_path(figures_dir, "strategy_tradeoffs.png")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    if strategy_tradeoffs.empty:
        ax.text(0.5, 0.5, "No strategies available", ha="center", va="center")
    else:
        sizes = strategy_tradeoffs["iterations"].astype(float).map(lambda value: 40.0 + value * 2.0)
        scatter = ax.scatter(
            strategy_tradeoffs["annual_cost_delta"],
            strategy_tradeoffs["annual_net_emissions_delta_grams"],
            c=strategy_tradeoffs["equity_population_gap_delta"],
            s=sizes,
            cmap="coolwarm_r",
            alpha=0.85,
            edgecolors="black",
            linewidths=strategy_tradeoffs["is_sacrifice_case"].astype(int) * 1.2,
        )
        for row in strategy_tradeoffs.loc[strategy_tradeoffs["is_sacrifice_case"]].itertuples(index=False):
            ax.annotate(str(row.run_label), (row.annual_cost_delta, row.annual_net_emissions_delta_grams), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.axhline(0.0, color="#334155", linewidth=0.9, alpha=0.5)
        ax.axvline(0.0, color="#334155", linewidth=0.9, alpha=0.5)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Equity Gap Delta")
    ax.set_title("Strategy Tradeoffs Across Sampled Runs")
    ax.set_xlabel("Annual Cost Delta")
    ax.set_ylabel("Annual Net Emissions Delta (g)")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sacrifice_cases(strategy_tradeoffs: pd.DataFrame, figures_dir: Path) -> Path:
    path = _figure_path(figures_dir, "sacrifice_cases.png")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sacrifice = strategy_tradeoffs.loc[strategy_tradeoffs["is_sacrifice_case"]].copy()
    if sacrifice.empty:
        ax.text(0.5, 0.5, "No sacrifice cases found in sampled strategies", ha="center", va="center")
        ax.axis("off")
    else:
        ordered = sacrifice.sort_values("best_objective").head(8)
        x_positions = range(len(ordered))
        width = 0.24
        ax.bar([x - width for x in x_positions], ordered["cost_percent_delta"], width=width, color=PILLAR_COLORS["cost"], label="Cost")
        ax.bar(list(x_positions), ordered["emissions_percent_delta"], width=width, color=PILLAR_COLORS["emissions"], label="Emissions")
        ax.bar([x + width for x in x_positions], ordered["equity_percent_delta"], width=width, color=PILLAR_COLORS["equity"], label="Equity")
        ax.axhline(0.0, color="#334155", linewidth=0.9)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([f"{row.scenario_id}\n{int(row.iterations)} iters" for row in ordered.itertuples(index=False)], fontsize=8)
        ax.legend(frameon=False)
        ax.set_ylabel("Percent Delta")
    ax.set_title("Pillar Sacrifice Cases")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_cost_benefit(strategy_tradeoffs: pd.DataFrame, figures_dir: Path) -> Path:
    path = _figure_path(figures_dir, "cost_benefit_summary.png")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    if strategy_tradeoffs.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No strategies available", ha="center", va="center")
            ax.axis("off")
    else:
        axes[0].scatter(
            strategy_tradeoffs["annual_cost_delta"],
            strategy_tradeoffs["annual_net_emissions_delta_grams"],
            c=strategy_tradeoffs["is_sacrifice_case"].map({True: "#be123c", False: "#334155"}),
            s=70,
            alpha=0.85,
        )
        axes[0].set_xlabel("Annual Cost Delta")
        axes[0].set_ylabel("Annual Net Emissions Delta (g)")
        axes[0].set_title("Cost vs Emissions")
        axes[1].scatter(
            strategy_tradeoffs["annual_cost_delta"],
            strategy_tradeoffs["equity_population_gap_delta"],
            c=strategy_tradeoffs["is_sacrifice_case"].map({True: "#be123c", False: "#334155"}),
            s=70,
            alpha=0.85,
        )
        axes[1].set_xlabel("Annual Cost Delta")
        axes[1].set_ylabel("Equity Gap Delta")
        axes[1].set_title("Cost vs Equity")
        for ax in axes:
            ax.axhline(0.0, color="#334155", linewidth=0.9, alpha=0.5)
            ax.axvline(0.0, color="#334155", linewidth=0.9, alpha=0.5)
            ax.grid(alpha=0.22)
    fig.suptitle("Cost-Benefit Summary Across Sampled Strategies", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#0f172a", linewidth=1.2))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=10, color="white", wrap=True)


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.5, color="#334155")
    ax.add_patch(arrow)


def plot_research_workflow(figures_dir: Path) -> Path:
    path = _figure_path(figures_dir, "research_workflow.png")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [
        (0.4, "Data\nIngestion", "#0f766e"),
        (2.2, "Route / Domain\nPreparation", "#2563eb"),
        (4.2, "Weight + Search\nSetup", "#ea580c"),
        (6.1, "Optimization\nSweep", "#7c3aed"),
        (8.0, "Artifact\nGeneration", "#be123c"),
        (9.8, "Comparative\nAnalysis", "#0369a1"),
    ]
    for index, (x, label, color) in enumerate(steps):
        _draw_box(ax, x, 1.0, 1.35, 0.9, label, color)
        if index < len(steps) - 1:
            _draw_arrow(ax, (x + 1.35, 1.45), (steps[index + 1][0], 1.45))
    _draw_box(ax, 10.9, 1.0, 0.8, 0.9, "PNG\nExport", "#1d4ed8")
    _draw_arrow(ax, (9.8 + 1.35, 1.45), (10.9, 1.45))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_model_components(figures_dir: Path) -> Path:
    path = _figure_path(figures_dir, "model_components.png")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _draw_box(ax, 0.6, 4.3, 2.1, 0.9, "Route Data\n+ Stop Geometry", "#0f766e")
    _draw_box(ax, 0.6, 2.8, 2.1, 0.9, "Ridership\nAssumptions", "#2563eb")
    _draw_box(ax, 0.6, 1.3, 2.1, 0.9, "Equity\nGeography", "#ea580c")
    _draw_box(ax, 3.5, 3.0, 2.0, 1.2, "Objective\nWeights", "#475569")
    _draw_box(ax, 6.1, 3.0, 2.0, 1.2, "TS / SA\nSearch", "#7c3aed")
    _draw_box(ax, 8.9, 4.3, 2.0, 0.9, "Route Deltas", "#0369a1")
    _draw_box(ax, 8.9, 2.8, 2.0, 0.9, "Tract Deltas", "#be123c")
    _draw_box(ax, 8.9, 1.3, 2.0, 0.9, "Figures / Maps", "#1d4ed8")
    pillar_y = 0.25
    _draw_box(ax, 3.3, pillar_y, 1.6, 0.7, "Cost", PILLAR_COLORS["cost"])
    _draw_box(ax, 5.1, pillar_y, 1.8, 0.7, "Emissions", PILLAR_COLORS["emissions"])
    _draw_box(ax, 7.2, pillar_y, 1.4, 0.7, "Equity", PILLAR_COLORS["equity"])
    arrows = [
        ((2.7, 4.75), (3.5, 3.95)),
        ((2.7, 3.25), (3.5, 3.55)),
        ((2.7, 1.75), (3.5, 3.15)),
        ((5.5, 3.6), (6.1, 3.6)),
        ((8.1, 3.9), (8.9, 4.75)),
        ((8.1, 3.6), (8.9, 3.25)),
        ((8.1, 3.3), (8.9, 1.75)),
    ]
    for start, end in arrows:
        _draw_arrow(ax, start, end)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_route_geometries(route_stops_csv: str | Path) -> gpd.GeoDataFrame:
    route_stops = pd.read_csv(route_stops_csv, dtype={"route_id": str, "direction_id": str})
    required = {"route_id", "direction_id", "route_stop_order", "stop_lat", "stop_lon"}
    missing = sorted(required.difference(route_stops.columns))
    if missing:
        raise ValueError(f"Route stops CSV {route_stops_csv} is missing required columns: {missing}")
    route_stops["route_stop_order"] = pd.to_numeric(route_stops["route_stop_order"], errors="coerce")
    route_stops["stop_lat"] = pd.to_numeric(route_stops["stop_lat"], errors="coerce")
    route_stops["stop_lon"] = pd.to_numeric(route_stops["stop_lon"], errors="coerce")
    route_stops = route_stops.dropna(subset=["route_stop_order", "stop_lat", "stop_lon"])
    route_stops = route_stops.sort_values(["route_id", "direction_id", "route_stop_order"])

    rows: list[dict[str, Any]] = []
    for (route_id, direction_id), group in route_stops.groupby(["route_id", "direction_id"], sort=True):
        coords = list(dict.fromkeys((float(row.stop_lon), float(row.stop_lat)) for row in group.itertuples(index=False)))
        if len(coords) < 2:
            continue
        rows.append(
            {
                "route_id": str(route_id),
                "direction_id": str(direction_id),
                "geometry": LineString(coords),
            }
        )

    if not rows:
        return gpd.GeoDataFrame(columns=["route_id", "direction_id", "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _plot_route_map(route_geometries: gpd.GeoDataFrame, value_table: pd.DataFrame, column: str, title: str, output_path: Path, *, diverging: bool = False) -> Path:
    merged = route_geometries.merge(value_table.loc[:, ["route_id", column]].drop_duplicates(subset=["route_id"]), on="route_id", how="left")
    fig, ax = plt.subplots(figsize=(8.5, 10))
    if merged.empty:
        ax.text(0.5, 0.5, "No route geometry available", ha="center", va="center")
        ax.axis("off")
    else:
        merged["plot_value"] = pd.to_numeric(merged[column], errors="coerce")
        missing = merged["plot_value"].isna()
        if missing.any():
            merged.loc[missing].plot(ax=ax, color="#cbd5e1", linewidth=1.2, alpha=0.5)
        present = merged.loc[~missing].copy()
        if not present.empty:
            max_abs = float(present["plot_value"].abs().max())
            if diverging:
                scale = max(max_abs, 1.0)
                norm: Normalize = TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale)
                cmap = plt.get_cmap("coolwarm")
            else:
                vmax = float(max(present["plot_value"].max(), 1.0))
                norm = Normalize(vmin=0.0, vmax=vmax)
                cmap = plt.get_cmap("YlGnBu")
            linewidths = 1.2 + (present["plot_value"].abs() / max(max_abs, 1.0)) * 2.8
            present.plot(
                ax=ax,
                column="plot_value",
                cmap=cmap,
                linewidth=linewidths,
                legend=True,
                legend_kwds={"shrink": 0.6, "label": column.replace("_", " ").title()},
                norm=norm,
            )
        bounds = merged.total_bounds
        if len(bounds) == 4 and not any(math.isnan(value) for value in bounds):
            dx = (bounds[2] - bounds[0]) * 0.05
            dy = (bounds[3] - bounds[1]) * 0.05
            ax.set_xlim(bounds[0] - dx, bounds[2] + dx)
            ax.set_ylim(bounds[1] - dy, bounds[3] + dy)
        ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_route_maps(strategy_tradeoffs: pd.DataFrame, route_deltas: pd.DataFrame, figures_dir: Path, route_stops_csv: str | Path) -> list[Path]:
    if strategy_tradeoffs.empty:
        outputs = []
        for filename, title in (
            ("baseline_route_map.png", "Baseline Route Map"),
            ("optimized_route_map.png", "Optimized Route Map"),
            ("route_delta_map.png", "Route Delta Map"),
        ):
            fig, ax = plt.subplots(figsize=(8.5, 10))
            ax.text(0.5, 0.5, "No strategies available", ha="center", va="center")
            ax.axis("off")
            out = _figure_path(figures_dir, filename)
            fig.savefig(out, dpi=180)
            plt.close(fig)
            outputs.append(out)
        return outputs

    best_run = strategy_tradeoffs.sort_values("best_objective").iloc[0]
    run_id = str(best_run["run_id"])
    selected_routes = route_deltas.loc[route_deltas["run_id"] == run_id].copy()
    route_geometries = build_route_geometries(route_stops_csv)
    baseline_path = _plot_route_map(
        route_geometries,
        selected_routes.rename(columns={"baseline_fleet": "baseline_buses"}),
        "baseline_buses",
        f"Baseline Buses by Route ({best_run['run_label']}, {int(best_run['iterations'])} iters)",
        _figure_path(figures_dir, "baseline_route_map.png"),
    )
    optimized_path = _plot_route_map(
        route_geometries,
        selected_routes.rename(columns={"optimized_fleet": "optimized_buses"}),
        "optimized_buses",
        f"Optimized Buses by Route ({best_run['run_label']}, {int(best_run['iterations'])} iters)",
        _figure_path(figures_dir, "optimized_route_map.png"),
    )
    delta_path = _plot_route_map(
        route_geometries,
        selected_routes.rename(columns={"delta_fleet": "fleet_delta"}),
        "fleet_delta",
        f"Fleet Delta by Route ({best_run['run_label']}, {int(best_run['iterations'])} iters)",
        _figure_path(figures_dir, "route_delta_map.png"),
        diverging=True,
    )
    return [baseline_path, optimized_path, delta_path]


def generate_figures_from_artifacts(
    artifacts_dir: str | Path,
    *,
    route_stops_csv: str | Path | None = None,
) -> list[Path]:
    manifest, frames = load_artifacts(artifacts_dir)
    artifact_root = Path(artifacts_dir)
    figures_dir = artifact_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    route_stops_source = route_stops_csv or manifest["source_files"]["route_stops_csv"]

    outputs = [
        plot_objective_vs_iteration(frames["strategy_tradeoffs"], figures_dir),
        plot_pillar_outcomes(frames["strategy_tradeoffs"], figures_dir),
        plot_strategy_tradeoffs(frames["strategy_tradeoffs"], figures_dir),
        plot_sacrifice_cases(frames["strategy_tradeoffs"], figures_dir),
        plot_cost_benefit(frames["strategy_tradeoffs"], figures_dir),
        plot_research_workflow(figures_dir),
        plot_model_components(figures_dir),
    ]
    outputs.extend(generate_route_maps(frames["strategy_tradeoffs"], frames["route_deltas"], figures_dir, route_stops_source))
    return outputs


def run_cli(args: argparse.Namespace) -> dict[str, Path]:
    if args.from_artifacts_dir:
        log_progress(f"Regenerating figures from artifacts in {args.from_artifacts_dir}")
        generate_figures_from_artifacts(args.from_artifacts_dir, route_stops_csv=args.route_stops_csv)
        return {"artifacts_dir": Path(args.from_artifacts_dir)}

    iteration_ladder = parse_iteration_ladder(args.iteration_ladder)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    log_progress(f"Output directory: {output_dir}")
    log_progress("Loading route fleet domain")
    domain = load_route_fleet_domain(args.routes_csv, route_stops_path=args.route_stops_csv)
    log_progress("Loading model parameters")
    parameters = load_parameters(args.parameters_json)
    log_progress("Loading weight scenarios")
    scenarios = load_weight_scenarios(args.scenarios_json, scenario_ids=args.scenario_ids)
    log_progress(f"Starting sweep with {len(scenarios)} scenario(s) over iteration ladder {list(iteration_ladder)}")
    paths = generate_analysis_artifacts(
        domain=domain,
        base_parameters=parameters,
        output_dir=output_dir,
        scenarios=scenarios,
        base_config=build_search_config(args),
        seed=args.seed,
        routes_csv=args.routes_csv,
        route_stops_csv=args.route_stops_csv,
        parameters_json=args.parameters_json,
        scenarios_json=args.scenarios_json,
        iteration_ladder=iteration_ladder,
    )
    if not args.skip_figures:
        log_progress("Generating figures from saved artifacts")
        generate_figures_from_artifacts(output_dir, route_stops_csv=args.route_stops_csv)
    return paths


def main() -> None:
    args = parse_args()
    paths = run_cli(args)
    if "run_manifest" in paths:
        print(f"Artifacts written to {paths['run_manifest'].parent}")
        print(f"Manifest: {paths['run_manifest']}")
        print(f"Figures directory: {paths['figures_dir']}")
    else:
        print(f"Figures regenerated from {paths['artifacts_dir']}")


if __name__ == "__main__":
    main()
