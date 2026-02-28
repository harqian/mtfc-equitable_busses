# Basic TS/SA Route Fleet Search Implementation Plan

## Overview

Implement a basic tabu search + simulated annealing (TS/SA) optimizer for route fleet allocation in this repository, with deterministic behavior controls and a comprehensive test suite in `test_model_search.py` including both unit and integration coverage.

## Current State Analysis

The repository currently has no optimization implementation:
- `objective_function(y)` is a stub returning `0` in `model.py`.
- `model.py` is currently focused on rendering route/intersection maps from static CSVs.
- There is no existing TS/SA search loop, tabu memory, acceptance logic, or optimization-facing tests.

Relevant route-level data already exists:
- `data/simplified_bus_routes.csv` includes route-level service features including `weekday_typical_buses`.
- For this basic v1, route bounds are derived directly from baseline fleets as `baseline ± 10` (with lower bound clamped at `0`) to avoid route-key normalization complexity across multiple files.

Reference behavior to preserve from the C++ solver (`~/github_projects/social-transit-solver`):
- Improvement acceptance and SA non-improvement acceptance.
- Tabu memory on inverse move types.
- Non-improvement counters for diversification/intensification.
- Candidate neighborhood including ADD/DROP/SWAP.

## Desired End State

After this plan is complete:
- A callable TS/SA route fleet optimizer exists in this repo.
- The optimizer works on route fleet vectors only (no stop relocation decisions).
- The search supports ADD/DROP/SWAP neighborhood moves, tabu memory decay, SA acceptance, and attractive-solution jumps.
- The optimizer has deterministic testability with fixed random seeds.
- `test_model_search.py` includes unit tests for algorithm mechanics and integration tests for end-to-end search behavior.

Verification of end state:
- Automated tests pass for core mechanics and integration scenarios.
- A manual run of the optimizer over local route CSVs completes and returns structured outputs with improved or equal best objective versus initial state.

### Key Discoveries
- Objective stub: [`model.py:9`](/Users/hq/github_projects/mtfc-equitable_busses/model.py:9)
- Current `model.py` is map-only flow: [`model.py:17`](/Users/hq/github_projects/mtfc-equitable_busses/model.py:17)
- Route service metrics already assembled in simplified routes build script: [`build_simplified_bus_routes_and_stops.py:252`](/Users/hq/github_projects/mtfc-equitable_busses/build_simplified_bus_routes_and_stops.py:252)
- TS/SA control flow patterns to mirror: [`search.cpp:185`](/Users/hq/github_projects/social-transit-solver/search.cpp:185), [`search.cpp:274`](/Users/hq/github_projects/social-transit-solver/search.cpp:274), [`search.cpp:361`](/Users/hq/github_projects/social-transit-solver/search.cpp:361), [`search.hpp:94`](/Users/hq/github_projects/social-transit-solver/search.hpp:94)

## What We're NOT Doing

- No stop location optimization or geometry edits.
- No GTFS regeneration or route topology rebuild.
- No attempt to replicate the full C++ solver’s expensive assignment/constraint subsystem.
- No long-running exhaustive local search stage.
- No CLI redesign or large visualization rework beyond minimal compatibility with current `model.py`.
- No use of `peak_vehicles_by_route.csv` in v1.

## Implementation Approach

Implement a pragmatic, testable TS/SA engine for route fleet vectors using local CSV data and a simplified objective. Keep components small and explicit:
- Data prep/normalization
- Objective and feasibility helpers
- Neighborhood generation
- Search loop and memory structures
- Result packaging

Prioritize deterministic behavior and correctness in mechanics over sophisticated transit modeling.

## Phase 1: Route Fleet Domain + Objective/Constraint Primitives

### Overview
Create the route fleet optimization domain model and all deterministic primitives required by the search loop.

### Changes Required:

#### 1. Route Fleet Data Preparation
**File**: `model.py`  
**Changes**:
- Add route-level loader for `simplified_bus_routes.csv` (weekday baseline fleet vector).
- Define stable route index ordering for vectorized operations.
- Create baseline vector, lower/upper bounds, and route metadata table.
- Set route bounds as:
  - `lower_i = max(0, baseline_i - 10)`
  - `upper_i = baseline_i + 10`

#### 2. Objective Function Definition
**File**: `model.py`  
**Changes**:
- Replace stub `objective_function(y)` with a real objective over route fleet vectors.
- Use a weighted objective with:
  - Deviation penalty from baseline fleet (`L1` or weighted `L2`).
  - Service proxy term using available route features (for example, trips or stop_count weighting).
- Ensure deterministic numeric behavior and input validation.

#### 3. Feasibility and Move Application Helpers
**File**: `model.py`  
**Changes**:
- Add feasibility checks:
  - Per-route bounds.
  - Global fleet conservation (`sum(y) == sum(y_baseline)`) for SWAP-only mass balance.
- Add move operators:
  - `ADD(route_i)`, `DROP(route_i)`, `SWAP(add_i, drop_j)`.
- Add canonical move representation suitable for tabu memory indexing.

### Success Criteria:

#### Automated Verification:
- [x] `python -m unittest -v test_model_search.py` passes primitive-focused tests.
- [x] Objective returns finite float for valid vectors and rejects invalid vectors cleanly.
- [x] Move helpers preserve invariants for all tested cases.

#### Manual Verification:
- [x] Loading route data yields expected vector length and stable route ordering.
- [x] Baseline vector is feasible under generated bounds.

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding to the next phase.

---

## Phase 2: Basic TS/SA Engine for Route Fleet Vectors

### Overview
Implement the iterative TS/SA search loop with simplified but faithful mechanics from the reference solver.

### Changes Required:

#### 1. Search State + Parameters
**File**: `model.py`  
**Changes**:
- Define a search config structure with:
  - `max_iterations`
  - `temp_init`, `temp_factor`
  - `tenure_init`, `tenure_factor`
  - `nonimp_in_max`, `nonimp_out_max`
  - neighborhood caps (`nbhd_add_lim`, `nbhd_drop_lim`, `nbhd_swap_lim`)
  - `attractive_max`
- Define search memory:
  - `add_tenure[]`, `drop_tenure[]`
  - `sol_current`, `sol_best`, `obj_current`, `obj_best`
  - `nonimp_in`, `nonimp_out`, `temperature`, `tenure`
  - `attractive_solutions`

#### 2. Neighborhood Search (Simplified Two-Pass)
**File**: `model.py`  
**Changes**:
- Pass 1:
  - Randomly sample feasible ADD/DROP candidates.
  - Score candidates by objective.
  - Filter tabu moves unless aspiration (`candidate_obj < obj_best`).
- Pass 2:
  - Keep best limited ADD/DROP candidates.
  - Build SWAP candidates from selected ADD/DROP pairs.
  - Return best and second-best feasible neighbors.

#### 3. Main Iteration Logic
**File**: `model.py`  
**Changes**:
- Improvement case:
  - Accept best move.
  - Reset tenure to `tenure_init`.
  - Apply inverse tabu on move endpoints.
- Non-improvement case:
  - Compute SA acceptance probability `exp(-(delta)/temperature)`.
  - If accepted: apply move, increase tenure (`* tenure_factor`), store second-best as attractive.
  - If rejected: no move, store best neighbor as attractive.
- End-of-iteration updates:
  - Decay `add_tenure` and `drop_tenure`.
  - Cool temperature (`* temp_factor`).
  - Trim attractive set.
  - Diversify when `nonimp_in` exceeds threshold (jump to random attractive solution).
  - Intensify when `nonimp_out` exceeds threshold (reset tenure).

#### 4. Structured Results
**File**: `model.py`  
**Changes**:
- Return structured outputs:
  - Best vector and objective.
  - Iteration/event log records suitable for assertions.
  - Route metadata joined with final fleet vector for downstream inspection.

### Success Criteria:

#### Automated Verification:
- [x] `python -m unittest -v test_model_search.py` passes search-loop behavior tests.
- [x] With fixed seed, repeated runs are deterministic.
- [x] Search never produces infeasible vectors under tested move sequences.

#### Manual Verification:
- [ ] A short run (`~50-100` iterations) completes on local data without exceptions.
- [ ] Event summaries show expected mix of improvement and non-improvement behavior.

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding to the next phase.

---

## Phase 3: Tests in `test_model_search.py` (Unit + Integration)

### Overview
Build a robust test suite that validates both algorithm internals and end-to-end execution.

### Changes Required:

#### 1. Unit Tests for Core Mechanics
**File**: `test_model_search.py`  
**Changes**:
- Add unit tests for:
  - Objective determinism and monotonic expectations on synthetic vectors.
  - Feasibility checks (bounds and global fleet conservation).
  - Move application correctness for ADD/DROP/SWAP.
  - Tabu aspiration behavior (tabu move allowed when improving best known objective).
  - SA acceptance behavior via seeded randomness.
  - Tenure decay and cooling schedule updates.

#### 2. Integration Tests on Synthetic Fixture
**File**: `test_model_search.py`  
**Changes**:
- Add a compact synthetic route dataset in-test (or fixture file) to validate:
  - End-to-end TS/SA run returns a valid result object.
  - `obj_best <= obj_initial`.
  - All intermediate vectors remain feasible.
  - Attractive-set diversification path is reachable under crafted settings.

#### 3. Integration Smoke Test on Real Local CSVs
**File**: `test_model_search.py`  
**Changes**:
- Add one bounded-iteration smoke integration test using local `data/simplified_bus_routes.csv`.
- Keep runtime bounded with small iteration count and deterministic seed.
- Assert:
  - No exceptions.
  - Output shape matches input route count.
  - Best objective is finite and not worse than initial by more than negligible numeric tolerance.

### Success Criteria:

#### Automated Verification:
- [ ] `python -m unittest -v test_model_search.py` passes all tests.
- [ ] Integration tests complete within practical local runtime bounds.

#### Manual Verification:
- [ ] Test output clearly distinguishes unit and integration coverage.
- [ ] Failures provide actionable messages (move type, iteration, feasibility reason).

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding to the next phase.

---

## Phase 4: Model Entrypoint Wiring and Minimal Usage Flow

### Overview
Expose a minimal runnable path in `model.py` for invoking optimization in addition to existing map behavior.

### Changes Required:

#### 1. Execution Path Control
**File**: `model.py`  
**Changes**:
- Add a simple mode switch (for example, optimize vs map render) without breaking current map output path.
- Ensure search invocation can run from repository root using existing data file discovery.

#### 2. Output Summaries
**File**: `model.py`  
**Changes**:
- Print concise optimization summary:
  - Initial objective
  - Best objective
  - Iteration count
  - Number of accepted improving/non-improving moves
- Provide a tabular route-level comparison (baseline vs optimized fleet counts) for quick inspection.

### Success Criteria:

#### Automated Verification:
- [ ] `python -m unittest -v test_model_search.py` remains green after wiring.

#### Manual Verification:
- [ ] Running optimization mode from CLI produces stable output and does not break map mode.
- [ ] Summary output is readable and sufficient for quick sanity checks.

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding to the next phase.

---

## Testing Strategy

### Unit Tests
- Objective validity and determinism.
- Move semantics and feasibility invariants.
- SA acceptance edge cases (`delta=0`, high/low temperature behavior).
- Tabu application and aspiration override.
- Counter/tenure/temperature update correctness.

### Integration Tests
- Full TS/SA execution on synthetic dataset with deterministic seed.
- Real-data smoke run on `simplified_bus_routes.csv` with low iteration budget.
- Regression guard for output dimensions and finite objective behavior.

### Manual Testing Steps
1. Run `python -m unittest -v test_model_search.py`.
2. Run `python model.py` in map mode and confirm existing map output still works.
3. Run optimization mode with fixed seed and verify summary indicates feasible final solution and best objective improvement or parity.

## Performance Considerations

- Keep neighborhood candidate limits modest to prevent combinatorial blow-up.
- Reuse precomputed route metadata and vectorized objective terms.
- Keep integration test iteration counts low to preserve fast feedback loops.

## Migration Notes

- No schema or persistent data migration is required.
- Changes are additive to current workflow.
- Existing map rendering behavior should remain available and unchanged by default.

## References

- Original task context: implement basic TS/SA search and tests for this repository.
- Current stub objective: [`model.py:9`](/Users/hq/github_projects/mtfc-equitable_busses/model.py:9)
- Current map flow: [`model.py:17`](/Users/hq/github_projects/mtfc-equitable_busses/model.py:17)
- Route service aggregation source: [`build_simplified_bus_routes_and_stops.py:252`](/Users/hq/github_projects/mtfc-equitable_busses/build_simplified_bus_routes_and_stops.py:252)
- Reference TS/SA behavior:
  - [`search.cpp:185`](/Users/hq/github_projects/social-transit-solver/search.cpp:185)
  - [`search.cpp:274`](/Users/hq/github_projects/social-transit-solver/search.cpp:274)
  - [`search.cpp:361`](/Users/hq/github_projects/social-transit-solver/search.cpp:361)
  - [`search.hpp:94`](/Users/hq/github_projects/social-transit-solver/search.hpp:94)
