"""V9.3 explicit exact Event-graph verifier.

The solver never receives a symbolic multi-event skeleton.  Python owns the
finite source search.  Every SMT check contains one already-fixed next-event
source appended to the current feasible prefix, so arithmetic feasibility and
kernel semantics remain in Z3 while combinatorial source enumeration does not.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import z3

from .carry_in import derive_protected_priority_prefix, reachable_carry_in_consistency
from .controller_encoder import enumerate_controller_policy_cases
from .event_depth_feasibility import derive_minimum_event_depth
from .event_kernel import (
    DispatchCase,
    EventSource,
    EventStepEncoding,
    build_event_candidates,
    encode_event_node_closure,
    encode_event_relative_edge,
    enumerate_dispatch_cases,
    enumerate_event_sources,
    event_source_time_formula,
    exact_periodic_countdown,
)
from .event_window_encoder import EventGraphProblem, EventWindowEncoding
from .symbolic_state import SymbolicKernelState, declare_state
from .target_projection import target_pending_after_origin
from .solver_runtime import DEFAULT_Z3_THREADS, SOURCE_TIME_WORKERS, make_solver


SOLVER_STRATEGY = "V9_3_PARALLEL_DISPATCH_SPECIALIZED_SHARED_CANDIDATE_EVENT_GRAPH_DFS"


@dataclass(slots=True)
class _DepthStats:
    nodes: int = 0
    closure_checks: int = 0
    closure_feasible: int = 0
    closure_infeasible: int = 0
    dispatch_checks: int = 0
    dispatch_feasible: int = 0
    dispatch_infeasible: int = 0
    edge_checks: int = 0
    source_time_checks: int = 0
    silent_service_checks: int = 0
    destination_checks: int = 0
    feasible_edges: int = 0
    infeasible_edges: int = 0
    terminal_checks: int = 0
    terminal_safe: int = 0
    structural_skips: int = 0
    solver_seconds: float = 0.0
    build_seconds: float = 0.0

    def as_dict(self, depth: int) -> dict[str, Any]:
        return {
            "depth": int(depth),
            "nodes": int(self.nodes),
            "closure_checks": int(self.closure_checks),
            "closure_feasible": int(self.closure_feasible),
            "closure_infeasible": int(self.closure_infeasible),
            "dispatch_checks": int(self.dispatch_checks),
            "dispatch_feasible": int(self.dispatch_feasible),
            "dispatch_infeasible": int(self.dispatch_infeasible),
            "edge_checks": int(self.edge_checks),
            "source_time_checks": int(self.source_time_checks),
            "silent_service_checks": int(self.silent_service_checks),
            "destination_checks": int(self.destination_checks),
            "feasible_edges": int(self.feasible_edges),
            "infeasible_edges": int(self.infeasible_edges),
            "terminal_checks": int(self.terminal_checks),
            "terminal_safe": int(self.terminal_safe),
            "structural_skips": int(self.structural_skips),
            "solver_check_seconds": round(float(self.solver_seconds), 6),
            "edge_build_seconds": round(float(self.build_seconds), 6),
        }


@dataclass(frozen=True, slots=True)
class EventGraphReceipt:
    obligation_id: str
    result: str
    solver_version: str
    strategy: str
    finite_event_bound: int
    minimum_feasible_depth: int
    explored_nodes: int
    closure_checks: int
    closure_feasible: int
    closure_infeasible: int
    dispatch_checks: int
    dispatch_feasible: int
    dispatch_infeasible: int
    edge_checks: int
    source_time_checks: int
    silent_service_checks: int
    destination_checks: int
    feasible_edges: int
    infeasible_edges: int
    terminal_checks: int
    structural_skips: int
    max_reached_depth: int
    solver_check_seconds: float
    edge_build_seconds: float
    depth_stats: tuple[dict[str, Any], ...]
    decisive_depth: int | None = None
    decisive_source_path: tuple[str, ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "obligation_id": self.obligation_id,
            "result": self.result,
            "solver_version": self.solver_version,
            "solver_strategy": self.strategy,
            "finite_event_bound": int(self.finite_event_bound),
            "minimum_feasible_depth": int(self.minimum_feasible_depth),
            "explored_nodes": int(self.explored_nodes),
            "closure_checks": int(self.closure_checks),
            "closure_feasible": int(self.closure_feasible),
            "closure_infeasible": int(self.closure_infeasible),
            "dispatch_checks": int(self.dispatch_checks),
            "dispatch_feasible": int(self.dispatch_feasible),
            "dispatch_infeasible": int(self.dispatch_infeasible),
            "edge_checks": int(self.edge_checks),
            "source_time_checks": int(self.source_time_checks),
            "silent_service_checks": int(self.silent_service_checks),
            "destination_checks": int(self.destination_checks),
            "feasible_edges": int(self.feasible_edges),
            "infeasible_edges": int(self.infeasible_edges),
            "terminal_checks": int(self.terminal_checks),
            "structural_skips": int(self.structural_skips),
            "max_reached_depth": int(self.max_reached_depth),
            "solver_check_seconds": round(float(self.solver_check_seconds), 6),
            "edge_build_seconds": round(float(self.edge_build_seconds), 6),
            "depth_stats": list(self.depth_stats),
            "symbolic_multi_event_skeleton": False,
            "canonical_disjoint_event_source_partition": True,
            "controller_branch_specialized_per_graph_node": True,
            "iterative_depth_restart": False,
            "solver_threads": int(DEFAULT_Z3_THREADS),
            "source_time_workers": int(SOURCE_TIME_WORKERS),
        }
        if self.decisive_depth is not None:
            row["decisive_depth"] = int(self.decisive_depth)
        if self.decisive_source_path:
            row["decisive_source_path"] = list(self.decisive_source_path)
        if self.reason is not None:
            row["reason"] = self.reason
        return row


@dataclass(slots=True)
class _SearchState:
    depth_stats: dict[int, _DepthStats] = field(default_factory=dict)
    explored_nodes: int = 0
    closure_checks: int = 0
    closure_feasible: int = 0
    closure_infeasible: int = 0
    dispatch_checks: int = 0
    dispatch_feasible: int = 0
    dispatch_infeasible: int = 0
    edge_checks: int = 0
    source_time_checks: int = 0
    silent_service_checks: int = 0
    destination_checks: int = 0
    feasible_edges: int = 0
    infeasible_edges: int = 0
    terminal_checks: int = 0
    structural_skips: int = 0
    max_reached_depth: int = 0
    solver_seconds: float = 0.0
    build_seconds: float = 0.0
    edge_serial: int = 0
    unknown_reason: str | None = None

    def at(self, depth: int) -> _DepthStats:
        return self.depth_stats.setdefault(int(depth), _DepthStats())


def _check(solver: z3.Solver) -> tuple[str, float, str | None]:
    started = perf_counter()
    result = solver.check()
    elapsed = perf_counter() - started
    if result == z3.sat:
        return "SAT", elapsed, None
    if result == z3.unsat:
        return "UNSAT", elapsed, None
    return "UNKNOWN", elapsed, solver.reason_unknown()


def _prepare_isolated_solver(
    assertions: tuple[z3.BoolRef, ...],
    formula: z3.BoolRef,
) -> z3.Solver:
    """Clone one exact source-time query into an isolated single-thread context."""

    ctx = z3.Context()
    local = make_solver(ctx=ctx, threads=1)
    for assertion in assertions:
        local.add(assertion.translate(ctx))
    local.add(formula.translate(ctx))
    return local


def _check_prepared_solver(solver: z3.Solver) -> tuple[str, float, str | None]:
    return _check(solver)


def _parallel_source_precheck(
    solver: z3.Solver,
    rows: tuple[tuple[EventSource, z3.BoolRef], ...],
    *,
    workers: int = SOURCE_TIME_WORKERS,
    progress_batch: Callable[[tuple[str, ...]], None] | None = None,
) -> dict[str, tuple[str, float, str | None]]:
    """Exact parallel source-time precheck using independent Z3 contexts.

    The current incremental path assertions are translated once per source into
    an isolated context.  At most ``workers`` checks run concurrently.  No Z3
    context is shared across threads.
    """

    results: dict[str, tuple[str, float, str | None]] = {}
    assertions = tuple(solver.assertions())
    width = max(1, int(workers))
    for offset in range(0, len(rows), width):
        batch = rows[offset: offset + width]
        if progress_batch is not None:
            progress_batch(tuple(source.source_id for source, _ in batch))
        prepared = [
            (source, _prepare_isolated_solver(assertions, formula))
            for source, formula in batch
        ]
        with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
            futures = [
                (source, pool.submit(_check_prepared_solver, local_solver))
                for source, local_solver in prepared
            ]
            for source, future in futures:
                results[source.source_id] = future.result()
    return results


def _search_sources(problem: EventGraphProblem) -> tuple[EventSource, ...]:
    """Static canonical source set; dispatch identity is specialized separately."""

    return enumerate_event_sources(
        problem.model,
        active_task_names=frozenset(problem.projection.active_task_names),
    )


def _progress(
    callback: Callable[[dict[str, Any]], None] | None,
    state: _SearchState,
    *,
    phase: str,
    depth: int,
    source: EventSource | None = None,
    path: tuple[str, ...] = (),
    **extra: Any,
) -> None:
    if callback is None:
        return
    row: dict[str, Any] = {
        "phase": phase,
        "depth": int(depth),
        "explored_nodes": int(state.explored_nodes),
        "closure_checks": int(state.closure_checks),
        "closure_feasible": int(state.closure_feasible),
        "closure_infeasible": int(state.closure_infeasible),
        "dispatch_checks": int(state.dispatch_checks),
        "dispatch_feasible": int(state.dispatch_feasible),
        "dispatch_infeasible": int(state.dispatch_infeasible),
        "edge_checks": int(state.edge_checks),
        "source_time_checks": int(state.source_time_checks),
        "silent_service_checks": int(state.silent_service_checks),
        "destination_checks": int(state.destination_checks),
        "feasible_edges": int(state.feasible_edges),
        "infeasible_edges": int(state.infeasible_edges),
        "terminal_checks": int(state.terminal_checks),
        "max_reached_depth": int(state.max_reached_depth),
        "source_path_tail": list(path[-8:]),
    }
    if source is not None:
        row["source_id"] = source.source_id
    row.update(extra)
    callback(row)


def solve_event_graph(
    obligation_id: str,
    problem: EventGraphProblem,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[EventGraphReceipt, EventWindowEncoding | None]:
    """Exhaustively prove one FirstBadEventWindow by explicit symbolic DFS."""

    solver = make_solver()
    solver.add(problem.base_formula)
    search = _SearchState()
    sources = _search_sources(problem)
    protected_task_names = frozenset(
        derive_protected_priority_prefix(problem.model).task_names
    )
    floor = derive_minimum_event_depth(problem.model, problem.target_task)
    finite_bound = int(problem.event_bound.finite_event_bound)
    decisive_encoding: EventWindowEncoding | None = None
    decisive_path: tuple[str, ...] = ()
    decisive_depth: int | None = None

    event_states: list[SymbolicKernelState] = [problem.start_state]
    event_steps: list[EventStepEncoding] = []
    path_formulas: list[z3.BoolRef] = []
    path_sources: list[str] = []

    def dfs(
        state: SymbolicKernelState,
        depth: int,
        controller_enabled: bool,
        controller_delta: z3.ArithRef,
    ) -> str:
        nonlocal decisive_encoding, decisive_path, decisive_depth
        search.explored_nodes += 1
        search.max_reached_depth = max(search.max_reached_depth, depth)
        search.at(depth).nodes += 1
        _progress(progress, search, phase="EVENT_GRAPH_NODE", depth=depth, path=tuple(path_sources))

        if depth >= finite_bound:
            # Every non-horizon edge is strictly before the horizon because
            # HORIZON is the first canonical minimum owner.  A feasible node at
            # the finite bound therefore contradicts the certified event bound.
            search.unknown_reason = "FINITE_EVENT_BOUND_REACHED_BEFORE_HORIZON"
            return "UNKNOWN"

        controller_cases = (
            enumerate_controller_policy_cases(problem.model)
            if controller_enabled else (None,)
        )
        for controller_case in controller_cases:
            closure_serial = search.edge_serial
            search.edge_serial += 1
            build_started = perf_counter()
            closure = encode_event_node_closure(
                state,
                problem.model,
                problem.environment,
                prefix=f"event.graph.node.{closure_serial}.{depth}",
                controller_enabled=controller_enabled,
                active_task_names=problem.projection.active_task_names,
                controller_case=controller_case,
            )
            build_elapsed = perf_counter() - build_started
            search.build_seconds += build_elapsed
            search.at(depth).build_seconds += build_elapsed
            solver.push()
            solver.add(closure.formula)
            search.closure_checks += 1
            search.at(depth).closure_checks += 1
            _progress(
                progress, search, phase="EVENT_GRAPH_NODE_CLOSURE_CHECK",
                depth=depth, path=tuple(path_sources),
                controller_enabled=controller_enabled,
                controller_case_id=(
                    None if controller_case is None else controller_case.case_id
                ),
            )
            closure_result, closure_elapsed, closure_reason = _check(solver)
            search.solver_seconds += closure_elapsed
            search.at(depth).solver_seconds += closure_elapsed
            if closure_result == "UNSAT":
                search.closure_infeasible += 1
                search.at(depth).closure_infeasible += 1
                solver.pop()
                continue
            if closure_result == "UNKNOWN":
                search.unknown_reason = closure_reason or "EVENT_GRAPH_NODE_CLOSURE_UNKNOWN"
                solver.pop()
                return "UNKNOWN"
            search.closure_feasible += 1
            search.at(depth).closure_feasible += 1

            dispatch_cases = enumerate_dispatch_cases(
                closure.dispatch_state,
                problem.model,
                active_task_names=problem.projection.active_task_names,
                protected_task_names=protected_task_names,
            )
            for dispatch_case in dispatch_cases:
                solver.push()
                solver.add(dispatch_case.formula)
                search.dispatch_checks += 1
                search.at(depth).dispatch_checks += 1
                _progress(
                    progress, search, phase="EVENT_GRAPH_DISPATCH_CASE_CHECK",
                    depth=depth, path=tuple(path_sources),
                    controller_case_id=(
                        None if controller_case is None else controller_case.case_id
                    ),
                    dispatch_case_id=dispatch_case.case_id,
                )
                dispatch_result, dispatch_elapsed, dispatch_reason = _check(solver)
                search.solver_seconds += dispatch_elapsed
                search.at(depth).solver_seconds += dispatch_elapsed
                if dispatch_result == "UNSAT":
                    search.dispatch_infeasible += 1
                    search.at(depth).dispatch_infeasible += 1
                    solver.pop()
                    continue
                if dispatch_result == "UNKNOWN":
                    search.unknown_reason = dispatch_reason or "EVENT_GRAPH_DISPATCH_CASE_UNKNOWN"
                    solver.pop(); solver.pop()
                    return "UNKNOWN"
                search.dispatch_feasible += 1
                search.at(depth).dispatch_feasible += 1

                # Build the event-time candidate normal form once for this exact
                # P6 winner.  Every source check below reuses these same terms.
                candidate_serial = search.edge_serial
                search.edge_serial += 1
                candidates = build_event_candidates(
                    closure.dispatch_state,
                    problem.model,
                    horizon_time=problem.horizon_time,
                    controller_delta=controller_delta,
                    prefix=f"event.graph.node.{candidate_serial}.{depth}.candidates",
                    source=None,
                    active_task_names=problem.projection.active_task_names,
                    selected_job_key=dispatch_case.selected_key,
                )
                solver.add(candidates.base_definition_formula)

                source_rows: list[tuple[EventSource, z3.BoolRef]] = []
                for source in sources:
                    if source.kind == "COMPLETION" and not dispatch_case.running:
                        continue
                    next_depth = depth + 1
                    if source.kind == "HORIZON" and next_depth < floor.minimum_depth:
                        search.structural_skips += 1
                        search.at(depth).structural_skips += 1
                        continue
                    source_rows.append((
                        source,
                        event_source_time_formula(
                            state, candidates, closure.dispatch_state, problem.model, source,
                            horizon_time=problem.horizon_time,
                        ),
                    ))

                def progress_batch(source_ids: tuple[str, ...]) -> None:
                    _progress(
                        progress, search, phase="EVENT_GRAPH_SOURCE_TIME_BATCH_CHECK",
                        depth=depth, path=tuple(path_sources),
                        controller_case_id=(
                            None if controller_case is None else controller_case.case_id
                        ),
                        dispatch_case_id=dispatch_case.case_id,
                        source_ids=list(source_ids),
                        source_workers=SOURCE_TIME_WORKERS,
                    )

                precheck = _parallel_source_precheck(
                    solver, tuple(source_rows),
                    workers=SOURCE_TIME_WORKERS, progress_batch=progress_batch,
                )
                feasible_source_rows: list[tuple[EventSource, z3.BoolRef]] = []
                for source, source_formula in source_rows:
                    search.edge_checks += 1
                    search.at(depth).edge_checks += 1
                    search.source_time_checks += 1
                    search.at(depth).source_time_checks += 1
                    stage_result, elapsed, reason = precheck[source.source_id]
                    search.solver_seconds += elapsed
                    search.at(depth).solver_seconds += elapsed
                    _progress(
                        progress, search, phase="EVENT_GRAPH_SOURCE_TIME_RESULT",
                        depth=depth, source=source, path=tuple(path_sources),
                        next_depth=depth + 1,
                        controller_case_id=(
                            None if controller_case is None else controller_case.case_id
                        ),
                        dispatch_case_id=dispatch_case.case_id,
                        source_time_result=stage_result,
                        source_workers=SOURCE_TIME_WORKERS,
                    )
                    if stage_result == "UNSAT":
                        search.infeasible_edges += 1
                        search.at(depth).infeasible_edges += 1
                        continue
                    if stage_result == "UNKNOWN":
                        search.unknown_reason = reason or "EVENT_GRAPH_SOURCE_TIME_UNKNOWN"
                        solver.pop(); solver.pop()
                        return "UNKNOWN"
                    feasible_source_rows.append((source, source_formula))

                for source, source_formula in feasible_source_rows:
                    next_depth = depth + 1
                    serial = search.edge_serial
                    search.edge_serial += 1
                    destination = declare_state(
                        f"event.graph.edge.{serial}.y.{next_depth}", problem.model
                    )
                    build_started = perf_counter()
                    step = encode_event_relative_edge(
                        state, destination, closure, problem.model,
                        horizon_time=problem.horizon_time,
                        controller_delta=controller_delta,
                        prefix=f"event.graph.edge.{serial}",
                        event_source=source,
                        active_task_names=problem.projection.active_task_names,
                        candidates=candidates,
                        selected_job_key=dispatch_case.selected_key,
                    )
                    destination_controller_delta = z3.Int(
                        f"event.graph.edge.{serial}.controller_delta"
                    )
                    if source.kind == "CONTROLLER":
                        controller_clock_update = (
                            destination_controller_delta == int(problem.model.agent_period)
                        )
                    elif source.kind == "HORIZON":
                        controller_clock_update = (destination_controller_delta == controller_delta)
                    else:
                        controller_clock_update = z3.And(
                            destination_controller_delta == controller_delta - step.delta,
                            destination_controller_delta >= 1,
                            destination_controller_delta <= int(problem.model.agent_period),
                        )
                    destination_terms: list[z3.BoolRef] = [
                        step.destination_formula,
                        controller_clock_update,
                        state.hi_miss_ledger == 0,
                        destination.hi_miss_ledger == 0,
                        reachable_carry_in_consistency(
                            destination, problem.projection.active_model
                        ),
                    ]
                    if source.kind != "HORIZON":
                        destination_terms.append(target_pending_after_origin(
                            destination, problem.model, problem.target_task
                        ))
                    destination_check_formula = z3.And(*destination_terms)
                    materialized_edge_formula = z3.And(
                        dispatch_case.formula,
                        candidates.base_definition_formula,
                        step.formula,
                        controller_clock_update,
                        state.hi_miss_ledger == 0,
                        destination.hi_miss_ledger == 0,
                        reachable_carry_in_consistency(
                            destination, problem.projection.active_model
                        ),
                        *((target_pending_after_origin(
                            destination, problem.model, problem.target_task
                        ),) if source.kind != "HORIZON" else ()),
                    )
                    build_elapsed = perf_counter() - build_started
                    search.build_seconds += build_elapsed
                    search.at(depth).build_seconds += build_elapsed

                    solver.push()
                    # The exact source-time query was already proved SAT in an
                    # isolated solver over byte-for-byte equivalent assertions.
                    solver.add(source_formula)

                    solver.add(step.silent_service_formula)
                    search.silent_service_checks += 1
                    search.at(depth).silent_service_checks += 1
                    _progress(
                        progress, search, phase="EVENT_GRAPH_SILENT_SERVICE_CHECK", depth=depth,
                        source=source, path=tuple(path_sources), next_depth=next_depth,
                        controller_case_id=(
                            None if controller_case is None else controller_case.case_id
                        ),
                        dispatch_case_id=dispatch_case.case_id,
                    )
                    stage_result, elapsed, reason = _check(solver)
                    search.solver_seconds += elapsed
                    search.at(depth).solver_seconds += elapsed
                    if stage_result == "UNSAT":
                        search.infeasible_edges += 1
                        search.at(depth).infeasible_edges += 1
                        solver.pop()
                        continue
                    if stage_result == "UNKNOWN":
                        search.unknown_reason = reason or "EVENT_GRAPH_SILENT_SERVICE_UNKNOWN"
                        solver.pop(); solver.pop(); solver.pop()
                        return "UNKNOWN"

                    solver.add(destination_check_formula)
                    search.destination_checks += 1
                    search.at(depth).destination_checks += 1
                    _progress(
                        progress, search, phase="EVENT_GRAPH_DESTINATION_CHECK", depth=depth,
                        source=source, path=tuple(path_sources), next_depth=next_depth,
                        controller_case_id=(
                            None if controller_case is None else controller_case.case_id
                        ),
                        dispatch_case_id=dispatch_case.case_id,
                    )
                    stage_result, elapsed, reason = _check(solver)
                    search.solver_seconds += elapsed
                    search.at(depth).solver_seconds += elapsed
                    if stage_result == "UNSAT":
                        search.infeasible_edges += 1
                        search.at(depth).infeasible_edges += 1
                        solver.pop()
                        continue
                    if stage_result == "UNKNOWN":
                        search.unknown_reason = reason or "EVENT_GRAPH_DESTINATION_UNKNOWN"
                        solver.pop(); solver.pop(); solver.pop()
                        return "UNKNOWN"

                    search.feasible_edges += 1
                    search.at(depth).feasible_edges += 1
                    event_states.append(destination)
                    event_steps.append(step)
                    path_formulas.append(materialized_edge_formula)
                    path_sources.append(source.source_id)

                    if source.kind == "HORIZON":
                        terminal = problem.build_terminal_bad_query(destination, depth=next_depth)
                        solver.push()
                        solver.add(terminal.formula)
                        search.terminal_checks += 1
                        search.at(depth).terminal_checks += 1
                        _progress(
                            progress, search, phase="EVENT_GRAPH_TERMINAL_BAD_CHECK",
                            depth=next_depth, source=source, path=tuple(path_sources),
                            dispatch_case_id=dispatch_case.case_id,
                        )
                        terminal_result, terminal_elapsed, terminal_reason = _check(solver)
                        search.solver_seconds += terminal_elapsed
                        search.at(depth).solver_seconds += terminal_elapsed
                        solver.pop()
                        if terminal_result == "SAT":
                            decisive_depth = next_depth
                            decisive_path = tuple(path_sources)
                            decisive_encoding = problem.materialize_sat_path(
                                root_case=current_root_formula,
                                event_states=tuple(event_states),
                                event_steps=tuple(event_steps),
                                path_formulas=tuple(path_formulas),
                                terminal=terminal,
                            )
                            path_sources.pop(); path_formulas.pop(); event_steps.pop(); event_states.pop()
                            solver.pop(); solver.pop(); solver.pop()
                            return "SAT"
                        if terminal_result == "UNKNOWN":
                            search.unknown_reason = terminal_reason or "EVENT_GRAPH_TERMINAL_UNKNOWN"
                            path_sources.pop(); path_formulas.pop(); event_steps.pop(); event_states.pop()
                            solver.pop(); solver.pop(); solver.pop()
                            return "UNKNOWN"
                        search.at(depth).terminal_safe += 1
                        child_result = "UNSAT"
                    else:
                        child_result = dfs(
                            destination,
                            next_depth,
                            controller_enabled=(source.kind == "CONTROLLER"),
                            controller_delta=destination_controller_delta,
                        )

                    path_sources.pop()
                    path_formulas.pop()
                    event_steps.pop()
                    event_states.pop()
                    solver.pop()
                    if child_result in {"SAT", "UNKNOWN"}:
                        solver.pop(); solver.pop()
                        return child_result

                solver.pop()

            solver.pop()

        return "UNSAT"

    overall = "UNSAT"
    current_root_case = z3.BoolVal(True)
    current_root_formula = z3.BoolVal(True)
    # Target release at the window origin can itself coincide with a controller
    # activation.  This is the only graph node whose controller phase is not
    # determined by an incoming canonical source, so split it once and exactly.
    for root_enabled in (False, True):
        current_root_case = (
            problem.start_state.t % problem.model.agent_period == 0
            if root_enabled
            else problem.start_state.t % problem.model.agent_period != 0
        )
        root_controller_delta = z3.Int(
            f"event.graph.root.controller_delta.{int(root_enabled)}"
        )
        root_clock_formula = z3.And(
            root_controller_delta
            == exact_periodic_countdown(
                problem.start_state.t, problem.model.agent_period
            ),
            root_controller_delta >= 1,
            root_controller_delta <= int(problem.model.agent_period),
        )
        current_root_formula = z3.And(current_root_case, root_clock_formula)
        solver.push()
        solver.add(current_root_case, root_clock_formula)
        root_result, root_elapsed, root_reason = _check(solver)
        search.solver_seconds += root_elapsed
        search.at(0).solver_seconds += root_elapsed
        if root_result == "UNKNOWN":
            search.unknown_reason = root_reason or "EVENT_GRAPH_ROOT_PHASE_UNKNOWN"
            solver.pop()
            overall = "UNKNOWN"
            break
        if root_result == "SAT":
            _progress(
                progress, search, phase="EVENT_GRAPH_ROOT_PHASE",
                depth=0, path=(), controller_enabled=root_enabled,
            )
            result = dfs(
                problem.start_state, 0, root_enabled, root_controller_delta
            )
            if result in {"SAT", "UNKNOWN"}:
                solver.pop()
                overall = result
                break
        solver.pop()

    depth_rows = tuple(
        search.depth_stats[depth].as_dict(depth)
        for depth in sorted(search.depth_stats)
    )
    receipt = EventGraphReceipt(
        obligation_id=obligation_id,
        result=overall,
        solver_version=z3.get_version_string(),
        strategy=SOLVER_STRATEGY,
        finite_event_bound=finite_bound,
        minimum_feasible_depth=floor.minimum_depth,
        explored_nodes=search.explored_nodes,
        closure_checks=search.closure_checks,
        closure_feasible=search.closure_feasible,
        closure_infeasible=search.closure_infeasible,
        dispatch_checks=search.dispatch_checks,
        dispatch_feasible=search.dispatch_feasible,
        dispatch_infeasible=search.dispatch_infeasible,
        edge_checks=search.edge_checks,
        source_time_checks=search.source_time_checks,
        silent_service_checks=search.silent_service_checks,
        destination_checks=search.destination_checks,
        feasible_edges=search.feasible_edges,
        infeasible_edges=search.infeasible_edges,
        terminal_checks=search.terminal_checks,
        structural_skips=search.structural_skips,
        max_reached_depth=search.max_reached_depth,
        solver_check_seconds=search.solver_seconds,
        edge_build_seconds=search.build_seconds,
        depth_stats=depth_rows,
        decisive_depth=decisive_depth,
        decisive_source_path=decisive_path,
        reason=search.unknown_reason,
    )
    return receipt, decisive_encoding


__all__ = ["EventGraphReceipt", "SOLVER_STRATEGY", "solve_event_graph"]
