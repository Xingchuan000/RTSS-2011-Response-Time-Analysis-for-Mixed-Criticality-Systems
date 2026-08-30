"""V9.3 explicit exact Event-graph verifier.

The solver never receives a symbolic multi-event skeleton.  Python owns the
finite source search.  Every SMT check contains one already-fixed next-event
source appended to the current feasible prefix, so arithmetic feasibility and
kernel semantics remain in Z3 while combinatorial source enumeration does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import z3

from .carry_in import derive_protected_priority_prefix, reachable_carry_in_consistency
from .controller_encoder import enumerate_controller_policy_cases
from .event_depth_feasibility import derive_minimum_event_depth
from .event_kernel import (
    EventSource,
    EventStepEncoding,
    encode_event_node_closure,
    encode_event_time_edge,
    enumerate_event_sources,
)
from .event_window_encoder import EventGraphProblem, EventWindowEncoding
from .symbolic_state import SymbolicKernelState, declare_state
from .target_projection import target_pending_after_origin


SOLVER_STRATEGY = "V9_3_TARGET_LOCAL_NORMALIZED_EVENT_GRAPH_DFS"


@dataclass(slots=True)
class _DepthStats:
    nodes: int = 0
    closure_checks: int = 0
    closure_feasible: int = 0
    closure_infeasible: int = 0
    edge_checks: int = 0
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
            "edge_checks": int(self.edge_checks),
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
    edge_checks: int
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
            "edge_checks": int(self.edge_checks),
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
    edge_checks: int = 0
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


def _search_sources(problem: EventGraphProblem) -> tuple[EventSource, ...]:
    """Static exact source set after proven slot-shape eliminations."""

    protected = set(derive_protected_priority_prefix(problem.model).task_names)
    active = frozenset(problem.projection.active_task_names)
    rows: list[EventSource] = []
    for source in enumerate_event_sources(
        problem.model, active_task_names=active
    ):
        if source.kind == "HI_DEADLINE" and source.slot != 0:
            continue
        if source.kind == "COMPLETION" and source.task_name is not None:
            task = problem.model.task_by_name[source.task_name]
            if task.criticality == "HI" and source.slot != 0:
                continue
            if task.criticality == "LO" and source.task_name in protected and source.slot == 0:
                continue
        rows.append(source)
    return tuple(rows)


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
        "edge_checks": int(state.edge_checks),
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

    solver = z3.Solver()
    solver.add(problem.base_formula)
    search = _SearchState()
    sources = _search_sources(problem)
    floor = derive_minimum_event_depth(problem.model, problem.target_task)
    finite_bound = int(problem.event_bound.finite_event_bound)
    decisive_encoding: EventWindowEncoding | None = None
    decisive_path: tuple[str, ...] = ()
    decisive_depth: int | None = None

    event_states: list[SymbolicKernelState] = [problem.start_state]
    event_steps: list[EventStepEncoding] = []
    path_formulas: list[z3.BoolRef] = []
    path_sources: list[str] = []

    def dfs(state: SymbolicKernelState, depth: int, controller_enabled: bool) -> str:
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

            for source in sources:
                next_depth = depth + 1
                if source.kind == "HORIZON" and next_depth < floor.minimum_depth:
                    search.structural_skips += 1
                    search.at(depth).structural_skips += 1
                    continue

                serial = search.edge_serial
                search.edge_serial += 1
                destination = declare_state(
                    f"event.graph.edge.{serial}.y.{next_depth}", problem.model
                )
                build_started = perf_counter()
                step = encode_event_time_edge(
                    state,
                    destination,
                    closure,
                    problem.model,
                    horizon_time=problem.horizon_time,
                    prefix=f"event.graph.edge.{serial}",
                    event_source=source,
                    active_task_names=problem.projection.active_task_names,
                )
                edge_terms: list[z3.BoolRef] = [
                    step.time_edge_formula,
                    state.hi_miss_ledger == 0,
                    destination.hi_miss_ledger == 0,
                    reachable_carry_in_consistency(
                        destination, problem.projection.active_model
                    ),
                ]
                if source.kind != "HORIZON":
                    edge_terms.append(target_pending_after_origin(
                        destination, problem.model, problem.target_task
                    ))
                edge_formula = z3.And(*edge_terms)
                build_elapsed = perf_counter() - build_started
                search.build_seconds += build_elapsed
                search.at(depth).build_seconds += build_elapsed

                solver.push()
                solver.add(edge_formula)
                search.edge_checks += 1
                search.at(depth).edge_checks += 1
                _progress(
                    progress, search, phase="EVENT_GRAPH_TIME_EDGE_CHECK", depth=depth,
                    source=source, path=tuple(path_sources), next_depth=next_depth,
                    controller_case_id=(
                        None if controller_case is None else controller_case.case_id
                    ),
                )
                edge_result, elapsed, reason = _check(solver)
                search.solver_seconds += elapsed
                search.at(depth).solver_seconds += elapsed

                if edge_result == "UNSAT":
                    search.infeasible_edges += 1
                    search.at(depth).infeasible_edges += 1
                    solver.pop()
                    continue
                if edge_result == "UNKNOWN":
                    search.unknown_reason = reason or "EVENT_GRAPH_EDGE_UNKNOWN"
                    solver.pop()
                    solver.pop()
                    return "UNKNOWN"

                search.feasible_edges += 1
                search.at(depth).feasible_edges += 1
                event_states.append(destination)
                event_steps.append(step)
                path_formulas.append(step.formula)
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
                    )
                    terminal_result, terminal_elapsed, terminal_reason = _check(solver)
                    search.solver_seconds += terminal_elapsed
                    search.at(depth).solver_seconds += terminal_elapsed
                    solver.pop()
                    if terminal_result == "SAT":
                        decisive_depth = next_depth
                        decisive_path = tuple(path_sources)
                        decisive_encoding = problem.materialize_sat_path(
                            root_case=current_root_case,
                            event_states=tuple(event_states),
                            event_steps=tuple(event_steps),
                            path_formulas=tuple(path_formulas),
                            terminal=terminal,
                        )
                        path_sources.pop(); path_formulas.pop(); event_steps.pop(); event_states.pop()
                        solver.pop()
                        solver.pop()
                        return "SAT"
                    if terminal_result == "UNKNOWN":
                        search.unknown_reason = terminal_reason or "EVENT_GRAPH_TERMINAL_UNKNOWN"
                        path_sources.pop(); path_formulas.pop(); event_steps.pop(); event_states.pop()
                        solver.pop()
                        solver.pop()
                        return "UNKNOWN"
                    search.at(depth).terminal_safe += 1
                    child_result = "UNSAT"
                else:
                    child_result = dfs(
                        destination,
                        next_depth,
                        controller_enabled=(source.kind == "CONTROLLER"),
                    )

                path_sources.pop()
                path_formulas.pop()
                event_steps.pop()
                event_states.pop()
                solver.pop()
                if child_result in {"SAT", "UNKNOWN"}:
                    solver.pop()
                    return child_result

            solver.pop()

        return "UNSAT"

    overall = "UNSAT"
    current_root_case = z3.BoolVal(True)
    # Target release at the window origin can itself coincide with a controller
    # activation.  This is the only graph node whose controller phase is not
    # determined by an incoming canonical source, so split it once and exactly.
    for root_enabled in (False, True):
        current_root_case = (
            problem.start_state.t % problem.model.agent_period == 0
            if root_enabled
            else problem.start_state.t % problem.model.agent_period != 0
        )
        solver.push()
        solver.add(current_root_case)
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
            result = dfs(problem.start_state, 0, root_enabled)
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
        edge_checks=search.edge_checks,
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
