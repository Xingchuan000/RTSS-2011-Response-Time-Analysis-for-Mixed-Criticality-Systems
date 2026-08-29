"""V9.3 exact terminal-depth Event BMC.

There is one solving lifecycle only:

1. derive a sound structural lower bound on Event depth;
2. skip impossible shallow depths without constructing a terminal solver;
3. at each feasible depth enumerate an exact finite specialization of the
   controller phase and penultimate event source;
4. create a brand-new solver for every specialization and call ``check`` once.

There are no probes, timeouts, push/pop refinement, same-solver resumes,
catch-all compatibility leaves, or monolithic fallback queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable

import z3

from .carry_in import derive_protected_priority_prefix
from .event_depth_feasibility import (
    derive_minimum_event_depth,
    penultimate_release_is_impossible,
)
from .event_window_encoder import EventWindowEncoding, IncrementalEventWindowEncoding


SOLVER_STRATEGY = "V9_3_FRESH_SPECIALIZED_EXACT_LEAF_BMC"


@dataclass(frozen=True, slots=True)
class DepthCaseReceipt:
    case_id: str
    result: str
    solver_check_seconds: float
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "case_id": self.case_id,
            "result": self.result,
            "solver_check_seconds": round(float(self.solver_check_seconds), 6),
        }
        if self.reason is not None:
            row["reason"] = self.reason
        return row


@dataclass(frozen=True, slots=True)
class IncrementalDepthReceipt:
    depth: int
    result: str
    solver_check_seconds: float
    terminal_build_seconds: float
    structurally_pruned: bool = False
    prune_reason: str | None = None
    case_receipts: tuple[DepthCaseReceipt, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "depth": int(self.depth),
            "result": self.result,
            "solver_check_seconds": round(float(self.solver_check_seconds), 6),
            "terminal_build_seconds": round(float(self.terminal_build_seconds), 6),
            "structurally_pruned": bool(self.structurally_pruned),
            "fresh_solver_per_exact_leaf": True,
        }
        if self.prune_reason is not None:
            row["prune_reason"] = self.prune_reason
        if self.case_receipts:
            row["case_receipts"] = [item.as_dict() for item in self.case_receipts]
        return row


@dataclass(frozen=True, slots=True)
class IncrementalBMCReceipt:
    obligation_id: str
    result: str
    solver_version: str
    strategy: str
    max_depth: int
    minimum_feasible_depth: int
    minimum_depth_witness_task: str | None
    checked_depth_count: int
    decisive_depth: int | None
    step_build_seconds: float
    terminal_build_seconds: float
    solver_check_seconds: float
    depth_receipts: tuple[IncrementalDepthReceipt, ...]
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "obligation_id": self.obligation_id,
            "result": self.result,
            "solver_version": self.solver_version,
            "solver_strategy": self.strategy,
            "max_depth": int(self.max_depth),
            "minimum_feasible_depth": int(self.minimum_feasible_depth),
            "minimum_depth_witness_task": self.minimum_depth_witness_task,
            "checked_depth_count": int(self.checked_depth_count),
            "step_build_seconds": round(float(self.step_build_seconds), 6),
            "terminal_build_seconds": round(float(self.terminal_build_seconds), 6),
            "solver_check_seconds": round(float(self.solver_check_seconds), 6),
            "depth_receipts": [item.as_dict() for item in self.depth_receipts],
            "exact_active_event_prefix": True,
            "fresh_solver_per_exact_leaf": True,
        }
        if self.decisive_depth is not None:
            row["decisive_depth"] = int(self.decisive_depth)
        if self.reason is not None:
            row["reason"] = self.reason
        return row


def _fresh_check(assertions: Iterable[z3.BoolRef]) -> tuple[str, float, str | None]:
    """Build one fully-specialized solver and check it exactly once."""

    solver = z3.Solver()
    solver.add(*tuple(assertions))
    started = perf_counter()
    result = solver.check()
    elapsed = perf_counter() - started
    if result == z3.sat:
        return "SAT", elapsed, None
    if result == z3.unsat:
        return "UNSAT", elapsed, None
    return "UNKNOWN", elapsed, solver.reason_unknown()


def _terminal_horizon_bridge(encoding: IncrementalEventWindowEncoding) -> z3.BoolRef:
    if not encoding.event_steps:
        return z3.BoolVal(True)
    last = encoding.event_steps[-1]
    return z3.And(
        last.candidates.next_time == encoding.horizon_time,
        *(value >= encoding.horizon_time for value in last.candidates.all_candidates),
    )


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


@dataclass(frozen=True, slots=True)
class _ExactCase:
    case_id: str
    cube: z3.BoolRef


def _controller_count_cases(
    encoding: IncrementalEventWindowEncoding,
) -> tuple[_ExactCase, ...]:
    """Exact controller-activation-count partition specialized before solving."""

    instances = encoding.controller_pool.instances
    rows: list[_ExactCase] = []
    for count in range(len(instances) + 1):
        inside = [
            instance.activation_time < encoding.horizon_time
            for instance in instances[:count]
        ]
        outside = [
            instance.activation_time >= encoding.horizon_time
            for instance in instances[count:]
        ]
        cube = z3.And(*(inside + outside)) if inside or outside else z3.BoolVal(True)
        rows.append(_ExactCase(f"CTRL_COUNT_{count}", cube))
    return tuple(rows)


def _penultimate_source_cases(
    encoding: IncrementalEventWindowEncoding,
) -> tuple[_ExactCase, ...]:
    """Exact source-member cover after structural tick elimination.

    Release timestamps remain an exact finite union inside one task member.
    Ticks that would force a protected higher-priority completion before the
    horizon are removed host-side and never reach Z3.
    """

    if len(encoding.event_steps) < 2:
        return (_ExactCase("NO_PENULTIMATE_SOURCE", z3.BoolVal(True)),)

    step = encoding.event_steps[-2]
    candidates = step.candidates
    nxt = candidates.next_time
    origin = encoding.environment.phase.origin_time
    deadline = int(encoding.deadline)
    rows: list[_ExactCase] = []
    protected = set(derive_protected_priority_prefix(encoding.model).task_names)

    for task_name, release_value in candidates.releases:
        feasible_ticks = tuple(
            int(tick)
            for tick in encoding.environment.allowed_ticks_by_task.get(task_name, ())
            if 0 < int(tick) < deadline
            and not penultimate_release_is_impossible(
                encoding.model, encoding.target_task, task_name, int(tick)
            )
        )
        if not feasible_ticks:
            continue
        rows.append(_ExactCase(
            f"SRC_RELEASE_TASK_{_safe_token(task_name)}",
            z3.And(
                nxt == release_value,
                z3.Or(*(nxt == origin + tick for tick in feasible_ticks)),
            ),
        ))

    rows.append(_ExactCase(
        "SRC_CONTROLLER",
        z3.And(nxt == candidates.controller, candidates.controller < encoding.horizon_time),
    ))

    for (task_name, slot), value in candidates.hi_deadlines:
        if slot != 0:
            continue
        rows.append(_ExactCase(
            f"SRC_HI_DEADLINE_JOB_{_safe_token(task_name)}_{slot}",
            z3.And(nxt == value, value < encoding.horizon_time),
        ))

    dispatch = step.phase_states[-1]
    for task_index, task in enumerate(encoding.model.tasks):
        for slot in range(encoding.model.max_jobs_per_task):
            if task.criticality == "HI" and slot != 0:
                continue
            if task.criticality == "LO" and task.name in protected and slot == 0:
                continue
            selected_index = task_index * encoding.model.max_jobs_per_task + slot
            rows.append(_ExactCase(
                f"SRC_COMPLETION_SLOT_{_safe_token(task.name)}_{slot}",
                z3.And(
                    nxt == candidates.completion,
                    dispatch.frontier.selected_slot == selected_index,
                    candidates.completion < encoding.horizon_time,
                ),
            ))

    return tuple(rows)


def _solve_depth_exact_cases(
    encoding: IncrementalEventWindowEncoding,
    assertions: tuple[z3.BoolRef, ...],
    *,
    depth: int,
    progress: Callable[[dict[str, Any]], None] | None,
) -> tuple[str, float, tuple[DepthCaseReceipt, ...], str | None]:
    rows: list[DepthCaseReceipt] = []
    total = 0.0
    controller_cases = _controller_count_cases(encoding)
    source_cases = _penultimate_source_cases(encoding)
    if not controller_cases or not source_cases:
        return "UNSAT", 0.0, (), None

    for controller in controller_cases:
        for source in source_cases:
            case_id = f"{controller.case_id}__{source.case_id}"
            if progress is not None:
                progress({
                    "depth": depth,
                    "phase": "FRESH_SPECIALIZED_LEAF_CHECK",
                    "case_id": case_id,
                })
            result, elapsed, reason = _fresh_check((
                *assertions,
                controller.cube,
                source.cube,
            ))
            total += elapsed
            rows.append(DepthCaseReceipt(case_id, result, elapsed, reason))
            if progress is not None:
                progress({
                    "depth": depth,
                    "phase": "FRESH_SPECIALIZED_LEAF_RESULT",
                    "case_id": case_id,
                    "case_result": result,
                    "case_solver_check_seconds": round(elapsed, 6),
                })
            if result == "SAT":
                return "SAT", total, tuple(rows), None
            if result == "UNKNOWN":
                return "UNKNOWN", total, tuple(rows), reason
    return "UNSAT", total, tuple(rows), None


def solve_incremental_event_window(
    obligation_id: str,
    encoding: IncrementalEventWindowEncoding,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[IncrementalBMCReceipt, EventWindowEncoding | None]:
    """Prove the exact FirstBadEventWindow with fresh specialized leaf solvers."""

    max_depth = int(encoding.event_bound.finite_event_bound)
    floor = derive_minimum_event_depth(encoding.model, encoding.target_task)
    rows: list[IncrementalDepthReceipt] = []
    step_build_seconds = 0.0
    terminal_build_seconds = 0.0
    solver_check_seconds = 0.0

    for depth in range(0, max_depth + 1):
        if depth > 0:
            started = perf_counter()
            encoding.append_exact_event_step()
            step_build_seconds += perf_counter() - started

        if depth < floor.minimum_depth:
            reason = (
                f"EVENT_DEPTH_BELOW_PROTECTED_STREAM_FLOOR:{floor.witness_task}:"
                f"{floor.minimum_depth}"
            )
            rows.append(IncrementalDepthReceipt(
                depth=depth,
                result="UNSAT",
                solver_check_seconds=0.0,
                terminal_build_seconds=0.0,
                structurally_pruned=True,
                prune_reason=reason,
            ))
            if progress is not None:
                progress({
                    "depth": depth,
                    "max_depth": max_depth,
                    "phase": "STRUCTURAL_DEPTH_PRUNE",
                    "minimum_feasible_depth": floor.minimum_depth,
                    "witness_task": floor.witness_task,
                })
            continue

        terminal_started = perf_counter()
        terminal = encoding.build_terminal_bad_query()
        bridge = _terminal_horizon_bridge(encoding)
        one_terminal_build = perf_counter() - terminal_started
        terminal_build_seconds += one_terminal_build
        assertions = (
            encoding.base_formula,
            *tuple(encoding.prefix_formulas),
            bridge,
            terminal.formula,
        )
        effective_result, one_check, case_rows, reason = _solve_depth_exact_cases(
            encoding,
            assertions,
            depth=depth,
            progress=progress,
        )
        solver_check_seconds += one_check
        rows.append(IncrementalDepthReceipt(
            depth=depth,
            result=effective_result,
            solver_check_seconds=one_check,
            terminal_build_seconds=one_terminal_build,
            case_receipts=case_rows,
        ))
        if progress is not None:
            progress({
                "depth": depth,
                "max_depth": max_depth,
                "phase": "DEPTH_RESULT",
                "depth_result": effective_result,
                "depth_solver_check_seconds": round(one_check, 6),
                "total_solver_check_seconds": round(solver_check_seconds, 6),
            })

        if effective_result == "SAT":
            decisive = encoding.materialize_depth(terminal)
            return IncrementalBMCReceipt(
                obligation_id=obligation_id,
                result="SAT",
                solver_version=z3.get_version_string(),
                strategy=SOLVER_STRATEGY,
                max_depth=max_depth,
                minimum_feasible_depth=floor.minimum_depth,
                minimum_depth_witness_task=floor.witness_task,
                checked_depth_count=len(rows),
                decisive_depth=depth,
                step_build_seconds=step_build_seconds,
                terminal_build_seconds=terminal_build_seconds,
                solver_check_seconds=solver_check_seconds,
                depth_receipts=tuple(rows),
            ), decisive
        if effective_result == "UNKNOWN":
            return IncrementalBMCReceipt(
                obligation_id=obligation_id,
                result="UNKNOWN",
                solver_version=z3.get_version_string(),
                strategy=SOLVER_STRATEGY,
                max_depth=max_depth,
                minimum_feasible_depth=floor.minimum_depth,
                minimum_depth_witness_task=floor.witness_task,
                checked_depth_count=len(rows),
                decisive_depth=depth,
                step_build_seconds=step_build_seconds,
                terminal_build_seconds=terminal_build_seconds,
                solver_check_seconds=solver_check_seconds,
                depth_receipts=tuple(rows),
                reason=reason,
            ), None

    # The only UNSAT exit: every depth is either structurally impossible by an
    # exact protected-stream theorem or every exact specialized leaf was UNSAT.
    return IncrementalBMCReceipt(
        obligation_id=obligation_id,
        result="UNSAT",
        solver_version=z3.get_version_string(),
        strategy=SOLVER_STRATEGY,
        max_depth=max_depth,
        minimum_feasible_depth=floor.minimum_depth,
        minimum_depth_witness_task=floor.witness_task,
        checked_depth_count=len(rows),
        decisive_depth=None,
        step_build_seconds=step_build_seconds,
        terminal_build_seconds=terminal_build_seconds,
        solver_check_seconds=solver_check_seconds,
        depth_receipts=tuple(rows),
    ), None


__all__ = [
    "SOLVER_STRATEGY",
    "DepthCaseReceipt",
    "IncrementalBMCReceipt",
    "IncrementalDepthReceipt",
    "solve_incremental_event_window",
]
