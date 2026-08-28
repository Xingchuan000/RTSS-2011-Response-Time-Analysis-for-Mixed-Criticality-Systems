"""Exact terminal-depth BMC for V9.2 Event FirstBadWindow.

The Event prefix is still extended one exact active macro at a time, but each
terminal depth is solved in a *fresh* Z3 solver.  This is intentional: after a
solver has been checked under push/pop and later extended, Z3 remains in an
incremental context and cannot apply some whole-query preprocessing as
aggressively.  The s313 receipts exposed this directly: very small exact depths
were timing out even though formula construction and memory were already under
control.

For depth ``k`` the queried formula is exactly

    Base & ActiveStep[0] & ... & ActiveStep[k-1] & BadAtHorizon(state[k])

for every ``k`` from 0 through the already-proved finite Event bound.  A small
redundant terminal bridge makes the already-implied final next-event time equal
to the horizon explicit.  If a fresh whole-depth check is UNKNOWN, the same
formula is retried under an exhaustive finite case partition over start mode and
the number of controller activations strictly before the horizon.  The case
partition is an exact Boolean disjunction, not an abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable

import z3

from formal_toolchain.core.hashing import sha256_object

from .event_window_encoder import (
    ENCODER_VERSION,
    EventWindowEncoding,
    IncrementalEventWindowEncoding,
)


SOLVER_STRATEGY = "EXACT_FRESH_TERMINAL_DEPTH_BMC_V2"


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
    reason: str | None = None
    primary_result: str | None = None
    case_receipts: tuple[DepthCaseReceipt, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "depth": int(self.depth),
            "result": self.result,
            "solver_check_seconds": round(float(self.solver_check_seconds), 6),
            "terminal_build_seconds": round(float(self.terminal_build_seconds), 6),
            "fresh_solver_per_depth": True,
        }
        if self.primary_result is not None:
            row["primary_result"] = self.primary_result
        if self.case_receipts:
            row["case_partition_used"] = True
            row["case_partition_kind"] = "START_MODE_X_CONTROLLER_COUNT"
            row["case_receipts"] = [item.as_dict() for item in self.case_receipts]
        else:
            row["case_partition_used"] = False
        if self.reason is not None:
            row["reason"] = self.reason
        return row


@dataclass(frozen=True, slots=True)
class IncrementalBMCReceipt:
    obligation_id: str
    result: str
    solver_version: str
    timeout_ms: int
    strategy: str
    query_plan_hash: str
    max_depth: int
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
            "timeout_ms": int(self.timeout_ms),
            "solver_strategy": self.strategy,
            "query_plan_hash": self.query_plan_hash,
            "max_depth": int(self.max_depth),
            "checked_depth_count": int(self.checked_depth_count),
            "step_build_seconds": round(float(self.step_build_seconds), 6),
            "terminal_build_seconds": round(float(self.terminal_build_seconds), 6),
            "solver_check_seconds": round(float(self.solver_check_seconds), 6),
            "depth_receipts": [item.as_dict() for item in self.depth_receipts],
            "monolithic_formula_materialized": False,
            "terminal_stutter_used": False,
            "exact_active_event_prefix": True,
            "fresh_solver_per_depth": True,
            "exhaustive_case_partition_on_unknown": True,
            "depth_partition_complete_if_unsat": self.result == "UNSAT",
        }
        if self.decisive_depth is not None:
            row["decisive_depth"] = int(self.decisive_depth)
        if self.reason is not None:
            row["reason"] = self.reason
        return row


def _query_plan_hash(encoding: IncrementalEventWindowEncoding) -> str:
    """Hash the exact finite query plan without materializing a huge SMT text."""

    return sha256_object({
        "schema": "v9_2_fresh_terminal_depth_query_plan_v2",
        "solver_strategy": SOLVER_STRATEGY,
        "event_window_encoder_version": ENCODER_VERSION,
        "target_task": encoding.target_task,
        "deadline": encoding.deadline,
        "event_bound": encoding.event_bound.as_dict(),
        "source_obligations": list(encoding.source_obligations),
        "depths": [0, int(encoding.event_bound.finite_event_bound)],
        "depth_interval_is_inclusive": True,
        "active_step_relation": "encode_event_step",
        "terminal_relation": "exact_P0_P1_P2_target_first_bad",
        "terminal_stutter_used": False,
        "fresh_solver_per_depth": True,
        "unknown_fallback_partition": "start_mode_x_controller_count",
    })


def _fresh_check(
    assertions: Iterable[z3.BoolRef], *, timeout_ms: int
) -> tuple[str, float, str | None]:
    """Solve one exact depth in a fresh non-incremental solver context."""

    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
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
    """Expose a redundant equality already implied by the exact last step.

    At depth > 0, the exact Event destination relation defines
    ``state[k].t == last_step.candidates.next_time`` and the terminal query
    defines ``state[k].t == horizon``.  Adding ``next_time == horizon`` directly
    is therefore a pure propagation lemma.  Candidate lower bounds are equally
    implied by exact-minimality and make the final silent interval easier to
    simplify without changing its model set.
    """

    if not encoding.event_steps:
        return z3.BoolVal(True)
    last = encoding.event_steps[-1]
    return z3.And(
        last.candidates.next_time == encoding.horizon_time,
        *(value >= encoding.horizon_time for value in last.candidates.all_candidates),
    )


def _exact_case_partition(
    encoding: IncrementalEventWindowEncoding,
) -> tuple[tuple[str, z3.BoolRef], ...]:
    """Exact finite partition used only after a whole-depth UNKNOWN.

    ``mode_hi`` is Boolean.  Controller instances are strictly ordered by one
    positive agent period, so the instances before the horizon always form a
    prefix.  Enumerating every prefix length 0..N and both start modes is
    exhaustive and disjoint enough for UNSAT aggregation; no phase or behavior
    is removed.
    """

    mode_cases = (
        ("START_MODE_LO", z3.Not(encoding.start_state.mode_hi)),
        ("START_MODE_HI", encoding.start_state.mode_hi),
    )
    instances = encoding.controller_pool.instances
    count_cases: list[tuple[str, z3.BoolRef]] = []
    for count in range(0, len(instances) + 1):
        inside = [
            instance.activation_time < encoding.horizon_time
            for instance in instances[:count]
        ]
        outside = [
            instance.activation_time >= encoding.horizon_time
            for instance in instances[count:]
        ]
        count_cases.append((
            f"CTRL_COUNT_{count}",
            z3.And(*(inside + outside)) if (inside or outside) else z3.BoolVal(True),
        ))
    return tuple(
        (f"{mode_id}__{count_id}", z3.And(mode_formula, count_formula))
        for mode_id, mode_formula in mode_cases
        for count_id, count_formula in count_cases
    )


def _solve_unknown_by_exact_cases(
    assertions: tuple[z3.BoolRef, ...],
    encoding: IncrementalEventWindowEncoding,
    *,
    timeout_ms: int,
    progress: Callable[[dict[str, Any]], None] | None,
    depth: int,
) -> tuple[str, float, tuple[DepthCaseReceipt, ...], str | None]:
    """Retry one UNKNOWN depth by an exhaustive exact finite case split."""

    rows: list[DepthCaseReceipt] = []
    total = 0.0
    unknown_cases: list[str] = []
    for case_id, cube in _exact_case_partition(encoding):
        if progress is not None:
            progress({
                "depth": depth,
                "phase": "CASE_CHECK",
                "case_id": case_id,
            })
        result, elapsed, reason = _fresh_check(
            (*assertions, cube), timeout_ms=timeout_ms
        )
        total += elapsed
        rows.append(DepthCaseReceipt(case_id, result, elapsed, reason=reason))
        if progress is not None:
            progress({
                "depth": depth,
                "phase": "CASE_RESULT",
                "case_id": case_id,
                "case_result": result,
                "case_solver_check_seconds": round(elapsed, 6),
            })
        if result == "SAT":
            return "SAT", total, tuple(rows), None
        if result == "UNKNOWN":
            unknown_cases.append(case_id)

    if unknown_cases:
        return (
            "UNKNOWN",
            total,
            tuple(rows),
            "EXACT_CASE_PARTITION_UNKNOWN:" + ",".join(unknown_cases),
        )
    return "UNSAT", total, tuple(rows), None


def solve_incremental_event_window(
    obligation_id: str,
    encoding: IncrementalEventWindowEncoding,
    *,
    timeout_ms: int = 120_000,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[IncrementalBMCReceipt, EventWindowEncoding | None]:
    """Check every exact first-horizon depth with fresh whole-depth solvers.

    Sound aggregation rule:
      * SAT at any depth -> SAT (and materialize exactly that depth for the
        existing independent SAT classifier/replayer),
      * UNKNOWN after the exhaustive exact fallback partition -> UNKNOWN
        immediately; it can never be hidden by later checks,
      * UNSAT only after *all* depths 0..finite_event_bound are UNSAT.
    """

    timeout_ms = int(timeout_ms)
    if timeout_ms <= 0:
        raise ValueError("solver timeout must be positive")

    max_depth = int(encoding.event_bound.finite_event_bound)
    rows: list[IncrementalDepthReceipt] = []
    step_build_seconds = 0.0
    terminal_build_seconds = 0.0
    solver_check_seconds = 0.0
    plan_hash = _query_plan_hash(encoding)

    for depth in range(0, max_depth + 1):
        # Prefix construction remains incremental and exact, but solver state is
        # deliberately *not* shared across depths.  This lets Z3 preprocess the
        # complete small depth formula from scratch instead of staying in a
        # post-push/pop incremental context.
        if depth > 0:
            started = perf_counter()
            encoding.append_exact_event_step()
            step_build_seconds += perf_counter() - started

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
        if progress is not None:
            progress({
                "depth": depth,
                "max_depth": max_depth,
                "checked_depth_count": len(rows),
                "prefix_active_event_steps": encoding.depth,
                "phase": "FRESH_CHECK",
            })
        primary_result, primary_seconds, primary_reason = _fresh_check(
            assertions, timeout_ms=timeout_ms
        )
        one_check = primary_seconds
        effective_result = primary_result
        effective_reason = primary_reason
        case_rows: tuple[DepthCaseReceipt, ...] = ()

        if primary_result == "UNKNOWN":
            effective_result, case_seconds, case_rows, effective_reason = (
                _solve_unknown_by_exact_cases(
                    assertions,
                    encoding,
                    timeout_ms=timeout_ms,
                    progress=progress,
                    depth=depth,
                )
            )
            one_check += case_seconds

        solver_check_seconds += one_check
        if progress is not None:
            progress({
                "depth": depth,
                "max_depth": max_depth,
                "checked_depth_count": len(rows) + 1,
                "prefix_active_event_steps": encoding.depth,
                "phase": "RESULT",
                "depth_result": effective_result,
                "primary_result": primary_result,
                "case_partition_used": bool(case_rows),
                "depth_solver_check_seconds": round(one_check, 6),
                "total_solver_check_seconds": round(solver_check_seconds, 6),
            })

        row = IncrementalDepthReceipt(
            depth,
            effective_result,
            one_check,
            one_terminal_build,
            reason=effective_reason,
            primary_result=primary_result,
            case_receipts=case_rows,
        )
        rows.append(row)

        if effective_result == "SAT":
            decisive = encoding.materialize_depth(terminal)
            return IncrementalBMCReceipt(
                obligation_id=obligation_id,
                result="SAT",
                solver_version=z3.get_version_string(),
                timeout_ms=timeout_ms,
                strategy=SOLVER_STRATEGY,
                query_plan_hash=plan_hash,
                max_depth=max_depth,
                checked_depth_count=len(rows),
                decisive_depth=depth,
                step_build_seconds=step_build_seconds,
                terminal_build_seconds=terminal_build_seconds,
                solver_check_seconds=solver_check_seconds,
                depth_receipts=tuple(rows),
            ), decisive

        if effective_result != "UNSAT":
            return IncrementalBMCReceipt(
                obligation_id=obligation_id,
                result="UNKNOWN",
                solver_version=z3.get_version_string(),
                timeout_ms=timeout_ms,
                strategy=SOLVER_STRATEGY,
                query_plan_hash=plan_hash,
                max_depth=max_depth,
                checked_depth_count=len(rows),
                decisive_depth=depth,
                step_build_seconds=step_build_seconds,
                terminal_build_seconds=terminal_build_seconds,
                solver_check_seconds=solver_check_seconds,
                depth_receipts=tuple(rows),
                reason=effective_reason,
            ), None

        del terminal

    # This is the only path allowed to report window UNSAT: every exact
    # terminal depth in the structural finite bound has been checked UNSAT.
    return IncrementalBMCReceipt(
        obligation_id=obligation_id,
        result="UNSAT",
        solver_version=z3.get_version_string(),
        timeout_ms=timeout_ms,
        strategy=SOLVER_STRATEGY,
        query_plan_hash=plan_hash,
        max_depth=max_depth,
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
