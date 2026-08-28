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
formula is retried under an exhaustive finite case partition over the number
of controller activations strictly before the horizon and, only for still-hard
count cases, by the exact event-source class of the penultimate
active macro.  These are exhaustive finite Boolean covers, not abstractions.
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


SOLVER_STRATEGY = "EXACT_FRESH_DEPTH_SCALAR_MIN_SOURCE_SPLIT_BMC_V3"


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
            row["case_partition_kind"] = "CONTROLLER_COUNT_THEN_PENULTIMATE_EVENT_SOURCE"
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
        "schema": "v9_2_fresh_scalar_min_source_split_query_plan_v3",
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
        "unknown_fallback_partition": "controller_count_then_penultimate_event_source_class",
    })


def _fresh_check(
    assertions: Iterable[z3.BoolRef], *, timeout_ms: int
) -> tuple[str, float, str | None]:
    """Solve one exact depth in a fresh non-incremental solver context."""

    solver = z3.Solver()
    # Z3 has no wall-clock deadline when the timeout parameter is omitted.
    # ``timeout_ms == 0`` is the trusted verifier's explicit unlimited mode
    # for final FirstBadEventWindow solving; negative values remain invalid.
    if int(timeout_ms) > 0:
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


def _controller_count_partition(
    encoding: IncrementalEventWindowEncoding,
) -> tuple[tuple[str, z3.BoolRef], ...]:
    """Exact partition by the controller activations strictly before horizon.

    Controller instances are ordered by one positive agent period, hence the
    in-window instances always form a prefix.  Unlike V2, start mode is not
    duplicated here because the s313 receipts showed that it did not isolate
    the hard arithmetic branch.
    """

    instances = encoding.controller_pool.instances
    rows: list[tuple[str, z3.BoolRef]] = []
    for count in range(0, len(instances) + 1):
        inside = [
            instance.activation_time < encoding.horizon_time
            for instance in instances[:count]
        ]
        outside = [
            instance.activation_time >= encoding.horizon_time
            for instance in instances[count:]
        ]
        rows.append((
            f"CTRL_COUNT_{count}",
            z3.And(*(inside + outside)) if (inside or outside) else z3.BoolVal(True),
        ))
    return tuple(rows)


def _penultimate_event_source_partition(
    encoding: IncrementalEventWindowEncoding,
) -> tuple[tuple[str, z3.BoolRef], ...]:
    """Exact cover of the penultimate active macro's next-event witness.

    At terminal depth >= 2, the last active macro reaches the query horizon,
    while the preceding macro reaches a boundary strictly before the horizon.
    The exact Event minimum definition guarantees that its ``next_time`` equals
    at least one closed-world candidate source.  Grouping those witnesses by
    source class is therefore an exhaustive (possibly overlapping) cover.
    Overlap on simultaneous events is harmless for UNSAT aggregation.
    """

    if len(encoding.event_steps) < 2:
        return ()
    step = encoding.event_steps[-2]
    candidates = step.candidates
    nxt = candidates.next_time
    release_values = [value for _, value in candidates.releases]
    deadline_values = [value for _, value in candidates.hi_deadlines]
    return (
        ("SRC_HORIZON", nxt == candidates.horizon),
        (
            "SRC_RELEASE_ANY",
            z3.Or(*(nxt == value for value in release_values))
            if release_values else z3.BoolVal(False),
        ),
        ("SRC_CONTROLLER", nxt == candidates.controller),
        (
            "SRC_HI_DEADLINE_ANY",
            z3.Or(*(nxt == value for value in deadline_values))
            if deadline_values else z3.BoolVal(False),
        ),
        ("SRC_COMPLETION", nxt == candidates.completion),
    )


def _solve_unknown_by_exact_cases(
    assertions: tuple[z3.BoolRef, ...],
    encoding: IncrementalEventWindowEncoding,
    *,
    timeout_ms: int,
    progress: Callable[[dict[str, Any]], None] | None,
    depth: int,
) -> tuple[str, float, tuple[DepthCaseReceipt, ...], str | None]:
    """Retry UNKNOWN via exact controller-count then event-source covers.

    A controller-count case that is still UNKNOWN is refined by the
    penultimate active macro's source class.  The source-class disjunction is
    already implied by the exact minimum relation.  SAT in any leaf is SAT; a
    count case is UNSAT only when every source class is UNSAT; any unresolved
    leaf keeps the depth UNKNOWN.
    """

    rows: list[DepthCaseReceipt] = []
    total = 0.0
    unresolved: list[str] = []
    source_partition = _penultimate_event_source_partition(encoding)

    for count_id, count_cube in _controller_count_partition(encoding):
        if progress is not None:
            progress({"depth": depth, "phase": "CASE_CHECK", "case_id": count_id})
        result, elapsed, reason = _fresh_check(
            (*assertions, count_cube), timeout_ms=timeout_ms
        )
        total += elapsed
        rows.append(DepthCaseReceipt(count_id, result, elapsed, reason=reason))
        if progress is not None:
            progress({
                "depth": depth, "phase": "CASE_RESULT", "case_id": count_id,
                "case_result": result, "case_solver_check_seconds": round(elapsed, 6),
            })
        if result == "SAT":
            return "SAT", total, tuple(rows), None
        if result == "UNSAT":
            continue

        # Depth 0/1 has no nonterminal penultimate macro to split.  Preserve
        # UNKNOWN rather than inventing a non-exhaustive fallback.
        if not source_partition:
            unresolved.append(count_id)
            continue

        count_unknown: list[str] = []
        for source_id, source_cube in source_partition:
            case_id = f"{count_id}__{source_id}"
            if progress is not None:
                progress({"depth": depth, "phase": "SOURCE_CASE_CHECK", "case_id": case_id})
            leaf_result, leaf_elapsed, leaf_reason = _fresh_check(
                (*assertions, count_cube, source_cube), timeout_ms=timeout_ms
            )
            total += leaf_elapsed
            rows.append(DepthCaseReceipt(
                case_id, leaf_result, leaf_elapsed, reason=leaf_reason
            ))
            if progress is not None:
                progress({
                    "depth": depth, "phase": "SOURCE_CASE_RESULT", "case_id": case_id,
                    "case_result": leaf_result,
                    "case_solver_check_seconds": round(leaf_elapsed, 6),
                })
            if leaf_result == "SAT":
                return "SAT", total, tuple(rows), None
            if leaf_result == "UNKNOWN":
                count_unknown.append(case_id)

        if count_unknown:
            unresolved.extend(count_unknown)
        # Otherwise all exact source classes were UNSAT, hence this controller
        # count case is UNSAT even though its unsplit parent timed out.

    if unresolved:
        return (
            "UNKNOWN",
            total,
            tuple(rows),
            "EXACT_SOURCE_PARTITION_UNKNOWN:" + ",".join(unresolved),
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
    if timeout_ms < 0:
        raise ValueError("solver timeout must be non-negative (0 means unlimited)")

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
