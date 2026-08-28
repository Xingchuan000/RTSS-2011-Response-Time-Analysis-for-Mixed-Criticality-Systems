"""Exact incremental terminal-depth BMC for V9.2 Event FirstBadWindow.

This module changes only how the finite Event window is *queried*.  The
monolithic reference encoding allows an exact active Event prefix to reach the
query horizon and then pads the remaining bounded slots with exact terminal
stutters.  Here, each possible first-horizon depth ``k`` is checked directly:

    Base & ActiveStep[0] & ... & ActiveStep[k-1] & BadAtHorizon(state[k])

for every ``k`` from 0 through the already-proved finite Event bound.  Hence no
Event behavior, environment choice, P5 behavior or finite depth is removed or
added.  The terminal-stutter suffix is omitted because it is semantically
identity after the first horizon state.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import z3

from formal_toolchain.core.hashing import sha256_object

from .event_window_encoder import (
    ENCODER_VERSION,
    EventWindowEncoding,
    IncrementalEventWindowEncoding,
)


SOLVER_STRATEGY = "EXACT_INCREMENTAL_TERMINAL_DEPTH_BMC_V1"


@dataclass(frozen=True, slots=True)
class IncrementalDepthReceipt:
    depth: int
    result: str
    solver_check_seconds: float
    terminal_build_seconds: float
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "depth": int(self.depth),
            "result": self.result,
            "solver_check_seconds": round(float(self.solver_check_seconds), 6),
            "terminal_build_seconds": round(float(self.terminal_build_seconds), 6),
        }
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
        "schema": "v9_2_incremental_terminal_depth_query_plan_v1",
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
    })


def solve_incremental_event_window(
    obligation_id: str,
    encoding: IncrementalEventWindowEncoding,
    *,
    timeout_ms: int = 120_000,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[IncrementalBMCReceipt, EventWindowEncoding | None]:
    """Check every exact first-horizon depth with one incremental Z3 solver.

    Sound aggregation rule:
      * SAT at any depth -> SAT (and materialize exactly that depth for the
        existing independent SAT classifier/replayer),
      * UNKNOWN at any depth -> UNKNOWN immediately; it can never be hidden by
        later checks,
      * UNSAT only after *all* depths 0..finite_event_bound are UNSAT.
    """

    timeout_ms = int(timeout_ms)
    if timeout_ms <= 0:
        raise ValueError("solver timeout must be positive")

    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(encoding.base_formula)

    max_depth = int(encoding.event_bound.finite_event_bound)
    rows: list[IncrementalDepthReceipt] = []
    step_build_seconds = 0.0
    terminal_build_seconds = 0.0
    solver_check_seconds = 0.0
    plan_hash = _query_plan_hash(encoding)

    for depth in range(0, max_depth + 1):
        # Depth k contains exactly k active Event macros.  When moving from
        # k-1 to k, make that one additional exact active macro permanent in
        # the shared prefix solver.  No terminal stutter relation is added.
        if depth > 0:
            started = perf_counter()
            encoding.append_exact_event_step()
            solver.add(encoding.prefix_formulas[-1])
            step_build_seconds += perf_counter() - started

        terminal_started = perf_counter()
        terminal = encoding.build_terminal_bad_query()
        one_terminal_build = perf_counter() - terminal_started
        terminal_build_seconds += one_terminal_build

        solver.push()
        solver.add(terminal.formula)
        if progress is not None:
            progress({
                "depth": depth,
                "max_depth": max_depth,
                "checked_depth_count": len(rows),
                "prefix_active_event_steps": encoding.depth,
                "phase": "CHECK",
            })
        check_started = perf_counter()
        result = solver.check()
        one_check = perf_counter() - check_started
        solver_check_seconds += one_check
        if progress is not None:
            progress({
                "depth": depth,
                "max_depth": max_depth,
                "checked_depth_count": len(rows) + 1,
                "prefix_active_event_steps": encoding.depth,
                "phase": "RESULT",
                "depth_result": (
                    "SAT" if result == z3.sat else
                    "UNSAT" if result == z3.unsat else "UNKNOWN"
                ),
                "depth_solver_check_seconds": round(one_check, 6),
                "total_solver_check_seconds": round(solver_check_seconds, 6),
            })

        if result == z3.sat:
            rows.append(IncrementalDepthReceipt(
                depth, "SAT", one_check, one_terminal_build
            ))
            solver.pop()
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

        if result != z3.unsat:
            reason = solver.reason_unknown()
            rows.append(IncrementalDepthReceipt(
                depth, "UNKNOWN", one_check, one_terminal_build, reason=reason
            ))
            solver.pop()
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
                reason=reason,
            ), None

        rows.append(IncrementalDepthReceipt(
            depth, "UNSAT", one_check, one_terminal_build
        ))
        solver.pop()
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
    "IncrementalBMCReceipt",
    "IncrementalDepthReceipt",
    "solve_incremental_event_window",
]
