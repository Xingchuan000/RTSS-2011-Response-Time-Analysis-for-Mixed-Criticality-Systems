"""Exact terminal-depth BMC for V9.2 Event FirstBadWindow.

The Event prefix is still extended one exact active macro at a time, and every
terminal depth receives a *fresh* Z3 solver.  Solver state is never reused
across depths, preserving the whole-query preprocessing benefit established by
the s313 fresh-depth experiments.  Within one fixed depth, however, the solver
is reused across exact controller/source/member refinements so the expensive
base preprocessing and learned clauses are not discarded.

For depth ``k`` the queried formula is exactly

    Base & ActiveStep[0] & ... & ActiveStep[k-1] & BadAtHorizon(state[k])

for every ``k`` from 0 through the already-proved finite Event bound.  A small
redundant terminal bridge makes the already-implied final next-event time equal
to the horizon explicit.  Each whole-depth check first receives only a bounded probe.  A hard query is
then refined under exhaustive finite covers: controller activation count,
penultimate event-source class, and finally the exact source member (release
task, HI-deadline job, or selected completion slot).  Only exact leaves may
inherit the terminal unlimited-time policy.  These are finite Boolean covers,
not abstractions.
"""

from __future__ import annotations

from contextlib import contextmanager
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


SOLVER_STRATEGY = "EXACT_LINEAR_PERIODIC_RELEASE_TICK_BMC_V6"


@dataclass(frozen=True, slots=True)
class DepthCaseReceipt:
    case_id: str
    result: str
    solver_check_seconds: float
    reason: str | None = None
    solver_context_reused: bool = True
    resumed_after_timeout: bool = False

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "case_id": self.case_id,
            "result": self.result,
            "solver_check_seconds": round(float(self.solver_check_seconds), 6),
            "solver_context_reused": bool(self.solver_context_reused),
            "resumed_after_timeout": bool(self.resumed_after_timeout),
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
            row["case_partition_kind"] = "CONTROLLER_COUNT_THEN_SOURCE_CLASS_THEN_EXACT_MEMBER_THEN_RELEASE_TICK"
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
    probe_timeout_ms: int
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
            "probe_timeout_ms": int(self.probe_timeout_ms),
            "leaf_timeout_ms": int(self.timeout_ms),
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
            "within_depth_solver_context_reused": True,
            "cross_depth_solver_context_reused": False,
            "same_solver_timeout_resume": True,
            "disjoint_exact_case_partition": True,
            "bounded_probe_before_exact_partition": True,
            "hierarchical_source_member_partition": True,
            "hierarchical_release_tick_partition": True,
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
        "schema": "v9_2_linear_periodic_release_tick_query_plan_v6",
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
        "within_depth_solver_context_reused": True,
        "cross_depth_solver_context_reused": False,
        "same_solver_timeout_resume": True,
        "unknown_fallback_partition": "controller_count_then_disjoint_source_class_then_disjoint_exact_source_member_then_release_tick",
        "coarse_queries_use_bounded_probe": True,
        "only_exact_leaf_cases_may_use_unlimited_timeout": True,
    })


def _solver_check(
    solver: z3.Solver, *, timeout_ms: int
) -> tuple[str, float, str | None]:
    """Check one already-built exact solver context.

    A depth owns exactly one solver.  Bounded parent probes and their refined
    child cases reuse that solver so preprocessing and learned clauses are not
    discarded.  ``timeout_ms == 0`` explicitly clears a prior bounded timeout
    on the same solver before an exact leaf resumes without a deadline.
    """

    timeout_ms = int(timeout_ms)
    if timeout_ms < 0:
        raise ValueError("solver timeout must be non-negative (0 means unlimited)")
    solver.set(timeout=timeout_ms)
    started = perf_counter()
    result = solver.check()
    elapsed = perf_counter() - started
    if result == z3.sat:
        return "SAT", elapsed, None
    if result == z3.unsat:
        return "UNSAT", elapsed, None
    return "UNKNOWN", elapsed, solver.reason_unknown()


def _new_depth_solver(assertions: Iterable[z3.BoolRef]) -> z3.Solver:
    """Create the one fresh solver owned by a single terminal depth."""

    solver = z3.Solver()
    solver.add(*tuple(assertions))
    return solver


@contextmanager
def _solver_scope(solver: z3.Solver, *extra: z3.BoolRef):
    """Temporarily refine one depth solver without losing parent learning."""

    solver.push()
    try:
        if extra:
            solver.add(*extra)
        yield
    finally:
        solver.pop()


def _ordered_disjoint_cover(
    rows: Iterable[tuple[str, z3.BoolRef]],
) -> tuple[tuple[str, z3.BoolRef], ...]:
    """Turn an overlapping finite cover into an equivalent disjoint cover.

    Simultaneous Event sources are legal, so the older source/member cases can
    overlap.  An ordered first-witness partition preserves the union exactly
    while ensuring a concrete model is solved in only one sibling case.
    """

    out: list[tuple[str, z3.BoolRef]] = []
    previous: list[z3.BoolRef] = []
    for case_id, cube in rows:
        if previous:
            disjoint = z3.And(cube, z3.Not(z3.Or(*previous)))
        else:
            disjoint = cube
        out.append((case_id, disjoint))
        previous.append(cube)
    return tuple(out)

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
    return _ordered_disjoint_cover((
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
    ))



def _safe_case_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _source_member_partition(
    encoding: IncrementalEventWindowEncoding,
    source_id: str,
) -> tuple[tuple[str, z3.BoolRef], ...]:
    """Refine a hard source class into an exact finite member cover.

    The parent source cube is always conjoined by the caller.  Release and
    HI-deadline classes are split by their exact candidate member.  Completion
    is split by the selected dispatch slot; a catch-all slot cube is retained
    so the cover remains exact even if a future frontier encoding widens the
    selected-slot domain.
    """

    if len(encoding.event_steps) < 2:
        return ()
    step = encoding.event_steps[-2]
    candidates = step.candidates
    nxt = candidates.next_time

    if source_id == "SRC_RELEASE_ANY":
        return _ordered_disjoint_cover(
            (
                f"SRC_RELEASE_TASK_{_safe_case_token(task_name)}",
                nxt == value,
            )
            for task_name, value in candidates.releases
        )

    if source_id == "SRC_HI_DEADLINE_ANY":
        return _ordered_disjoint_cover(
            (
                f"SRC_HI_DEADLINE_JOB_{_safe_case_token(task_name)}_{slot}",
                nxt == value,
            )
            for (task_name, slot), value in candidates.hi_deadlines
        )

    if source_id == "SRC_COMPLETION":
        dispatch_state = step.phase_states[-1]
        known: list[z3.BoolRef] = []
        rows: list[tuple[str, z3.BoolRef]] = []
        for task_index, task in enumerate(encoding.model.tasks):
            for slot in range(encoding.model.max_jobs_per_task):
                selected_index = task_index * encoding.model.max_jobs_per_task + slot
                cube = dispatch_state.frontier.selected_slot == selected_index
                known.append(cube)
                rows.append((
                    f"SRC_COMPLETION_SLOT_{_safe_case_token(task.name)}_{slot}",
                    cube,
                ))
        rows.append((
            "SRC_COMPLETION_SLOT_OTHER",
            z3.Not(z3.Or(*known)) if known else z3.BoolVal(True),
        ))
        return _ordered_disjoint_cover(rows)

    return ()


def _release_tick_subpartition(
    encoding: IncrementalEventWindowEncoding,
    member_id: str,
) -> tuple[tuple[str, z3.BoolRef], ...]:
    """Exact finite refinement of one periodic-release task by relative tick.

    ``SRC_RELEASE_TASK_*`` still leaves the absolute periodic phase symbolic.
    For a target-relative first-bad window, however, every admissible release
    timestamp belongs to the finite environment tick domain.  Splitting by
    ``next_time == origin + tick`` removes that hidden phase disjunction before
    any unlimited solve.

    A final OTHER cube makes the partition tautologically exhaustive even if a
    future environment-domain implementation changes.  Therefore this helper
    is a solver decomposition only and cannot remove a concrete model.
    """

    if len(encoding.event_steps) < 2:
        return ()
    step = encoding.event_steps[-2]
    candidates = step.candidates
    nxt = candidates.next_time
    origin = encoding.environment.phase.origin_time

    task_name: str | None = None
    for candidate_task, _ in candidates.releases:
        expected = f"SRC_RELEASE_TASK_{_safe_case_token(candidate_task)}"
        if member_id == expected:
            task_name = candidate_task
            break
    if task_name is None:
        return ()

    ticks = tuple(int(tick) for tick in encoding.environment.allowed_ticks_by_task.get(task_name, ()))
    raw_rows: list[tuple[str, z3.BoolRef]] = []
    raw_cubes: list[z3.BoolRef] = []
    for tick in ticks:
        cube = nxt == origin + tick
        raw_cubes.append(cube)
        raw_rows.append((f"SRC_RELEASE_TICK_{tick}", cube))
    other = z3.Not(z3.Or(*raw_cubes)) if raw_cubes else z3.BoolVal(True)
    raw_rows.append(("SRC_RELEASE_TICK_OTHER", other))
    return _ordered_disjoint_cover(raw_rows)

def _solve_unknown_by_exact_cases(
    solver: z3.Solver,
    encoding: IncrementalEventWindowEncoding,
    *,
    probe_timeout_ms: int,
    leaf_timeout_ms: int,
    progress: Callable[[dict[str, Any]], None] | None,
    depth: int,
) -> tuple[str, float, tuple[DepthCaseReceipt, ...], str | None]:
    """Refine one hard depth while reusing its single Z3 solver context.

    The solver already contains the exact depth formula and has just completed
    the whole-depth bounded probe.  All count/source/member refinements are
    temporary scopes on that same solver, so the expensive base preprocessing
    and learned clauses are retained.  No solver object is shared across
    different depths.

    Exact leaves use a bounded probe first.  If that probe times out, the same
    solver in the same pushed scope has its timeout cleared and is checked again
    unlimited; the first 120 seconds are therefore not thrown away.
    """

    rows: list[DepthCaseReceipt] = []
    total = 0.0
    unresolved: list[str] = []
    source_partition = _penultimate_event_source_partition(encoding)

    def run_current(
        case_id: str,
        *,
        timeout_ms: int,
        phase: str,
        resumed_after_timeout: bool = False,
    ) -> tuple[str, str | None]:
        nonlocal total
        if progress is not None:
            progress({
                "depth": depth,
                "phase": phase,
                "case_id": case_id,
                "case_timeout_ms": int(timeout_ms),
                "within_depth_solver_context_reused": True,
                "resumed_after_timeout": bool(resumed_after_timeout),
            })
        result, elapsed, reason = _solver_check(solver, timeout_ms=timeout_ms)
        total += elapsed
        rows.append(DepthCaseReceipt(
            case_id, result, elapsed, reason=reason,
            solver_context_reused=True,
            resumed_after_timeout=resumed_after_timeout,
        ))
        if progress is not None:
            progress({
                "depth": depth,
                "phase": phase.replace("_CHECK", "_RESULT"),
                "case_id": case_id,
                "case_result": result,
                "case_timeout_ms": int(timeout_ms),
                "case_solver_check_seconds": round(elapsed, 6),
                "within_depth_solver_context_reused": True,
                "resumed_after_timeout": bool(resumed_after_timeout),
            })
        return result, reason

    def resume_exact_leaf(case_id: str) -> tuple[str, str | None]:
        """Continue an already-timed-out exact leaf in the same solver context."""

        return run_current(
            f"{case_id}__RESUME",
            timeout_ms=leaf_timeout_ms,
            phase="EXACT_LEAF_RESUME_CHECK",
            resumed_after_timeout=True,
        )

    def probe_then_resume_exact_leaf(case_id: str) -> tuple[str, str | None]:
        """Probe a new exact leaf once, then continue that same context."""

        probe_id = f"{case_id}__PROBE"
        result, reason = run_current(
            probe_id, timeout_ms=probe_timeout_ms, phase="LEAF_PROBE_CHECK"
        )
        if result != "UNKNOWN":
            return result, reason
        return resume_exact_leaf(case_id)

    for count_id, count_cube in _controller_count_partition(encoding):
        with _solver_scope(solver, count_cube):
            count_result, _ = run_current(
                count_id,
                timeout_ms=probe_timeout_ms,
                phase="COUNT_PROBE_CHECK",
            )
            if count_result == "SAT":
                return "SAT", total, tuple(rows), None
            if count_result == "UNSAT":
                continue

            # Depth 0/1 has no nonterminal penultimate macro.  The count cube is
            # already exact, so resume this same context instead of restarting.
            if not source_partition:
                leaf_result, leaf_reason = resume_exact_leaf(
                    f"{count_id}__EXACT_LEAF"
                )
                if leaf_result == "SAT":
                    return "SAT", total, tuple(rows), None
                if leaf_result == "UNKNOWN":
                    unresolved.append(
                        count_id + (f":{leaf_reason}" if leaf_reason else "")
                    )
                continue

            count_unresolved = False
            for source_id, source_cube in source_partition:
                source_case_id = f"{count_id}__{source_id}"
                with _solver_scope(solver, source_cube):
                    source_result, _ = run_current(
                        source_case_id,
                        timeout_ms=probe_timeout_ms,
                        phase="SOURCE_PROBE_CHECK",
                    )
                    if source_result == "SAT":
                        return "SAT", total, tuple(rows), None
                    if source_result == "UNSAT":
                        continue

                    members = _source_member_partition(encoding, source_id)
                    if not members:
                        # HORIZON/CONTROLLER are exact leaves.  Continue the
                        # timed-out source query in this very solver context.
                        leaf_result, leaf_reason = resume_exact_leaf(
                            f"{source_case_id}__EXACT_LEAF"
                        )
                        if leaf_result == "SAT":
                            return "SAT", total, tuple(rows), None
                        if leaf_result == "UNKNOWN":
                            unresolved.append(
                                source_case_id
                                + (f":{leaf_reason}" if leaf_reason else "")
                            )
                            count_unresolved = True
                        continue

                    member_unknown = False
                    for member_id, member_cube in members:
                        member_case_id = f"{source_case_id}__{member_id}"
                        with _solver_scope(solver, member_cube):
                            release_ticks = _release_tick_subpartition(encoding, member_id)
                            if release_ticks:
                                member_result, member_reason = run_current(
                                    f"{member_case_id}__PROBE",
                                    timeout_ms=probe_timeout_ms,
                                    phase="MEMBER_PROBE_CHECK",
                                )
                                if member_result == "SAT":
                                    return "SAT", total, tuple(rows), None
                                if member_result == "UNSAT":
                                    continue

                                tick_unknown = False
                                for tick_id, tick_cube in release_ticks:
                                    tick_case_id = f"{member_case_id}__{tick_id}"
                                    with _solver_scope(solver, tick_cube):
                                        leaf_result, leaf_reason = probe_then_resume_exact_leaf(
                                            f"{tick_case_id}__EXACT_LEAF"
                                        )
                                        if leaf_result == "SAT":
                                            return "SAT", total, tuple(rows), None
                                        if leaf_result == "UNKNOWN":
                                            tick_unknown = True
                                            unresolved.append(
                                                tick_case_id
                                                + (f":{leaf_reason}" if leaf_reason else "")
                                            )
                                if tick_unknown:
                                    member_unknown = True
                                continue

                            leaf_result, leaf_reason = probe_then_resume_exact_leaf(
                                f"{member_case_id}__EXACT_LEAF"
                            )
                            if leaf_result == "SAT":
                                return "SAT", total, tuple(rows), None
                            if leaf_result == "UNKNOWN":
                                member_unknown = True
                                unresolved.append(
                                    member_case_id
                                    + (f":{leaf_reason}" if leaf_reason else "")
                                )

                    if member_unknown:
                        count_unresolved = True
                    # Otherwise every disjoint exact member/tick leaf is UNSAT.

            if count_unresolved:
                continue
            # Otherwise every disjoint exact source class is UNSAT.

    if unresolved:
        return (
            "UNKNOWN",
            total,
            tuple(rows),
            "EXACT_HIERARCHICAL_REUSED_LEAF_UNKNOWN:" + ",".join(unresolved),
        )
    return "UNSAT", total, tuple(rows), None

def solve_incremental_event_window(
    obligation_id: str,
    encoding: IncrementalEventWindowEncoding,
    *,
    timeout_ms: int = 120_000,
    probe_timeout_ms: int = 120_000,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[IncrementalBMCReceipt, EventWindowEncoding | None]:
    """Check every exact first-horizon depth with bounded probes and exact leaves.

    Sound aggregation rule:
      * SAT at any depth -> SAT (and materialize exactly that depth for the
        existing independent SAT classifier/replayer),
      * UNKNOWN after the exhaustive exact fallback partition -> UNKNOWN
        immediately; it can never be hidden by later checks,
      * UNSAT only after *all* depths 0..finite_event_bound are UNSAT.
    """

    timeout_ms = int(timeout_ms)
    probe_timeout_ms = int(probe_timeout_ms)
    if timeout_ms < 0:
        raise ValueError("solver timeout must be non-negative (0 means unlimited)")
    if probe_timeout_ms <= 0:
        raise ValueError("probe timeout must be positive so coarse queries cannot run unlimited")

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
                "probe_timeout_ms": probe_timeout_ms,
                "leaf_timeout_ms": timeout_ms,
            })
        # The whole-depth query is only a bounded probe.  Unlimited solving is
        # reserved for exact hierarchical leaves after controller/source/member
        # partitioning, otherwise the fallback can never be reached.
        depth_solver = _new_depth_solver(assertions)
        primary_result, primary_seconds, primary_reason = _solver_check(
            depth_solver, timeout_ms=probe_timeout_ms
        )
        one_check = primary_seconds
        effective_result = primary_result
        effective_reason = primary_reason
        case_rows: tuple[DepthCaseReceipt, ...] = ()

        if primary_result == "UNKNOWN":
            effective_result, case_seconds, case_rows, effective_reason = (
                _solve_unknown_by_exact_cases(
                    depth_solver,
                    encoding,
                    probe_timeout_ms=probe_timeout_ms,
                    leaf_timeout_ms=timeout_ms,
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
                probe_timeout_ms=probe_timeout_ms,
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
                probe_timeout_ms=probe_timeout_ms,
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

        del depth_solver
        del terminal

    # This is the only path allowed to report window UNSAT: every exact
    # terminal depth in the structural finite bound has been checked UNSAT.
    return IncrementalBMCReceipt(
        obligation_id=obligation_id,
        result="UNSAT",
        solver_version=z3.get_version_string(),
        timeout_ms=timeout_ms,
        probe_timeout_ms=probe_timeout_ms,
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
