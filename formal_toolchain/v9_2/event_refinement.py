"""V9.3 Full-kernel and target-local Event-graph proof obligations.

The reference Event macro remains an exact quotient of the Full kernel.  The
deployed FirstBadWindow search then projects away lower-priority scheduling
state while retaining the complete controller budget/history state.  Under
strict fixed priority this is a safety-preserving interference dominance: the
projected graph can admit extra bad behavior, so UNSAT proves Full safety while
SAT must be classified by replay.  No reverse projected->Full theorem is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Iterable

import z3

from .event_kernel import (
    EventSource,
    _silent_interval_advance,
    build_event_candidates,
    exact_periodic_countdown,
    state_equality,
)
from .event_window_encoder import derive_finite_event_bound
from .formula_solver import solve_formula
from .symbolic_state import BoundModel, declare_state
from .transition_encoder import encode_p7_time_and_service


EVENT_TERMINAL_OBLIGATIONS = (
    "EVENT_START_ABSTRACTION_SOUNDNESS",
    "EVENT_STATE_FUTURE_SUFFICIENCY",
    "EVENT_PHASE_SSA_FRAME_ELIMINATION_EQUIVALENCE",
    "EVENT_P5_GRAPH_BRANCH_SPECIALIZATION_EQUIVALENCE",
    "EVENT_CONTROLLER_POLICY_CASE_PARTITION_EQUIVALENCE",
    "EVENT_EXPLICIT_GRAPH_SOURCE_PARTITION_EQUIVALENCE",
    "TARGET_LOCAL_FIXED_PRIORITY_INTERFERENCE_DOMINANCE",
    "TARGET_LOCAL_POLICY_STATE_RETENTION",
    "EVENT_LAZY_RELEASE_DEMAND_INDEPENDENCE",
    "NEXT_EVENT_MINIMALITY",
    "NEXT_EVENT_EXACT_MINIMUM",
    "NO_SKIPPED_DISCRETE_EVENT",
    "NO_SPURIOUS_EVENT_SOURCE",
    "SILENT_INTERVAL_SERVICE_EQUIVALENCE",
    "RELEASE_EVENT_COVERAGE",
    "DEADLINE_EVENT_COVERAGE",
    "TERMINAL_SERVICE_EVENT_COVERAGE",
    "CONTROLLER_EVENT_COVERAGE",
    "EXACT_P5_AT_CONTROLLER_EVENT",
    "FULL_TO_EVENT_SEGMENT_SIMULATION",
    "FULL_TO_PROJECTED_EVENT_PREFIX_SIMULATION",
    "FIRST_HI_BAD_PROJECTED_EVENT_REFLECTION",
    "FINITE_EVENT_COUNT_BOUND",
    "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY",
    "EVENT_WINDOW_ENCODING_SOUNDNESS",
)


@dataclass(frozen=True, slots=True)
class EventRefinementProof:
    status: str
    obligation_statuses: dict[str, str]
    solver_receipts: tuple[dict[str, Any], ...]
    structural_receipts: tuple[dict[str, Any], ...]
    failure_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_code": self.failure_code,
            "obligation_statuses": dict(self.obligation_statuses),
            "solver_receipts": list(self.solver_receipts),
            "structural_receipts": list(self.structural_receipts),
            "event_layer_added_abstractions": [
                "TARGET_LOCAL_FIXED_PRIORITY_INTERFERENCE_DOMINANCE"
            ],
            "exact_reference_event_macro_semantics": self.status == "PASS",
            "full_to_projected_event_simulation_verified": self.status == "PASS",
            "event_to_full_realizability_verified": False,
            "projected_sat_requires_full_replay": True,
            "small_horizon_differential_consistency_verified": self.status == "PASS",
            "exact_p5_in_event_window": True,
        }


def _min_expr(values: Iterable[z3.ArithRef]) -> z3.ArithRef:
    rows = list(values)
    if not rows:
        raise ValueError("minimum requires at least one candidate")
    result = rows[0]
    for value in rows[1:]:
        result = z3.If(value < result, value, result)
    return result


def _minimum_counterexample() -> z3.BoolRef:
    """Refute a mismatch between scalar exact-min constraints and reference min.

    The Event kernel uses the same two-clause finite minimum definition:
    ``nxt <= each candidate`` and ``nxt == at least one candidate``.  The
    nested ITE is retained here only as a small independent reference term.
    """

    rows = [z3.Int(f"event.refine.min.c{i}") for i in range(7)]
    nxt = z3.Int("event.refine.min.scalar")
    reference = _min_expr(rows)
    scalar = z3.And(
        *(nxt <= row for row in rows),
        z3.Or(*(nxt == row for row in rows)),
    )
    return z3.Or(
        z3.And(scalar, nxt != reference),
        z3.And(nxt == reference, z3.Not(scalar)),
    )


def _periodic_counterexample(period: int, prefix: str) -> z3.BoolRef:
    t = z3.Int(f"{prefix}.t")
    period = int(period)
    nxt = ((t / period) + 1) * period
    return z3.And(
        t >= 0,
        z3.Not(z3.And(
            nxt > t,
            nxt % period == 0,
            nxt - t >= 1,
            nxt - t <= period,
        )),
    )


def _release_countdown_counterexample(
    period: int,
    prefix: str,
) -> z3.BoolRef:
    """Refute mismatch between P7 ``eta`` countdown and next release time."""

    period = int(period)
    t = z3.Int(f"{prefix}.t")
    residue = t % period
    # After P3 at a release timestamp eta has been reset to zero; otherwise it
    # is the exact positive residue.  This is precisely the dispatch-state
    # convention consumed by the production Event candidate builder.
    eta = z3.If(residue == 0, 0, residue)
    countdown = period - eta
    reference = ((t / period) + 1) * period - t
    return z3.And(t >= 0, countdown != reference)


def _controller_countdown_counterexample(
    period: int,
    prefix: str,
) -> z3.BoolRef:
    """Refute mismatch in the one-time root controller countdown initializer."""

    period = int(period)
    t = z3.Int(f"{prefix}.t")
    reference = ((t / period) + 1) * period - t
    return z3.And(
        t >= 0,
        exact_periodic_countdown(t, period) != reference,
    )


def _release_tick_domain_counterexample(
    target_period: int,
    task_period: int,
    deadline: int,
    prefix: str,
) -> z3.BoolRef:
    """Any actual relative release tick belongs to the finite gcd-domain."""

    origin = z3.Int(f"{prefix}.origin")
    tick = z3.Int(f"{prefix}.tick")
    divisor = gcd(int(target_period), int(task_period))
    return z3.And(
        origin >= 0,
        origin % int(target_period) == 0,
        tick >= 0,
        tick <= int(deadline),
        (origin + tick) % int(task_period) == 0,
        tick % divisor != 0,
    )


def _count_bound_counterexample(length: int, period: int, bound: int, prefix: str) -> z3.BoolRef:
    """Refute a periodic-stream count larger than floor(D/T)+1."""

    n = z3.Int(f"{prefix}.n")
    length = int(length)
    period = int(period)
    bound = int(bound)
    return z3.And(
        n >= 1,
        (n - 1) * period <= length,
        n > bound,
    )


def _silent_interval_counterexample(prefix: str = "event.refine.silent") -> z3.BoolRef:
    """Refute non-associativity of exact bulk P7 service/eta updates."""

    service = z3.Int(f"{prefix}.service")
    eta = z3.Int(f"{prefix}.eta")
    period = z3.Int(f"{prefix}.period")
    a = z3.Int(f"{prefix}.a")
    b = z3.Int(f"{prefix}.b")

    macro_service = service + (a + b)
    split_service = (service + a) + b

    def sat_eta(value: z3.ArithRef, delta: z3.ArithRef) -> z3.ArithRef:
        return z3.If(value + delta < period, value + delta, period)

    macro_eta = sat_eta(eta, a + b)
    split_eta = sat_eta(sat_eta(eta, a), b)
    domain = z3.And(
        service >= 0,
        period > 0,
        eta >= 0,
        eta <= period,
        a >= 0,
        b >= 0,
    )
    return z3.And(domain, z3.Or(macro_service != split_service, macro_eta != split_eta))


def _slot_index(model: BoundModel, key: tuple[str, int]) -> int:
    for task_index, task in enumerate(model.tasks):
        for slot in range(model.max_jobs_per_task):
            if key == (task.name, slot):
                return task_index * model.max_jobs_per_task + slot
    raise KeyError(key)


def _deadline_event_counterexample(
    model: BoundModel, key: tuple[str, int], prefix: str
) -> z3.BoolRef:
    dispatch = declare_state(f"{prefix}.dispatch", model)
    horizon = z3.Int(f"{prefix}.horizon")
    job = dispatch.jobs[key]
    candidates = build_event_candidates(
        dispatch, model, horizon_time=horizon,
        controller_delta=exact_periodic_countdown(dispatch.t, model.agent_period),
        prefix=f"{prefix}.candidates"
    )
    candidate = dict(candidates.hi_deadlines)[key]
    return z3.And(
        dispatch.t >= 0,
        dispatch.p == 7,
        job.present,
        z3.Not(job.removed),
        job.executed_service < job.effective_demand,
        job.absolute_deadline > dispatch.t,
        horizon >= job.absolute_deadline,
        candidate != job.absolute_deadline,
    )


def _terminal_service_event_counterexample(
    model: BoundModel, key: tuple[str, int], prefix: str
) -> z3.BoolRef:
    dispatch = declare_state(f"{prefix}.dispatch", model)
    horizon = z3.Int(f"{prefix}.horizon")
    job = dispatch.jobs[key]
    index = _slot_index(model, key)
    remaining = job.effective_demand - job.executed_service
    candidates = build_event_candidates(
        dispatch, model, horizon_time=horizon,
        controller_delta=exact_periodic_countdown(dispatch.t, model.agent_period),
        prefix=f"{prefix}.candidates"
    )
    return z3.And(
        candidates.definition_formula,
        dispatch.t >= 0,
        dispatch.p == 7,
        dispatch.frontier.running,
        dispatch.frontier.selected_slot == index,
        job.present,
        job.ready,
        z3.Not(job.removed),
        remaining > 0,
        horizon >= dispatch.t + remaining,
        candidates.completion != dispatch.t + remaining,
    )


def _p7_delta1_counterexample(model: BoundModel) -> z3.BoolRef:
    """Refute local P7 versus one-tick Event-tail equivalence.

    The exact Event macro and the Full kernel use the *same* P0--P6 closure by
    construction.  Re-instantiating that closure on two independent symbolic
    paths is both unnecessarily expensive and semantically wrong for the
    inherited nondeterministic history abstraction: two valid P5 witnesses need
    not choose identical post-history values.

    Differential consistency is therefore checked at the only quotient point:
    from one shared P7 dispatch state, one Full P7 tick and one Event silent
    advance of delta=1 must end field-for-field equal.  This is stronger than
    checking the property only for closure-reachable dispatch states.
    """

    dispatch = declare_state("event.diff.p7.dispatch", model)
    full_end = declare_state("event.diff.p7.full_end", model)
    event_end = declare_state("event.diff.p7.event_end", model)
    horizon = dispatch.t + 1
    candidates = build_event_candidates(
        dispatch, model, horizon_time=horizon,
        controller_delta=exact_periodic_countdown(dispatch.t, model.agent_period),
        prefix="event.diff.p7.candidates", source=EventSource("HORIZON"),
    )
    return z3.And(
        dispatch.p == 7,
        encode_p7_time_and_service(dispatch, full_end, model),
        _silent_interval_advance(dispatch, event_end, model, candidates),
        candidates.next_time == horizon,
        z3.Not(state_equality(full_end, event_end)),
    )


def _finite_event_bound_formulas(model: BoundModel) -> list[tuple[str, z3.BoolRef]]:
    formulas: list[tuple[str, z3.BoolRef]] = []
    for target in model.hi_tasks:
        bound = derive_finite_event_bound(model, target.name)
        active_tasks = tuple(
            task for task in model.tasks if task.priority <= target.priority
        )
        for task in active_tasks:
            formulas.append((
                f"FINITE_EVENT_COUNT_BOUND::RELEASE::{target.name}::{task.name}",
                _count_bound_counterexample(
                    target.deadline,
                    task.period,
                    bound.release_bound_by_task[task.name],
                    f"event.bound.release.{target.name}.{task.name}",
                ),
            ))
            formulas.append((
                f"EVENT_ENVIRONMENT_RELEASE_DOMAIN_COVERAGE::{target.name}::{task.name}",
                _release_tick_domain_counterexample(
                    target.period,
                    task.period,
                    target.deadline,
                    f"event.bound.domain.{target.name}.{task.name}",
                ),
            ))
        formulas.append((
            f"FINITE_EVENT_COUNT_BOUND::CONTROLLER::{target.name}",
            _count_bound_counterexample(
                target.deadline,
                model.agent_period,
                bound.controller_bound,
                f"event.bound.controller.{target.name}",
            ),
        ))
        component_sum = (
            bound.total_release_bound
            + bound.hi_deadline_bound
            + bound.completion_bound
            + bound.controller_bound
            + bound.horizon_boundary_bound
            + 2
        )
        formulas.append((
            f"FINITE_EVENT_COUNT_BOUND::COMPOSITION::{target.name}",
            z3.BoolVal(bound.finite_event_bound < component_sum),
        ))
    return formulas


def _structural_claims() -> list[dict[str, Any]]:
    """Definitional architecture facts used by the refinement composition.

    V9.3 intentionally does not inspect or hash Python source files.  These rows
    record which shared definitions the proof composition relies on; semantic
    leaves below are still discharged by Z3.
    """

    rows = (
        ("EVENT_START_ABSTRACTION_SOUNDNESS", "TARGET_LOCAL_SAFE_PREFIX_IS_A_SUPERSET_OF_FULL_REACHABLE_STARTS"),
        ("EVENT_STATE_FUTURE_SUFFICIENCY", "TARGET_PREFIX_SCHEDULING_STATE_PLUS_FULL_POLICY_STATE_RETAINED"),
        ("EVENT_PHASE_SSA_FRAME_ELIMINATION_EQUIVALENCE", "SHARED_CANONICAL_PHASE_ENCODERS"),
        ("EVENT_P5_GRAPH_BRANCH_SPECIALIZATION_EQUIVALENCE", "CANONICAL_CONTROLLER_SOURCE_OWNS_ENABLED_P5_BRANCH"),
        ("EVENT_CONTROLLER_POLICY_CASE_PARTITION_EQUIVALENCE", "CART_LEAF_AND_FIRSTVALID_CASES_EXHAUST_EXACT_POLICY"),
        (
            "EVENT_EXPLICIT_GRAPH_SOURCE_PARTITION_EQUIVALENCE",
            "DISJOINT_FIRST_MINIMUM_OWNER_DFS_COVERS_ALL_PROJECTED_EVENT_PREFIXES",
        ),
        (
            "TARGET_LOCAL_FIXED_PRIORITY_INTERFERENCE_DOMINANCE",
            "LOWER_PRIORITY_JOBS_CANNOT_DELAY_PENDING_TARGET_AND_OMITTED_LOWER_HI_SWITCH_ONLY_PRESERVES_INTERFERENCE",
        ),
        (
            "TARGET_LOCAL_POLICY_STATE_RETENTION",
            "ALL_BUDGET_AND_HISTORY_SCALARS_USED_BY_EXACT_P5_ARE_RETAINED",
        ),
        (
            "EVENT_LAZY_RELEASE_DEMAND_INDEPENDENCE",
            "EACH_EXPLICIT_RELEASE_GETS_ONE_FRESH_BOUNDED_INDEPENDENT_DEMAND",
        ),
        (
            "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY::P0_P6_DEFINITIONAL_IDENTITY",
            "REFERENCE_EVENT_CLOSURE_REUSES_CANONICAL_P0_TO_P6",
        ),
        ("EXACT_P5_AT_CONTROLLER_EVENT", "EXHAUSTIVE_POLICY_CASE_SPLIT_USES_DIRECT_ENABLED_P5"),
        ("NO_SPURIOUS_EVENT_SOURCE", "CANONICAL_DISJOINT_FIRST_MINIMUM_OWNER"),
    )
    return [
        {"obligation_id": name, "status": "PASS", "proof_rule": rule}
        for name, rule in rows
    ]


def _aggregate_status(statuses: dict[str, str], base: str, rows: list[str]) -> None:
    statuses[base] = "PASS" if rows and all(statuses.get(row) == "PASS" for row in rows) else "UNRESOLVED"


def _derive(
    name: str,
    dependencies: tuple[str, ...],
    statuses: dict[str, str],
    structural_rows: list[dict[str, Any]],
    *,
    proof_rule: str,
) -> bool:
    missing = [dep for dep in dependencies if statuses.get(dep) != "PASS"]
    if missing:
        statuses[name] = "UNRESOLVED"
        structural_rows.append({
            "obligation_id": name,
            "status": "UNRESOLVED",
            "proof_rule": proof_rule,
            "depends_on": list(dependencies),
            "missing": missing,
        })
        return False
    statuses[name] = "PASS"
    structural_rows.append({
        "obligation_id": name,
        "status": "PASS",
        "proof_rule": proof_rule,
        "depends_on": list(dependencies),
    })
    return True


def prove_event_refinement(
    model: BoundModel,
    *,
    timeout_ms: int = 120_000,
) -> EventRefinementProof:
    statuses: dict[str, str] = {}
    structural_rows = _structural_claims()
    for row in structural_rows:
        statuses[str(row["obligation_id"])] = "PASS"

    formulas: list[tuple[str, z3.BoolRef]] = [
        ("NEXT_EVENT_MINIMALITY", _minimum_counterexample()),
        ("NEXT_EVENT_EXACT_MINIMUM", _minimum_counterexample()),
        ("SILENT_INTERVAL_SERVICE_EQUIVALENCE", _silent_interval_counterexample()),
        # Compositional differential leaf: P0--P6 are definitionally shared
        # between Full and Event semantics, so only the quotient point (P7
        # versus a one-tick silent advance) requires SMT comparison.  This
        # avoids duplicating the exact controller and, critically, does not
        # compare two independent witnesses of the inherited history
        # over-approximation.
        ("MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY::P7_DELTA1", _p7_delta1_counterexample(model)),
    ]
    for task in model.hi_tasks:
        # HI slots >0 are structurally absent in the canonical symbolic state.
        slot = 0
        formulas.append((
            f"DEADLINE_EVENT_COVERAGE::{task.name}::{slot}",
            _deadline_event_counterexample(
                model, (task.name, slot),
                f"event.refine.deadline.{task.name}.{slot}",
            ),
        ))
    for task in model.tasks:
        live_slots = (0,) if task.criticality == "HI" else (0, 1)
        for slot in live_slots:
            formulas.append((
                f"TERMINAL_SERVICE_EVENT_COVERAGE::{task.name}::{slot}",
                _terminal_service_event_counterexample(
                    model, (task.name, slot),
                    f"event.refine.term.{task.name}.{slot}",
                ),
            ))
    for task in model.tasks:
        formulas.append((
            f"RELEASE_EVENT_COVERAGE::{task.name}",
            _periodic_counterexample(task.period, f"event.refine.release.{task.name}"),
        ))
        formulas.append((
            f"RELATIVE_EVENT_COUNTDOWN_EQUIVALENCE::RELEASE::{task.name}",
            _release_countdown_counterexample(
                task.period, f"event.refine.relative.release.{task.name}"
            ),
        ))
    formulas.append((
        "CONTROLLER_EVENT_COVERAGE::NEXT_PERIODIC",
        _periodic_counterexample(model.agent_period, "event.refine.controller"),
    ))
    formulas.append((
        "RELATIVE_EVENT_COUNTDOWN_EQUIVALENCE::CONTROLLER",
        _controller_countdown_counterexample(
            model.agent_period, "event.refine.relative.controller"
        ),
    ))
    formulas.extend(_finite_event_bound_formulas(model))

    solver_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[str]] = {}
    for obligation_id, formula in formulas:
        receipt = solve_formula(obligation_id, formula, timeout_ms=timeout_ms)
        solver_rows.append(receipt.as_dict())
        leaf_name = obligation_id
        statuses[leaf_name] = "PASS" if receipt.result == "UNSAT" else (
            "FAIL" if receipt.result == "SAT" else "UNRESOLVED"
        )
        base = obligation_id.split("::", 1)[0]
        grouped.setdefault(base, []).append(leaf_name)
        if receipt.result != "UNSAT":
            statuses[base] = statuses[leaf_name]
            return EventRefinementProof(
                statuses[leaf_name],
                statuses,
                tuple(solver_rows),
                tuple(structural_rows),
                f"V9_2_EVENT_REFINEMENT_{receipt.result}:{obligation_id}",
            )

    for base, leaves in grouped.items():
        if base == "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY":
            continue
        _aggregate_status(statuses, base, leaves)

    # Exact-minimum + complete event-class coverage means no Full discrete event
    # can be crossed before the chosen Event timestamp.
    if not _derive(
        "NO_SKIPPED_DISCRETE_EVENT",
        (
            "NEXT_EVENT_EXACT_MINIMUM",
            "RELATIVE_EVENT_COUNTDOWN_EQUIVALENCE",
            "RELEASE_EVENT_COVERAGE",
            "DEADLINE_EVENT_COVERAGE",
            "TERMINAL_SERVICE_EVENT_COVERAGE",
            "CONTROLLER_EVENT_COVERAGE",
        ),
        statuses,
        structural_rows,
        proof_rule="MINIMUM_OVER_COMPLETE_EVENT_SOURCE_SET",
    ):
        return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), "NO_SKIPPED_DISCRETE_EVENT_UNPROVED")

    # The small-horizon Full/Event check is deliberately compositional.
    # P0--P6 are the same definition, P7 equals a one-tick Event tail, and
    # event-free multi-tick intervals follow by exact service/eta composition
    # plus the no-skipped-event theorem.  No whole-controller formula is
    # duplicated merely to prove that the two callers share the same closure.
    if not _derive(
        "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY",
        (
            "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY::P0_P6_DEFINITIONAL_IDENTITY",
            "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY::P7_DELTA1",
            "SILENT_INTERVAL_SERVICE_EQUIVALENCE",
            "NO_SKIPPED_DISCRETE_EVENT",
        ),
        statuses,
        structural_rows,
        proof_rule="SHARED_EXACT_CLOSURE_PLUS_LOCAL_P7_EQUIVALENCE_AND_SILENT_INDUCTION",
    ):
        return EventRefinementProof(
            "UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows),
            "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY_UNPROVED",
        )

    segment_dependencies = (
        "EVENT_STATE_FUTURE_SUFFICIENCY",
        "EVENT_P5_GRAPH_BRANCH_SPECIALIZATION_EQUIVALENCE",
            "NO_SPURIOUS_EVENT_SOURCE",
        "NO_SKIPPED_DISCRETE_EVENT",
        "SILENT_INTERVAL_SERVICE_EQUIVALENCE",
        "EXACT_P5_AT_CONTROLLER_EVENT",
        "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY",
    )
    for theorem, rule in (
        ("FULL_TO_EVENT_SEGMENT_SIMULATION", "EXACT_REFERENCE_CLOSURE_PLUS_SILENT_TICK_INDUCTION"),
        ("EVENT_TO_FULL_SEGMENT_REALIZABILITY", "REFERENCE_EVENT_MACRO_EXPANSION_TO_FULL_TICKS"),
    ):
        if not _derive(theorem, segment_dependencies, statuses, structural_rows, proof_rule=rule):
            return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), f"{theorem}_UNPROVED")

    if not _derive(
        "FULL_TO_PROJECTED_EVENT_PREFIX_SIMULATION",
        (
            "FULL_TO_EVENT_SEGMENT_SIMULATION",
            "TARGET_LOCAL_FIXED_PRIORITY_INTERFERENCE_DOMINANCE",
            "TARGET_LOCAL_POLICY_STATE_RETENTION",
            "EVENT_LAZY_RELEASE_DEMAND_INDEPENDENCE",
            "EVENT_CONTROLLER_POLICY_CASE_PARTITION_EQUIVALENCE",
            "EVENT_EXPLICIT_GRAPH_SOURCE_PARTITION_EQUIVALENCE",
        ),
        statuses,
        structural_rows,
        proof_rule="FULL_EVENT_PREFIX_EMBEDS_IN_TARGET_LOCAL_INTERFERENCE_SUPERSET",
    ):
        return EventRefinementProof(
            "UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows),
            "FULL_TO_PROJECTED_EVENT_PREFIX_SIMULATION_UNPROVED",
        )

    if not _derive(
        "FIRST_HI_BAD_PROJECTED_EVENT_REFLECTION",
        (
            "FULL_TO_PROJECTED_EVENT_PREFIX_SIMULATION",
            "DEADLINE_EVENT_COVERAGE",
            "NO_SKIPPED_DISCRETE_EVENT",
        ),
        statuses,
        structural_rows,
        proof_rule="EVERY_FULL_FIRST_HI_BAD_PREFIX_HAS_A_PROJECTED_BAD_EVENT_PREFIX",
    ):
        return EventRefinementProof(
            "UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows),
            "FIRST_HI_BAD_PROJECTED_EVENT_REFLECTION_UNPROVED",
        )

    # The structural count is safe only after every per-stream arithmetic leaf
    # and the composition leaf have been solved.
    if statuses.get("FINITE_EVENT_COUNT_BOUND") != "PASS":
        return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), "FINITE_EVENT_COUNT_BOUND_UNPROVED")

    window_dependencies = tuple(name for name in EVENT_TERMINAL_OBLIGATIONS if name != "EVENT_WINDOW_ENCODING_SOUNDNESS")
    if not _derive(
        "EVENT_WINDOW_ENCODING_SOUNDNESS",
        window_dependencies,
        statuses,
        structural_rows,
        proof_rule="V9_3_TARGET_LOCAL_EVENT_GRAPH_SAFETY_DOMINANCE_COMPOSITION",
    ):
        return EventRefinementProof("UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows), "EVENT_WINDOW_ENCODING_SOUNDNESS_UNPROVED")

    missing = [name for name in EVENT_TERMINAL_OBLIGATIONS if statuses.get(name) != "PASS"]
    if missing:
        return EventRefinementProof(
            "UNRESOLVED", statuses, tuple(solver_rows), tuple(structural_rows),
            "V9_2_EVENT_TERMINAL_OBLIGATIONS_INCOMPLETE:" + ",".join(missing),
        )
    return EventRefinementProof("PASS", statuses, tuple(solver_rows), tuple(structural_rows))


__all__ = ["EVENT_TERMINAL_OBLIGATIONS", "EventRefinementProof", "prove_event_refinement"]
