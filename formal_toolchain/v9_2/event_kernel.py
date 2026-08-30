"""Exact event-driven macro semantics for the V9.2 finite proof realization.

The Full P0--P7 kernel remains the reference semantics.  This module does not
introduce a new safety abstraction: an event step executes the exact P0--P6
closure at the current timestamp and replaces only a run of event-free P7
service ticks by their algebraically identical bulk update.

Event-boundary states deliberately retain the complete Full-kernel persistent
state.  V9.2 therefore gets its memory reduction from *fewer symbolic steps*,
not from merging job identities or widening controller behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import z3

from .environment_encoder import SymbolicEnvironment
from .symbolic_state import (
    BoundModel, SymbolicJob, SymbolicKernelState, declare_sparse_successor, declare_state,
)
from .transition_encoder import (
    encode_p0_settle,
    encode_p1_idle_recovery,
    encode_p2_deadline_observe,
    encode_p3_arrival_freeze,
    encode_p4_mode_switch,
    encode_p5_controller,
    encode_p5_controller_effect,
    encode_p5_controller_frame,
    encode_p5_identity,
    encode_p6_dispatch,
)


EVENT_KERNEL_VERSION = "V9_2_EXACT_EVENT_MACRO_V6_LINEAR_PERIODIC_CANDIDATES"


def _job_fields(job: SymbolicJob) -> tuple[z3.ExprRef, ...]:
    return (
        job.present,
        job.release_index,
        job.release_time,
        job.absolute_deadline,
        job.tie_break,
        job.release_entry_mode_hi,
        job.classification_abnormal,
        job.budget_at_release,
        job.actual_demand,
        job.effective_demand,
        job.executed_service,
        job.removed,
        job.ready,
    )


def state_equality(left: SymbolicKernelState, right: SymbolicKernelState) -> z3.BoolRef:
    """Field-for-field equality of two Full-kernel states."""

    clauses: list[z3.BoolRef] = [
        left.t == right.t,
        left.p == right.p,
        left.mode_hi == right.mode_hi,
        left.frontier.selected_slot == right.frontier.selected_slot,
        left.frontier.running == right.frontier.running,
        left.hi_miss_ledger == right.hi_miss_ledger,
    ]
    for name in left.budgets:
        clauses.extend((
            left.budgets[name] == right.budgets[name],
            left.eta[name] == right.eta[name],
            left.chi.recent_cost[name] == right.chi.recent_cost[name],
            left.chi.ema_cost[name] == right.chi.ema_cost[name],
            left.chi.overrun_ema[name] == right.chi.overrun_ema[name],
            left.chi.max_cost_k[name] == right.chi.max_cost_k[name],
        ))
    for key in left.jobs:
        clauses.extend(a == b for a, b in zip(_job_fields(left.jobs[key]), _job_fields(right.jobs[key])))
    for lhs, rhs in (
        (left.chi.mode_change_window, right.chi.mode_change_window),
        (left.chi.lo_cancel_window, right.chi.lo_cancel_window),
        (left.chi.hi_overrun_window, right.chi.hi_overrun_window),
        (left.chi.lo_overrun_window, right.chi.lo_overrun_window),
        (left.chi.job_start_window, right.chi.job_start_window),
    ):
        clauses.extend(a == b for a, b in zip(lhs, rhs))
    return z3.And(*clauses)


def _append_if_distinct(
    clauses: list[z3.BoolRef], left: z3.ExprRef, right: z3.ExprRef
) -> None:
    """Append an equality only when the two ASTs are not already shared."""

    if not left.eq(right):
        clauses.append(left == right)


def _controller_effect_state_equality(
    left: SymbolicKernelState,
    right: SymbolicKernelState,
    *,
    include_time: bool,
) -> z3.BoolRef:
    """Equality over the exact read/write support of deployed P5.

    The controller reads ``t``, budgets and RuntimeFeatureState history and
    writes budgets/history.  All other Full-state fields are exact P5 frame
    fields and are handled by ``encode_p5_controller_frame`` in the event-local
    SSA chain.  Keeping them out of every pool link prevents the same hundreds
    of job/frame equalities from being copied for every Event slot.
    """

    clauses: list[z3.BoolRef] = []
    if include_time:
        _append_if_distinct(clauses, left.t, right.t)
    for name in left.budgets:
        _append_if_distinct(clauses, left.budgets[name], right.budgets[name])
    for lhs, rhs in (
        (left.chi.recent_cost, right.chi.recent_cost),
        (left.chi.ema_cost, right.chi.ema_cost),
        (left.chi.overrun_ema, right.chi.overrun_ema),
        (left.chi.max_cost_k, right.chi.max_cost_k),
    ):
        for name in lhs:
            _append_if_distinct(clauses, lhs[name], rhs[name])
    for lhs_values, rhs_values in (
        (left.chi.mode_change_window, right.chi.mode_change_window),
        (left.chi.lo_cancel_window, right.chi.lo_cancel_window),
        (left.chi.hi_overrun_window, right.chi.hi_overrun_window),
        (left.chi.lo_overrun_window, right.chi.lo_overrun_window),
        (left.chi.job_start_window, right.chi.job_start_window),
    ):
        for lhs, rhs in zip(lhs_values, rhs_values):
            _append_if_distinct(clauses, lhs, rhs)
    return z3.And(*clauses)


def _exact_minimum_definition(
    result: z3.ArithRef, values: Iterable[z3.ArithRef]
) -> z3.BoolRef:
    """Exact finite minimum without a nested ITE expression.

    ``result <= value`` for every candidate plus equality to at least one
    candidate uniquely defines the mathematical minimum.  This is an exact
    existential/SSA factoring of the previous nested ``If`` minimum, not an
    approximation.  Keeping the minimum as a scalar prevents the full nested
    candidate term from being substituted into every service/eta/state update.
    """

    rows = tuple(values)
    if not rows:
        raise ValueError("event candidate set must be non-empty")
    return z3.And(
        *(result <= value for value in rows),
        z3.Or(*(result == value for value in rows)),
    )


def _next_periodic_after(t: z3.ArithRef, period: int) -> z3.ArithRef:
    """Reference expression for the least phase-zero timestamp after ``t``.

    The production Event kernel no longer substitutes this symbolic division
    expression into every candidate relation.  It is retained as an
    independent reference term for machine equivalence checks.
    """

    period = int(period)
    if period <= 0:
        raise ValueError("period must be positive")
    return ((t / period) + 1) * period


def _declare_exact_periodic_successor(
    t: z3.ArithRef,
    period: int,
    *,
    prefix: str,
) -> tuple[z3.ArithRef, z3.BoolRef]:
    """Exact quotient-free encoding of the next phase-zero periodic timestamp.

    For positive integer ``period`` there is exactly one multiple ``nxt`` of
    ``period`` in the half-open interval ``(t, t + period]``.  Introducing an
    integer period index keeps the relation in linear integer arithmetic:

        nxt = k * period
        t < nxt <= t + period

    This is formula-equivalent to ``((t / period) + 1) * period`` for the
    non-negative runtime timestamps used by the Event kernel, while avoiding
    symbolic integer division in every release/controller candidate.
    """

    period = int(period)
    if period <= 0:
        raise ValueError("period must be positive")
    period_index = z3.Int(f"{prefix}.period_index")
    nxt = z3.Int(f"{prefix}.time")
    definition = z3.And(
        nxt == period_index * period,
        nxt > t,
        nxt <= t + period,
    )
    return nxt, definition


def _slot_index(model: BoundModel, key: tuple[str, int]) -> int:
    for task_index, task in enumerate(model.tasks):
        for slot in range(model.max_jobs_per_task):
            if (task.name, slot) == key:
                return task_index * model.max_jobs_per_task + slot
    raise KeyError(key)


@dataclass(frozen=True, slots=True)
class ExactControllerInstance:
    """One exact deployed P5 instance at a concrete periodic activation time.

    Instances are pooled per finite event window.  The expensive exact
    controller-owned observation->tree->mask->FirstValid/budget/history
    effect is constructed once per *possible controller activation*, not once
    per Event slot.  The non-controller Full-state frame remains event-local
    and exact.
    """

    activation_time: z3.ArithRef
    pre_state: SymbolicKernelState
    post_state: SymbolicKernelState
    formula: z3.BoolRef


@dataclass(frozen=True, slots=True)
class ExactControllerPool:
    """Finite exact-P5 pool for one target-deadline window."""

    origin_time: z3.ArithRef
    horizon_time: z3.ArithRef
    first_activation: z3.ArithRef
    instances: tuple[ExactControllerInstance, ...]
    formula: z3.BoolRef


def _first_periodic_at_or_after(t: z3.ArithRef, period: int) -> z3.ArithRef:
    """Least phase-zero periodic timestamp greater than or equal to ``t``."""

    period = int(period)
    if period <= 0:
        raise ValueError("period must be positive")
    return ((t + period - 1) / period) * period


def build_exact_controller_pool(
    model: BoundModel,
    *,
    origin_time: z3.ArithRef,
    horizon_time: z3.ArithRef,
    controller_bound: int,
    prefix: str,
) -> ExactControllerPool:
    """Construct exactly the finite controller activations that can affect a window.

    ``controller_bound`` is the structural bound ``floor(D/Tctrl)+1`` already
    proved by the V9.2 finite-event theorem.  Every exact controller-owned P5
    effect is built once here and later shared by every Event slot whose P5
    timestamp equals the corresponding activation time.  The exact P5 frame
    is kept in the event-local sparse state chain.  No controller summary or
    widened behavior is introduced.
    """

    controller_bound = int(controller_bound)
    if controller_bound <= 0:
        raise ValueError("controller_bound must be positive")
    first = _first_periodic_at_or_after(origin_time, model.agent_period)
    instances: list[ExactControllerInstance] = []
    constraints: list[z3.BoolRef] = []
    for index in range(controller_bound):
        activation_time = first + index * int(model.agent_period)
        pre = declare_state(f"{prefix}.{index}.pre", model)
        post = declare_state(f"{prefix}.{index}.post", model)
        # Pool only the controller-owned exact relation.  The complete deployed
        # P5 is definitionally ``effect AND frame``; event-local sparse P5
        # successors supply that exact frame without re-copying it into every
        # pooled instance and every Event-slot link.
        exact = encode_p5_controller_effect(pre, post, model)
        # Only activations strictly before the target horizon are part of the
        # event-window P5 sequence.  The exact formula is guarded rather than
        # used as an unconstrained terminal-side witness.
        constraints.append(z3.Implies(
            activation_time < horizon_time,
            z3.And(
                pre.t == activation_time,
                activation_time % model.agent_period == 0,
                exact,
            ),
        ))
        instances.append(ExactControllerInstance(activation_time, pre, post, exact))
    return ExactControllerPool(
        origin_time=origin_time,
        horizon_time=horizon_time,
        first_activation=first,
        instances=tuple(instances),
        formula=z3.And(*constraints),
    )


def encode_p5_from_exact_pool(
    z: SymbolicKernelState,
    zp: SymbolicKernelState,
    model: BoundModel,
    pool: ExactControllerPool,
) -> z3.BoolRef:
    """Exact P5 with window-global sharing of expensive controller formulas.

    When ``t mod Tctrl != 0`` this is strict identity, exactly as deployed P5.
    At an activation timestamp, only the exact P5 read/write support
    (time/budgets/history on input and budgets/history on output) is linked to
    the unique pooled controller effect for that timestamp; the complete
    non-controller frame is imposed locally by ``encode_p5_controller_frame``.
    The conjunction is definitionally the deployed P5 relation, so this is
    formula factoring rather than a controller abstraction.
    """

    enabled = (z.t % model.agent_period) == 0
    active_rows: list[z3.BoolRef] = []
    for instance in pool.instances:
        in_window = instance.activation_time < pool.horizon_time
        at_instance = z.t == instance.activation_time
        active_rows.append(z3.And(
            in_window,
            at_instance,
            _controller_effect_state_equality(
                z, instance.pre_state, include_time=True
            ),
            _controller_effect_state_equality(
                zp, instance.post_state, include_time=False
            ),
        ))

    # Exact periodic arithmetic plus the finite pool must cover every enabled
    # P5 inside the proof window.  If this ever becomes false, the formula is
    # UNSAT rather than silently falling back to identity or a summary.
    exact_active = z3.Or(*active_rows) if active_rows else z3.BoolVal(False)
    return z3.And(
        z.p == 5,
        zp.p == 6,
        encode_p5_controller_frame(z, zp, model),
        z3.Implies(z3.Not(enabled), encode_p5_identity(z, zp, model)),
        z3.Implies(enabled, exact_active),
    )


@dataclass(frozen=True, slots=True)
class EventCandidateSet:
    """Exact candidate timestamps used by one event macro."""

    horizon: z3.ArithRef
    releases: tuple[tuple[str, z3.ArithRef], ...]
    controller: z3.ArithRef
    hi_deadlines: tuple[tuple[tuple[str, int], z3.ArithRef], ...]
    completion: z3.ArithRef
    all_candidates: tuple[z3.ArithRef, ...]
    next_time: z3.ArithRef
    definition_formula: z3.BoolRef


@dataclass(frozen=True, slots=True)
class EventStepEncoding:
    """One exact event macro from a P0 event boundary to the next P0 boundary."""

    source: SymbolicKernelState
    destination: SymbolicKernelState
    phase_states: tuple[SymbolicKernelState, ...]
    candidates: EventCandidateSet
    delta: z3.ArithRef
    closure_formula: z3.BoolRef
    silent_core_formula: z3.BoolRef
    formula: z3.BoolRef


def _exact_p0_to_p7_closure(
    source: SymbolicKernelState,
    model: BoundModel,
    env: SymbolicEnvironment,
    *,
    prefix: str,
    controller_pool: ExactControllerPool | None = None,
) -> tuple[tuple[SymbolicKernelState, ...], z3.BoolRef]:
    """Execute exact P0..P6, ending at the P7 dispatch state."""

    # Exact SSA-style phase chain.  A phase declares fresh Z3 symbols only for
    # fields it owns; every frame component is structurally shared with the
    # preceding state.  Canonical phase encoders are still used unchanged, so
    # this is existential/frame elimination rather than a new abstraction.
    p1 = declare_sparse_successor(
        f"{prefix}.p1", source, model, phase=1,
        mutable=frozenset({"jobs.present", "jobs.removed", "jobs.ready"}),
    )
    p2 = declare_sparse_successor(
        f"{prefix}.p2", p1, model, phase=2, mutable=frozenset({"mode"}),
    )
    p3 = declare_sparse_successor(
        f"{prefix}.p3", p2, model, phase=3, mutable=frozenset({"hi_miss_ledger"}),
    )
    p4 = declare_sparse_successor(
        f"{prefix}.p4", p3, model, phase=4, mutable=frozenset({"eta", "jobs"}),
    )
    p5 = declare_sparse_successor(
        f"{prefix}.p5", p4, model, phase=5, mutable=frozenset({"mode"}),
    )
    p6 = declare_sparse_successor(
        f"{prefix}.p6", p5, model, phase=6, mutable=frozenset({"budgets", "history"}),
    )
    p7 = declare_sparse_successor(
        f"{prefix}.p7", p6, model, phase=7, mutable=frozenset({"frontier"}),
    )
    states = (p1, p2, p3, p4, p5, p6, p7)
    formula = z3.And(
        encode_p0_settle(source, p1, model),
        encode_p1_idle_recovery(p1, p2, model),
        encode_p2_deadline_observe(p2, p3, model),
        encode_p3_arrival_freeze(p3, p4, model, env),
        encode_p4_mode_switch(p4, p5, model),
        # Event windows always use exact deployed P5 semantics.  A finite
        # window may share those exact formulas through ``controller_pool``;
        # the P5 invariant summary is intentionally never imported here.
        (encode_p5_controller(p5, p6, model) if controller_pool is None
         else encode_p5_from_exact_pool(p5, p6, model, controller_pool)),
        encode_p6_dispatch(p6, p7, model),
    )
    return states, formula


def build_event_candidates(
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    *,
    horizon_time: z3.ArithRef,
    prefix: str,
) -> EventCandidateSet:
    """Build the exact next-event candidate set after P6 dispatch.

    Candidate classes are exactly the discrete changes represented by the Full
    P0--P7 kernel plus the proof query horizon.  The two finite minima
    (selected-job completion and global next-event time) are represented by
    fresh scalar SSA symbols constrained by ``_exact_minimum_definition``.
    This is formula-equivalent to the previous nested-ITE minima but exposes
    substantially better arithmetic propagation to SMT preprocessing.
    """

    t = dispatch_state.t
    periodic_definitions: list[z3.BoolRef] = []
    release_rows: list[tuple[str, z3.ArithRef]] = []
    for task in model.tasks:
        release_time, release_definition = _declare_exact_periodic_successor(
            t,
            task.period,
            prefix=f"{prefix}.candidate.release.{task.name}",
        )
        release_rows.append((task.name, release_time))
        periodic_definitions.append(release_definition)
    releases = tuple(release_rows)
    controller, controller_definition = _declare_exact_periodic_successor(
        t,
        model.agent_period,
        prefix=f"{prefix}.candidate.controller",
    )
    periodic_definitions.append(controller_definition)

    hi_deadlines: list[tuple[tuple[str, int], z3.ArithRef]] = []
    for task in model.hi_tasks:
        for slot in range(model.max_jobs_per_task):
            job = dispatch_state.jobs[(task.name, slot)]
            candidate = z3.If(
                z3.And(
                    job.present,
                    z3.Not(job.removed),
                    job.executed_service < job.effective_demand,
                    job.absolute_deadline > t,
                ),
                job.absolute_deadline,
                horizon_time,
            )
            hi_deadlines.append(((task.name, slot), candidate))

    completion = z3.Int(f"{prefix}.candidate.completion")
    completion_rows: list[z3.BoolRef] = [
        z3.Implies(z3.Not(dispatch_state.frontier.running), completion == horizon_time),
    ]
    for key, job in dispatch_state.jobs.items():
        index = _slot_index(model, key)
        completion_rows.append(z3.Implies(
            dispatch_state.frontier.selected_slot == index,
            completion == t + (job.effective_demand - job.executed_service),
        ))
    completion_definition = z3.And(*completion_rows)

    all_candidates = (
        horizon_time,
        *(value for _, value in releases),
        controller,
        *(value for _, value in hi_deadlines),
        completion,
    )
    next_time = z3.Int(f"{prefix}.candidate.next_time")
    next_time_definition = _exact_minimum_definition(next_time, all_candidates)
    definition_formula = z3.And(*periodic_definitions, completion_definition, next_time_definition)
    return EventCandidateSet(
        horizon=horizon_time,
        releases=releases,
        controller=controller,
        hi_deadlines=tuple(hi_deadlines),
        completion=completion,
        all_candidates=tuple(all_candidates),
        next_time=next_time,
        definition_formula=definition_formula,
    )


def _silent_interval_core(
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    candidates: EventCandidateSet,
) -> z3.BoolRef:
    """Destination-free side conditions of an exact silent P7 interval."""

    next_time = candidates.next_time
    delta = next_time - dispatch_state.t
    clauses: list[z3.BoolRef] = [
        candidates.definition_formula,
        delta >= 1,
        next_time <= candidates.horizon,
    ]

    selected_remaining_terms: list[z3.BoolRef] = []
    for key, job in dispatch_state.jobs.items():
        index = _slot_index(model, key)
        selected = dispatch_state.frontier.selected_slot == index
        selected_remaining_terms.append(z3.Implies(
            selected,
            z3.And(
                job.present,
                z3.Not(job.removed),
                job.ready,
                job.effective_demand > job.executed_service,
                delta <= job.effective_demand - job.executed_service,
            ),
        ))

    clauses.extend(selected_remaining_terms)
    # Exact-minimum constraints live once in ``definition_formula`` instead of
    # re-expanding a nested minimum expression at every consumer.
    return z3.And(*clauses)


def _event_destination_update(
    dispatch_state: SymbolicKernelState,
    destination: SymbolicKernelState,
    model: BoundModel,
    candidates: EventCandidateSet,
    *,
    active_guard: z3.BoolRef | None = None,
    stutter_source: SymbolicKernelState | None = None,
) -> z3.BoolRef:
    """Write one event-boundary successor, optionally factoring terminal stutter.

    With ``active_guard=None`` this is the ordinary exact silent-interval
    destination relation.  With a guard and ``stutter_source``, each destination
    field is assigned once as ``If(active, active_rhs, source_rhs)``.  This is
    Boolean/SSA factoring of

      ``(active & ActiveUpdate(dst)) | (!active & dst=src)``

    and avoids serializing a second complete Full-state equality at every
    bounded Event slot.
    """

    if (active_guard is None) != (stutter_source is None):
        raise ValueError("active_guard and stutter_source must be supplied together")

    def choose(active_value: z3.ExprRef, stutter_value: z3.ExprRef | None = None) -> z3.ExprRef:
        if active_guard is None:
            return active_value
        assert stutter_value is not None
        return z3.If(active_guard, active_value, stutter_value)

    next_time = candidates.next_time
    delta = next_time - dispatch_state.t
    source = stutter_source
    clauses: list[z3.BoolRef] = [
        destination.t == choose(next_time, None if source is None else source.t),
        destination.p == choose(z3.IntVal(0), None if source is None else source.p),
        destination.mode_hi == choose(
            dispatch_state.mode_hi, None if source is None else source.mode_hi
        ),
        destination.hi_miss_ledger == choose(
            dispatch_state.hi_miss_ledger,
            None if source is None else source.hi_miss_ledger,
        ),
        destination.frontier.selected_slot == choose(
            z3.IntVal(-1), None if source is None else source.frontier.selected_slot
        ),
        destination.frontier.running == choose(
            z3.BoolVal(False), None if source is None else source.frontier.running
        ),
    ]

    for task in model.tasks:
        clauses.extend((
            destination.budgets[task.name] == choose(
                dispatch_state.budgets[task.name],
                None if source is None else source.budgets[task.name],
            ),
            destination.eta[task.name] == choose(
                z3.If(
                    dispatch_state.eta[task.name] + delta < task.period,
                    dispatch_state.eta[task.name] + delta,
                    task.period,
                ),
                None if source is None else source.eta[task.name],
            ),
            destination.chi.recent_cost[task.name] == choose(
                dispatch_state.chi.recent_cost[task.name],
                None if source is None else source.chi.recent_cost[task.name],
            ),
            destination.chi.ema_cost[task.name] == choose(
                dispatch_state.chi.ema_cost[task.name],
                None if source is None else source.chi.ema_cost[task.name],
            ),
            destination.chi.overrun_ema[task.name] == choose(
                dispatch_state.chi.overrun_ema[task.name],
                None if source is None else source.chi.overrun_ema[task.name],
            ),
            destination.chi.max_cost_k[task.name] == choose(
                dispatch_state.chi.max_cost_k[task.name],
                None if source is None else source.chi.max_cost_k[task.name],
            ),
        ))

    for lhs, active_values, stutter_values in (
        (
            destination.chi.mode_change_window,
            dispatch_state.chi.mode_change_window,
            None if source is None else source.chi.mode_change_window,
        ),
        (
            destination.chi.lo_cancel_window,
            dispatch_state.chi.lo_cancel_window,
            None if source is None else source.chi.lo_cancel_window,
        ),
        (
            destination.chi.hi_overrun_window,
            dispatch_state.chi.hi_overrun_window,
            None if source is None else source.chi.hi_overrun_window,
        ),
        (
            destination.chi.lo_overrun_window,
            dispatch_state.chi.lo_overrun_window,
            None if source is None else source.chi.lo_overrun_window,
        ),
        (
            destination.chi.job_start_window,
            dispatch_state.chi.job_start_window,
            None if source is None else source.chi.job_start_window,
        ),
    ):
        if stutter_values is None:
            clauses.extend(a == b for a, b in zip(lhs, active_values))
        else:
            clauses.extend(
                a == choose(b, c)
                for a, b, c in zip(lhs, active_values, stutter_values)
            )

    job_field_names = (
        "present", "release_index", "release_time", "absolute_deadline",
        "tie_break", "release_entry_mode_hi", "classification_abnormal",
        "budget_at_release", "actual_demand", "effective_demand",
        "removed", "ready",
    )
    for key, job in dispatch_state.jobs.items():
        other = destination.jobs[key]
        source_job = None if source is None else source.jobs[key]
        index = _slot_index(model, key)
        selected = dispatch_state.frontier.selected_slot == index
        for field_name in job_field_names:
            clauses.append(
                getattr(other, field_name)
                == choose(
                    getattr(job, field_name),
                    None if source_job is None else getattr(source_job, field_name),
                )
            )
        clauses.append(
            other.executed_service
            == choose(
                job.executed_service + z3.If(selected, delta, 0),
                None if source_job is None else source_job.executed_service,
            )
        )
    return z3.And(*clauses)


def _silent_interval_advance(
    dispatch_state: SymbolicKernelState,
    destination: SymbolicKernelState,
    model: BoundModel,
    candidates: EventCandidateSet,
) -> z3.BoolRef:
    """Exact quotient of repeated P7 ticks over an event-free interval."""

    return z3.And(
        _silent_interval_core(dispatch_state, model, candidates),
        _event_destination_update(dispatch_state, destination, model, candidates),
    )


def encode_event_step(
    source: SymbolicKernelState,
    destination: SymbolicKernelState,
    model: BoundModel,
    env: SymbolicEnvironment,
    *,
    horizon_time: z3.ArithRef,
    prefix: str,
    controller_pool: ExactControllerPool | None = None,
) -> EventStepEncoding:
    """Encode one exact V9.2 event macro.

    ``source`` must be a P0 boundary strictly before ``horizon_time``.  The
    destination is the next P0 boundary at the exact minimum candidate time.
    """

    phase_states, closure = _exact_p0_to_p7_closure(
        source, model, env, prefix=prefix, controller_pool=controller_pool
    )
    dispatch_state = phase_states[-1]
    candidates = build_event_candidates(
        dispatch_state, model, horizon_time=horizon_time, prefix=prefix
    )
    delta = candidates.next_time - source.t
    silent_core = _silent_interval_core(dispatch_state, model, candidates)
    formula = z3.And(
        source.p == 0,
        source.t < horizon_time,
        closure,
        silent_core,
        _event_destination_update(dispatch_state, destination, model, candidates),
    )
    return EventStepEncoding(
        source=source,
        destination=destination,
        phase_states=phase_states,
        candidates=candidates,
        delta=delta,
        closure_formula=closure,
        silent_core_formula=silent_core,
        formula=formula,
    )



__all__ = [
    "EVENT_KERNEL_VERSION",
    "ExactControllerInstance",
    "ExactControllerPool",
    "build_exact_controller_pool",
    "encode_p5_from_exact_pool",
    "EventCandidateSet",
    "EventStepEncoding",
    "build_event_candidates",
    "encode_event_step",
    "state_equality",
]
