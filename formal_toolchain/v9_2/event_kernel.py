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
    encode_p5_identity,
    encode_p6_dispatch,
)


EVENT_KERNEL_VERSION = "V9_2_EXACT_EVENT_MACRO_V3_SSA_EXACT_P5_POOL"


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


def _min_expr(values: Iterable[z3.ArithRef]) -> z3.ArithRef:
    rows = list(values)
    if not rows:
        raise ValueError("event candidate set must be non-empty")
    result = rows[0]
    for value in rows[1:]:
        result = z3.If(value < result, value, result)
    return result


def _next_periodic_after(t: z3.ArithRef, period: int) -> z3.ArithRef:
    """Least phase-zero periodic timestamp strictly larger than ``t``."""

    period = int(period)
    if period <= 0:
        raise ValueError("period must be positive")
    return ((t / period) + 1) * period


def _slot_index(model: BoundModel, key: tuple[str, int]) -> int:
    for task_index, task in enumerate(model.tasks):
        for slot in range(model.max_jobs_per_task):
            if (task.name, slot) == key:
                return task_index * model.max_jobs_per_task + slot
    raise KeyError(key)


@dataclass(frozen=True, slots=True)
class ExactControllerInstance:
    """One exact deployed P5 instance at a concrete periodic activation time.

    Instances are pooled per finite event window.  The expensive
    observation->tree->mask->FirstValid relation is constructed once per
    *possible controller activation*, not once per Event slot.
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
    proved by the V9.2 finite-event theorem.  Every exact P5 formula is built
    once here and later shared by every Event slot whose P5 timestamp equals
    the corresponding activation time.  No controller summary or widened
    behavior is introduced.
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
        exact = encode_p5_controller(pre, post, model)
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
    At an activation timestamp, the event-local pre/post states are equated to
    the unique pooled exact-P5 instance for that timestamp.  Hence this is a
    formula-factoring transformation, not a controller abstraction.
    """

    enabled = (z.t % model.agent_period) == 0
    active_rows: list[z3.BoolRef] = []
    for instance in pool.instances:
        in_window = instance.activation_time < pool.horizon_time
        at_instance = z.t == instance.activation_time
        active_rows.append(z3.And(
            in_window,
            at_instance,
            state_equality(z, instance.pre_state),
            state_equality(zp, instance.post_state),
        ))

    # Exact periodic arithmetic plus the finite pool must cover every enabled
    # P5 inside the proof window.  If this ever becomes false, the formula is
    # UNSAT rather than silently falling back to identity or a summary.
    exact_active = z3.Or(*active_rows) if active_rows else z3.BoolVal(False)
    return z3.And(
        z.p == 5,
        zp.p == 6,
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


@dataclass(frozen=True, slots=True)
class EventStepEncoding:
    """One exact event macro from a P0 event boundary to the next P0 boundary."""

    source: SymbolicKernelState
    destination: SymbolicKernelState
    phase_states: tuple[SymbolicKernelState, ...]
    candidates: EventCandidateSet
    delta: z3.ArithRef
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
) -> EventCandidateSet:
    """Build the exact next-event candidate set after P6 dispatch.

    Candidate classes are exactly the discrete changes represented by the Full
    P0--P7 kernel plus the proof query horizon:
      * phase-zero periodic release,
      * HI deadline observation,
      * controller activation,
      * selected-job completion/cap terminal service,
      * target proof horizon.

    The horizon is an observation boundary, not an environment event.  It is
    required so the event window stops exactly at the target deadline even when
    the target completed earlier.
    """

    t = dispatch_state.t
    releases = tuple(
        (task.name, _next_periodic_after(t, task.period))
        for task in model.tasks
    )
    controller = _next_periodic_after(t, model.agent_period)

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

    completion_terms: list[z3.ArithRef] = []
    for key, job in dispatch_state.jobs.items():
        index = _slot_index(model, key)
        completion_terms.append(z3.If(
            dispatch_state.frontier.selected_slot == index,
            t + (job.effective_demand - job.executed_service),
            horizon_time,
        ))
    completion = _min_expr(completion_terms) if completion_terms else horizon_time

    all_candidates = (
        horizon_time,
        *(value for _, value in releases),
        controller,
        *(value for _, value in hi_deadlines),
        completion,
    )
    next_time = _min_expr(all_candidates)
    return EventCandidateSet(
        horizon=horizon_time,
        releases=releases,
        controller=controller,
        hi_deadlines=tuple(hi_deadlines),
        completion=completion,
        all_candidates=tuple(all_candidates),
        next_time=next_time,
    )


def _silent_interval_advance(
    dispatch_state: SymbolicKernelState,
    destination: SymbolicKernelState,
    model: BoundModel,
    candidates: EventCandidateSet,
) -> z3.BoolRef:
    """Exact quotient of repeated P7 ticks over an event-free interval."""

    next_time = candidates.next_time
    delta = next_time - dispatch_state.t
    clauses: list[z3.BoolRef] = [
        destination.t == next_time,
        destination.p == 0,
        destination.mode_hi == dispatch_state.mode_hi,
        destination.hi_miss_ledger == dispatch_state.hi_miss_ledger,
        destination.frontier.selected_slot == -1,
        z3.Not(destination.frontier.running),
        delta >= 1,
        next_time <= candidates.horizon,
    ]

    for task in model.tasks:
        clauses.extend((
            destination.budgets[task.name] == dispatch_state.budgets[task.name],
            destination.eta[task.name] == z3.If(
                dispatch_state.eta[task.name] + delta < task.period,
                dispatch_state.eta[task.name] + delta,
                task.period,
            ),
            destination.chi.recent_cost[task.name] == dispatch_state.chi.recent_cost[task.name],
            destination.chi.ema_cost[task.name] == dispatch_state.chi.ema_cost[task.name],
            destination.chi.overrun_ema[task.name] == dispatch_state.chi.overrun_ema[task.name],
            destination.chi.max_cost_k[task.name] == dispatch_state.chi.max_cost_k[task.name],
        ))

    for lhs, rhs in (
        (destination.chi.mode_change_window, dispatch_state.chi.mode_change_window),
        (destination.chi.lo_cancel_window, dispatch_state.chi.lo_cancel_window),
        (destination.chi.hi_overrun_window, dispatch_state.chi.hi_overrun_window),
        (destination.chi.lo_overrun_window, dispatch_state.chi.lo_overrun_window),
        (destination.chi.job_start_window, dispatch_state.chi.job_start_window),
    ):
        clauses.extend(a == b for a, b in zip(lhs, rhs))

    selected_remaining_terms: list[z3.BoolRef] = []
    for key, job in dispatch_state.jobs.items():
        other = destination.jobs[key]
        index = _slot_index(model, key)
        selected = dispatch_state.frontier.selected_slot == index
        for field_name in (
            "present", "release_index", "release_time", "absolute_deadline",
            "tie_break", "release_entry_mode_hi", "classification_abnormal",
            "budget_at_release", "actual_demand", "effective_demand",
            "removed", "ready",
        ):
            clauses.append(getattr(other, field_name) == getattr(job, field_name))
        clauses.append(
            other.executed_service
            == job.executed_service + z3.If(selected, delta, 0)
        )
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
    # Exact minimum: every candidate is at or after the chosen timestamp and
    # the chosen timestamp equals at least one candidate source.
    clauses.extend(candidates.next_time <= value for value in candidates.all_candidates)
    clauses.append(z3.Or(*(candidates.next_time == value for value in candidates.all_candidates)))
    return z3.And(*clauses)


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
    candidates = build_event_candidates(dispatch_state, model, horizon_time=horizon_time)
    delta = candidates.next_time - source.t
    formula = z3.And(
        source.p == 0,
        source.t < horizon_time,
        closure,
        _silent_interval_advance(dispatch_state, destination, model, candidates),
    )
    return EventStepEncoding(
        source=source,
        destination=destination,
        phase_states=phase_states,
        candidates=candidates,
        delta=delta,
        formula=formula,
    )


def event_boundary_stutter(
    source: SymbolicKernelState,
    destination: SymbolicKernelState,
    *,
    horizon_time: z3.ArithRef,
) -> z3.BoolRef:
    """Canonical terminal self-loop used only after reaching the query horizon."""

    return z3.And(
        source.t == horizon_time,
        source.p == 0,
        state_equality(source, destination),
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
    "event_boundary_stutter",
    "state_equality",
]
