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

import z3

from .environment_encoder import SymbolicEnvironment
from .symbolic_state import (
    BoundModel, SymbolicJob, SymbolicKernelState, declare_sparse_successor,
)
from .transition_encoder import (
    encode_p0_settle,
    encode_p1_idle_recovery,
    encode_p2_deadline_observe,
    encode_p3_arrival_freeze,
    encode_p4_mode_switch,
    encode_p5_controller_enabled,
    encode_p5_identity,
    encode_p6_dispatch,
)


EVENT_KERNEL_VERSION = "V9_3_EXACT_EVENT_GRAPH_SOURCE_SPECIALIZED_V1"


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
class EventSource:
    """One member of the canonical disjoint next-event source partition.

    The ordered partition uses HORIZON first, CONTROLLER second, then task
    releases, HI deadlines and finally selected-job completion.  If several
    physical events share the same timestamp, the earliest source in this
    order owns that timestamp; P0--P6 still processes *all* simultaneous
    effects at the destination, so ownership changes no kernel semantics.
    """

    kind: str
    task_name: str | None = None
    slot: int | None = None

    @property
    def source_id(self) -> str:
        if self.task_name is None:
            return self.kind
        suffix = self.task_name if self.slot is None else f"{self.task_name}_{self.slot}"
        return f"{self.kind}_{suffix}"


def enumerate_event_sources(model: BoundModel) -> tuple[EventSource, ...]:
    """Canonical complete source order used by explicit Event-graph search."""

    rows: list[EventSource] = [EventSource("HORIZON"), EventSource("CONTROLLER")]
    rows.extend(EventSource("RELEASE", task.name) for task in model.tasks)
    for task in model.hi_tasks:
        for slot in range(model.max_jobs_per_task):
            rows.append(EventSource("HI_DEADLINE", task.name, slot))
    for task in model.tasks:
        for slot in range(model.max_jobs_per_task):
            rows.append(EventSource("COMPLETION", task.name, slot))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class EventCandidateSet:
    """Exact candidate timestamps used by one source-specialized Event edge."""

    horizon: z3.ArithRef
    releases: tuple[tuple[str, z3.ArithRef], ...]
    controller: z3.ArithRef
    hi_deadlines: tuple[tuple[tuple[str, int], z3.ArithRef], ...]
    completion: z3.ArithRef
    all_candidates: tuple[z3.ArithRef, ...]
    next_time: z3.ArithRef
    base_definition_formula: z3.BoolRef
    source_definition_formula: z3.BoolRef
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
    controller_enabled: bool,
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
        # Event-graph ownership tells us exactly whether this timestamp is a
        # controller activation.  Compile only that exact P5 branch instead of
        # keeping a symbolic enabled/disabled split inside every event macro.
        (encode_p5_controller_enabled(p5, p6, model)
         if controller_enabled else encode_p5_identity(p5, p6, model)),
        encode_p6_dispatch(p6, p7, model),
    )
    return states, formula


def _source_partition_definition(
    candidates: EventCandidateSet,
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    source: EventSource,
) -> z3.BoolRef:
    """Exact disjoint first-minimum owner for one next-event source."""

    if source.kind == "HORIZON":
        chosen, index, guard = candidates.horizon, 0, z3.BoolVal(True)
    elif source.kind == "CONTROLLER":
        chosen, index, guard = candidates.controller, 1, z3.BoolVal(True)
    elif source.kind == "RELEASE":
        release_index = next(
            i for i, (task_name, _) in enumerate(candidates.releases)
            if task_name == source.task_name
        )
        chosen = candidates.releases[release_index][1]
        index = 2 + release_index
        guard = z3.BoolVal(True)
    elif source.kind == "HI_DEADLINE":
        deadline_index = next(
            i for i, ((task_name, slot), _) in enumerate(candidates.hi_deadlines)
            if task_name == source.task_name and slot == source.slot
        )
        chosen = candidates.hi_deadlines[deadline_index][1]
        index = 2 + len(candidates.releases) + deadline_index
        guard = z3.BoolVal(True)
    elif source.kind == "COMPLETION":
        if source.task_name is None or source.slot is None:
            raise KeyError(source.source_id)
        chosen = candidates.completion
        index = len(candidates.all_candidates) - 1
        selected_index = _slot_index(model, (source.task_name, source.slot))
        guard = dispatch_state.frontier.selected_slot == selected_index
    else:
        raise KeyError(source.source_id)

    earlier = candidates.all_candidates[:index]
    later = candidates.all_candidates[index + 1:]
    return z3.And(
        guard,
        candidates.next_time == chosen,
        *(value > chosen for value in earlier),
        *(value >= chosen for value in later),
    )


def build_event_candidates(
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    *,
    horizon_time: z3.ArithRef,
    prefix: str,
    source: EventSource | None = None,
) -> EventCandidateSet:
    """Build candidates with one exact disjoint next-event source fixed.

    Global ``min`` disjunction is intentionally absent.  Python graph search
    chooses one source member, and this function enforces that member is the
    canonical first owner of the mathematical minimum.  Simultaneous physical
    events are still all processed by the destination P0 closure.
    """

    t = dispatch_state.t
    periodic_definitions: list[z3.BoolRef] = []
    release_rows: list[tuple[str, z3.ArithRef]] = []
    for task in model.tasks:
        release_time, release_definition = _declare_exact_periodic_successor(
            t, task.period, prefix=f"{prefix}.candidate.release.{task.name}"
        )
        release_rows.append((task.name, release_time))
        periodic_definitions.append(release_definition)
    releases = tuple(release_rows)
    controller, controller_definition = _declare_exact_periodic_successor(
        t, model.agent_period, prefix=f"{prefix}.candidate.controller"
    )
    periodic_definitions.append(controller_definition)

    hi_deadlines: list[tuple[tuple[str, int], z3.ArithRef]] = []
    for task in model.hi_tasks:
        for slot in range(model.max_jobs_per_task):
            job = dispatch_state.jobs[(task.name, slot)]
            candidate = z3.If(
                z3.And(
                    job.present, z3.Not(job.removed),
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
        selected_index = _slot_index(model, key)
        completion_rows.append(z3.Implies(
            dispatch_state.frontier.selected_slot == selected_index,
            completion == t + (job.effective_demand - job.executed_service),
        ))
    completion_definition = z3.And(*completion_rows)

    # HORIZON precedes CONTROLLER deliberately.  A controller activation at the
    # target horizon is irrelevant because the terminal bad observation stops
    # after P2.  CONTROLLER precedes every nonterminal source so a graph child
    # is controller-enabled iff its incoming source is CONTROLLER.
    all_candidates = (
        horizon_time,
        controller,
        *(value for _, value in releases),
        *(value for _, value in hi_deadlines),
        completion,
    )
    next_time = z3.Int(f"{prefix}.candidate.next_time")
    base_definition = z3.And(*periodic_definitions, completion_definition)
    shell = EventCandidateSet(
        horizon=horizon_time, releases=releases, controller=controller,
        hi_deadlines=tuple(hi_deadlines), completion=completion,
        all_candidates=tuple(all_candidates), next_time=next_time,
        base_definition_formula=base_definition,
        source_definition_formula=z3.BoolVal(True),
        definition_formula=z3.BoolVal(True),
    )
    source_definition = (
        z3.BoolVal(True) if source is None
        else _source_partition_definition(shell, dispatch_state, model, source)
    )
    definition = z3.And(base_definition, source_definition)
    return EventCandidateSet(
        horizon=horizon_time, releases=releases, controller=controller,
        hi_deadlines=tuple(hi_deadlines), completion=completion,
        all_candidates=tuple(all_candidates), next_time=next_time,
        base_definition_formula=base_definition,
        source_definition_formula=source_definition,
        definition_formula=definition,
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


def encode_event_step_for_source(
    source_state: SymbolicKernelState,
    destination: SymbolicKernelState,
    model: BoundModel,
    env: SymbolicEnvironment,
    *,
    horizon_time: z3.ArithRef,
    prefix: str,
    event_source: EventSource,
    controller_enabled: bool,
) -> EventStepEncoding:
    """Encode one exact edge of the explicit Event graph.

    ``event_source`` is fixed by host-side graph search.  The arithmetic formula
    proves that this source is the canonical owner of the exact minimum next
    timestamp.  ``controller_enabled`` is a graph-node fact: root nodes are
    split on controller phase, and every later node is enabled exactly when its
    incoming canonical source was CONTROLLER.
    """

    phase_states, closure = _exact_p0_to_p7_closure(
        source_state, model, env, prefix=prefix,
        controller_enabled=controller_enabled,
    )
    dispatch_state = phase_states[-1]
    candidates = build_event_candidates(
        dispatch_state, model, horizon_time=horizon_time, prefix=prefix,
        source=event_source,
    )
    delta = candidates.next_time - source_state.t
    silent_core = _silent_interval_core(dispatch_state, model, candidates)
    formula = z3.And(
        source_state.p == 0,
        source_state.t < horizon_time,
        closure,
        silent_core,
        _event_destination_update(dispatch_state, destination, model, candidates),
    )
    return EventStepEncoding(
        source=source_state, destination=destination, phase_states=phase_states,
        candidates=candidates, delta=delta, closure_formula=closure,
        silent_core_formula=silent_core, formula=formula,
    )



__all__ = [
    "EVENT_KERNEL_VERSION",
    "EventCandidateSet",
    "EventSource",
    "EventStepEncoding",
    "build_event_candidates",
    "encode_event_step_for_source",
    "enumerate_event_sources",
    "state_equality",
]
