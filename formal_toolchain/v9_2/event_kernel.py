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
from typing import Collection

import z3

from .controller_encoder import ControllerPolicyCase
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
    encode_p5_controller_enabled_case,
    encode_p5_identity,
    encode_p6_dispatch_case,
)


EVENT_KERNEL_VERSION = "V9_3_MODFREE_DIRECT_DISPATCH_EVENT_GRAPH_V3"


def exact_periodic_countdown(t: z3.ArithRef, period: int) -> z3.ArithRef:
    """Ticks from ``t`` to the next strictly-later phase-zero timestamp.

    Production Event edges use propagated countdown state and do not call this
    helper.  It is used only to initialize the controller clock at the graph
    root and by small refinement obligations.
    """

    period = int(period)
    residue = t % period
    return z3.If(residue == 0, period, period - residue)


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


def _slot_index(model: BoundModel, key: tuple[str, int]) -> int:
    for task_index, task in enumerate(model.tasks):
        for slot in range(model.max_jobs_per_task):
            if (task.name, slot) == key:
                return task_index * model.max_jobs_per_task + slot
    raise KeyError(key)


@dataclass(frozen=True, slots=True)
class DispatchCase:
    """One exact P6 frontier winner used by the Event-graph outer search."""

    case_id: str
    selected_key: tuple[str, int] | None
    dispatch_state: SymbolicKernelState
    formula: z3.BoolRef

    @property
    def running(self) -> bool:
        return self.selected_key is not None


def enumerate_dispatch_cases(
    pre_dispatch_state: SymbolicKernelState,
    model: BoundModel,
    *,
    prefix: str,
    active_task_names: Collection[str] | None = None,
    protected_task_names: Collection[str] = (),
) -> tuple[DispatchCase, ...]:
    """Exact finite P6 partition compiled as winner-specialized relations."""

    active = (
        frozenset(task.name for task in model.tasks)
        if active_task_names is None else frozenset(active_task_names)
    )
    protected = frozenset(protected_task_names)

    selected_keys: list[tuple[str, int] | None] = [None]
    for task in model.tasks:
        if task.name not in active:
            continue
        if task.criticality == "HI":
            slots = (0,)
        elif task.name in protected:
            slots = (1,)
        else:
            slots = tuple(range(model.max_jobs_per_task))
        selected_keys.extend((task.name, slot) for slot in slots)

    rows: list[DispatchCase] = []
    for ordinal, key in enumerate(selected_keys):
        dispatch_state = declare_sparse_successor(
            f"{prefix}.{ordinal}.p7", pre_dispatch_state, model, phase=7,
            mutable=frozenset({"frontier"}),
        )
        formula = encode_p6_dispatch_case(
            pre_dispatch_state, dispatch_state, model,
            selected_key=key,
            active_task_names=active,
            protected_task_names=protected,
        )
        rows.append(DispatchCase(
            case_id=("IDLE" if key is None else f"RUN_{key[0]}_{key[1]}"),
            selected_key=key,
            dispatch_state=dispatch_state,
            formula=formula,
        ))
    return tuple(rows)


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


def enumerate_event_sources(
    model: BoundModel,
    *,
    active_task_names: Collection[str] | None = None,
) -> tuple[EventSource, ...]:
    """Canonical complete source order used by explicit Event-graph search."""

    active = (
        frozenset(task.name for task in model.tasks)
        if active_task_names is None else frozenset(active_task_names)
    )
    rows: list[EventSource] = [EventSource("HORIZON"), EventSource("CONTROLLER")]
    rows.extend(EventSource("RELEASE", task.name) for task in model.tasks if task.name in active)
    for task in model.hi_tasks:
        if task.name in active:
            rows.append(EventSource("HI_DEADLINE", task.name, 0))
    # Completion identity is determined by the already-specialized P6
    # dispatch case, so completion is one generic event source.
    rows.append(EventSource("COMPLETION"))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class EventCandidateSet:
    """Exact next-event candidates in relative-countdown normal form.

    Absolute timestamps are retained only for diagnostics/refinement consumers.
    The source partition itself compares the corresponding ``*_delta`` terms,
    so no fresh periodic quotient/index variables occur in an Event edge.
    """

    horizon: z3.ArithRef
    releases: tuple[tuple[str, z3.ArithRef], ...]
    controller: z3.ArithRef
    hi_deadlines: tuple[tuple[tuple[str, int], z3.ArithRef], ...]
    completion: z3.ArithRef
    all_candidates: tuple[z3.ArithRef, ...]
    next_time: z3.ArithRef
    horizon_delta: z3.ArithRef
    release_deltas: tuple[tuple[str, z3.ArithRef], ...]
    controller_delta: z3.ArithRef
    hi_deadline_deltas: tuple[tuple[tuple[str, int], z3.ArithRef], ...]
    hi_deadline_enabled: tuple[tuple[tuple[str, int], z3.BoolRef], ...]
    completion_delta: z3.ArithRef
    completion_enabled: z3.BoolRef
    all_deltas: tuple[z3.ArithRef, ...]
    next_delta: z3.ArithRef
    base_definition_formula: z3.BoolRef
    source_definition_formula: z3.BoolRef
    definition_formula: z3.BoolRef


@dataclass(frozen=True, slots=True)
class EventStepEncoding:
    """One exact Event edge factored into three independently solvable layers."""

    source: SymbolicKernelState
    destination: SymbolicKernelState
    phase_states: tuple[SymbolicKernelState, ...]
    candidates: EventCandidateSet
    delta: z3.ArithRef
    closure_formula: z3.BoolRef
    source_time_formula: z3.BoolRef
    silent_service_formula: z3.BoolRef
    destination_formula: z3.BoolRef
    formula: z3.BoolRef


@dataclass(frozen=True, slots=True)
class EventNodeClosureEncoding:
    """Exact P0--P5 normalization of one Event-graph node before dispatch."""

    source: SymbolicKernelState
    phase_states: tuple[SymbolicKernelState, ...]
    pre_dispatch_state: SymbolicKernelState
    formula: z3.BoolRef


def _exact_p0_to_p6_closure(
    source: SymbolicKernelState,
    model: BoundModel,
    env: SymbolicEnvironment,
    *,
    prefix: str,
    controller_enabled: bool,
    active_task_names: frozenset[str] | None = None,
    controller_case: ControllerPolicyCase | None = None,
) -> tuple[tuple[SymbolicKernelState, ...], z3.BoolRef]:
    """Execute exact P0..P5, ending at the P6 pre-dispatch state."""

    # Exact SSA-style phase chain.  A phase declares fresh Z3 symbols only for
    # fields it owns; every frame component is structurally shared with the
    # preceding state.  Canonical phase encoders are still used unchanged, so
    # this is existential/frame elimination rather than a new abstraction.
    p1 = declare_sparse_successor(
        f"{prefix}.p1", source, model, phase=1,
        mutable=frozenset({"jobs.present", "jobs.removed", "jobs.ready"}),
        active_task_names=active_task_names,
    )
    p2 = declare_sparse_successor(
        f"{prefix}.p2", p1, model, phase=2, mutable=frozenset({"mode"}),
    )
    p3 = declare_sparse_successor(
        f"{prefix}.p3", p2, model, phase=3, mutable=frozenset({"hi_miss_ledger"}),
    )
    p4 = declare_sparse_successor(
        f"{prefix}.p4", p3, model, phase=4, mutable=frozenset({"eta", "jobs"}),
        active_task_names=active_task_names,
    )
    p5 = declare_sparse_successor(
        f"{prefix}.p5", p4, model, phase=5, mutable=frozenset({"mode"}),
    )
    p6 = declare_sparse_successor(
        f"{prefix}.p6", p5, model, phase=6, mutable=frozenset({"budgets", "history"}),
    )
    states = (p1, p2, p3, p4, p5, p6)
    formula = z3.And(
        encode_p0_settle(source, p1, model, active_task_names=active_task_names),
        encode_p1_idle_recovery(p1, p2, model, active_task_names=active_task_names),
        encode_p2_deadline_observe(p2, p3, model, active_task_names=active_task_names),
        encode_p3_arrival_freeze(
            p3, p4, model, env, active_task_names=active_task_names,
            event_relative_phase=True,
        ),
        encode_p4_mode_switch(p4, p5, model, active_task_names=active_task_names),
        # Event-graph ownership tells us exactly whether this timestamp is a
        # controller activation.  Compile only that exact P5 branch instead of
        # keeping a symbolic enabled/disabled split inside every event macro.
        (
            encode_p5_controller_enabled_case(p5, p6, model, controller_case)
            if controller_enabled and controller_case is not None
            else encode_p5_controller_enabled(p5, p6, model)
            if controller_enabled
            else encode_p5_identity(p5, p6, model)
        ),
    )
    return states, formula


def encode_event_node_closure(
    source_state: SymbolicKernelState,
    model: BoundModel,
    env: SymbolicEnvironment,
    *,
    prefix: str,
    controller_enabled: bool,
    active_task_names: Collection[str] | None = None,
    controller_case: ControllerPolicyCase | None = None,
) -> EventNodeClosureEncoding:
    active = None if active_task_names is None else frozenset(active_task_names)
    phase_states, formula = _exact_p0_to_p6_closure(
        source_state,
        model,
        env,
        prefix=prefix,
        controller_enabled=controller_enabled,
        active_task_names=active,
        controller_case=controller_case,
    )
    return EventNodeClosureEncoding(
        source=source_state,
        phase_states=phase_states,
        pre_dispatch_state=phase_states[-1],
        formula=z3.And(source_state.p == 0, formula),
    )


def event_source_time_formula(
    source_state: SymbolicKernelState,
    candidates: EventCandidateSet,
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    source: EventSource,
    *,
    horizon_time: z3.ArithRef,
) -> z3.BoolRef:
    """Cheap exact source-ordering formula over a prebuilt candidate normal form."""

    return z3.And(
        source_state.t < horizon_time,
        event_source_partition_formula(candidates, dispatch_state, model, source),
        candidates.next_delta >= 1,
        candidates.next_delta <= candidates.horizon_delta,
    )


def encode_event_relative_edge(
    source_state: SymbolicKernelState,
    destination: SymbolicKernelState,
    closure: EventNodeClosureEncoding,
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    *,
    horizon_time: z3.ArithRef,
    controller_delta: z3.ArithRef,
    prefix: str,
    event_source: EventSource,
    active_task_names: Collection[str] | None = None,
    candidates: EventCandidateSet | None = None,
    selected_job_key: tuple[str, int] | None | str = "SYMBOLIC",
) -> EventStepEncoding:
    """Build one exact edge in source-time/service/destination normal form."""

    active = None if active_task_names is None else frozenset(active_task_names)
    if candidates is None:
        candidates = build_event_candidates(
            dispatch_state,
            model,
            horizon_time=horizon_time,
            controller_delta=controller_delta,
            prefix=prefix,
            source=None,
            active_task_names=active,
            selected_job_key=selected_job_key,
        )
    delta = candidates.next_delta
    source_time = event_source_time_formula(
        source_state, candidates, dispatch_state, model, event_source,
        horizon_time=horizon_time,
    )
    silent_service = _silent_interval_service(
        dispatch_state, model, candidates, active_task_names=active,
        selected_job_key=selected_job_key,
    )
    destination_formula = _event_destination_update(
        dispatch_state,
        destination,
        model,
        candidates,
        active_task_names=active,
        selected_job_key=selected_job_key,
    )
    return EventStepEncoding(
        source=source_state,
        destination=destination,
        phase_states=closure.phase_states + (dispatch_state,),
        candidates=candidates,
        delta=delta,
        closure_formula=closure.formula,
        source_time_formula=source_time,
        silent_service_formula=silent_service,
        destination_formula=destination_formula,
        formula=z3.And(
            closure.formula, candidates.base_definition_formula,
            source_time, silent_service, destination_formula,
        ),
    )


def _canonical_candidate_terms(
    candidates: EventCandidateSet,
) -> tuple[tuple[str, z3.ArithRef, z3.BoolRef], ...]:
    rows: list[tuple[str, z3.ArithRef, z3.BoolRef]] = [
        ("HORIZON", candidates.horizon_delta, z3.BoolVal(True)),
        ("CONTROLLER", candidates.controller_delta, z3.BoolVal(True)),
    ]
    rows.extend(
        (f"RELEASE_{name}", delta, z3.BoolVal(True))
        for name, delta in candidates.release_deltas
    )
    enabled_by_key = dict(candidates.hi_deadline_enabled)
    rows.extend(
        (f"HI_DEADLINE_{name}_{slot}", delta, enabled_by_key[(name, slot)])
        for (name, slot), delta in candidates.hi_deadline_deltas
    )
    rows.append(("COMPLETION", candidates.completion_delta, candidates.completion_enabled))
    return tuple(rows)


def _source_partition_definition(
    candidates: EventCandidateSet,
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    source: EventSource,
) -> z3.BoolRef:
    """Exact disjoint first-minimum owner with guarded optional candidates."""

    rows = _canonical_candidate_terms(candidates)
    target_id = source.source_id
    try:
        index = next(i for i, (source_id, _, _) in enumerate(rows) if source_id == target_id)
    except StopIteration as exc:
        raise KeyError(target_id) from exc
    _, chosen, chosen_enabled = rows[index]
    earlier = rows[:index]
    later = rows[index + 1:]
    return z3.And(
        chosen_enabled,
        candidates.next_delta == chosen,
        *(z3.Or(z3.Not(enabled), value > chosen) for _, value, enabled in earlier),
        *(z3.Or(z3.Not(enabled), value >= chosen) for _, value, enabled in later),
    )


def event_source_partition_formula(
    candidates: EventCandidateSet,
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    source: EventSource,
) -> z3.BoolRef:
    """Public exact source-owner constraint over one shared node candidate set."""

    return _source_partition_definition(candidates, dispatch_state, model, source)


def build_event_candidates(
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    *,
    horizon_time: z3.ArithRef,
    controller_delta: z3.ArithRef,
    prefix: str,
    source: EventSource | None = None,
    active_task_names: Collection[str] | None = None,
    selected_job_key: tuple[str, int] | None | str = "SYMBOLIC",
) -> EventCandidateSet:
    """Build exact candidates from relative phase/countdown state.

    P3/P7 already maintains ``eta`` as the exact periodic release age.  At a
    P7 dispatch boundary, the next release offset is therefore simply
    ``T_i - eta_i``.  Reconstructing the same timestamp with a fresh integer
    period index on every edge is redundant and was the dominant Presburger
    arithmetic hotspot.  Controller phase is supplied by the Event-graph
    clock and propagated linearly between nodes.
    """

    t = dispatch_state.t
    active = (
        frozenset(task.name for task in model.tasks)
        if active_task_names is None else frozenset(active_task_names)
    )
    horizon_delta = horizon_time - t

    release_delta_rows: list[tuple[str, z3.ArithRef]] = []
    release_rows: list[tuple[str, z3.ArithRef]] = []
    for task in model.tasks:
        if task.name not in active:
            continue
        delta = z3.IntVal(int(task.period)) - dispatch_state.eta[task.name]
        release_delta_rows.append((task.name, delta))
        release_rows.append((task.name, t + delta))

    hi_deadline_delta_rows: list[tuple[tuple[str, int], z3.ArithRef]] = []
    hi_deadline_enabled_rows: list[tuple[tuple[str, int], z3.BoolRef]] = []
    hi_deadlines: list[tuple[tuple[str, int], z3.ArithRef]] = []
    for task in model.hi_tasks:
        if task.name not in active:
            continue
        for slot in (0,):
            job = dispatch_state.jobs[(task.name, slot)]
            enabled = z3.And(
                job.present,
                z3.Not(job.removed),
                job.executed_service < job.effective_demand,
                job.absolute_deadline > t,
            )
            delta = job.absolute_deadline - t
            hi_deadline_delta_rows.append(((task.name, slot), delta))
            hi_deadline_enabled_rows.append(((task.name, slot), enabled))
            hi_deadlines.append(((task.name, slot), t + delta))

    # Production Event Graph passes a concrete dispatch winner produced by a
    # winner-specialized P6 relation.  Refinement obligations may leave the winner symbolic, in which case this generic
    # expression is used only to prove the dispatch-independent identities.
    if selected_job_key == "SYMBOLIC":
        completion_terms: list[z3.ArithRef] = []
        for key, job in dispatch_state.jobs.items():
            if key[0] not in active:
                continue
            task = model.task_by_name[key[0]]
            if task.criticality == "HI" and key[1] != 0:
                continue
            selected_index = _slot_index(model, key)
            completion_terms.append(z3.If(
                dispatch_state.frontier.selected_slot == selected_index,
                job.effective_demand - job.executed_service,
                0,
            ))
        completion_delta = z3.Sum(*completion_terms) if completion_terms else z3.IntVal(0)
        completion_enabled = dispatch_state.frontier.running
    elif selected_job_key is None:
        completion_delta = z3.IntVal(1)
        completion_enabled = z3.BoolVal(False)
    else:
        key = selected_job_key
        job = dispatch_state.jobs[key]
        completion_delta = job.effective_demand - job.executed_service
        completion_enabled = z3.BoolVal(True)

    release_deltas = tuple(release_delta_rows)
    releases = tuple(release_rows)
    hi_deadline_deltas = tuple(hi_deadline_delta_rows)
    hi_deadline_enabled = tuple(hi_deadline_enabled_rows)
    controller = t + controller_delta
    completion = t + completion_delta
    all_deltas = (
        horizon_delta,
        controller_delta,
        *(value for _, value in release_deltas),
        *(value for _, value in hi_deadline_deltas),
        completion_delta,
    )
    all_candidates = tuple(t + value for value in all_deltas)
    next_delta = z3.Int(f"{prefix}.candidate.next_delta")
    next_time = t + next_delta
    base_definition = z3.And(
        horizon_delta >= 1,
        controller_delta >= 1,
        controller_delta <= int(model.agent_period),
        *(value >= 1 for _, value in release_deltas),
        *(value <= int(model.task_by_name[name].period) for name, value in release_deltas),
        *(z3.Implies(enabled, delta >= 1) for (_, delta), (_, enabled) in zip(hi_deadline_deltas, hi_deadline_enabled)),
        z3.Implies(completion_enabled, completion_delta >= 1),
        next_time == t + next_delta,
    )
    shell = EventCandidateSet(
        horizon=horizon_time, releases=releases, controller=controller,
        hi_deadlines=hi_deadlines, completion=completion,
        all_candidates=all_candidates, next_time=next_time,
        horizon_delta=horizon_delta, release_deltas=release_deltas,
        controller_delta=controller_delta,
        hi_deadline_deltas=hi_deadline_deltas,
        hi_deadline_enabled=hi_deadline_enabled,
        completion_delta=completion_delta, completion_enabled=completion_enabled,
        all_deltas=tuple(all_deltas),
        next_delta=next_delta,
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
        hi_deadlines=hi_deadlines, completion=completion,
        all_candidates=all_candidates, next_time=next_time,
        horizon_delta=horizon_delta, release_deltas=release_deltas,
        controller_delta=controller_delta,
        hi_deadline_deltas=hi_deadline_deltas,
        hi_deadline_enabled=hi_deadline_enabled,
        completion_delta=completion_delta, completion_enabled=completion_enabled,
        all_deltas=tuple(all_deltas),
        next_delta=next_delta,
        base_definition_formula=base_definition,
        source_definition_formula=source_definition,
        definition_formula=definition,
    )


def _silent_interval_service(
    dispatch_state: SymbolicKernelState,
    model: BoundModel,
    candidates: EventCandidateSet,
    *,
    active_task_names: Collection[str] | None = None,
    selected_job_key: tuple[str, int] | None | str = "SYMBOLIC",
) -> z3.BoolRef:
    """Only the processor-service side conditions of one silent interval."""

    delta = candidates.next_delta
    clauses: list[z3.BoolRef] = []

    active = (
        frozenset(task.name for task in model.tasks)
        if active_task_names is None else frozenset(active_task_names)
    )
    if selected_job_key is None:
        return z3.BoolVal(True)
    if selected_job_key != "SYMBOLIC":
        job = dispatch_state.jobs[selected_job_key]
        return z3.And(
            job.present,
            z3.Not(job.removed),
            job.ready,
            job.effective_demand > job.executed_service,
            delta <= job.effective_demand - job.executed_service,
        )

    selected_remaining_terms: list[z3.BoolRef] = []
    for key, job in dispatch_state.jobs.items():
        if key[0] not in active:
            continue
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
    active_task_names: Collection[str] | None = None,
    selected_job_key: tuple[str, int] | None | str = "SYMBOLIC",
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
    delta = candidates.next_delta
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

    active = (
        frozenset(task.name for task in model.tasks)
        if active_task_names is None else frozenset(active_task_names)
    )
    for task in model.tasks:
        clauses.extend((
            destination.budgets[task.name] == choose(
                dispatch_state.budgets[task.name],
                None if source is None else source.budgets[task.name],
            ),
            destination.eta[task.name] == choose(
                (
                    z3.If(
                        dispatch_state.eta[task.name] + delta < task.period,
                        dispatch_state.eta[task.name] + delta,
                        task.period,
                    )
                    if task.name in active else dispatch_state.eta[task.name]
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
        if key[0] not in active:
            continue
        other = destination.jobs[key]
        source_job = None if source is None else source.jobs[key]
        index = _slot_index(model, key)
        if selected_job_key == "SYMBOLIC":
            selected = dispatch_state.frontier.selected_slot == index
        else:
            selected = z3.BoolVal(selected_job_key == key)
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
    *,
    active_task_names: Collection[str] | None = None,
) -> z3.BoolRef:
    """Exact quotient of repeated P7 ticks over an event-free interval."""

    return z3.And(
        candidates.definition_formula,
        candidates.next_delta >= 1,
        candidates.next_delta <= candidates.horizon_delta,
        _silent_interval_service(
            dispatch_state, model, candidates, active_task_names=active_task_names
        ),
        _event_destination_update(
            dispatch_state, destination, model, candidates,
            active_task_names=active_task_names,
        ),
    )




__all__ = [
    "EVENT_KERNEL_VERSION",
    "DispatchCase",
    "EventCandidateSet",
    "EventNodeClosureEncoding",
    "EventSource",
    "EventStepEncoding",
    "build_event_candidates",
    "encode_event_node_closure",
    "encode_event_relative_edge",
    "enumerate_dispatch_cases",
    "enumerate_event_sources",
    "event_source_partition_formula",
    "event_source_time_formula",
    "exact_periodic_countdown",
    "state_equality",
]
