"""Canonical eight-phase transition relation for the V9.2 kernel."""

from __future__ import annotations

from typing import Callable

import z3

from .controller_encoder import encode_controller_decision
from .environment_encoder import (
    SymbolicEnvironment, classify_from_actual_demand, demand_for_time,
)
from .symbolic_state import BoundModel, SymbolicJob, SymbolicKernelState


def _job_fields(job: SymbolicJob):
    return (job.present, job.release_index, job.release_time, job.absolute_deadline,
            job.tie_break, job.release_entry_mode_hi, job.classification_abnormal,
            job.budget_at_release, job.actual_demand, job.effective_demand,
            job.executed_service, job.removed, job.ready)


def _frame_eq(left: z3.ExprRef, right: z3.ExprRef) -> z3.BoolRef | None:
    """Return an equality only when SSA sharing has not made it tautological."""
    return None if left.eq(right) else left == right


def _append_frame(clauses: list[z3.BoolRef], left: z3.ExprRef, right: z3.ExprRef) -> None:
    equality = _frame_eq(left, right)
    if equality is not None:
        clauses.append(equality)


def _frame_state(
    z: SymbolicKernelState,
    zp: SymbolicKernelState,
    model: BoundModel,
    *,
    mutable: frozenset[str] = frozenset(),
) -> list[z3.BoolRef]:
    """Frame state fields, omitting identities already shared by SSA states."""

    clauses: list[z3.BoolRef] = []
    if "t" not in mutable:
        _append_frame(clauses, zp.t, z.t)
    if "mode" not in mutable:
        _append_frame(clauses, zp.mode_hi, z.mode_hi)
    if "hi_miss_ledger" not in mutable:
        _append_frame(clauses, zp.hi_miss_ledger, z.hi_miss_ledger)
    if "frontier" not in mutable:
        _append_frame(clauses, zp.frontier.selected_slot, z.frontier.selected_slot)
        _append_frame(clauses, zp.frontier.running, z.frontier.running)
    if "budgets" not in mutable:
        for name in z.budgets:
            _append_frame(clauses, zp.budgets[name], z.budgets[name])
    if "eta" not in mutable:
        for name in z.eta:
            _append_frame(clauses, zp.eta[name], z.eta[name])

    job_field_names = (
        "present", "release_index", "release_time", "absolute_deadline",
        "tie_break", "release_entry_mode_hi", "classification_abnormal",
        "budget_at_release", "actual_demand", "effective_demand",
        "executed_service", "removed", "ready",
    )
    for key, job in z.jobs.items():
        other = zp.jobs[key]
        for field_name in job_field_names:
            if f"jobs.{field_name}" in mutable or "jobs" in mutable:
                continue
            _append_frame(clauses, getattr(other, field_name), getattr(job, field_name))

    if "history" not in mutable:
        for left, right in (
            (zp.chi.recent_cost, z.chi.recent_cost),
            (zp.chi.ema_cost, z.chi.ema_cost),
            (zp.chi.overrun_ema, z.chi.overrun_ema),
            (zp.chi.max_cost_k, z.chi.max_cost_k),
        ):
            for name in right:
                _append_frame(clauses, left[name], right[name])
        for left_values, right_values in (
            (zp.chi.mode_change_window, z.chi.mode_change_window),
            (zp.chi.lo_cancel_window, z.chi.lo_cancel_window),
            (zp.chi.hi_overrun_window, z.chi.hi_overrun_window),
            (zp.chi.lo_overrun_window, z.chi.lo_overrun_window),
            (zp.chi.job_start_window, z.chi.job_start_window),
        ):
            for left, right in zip(left_values, right_values):
                _append_frame(clauses, left, right)
    return clauses


def _phase(z: SymbolicKernelState, zp: SymbolicKernelState, current: int, next_phase: int) -> list[z3.BoolRef]:
    return [z.p == current, zp.p == next_phase]


def encode_p0_settle(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 0, 1) + _frame_state(
        z, zp, model, mutable=frozenset({"jobs.present", "jobs.removed", "jobs.ready"})
    )
    for key, job in z.jobs.items():
        other = zp.jobs[key]
        completed = job.present & (job.executed_service >= job.effective_demand)
        clauses.extend((other.present == z3.And(job.present, z3.Not(completed)),
                        other.removed == z3.Or(job.removed, completed),
                        other.ready == z3.And(job.ready, z3.Not(completed))))
    return z3.And(*clauses)


def encode_p1_idle_recovery(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 1, 2) + _frame_state(z, zp, model, mutable=frozenset({"mode"}))
    active = z3.Or(*(job.present & (job.remaining > 0) for job in z.jobs.values()))
    clauses.append(zp.mode_hi == z3.And(z.mode_hi, active))
    return z3.And(*clauses)


def encode_p2_deadline_observe(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 2, 3) + _frame_state(
        z, zp, model, mutable=frozenset({"hi_miss_ledger"})
    )
    misses = []
    for task in model.tasks:
        if task.criticality != "HI":
            continue
        misses.extend(job.present & (job.absolute_deadline == z.t) &
                      (job.executed_service < job.effective_demand)
                      for (name, _), job in z.jobs.items() if name == task.name)
    new_miss = z3.Or(*misses) if misses else z3.BoolVal(False)
    clauses.append(zp.hi_miss_ledger == z.hi_miss_ledger + z3.If(new_miss, 1, 0))
    # No clause changes present/removed here: deadline observation is observe-only.
    return z3.And(*clauses)


def _release_due(z: SymbolicKernelState, task) -> z3.BoolRef:
    return z3.And(z.t % task.period == 0, z.eta[task.name] == task.period)


def _remaining(job: SymbolicJob) -> z3.ArithRef:
    return z3.If(
        z3.And(job.present, job.effective_demand > job.executed_service),
        job.effective_demand - job.executed_service,
        0,
    )


def _encode_lo_aggregate_after_release(
    z: SymbolicKernelState,
    zp: SymbolicKernelState,
    task,
    *,
    due: z3.BoolRef,
) -> list[z3.BoolRef]:
    aggregate = z.jobs[(task.name, 0)]
    exact = z.jobs[(task.name, 1)]
    other = zp.jobs[(task.name, 0)]
    folded = _remaining(aggregate) + _remaining(exact)
    present_after = folded > 0
    degraded = int(task.degraded_cost or task.c_lo)
    return [
        other.present == z3.If(due, present_after, aggregate.present),
        other.release_index == z3.If(due, -1, aggregate.release_index),
        other.release_time == z3.If(due, z.t, aggregate.release_time),
        # LO aggregate deadlines are observationally irrelevant to HI safety;
        # keep a canonical timestamp solely for structural well-formedness.
        other.absolute_deadline == z3.If(due, z.t, aggregate.absolute_deadline),
        other.tie_break == z3.If(due, -1, aggregate.tie_break),
        other.release_entry_mode_hi == z3.If(due, z.mode_hi, aggregate.release_entry_mode_hi),
        other.classification_abnormal == z3.If(due, False, aggregate.classification_abnormal),
        other.budget_at_release == z3.If(due, z.budgets[task.name], aggregate.budget_at_release),
        other.actual_demand == z3.If(due, z3.If(present_after, folded, 1), aggregate.actual_demand),
        other.effective_demand == z3.If(due, z3.If(present_after, folded, 1), aggregate.effective_demand),
        other.executed_service == z3.If(due, 0, aggregate.executed_service),
        other.removed == z3.If(due, z3.Not(present_after), aggregate.removed),
        other.ready == z3.If(due, present_after, aggregate.ready),
        # Keep the frozen degraded value materially consumed in this relation;
        # it also rejects malformed bindings before any proof starts.
        z3.BoolVal(1 <= degraded <= task.c_lo),
    ]


def _encode_exact_release_slot(
    z: SymbolicKernelState,
    zp: SymbolicKernelState,
    task,
    slot: int,
    *,
    due: z3.BoolRef,
    demand: z3.ArithRef,
) -> list[z3.BoolRef]:
    job = z.jobs[(task.name, slot)]
    other = zp.jobs[(task.name, slot)]
    release_index = z.t / task.period
    degraded = int(task.degraded_cost or task.c_lo)
    effective = (
        demand
        if task.criticality == "HI"
        else z3.If(
            z.mode_hi,
            z3.If(demand < degraded, demand, degraded),
            z3.If(demand < z.budgets[task.name] + 1, demand, z.budgets[task.name] + 1),
        )
    )
    budget_at_release = z3.If(
        z.mode_hi if task.criticality == "LO" else z3.BoolVal(False),
        degraded,
        z.budgets[task.name],
    )
    return [
        other.present == z3.If(due, True, job.present),
        other.release_index == z3.If(due, release_index, job.release_index),
        other.release_time == z3.If(due, z.t, job.release_time),
        other.absolute_deadline == z3.If(due, z.t + task.deadline, job.absolute_deadline),
        other.tie_break == z3.If(due, release_index, job.tie_break),
        other.release_entry_mode_hi == z3.If(due, z.mode_hi, job.release_entry_mode_hi),
        other.classification_abnormal == z3.If(
            due, classify_from_actual_demand(demand, task), job.classification_abnormal
        ),
        other.budget_at_release == z3.If(due, budget_at_release, job.budget_at_release),
        other.actual_demand == z3.If(due, demand, job.actual_demand),
        other.effective_demand == z3.If(due, effective, job.effective_demand),
        other.executed_service == z3.If(due, 0, job.executed_service),
        other.removed == z3.If(due, False, job.removed),
        other.ready == z3.If(due, True, job.ready),
    ]


def encode_p3_arrival_freeze(
    z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel, env: SymbolicEnvironment
) -> z3.BoolRef:
    if model.max_jobs_per_task < 2 and any(task.criticality == "LO" for task in model.tasks):
        raise ValueError("V9_2_TWO_SLOT_LO_CARRY_IN_REQUIRES_TWO_JOB_SLOTS")
    clauses = _phase(z, zp, 3, 4) + _frame_state(
        z, zp, model, mutable=frozenset({"eta", "jobs"})
    )
    for task in model.tasks:
        due = _release_due(z, task)
        demand, covered = demand_for_time(env, task, z.t)
        clauses.append(z3.Implies(due, covered))
        clauses.append(zp.eta[task.name] == z3.If(due, 0, z.eta[task.name]))
        if task.criticality == "HI":
            exact_slot = 0
            # In a NoPriorHIMiss prefix with D<=T the preceding HI job must
            # have settled before the next release.  Refuse overlap rather than
            # silently replacing an exact HI job.
            clauses.append(z3.Implies(due, z3.Not(z.jobs[(task.name, exact_slot)].present)))
            clauses.extend(_encode_exact_release_slot(
                z, zp, task, exact_slot, due=due, demand=demand
            ))
            for slot in range(1, model.max_jobs_per_task):
                job = z.jobs[(task.name, slot)]
                other = zp.jobs[(task.name, slot)]
                clauses.extend(a == b for a, b in zip(_job_fields(other), _job_fields(job)))
        else:
            clauses.extend(_encode_lo_aggregate_after_release(z, zp, task, due=due))
            clauses.extend(_encode_exact_release_slot(
                z, zp, task, 1, due=due, demand=demand
            ))
            for slot in range(2, model.max_jobs_per_task):
                job = z.jobs[(task.name, slot)]
                other = zp.jobs[(task.name, slot)]
                clauses.extend(a == b for a, b in zip(_job_fields(other), _job_fields(job)))
    return z3.And(*clauses)


def encode_p4_mode_switch(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 4, 5) + _frame_state(z, zp, model, mutable=frozenset({"mode"}))
    abnormal_batch = z3.Or(*(job.present & (job.release_time == z.t) & job.classification_abnormal
                             for task in model.hi_tasks
                             for (name, _), job in z.jobs.items() if name == task.name))
    clauses.append(zp.mode_hi == z3.Or(z.mode_hi, abnormal_batch))
    return z3.And(*clauses)


def _history_domain(state: SymbolicKernelState, model: BoundModel) -> list[z3.BoolRef]:
    """Conservative deployed-history abstraction used at controller boundaries.

    Concrete history is a subset of this domain.  Allowing additional history
    values can create spurious SAT counterexamples but cannot hide an unsafe
    controller behavior behind an UNSAT result.
    """

    clauses: list[z3.BoolRef] = []
    for task in model.tasks:
        upper = task.history_cost_upper
        clauses.extend((
            state.chi.recent_cost[task.name] >= 0,
            state.chi.recent_cost[task.name] <= upper,
            state.chi.ema_cost[task.name] >= 0,
            state.chi.ema_cost[task.name] <= upper,
            state.chi.max_cost_k[task.name] >= 0,
            state.chi.max_cost_k[task.name] <= upper,
            state.chi.overrun_ema[task.name] >= 0,
            state.chi.overrun_ema[task.name] <= 1,
        ))
    for window in (
        state.chi.mode_change_window, state.chi.lo_cancel_window,
        state.chi.hi_overrun_window, state.chi.lo_overrun_window,
        state.chi.job_start_window,
    ):
        clauses.extend(value >= 0 for value in window)
    return clauses


def _copy_history(z: SymbolicKernelState, zp: SymbolicKernelState) -> list[z3.BoolRef]:
    clauses: list[z3.BoolRef] = []
    for left, right in (
        (zp.chi.recent_cost, z.chi.recent_cost),
        (zp.chi.ema_cost, z.chi.ema_cost),
        (zp.chi.overrun_ema, z.chi.overrun_ema),
        (zp.chi.max_cost_k, z.chi.max_cost_k),
    ):
        clauses.extend(left[name] == right[name] for name in right)
    for left_values, right_values in (
        (zp.chi.mode_change_window, z.chi.mode_change_window),
        (zp.chi.lo_cancel_window, z.chi.lo_cancel_window),
        (zp.chi.hi_overrun_window, z.chi.hi_overrun_window),
        (zp.chi.lo_overrun_window, z.chi.lo_overrun_window),
        (zp.chi.job_start_window, z.chi.job_start_window),
    ):
        clauses.extend(left == right for left, right in zip(left_values, right_values))
    return clauses


def encode_p5_identity(
    z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel
) -> z3.BoolRef:
    """Exact deployed P5 relation when the periodic controller is disabled."""

    return z3.And(*(_phase(z, zp, 5, 6) + _frame_state(z, zp, model)))


def encode_p5_invariant_summary(
    z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel
) -> z3.BoolRef:
    """Sound over-approximation of P5 used only for Psi inductiveness.

    The exact deployed controller remains encoded by :func:`encode_p5_controller`
    in every FirstBadWindow.  For invariant preservation, observation/tree details
    are irrelevant once separate machine obligations establish that FirstValid can
    only choose a mask-valid candidate and every such candidate obeys controller
    budget bounds.  Enabled P5 therefore permits *any* bounded post budget and any
    post history in the conservative history domain; disabled P5 is exact identity.
    Proving Psi under this superset is stronger than proving it under the exact P5.
    """

    enabled = (z.t % model.agent_period) == 0
    clauses = _phase(z, zp, 5, 6) + _frame_state(
        z, zp, model, mutable=frozenset({"budgets", "history"})
    )
    history_identity = z3.And(*_copy_history(z, zp))
    clauses.append(z3.Implies(enabled, z3.And(*_history_domain(zp, model))))
    clauses.append(z3.Implies(z3.Not(enabled), history_identity))
    for task in model.tasks:
        clauses.append(z3.Implies(
            enabled,
            z3.And(
                zp.budgets[task.name] >= task.budget_floor,
                zp.budgets[task.name] <= task.budget_upper,
            ),
        ))
        clauses.append(z3.Implies(
            z3.Not(enabled), zp.budgets[task.name] == z.budgets[task.name]
        ))
    return z3.And(*clauses)


def encode_p5_controller(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    """Exact deployed P5 relation.

    Keep the controller-owned effect separate from the phase frame.  Event
    windows can then pool the expensive observation/tree/mask relation over
    only the fields that P5 actually reads or writes, while the event-local
    SSA chain supplies the unchanged Full-state frame exactly.  The conjunction
    here is definitionally identical to the former monolithic relation.
    """

    return z3.And(
        encode_p5_controller_effect(z, zp, model),
        encode_p5_controller_frame(z, zp, model),
    )


def encode_p5_controller_effect(
    z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel
) -> z3.BoolRef:
    """Exact P5 projection onto controller-owned state.

    Production P5 observes only ``t``, budgets and RuntimeFeatureState history,
    and it writes only budgets/history.  Jobs, eta, mode, frontier and the miss
    ledger are pure frame fields.  Separating those frame equalities is formula
    factoring only; :func:`encode_p5_controller` still exposes the complete
    Full-kernel relation.
    """

    decision = encode_controller_decision(z, model)
    clauses = _phase(z, zp, 5, 6)
    # The deployed agent is periodic.  Outside an activation timestamp P5 is an
    # identity budget transition.  At an activation timestamp every auxiliary
    # policy equation is enforced and the selected candidate becomes B'.
    clauses.extend(z3.Implies(decision.enabled, clause) for clause in decision.constraints)
    clauses.extend(
        zp.budgets[name]
        == z3.If(decision.enabled, decision.budget_after[name], z.budgets[name])
        for name in z.budgets
    )
    # RuntimeFeatureState is updated once per controller step.  For HI-safety
    # proof we deliberately over-approximate that update by the complete
    # numeric history domain; when P5 is disabled history is exact identity.
    history_identity = z3.And(*_copy_history(z, zp))
    clauses.append(z3.Implies(decision.enabled, z3.And(*_history_domain(zp, model))))
    clauses.append(z3.Implies(z3.Not(decision.enabled), history_identity))
    return z3.And(*clauses)


def encode_p5_controller_frame(
    z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel
) -> z3.BoolRef:
    """Exact non-controller frame of P5.

    ``budgets`` and ``history`` are intentionally excluded because they are the
    controller-owned effect encoded by :func:`encode_p5_controller_effect`.
    With sparse P5 successors this normally simplifies to ``True`` because the
    unchanged fields are structurally shared rather than re-declared.
    """

    return z3.And(*_frame_state(
        z, zp, model, mutable=frozenset({"budgets", "history"})
    ))


def _slot_index(model: BoundModel, key: tuple[str, int]) -> int:
    return next(index for index, item in enumerate(model.tasks) for slot in range(model.max_jobs_per_task)
                if (item.name, slot) == key for index in [index * model.max_jobs_per_task + slot])


def encode_p6_dispatch(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    """Exact fixed-priority dispatch with linear task-priority structure.

    Task priorities are static and already stored in canonical order.  The old
    encoding compared every job slot against every other slot and then folded
    all winner booleans through a nested ITE.  This version first chooses the
    highest-priority task with any eligible job, then resolves only that task's
    local slots by tie_break (and slot order for an exact tie, matching the old
    winner-fold behavior).
    """

    clauses = _phase(z, zp, 6, 7) + _frame_state(
        z, zp, model, mutable=frozenset({"frontier"})
    )
    eligible: dict[tuple[str, int], z3.BoolRef] = {}
    task_has: dict[str, z3.BoolRef] = {}
    for task in model.tasks:
        rows: list[z3.BoolRef] = []
        for slot in range(model.max_jobs_per_task):
            key = (task.name, slot)
            job = z.jobs[key]
            value = job.present & job.ready & (job.remaining > 0) & z3.Not(job.removed)
            eligible[key] = value
            rows.append(value)
        task_has[task.name] = z3.Or(*rows)

    any_eligible = z3.Or(*(task_has[task.name] for task in model.tasks))
    clauses.append(zp.frontier.running == any_eligible)
    clauses.append(z3.Implies(z3.Not(any_eligible), zp.frontier.selected_slot == -1))

    higher_task_busy = z3.BoolVal(False)
    for task in model.tasks:
        task_wins = z3.And(task_has[task.name], z3.Not(higher_task_busy))
        for slot in range(model.max_jobs_per_task):
            key = (task.name, slot)
            job = z.jobs[key]
            index = _slot_index(model, key)
            local_predecessors: list[z3.BoolRef] = []
            for other_slot in range(model.max_jobs_per_task):
                if other_slot == slot:
                    continue
                other_key = (task.name, other_slot)
                other = z.jobs[other_key]
                other_index = _slot_index(model, other_key)
                local_predecessors.append(z3.And(
                    eligible[other_key],
                    z3.Or(
                        other.tie_break < job.tie_break,
                        z3.And(other.tie_break == job.tie_break, other_index < index),
                    ),
                ))
            local_wins = z3.And(
                eligible[key],
                *(z3.Not(value) for value in local_predecessors),
            )
            clauses.append(z3.Implies(
                z3.And(task_wins, local_wins),
                zp.frontier.selected_slot == index,
            ))
        higher_task_busy = z3.Or(higher_task_busy, task_has[task.name])

    return z3.And(*clauses)


def encode_p7_time_and_service(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 7, 0) + _frame_state(
        z, zp, model, mutable=frozenset({"t", "eta", "frontier", "jobs.executed_service"})
    )
    clauses.extend((zp.t == z.t + 1, zp.p == 0,
                    zp.frontier.selected_slot == -1,
                    zp.frontier.running == z3.BoolVal(False)))
    for task in model.tasks:
        clauses.append(zp.eta[task.name] == z3.If(z.eta[task.name] < task.period,
                                                   z.eta[task.name] + 1, task.period))
        for slot in range(model.max_jobs_per_task):
            job = z.jobs[(task.name, slot)]
            other = zp.jobs[(task.name, slot)]
            index = _slot_index(model, (task.name, slot))
            clauses.append(other.executed_service == job.executed_service +
                           z3.If(z.frontier.selected_slot == index, 1, 0))
    return z3.And(*clauses)



def encode_phase_step(
    z: SymbolicKernelState,
    zp: SymbolicKernelState,
    model: BoundModel,
    env: SymbolicEnvironment,
    *,
    phase: int,
    controller_may_fire: bool = True,
) -> z3.BoolRef:
    """Encode one statically known phase without constructing seven dead branches.

    Finite first-bad windows start at P0 and follow the canonical P0..P7
    sequence, so their phase is known from the unroll index.  Using this
    dispatcher is logically identical to ``encode_step`` under ``z.p=phase``
    but avoids rebuilding the full eight-way disjunction at every microstep.
    ``controller_may_fire=False`` is permitted only when the caller has proved
    the absolute-time congruence cannot satisfy the deployed controller period.
    In that case P5 is exact identity on every state component except ``p``.
    """

    phase = int(phase)
    if phase == 0:
        return encode_p0_settle(z, zp, model)
    if phase == 1:
        return encode_p1_idle_recovery(z, zp, model)
    if phase == 2:
        return encode_p2_deadline_observe(z, zp, model)
    if phase == 3:
        return encode_p3_arrival_freeze(z, zp, model, env)
    if phase == 4:
        return encode_p4_mode_switch(z, zp, model)
    if phase == 5:
        if controller_may_fire:
            return encode_p5_controller(z, zp, model)
        return z3.And(*(_phase(z, zp, 5, 6) + _frame_state(z, zp, model)))
    if phase == 6:
        return encode_p6_dispatch(z, zp, model)
    if phase == 7:
        return encode_p7_time_and_service(z, zp, model)
    raise ValueError(f"invalid V9.2 kernel phase: {phase}")

def encode_step(
    z: SymbolicKernelState,
    zp: SymbolicKernelState,
    model: BoundModel,
    env: SymbolicEnvironment,
) -> z3.BoolRef:
    """The only V9.2 transition relation: exactly the canonical P0..P7 disjunction."""

    return z3.Or(
        encode_p0_settle(z, zp, model),
        encode_p1_idle_recovery(z, zp, model),
        encode_p2_deadline_observe(z, zp, model),
        encode_p3_arrival_freeze(z, zp, model, env),
        encode_p4_mode_switch(z, zp, model),
        encode_p5_controller(z, zp, model),
        encode_p6_dispatch(z, zp, model),
        encode_p7_time_and_service(z, zp, model),
    )


__all__ = [
    "encode_p0_settle", "encode_p1_idle_recovery", "encode_p2_deadline_observe",
    "encode_p3_arrival_freeze", "encode_p4_mode_switch", "encode_p5_identity",
    "encode_p5_controller", "encode_p5_controller_effect", "encode_p5_controller_frame",
    "encode_p6_dispatch", "encode_p7_time_and_service",
    "encode_phase_step", "encode_step",
]
