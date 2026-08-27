"""Canonical eight-phase transition relation for the V9.1 kernel."""

from __future__ import annotations

from typing import Callable

import z3

from .environment_encoder import SymbolicEnvironment, classify_from_actual_demand
from .symbolic_state import BoundModel, SymbolicJob, SymbolicKernelState


def _job_fields(job: SymbolicJob):
    return (job.present, job.release_index, job.release_time, job.absolute_deadline,
            job.tie_break, job.release_entry_mode_hi, job.classification_abnormal,
            job.budget_at_release, job.actual_demand, job.effective_demand,
            job.executed_service, job.removed, job.ready)


def _copy_state(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> list[z3.BoolRef]:
    clauses: list[z3.BoolRef] = [zp.t == z.t, zp.mode_hi == z.mode_hi,
                                  zp.hi_miss_ledger == z.hi_miss_ledger,
                                  zp.frontier.selected_slot == z.frontier.selected_slot,
                                  zp.frontier.running == z.frontier.running]
    clauses.extend(zp.budgets[name] == z.budgets[name] for name in z.budgets)
    clauses.extend(zp.eta[name] == z.eta[name] for name in z.eta)
    for key, job in z.jobs.items():
        other = zp.jobs[key]
        clauses.extend(a == b for a, b in zip(_job_fields(other), _job_fields(job)))
    for left, right in (
        (zp.chi.recent_cost, z.chi.recent_cost),
        (zp.chi.ema_cost, z.chi.ema_cost),
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


def _phase(z: SymbolicKernelState, zp: SymbolicKernelState, current: int, next_phase: int) -> list[z3.BoolRef]:
    return [z.p == current, zp.p == next_phase]


def encode_p0_settle(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 0, 1) + _copy_state(z, zp, model)
    for key, job in z.jobs.items():
        other = zp.jobs[key]
        completed = job.present & (job.executed_service >= job.effective_demand)
        clauses.extend((other.present == z3.And(job.present, z3.Not(completed)),
                        other.removed == z3.Or(job.removed, completed),
                        other.ready == z3.And(job.ready, z3.Not(completed))))
    return z3.And(*clauses)


def encode_p1_idle_recovery(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 1, 2) + _copy_state(z, zp, model)
    active = z3.Or(*(job.present & (job.remaining > 0) for job in z.jobs.values()))
    clauses.append(zp.mode_hi == z3.And(z.mode_hi, active))
    return z3.And(*clauses)


def encode_p2_deadline_observe(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 2, 3) + _copy_state(z, zp, model)
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


def _release_expr(z: SymbolicKernelState, job: SymbolicJob, task, slot: int) -> z3.BoolRef:
    return z3.And(z.t == slot * task.period, z.eta[task.name] == task.period,
                  z3.Not(job.present), z3.Not(job.removed))


def encode_p3_arrival_freeze(
    z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel, env: SymbolicEnvironment
) -> z3.BoolRef:
    clauses = _phase(z, zp, 3, 4) + _copy_state(z, zp, model)
    for task_id, task in enumerate(model.tasks):
        clauses.append(zp.eta[task.name] == z3.If(
            z3.Or(*(_release_expr(z, z.jobs[(task.name, slot)], task, slot)
                    for slot in range(model.max_jobs_per_task))),
            0, z.eta[task.name]))
        for slot in range(model.max_jobs_per_task):
            job = z.jobs[(task.name, slot)]
            other = zp.jobs[(task.name, slot)]
            release = _release_expr(z, job, task, slot)
            demand = env.actual_demands.get((task.name, slot))
            if demand is None:
                clauses.append(z3.Not(release))
                continue
            degraded = max(1, min(task.c_lo, task.c_lo // 2))
            effective = z3.If(
                task.criticality == "HI",
                demand,
                z3.If(z.mode_hi, z3.If(demand < degraded, demand, degraded),
                      z3.If(demand < z.budgets[task.name] + 1, demand, z.budgets[task.name] + 1)),
            )
            clauses.extend((other.present == z3.Or(job.present, release),
                            other.release_index == z3.If(release, slot, job.release_index),
                            other.release_time == z3.If(release, z.t, job.release_time),
                            other.absolute_deadline == z3.If(release, z.t + task.deadline, job.absolute_deadline),
                            other.tie_break == z3.If(release, slot, job.tie_break),
                            other.release_entry_mode_hi == z3.If(release, z.mode_hi, job.release_entry_mode_hi),
                            other.classification_abnormal == z3.If(
                                release, classify_from_actual_demand(demand, task), job.classification_abnormal),
                            other.budget_at_release == z3.If(release, z.budgets[task.name], job.budget_at_release),
                            other.actual_demand == z3.If(release, demand, job.actual_demand),
                            other.effective_demand == z3.If(release, effective, job.effective_demand),
                            other.executed_service == z3.If(release, 0, job.executed_service),
                            other.removed == z3.If(release, False, job.removed),
                            other.ready == z3.If(release, True, job.ready)))
    return z3.And(*clauses)


def encode_p4_mode_switch(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 4, 5) + _copy_state(z, zp, model)
    abnormal_batch = z3.Or(*(job.present & (job.release_time == z.t) & job.classification_abnormal
                             for task in model.hi_tasks
                             for (name, _), job in z.jobs.items() if name == task.name))
    clauses.append(zp.mode_hi == z3.Or(z.mode_hi, abnormal_batch))
    return z3.And(*clauses)


def encode_p5_controller(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    # Task C replaces only this phase's budget/action equations; its boundary
    # remains fixed here so no alternate V8/V9 transition relation can arise.
    return z3.And(*(_phase(z, zp, 5, 6) + _copy_state(z, zp, model)))


def _slot_index(model: BoundModel, key: tuple[str, int]) -> int:
    return next(index for index, item in enumerate(model.tasks) for slot in range(model.max_jobs_per_task)
                if (item.name, slot) == key for index in [index * model.max_jobs_per_task + slot])


def encode_p6_dispatch(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 6, 7) + _copy_state(z, zp, model)
    jobs = list(z.jobs.items())
    eligible = {key: job.present & job.ready & (job.remaining > 0) & z3.Not(job.removed)
                for key, job in jobs}
    winners = []
    for key, job in jobs:
        higher = [eligible[other_key] for other_key, other in jobs
                  if (other.priority, other.tie_break.as_long() if z3.is_rational_value(other.tie_break) else 0)
                  < (job.priority, 0) or
                  (other.priority == job.priority and other_key[1] < key[1])]
        winners.append(eligible[key] & z3.And(*(z3.Not(value) for value in higher)))
    selected = -1
    for (key, _), winner in reversed(list(zip(jobs, winners))):
        selected = z3.If(winner, _slot_index(model, key), selected)
    clauses.extend((zp.frontier.selected_slot == selected,
                    zp.frontier.running == z3.Or(*eligible.values())))
    return z3.And(*clauses)


def encode_p7_time_and_service(z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses = _phase(z, zp, 7, 0) + _copy_state(z, zp, model)
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


def encode_step(
    z: SymbolicKernelState,
    zp: SymbolicKernelState,
    model: BoundModel,
    env: SymbolicEnvironment,
) -> z3.BoolRef:
    """The only V9.1 transition relation: exactly the canonical P0..P7 disjunction."""

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
    "encode_p3_arrival_freeze", "encode_p4_mode_switch", "encode_p5_controller",
    "encode_p6_dispatch", "encode_p7_time_and_service", "encode_step",
]
