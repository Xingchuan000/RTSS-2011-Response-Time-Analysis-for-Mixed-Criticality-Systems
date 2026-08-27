"""Named, reviewable clauses of the V9.1 safe-prefix invariant."""

from __future__ import annotations

import z3

from .symbolic_state import BoundModel, SymbolicKernelState


def state_well_formedness(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    from .symbolic_state import well_formed
    return well_formed(state, model)


def budget_bounds(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    return z3.And(*(z3.And(state.budgets[task.name] >= task.budget_floor,
                            state.budgets[task.name] <= task.budget_upper)
                    for task in model.tasks))


def exact_periodic_eta(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    return z3.And(*(state.eta[task.name] == z3.If(state.t % task.period == 0,
                                                  task.period, state.t % task.period)
                    for task in model.tasks))


def job_field_consistency(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses: list[z3.BoolRef] = []
    for task in model.tasks:
        for slot in range(model.max_jobs_per_task):
            job = state.jobs[(task.name, slot)]
            clauses.extend((z3.Implies(job.present, job.release_time == job.release_index * task.period),
                            z3.Implies(job.present, job.absolute_deadline == job.release_time + task.deadline),
                            z3.Implies(job.present, job.actual_demand <=
                                       (task.c_hi if task.criticality == "HI" else task.c_lo)),
                            z3.Implies(job.present, job.actual_demand >= 1)))
            if task.criticality == "HI":
                clauses.append(z3.Implies(job.present, job.effective_demand == job.actual_demand))
    return z3.And(*clauses)


def no_prior_hi_miss_consistency(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses: list[z3.BoolRef] = []
    for task in model.hi_tasks:
        for job in state.jobs.values():
            if job.task_id != task.priority:
                continue
            clauses.append(z3.Implies(z3.And(state.hi_miss_ledger == 0, job.present),
                                      job.absolute_deadline >= state.t))
    return z3.And(*clauses)


def history_bounds(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    values = []
    for mapping in (state.chi.recent_cost, state.chi.ema_cost, state.chi.max_cost_k):
        values.extend(value <= max(task.c_hi for task in model.tasks)
                     for value in mapping.values())
    for window in (state.chi.mode_change_window, state.chi.lo_cancel_window,
                   state.chi.hi_overrun_window, state.chi.lo_overrun_window,
                   state.chi.job_start_window):
        values.extend((value >= 0, value <= 1) for value in window)
    return z3.And(*(item for value in values for item in (value if isinstance(value, tuple) else (value,))))


def carry_in_consistency(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    # The finite slot representation is itself the carry-in boundary.  Adequacy
    # (including saturation tails) is discharged independently by carry_in.py.
    return z3.And(*(z3.Implies(job.present, job.release_time <= state.t)
                    for job in state.jobs.values()))


def build_psi(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    return z3.And(
        state_well_formedness(state, model),
        budget_bounds(state, model),
        exact_periodic_eta(state, model),
        job_field_consistency(state, model),
        no_prior_hi_miss_consistency(state, model),
        history_bounds(state, model),
        carry_in_consistency(state, model),
    )


__all__ = ["build_psi", "budget_bounds", "carry_in_consistency", "exact_periodic_eta",
           "history_bounds", "job_field_consistency", "no_prior_hi_miss_consistency",
           "state_well_formedness"]
