"""Named, reviewable clauses of the V9.2 safe-prefix invariant."""

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
    """Phase-aware release age for exact-periodic phase-zero releases."""

    clauses = []
    for task in model.tasks:
        residue = state.t % task.period
        expected = z3.If(
            residue == 0,
            z3.If(state.p <= 3, task.period, 0),
            residue,
        )
        clauses.append(state.eta[task.name] == expected)
    return z3.And(*clauses)


def job_field_consistency(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses: list[z3.BoolRef] = []
    for task in model.tasks:
        for slot in range(model.max_jobs_per_task):
            job = state.jobs[(task.name, slot)]
            lo_aggregate = task.criticality == "LO" and slot == 0
            unused = (task.criticality == "HI" and slot != 0) or (
                task.criticality == "LO" and slot not in {0, 1}
            )
            if unused:
                clauses.extend((z3.Not(job.present), z3.Not(job.ready)))
                continue
            if lo_aggregate:
                clauses.extend((
                    z3.Implies(job.present, job.release_index == -1),
                    z3.Implies(job.present, job.release_time <= state.t),
                    z3.Implies(job.present, job.tie_break == -1),
                    z3.Implies(job.present, job.actual_demand == job.effective_demand),
                ))
                continue
            clauses.extend((
                z3.Implies(job.present, job.release_time == job.release_index * task.period),
                z3.Implies(job.present, job.absolute_deadline == job.release_time + task.deadline),
                z3.Implies(job.present, job.actual_demand <= task.actual_demand_upper),
                z3.Implies(job.present, job.actual_demand >= task.actual_demand_min),
                z3.Implies(job.present, job.tie_break == job.release_index),
            ))
            if task.criticality == "HI":
                clauses.extend((
                    z3.Implies(job.present, job.budget_at_release >= task.budget_floor),
                    z3.Implies(job.present, job.budget_at_release <= task.budget_upper),
                    z3.Implies(job.present, job.effective_demand == job.actual_demand),
                ))
            else:
                degraded = int(task.degraded_cost or task.c_lo)
                normal_snapshot = z3.And(
                    job.budget_at_release >= task.budget_floor,
                    job.budget_at_release <= task.budget_upper,
                )
                degraded_snapshot = job.budget_at_release == degraded
                expected_effective = z3.If(
                    job.release_entry_mode_hi,
                    z3.If(job.actual_demand < degraded, job.actual_demand, degraded),
                    z3.If(
                        job.actual_demand < job.budget_at_release + 1,
                        job.actual_demand,
                        job.budget_at_release + 1,
                    ),
                )
                clauses.extend((
                    z3.Implies(
                        job.present,
                        z3.If(job.release_entry_mode_hi, degraded_snapshot, normal_snapshot),
                    ),
                    z3.Implies(job.present, job.effective_demand == expected_effective),
                ))
    return z3.And(*clauses)


def no_prior_hi_miss_consistency(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses: list[z3.BoolRef] = []
    for task in model.hi_tasks:
        for slot in range(model.max_jobs_per_task):
            job = state.jobs[(task.name, slot)]
            deadline_bound = z3.If(state.p <= 2, job.absolute_deadline >= state.t,
                                   job.absolute_deadline > state.t)
            clauses.append(z3.Implies(
                z3.And(state.hi_miss_ledger == 0, job.present),
                deadline_bound,
            ))
    return z3.And(*clauses)


def settled_job_consistency(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    """After P0 no completed job remains present until the next time advance."""

    clauses: list[z3.BoolRef] = []
    for job in state.jobs.values():
        clauses.append(z3.Implies(
            z3.And(state.p >= 1, job.present),
            job.executed_service < job.effective_demand,
        ))
    return z3.And(*clauses)


def frontier_consistency(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    clauses: list[z3.BoolRef] = []
    clauses.append(z3.Implies(
        state.p <= 6,
        z3.And(state.frontier.selected_slot == -1, z3.Not(state.frontier.running)),
    ))
    eligible_by_index: list[z3.BoolRef] = []
    for index, job in enumerate(state.jobs.values()):
        eligible = job.present & job.ready & (job.remaining > 0) & z3.Not(job.removed)
        eligible_by_index.append(z3.And(state.frontier.selected_slot == index, eligible))
    selected_eligible = z3.Or(*eligible_by_index) if eligible_by_index else z3.BoolVal(False)
    clauses.append(z3.Implies(
        state.p == 7,
        z3.And(
            state.frontier.running == (state.frontier.selected_slot >= 0),
            z3.Implies(state.frontier.selected_slot >= 0, selected_eligible),
        ),
    ))
    return z3.And(*clauses)


def history_bounds(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
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
    # Runtime windows store event counts, not booleans.  Non-negativity is the
    # sound finite abstraction; observation itself clips the resulting rates.
    for window in (
        state.chi.mode_change_window, state.chi.lo_cancel_window,
        state.chi.hi_overrun_window, state.chi.lo_overrun_window,
        state.chi.job_start_window,
    ):
        clauses.extend(value >= 0 for value in window)
    return z3.And(*clauses)


def carry_in_consistency(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    """Base SafePrefix carry-in shape; reachable bounds are proved separately."""

    clauses: list[z3.BoolRef] = []
    for task in model.tasks:
        if task.deadline > task.period:
            clauses.append(z3.BoolVal(False))
        for slot in range(model.max_jobs_per_task):
            job = state.jobs[(task.name, slot)]
            clauses.append(z3.Implies(job.present, job.release_time <= state.t))
    return z3.And(*clauses)


def named_psi_clauses(
    state: SymbolicKernelState, model: BoundModel
) -> dict[str, z3.BoolRef]:
    return {
        "state_well_formedness": state_well_formedness(state, model),
        "budget_bounds": budget_bounds(state, model),
        "exact_periodic_eta": exact_periodic_eta(state, model),
        "job_field_consistency": job_field_consistency(state, model),
        "no_prior_hi_miss_consistency": no_prior_hi_miss_consistency(state, model),
        "settled_job_consistency": settled_job_consistency(state, model),
        "frontier_consistency": frontier_consistency(state, model),
        "history_bounds": history_bounds(state, model),
        "carry_in_consistency": carry_in_consistency(state, model),
    }


def build_psi(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    return z3.And(*named_psi_clauses(state, model).values())


__all__ = ["build_psi", "budget_bounds", "carry_in_consistency", "exact_periodic_eta",
           "frontier_consistency", "history_bounds", "job_field_consistency",
           "named_psi_clauses", "no_prior_hi_miss_consistency",
           "settled_job_consistency", "state_well_formedness"]
