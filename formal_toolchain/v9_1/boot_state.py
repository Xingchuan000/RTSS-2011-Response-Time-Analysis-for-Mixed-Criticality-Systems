"""Canonical deployed boot-state contract for the V9.1 kernel.

Every proof obligation that starts from boot must use this single encoder.  The
contract deliberately canonicalizes ghost fields of absent jobs so arbitrary
SMT values cannot create fake initial-state counterexamples or obstruct later
boot-state pinning/replay.
"""

from __future__ import annotations

import z3

from .symbolic_state import BoundModel, SymbolicKernelState


def encode_canonical_boot_state(
    state: SymbolicKernelState,
    model: BoundModel,
) -> tuple[z3.BoolRef, ...]:
    clauses: list[z3.BoolRef] = [
        state.t == 0,
        state.p == 0,
        z3.Not(state.mode_hi),
        state.hi_miss_ledger == 0,
        state.frontier.selected_slot == -1,
        z3.Not(state.frontier.running),
    ]
    for task in model.tasks:
        clauses.extend((
            state.budgets[task.name] == task.initial_budget,
            state.eta[task.name] == task.period,
        ))
        for slot in range(model.max_jobs_per_task):
            job = state.jobs[(task.name, slot)]
            clauses.extend((
                z3.Not(job.present),
                job.release_index == -1,
                job.release_time == 0,
                job.absolute_deadline == 0,
                job.tie_break == -1,
                z3.Not(job.release_entry_mode_hi),
                z3.Not(job.classification_abnormal),
                job.budget_at_release == 0,
                job.actual_demand == 0,
                job.effective_demand == 0,
                job.executed_service == 0,
                z3.Not(job.removed),
                z3.Not(job.ready),
            ))

    clauses.extend(value == 0 for value in state.chi.recent_cost.values())
    for task in model.tasks:
        # RuntimeFeatureState initializes EMA and max-cost history from C_LO.
        clauses.extend((
            state.chi.ema_cost[task.name] == task.c_lo,
            state.chi.overrun_ema[task.name] == 0,
            state.chi.max_cost_k[task.name] == task.c_lo,
        ))
    for window in (
        state.chi.mode_change_window,
        state.chi.lo_cancel_window,
        state.chi.hi_overrun_window,
        state.chi.lo_overrun_window,
        state.chi.job_start_window,
    ):
        clauses.extend(value == 0 for value in window)
    return tuple(clauses)


__all__ = ["encode_canonical_boot_state"]
