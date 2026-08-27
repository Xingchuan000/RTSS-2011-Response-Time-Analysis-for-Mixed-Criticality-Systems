"""Proof-object builders for the V9.1 safe-prefix invariant Psi."""

from __future__ import annotations

from dataclasses import dataclass

import z3

from .environment_encoder import SymbolicEnvironment
from .invariant_templates import build_psi
from .symbolic_state import BoundModel, SymbolicKernelState, declare_state
from .transition_encoder import encode_step


@dataclass(frozen=True, slots=True)
class SafePrefixInvariant:
    model: BoundModel
    template_name: str = "V9_1_SAFE_PREFIX_PSI_V1"

    def formula(self, state: SymbolicKernelState) -> z3.BoolRef:
        return build_psi(state, self.model)

    def initial_constraints(self, state: SymbolicKernelState) -> tuple[z3.BoolRef, ...]:
        clauses: list[z3.BoolRef] = [state.t == 0, state.p == 0, z3.Not(state.mode_hi),
                                     state.hi_miss_ledger == 0,
                                     state.frontier.selected_slot == -1,
                                     z3.Not(state.frontier.running)]
        for task in self.model.tasks:
            clauses.extend((state.budgets[task.name] == task.initial_budget,
                            state.eta[task.name] == task.period))
        for job in state.jobs.values():
            clauses.extend((z3.Not(job.present), z3.Not(job.removed),
                            job.executed_service == 0))
        for mapping in (state.chi.recent_cost, state.chi.ema_cost, state.chi.max_cost_k):
            clauses.extend(value == 0 for value in mapping.values())
        for window in (state.chi.mode_change_window, state.chi.lo_cancel_window,
                       state.chi.hi_overrun_window, state.chi.lo_overrun_window,
                       state.chi.job_start_window):
            clauses.extend(value == 0 for value in window)
        return tuple(clauses)

    def initial_counterexample(self, *, prefix: str = "initial") -> z3.BoolRef:
        state = declare_state(prefix, self.model)
        return z3.And(*self.initial_constraints(state), z3.Not(self.formula(state)))

    def conditional_inductiveness_counterexample(
        self, env: SymbolicEnvironment, *, prefix: str = "ind"
    ) -> z3.BoolRef:
        state = declare_state(f"{prefix}.z", self.model)
        next_state = declare_state(f"{prefix}.zp", self.model)
        no_new_miss = next_state.hi_miss_ledger == state.hi_miss_ledger
        return z3.And(self.formula(state), encode_step(state, next_state, self.model, env),
                      no_new_miss, z3.Not(self.formula(next_state)))


__all__ = ["SafePrefixInvariant"]
