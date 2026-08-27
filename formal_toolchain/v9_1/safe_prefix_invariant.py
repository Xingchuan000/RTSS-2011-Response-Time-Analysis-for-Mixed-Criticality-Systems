"""Proof-object builders for the V9.1 safe-prefix invariant Psi."""

from __future__ import annotations

from dataclasses import dataclass

import z3

from .boot_state import encode_canonical_boot_state
from .environment_encoder import SymbolicEnvironment
from .invariant_templates import build_psi, named_psi_clauses
from .symbolic_state import BoundModel, SymbolicKernelState, declare_state
from .transition_encoder import encode_phase_step, encode_step


@dataclass(frozen=True, slots=True)
class SafePrefixInvariant:
    model: BoundModel
    template_name: str = "V9_1_SAFE_PREFIX_PSI_V1"

    def formula(self, state: SymbolicKernelState) -> z3.BoolRef:
        return build_psi(state, self.model)

    def initial_constraints(self, state: SymbolicKernelState) -> tuple[z3.BoolRef, ...]:
        return encode_canonical_boot_state(state, self.model)


    def named_clauses(self, state: SymbolicKernelState) -> dict[str, z3.BoolRef]:
        return named_psi_clauses(state, self.model)

    def initial_clause_counterexamples(
        self, *, prefix: str = "initial.diag"
    ) -> dict[str, z3.BoolRef]:
        state = declare_state(prefix, self.model)
        boot = self.initial_constraints(state)
        return {
            name: z3.And(*boot, z3.Not(clause))
            for name, clause in self.named_clauses(state).items()
        }

    def initial_counterexample(self, *, prefix: str = "initial") -> z3.BoolRef:
        state = declare_state(prefix, self.model)
        return z3.And(*self.initial_constraints(state), z3.Not(self.formula(state)))

    def conditional_inductiveness_counterexample(
        self, env: SymbolicEnvironment, *, prefix: str = "ind"
    ) -> z3.BoolRef:
        """Counterexample to safe-prefix-only inductiveness.

        The theorem is intentionally conditioned on NoPriorHIMiss(z) and on no
        HI miss being present after the step.  Post-miss states are outside the
        induction domain and must not create irrelevant proof failures.
        """

        state = declare_state(f"{prefix}.z", self.model)
        next_state = declare_state(f"{prefix}.zp", self.model)
        return z3.And(
            *env.constraints,
            env.phase.origin_time == state.t,
            self.formula(state),
            state.hi_miss_ledger == 0,
            encode_step(state, next_state, self.model, env),
            next_state.hi_miss_ledger == 0,
            z3.Not(self.formula(next_state)),
        )

    def phase_inductiveness_counterexample(
        self, env: SymbolicEnvironment, phase: int, *, prefix: str = "ind.phase"
    ) -> z3.BoolRef:
        """Phase-local form of the same safe-prefix induction obligation."""

        phase = int(phase)
        if not 0 <= phase <= 7:
            raise ValueError("V9_1_PHASE_OUT_OF_RANGE")
        state = declare_state(f"{prefix}.p{phase}.z", self.model)
        next_state = declare_state(f"{prefix}.p{phase}.zp", self.model)
        return z3.And(
            *env.constraints,
            env.phase.origin_time == state.t,
            self.formula(state),
            state.hi_miss_ledger == 0,
            encode_phase_step(state, next_state, self.model, env, phase=phase),
            next_state.hi_miss_ledger == 0,
            z3.Not(self.formula(next_state)),
        )



__all__ = ["SafePrefixInvariant"]
