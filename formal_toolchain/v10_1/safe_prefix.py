"""V10.1 scheduler SafePrefix invariant and P5 summary.

Policy-history numeric bounds are deliberately *not* part of the scheduler
SafePrefix invariant.  They are proved/abstracted by the separate V10.1
feature-transfer layer.  This avoids importing the old Real-valued EMA bound
into a runtime whose history is actually updated in IEEE-754 binary64.
"""

from __future__ import annotations

from dataclasses import dataclass

import z3

from .kernel.action_encoder import encode_first_valid_leaf_cases
from .kernel.boot_state import encode_canonical_boot_state
from .kernel.environment_encoder import SymbolicEnvironment
from .kernel.invariant_templates import named_psi_clauses
from .kernel.mask_encoder import encode_action_mask
from .kernel.symbolic_state import BoundModel, SymbolicKernelState, declare_state
from .kernel.transition_encoder import (
    _copy_history, _frame_state, _phase, encode_phase_step,
)
from .kernel.tree_encoder import TreeLeafCase


@dataclass(frozen=True, slots=True)
class P5SummaryObligation:
    obligation_id: str
    counterexample: z3.BoolRef
    explanation: str


def named_scheduler_psi_clauses(
    state: SymbolicKernelState, model: BoundModel
) -> dict[str, z3.BoolRef]:
    rows = dict(named_psi_clauses(state, model))
    rows.pop("history_bounds", None)
    return rows


def build_scheduler_psi(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    return z3.And(*named_scheduler_psi_clauses(state, model).values())


def encode_p5_scheduler_summary(
    z: SymbolicKernelState, zp: SymbolicKernelState, model: BoundModel
) -> z3.BoolRef:
    """Superset P5 relation sufficient for scheduler-SafePrefix induction.

    Enabled P5 may choose any post budget inside the frozen legal bounds.
    Policy history is not given any EMA/sample upper bound here, but it must
    remain inside the structural runtime domain (all counters/statistics are
    non-negative).  Disabled P5 is exact identity.  This keeps the summary a
    sound superset of the deployed P5 while avoiding the old Real-valued EMA
    bound in the scheduler SafePrefix invariant.
    """

    enabled = (z.t % model.agent_period) == 0
    clauses = _phase(z, zp, 5, 6) + _frame_state(
        z, zp, model, mutable=frozenset({"budgets", "history"})
    )
    history_identity = z3.And(*_copy_history(z, zp))
    clauses.append(z3.Implies(enabled, _history_structural_domain(zp, model)))
    clauses.append(z3.Implies(z3.Not(enabled), history_identity))
    for task in model.tasks:
        clauses.append(z3.Implies(
            enabled,
            z3.And(
                zp.budgets[task.name] >= int(task.budget_floor),
                zp.budgets[task.name] <= int(task.budget_upper),
            ),
        ))
        clauses.append(z3.Implies(
            z3.Not(enabled), zp.budgets[task.name] == z.budgets[task.name]
        ))
    return z3.And(*clauses)


def _history_structural_domain(
    state: SymbolicKernelState, model: BoundModel
) -> z3.BoolRef:
    """Structural policy-history domain retained by scheduler well-formedness.

    V10.1 deliberately removes history *upper bounds* from scheduler SafePrefix,
    because those numeric bounds belong to the feature-transfer proof.  The
    Full-kernel state contract still requires runtime statistics/event counters
    to be non-negative.  The P5 summary must preserve that structural fact;
    otherwise it admits impossible negative-history post states and creates a
    spurious inductiveness counterexample.
    """

    clauses: list[z3.BoolRef] = []
    for task in model.tasks:
        clauses.extend((
            state.chi.recent_cost[task.name] >= 0,
            state.chi.ema_cost[task.name] >= 0,
            state.chi.overrun_ema[task.name] >= 0,
            state.chi.max_cost_k[task.name] >= 0,
        ))
    for window in (
        state.chi.mode_change_window,
        state.chi.lo_cancel_window,
        state.chi.hi_overrun_window,
        state.chi.lo_overrun_window,
        state.chi.job_start_window,
    ):
        clauses.extend(value >= 0 for value in window)
    return z3.And(*clauses)


def _budget_precondition(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    return z3.And(*(
        z3.And(
            state.budgets[task.name] >= int(task.budget_floor),
            state.budgets[task.name] <= int(task.budget_upper),
        )
        for task in model.tasks
    ))


def _candidate_out_of_bounds(candidate: dict[str, z3.ArithRef], model: BoundModel) -> z3.BoolRef:
    return z3.Or(*(
        z3.Or(
            candidate[task.name] < int(task.budget_floor),
            candidate[task.name] > int(task.budget_upper),
        )
        for task in model.tasks
    ))


def build_p5_scheduler_summary_soundness_obligations(
    model: BoundModel, *, prefix: str = "v10.p5.summary"
) -> tuple[P5SummaryObligation, ...]:
    """Machine obligations making the P5 scheduler summary a sound superset."""

    if model.tree is None or model.noop_id is None or model.action_dim <= 0:
        raise ValueError("V10_1_P5_POLICY_ARTIFACT_UNBOUND")
    rows: list[P5SummaryObligation] = []
    rankings = sorted({
        tuple(int(value) for value in leaf.action_ranking)
        for leaf in model.tree.leaves
    })
    if not rankings:
        raise ValueError("TREE_HAS_NO_LEAVES")
    for ranking_index, ranking in enumerate(rankings):
        if len(ranking) != model.action_dim or set(ranking) != set(range(model.action_dim)):
            raise ValueError("POLICY_SELECTION_SEMANTICS_FAILED")
        mask = tuple(
            z3.Bool(f"{prefix}.ranking_{ranking_index}.mask.{i}")
            for i in range(model.action_dim)
        )
        case = TreeLeafCase(ranking_index, ranking, z3.BoolVal(True))
        selected, constraints = encode_first_valid_leaf_cases(
            (case,), mask, action_dim=model.action_dim, noop_id=int(model.noop_id),
            name=f"{prefix}.ranking_{ranking_index}.selected",
        )
        selected_is_masked = z3.Or(*(
            z3.And(selected == index, mask[index]) for index in range(model.action_dim)
        ))
        rows.append(P5SummaryObligation(
            f"P5_LEAF_FIRST_VALID_SELECTED_ACTION_IS_MASKED_{ranking_index}",
            z3.And(*constraints, z3.Not(selected_is_masked)),
            "ranked FirstValid with explicit noop can only return a mask-valid action",
        ))

    state = declare_state(f"{prefix}.budget", model)
    masks, candidates, mask_constraints = encode_action_mask(
        state.budgets, model.action_definitions, model
    )
    bad = z3.Or(*(
        z3.And(masks[index], _candidate_out_of_bounds(dict(candidates[index]), model))
        for index in range(model.action_dim)
    ))
    rows.append(P5SummaryObligation(
        "P5_MASK_VALID_CANDIDATES_RESPECT_BUDGET_BOUNDS",
        z3.And(_budget_precondition(state, model), *mask_constraints, bad),
        "every action admitted by the exact deployed mask stays inside controller budget bounds",
    ))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class SchedulerSafePrefixInvariant:
    model: BoundModel
    template_name: str = "V10_1_SCHEDULER_SAFE_PREFIX_PSI_V1"

    def formula(self, state: SymbolicKernelState) -> z3.BoolRef:
        return build_scheduler_psi(state, self.model)

    def initial_constraints(self, state: SymbolicKernelState) -> tuple[z3.BoolRef, ...]:
        return encode_canonical_boot_state(state, self.model)

    def initial_counterexample(self, *, prefix: str = "initial") -> z3.BoolRef:
        state = declare_state(prefix, self.model)
        return z3.And(*self.initial_constraints(state), z3.Not(self.formula(state)))

    def phase_inductiveness_counterexample(
        self, env: SymbolicEnvironment, phase: int, *, prefix: str = "ind.phase",
        use_p5_summary: bool = False,
    ) -> z3.BoolRef:
        phase = int(phase)
        if not 0 <= phase <= 7:
            raise ValueError("V10_1_PHASE_OUT_OF_RANGE")
        state = declare_state(f"{prefix}.p{phase}.z", self.model)
        next_state = declare_state(f"{prefix}.p{phase}.zp", self.model)
        transition = (
            encode_p5_scheduler_summary(state, next_state, self.model)
            if phase == 5 and use_p5_summary
            else encode_phase_step(state, next_state, self.model, env, phase=phase)
        )
        return z3.And(
            *env.constraints,
            env.phase.origin_time == state.t,
            self.formula(state),
            state.hi_miss_ledger == 0,
            transition,
            next_state.hi_miss_ledger == 0,
            z3.Not(self.formula(next_state)),
        )


__all__ = [
    "P5SummaryObligation", "SchedulerSafePrefixInvariant",
    "build_p5_scheduler_summary_soundness_obligations",
    "build_scheduler_psi", "encode_p5_scheduler_summary",
    "named_scheduler_psi_clauses",
]
