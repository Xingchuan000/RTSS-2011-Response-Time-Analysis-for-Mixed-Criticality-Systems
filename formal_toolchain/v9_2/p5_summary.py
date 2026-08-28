"""Compositional soundness obligations for the V9.2 P5 invariant summary.

The exact deployed P5 relation remains the one used by FirstBadWindow.  Safe-prefix
inductiveness only needs the weaker fact that P5 preserves controller-budget bounds
and the declared conservative history domain.  This module proves that weakening
compositionally without rebuilding observation -> CART -> FirstValid inside the
induction theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import z3

from .action_encoder import encode_first_valid_explicit_noop
from .invariant_templates import history_bounds
from .mask_encoder import encode_action_mask
from .symbolic_state import BoundModel, declare_state
from .transition_encoder import _history_domain


@dataclass(frozen=True, slots=True)
class P5SummaryObligation:
    obligation_id: str
    counterexample: z3.BoolRef
    explanation: str


def _budget_precondition(state, model: BoundModel) -> z3.BoolRef:
    return z3.And(*(
        z3.And(
            state.budgets[task.name] >= task.budget_floor,
            state.budgets[task.name] <= task.budget_upper,
        )
        for task in model.tasks
    ))


def _candidate_out_of_bounds(candidate, model: BoundModel) -> z3.BoolRef:
    return z3.Or(*(
        z3.Or(
            candidate[task.name] < task.budget_floor,
            candidate[task.name] > task.budget_upper,
        )
        for task in model.tasks
    ))


def build_p5_summary_soundness_obligations(
    model: BoundModel, *, prefix: str = "p5.summary"
) -> tuple[P5SummaryObligation, ...]:
    """Return closed counterexample formulas whose UNSAT proves the P5 summary.

    Proof decomposition:

    1. FirstValid always returns a masked action when rankings are permutations.
       Bindings already machine-check every deployed tree leaf ranking as a full
       action permutation, so this theorem is independent of observation/tree
       arithmetic.
    2. Every masked static action candidate obeys the frozen controller budget
       interval.  This uses the exact deployed mask/candidate arithmetic, including
       binary64 rounding and the explicit noop.
    3. The controller-boundary history abstraction is contained in Psi's history
       clause.  Therefore the exact P5 post-history contract implies the summary.

    Together these establish that every exact P5 transition is contained in the
    cheaper invariant-summary transition used only for safe-prefix induction.
    """

    if model.action_dim <= 0 or model.noop_id is None:
        raise ValueError("V9_2_P5_POLICY_ARTIFACT_UNBOUND")

    obligations: list[P5SummaryObligation] = []

    # Selector theorem over arbitrary permutation rankings and arbitrary masks.
    # It deliberately does not depend on numeric observation or tree guards.
    ranking = tuple(z3.Int(f"{prefix}.ranking.{i}") for i in range(model.action_dim))
    mask = tuple(z3.Bool(f"{prefix}.mask.{i}") for i in range(model.action_dim))
    selected, selector_constraints = encode_first_valid_explicit_noop(
        ranking,
        mask,
        action_dim=model.action_dim,
        noop_id=model.noop_id,
        name=f"{prefix}.selected",
    )
    permutation = z3.And(
        *(z3.And(value >= 0, value < model.action_dim) for value in ranking),
        z3.Distinct(*ranking),
    )
    selected_is_masked = z3.Or(*(
        z3.And(selected == index, mask[index]) for index in range(model.action_dim)
    ))
    obligations.append(P5SummaryObligation(
        "P5_FIRST_VALID_SELECTED_ACTION_IS_MASKED",
        z3.And(permutation, *selector_constraints, z3.Not(selected_is_masked)),
        "ranked FirstValid with the explicit noop can only return a mask-valid action",
    ))

    # Candidate theorem uses the exact deployed mask and binary64 budget-update
    # arithmetic, but not the observation/tree path that caused the induction
    # timeout.  A single disjunction covers the entire frozen action alphabet.
    state = declare_state(f"{prefix}.budget", model)
    masks, candidates, mask_constraints = encode_action_mask(
        state.budgets, model.action_definitions, model
    )
    bad_masked_candidate = z3.Or(*(
        z3.And(masks[index], _candidate_out_of_bounds(candidates[index], model))
        for index in range(model.action_dim)
    ))
    obligations.append(P5SummaryObligation(
        "P5_MASK_VALID_CANDIDATES_RESPECT_BUDGET_BOUNDS",
        z3.And(
            _budget_precondition(state, model),
            *mask_constraints,
            bad_masked_candidate,
        ),
        "every action admitted by the exact deployed mask stays inside frozen controller budget bounds",
    ))

    # Exact P5 constrains enabled post-history with _history_domain.  Verify that
    # this implementation contract is at least as strong as Psi's public history
    # clause instead of relying on two hand-maintained definitions remaining equal.
    history_state = declare_state(f"{prefix}.history", model)
    obligations.append(P5SummaryObligation(
        "P5_HISTORY_DOMAIN_IMPLIES_PSI_HISTORY_BOUNDS",
        z3.And(
            *_history_domain(history_state, model),
            z3.Not(history_bounds(history_state, model)),
        ),
        "the controller-boundary history over-approximation is contained in Psi's history domain",
    ))

    return tuple(obligations)


__all__ = ["P5SummaryObligation", "build_p5_summary_soundness_obligations"]
