"""Leaf-specialized FirstValid selector and sparse symbolic budget update."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import z3

from .symbolic_state import BoundModel


def encode_first_valid_leaf_cases(
    leaf_cases: Sequence[Any],
    mask: Sequence[z3.BoolRef],
    *,
    action_dim: int,
    noop_id: int,
    name: str = "selected_action",
) -> tuple[z3.ArithRef, tuple[z3.BoolRef, ...]]:
    """Encode ranked FirstValid directly under each concrete CART leaf.

    Every leaf ranking is a concrete permutation.  Therefore mask lookup is a
    direct array index and no symbolic ``ranking[j] == action`` multiplexor is
    needed.  Explicit noop makes every leaf total, so positions after noop are
    unreachable and are not encoded.
    """

    if len(mask) != action_dim or not (0 <= noop_id < action_dim):
        raise ValueError("POLICY_SELECTION_SEMANTICS_FAILED")
    selected = z3.Int(name)
    clauses: list[z3.BoolRef] = [
        mask[noop_id],
        selected >= 0,
        selected < action_dim,
    ]

    for case in leaf_cases:
        ranking = tuple(int(value) for value in case.ranking)
        if len(ranking) != action_dim or set(ranking) != set(range(action_dim)):
            raise ValueError("POLICY_SELECTION_SEMANTICS_FAILED")
        prior_invalid: list[z3.BoolRef] = []
        reached_noop = False
        for action in ranking:
            choose = z3.And(
                case.guard,
                mask[action],
                *(z3.Not(value) for value in prior_invalid),
            )
            clauses.append(z3.Implies(choose, selected == action))
            prior_invalid.append(mask[action])
            if action == noop_id:
                reached_noop = True
                break
        if not reached_noop:
            raise ValueError("EXPLICIT_NOOP_IDENTITY_UNRESOLVED")

    return selected, tuple(clauses)


def encode_budget_after_selected_action(
    selected: z3.ArithRef,
    candidates: Sequence[Mapping[str, z3.ArithRef]],
    budgets: Mapping[str, z3.ArithRef],
    *,
    action_dim: int,
    prefix: str = "budget_after",
) -> tuple[dict[str, z3.ArithRef], tuple[z3.BoolRef, ...]]:
    """Use fresh post-budget scalars plus action implications, not ITE ladders."""

    if len(candidates) != action_dim:
        raise ValueError("BUDGET_UPDATE_ACTION_DIM_MISMATCH")
    result: dict[str, z3.ArithRef] = {}
    clauses: list[z3.BoolRef] = []
    for name in budgets:
        after = z3.Int(f"{prefix}.{name}")
        result[name] = after
        for action in range(action_dim):
            clauses.append(z3.Implies(
                selected == action,
                after == candidates[action][name],
            ))
    return result, tuple(clauses)


__all__ = ["encode_budget_after_selected_action", "encode_first_valid_leaf_cases"]
