"""Ranked FirstValid selector and symbolic budget update."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import z3

from .symbolic_state import BoundModel


def encode_first_valid_explicit_noop(
    ranking: Sequence[z3.ArithRef | int],
    mask: Sequence[z3.BoolRef],
    *,
    action_dim: int,
    noop_id: int,
    name: str = "selected_action",
) -> tuple[z3.ArithRef, tuple[z3.BoolRef, ...]]:
    if len(ranking) != action_dim or len(mask) != action_dim:
        raise ValueError("POLICY_SELECTION_SEMANTICS_FAILED")
    normalized = tuple(item if isinstance(item, z3.ExprRef) else z3.IntVal(int(item)) for item in ranking)
    if not (0 <= noop_id < action_dim):
        raise ValueError("EXPLICIT_NOOP_IDENTITY_UNRESOLVED")
    selected = z3.Int(name)
    clauses: list[z3.BoolRef] = [mask[noop_id]]

    def mask_at(action: z3.ArithRef) -> z3.BoolRef:
        if z3.is_int_value(action):
            value = action.as_long()
            if not (0 <= value < action_dim):
                raise ValueError("POLICY_SELECTION_SEMANTICS_FAILED")
            return mask[value]
        return z3.Or(*(z3.And(action == index, mask[index]) for index in range(action_dim)))

    for index, action in enumerate(normalized):
        prior_invalid = z3.And(*(z3.Not(mask_at(normalized[j])) for j in range(index)))
        clauses.append(z3.Implies(selected == action, z3.And(mask_at(action), prior_invalid)))
    clauses.append(z3.Or(*(selected == action for action in normalized)))
    # Every enabled prefix candidate must be the selected first candidate.
    for index, action in enumerate(normalized):
        prior_invalid = z3.And(*(z3.Not(mask_at(normalized[j])) for j in range(index)))
        clauses.append(z3.Implies(z3.And(mask_at(action), prior_invalid), selected == action))
    return selected, tuple(clauses)


def encode_budget_after_selected_action(
    selected: z3.ArithRef,
    candidates: Sequence[Mapping[str, z3.ArithRef]],
    budgets: Mapping[str, z3.ArithRef],
    *,
    action_dim: int,
) -> dict[str, z3.ArithRef]:
    if len(candidates) != action_dim:
        raise ValueError("BUDGET_UPDATE_ACTION_DIM_MISMATCH")
    result: dict[str, z3.ArithRef] = {}
    for name, before in budgets.items():
        value = before
        for index in reversed(range(action_dim)):
            value = z3.If(selected == index, candidates[index][name], value)
        result[name] = value
    return result


__all__ = ["encode_budget_after_selected_action", "encode_first_valid_explicit_noop"]
