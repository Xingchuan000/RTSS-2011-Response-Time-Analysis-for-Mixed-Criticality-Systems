"""Symbolic action-mask and candidate-budget semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import z3

from .symbolic_state import BoundModel


def _field(action: Any, name: str, default: Any = None) -> Any:
    if isinstance(action, Mapping):
        return action.get(name, default)
    return getattr(action, name, default)


def _ceil_ratio(value: z3.ArithRef, ratio: Any) -> z3.ArithRef:
    # For positive integer budgets, ceil(value * ratio) = -floor(-x).
    rational = z3.RealVal(str(ratio))
    return -z3.ToInt(-(value * rational))


def encode_budget_update(
    budgets: Mapping[str, z3.ArithRef], action: Any, model: BoundModel, *, prefix: str = "candidate"
) -> tuple[dict[str, z3.ArithRef], list[z3.BoolRef]]:
    updates = dict(budgets)
    constraints: list[z3.BoolRef] = []
    if bool(_field(action, "is_noop", False)):
        return updates, constraints
    inc = _field(action, "increase_task", _field(action, "target_task"))
    direction = _field(action, "direction")
    if inc is not None and direction is None:
        direction = "increase"
    dec_tasks = tuple(_field(action, "decrease_tasks", ()) or ())
    if direction not in {None, "increase", "decrease"} and not dec_tasks:
        raise ValueError("ACTION_SEMANTICS_UNBOUND")
    if _field(action, "is_constraint_guided_pair", False) or _field(action, "is_residual_ranked", False):
        raise ValueError("ACTION_SEMANTICS_UNBOUND")
    if inc is not None and direction == "increase":
        ratio = _field(action, "increase_ratio")
        delta = _ceil_ratio(budgets[str(inc)], ratio) if ratio is not None else z3.IntVal(1)
        updates[str(inc)] = budgets[str(inc)] + z3.If(delta > 0, delta, 1)
    elif inc is not None and direction == "decrease":
        ratio = _field(action, "decrease_ratio")
        delta = z3.ToInt(budgets[str(inc)] * z3.RealVal(str(ratio))) if ratio is not None else z3.IntVal(1)
        updates[str(inc)] = budgets[str(inc)] - z3.If(delta > 0, delta, 1)
    elif dec_tasks:
        ratio = _field(action, "decrease_ratio")
        for name in dec_tasks:
            delta = z3.ToInt(budgets[str(name)] * z3.RealVal(str(ratio))) if ratio is not None else z3.IntVal(1)
            updates[str(name)] = budgets[str(name)] - z3.If(delta > 0, delta, 1)
    else:
        raise ValueError("ACTION_SEMANTICS_UNBOUND")
    return updates, constraints


def encode_action_mask(
    budgets: Mapping[str, z3.ArithRef], actions: Sequence[Any], model: BoundModel
) -> tuple[tuple[z3.BoolRef, ...], tuple[dict[str, z3.ArithRef], ...], tuple[z3.BoolRef, ...]]:
    if model.noop_id is None or sum(bool(_field(action, "is_noop", False)) for action in actions) != 1:
        raise ValueError("EXPLICIT_NOOP_IDENTITY_UNRESOLVED")
    masks: list[z3.BoolRef] = []
    candidates: list[dict[str, z3.ArithRef]] = []
    constraints: list[z3.BoolRef] = []
    task_by_name = model.task_by_name
    for index, action in enumerate(actions):
        if bool(_field(action, "is_noop", False)):
            masks.append(z3.BoolVal(True))
            candidates.append(dict(budgets))
            continue
        candidate, local = encode_budget_update(budgets, action, model, prefix=f"candidate.{index}")
        candidates.append(candidate)
        constraints.extend(local)
        valid: list[z3.BoolRef] = []
        for name, task in task_by_name.items():
            valid.append(z3.And(candidate[name] >= task.budget_floor, candidate[name] <= task.budget_upper))
        if _field(action, "decrease_tasks", ()):
            for name in _field(action, "decrease_tasks", ()):
                if task_by_name[str(name)].criticality == "HI":
                    valid.append(z3.BoolVal(False))
        masks.append(z3.And(*valid))
    if not z3.is_true(masks[model.noop_id]):
        raise ValueError("EXPLICIT_NOOP_IDENTITY_UNRESOLVED")
    return tuple(masks), tuple(candidates), tuple(constraints)


__all__ = ["encode_action_mask", "encode_budget_update"]
