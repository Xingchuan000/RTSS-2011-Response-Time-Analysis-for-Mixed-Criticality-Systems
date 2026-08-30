"""Exact static action-mask, candidate-budget, and safety-margin semantics for P5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from math import ceil, floor
from typing import Any

import z3

from .symbolic_state import BoundModel


def _field(action: Any, name: str, default: Any = None) -> Any:
    if isinstance(action, Mapping):
        return action.get(name, default)
    return getattr(action, name, default)


def _binary64(value: float) -> z3.FPRef:
    return z3.FPVal(float(value), z3.Float64())


def _budget_as_binary64(value: z3.ArithRef) -> z3.FPRef:
    # Runtime budgets are small Python ints and therefore exactly representable
    # as binary64.  Keep the conversion explicit so multiplication follows the
    # executable CPython float path rather than mathematical-real arithmetic.
    return z3.fpToFP(z3.RNE(), z3.ToReal(value), z3.Float64())





@lru_cache(maxsize=None)
def _verified_rounding_fraction(
    direction: str, ratio: float, lower: int, upper: int
) -> tuple[int, int]:
    """Replace bounded CPython binary64 ratio updates by exact integer arithmetic.

    Runtime budgets are finite integers.  For the concrete action ratio, verify
    every admissible budget value against CPython's binary64 multiply followed
    by ceil/floor.  Only after the finite equality check succeeds do we return
    the short decimal rational used in SMT.  Thus the SMT relation contains no
    floating-point theory while remaining exact on the complete bound domain.
    """

    if lower < 1 or upper < lower:
        raise ValueError("INVALID_BUDGET_DOMAIN")
    if direction == "increase":
        factor = 1.0 + float(ratio)
        op = ceil
    elif direction == "decrease":
        factor = 1.0 - float(ratio)
        op = floor
    else:
        raise ValueError("UNSUPPORTED_BUDGET_DIRECTION")
    rational = Fraction(str(factor))
    p, q = int(rational.numerator), int(rational.denominator)
    for budget in range(int(lower), int(upper) + 1):
        runtime = int(op(float(budget) * factor))
        if direction == "increase":
            exact = (p * budget + q - 1) // q
        else:
            exact = (p * budget) // q
        if runtime != exact:
            raise ValueError("BINARY64_BUDGET_ROUNDING_NOT_RATIONALIZABLE_ON_BOUND_DOMAIN")
    return p, q


def _exact_rounded_ratio(
    value: z3.ArithRef, task: Any, ratio: float, *, direction: str
) -> z3.ArithRef:
    p, q = _verified_rounding_fraction(
        direction, float(ratio), int(task.budget_floor), int(task.budget_upper)
    )
    numerator = z3.IntVal(p) * value
    if direction == "increase":
        return (numerator + (q - 1)) / q
    return numerator / q


@lru_cache(maxsize=None)
def _verified_deploy_cap_cutoff(
    initial_budget: int, lower: int, upper: int, ratio: float
) -> int | None:
    """Compile the bounded binary64 deploy-cap predicate to one integer cutoff."""

    truth = [
        (budget, (float(budget) / float(initial_budget)) < float(ratio))
        for budget in range(int(lower), int(upper) + 1)
    ]
    seen_false = False
    max_valid: int | None = None
    for budget, valid in truth:
        if valid:
            if seen_false:
                raise ValueError("DEPLOY_CAP_BINARY64_PREDICATE_NOT_MONOTONE")
            max_valid = budget
        else:
            seen_false = True
    return max_valid


def _deploy_cap_guard(budget: z3.ArithRef, task: Any, ratio: float) -> z3.BoolRef:
    cutoff = _verified_deploy_cap_cutoff(
        int(task.initial_budget), int(task.budget_floor), int(task.budget_upper), float(ratio)
    )
    if cutoff is None:
        return z3.BoolVal(False)
    if cutoff >= int(task.budget_upper):
        return z3.BoolVal(True)
    return budget <= int(cutoff)


def _primitive_candidate(
    budgets: Mapping[str, z3.ArithRef], action: Any, model: BoundModel
) -> dict[str, z3.ArithRef]:
    if model.rounding_mode != "ceil_floor":
        raise ValueError("V9_2_P5_ONLY_CEIL_FLOOR_ROUNDING_IS_BOUND")
    candidate = dict(budgets)
    if bool(_field(action, "is_noop", False)):
        return candidate
    if bool(_field(action, "is_constraint_guided_pair", False)) or bool(
        _field(action, "is_residual_ranked", False)
    ):
        raise ValueError("ACTION_SEMANTICS_UNBOUND")

    task_by_name = model.task_by_name
    inc_name = _field(action, "increase_task", _field(action, "target_task"))
    direction = _field(action, "direction")
    if direction == "decrease" and _field(action, "increase_task") is None:
        inc_name = None
    decrease_tasks = tuple(str(name) for name in (_field(action, "decrease_tasks", ()) or ()))
    if direction == "decrease" and not decrease_tasks:
        target = _field(action, "target_task")
        if target is not None:
            decrease_tasks = (str(target),)

    if inc_name is not None:
        inc_name = str(inc_name)
        if inc_name not in task_by_name:
            raise ValueError("ACTION_TARGET_TASK_UNBOUND")
        ratio = float(_field(action, "increase_ratio", 0.10))
        old = budgets[inc_name]
        rounded = _exact_rounded_ratio(
            old, task_by_name[inc_name], ratio, direction="increase"
        )
        increased = z3.If(rounded >= old + model.min_budget_delta, rounded, old + model.min_budget_delta)
        upper = task_by_name[inc_name].budget_upper
        candidate[inc_name] = z3.If(increased > upper, upper, z3.If(increased < 1, 1, increased))

    for name in decrease_tasks:
        if name not in task_by_name:
            raise ValueError("ACTION_TARGET_TASK_UNBOUND")
        ratio = float(_field(action, "decrease_ratio", 0.05))
        old = budgets[name]
        rounded = _exact_rounded_ratio(
            old, task_by_name[name], ratio, direction="decrease"
        )
        decreased = z3.If(rounded <= old - model.min_budget_delta, rounded, old - model.min_budget_delta)
        candidate[name] = z3.If(decreased < 1, 1, decreased)

    if inc_name is None and not decrease_tasks:
        raise ValueError("ACTION_SEMANTICS_UNBOUND")
    return candidate


def encode_budget_update(
    budgets: Mapping[str, z3.ArithRef], action: Any, model: BoundModel, *, prefix: str = "candidate"
) -> tuple[dict[str, z3.ArithRef], list[z3.BoolRef]]:
    del prefix
    return _primitive_candidate(budgets, action, model), []


def _safety_rows_hold(candidate: Mapping[str, z3.ArithRef], model: BoundModel) -> z3.BoolRef:
    if not model.check_safety:
        return z3.BoolVal(True)
    names = tuple(task.name for task in model.tasks)
    clauses: list[z3.BoolRef] = []
    for row in model.safety_constraints:
        coeff = tuple(int(value) for value in row["coefficients"])
        if len(coeff) != len(names):
            raise ValueError("V9_2_P5_SAFETY_ROW_DIMENSION_MISMATCH")
        lhs = z3.Sum(*(z3.IntVal(coeff[index]) * candidate[name] for index, name in enumerate(names)))
        clauses.append(lhs <= int(row["rhs"]))
    return z3.And(*clauses)


def encode_safety_margin_min(
    budgets: Mapping[str, z3.ArithRef], model: BoundModel
) -> z3.ArithRef:
    """Encode ``AmcBudgetEnv._compute_current_safety_margin_min``.

    Checker rows and budget vectors are integral and remain below the exact
    binary64 integer range.  Dot/slack values are therefore exact; division,
    min, and clip are encoded in FP64 before converting the resulting float to
    a Real consumed by the observation encoder.
    """

    if not model.check_safety:
        return z3.RealVal(1)
    names = tuple(task.name for task in model.tasks)
    margins: list[z3.FPRef] = []
    for row in model.safety_constraints:
        coeff = tuple(int(value) for value in row["coefficients"])
        if len(coeff) != len(names):
            raise ValueError("V9_2_P5_SAFETY_ROW_DIMENSION_MISMATCH")
        rhs = int(row["rhs"])
        lhs = z3.Sum(*(z3.IntVal(coeff[index]) * budgets[name] for index, name in enumerate(names)))
        slack = z3.fpSub(
            z3.RNE(),
            _binary64(float(rhs)),
            _budget_as_binary64(lhs),
        )
        margins.append(
            z3.fpDiv(z3.RNE(), slack, _binary64(float(max(1, abs(rhs)))))
        )
    if not margins:
        raise ValueError("V9_2_P5_SAFETY_CHECKER_HAS_NO_ROWS")
    minimum = margins[0]
    for value in margins[1:]:
        minimum = z3.If(z3.fpLEQ(minimum, value), minimum, value)
    zero = _binary64(0.0)
    one = _binary64(1.0)
    clipped = z3.If(z3.fpLT(minimum, zero), zero, z3.If(z3.fpGT(minimum, one), one, minimum))
    return z3.fpToReal(clipped)


def encode_action_mask(
    budgets: Mapping[str, z3.ArithRef], actions: Sequence[Any], model: BoundModel
) -> tuple[tuple[z3.BoolRef, ...], tuple[dict[str, z3.ArithRef], ...], tuple[z3.BoolRef, ...]]:
    if len(actions) != model.action_dim:
        raise ValueError("POLICY_ACTION_DIMENSION_MISMATCH")
    if model.noop_id is None or sum(bool(_field(action, "is_noop", False)) for action in actions) != 1:
        raise ValueError("EXPLICIT_NOOP_IDENTITY_UNRESOLVED")

    masks: list[z3.BoolRef] = []
    candidates: list[dict[str, z3.ArithRef]] = []
    task_by_name = model.task_by_name
    for index, action in enumerate(actions):
        action_id = int(_field(action, "action_id", index))
        if action_id != index:
            raise ValueError("V9_2_P5_ACTION_IDS_MUST_MATCH_ALPHABET_INDEX")
        if bool(_field(action, "is_noop", False)):
            if index != model.noop_id:
                raise ValueError("EXPLICIT_NOOP_IDENTITY_UNRESOLVED")
            masks.append(z3.BoolVal(True))
            candidates.append(dict(budgets))
            continue

        candidate = _primitive_candidate(budgets, action, model)
        candidates.append(candidate)
        valid: list[z3.BoolRef] = []

        # Runtime deploy-cap guard is evaluated before the primitive update.
        inc_name = _field(action, "increase_task", _field(action, "target_task"))
        direction = _field(action, "direction")
        if direction == "decrease" and _field(action, "increase_task") is None:
            inc_name = None
        if model.enable_deploy_cap_mask and inc_name is not None:
            inc_name = str(inc_name)
            task = task_by_name[inc_name]
            cap_applies = model.deploy_cap_mask_criticality == "all" or task.criticality == "LO"
            if cap_applies:
                valid.append(_deploy_cap_guard(
                    budgets[inc_name], task, model.deploy_cap_mask_ratio
                ))

        decrease_tasks = tuple(str(name) for name in (_field(action, "decrease_tasks", ()) or ()))
        if direction == "decrease" and not decrease_tasks:
            target = _field(action, "target_task")
            if target is not None:
                decrease_tasks = (str(target),)
        if model.forbid_decreasing_hi_budgets and any(
            task_by_name[name].criticality == "HI" for name in decrease_tasks
        ):
            valid.append(z3.BoolVal(False))

        # evaluate_budget_candidate rejects a static action whose clipped update
        # leaves the whole budget vector unchanged.
        valid.append(z3.Or(*(candidate[name] != budgets[name] for name in budgets)))
        for name, task in task_by_name.items():
            valid.extend((candidate[name] >= task.budget_floor, candidate[name] <= task.budget_upper))
        valid.append(_safety_rows_hold(candidate, model))
        masks.append(z3.And(*valid))

    if not z3.is_true(masks[model.noop_id]):
        raise ValueError("EXPLICIT_NOOP_IDENTITY_UNRESOLVED")
    return tuple(masks), tuple(candidates), ()


__all__ = ["encode_action_mask", "encode_budget_update", "encode_safety_margin_min"]
