"""Frozen single/24 budget-action semantics for the C-AMC-sem/P0 proof.

This module is a proof model, not the mutable training/runtime environment.  It
captures the action ordering, candidate update, mask, fallback, and step
semantics certified by the existing PPP proof route.  q-AMC and future
experimental action branches may evolve independently under ``amc_py``.
"""

from __future__ import annotations

import math


FORMAL_ACTION_CONTRACT_VERSION = "c_amc_sem_p0_single24_action_v1"


def build_budget_action_space(
    ordered_task_names: list[str],
    *,
    action_space: str = "single",
    budget_increase_ratio: float = 0.10,
    budget_decrease_ratio: float = 0.05,
    include_explicit_noop: bool = False,
) -> tuple[dict[str, object], ...]:
    """Build the deterministic single-action table used by the proof model."""

    if action_space != "single":
        raise ValueError("UNSUPPORTED_ACTION_SCOPE")
    if include_explicit_noop:
        raise ValueError("EXPLICIT_NOOP_NOT_CERTIFIED")
    if budget_increase_ratio <= 0.0:
        raise ValueError("INVALID_INCREASE_RATIO")
    if budget_decrease_ratio <= 0.0 or budget_decrease_ratio >= 1.0:
        raise ValueError("INVALID_DECREASE_RATIO")

    actions: list[dict[str, object]] = []
    action_id = 0
    for task_index, task_name in enumerate(ordered_task_names):
        actions.append({
            "action_id": action_id,
            "increase_task": task_name,
            "decrease_tasks": (),
            "increase_idx": task_index,
            "decrease_indices": (),
            "increase_ratio": budget_increase_ratio,
            "decrease_ratio": budget_decrease_ratio,
            "is_noop": False,
        })
        action_id += 1
    for task_index, task_name in enumerate(ordered_task_names):
        actions.append({
            "action_id": action_id,
            "increase_task": None,
            "decrease_tasks": (task_name,),
            "increase_idx": None,
            "decrease_indices": (task_index,),
            "increase_ratio": budget_increase_ratio,
            "decrease_ratio": budget_decrease_ratio,
            "is_noop": False,
        })
        action_id += 1
    return tuple(actions)


def apply_budget_action_candidate(
    *,
    action: dict[str, object],
    budgets: dict[str, int],
    task_names: list[str],
    task_criticalities: list[str],
    task_deadlines: list[int],
    task_c_hi: list[int],
    rounding_mode: str = "ceil_floor",
    min_budget_delta: int = 1,
) -> dict[str, int]:
    """Replay one certified static action without mutating the input budgets."""

    if bool(action["is_noop"]):
        return {}
    if rounding_mode not in {"ceil_floor", "nearest"}:
        raise ValueError("UNSUPPORTED_BUDGET_ROUNDING_MODE")
    if min_budget_delta <= 0:
        raise ValueError("INVALID_MIN_BUDGET_DELTA")

    updates: dict[str, int] = {}
    increase_idx = action["increase_idx"]
    if increase_idx is not None:
        inc_idx = int(increase_idx)
        inc_name = task_names[inc_idx]
        old_inc = budgets[inc_name]
        raw_inc = old_inc * (1.0 + float(action["increase_ratio"]))
        inc_value = math.ceil(raw_inc) if rounding_mode == "ceil_floor" else int(round(raw_inc))
        inc_value = max(inc_value, old_inc + int(min_budget_delta))
        upper_bound = task_c_hi[inc_idx] if task_criticalities[inc_idx] == "HI" else task_deadlines[inc_idx]
        updates[inc_name] = max(1, min(inc_value, upper_bound))

    decrease_indices = action["decrease_indices"]
    for decrease_idx in decrease_indices:
        dec_idx = int(decrease_idx)
        dec_name = task_names[dec_idx]
        old_dec = budgets[dec_name]
        raw_dec = old_dec * (1.0 - float(action["decrease_ratio"]))
        dec_value = math.floor(raw_dec) if rounding_mode == "ceil_floor" else int(round(raw_dec))
        dec_value = min(dec_value, old_dec - int(min_budget_delta))
        updates[dec_name] = max(1, dec_value)
    return updates


def deploy_cap_increase_reject_reason(
    *,
    action: dict[str, object],
    budgets: dict[str, int],
    task_names: list[str],
    task_criticalities: list[str],
    initial_budgets: dict[str, int],
    enabled: bool,
    cap_ratio: float,
    cap_criticality: str,
) -> str | None:
    """Certified deployment cap guard."""

    if not enabled or action["increase_idx"] is None:
        return None
    inc_idx = int(action["increase_idx"])
    if task_criticalities[inc_idx] != cap_criticality:
        return None
    task_name = task_names[inc_idx]
    cap_value = math.ceil(initial_budgets[task_name] * cap_ratio)
    if budgets[task_name] >= cap_value:
        return "deploy_cap_reached"
    return None


def budget_floor_violation(
    *,
    updates: dict[str, int],
    initial_budgets: dict[str, int],
    budget_floor_ratio: float,
) -> str | None:
    """Certified initial-budget floor guard, without q-AMC profile overrides."""

    if budget_floor_ratio <= 0.0:
        return None
    for task_name, candidate_budget in updates.items():
        floor_value = max(1, math.ceil(initial_budgets[task_name] * budget_floor_ratio))
        if candidate_budget < floor_value:
            return "budget_floor_violation"
    return None


def evaluate_budget_candidate(
    *,
    action: dict[str, object] | None,
    budgets: dict[str, int],
    task_names: list[str],
    task_criticalities: list[str],
    task_deadlines: list[int],
    task_c_hi: list[int],
    initial_budgets: dict[str, int],
    rounding_mode: str,
    min_budget_delta: int,
    forbid_decreasing_hi_budgets: bool,
    budget_floor_ratio: float,
    deploy_cap_enabled: bool,
    deploy_cap_ratio: float,
    deploy_cap_criticality: str,
) -> dict[str, object]:
    """Single shared evaluator consumed by both mask and step in the proof model."""

    before = dict(budgets)
    if action is None or bool(action["is_noop"]):
        return {
            "accepted": True,
            "reject_reason": None,
            "candidate_budgets": before,
            "updates": {},
        }

    cap_reason = deploy_cap_increase_reject_reason(
        action=action,
        budgets=before,
        task_names=task_names,
        task_criticalities=task_criticalities,
        initial_budgets=initial_budgets,
        enabled=deploy_cap_enabled,
        cap_ratio=deploy_cap_ratio,
        cap_criticality=deploy_cap_criticality,
    )
    if cap_reason is not None:
        return {
            "accepted": False,
            "reject_reason": cap_reason,
            "candidate_budgets": before,
            "updates": {},
        }

    if forbid_decreasing_hi_budgets:
        decrease_indices = action["decrease_indices"]
        for decrease_idx in decrease_indices:
            if task_criticalities[int(decrease_idx)] == "HI":
                return {
                    "accepted": False,
                    "reject_reason": "decrease_hi_forbidden",
                    "candidate_budgets": before,
                    "updates": {},
                }

    updates = apply_budget_action_candidate(
        action=action,
        budgets=before,
        task_names=task_names,
        task_criticalities=task_criticalities,
        task_deadlines=task_deadlines,
        task_c_hi=task_c_hi,
        rounding_mode=rounding_mode,
        min_budget_delta=min_budget_delta,
    )
    floor_reason = budget_floor_violation(
        updates=updates,
        initial_budgets=initial_budgets,
        budget_floor_ratio=budget_floor_ratio,
    )
    if floor_reason is not None:
        return {
            "accepted": False,
            "reject_reason": floor_reason,
            "candidate_budgets": before,
            "updates": {},
        }

    candidate = dict(before)
    candidate.update(updates)
    return {
        "accepted": True,
        "reject_reason": None,
        "candidate_budgets": candidate,
        "updates": updates,
    }


def formal_valid_action_mask(
    *,
    actions: tuple[dict[str, object], ...],
    budgets: dict[str, int],
    task_names: list[str],
    task_criticalities: list[str],
    task_deadlines: list[int],
    task_c_hi: list[int],
    initial_budgets: dict[str, int],
    rounding_mode: str,
    min_budget_delta: int,
    forbid_decreasing_hi_budgets: bool,
    budget_floor_ratio: float,
    deploy_cap_enabled: bool,
    deploy_cap_ratio: float,
    deploy_cap_criticality: str,
) -> tuple[bool, ...]:
    """Compute the mask exclusively through the shared evaluator."""

    mask: list[bool] = []
    for action in actions:
        evaluation = evaluate_budget_candidate(
            action=action,
            budgets=budgets,
            task_names=task_names,
            task_criticalities=task_criticalities,
            task_deadlines=task_deadlines,
            task_c_hi=task_c_hi,
            initial_budgets=initial_budgets,
            rounding_mode=rounding_mode,
            min_budget_delta=min_budget_delta,
            forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
            budget_floor_ratio=budget_floor_ratio,
            deploy_cap_enabled=deploy_cap_enabled,
            deploy_cap_ratio=deploy_cap_ratio,
            deploy_cap_criticality=deploy_cap_criticality,
        )
        mask.append(bool(evaluation["accepted"]))
    return tuple(mask)


def valid_action_mask(**kwargs: object) -> tuple[bool, ...]:
    """Public mask alias in the frozen proof contract."""

    return formal_valid_action_mask(**kwargs)


def step(
    *,
    action_id: int,
    actions: tuple[dict[str, object], ...],
    budgets: dict[str, int],
    evaluator_kwargs: dict[str, object],
) -> dict[str, object]:
    """Apply exactly the action indexed by ``action_id`` or preserve budgets."""

    action = actions[action_id]
    evaluation = evaluate_budget_candidate(action=action, budgets=budgets, **evaluator_kwargs)
    if not bool(evaluation["accepted"]):
        return {
            "selected_action_id": None,
            "budgets": dict(budgets),
            "reject_reason": evaluation["reject_reason"],
        }
    return {
        "selected_action_id": action_id,
        "budgets": dict(evaluation["candidate_budgets"]),
        "reject_reason": None,
    }
