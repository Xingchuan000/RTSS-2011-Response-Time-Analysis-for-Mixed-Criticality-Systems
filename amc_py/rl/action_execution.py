"""Shared budget-action candidate/guard pipeline.

The executor is intentionally quality-agnostic: q-AMC quality transitions are
owned by the native runtime, while this module only applies existing budget
actions.  Environment and wrapper callers can use it for parity checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task

from .actions import BudgetAction, action_violates_hi_decrease_guard, apply_budget_action_candidate
from .safety import RuntimeBudgetSafetyChecker


@dataclass(frozen=True, slots=True)
class BudgetActionExecutionConfig:
    rounding_mode: str = "ceil_floor"
    min_budget_delta: int = 1
    budget_floor_ratio: float = 0.0
    fixed_floor_by_task: Mapping[str, int] | None = None
    enable_deploy_cap_mask: bool = False
    deploy_cap_mask_ratio: float = 4.0
    deploy_cap_mask_criticality: str = "lo"
    forbid_decreasing_hi_budgets: bool = False
    check_safety: bool = True


@dataclass(frozen=True, slots=True)
class BudgetActionExecutionResult:
    accepted: bool
    noop: bool
    updates: dict[str, int]
    candidate_budgets: dict[str, int]
    reject_reason: str | None
    safety_checked: bool
    diagnostics: tuple[dict, ...] = ()


def evaluate_budget_action(
    *,
    action: BudgetAction | None,
    ordered_tasks: Sequence[Task],
    budget_state: BudgetState,
    initial_budgets: Mapping[str, int] | None = None,
    config: BudgetActionExecutionConfig | None = None,
    safety_checker: RuntimeBudgetSafetyChecker | None = None,
) -> BudgetActionExecutionResult:
    cfg = config or BudgetActionExecutionConfig()
    before = dict(budget_state.budgets)
    if action is None or action.is_noop:
        return BudgetActionExecutionResult(True, True, {}, before, None, False)
    if action_violates_hi_decrease_guard(action, ordered_tasks, cfg.forbid_decreasing_hi_budgets):
        return BudgetActionExecutionResult(False, False, {}, before, "decrease_hi_forbidden", False)
    try:
        updates = apply_budget_action_candidate(
            action=action,
            budget_state=budget_state,
            ordered_tasks=ordered_tasks,
            rounding_mode=cfg.rounding_mode,
            min_budget_delta=cfg.min_budget_delta,
        )
    except ValueError as exc:
        return BudgetActionExecutionResult(False, False, {}, before, str(exc), False)
    candidate = dict(before)
    candidate.update(updates)
    initial = dict(initial_budgets or budget_state.initial_budgets or before)
    fixed_floor = dict(cfg.fixed_floor_by_task or {})
    for task_name, value in updates.items():
        floor = max(1, math.ceil(initial[task_name] * cfg.budget_floor_ratio)) if cfg.budget_floor_ratio > 0 else 1
        floor = max(floor, int(fixed_floor.get(task_name, 1)))
        if value < floor:
            return BudgetActionExecutionResult(False, False, {}, before, f"budget_floor_violation:{task_name}", False)
    if cfg.enable_deploy_cap_mask and action.increase_idx is not None:
        task = ordered_tasks[action.increase_idx]
        if cfg.deploy_cap_mask_criticality == "all" or task.criticality is Criticality.LO:
            current_ratio = before[task.name] / max(1, initial[task.name])
            if current_ratio >= cfg.deploy_cap_mask_ratio:
                return BudgetActionExecutionResult(
                    False,
                    False,
                    {},
                    before,
                    (
                        f"deploy_cap_increase_mask:{task.name}:"
                        f"ratio={current_ratio:.6g}:cap={cfg.deploy_cap_mask_ratio:.6g}"
                    ),
                    False,
                )
    # The primitive already clips to the legacy upper bounds; retain an explicit
    # check so callers receive a deterministic result if a custom action is used.
    for task in ordered_tasks:
        upper = task.c_hi if task.criticality is Criticality.HI else task.deadline
        if candidate[task.name] > upper:
            return BudgetActionExecutionResult(False, False, {}, before, f"budget_upper_bound_violation:{task.name}", False)
    safety_checked = False
    diagnostics: tuple[dict, ...] = ()
    if cfg.check_safety and safety_checker is not None:
        safety_checked = True
        report = safety_checker.validate_candidate(candidate)
        diagnostics = tuple(report.diagnostics)
        if not report.accepted:
            return BudgetActionExecutionResult(False, False, {}, before, report.reason, True, diagnostics)
    return BudgetActionExecutionResult(True, False, updates, candidate, None, safety_checked, diagnostics)


__all__ = [
    "BudgetActionExecutionConfig",
    "BudgetActionExecutionResult",
    "evaluate_budget_action",
]
