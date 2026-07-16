"""synthetic target 的唯一 executable-policy runtime adapter。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.actions import (BudgetAction, action_violates_hi_decrease_guard,
                               apply_budget_action_candidate, build_budget_action_space)


def observation_from_state(state: Mapping[str, Any], tasks: Sequence[Task], feature_names: Sequence[str]) -> tuple[float, ...]:
    budgets = state["budgets"]
    caps = state["caps"]
    values: list[float] = []
    for task in tasks:
        ratio = float(budgets[task.name]) / float(caps[task.name])
        values.extend((ratio, ratio, ratio, ratio, 0.0, 0.0, 1.0 - ratio,
                       1.0 if task.criticality is Criticality.HI else 0.0,
                       1.0 - task.period / max(task.period, 1), ratio))
    values.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, min(float(values[0]), 1.0)))
    if len(values) != len(feature_names):
        raise ValueError("synthetic observation 与 feature_names 维度不一致")
    return tuple(values)


def _apply(action: BudgetAction, state: Mapping[str, Any], tasks: Sequence[Task]) -> dict[str, int]:
    return apply_budget_action_candidate(action=action,
                                         budget_state=BudgetState(dict(state["budgets"])),
                                         ordered_tasks=tasks)


def mask_and_reasons(state: Mapping[str, Any], actions: Sequence[BudgetAction], tasks: Sequence[Task]) -> tuple[tuple[bool, ...], tuple[str, ...]]:
    budgets = state["budgets"]
    floors = state.get("floors", {task.name: 1 for task in tasks})
    caps = state.get("caps", {task.name: (task.c_hi if task.criticality is Criticality.HI else task.deadline)
                                for task in tasks})
    config = state.get("config", type("SyntheticDefaultConfig", (), {"forbid_decreasing_hi_budgets": True,
        "enable_deploy_cap_mask": False, "deploy_cap_mask_criticality": "lo", "deploy_cap_mask_ratio": 1.0})())
    masks: list[bool] = []
    reasons: list[str] = []
    for action in actions:
        updates = _apply(action, state, tasks)
        reason = "accepted"
        valid = True
        if action_violates_hi_decrease_guard(action, tasks, bool(getattr(config, "forbid_decreasing_hi_budgets", True))):
            valid, reason = False, "decrease_hi_forbidden"
        elif not updates or all(int(budgets[name]) == int(value) for name, value in updates.items()):
            valid, reason = False, "no_effective_budget_change"
        elif any(int(value) < int(floors[name]) for name, value in updates.items()):
            valid, reason = False, "budget_floor_violation"
        elif any(int(value) > int(caps[name]) for name, value in updates.items()):
            valid, reason = False, "budget_upper_bound_violation"
        elif (getattr(config, "enable_deploy_cap_mask", False) and action.increase_idx is not None):
            task = tasks[action.increase_idx]
            if (getattr(config, "deploy_cap_mask_criticality", "lo") == "all" or task.criticality is Criticality.LO):
                initial = float(state["initial_budgets"][task.name])
                if initial and float(budgets[task.name]) / initial >= float(getattr(config, "deploy_cap_mask_ratio", 1.0)):
                    valid, reason = False, "deploy_cap_increase_mask"
        masks.append(valid)
        reasons.append(reason)
    return tuple(masks), tuple(reasons)


def build_runtime_adapter(target: Any, actions: Sequence[BudgetAction]) -> dict[str, Any]:
    tasks = target.ordered_tasks
    feature_names = target.feature_names
    def evaluate(state: Mapping[str, Any]) -> dict[str, Any]:
        observation = observation_from_state(state, tasks, feature_names)
        mask, reasons = mask_and_reasons(state, actions, tasks)
        return {"observation": observation, "mask": mask, "reasons": reasons}
    return {"evaluate": evaluate, "apply": lambda state, action_id: _apply(actions[action_id], state, tasks),
            "tasks": tasks, "actions": actions, "feature_names": feature_names}


def build_transition_witness(domain: Mapping[str, Any], tasks: Sequence[Task]) -> dict[str, dict[str, Any]]:
    """生成 synthetic 正常 release/completion/cancellation/recovery witness。"""
    if len(tasks) < 2:
        raise ValueError("synthetic transition witness 至少需要两个 task")
    initial = {task.name: int(domain["tasks"][task.name]["initial"]) for task in tasks}
    first_job = f"{tasks[0].name}:0"
    second_job = f"{tasks[1].name}:0"
    release_before = {first_job: initial[tasks[0].name]}
    release_after = dict(release_before); release_after[second_job] = initial[tasks[1].name]
    completion_before = dict(release_after); completion_after = {second_job: initial[tasks[1].name]}
    cancellation_before = dict(completion_after); cancellation_after = {}
    recovery_before = {f"{tasks[0].name}:1": initial[tasks[0].name]}; recovery_after = dict(recovery_before)
    rows = {"boot": {"budget_write": False, "budget_before": {}, "budget_after": initial,
                      "active_release_before": {}, "active_release_after": {}, "transition_semantics": "boot"},
            "release_snapshot": {"budget_write": False, "budget_before": initial, "budget_after": initial,
                      "active_release_before": release_before, "active_release_after": release_after,
                      "transition_semantics": "release", "released_task": tasks[1].name, "released_job": second_job},
            "completion": {"budget_write": False, "budget_before": initial, "budget_after": initial,
                      "active_release_before": completion_before, "active_release_after": completion_after,
                      "transition_semantics": "remove", "removed_job": first_job},
            "cancellation": {"budget_write": False, "budget_before": initial, "budget_after": initial,
                      "active_release_before": cancellation_before, "active_release_after": cancellation_after,
                      "transition_semantics": "remove", "removed_job": second_job},
            "recovery": {"budget_write": False, "budget_before": initial, "budget_after": initial,
                      "active_release_before": recovery_before, "active_release_after": recovery_after,
                      "transition_semantics": "preserve"},
            "active_release_snapshot": {"budget_write": False, "budget_before": initial, "budget_after": initial,
                      "active_release_before": recovery_before, "active_release_after": recovery_after,
                      "transition_semantics": "preserve"}}
    return rows
