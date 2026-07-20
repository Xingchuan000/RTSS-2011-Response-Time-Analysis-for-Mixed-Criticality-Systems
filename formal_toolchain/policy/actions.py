"""Phase G03：生产预算动作与独立 binary64 重放。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from amc_py.budget_runtime import BudgetState
from amc_py.rl.actions import apply_budget_action_candidate
from amc_py.models import Criticality
from amc_py.rl.actions import BudgetAction
from formal_toolchain.core.hashing import sha256_object


def replay_action(
    action: BudgetAction, budgets: Mapping[str, int], tasks: Sequence[Any],
    *, rounding_mode: str = "ceil_floor", min_budget_delta: int = 1,
) -> dict[str, int]:
    """独立按 CPython binary64、ceil/floor 和 P0 clamp 顺序重算动作。"""
    if action.is_constraint_guided_pair or action.is_residual_ranked:
        raise ValueError("动态动作不属于第一轮 single verifier 子集")
    names = [str(task.name) for task in tasks]
    result: dict[str, int] = {}
    if action.is_noop:
        return result
    if action.increase_idx is not None:
        task = tasks[action.increase_idx]; name = names[action.increase_idx]
        raw_value = int(budgets[name]) * (1.0 + float(action.increase_ratio))
        value = math.ceil(raw_value) if rounding_mode == "ceil_floor" else int(round(raw_value))
        value = max(value, int(budgets[name]) + int(min_budget_delta))
        cap = task.c_hi if task.criticality is Criticality.HI else task.deadline
        result[name] = max(1, min(value, cap))
    for index in action.decrease_indices:
        name = names[index]
        raw_value = int(budgets[name]) * (1.0 - float(action.decrease_ratio))
        value = math.floor(raw_value) if rounding_mode == "ceil_floor" else int(round(raw_value))
        value = min(value, int(budgets[name]) - int(min_budget_delta))
        result[name] = max(1, value)
    return result


def _affected_task_indices(action: BudgetAction) -> tuple[int, ...]:
    indices: list[int] = []
    if action.increase_idx is not None:
        indices.append(int(action.increase_idx))
    indices.extend(int(index) for index in action.decrease_indices)
    return tuple(dict.fromkeys(indices))


def build_action_transition_table(
    actions: Sequence[BudgetAction], tasks: Sequence[Any],
    budget_domain: Mapping[str, Mapping[str, Any]],
    *, rounding_mode: str = "ceil_floor", min_budget_delta: int = 1,
) -> dict[str, Any]:
    task_names = [str(task.name) for task in tasks]
    initial = {name: int(budget_domain[name]["initial"]) for name in task_names}
    summaries: list[dict[str, Any]] = []
    for action in actions:
        if action.is_constraint_guided_pair or action.is_residual_ranked:
            raise ValueError("Phase G03 structural backend 只接受 single action")

        affected = _affected_task_indices(action)
        if action.is_noop:
            affected = ()
        if len(affected) > 1:
            raise ValueError("single backend 遇到多目标 action")

        digest_rows: list[dict[str, Any]] = []
        checked = 0
        min_after: int | None = None
        max_after: int | None = None
        if not affected:
            probes = [None]
        else:
            index = affected[0]
            name = task_names[index]
            interval = budget_domain[name]["integer_interval"]
            probes = range(int(interval["lower"]), int(interval["upper"]) + 1)

        for probe in probes:
            budgets = dict(initial)
            if affected:
                name = task_names[affected[0]]
                budgets[name] = int(probe)

            formal = replay_action(
                action, budgets, tasks, rounding_mode=rounding_mode,
                min_budget_delta=min_budget_delta,
            )
            production = apply_budget_action_candidate(
                action=action,
                budget_state=BudgetState(dict(budgets)),
                ordered_tasks=tasks,
                rounding_mode=rounding_mode,
                min_budget_delta=min_budget_delta,
            )
            if production != formal:
                return {
                    "status": "FAIL",
                    "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {
                        "code": "ACTION_TRANSITION_PRODUCTION_MISMATCH",
                        "action_id": int(action.action_id),
                        "probe": probe,
                        "formal": formal,
                        "production": production,
                    },
                }

            digest_rows.append({"before": probe, "updates": formal})
            checked += 1
            for value in formal.values():
                min_after = value if min_after is None else min(min_after, value)
                max_after = value if max_after is None else max(max_after, value)

        summaries.append(
            {
                "action_id": int(action.action_id),
                "affected_task_indices": list(affected),
                "checked_value_count": checked,
                "transition_digest": sha256_object(digest_rows),
                "min_after": min_after,
                "max_after": max_after,
                "increase_ratio_hex": float(action.increase_ratio).hex(),
                "decrease_ratio_hex": float(action.decrease_ratio).hex(),
                "frame_condition": [
                    name for idx, name in enumerate(task_names) if idx not in affected
                ],
            }
        )
    return {
        "status": "PASS",
        "schema_version": "action_transition_table_v2",
        "action_count": len(actions),
        "actions": summaries,
        "semantic": "production_action_primitive_1d_complete_replay",
        "rounding_mode": rounding_mode,
        "min_budget_delta": int(min_budget_delta),
    }


def verify_action_against_production(action: BudgetAction, budgets: Mapping[str, int], tasks: Sequence[Any],
                                     production_apply: Any) -> dict[str, Any]:
    """由 compiler/adapter 显式传入 production callable，执行单点差分。"""
    expected = replay_action(action, budgets, tasks)
    actual = production_apply(action=action, budget_state=BudgetState(dict(budgets)), ordered_tasks=tasks)
    if expected != actual:
        return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                "failure": {"code": "ACTION_TRANSITION_MISMATCH", "expected": expected, "actual": actual}}
    return {"status": "PASS", "expected": expected, "actual": actual}
