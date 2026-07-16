"""Phase G03：生产预算动作与独立 binary64 重放。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from amc_py.budget_runtime import BudgetState
from amc_py.rl.actions import apply_budget_action_candidate
from amc_py.models import Criticality
from amc_py.rl.actions import BudgetAction


def replay_action(action: BudgetAction, budgets: Mapping[str, int], tasks: Sequence[Any]) -> dict[str, int]:
    """独立按 CPython binary64、ceil/floor 和 P0 clamp 顺序重算动作。"""
    if action.is_constraint_guided_pair or action.is_residual_ranked:
        raise ValueError("动态动作不属于第一轮 single verifier 子集")
    names = [str(task.name) for task in tasks]
    result: dict[str, int] = {}
    if action.is_noop:
        return result
    if action.increase_idx is not None:
        task = tasks[action.increase_idx]; name = names[action.increase_idx]
        value = math.ceil(int(budgets[name]) * (1.0 + float(action.increase_ratio)))
        cap = task.c_hi if task.criticality is Criticality.HI else task.deadline
        result[name] = max(1, min(value, cap))
    for index in action.decrease_indices:
        name = names[index]
        result[name] = max(1, math.floor(int(budgets[name]) * (1.0 - float(action.decrease_ratio))))
    return result


def build_action_transition_table(actions: Sequence[BudgetAction], tasks: Sequence[Any],
                                  budget_domain: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for action in actions:
        if action.is_constraint_guided_pair or action.is_residual_ranked:
            raise ValueError("Phase G03 只接受已绑定的 s185 single action subset")
        for task_name, info in budget_domain.items():
            # s185 第一轮只激活 single action，因此让每个被选任务遍历完整
            # finite domain；其它任务固定为已认证 initial，不用随机抽样代替穷举。
            values = info.get("finite_integer_domain")
            if not isinstance(values, list):
                raise ValueError(f"{task_name} 缺少 finite integer domain")
            for probe_value in values:
                budgets = {name: int(row["initial"]) for name, row in budget_domain.items()}
                budgets[task_name] = int(probe_value)
                before = dict(budgets)
                after = replay_action(action, budgets, tasks)
                production_updates = apply_budget_action_candidate(
                    action=action, budget_state=BudgetState(dict(budgets)), ordered_tasks=tasks)
                if production_updates != after:
                    return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                            "failure": {"code": "ACTION_TRANSITION_PRODUCTION_MISMATCH",
                                        "action_id": action.action_id, "budget_before": before,
                                        "formal": after, "production": production_updates}}
                rows.append({"action_id": action.action_id, "probe_task": task_name,
                         "budget_before": before, "updates": after,
                         "increase_ratio_hex": float(action.increase_ratio).hex(),
                         "decrease_ratio_hex": float(action.decrease_ratio).hex()})
    return {"status": "PASS", "schema_version": "action_transition_table_v1",
            "action_count": len(actions), "rows": rows, "semantic": "production_action_primitive"}


def verify_action_against_production(action: BudgetAction, budgets: Mapping[str, int], tasks: Sequence[Any],
                                     production_apply: Any) -> dict[str, Any]:
    """由 compiler/adapter 显式传入 production callable，执行单点差分。"""
    expected = replay_action(action, budgets, tasks)
    actual = production_apply(action=action, budget_state=BudgetState(dict(budgets)), ordered_tasks=tasks)
    if expected != actual:
        return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                "failure": {"code": "ACTION_TRANSITION_MISMATCH", "expected": expected, "actual": actual}}
    return {"status": "PASS", "expected": expected, "actual": actual}
