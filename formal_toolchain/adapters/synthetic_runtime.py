"""synthetic fixture 的运行时 mask adapter。

该逻辑属于 fixture adapter，而不是验收脚本本身；因此 fresh verifier 可以
复用同一份明确的运行时语义，并且不会由验收入口临时拼出一套假运行时。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def evaluate_synthetic_runtime_mask(
    state: Mapping[str, Any], definitions: Sequence[Mapping[str, Any]], forbid_hi_decrease: bool = True
) -> tuple[tuple[bool, ...], tuple[str, ...]]:
    """按 fixture 声明的预算上下界重算 runtime mask。

    ``formal_toolchain.policy.mask_fallback`` 会把这个函数与独立形式化
    evaluator 做逐字段比较。这里不读取调用方已经算好的 mask；floor/cap
    也必须来自 state，缺失时不擅自使用默认值。
    """
    masks: list[bool] = []
    reasons: list[str] = []
    budgets = state.get("budgets", {})
    criticality = state.get("criticality", {})
    floors = state.get("floors", state.get("floor", {}))
    caps = state.get("caps", {})
    for definition in definitions:
        task = definition.get("target_task")
        direction = definition.get("direction", "increase")
        valid = isinstance(task, str) and task in budgets
        reason = "accepted"
        if not valid:
            reason = "unknown_target"
        elif direction == "decrease" and forbid_hi_decrease and criticality.get(task) == "HI":
            valid = False
            reason = "hi_decrease_guard"
        elif direction == "increase" and task in caps and int(budgets[task]) >= int(caps[task]):
            valid = False
            reason = "budget_upper_bound"
        elif direction == "decrease" and task in floors and int(budgets[task]) <= int(floors[task]):
            valid = False
            reason = "budget_floor"
        masks.append(valid)
        reasons.append(reason)
    return tuple(masks), tuple(reasons)
