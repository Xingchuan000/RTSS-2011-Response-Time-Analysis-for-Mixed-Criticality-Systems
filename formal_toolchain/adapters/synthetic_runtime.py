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
    masks: list[bool] = []
    reasons: list[str] = []
    budgets = state.get("budgets", {})
    criticality = state.get("criticality", {})
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
        masks.append(valid)
        reasons.append(reason)
    return tuple(masks), tuple(reasons)

