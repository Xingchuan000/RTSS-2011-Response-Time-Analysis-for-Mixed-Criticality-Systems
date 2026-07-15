"""FormalTarget 的 task/priority canonical 导出与 fingerprint。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
from typing import Any, Mapping, Sequence

from formal_toolchain.core.hashing import sha256_object


def canonical_task(task: Any, *, priority_index: int, budget_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not is_dataclass(task):
        raise TypeError("task 必须是实际 dataclass Task")
    info = dict(budget_info or {})
    initial = int(info.get("initial_runtime_budget", task.c_lo))
    floor = int(info.get("budget_floor", 1))
    cap = int(info.get("budget_cap", task.c_hi))
    record = {"priority_index": int(priority_index), "name": str(task.name),
              "criticality": getattr(task.criticality, "value", str(task.criticality)),
              "period": int(task.period), "deadline": int(task.deadline),
              "code_c_lo": int(task.c_lo), "code_c_hi": int(task.c_hi),
              "initial_runtime_budget": initial, "budget_floor": floor,
              "budget_cap": cap}
    if not (floor <= initial <= cap and cap >= int(task.c_hi)):
        raise ValueError(f"task {task.name} 的 budget 区间无效")
    return record


def export_taskset(ordered_tasks: Sequence[Any], budget_metadata: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if budget_metadata is None or any(str(task.name) not in budget_metadata for task in ordered_tasks):
        raise ValueError("formal taskset 必须从明确 budget provenance 提供所有任务预算字段")
    tasks = [canonical_task(task, priority_index=index,
                            budget_info=(budget_metadata or {}).get(str(task.name)))
             for index, task in enumerate(ordered_tasks)]
    priority = [task["name"] for task in tasks]
    return {"schema_version": "code_taskset_canonical_v1", "ordered_tasks": tasks,
            "priority_order": priority, "fingerprint": sha256_object({"tasks": tasks, "priority": priority})}


def derive_feature_task_order(feature_names: Sequence[str]) -> list[str]:
    """从 feature names 的 T00..T11 前缀提取顺序，不做排序修复。"""
    result: list[str] = []
    pattern = re.compile(r"^T(?P<slot>\d{2})\.(?P<task>[^.]+)\.(?P<feature>[^.]+)$")
    slots: dict[int, str] = {}
    counts: dict[str, int] = {}
    for name in feature_names:
        match = pattern.fullmatch(str(name))
        if match is None:
            continue
        slot = int(match.group("slot"))
        task_name = match.group("task")
        previous = slots.setdefault(slot, task_name)
        if previous != task_name:
            raise ValueError(f"feature slot T{slot:02d} 绑定了多个 task")
        counts[task_name] = counts.get(task_name, 0) + 1
    if slots and (min(slots) != 0 or sorted(slots) != list(range(0, max(slots) + 1))):
        raise ValueError("feature task slot 不连续")
    for slot in sorted(slots):
        task_name = slots[slot]
        if task_name not in result:
            result.append(task_name)
    if any(count <= 0 for count in counts.values()):
        raise ValueError("feature task slot 不得为空")
    # 合成 profile 的 global feature 必须全部位于 task feature 之后；混入 task
    # block 后的 Txx 或出现不符合 G. 约定的尾部字段都拒绝。
    task_positions = [index for index, name in enumerate(feature_names) if pattern.fullmatch(str(name))]
    if task_positions and task_positions != list(range(max(task_positions) + 1)):
        raise ValueError("task feature block 必须位于 global feature 之前")
    if any(not str(name).startswith("G.") for name in feature_names[max(task_positions, default=-1) + 1:]):
        raise ValueError("global feature 必须使用 G. 前缀并位于固定尾部")
    if slots and (any(count != 10 for count in counts.values()) or
                  len(feature_names) - len(task_positions) != 8):
        raise ValueError("P0 feature profile 要求每个 task 10 个 feature、global 8 个 feature")
    return result


def derive_action_task_order(action_definitions: Sequence[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for action in action_definitions:
        for key in ("increase_task", "target_task", "task_name"):
            value = action.get(key)
            if isinstance(value, str) and value not in result:
                result.append(value)
        for key in ("decrease_tasks", "target_tasks"):
            value = action.get(key)
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and item not in result:
                        result.append(item)
    return result
