from __future__ import annotations

from typing import Any, Iterable, Mapping

from .models import ActionRiskRecord


def classify_actions(
    *,
    action_definitions: Iterable[Mapping[str, Any]],
    tasks: Iterable[Mapping[str, Any]],
) -> list[ActionRiskRecord]:
    task_by_name = {
        str(row["name"]): _normalize_task(row)
        for row in tasks
        if isinstance(row, Mapping) and "name" in row
    }
    hi_tasks = [row for row in task_by_name.values() if _crit(row) == "HI"]
    records: list[ActionRiskRecord] = []
    for action in action_definitions:
        action_id = int(action["action_id"])
        target = _target_task(action)
        task = task_by_name.get(target) if target is not None else None
        direction = _direction(action)
        interfered: list[str] = []
        if task is not None and direction == "INCREASE":
            target_priority = int(task["priority"])
            for hi in hi_tasks:
                if target_priority < int(hi["priority"]):
                    interfered.append(str(hi["name"]))
        if task is not None and _crit(task) == "HI" and direction == "DECREASE":
            risk_class = "HI_BUDGET_DECREASE"
        elif task is not None and _crit(task) == "LO" and interfered:
            risk_class = "HIGHER_PRIORITY_LO_INCREASE"
        elif direction == "INCREASE":
            risk_class = "BUDGET_INCREASE"
        else:
            risk_class = "BENIGN_OR_UNKNOWN"
        records.append(
            ActionRiskRecord(
                action_id=action_id,
                action_kind=str(action.get("kind", action.get("type", direction or "unknown"))),
                target_task=target,
                target_criticality=_crit(task),
                target_priority=int(task["priority"]) if task is not None else None,
                budget_direction=direction,
                interferes_with_hi_tasks=tuple(sorted(interfered)),
                risk_class=risk_class,
            )
        )
    return records


def _normalize_task(task: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(task)
    priority = result.get("priority", result.get("priority_index"))
    if priority is None:
        raise ValueError(f"task 缺少 priority/priority_index: {result.get('name')}")
    result["priority"] = int(priority)
    return result


def _crit(task: Mapping[str, Any] | None) -> str | None:
    if task is None:
        return None
    return str(task.get("criticality", task.get("level", ""))).upper()


def _target_task(action: Mapping[str, Any]) -> str | None:
    for key in ("task", "task_name", "target_task", "increase_task"):
        value = action.get(key)
        if isinstance(value, str) and value:
            return value
    decreases = action.get("decrease_tasks", action.get("decrease_task"))
    if isinstance(decreases, str) and decreases:
        return decreases
    if isinstance(decreases, (list, tuple)) and len(decreases) == 1:
        return str(decreases[0])
    return None


def _direction(action: Mapping[str, Any]) -> str | None:
    explicit = action.get("direction")
    if isinstance(explicit, str):
        normalized = explicit.strip().upper()
        if normalized in {"INCREASE", "DECREASE"}:
            return normalized
    if action.get("increase_task") is not None or action.get("increase_idx") is not None:
        return "INCREASE"
    decreases = action.get("decrease_tasks", action.get("decrease_indices", action.get("decrease_task")))
    if isinstance(decreases, str) and decreases:
        return "DECREASE"
    if isinstance(decreases, (list, tuple)) and bool(decreases):
        return "DECREASE"
    return None
