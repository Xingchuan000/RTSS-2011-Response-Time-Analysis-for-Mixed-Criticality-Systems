"""Phase K04：事件投影，只保留 timing-relevant phase。"""

from __future__ import annotations

from .state_relation import P0Event

_ERASED = frozenset({"CONTROLLER", "OBSERVATION", "TREE", "MASK"})


def project_event(event: P0Event) -> P0Event | None:
    """按计划规则投影一个 concrete event。"""
    if event.kind in _ERASED:
        return None
    # 计划要求擦除 budget update 的标签；预算变化必须由投影状态中的
    # future_budget 字段承载，不能伪造为一个 reference event。
    if event.kind == "BUDGET_UPDATE_LABEL":
        return None
    if event.kind == "PRIMARY_LO_CANCELLATION":
        return P0Event(event.time, "JOB_COMPLETION", event.job_key, event.payload)
    if event.kind in {"NORMAL_COMPLETION", "DEGRADED_COMPLETION"}:
        return P0Event(event.time, "JOB_COMPLETION", event.job_key, event.payload)
    return event


def project_events(events: tuple[P0Event, ...] | list[P0Event]) -> tuple[P0Event, ...]:
    return tuple(projected for event in events if (projected := project_event(event)) is not None)
