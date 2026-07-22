from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True, order=True)
class EffectiveEvent:
    time: int
    order_rank: int
    event_type: str
    task_name: str | None
    release_index: int | None
    token: int | None


def _event_type_name(event) -> str:
    event_type = getattr(event, "event_type")
    return str(getattr(event_type, "value", event_type))


def is_effective_event(event, snapshot) -> bool:
    event_type = _event_type_name(event)
    key = None
    if event.task_name is not None and event.release_index is not None:
        key = (event.task_name, int(event.release_index))

    if event_type == "JOB_COMPLETION":
        return key is not None and snapshot.completion_token(key) == event.token
    if event_type == "BUDGET_OVERRUN":
        return key is not None and snapshot.overrun_token(key) == event.token
    if event_type == "RESPONSE_TIME_EXPIRY":
        return key is not None and snapshot.response_token(key) == event.token
    if event_type == "DEADLINE_CHECK":
        return key in getattr(snapshot, "active_job_keys", ())
    if event_type == "JOB_ARRIVAL":
        return True
    if event_type in {"RECOVERY", "CONTROLLER"}:
        return True
    raise ValueError(f"unknown event type: {event_type}")


def effective_frontier(queue_snapshot, runtime_snapshot):
    events = []
    order_table = {
        "RECOVERY": 0,
        "DEADLINE_CHECK": 1,
        "JOB_ARRIVAL": 2,
        "BUDGET_UPDATE": 3,
        "JOB_COMPLETION": 4,
        "BUDGET_OVERRUN": 5,
        "RESPONSE_TIME_EXPIRY": 6,
        "CONTROLLER": 7,
    }
    for event in queue_snapshot:
        if is_effective_event(event, runtime_snapshot):
            event_type = _event_type_name(event)
            events.append(
                EffectiveEvent(
                    time=int(event.time),
                    order_rank=int(order_table.get(event_type, 99)),
                    event_type=event_type,
                    task_name=event.task_name,
                    release_index=event.release_index,
                    token=event.token,
                )
            )
    return tuple(sorted(events))
