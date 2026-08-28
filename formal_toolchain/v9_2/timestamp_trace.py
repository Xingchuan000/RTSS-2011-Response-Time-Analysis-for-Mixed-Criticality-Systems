"""Machine-readable same-timestamp records exported by concrete runtime replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class TimestampSemanticRecord:
    time: int
    settled_completions: tuple[Any, ...] = ()
    recovery_before_deadline: Any = None
    deadline_observations: tuple[Any, ...] = ()
    frozen_arrivals: tuple[Any, ...] = ()
    mode_switch: Any = None
    controller_observation: Any = None
    controller_action: Any = None
    budget_after_controller: Mapping[str, int] | None = None
    preliminary_dispatch: Any = None
    final_dispatch: Any = None
    service_quantum: int = 0
    phase_order: tuple[str, ...] = (
        "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"
    )


def records_from_debug_events(events: Iterable[Mapping[str, Any]]) -> tuple[TimestampSemanticRecord, ...]:
    """Group concrete debug events by timestamp without inventing missing events."""

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for event in events:
        if "time" not in event:
            raise ValueError("CONCRETE_TRACE_EVENT_TIME_MISSING")
        grouped.setdefault(int(event["time"]), []).append(event)
    result = []
    for time, rows in sorted(grouped.items()):
        arrivals = tuple(row for row in rows if row.get("event") == "job_arrival")
        switches = tuple(row for row in rows if row.get("event") == "mode_switch")
        actions = tuple(row for row in rows if row.get("event") in {"budget_update", "controller_action"})
        result.append(TimestampSemanticRecord(
            time=time,
            frozen_arrivals=arrivals,
            mode_switch=switches[0] if switches else None,
            controller_observation=actions[0].get("observation") if actions else None,
            controller_action=actions[0].get("action_id") if actions else None,
            budget_after_controller=actions[0].get("updates") if actions else None,
        ))
    return tuple(result)


__all__ = ["TimestampSemanticRecord", "records_from_debug_events"]
