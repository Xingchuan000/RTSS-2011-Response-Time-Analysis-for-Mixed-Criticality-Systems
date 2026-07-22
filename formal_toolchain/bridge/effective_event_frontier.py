from __future__ import annotations

from typing import Any

from formal_toolchain.bridge.logical_events import (
    LogicalEvent,
    LogicalEventKind,
    PHASE_RANK,
    project_raw_event,
    project_arrival_batch,
    job_key_of,
    raw_event_kind,
)


def effective_frontier(queue_snapshot, runtime_snapshot):
    projected: list[LogicalEvent] = []
    arrivals_by_time: dict[int, list[Any]] = {}
    fifo_counter = 0
    for event in queue_snapshot:
        event_copy = event
        if not hasattr(event_copy, "fifo_rank"):
            event_copy = type("_EventWithRank", (), {
                "task_name": getattr(event, "task_name", None),
                "release_index": getattr(event, "release_index", None),
                "time": getattr(event, "time", 0),
                "event_type": getattr(event, "event_type", None),
                "token": getattr(event, "token", None),
                "fifo_rank": fifo_counter,
            })()
            fifo_counter += 1
        raw = raw_event_kind(event_copy)
        if raw == "JOB_ARRIVAL":
            t = int(event_copy.time)
            arrivals_by_time.setdefault(t, []).append(event_copy)
            continue
        projected.extend(project_raw_event(event_copy, runtime_snapshot))
    for t, evts in sorted(arrivals_by_time.items()):
        batch = project_arrival_batch(evts, time=t)
        projected.append(batch)
    return tuple(sorted(projected))
