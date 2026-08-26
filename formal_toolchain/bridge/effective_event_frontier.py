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


def _controller_frontier_events(value):
    if isinstance(value, dict) and "frontier" in value:
        return tuple(value["frontier"])
    if hasattr(value, "effective_event_frontier"):
        return tuple(value.effective_event_frontier)
    return tuple(value)


def _controller_logical_event_key(event):
    def scalar(value):
        return value.value if hasattr(value, "value") else value

    if isinstance(event, dict):
        get = event.get
        kind = scalar(get("kind"))
        job_key = get("job_key")
        batch_jobs = get("batch_jobs", ())
        return (
            scalar(get("time", 0)),
            scalar(get("phase_rank", 0)),
            kind,
            tuple(job_key) if job_key is not None else None,
            tuple(tuple(item) for item in batch_jobs),
        )
    return (
        scalar(getattr(event, "time", 0)),
        scalar(getattr(event, "phase_rank", 0)),
        scalar(getattr(event, "kind", None)),
        tuple(getattr(event, "job_key")) if getattr(event, "job_key", None) is not None else None,
        tuple(tuple(item) for item in getattr(event, "batch_jobs", ())),
    )


def compare_controller_reschedule_frontier(*, before, after):
    """Compare logical ``EffQ`` frontiers while ignoring raw token identity."""

    before_keys = tuple(sorted(_controller_logical_event_key(event)
                               for event in _controller_frontier_events(before)))
    after_keys = tuple(sorted(_controller_logical_event_key(event)
                              for event in _controller_frontier_events(after)))
    def by_kind(kind):
        return tuple(item for item in before_keys if item[2] == kind), tuple(item for item in after_keys if item[2] == kind)
    removals_before, removals_after = by_kind("REM")
    arrivals_before, arrivals_after = by_kind("ARR_BATCH")
    deadlines_before, deadlines_after = by_kind("DDL")
    unchanged = before_keys == after_keys
    result = {
        "status": "PASS" if unchanged else "FAIL",
        "before_keys": before_keys,
        "after_keys": after_keys,
        "effective_frontier_unchanged": unchanged,
        "logical_removal_key_unchanged": tuple(item[3] for item in removals_before) == tuple(item[3] for item in removals_after),
        "logical_removal_kind_unchanged": tuple(item[2] for item in removals_before) == tuple(item[2] for item in removals_after),
        "logical_removal_time_unchanged": tuple(item[0] for item in removals_before) == tuple(item[0] for item in removals_after),
        "arrival_frontier_unchanged": arrivals_before == arrivals_after,
        "deadline_frontier_unchanged": deadlines_before == deadlines_after,
        "raw_token_ids_ignored": True,
    }
    result["status"] = "PASS" if all(
        result[field] is True for field in (
            "effective_frontier_unchanged",
            "logical_removal_key_unchanged",
            "logical_removal_kind_unchanged",
            "logical_removal_time_unchanged",
            "arrival_frontier_unchanged",
            "deadline_frontier_unchanged",
        )
    ) else "FAIL"
    return result
