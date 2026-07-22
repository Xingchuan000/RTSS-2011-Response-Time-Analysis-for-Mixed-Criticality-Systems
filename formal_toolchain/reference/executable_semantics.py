from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from formal_toolchain.reference.reference_state import (
    ReferenceState, ReferenceJob, JobKey, PendingReferenceRelease,
    ReferenceModeSwitch,
)
from formal_toolchain.reference.c_amc_sem_semantics import (
    classify_arrival_batch, release_class_and_budget,
)
from formal_toolchain.bridge.logical_events import LogicalEvent, LogicalEventKind, PHASE_RANK
from formal_toolchain.adapters.formal_runtime_snapshot import (
    ReleasedJobRecord, TerminalRecord, MissRecord,
)


REFERENCE_TRANSITION_CASES = (
    "SAME_TIME_REMOVAL",
    "IDLE_RECOVERY",
    "DEADLINE_OBSERVATION",
    "ARRIVAL_BATCH",
    "MODE_SWITCH",
    "JOB_RELEASE",
    "DISPATCH",
    "SERVICE_TICK",
    "IDLE_JUMP_TO_NEXT_RELEASE",
)


def _tasks(taskset: Any) -> tuple[Any, ...]:
    tasks = taskset.tasks if hasattr(taskset, "tasks") else taskset.get("tasks", ())
    return tuple(tasks)


def _field(task: Any, name: str) -> Any:
    return task[name] if isinstance(task, Mapping) else getattr(task, name)


def _task_name(task: Any) -> str:
    return str(_field(task, "name"))


def _task_int(task: Any, name: str) -> int:
    value = _field(task, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"REFERENCE_TASK_FIELD_NOT_INT:{_task_name(task)}:{name}")
    return value


def _task_priority(task: Any) -> int:
    return _task_int(task, "priority_index")


def _task_by_name(taskset: Any, name: str) -> Any:
    matches = [task for task in _tasks(taskset) if _task_name(task) == name]
    if len(matches) != 1:
        raise ValueError(f"REFERENCE_TASK_NOT_UNIQUE:{name}")
    return matches[0]


def _canonical_frontier(
    events: Iterable[LogicalEvent],
    taskset: Any,
) -> tuple[LogicalEvent, ...]:
    grouped: dict[int, list[LogicalEvent]] = {}
    for event in events:
        grouped.setdefault(event.time, []).append(event)

    result: list[LogicalEvent] = []
    for time in sorted(grouped):
        batch_events = [e for e in grouped[time] if e.kind == LogicalEventKind.ARR_BATCH]
        non_batch = [e for e in grouped[time] if e.kind != LogicalEventKind.ARR_BATCH]

        merged_batches: list[LogicalEvent] = []
        seen_jobs: set[JobKey] = set()
        all_fifo: list[int] = []
        for b in sorted(batch_events, key=lambda e: e.fifo_rank):
            for jk in b.batch_jobs:
                if jk not in seen_jobs:
                    seen_jobs.add(jk)
                    all_fifo.append(b.fifo_rank)
        if batch_events:
            merged_batches.append(LogicalEvent(
                time=time,
                phase_rank=PHASE_RANK[LogicalEventKind.ARR_BATCH],
                kind=LogicalEventKind.ARR_BATCH,
                batch_jobs=tuple(sorted(
                    seen_jobs,
                    key=lambda key: (
                        _task_priority(_task_by_name(taskset, key[0])),
                        key[0], key[1],
                    ),
                )),
                fifo_rank=min(all_fifo) if all_fifo else 0,
            ))

        time_events = merged_batches + non_batch
        time_events.sort(key=lambda e: (e.phase_rank, e.fifo_rank, e.kind.value, e.job_key or ("", -1), e.batch_jobs))
        result.extend(time_events)

    return tuple(result)


def _arrival_event_for_job(
    *,
    task: Any,
    release_index: int,
    release_time: int,
    fifo_rank: int,
) -> LogicalEvent:
    if release_index < 0:
        raise ValueError("REFERENCE_RELEASE_INDEX_NEGATIVE")
    return LogicalEvent(
        time=release_time,
        phase_rank=PHASE_RANK[LogicalEventKind.ARR_BATCH],
        kind=LogicalEventKind.ARR_BATCH,
        batch_jobs=((_task_name(task), release_index),),
        fifo_rank=fifo_rank,
    )


def pop_event(state: ReferenceState, event: LogicalEvent, taskset: Any) -> ReferenceState:
    frontier = list(state.frontier)
    try:
        frontier.remove(event)
    except ValueError:
        pass
    return replace(state, frontier=_canonical_frontier(frontier, taskset))


def _job_schedule_key(
    job_key: JobKey,
    jobs: Mapping[JobKey, ReferenceJob],
    taskset: Any,
) -> tuple[int, int, str, int]:
    job = jobs[job_key]
    task = _task_by_name(taskset, job_key[0])
    return (_task_priority(task), job.release_time, job_key[0], job_key[1])


def _normalize_dispatch(state: ReferenceState, taskset: Any) -> ReferenceState:
    if not state.jobs:
        return replace(state, running=None, ready_order=())
    ordered = tuple(sorted(
        state.jobs,
        key=lambda key: _job_schedule_key(key, state.jobs, taskset),
    ))
    return replace(state, running=ordered[0], ready_order=ordered[1:])


def has_same_time_non_service_event(state: ReferenceState) -> bool:
    for event in state.frontier:
        if event.time == state.time and event.kind != LogicalEventKind.SVC:
            return True
    return False


def _append_generated_event(
    *,
    frontier: list[LogicalEvent],
    parent_event: LogicalEvent | None,
    generated_event: LogicalEvent,
) -> None:
    if parent_event is not None:
        if generated_event.time < parent_event.time:
            raise ValueError("REFERENCE_GENERATED_EVENT_IN_PAST")

        if (
            generated_event.time == parent_event.time
            and generated_event.kind is not LogicalEventKind.SVC
            and generated_event.phase_rank <= parent_event.phase_rank
        ):
            raise ValueError("REFERENCE_GENERATED_EVENT_PHASE_NOT_STRICT")

    frontier.append(generated_event)


def next_logical_event(state: ReferenceState) -> LogicalEvent | None:
    candidates = [e for e in state.frontier if e.time >= state.time]
    if not candidates:
        return None
    return min(candidates, key=lambda e: (e.time, e.phase_rank, e.fifo_rank, e.kind.value, e.job_key or ("", -1), e.batch_jobs))


def _update_ready_order(state: ReferenceState, taskset: Any) -> tuple[JobKey, ...]:
    active = {k: v for k, v in state.jobs.items()}
    ready = []
    for jk, job in active.items():
        if state.running == jk:
            continue
        task = _task_by_name(taskset, jk[0])
        priority = _task_priority(task)
        ready.append((priority, job.release_time, jk[0], jk[1], jk))
    ready.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return tuple(r[4] for r in ready)


def apply_removal(state: ReferenceState, event: LogicalEvent, taskset: Any) -> ReferenceState:
    jobs = dict(state.jobs)
    key = event.job_key
    if key in jobs:
        job = jobs[key]
        del jobs[key]
        terminal = dict(state.terminal)
        terminal[key] = TerminalRecord(
            job_key=key,
            terminal_kind="COMPLETED",
            terminal_time=state.time,
            executed_service=job.executed,
        )
        new_state = replace(state, jobs=jobs, terminal=terminal)
    else:
        new_state = state
    new_state = pop_event(new_state, event, taskset)
    normalized = _normalize_dispatch(new_state, taskset)
    if (
        normalized.mode == "HI"
        and not normalized.jobs
        and normalized.running is None
        and not normalized.pending_releases
    ):
        rec = LogicalEvent(
            time=state.time,
            phase_rank=PHASE_RANK[LogicalEventKind.REC],
            kind=LogicalEventKind.REC,
            fifo_rank=event.fifo_rank + 1,
        )
        frontier = list(normalized.frontier)
        _append_generated_event(
            frontier=frontier,
            parent_event=event,
            generated_event=rec,
        )
        normalized = replace(normalized, frontier=_canonical_frontier(frontier, taskset))
    return normalized


def apply_recovery(state: ReferenceState, event: LogicalEvent, taskset: Any) -> ReferenceState:
    if state.mode != "HI":
        raise ValueError("REFERENCE_RECOVERY_OUTSIDE_HI")
    if state.jobs or state.running is not None or state.pending_releases:
        raise ValueError("REFERENCE_RECOVERY_NOT_QUIESCENT")
    return pop_event(replace(state, mode="LO"), event, taskset)


def apply_deadline_observation(state: ReferenceState, event: LogicalEvent, taskset: Any) -> ReferenceState:
    key = event.job_key
    if key is None:
        return pop_event(state, event, taskset)
    if key in state.terminal:
        return pop_event(state, event, taskset)
    if key not in state.jobs:
        return pop_event(state, event, taskset)
    job = state.jobs[key]
    if job.executed >= job.budget:
        return pop_event(state, event, taskset)
    misses = list(state.misses)
    task = _task_by_name(taskset, key[0])
    misses.append(MissRecord(
        job_key=key,
        criticality=job.criticality,
        release_time=job.release_time,
        release_class=state.released[key].release_class,
        mode_at_miss=state.mode,
        miss_time=state.time,
        absolute_deadline=job.absolute_deadline,
        executed_at_miss=job.executed,
        priority_index=_task_priority(task),
    ))
    return pop_event(replace(state, misses=tuple(misses)), event, taskset)


def _task_for_job_key(jk: tuple[str, int], taskset: Any) -> Any:
    return _task_by_name(taskset, jk[0])


def apply_arrival_batch(state: ReferenceState, event: LogicalEvent, taskset: Any) -> ReferenceState:
    pending = dict(state.pending_releases)
    frontier = list(state.frontier)
    classification = classify_arrival_batch(
        mode_before=state.mode,
        batch_jobs=event.batch_jobs,
        taskset=taskset,
        abnormal_hi_releases=state.abnormal_hi_releases,
    )

    for jk in event.batch_jobs:
        task = _task_by_name(taskset, jk[0])
        period = _task_int(task, "period")
        deadline = _task_int(task, "deadline")
        priority = _task_priority(task)
        release_time = event.time
        if jk in state.released or jk in state.jobs or jk in state.terminal or jk in pending:
            raise ValueError(f"REFERENCE_DUPLICATE_RELEASE:{jk[0]}:{jk[1]}")
        release_class, effective_mode, budget = release_class_and_budget(
            task=task,
            mode_before_batch=classification.mode_before,
            mode_after_batch=classification.mode_after,
            abnormal_hi=jk in classification.abnormal_hi_jobs,
            switched_in_this_batch=classification.switch_trigger is not None,
            primary_on_switch_time=state.primary_on_switch_time,
        )
        pending[jk] = PendingReferenceRelease(
            job_key=jk,
            release_time=release_time,
            absolute_deadline=release_time + deadline,
            criticality=str(_field(task, "criticality")),
            priority_index=priority,
            abnormal_hi=jk in classification.abnormal_hi_jobs,
            effective_release_mode=effective_mode,
            release_class=release_class,
            release_budget=budget,
        )
        next_arrival = _arrival_event_for_job(
            task=task,
            release_index=jk[1] + 1,
            release_time=release_time + period,
            fifo_rank=event.fifo_rank + jk[1] + 1,
        )
        _append_generated_event(frontier=frontier, parent_event=event, generated_event=next_arrival)

    if classification.switch_trigger is not None:
        _append_generated_event(
            frontier=frontier,
            parent_event=event,
            generated_event=LogicalEvent(
                time=event.time,
                phase_rank=PHASE_RANK[LogicalEventKind.SW],
                kind=LogicalEventKind.SW,
                job_key=classification.switch_trigger,
                fifo_rank=event.fifo_rank,
            ),
        )
    for jk in event.batch_jobs:
        _append_generated_event(
            frontier=frontier,
            parent_event=event,
            generated_event=LogicalEvent(
                time=event.time,
                phase_rank=PHASE_RANK[LogicalEventKind.REL],
                kind=LogicalEventKind.REL,
                job_key=jk,
                fifo_rank=event.fifo_rank + jk[1],
            ),
        )

    frontier = [e for e in frontier if e != event]
    new_state = replace(
        state,
        pending_releases=pending,
        frontier=_canonical_frontier(frontier, taskset),
    )
    return _normalize_dispatch(new_state, taskset)


def apply_mode_switch(state: ReferenceState, event: LogicalEvent, taskset: Any) -> ReferenceState:
    if state.mode != "LO":
        raise ValueError("REFERENCE_MODE_SWITCH_OUTSIDE_LO")
    key = event.job_key
    if key is None or key not in state.pending_releases or not state.pending_releases[key].abnormal_hi:
        raise ValueError("REFERENCE_MODE_SWITCH_TRIGGER_NOT_PENDING_ABNORMAL_HI")
    if event.time != state.pending_releases[key].release_time:
        raise ValueError("REFERENCE_MODE_SWITCH_TIME_MISMATCH")
    if any(record.switch_time == event.time for record in state.mode_switches):
        raise ValueError("REFERENCE_MODE_SWITCH_DUPLICATE_TIME")
    switch = ReferenceModeSwitch(event.time, key, "ABNORMAL_HI_ARRIVAL")
    return pop_event(replace(state, mode="HI", mode_switches=state.mode_switches + (switch,)), event, taskset)


def apply_release(state: ReferenceState, event: LogicalEvent, taskset: Any) -> ReferenceState:
    key = event.job_key
    if key is None or key not in state.pending_releases:
        raise ValueError("REFERENCE_RELEASE_PLAN_MISSING")
    plan = state.pending_releases[key]
    if key in state.released or key in state.jobs or key in state.terminal:
        raise ValueError(f"REFERENCE_DUPLICATE_RELEASE:{key[0]}:{key[1]}")
    released = dict(state.released)
    released[key] = ReleasedJobRecord(
        job_key=key,
        release_time=plan.release_time,
        absolute_deadline=plan.absolute_deadline,
        criticality=plan.criticality,
        released_mode=plan.effective_release_mode,
        release_class=plan.release_class,
        release_budget=plan.release_budget,
        raw_actual_cost=plan.release_budget,
        removal_demand=plan.release_budget,
        priority_index=plan.priority_index,
        provenance="reference_arrival",
    )
    jobs = dict(state.jobs)
    jobs[key] = ReferenceJob(
        job_key=key,
        release_time=plan.release_time,
        absolute_deadline=plan.absolute_deadline,
        criticality=plan.criticality,
        released_mode=plan.effective_release_mode,
        release_class=plan.release_class,
        budget=plan.release_budget,
        removal_demand=plan.release_budget,
    )
    frontier = [e for e in state.frontier if e != event]
    _append_generated_event(
        frontier=frontier,
        parent_event=event,
        generated_event=LogicalEvent(
            time=plan.absolute_deadline,
            phase_rank=PHASE_RANK[LogicalEventKind.DDL],
            kind=LogicalEventKind.DDL,
            job_key=key,
            fifo_rank=event.fifo_rank,
        ),
    )
    return _normalize_dispatch(replace(
        state, jobs=jobs, released=released,
        pending_releases={k: v for k, v in state.pending_releases.items() if k != key},
        frontier=_canonical_frontier(frontier, taskset),
    ), taskset)


def apply_dispatch(state: ReferenceState, event: LogicalEvent, taskset: Any) -> ReferenceState:
    return pop_event(_normalize_dispatch(state, taskset), event, taskset)


def apply_service_tick(state: ReferenceState, taskset: Any) -> ReferenceState:
    normalized = _normalize_dispatch(state, taskset)
    if normalized.running is None:
        raise ValueError("REFERENCE_SERVICE_TICK_WITHOUT_RUNNING")
    if normalized.running != state.running or normalized.ready_order != state.ready_order:
        raise ValueError("REFERENCE_SERVICE_STATE_NOT_NORMALIZED")
    future_times = [event.time for event in normalized.frontier if event.time > normalized.time]
    if future_times and min(future_times) < normalized.time + 1:
        raise ValueError("REFERENCE_SERVICE_CROSSES_EVENT_BOUNDARY")
    jobs = dict(state.jobs)
    rk = normalized.running
    if rk in jobs:
        old = jobs[rk]
        jobs[rk] = replace(old, executed=old.executed + 1)
        if jobs[rk].executed >= jobs[rk].budget:
            rem = LogicalEvent(
                time=state.time + 1,
                phase_rank=PHASE_RANK[LogicalEventKind.REM],
                kind=LogicalEventKind.REM,
                job_key=rk,
                fifo_rank=0,
            )
            frontier = list(state.frontier)
            _append_generated_event(
                frontier=frontier,
                parent_event=LogicalEvent(
                    time=state.time,
                    phase_rank=PHASE_RANK[LogicalEventKind.SVC],
                    kind=LogicalEventKind.SVC,
                    fifo_rank=0,
                ),
                generated_event=rem,
            )
            after = replace(state, jobs=jobs, frontier=_canonical_frontier(frontier, taskset))
        else:
            after = replace(state, jobs=jobs)
    else:
        after = state
    return _normalize_dispatch(replace(after, time=state.time + 1), taskset)


def apply_logical_event(state: ReferenceState, event: LogicalEvent, taskset: Any) -> ReferenceState:
    kind = event.kind
    if kind == LogicalEventKind.REM:
        return apply_removal(state, event, taskset)
    elif kind == LogicalEventKind.REC:
        return apply_recovery(state, event, taskset)
    elif kind == LogicalEventKind.DDL:
        return apply_deadline_observation(state, event, taskset)
    elif kind == LogicalEventKind.ARR_BATCH:
        return apply_arrival_batch(state, event, taskset)
    elif kind == LogicalEventKind.SW:
        return apply_mode_switch(state, event, taskset)
    elif kind == LogicalEventKind.REL:
        return apply_release(state, event, taskset)
    elif kind == LogicalEventKind.DSP:
        return apply_dispatch(state, event, taskset)
    elif kind == LogicalEventKind.SVC:
        return apply_service_tick(state, taskset)
    else:
        return pop_event(state, event, taskset)


def dispatch_if_needed(state: ReferenceState, taskset: Any) -> ReferenceState:
    return _normalize_dispatch(state, taskset)


def closure_measure(state: ReferenceState) -> tuple[int, ...]:
    ranks = sorted(set(PHASE_RANK.values()))
    return tuple(
        sum(
            1
            for event in state.frontier
            if event.time == state.time
            and event.kind is not LogicalEventKind.SVC
            and event.phase_rank == rank
        )
        for rank in ranks
    )


def close_timestamp(state: ReferenceState, taskset: Any) -> ReferenceState:
    current = state
    while has_same_time_non_service_event(current):
        event = next_logical_event(current)
        if event is None:
            break
        previous_measure = closure_measure(current)
        current = apply_logical_event(current, event, taskset)
        new_measure = closure_measure(current)
        if not (new_measure < previous_measure):
            raise ValueError("REFERENCE_CLOSURE_PHASE_REGRESSION")
    return dispatch_if_needed(current, taskset)


def initial_reference_state(
    taskset: Any,
    *,
    abnormal_hi_releases: Iterable[JobKey] = (),
    primary_on_switch_time: bool = True,
) -> ReferenceState:
    tasks = _tasks(taskset)
    if not tasks:
        raise ValueError("REFERENCE_TASKSET_EMPTY")

    arrivals = [
        _arrival_event_for_job(
            task=task,
            release_index=0,
            release_time=_task_int(task, "offset"),
            fifo_rank=index,
        )
        for index, task in enumerate(tasks)
    ]

    state = ReferenceState(
        time=0,
        mode="LO",
        jobs={},
        released={},
        terminal={},
        misses=(),
        ready_order=(),
        running=None,
        frontier=_canonical_frontier(arrivals, taskset),
        pending_releases={},
        mode_switches=(),
        abnormal_hi_releases=frozenset(abnormal_hi_releases),
        primary_on_switch_time=bool(primary_on_switch_time),
    )
    validate_reference_state(state, taskset)
    return state


def validate_reference_state(
    state: ReferenceState,
    taskset: Any,
) -> None:
    if state.time < 0:
        raise ValueError("REFERENCE_STATE_TIME_NEGATIVE")
    if state.mode not in ("LO", "HI"):
        raise ValueError("REFERENCE_STATE_MODE_INVALID")

    for event in state.frontier:
        if event.time < state.time:
            raise ValueError("REFERENCE_FRONTIER_EVENT_BEFORE_TIME")

    active_keys = set(state.jobs.keys())
    terminal_keys = set(state.terminal.keys())
    released_keys = set(state.released.keys())
    pending_keys = set(state.pending_releases.keys())
    if (active_keys & terminal_keys) or (active_keys & pending_keys) or (terminal_keys & pending_keys):
        raise ValueError("REFERENCE_STATE_LEDGER_SET_OVERLAP")
    if not active_keys <= released_keys:
        raise ValueError("REFERENCE_ACTIVE_NOT_SUBSET_RELEASED")
    if not terminal_keys <= released_keys:
        raise ValueError("REFERENCE_TERMINAL_NOT_SUBSET_RELEASED")
    if len(active_keys) != len(state.jobs):
        raise ValueError("REFERENCE_ACTIVE_KEYS_NOT_UNIQUE")

    for jk, job in state.jobs.items():
        record = state.released[jk]
        if (
            job.job_key != jk
            or record.job_key != jk
            or job.release_time != record.release_time
            or job.absolute_deadline != record.absolute_deadline
            or job.criticality != record.criticality
            or job.budget != record.release_budget
            or job.released_mode != record.released_mode
            or job.release_class != record.release_class
            or job.removal_demand != record.removal_demand
            or job.removal_demand != record.removal_demand
            or not 0 <= job.executed <= job.budget
        ):
            raise ValueError(f"REFERENCE_JOB_LEDGER_MISMATCH:{jk}")

    ready_set = set(state.ready_order)
    if len(ready_set) != len(state.ready_order):
        raise ValueError("REFERENCE_READY_ORDER_DUPLICATE")

    runnable = {k for k in active_keys if k != state.running}
    if state.running is not None and state.running not in active_keys:
        raise ValueError("REFERENCE_RUNNING_NOT_ACTIVE")
    if ready_set != runnable:
        raise ValueError("REFERENCE_READY_ORDER_MISMATCH")

    normalized = _normalize_dispatch(state, taskset)
    if normalized.running != state.running:
        raise ValueError("REFERENCE_RUNNING_NOT_HIGHEST_PRIORITY")
    if normalized.ready_order != state.ready_order:
        raise ValueError("REFERENCE_READY_ORDER_NOT_CANONICAL")

    active_deadlines = [
        event for event in state.frontier
        if event.kind == LogicalEventKind.DDL and event.job_key in active_keys
    ]
    deadline_keys = {event.job_key for event in active_deadlines}
    miss_keys = {miss.job_key for miss in state.misses}
    if any(key not in deadline_keys and key not in miss_keys for key in active_keys):
        raise ValueError("REFERENCE_ACTIVE_DEADLINE_EVENT_NOT_UNIQUE")
    if len(deadline_keys) != len(active_deadlines):
        raise ValueError("REFERENCE_ACTIVE_DEADLINE_EVENT_NOT_UNIQUE")
    rel_events = {
        event.job_key: event
        for event in state.frontier
        if event.kind == LogicalEventKind.REL and event.job_key is not None
    }
    if set(rel_events) != pending_keys:
        raise ValueError("REFERENCE_PENDING_RELEASE_EVENT_NOT_UNIQUE")
    for key, plan in state.pending_releases.items():
        if plan.job_key != key or plan.release_budget <= 0:
            raise ValueError(f"REFERENCE_PENDING_RELEASE_INVALID:{key}")
        if rel_events[key].time != plan.release_time:
            raise ValueError(f"REFERENCE_PENDING_RELEASE_TIME_MISMATCH:{key}")
    if len({record.switch_time for record in state.mode_switches}) != len(state.mode_switches):
        raise ValueError("REFERENCE_MODE_SWITCH_TIME_NOT_UNIQUE")
    for record in state.mode_switches:
        plan = state.pending_releases.get(record.triggering_job_key) or state.released.get(record.triggering_job_key)
        if plan is None or getattr(plan, "criticality", None) != "HI":
            raise ValueError("REFERENCE_MODE_SWITCH_TRIGGER_NOT_HI")
        if isinstance(plan, PendingReferenceRelease) and not plan.abnormal_hi:
            raise ValueError("REFERENCE_MODE_SWITCH_TRIGGER_NOT_ABNORMAL")
        if isinstance(plan, ReleasedJobRecord) and plan.release_class != "HI_ABNORMAL_SWITCH_TRIGGER":
            raise ValueError("REFERENCE_MODE_SWITCH_TRIGGER_NOT_ABNORMAL")
    for event in state.frontier:
        if event.kind == LogicalEventKind.REC and (
            state.mode != "HI" or state.jobs or state.running is not None or state.pending_releases
        ):
            raise ValueError("REFERENCE_RECOVERY_EVENT_NOT_QUIESCENT_HI")
    tasks = _tasks(taskset)
    arrival_events = [
        event for event in state.frontier
        if event.kind == LogicalEventKind.ARR_BATCH and event.time >= state.time
    ]
    for task in tasks:
        name = _task_name(task)
        generators = [event for event in arrival_events if any(jk[0] == name for jk in event.batch_jobs)]
        if len(generators) != 1:
            raise ValueError(f"REFERENCE_PERIODIC_GENERATOR_NOT_UNIQUE:{name}")
        event = generators[0]
        jk = next(jk for jk in event.batch_jobs if jk[0] == name)
        offset = _task_int(task, "offset")
        period = _task_int(task, "period")
        expected_time = offset + jk[1] * period
        if event.time != expected_time or event.time < state.time or jk[1] < 0:
            raise ValueError(f"REFERENCE_PERIODIC_GENERATOR_MISMATCH:{name}")
    if len(released_keys) != len({key for key in released_keys}):
        raise ValueError("REFERENCE_RELEASE_INDEX_DUPLICATE")

    expected_canonical = _canonical_frontier(state.frontier, taskset)
    if state.frontier != expected_canonical:
        raise ValueError("REFERENCE_FRONTIER_NOT_CANONICAL")


def is_closed_reference_state(
    state: ReferenceState,
    taskset: Any,
) -> bool:
    validate_reference_state(state, taskset)

    if has_same_time_non_service_event(state):
        return False

    normalized = _normalize_dispatch(state, taskset)
    return bool(
        normalized.running == state.running
        and normalized.ready_order == state.ready_order
    )


def least_future_release(
    *,
    time: int,
    offset: int,
    period: int,
) -> tuple[int, int]:
    if period <= 0 or not 0 <= offset < period or time < 0:
        raise ValueError("REFERENCE_PERIODIC_DOMAIN_INVALID")

    k = 0 if time < offset else (time - offset) // period + 1
    release_time = offset + k * period

    if not (
        k >= 0
        and release_time > time
        and (release_time - offset) % period == 0
    ):
        raise AssertionError("REFERENCE_PERIODIC_SUCCESSOR_INVALID")
    return release_time, k


def _checked_successor(
    *,
    state: ReferenceState,
    taskset: Any,
    case_id: str,
    events: list[LogicalEvent],
) -> tuple[ReferenceState, str, list[LogicalEvent]]:
    validate_reference_state(state, taskset)
    return state, case_id, events


def step_reference(state: ReferenceState, taskset: Any) -> tuple[ReferenceState, str, list]:
    validate_reference_state(state, taskset)

    if has_same_time_non_service_event(state):
        event = next_logical_event(state)
        before = state
        after = apply_logical_event(state, event, taskset)
        new_measure = closure_measure(after)
        old_measure = closure_measure(before)
        if not (new_measure < old_measure):
            raise ValueError("REFERENCE_CLOSURE_PHASE_REGRESSION")
        return _checked_successor(
            state=after,
            taskset=taskset,
            case_id=f"SAME_TIME_{event.kind.value}",
            events=[event],
        )

    closed = dispatch_if_needed(state, taskset)

    if closed.running is not None:
        after = apply_service_tick(closed, taskset)
        return _checked_successor(
            state=after,
            taskset=taskset,
            case_id="READY_SERVICE_OR_EARLIER_BOUNDARY",
            events=[LogicalEvent(
                time=closed.time,
                phase_rank=PHASE_RANK[LogicalEventKind.SVC],
                kind=LogicalEventKind.SVC,
                fifo_rank=0,
            )],
        )

    future = [event for event in closed.frontier if event.time > closed.time]
    if future:
        next_time = min(event.time for event in future)
        jumped = replace(closed, time=next_time)
        return _checked_successor(
            state=jumped,
            taskset=taskset,
            case_id="IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT",
            events=[],
        )

    raise ValueError("REFERENCE_VALID_STATE_PERIODIC_GENERATOR_INVARIANT_BROKEN")


def verify_frame_rule(before: ReferenceState, after: ReferenceState, footprint: set[JobKey]) -> bool:
    untouched = {k: v for k, v in before.jobs.items() if k not in footprint}
    for key, job in untouched.items():
        if key not in after.jobs or after.jobs[key] != job:
            return False
    return True
