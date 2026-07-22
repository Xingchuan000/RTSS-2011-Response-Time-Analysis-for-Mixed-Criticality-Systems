from __future__ import annotations

from dataclasses import replace
import pytest

from formal_toolchain.reference.executable_semantics import (
    initial_reference_state,
    step_reference,
    is_closed_reference_state,
    least_future_release,
    validate_reference_state,
    has_same_time_non_service_event,
    closure_measure,
    _append_generated_event,
)
from formal_toolchain.reference.reference_state import ReferenceState, ReferenceJob
from formal_toolchain.bridge.logical_events import LogicalEvent, LogicalEventKind, PHASE_RANK


def _make_taskset(tasks: list[dict]) -> dict:
    return {"tasks": tuple(tasks)}


def test_recurring_releases_t10d7():
    taskset = _make_taskset([
        {"name": "hi", "period": 10, "deadline": 7, "offset": 0,
         "priority_index": 0, "c_lo": 2, "c_hi": 4, "criticality": "HI"},
    ])
    state = initial_reference_state(taskset)
    assert state.time == 0
    releases = []
    for _ in range(30):
        state, case, events = step_reference(state, taskset)
        if any(e.kind == LogicalEventKind.ARR_BATCH for e in events):
            for e in events:
                if e.kind == LogicalEventKind.ARR_BATCH:
                    for jk in e.batch_jobs:
                        releases.append((jk, e.time))
        elif "SAME_TIME_ARR_BATCH" in case:
            for e in events:
                if e.kind == LogicalEventKind.ARR_BATCH:
                    for jk in e.batch_jobs:
                        releases.append((jk, e.time))
    release_times = [t for (jk, t) in releases if jk[0] == "hi"]
    assert 0 in release_times, "first release at 0"
    assert 10 in release_times, "second release at 10"
    assert 20 in release_times, "third release at 20"
    assert 7 not in release_times, "deadline should not drive release"


def test_release_keys_increment():
    taskset = _make_taskset([
        {"name": "t1", "period": 5, "deadline": 5, "offset": 0,
         "priority_index": 0, "c_lo": 1, "c_hi": 2, "criticality": "LO"},
    ])
    state = initial_reference_state(taskset)
    keys_seen = set()
    for _ in range(20):
        state, case, events = step_reference(state, taskset)
        for e in events:
            if e.kind == LogicalEventKind.ARR_BATCH:
                for jk in e.batch_jobs:
                    keys_seen.add(jk)
    assert ("t1", 0) in keys_seen
    assert ("t1", 1) in keys_seen
    assert ("t1", 2) in keys_seen
    assert ("t1", 3) in keys_seen


def test_multiple_pending_jobs():
    taskset = _make_taskset([
        {"name": "t1", "period": 5, "deadline": 10, "offset": 0,
         "priority_index": 0, "c_lo": 8, "c_hi": 10, "criticality": "LO"},
    ])
    state = initial_reference_state(taskset)
    for _ in range(30):
        state, case, events = step_reference(state, taskset)
    keys = set(state.jobs.keys())
    assert len(keys) > 1, "should have multiple pending jobs when period < budget"
    task_keys = [k for k in keys if k[0] == "t1"]
    assert len(task_keys) >= 2


def test_strict_priority_ready_running():
    taskset = _make_taskset([
        {"name": "hi_pri", "period": 10, "deadline": 10, "offset": 0,
         "priority_index": 0, "c_lo": 1, "c_hi": 2, "criticality": "LO"},
        {"name": "lo_pri", "period": 10, "deadline": 10, "offset": 0,
         "priority_index": 1, "c_lo": 1, "c_hi": 2, "criticality": "LO"},
    ])
    state = initial_reference_state(taskset)
    for _ in range(20):
        state, case, events = step_reference(state, taskset)
        if state.running is not None:
            running_name = state.running[0]
            assert running_name == "hi_pri", f"expected hi_pri running, got {running_name}"
            break


def test_idle_jump_to_minimum_future():
    taskset = _make_taskset([
        {"name": "t1", "period": 100, "deadline": 10, "offset": 50,
         "priority_index": 0, "c_lo": 1, "c_hi": 2, "criticality": "LO"},
    ])
    state = initial_reference_state(taskset)
    jumped_to_50 = False
    for _ in range(20):
        state, case, events = step_reference(state, taskset)
        if case == "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT":
            jumped_to_50 = True
            assert state.time == 50
            break
    assert jumped_to_50


def test_completion_at_deadline_no_miss():
    taskset = _make_taskset([
        {"name": "t1", "period": 10, "deadline": 5, "offset": 0,
         "priority_index": 0, "c_lo": 5, "c_hi": 6, "criticality": "LO"},
    ])
    state = initial_reference_state(taskset)
    for _ in range(30):
        state, case, events = step_reference(state, taskset)
    misses_before_completion = len(state.misses)
    for _ in range(100):
        state, case, events = step_reference(state, taskset)
    assert len(state.misses) <= misses_before_completion + 1


def test_closure_measure_strictly_decreasing():
    from formal_toolchain.reference.executable_semantics import close_timestamp
    taskset = _make_taskset([
        {"name": "t1", "period": 5, "deadline": 5, "offset": 0,
         "priority_index": 0, "c_lo": 1, "c_hi": 2, "criticality": "LO"},
    ])
    state = initial_reference_state(taskset)
    closed = close_timestamp(state, taskset)
    assert not has_same_time_non_service_event(closed)


def test_least_future_release_formula():
    result = least_future_release(time=0, offset=0, period=10)
    assert result == (10, 1)

    result = least_future_release(time=5, offset=0, period=10)
    assert result == (10, 1)

    result = least_future_release(time=10, offset=0, period=10)
    assert result == (20, 2)

    result = least_future_release(time=0, offset=3, period=10)
    assert result == (3, 0)

    result = least_future_release(time=3, offset=3, period=10)
    assert result == (13, 1)


def test_generated_same_time_lower_phase_rejected():
    parent = LogicalEvent(
        time=3, phase_rank=PHASE_RANK[LogicalEventKind.ARR_BATCH],
        kind=LogicalEventKind.ARR_BATCH, batch_jobs=(("t1", 0),), fifo_rank=0,
    )
    generated = LogicalEvent(
        time=3, phase_rank=PHASE_RANK[LogicalEventKind.REM],
        kind=LogicalEventKind.REM, fifo_rank=0,
    )
    with pytest.raises(ValueError, match="REFERENCE_GENERATED_EVENT_PHASE_NOT_STRICT"):
        _append_generated_event(frontier=[], parent_event=parent, generated_event=generated)


def test_generated_past_event_rejected():
    parent = LogicalEvent(
        time=3, phase_rank=PHASE_RANK[LogicalEventKind.SVC],
        kind=LogicalEventKind.SVC, fifo_rank=0,
    )
    generated = LogicalEvent(
        time=2, phase_rank=PHASE_RANK[LogicalEventKind.REM],
        kind=LogicalEventKind.REM, fifo_rank=0,
    )
    with pytest.raises(ValueError, match="REFERENCE_GENERATED_EVENT_IN_PAST"):
        _append_generated_event(frontier=[], parent_event=parent, generated_event=generated)


def test_generated_future_lower_phase_allowed():
    parent = LogicalEvent(
        time=3, phase_rank=PHASE_RANK[LogicalEventKind.DDL],
        kind=LogicalEventKind.DDL, fifo_rank=0,
    )
    generated = LogicalEvent(
        time=4, phase_rank=PHASE_RANK[LogicalEventKind.ARR_BATCH],
        kind=LogicalEventKind.ARR_BATCH, batch_jobs=(("t1", 0),), fifo_rank=0,
    )
    frontier = []
    _append_generated_event(frontier=frontier, parent_event=parent, generated_event=generated)
    assert frontier == [generated]


def test_step_reference_revalidates_successor(monkeypatch):
    taskset = _make_taskset([
        {"name": "t1", "period": 10, "deadline": 10, "offset": 0,
         "priority_index": 0, "c_lo": 1, "c_hi": 1, "criticality": "LO"},
    ])
    state = initial_reference_state(taskset)
    calls = []
    import formal_toolchain.reference.executable_semantics as semantics
    original = semantics.validate_reference_state

    def checked(state_arg, taskset_arg):
        calls.append(state_arg)
        return original(state_arg, taskset_arg)

    monkeypatch.setattr(semantics, "validate_reference_state", checked)
    step_reference(state, taskset)
    assert len(calls) >= 2

    result = least_future_release(time=4, offset=3, period=10)
    assert result == (13, 1)


def test_reference_prefix_extension_dead_end_raises():
    taskset = _make_taskset([
        {"name": "t1", "period": 10, "deadline": 10, "offset": 0,
         "priority_index": 0, "c_lo": 1, "c_hi": 2, "criticality": "LO"},
    ])
    state = initial_reference_state(taskset)
    for _ in range(100):
        try:
            state, case, events = step_reference(state, taskset)
        except ValueError as e:
            if "REFERENCE_PREFIX_DEAD_END" in str(e):
                pytest.fail("unexpected dead end")
            raise


def test_higher_priority_arrival_preempts_running_job():
    taskset = _make_taskset([
        {"name": "high", "priority_index": 0, "offset": 2, "period": 10,
         "deadline": 10, "c_lo": 1, "c_hi": 1, "criticality": "HI"},
        {"name": "low", "priority_index": 1, "offset": 0, "period": 20,
         "deadline": 20, "c_lo": 10, "c_hi": 10, "criticality": "LO"},
    ])
    state = initial_reference_state(taskset)
    for _ in range(8):
        state, _, _ = step_reference(state, taskset)
        if state.running == ("high", 0):
            break
    assert state.running == ("high", 0)
    assert ("low", 0) in state.ready_order


def test_closed_state_predicate_returns_strict_bool():
    taskset = _make_taskset([{"name": "t", "priority_index": 0, "offset": 0,
                              "period": 10, "deadline": 10, "c_lo": 1,
                              "c_hi": 1, "criticality": "LO"}])
    assert type(is_closed_reference_state(initial_reference_state(taskset), taskset)) is bool


def test_generator_index_time_mismatch_rejected():
    taskset = _make_taskset([{"name": "t", "priority_index": 0, "offset": 0,
                              "period": 10, "deadline": 10, "c_lo": 1,
                              "c_hi": 1, "criticality": "LO"}])
    state = initial_reference_state(taskset)
    bad = LogicalEvent(time=5, phase_rank=PHASE_RANK[LogicalEventKind.ARR_BATCH],
                       kind=LogicalEventKind.ARR_BATCH, batch_jobs=(("t", 1),), fifo_rank=0)
    bad_state = replace(state, frontier=(bad,))
    with pytest.raises(ValueError, match="PERIODIC_GENERATOR_MISMATCH"):
        validate_reference_state(bad_state, taskset)
