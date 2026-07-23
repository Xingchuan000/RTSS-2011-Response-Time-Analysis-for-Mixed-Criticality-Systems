from __future__ import annotations

from typing import Any

import pytest

from formal_toolchain.adapters.formal_runtime_snapshot import (
    _mode_name,
    build_formal_runtime_snapshot,
    _release_class_from_provenance,
    ReleasedJobRecord,
)
from formal_toolchain.reference.executable_semantics import (
    initial_reference_state,
    step_reference,
)


def test_released_mode_is_canonical_name():
    from amc_py.runtime_models import (
        SystemMode,
    )

    assert _mode_name(SystemMode.LO) == "LO"
    assert _mode_name(SystemMode.HI) == "HI"


def test_normal_hi_not_reclassified_by_subsequent_switch():
    taskset = {
        "tasks": (
            {
                "name": "hi",
                "criticality": "HI",
                "priority_index": 0,
                "period": 10,
                "offset": 0,
                "deadline": 10,
                "c_lo": 2,
                "c_hi": 5,
            },
            {
                "name": "lo",
                "criticality": "LO",
                "priority_index": 1,
                "period": 10,
                "offset": 0,
                "deadline": 10,
                "c_lo": 3,
                "c_hi": 1,
            },
        )
    }

    state = initial_reference_state(taskset)
    for _ in range(4):
        state, _, _ = step_reference(state, taskset)

    hi = state.released[("hi", 0)]
    assert hi.release_class == "HI_NORMAL"
    assert hi.released_mode == "LO"


def test_only_trigger_key_is_hi_abnormal_switch_trigger():
    taskset = {
        "tasks": (
            {
                "name": "hi",
                "criticality": "HI",
                "priority_index": 0,
                "period": 10,
                "offset": 0,
                "deadline": 10,
                "c_lo": 2,
                "c_hi": 5,
            },
            {
                "name": "hi2",
                "criticality": "HI",
                "priority_index": 1,
                "period": 10,
                "offset": 0,
                "deadline": 10,
                "c_lo": 2,
                "c_hi": 5,
            },
        )
    }

    state = initial_reference_state(
        taskset,
        abnormal_hi_releases={
            ("hi", 0),
            ("hi2", 0),
        },
    )

    for _ in range(4):
        state, _, _ = step_reference(state, taskset)

    assert state.released[("hi", 0)].release_class == "HI_ABNORMAL_SWITCH_TRIGGER"
    assert state.released[("hi2", 0)].release_class == "HI_NORMAL"


def test_same_batch_lo_is_lo_primary_same_batch_switch_time():
    taskset = {
        "tasks": (
            {
                "name": "hi",
                "criticality": "HI",
                "priority_index": 0,
                "period": 10,
                "offset": 0,
                "deadline": 10,
                "c_lo": 2,
                "c_hi": 5,
            },
            {
                "name": "lo",
                "criticality": "LO",
                "priority_index": 1,
                "period": 10,
                "offset": 0,
                "deadline": 10,
                "c_lo": 3,
                "c_hi": 1,
            },
        )
    }

    state = initial_reference_state(
        taskset,
        abnormal_hi_releases={("hi", 0)},
    )

    for _ in range(4):
        state, _, _ = step_reference(state, taskset)

    lo = state.released[("lo", 0)]
    assert lo.release_class == "LO_PRIMARY_SAME_BATCH_SWITCH_TIME"
    assert lo.released_mode == "LO"


def test_lo_before_switch_stays_lo_primary_normal():
    taskset = {
        "tasks": (
            {
                "name": "lo",
                "criticality": "LO",
                "priority_index": 0,
                "period": 10,
                "offset": 0,
                "deadline": 10,
                "c_lo": 3,
                "c_hi": 1,
            },
            {
                "name": "hi",
                "criticality": "HI",
                "priority_index": 1,
                "period": 10,
                "offset": 5,
                "deadline": 10,
                "c_lo": 2,
                "c_hi": 5,
            },
        )
    }

    state = initial_reference_state(
        taskset,
        abnormal_hi_releases={("hi", 0)},
    )

    for _ in range(6):
        state, _, _ = step_reference(state, taskset)

    lo = state.released[("lo", 0)]
    assert lo.release_class == "LO_PRIMARY_NORMAL"
    assert lo.released_mode == "LO"


def test_degraded_lo_is_lo_degraded_hi_mode():
    taskset = {
        "tasks": (
            {
                "name": "hi",
                "criticality": "HI",
                "priority_index": 0,
                "period": 10,
                "offset": 0,
                "deadline": 10,
                "c_lo": 2,
                "c_hi": 5,
            },
            {
                "name": "lo",
                "criticality": "LO",
                "priority_index": 1,
                "period": 6,
                "offset": 0,
                "deadline": 6,
                "c_lo": 3,
                "c_hi": 1,
            },
        )
    }

    state = initial_reference_state(
        taskset,
        abnormal_hi_releases={("hi", 0)},
    )
    for _ in range(14):
        state, _, _ = step_reference(state, taskset)

    lo = state.released[("lo", 1)]
    assert lo.release_class == "LO_DEGRADED_HI_MODE"
    assert lo.released_mode == "HI"


def test_removal_demand_is_fixed_across_switch():
    taskset = {
        "tasks": (
            {
                "name": "hi",
                "criticality": "HI",
                "priority_index": 0,
                "period": 10,
                "offset": 0,
                "deadline": 10,
                "c_lo": 2,
                "c_hi": 5,
            },
            {
                "name": "lo",
                "criticality": "LO",
                "priority_index": 1,
                "period": 20,
                "offset": 0,
                "deadline": 20,
                "c_lo": 3,
                "c_hi": 1,
            },
        )
    }

    state = initial_reference_state(
        taskset,
        abnormal_hi_releases={("hi", 0)},
    )

    snapshots = []
    for _ in range(8):
        state, _, _ = step_reference(state, taskset)
    for key, job in state.released.items():
        snapshots.append(job)

    for _ in range(12):
        state, _, _ = step_reference(state, taskset)

    for job in snapshots:
        key = job.job_key
        if key in state.released:
            after = state.released[key]
            assert job.release_class == after.release_class
            assert job.released_mode == after.released_mode
            assert job.release_budget == after.release_budget
            assert job.removal_demand == after.removal_demand
