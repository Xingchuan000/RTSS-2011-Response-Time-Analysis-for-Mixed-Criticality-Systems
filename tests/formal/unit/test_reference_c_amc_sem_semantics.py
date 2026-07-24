from formal_toolchain.reference.c_amc_sem_semantics import (
    ReferenceReleaseDecision,
    classify_arrival_batch,
    decide_reference_release,
)
from formal_toolchain.reference.executable_semantics import initial_reference_state, step_reference
from formal_toolchain.bridge.logical_events import LogicalEvent, LogicalEventKind, PHASE_RANK
import pytest


TASKSET = {"tasks": (
    {"name": "hi", "criticality": "HI", "priority_index": 0, "c_lo": 2, "c_hi": 5},
    {"name": "lo", "criticality": "LO", "priority_index": 1, "c_lo": 3, "c_hi": 1},
)}


def test_abnormal_hi_arrival_triggers_unique_switch():
    result = classify_arrival_batch(
        mode_before="LO", batch_jobs=(("hi", 0), ("lo", 0)), taskset=TASKSET,
        abnormal_hi_releases=frozenset({("hi", 0)}),
    )
    assert result.mode_after == "HI"
    assert result.switch_trigger == ("hi", 0)


def test_normal_hi_arrival_does_not_switch():
    result = classify_arrival_batch(
        mode_before="LO", batch_jobs=(("hi", 0),), taskset=TASKSET,
        abnormal_hi_releases=frozenset(),
    )
    assert result.switch_trigger is None
    assert result.mode_after == "LO"


def test_abnormal_hi_uses_c_hi():
    decision = decide_reference_release(
        task=TASKSET["tasks"][0],
        mode_before_batch="LO",
        mode_after_batch="HI",
        abnormal_hi=True,
        is_switch_trigger=True,
        switched_in_this_batch=True,
        primary_on_switch_time=True,
    )

    assert decision == ReferenceReleaseDecision(
        release_class=
            "HI_ABNORMAL_SWITCH_TRIGGER",
        effective_release_mode="LO",
        release_budget=5,
    )


def test_same_batch_lo_release_is_primary():
    decision = decide_reference_release(
        task=TASKSET["tasks"][1],
        mode_before_batch="LO",
        mode_after_batch="HI",
        abnormal_hi=False,
        is_switch_trigger=False,
        switched_in_this_batch=True,
        primary_on_switch_time=True,
    )

    assert decision == ReferenceReleaseDecision(
        release_class=
            "LO_PRIMARY_SAME_BATCH_SWITCH_TIME",
        effective_release_mode="LO",
        release_budget=3,
    )


def test_later_hi_mode_lo_release_is_degraded():
    decision = decide_reference_release(
        task=TASKSET["tasks"][1],
        mode_before_batch="HI",
        mode_after_batch="HI",
        abnormal_hi=False,
        is_switch_trigger=False,
        switched_in_this_batch=False,
        primary_on_switch_time=True,
    )

    assert decision == ReferenceReleaseDecision(
        release_class="LO_DEGRADED_HI_MODE",
        effective_release_mode="HI",
        release_budget=1,
    )


def test_hi_mode_normal_hi_release_uses_c_lo():
    decision = decide_reference_release(
        task=TASKSET["tasks"][0],
        mode_before_batch="HI",
        mode_after_batch="HI",
        abnormal_hi=False,
        is_switch_trigger=False,
        switched_in_this_batch=False,
        primary_on_switch_time=True,
    )

    assert decision == ReferenceReleaseDecision(
        release_class="HI_NORMAL",
        effective_release_mode="HI",
        release_budget=2,
    )


def test_same_batch_abnormal_non_trigger_uses_c_hi():
    decision = decide_reference_release(
        task=TASKSET["tasks"][0],
        mode_before_batch="LO",
        mode_after_batch="HI",
        abnormal_hi=True,
        is_switch_trigger=False,
        switched_in_this_batch=True,
        primary_on_switch_time=True,
    )
    assert decision == ReferenceReleaseDecision(
        release_class="HI_ABNORMAL",
        effective_release_mode="LO",
        release_budget=5,
    )


def test_existing_hi_mode_abnormal_release_uses_c_hi():
    decision = decide_reference_release(
        task=TASKSET["tasks"][0],
        mode_before_batch="HI",
        mode_after_batch="HI",
        abnormal_hi=True,
        is_switch_trigger=False,
        switched_in_this_batch=False,
        primary_on_switch_time=True,
    )
    assert decision == ReferenceReleaseDecision(
        release_class="HI_ABNORMAL",
        effective_release_mode="HI",
        release_budget=5,
    )


def test_same_batch_normal_hi_keeps_lo_release_mode():
    decision = decide_reference_release(
        task=TASKSET["tasks"][0],
        mode_before_batch="LO",
        mode_after_batch="HI",
        abnormal_hi=False,
        is_switch_trigger=False,
        switched_in_this_batch=True,
        primary_on_switch_time=True,
    )

    assert decision == ReferenceReleaseDecision(
        release_class="HI_NORMAL",
        effective_release_mode="LO",
        release_budget=2,
    )


def test_arrival_batch_decomposes_to_sw_and_rel():
    taskset = {"tasks": (
        {"name": "hi", "criticality": "HI", "priority_index": 0,
         "period": 10, "offset": 0, "deadline": 10, "c_lo": 2, "c_hi": 5},
    )}
    state = initial_reference_state(taskset, abnormal_hi_releases={("hi", 0)})
    state, case, events = step_reference(state, taskset)
    assert case == "SAME_TIME_ARR_BATCH"
    assert state.pending_releases
    assert any(event.kind == LogicalEventKind.SW for event in state.frontier)
    assert any(event.kind == LogicalEventKind.REL for event in state.frontier)


def test_recovery_requires_quiescence():
    taskset = {"tasks": (
        {"name": "hi", "criticality": "HI", "priority_index": 0,
         "period": 10, "offset": 0, "deadline": 10, "c_lo": 2, "c_hi": 5},
    )}
    state = initial_reference_state(taskset)
    event = LogicalEvent(time=0, phase_rank=PHASE_RANK[LogicalEventKind.REC],
                         kind=LogicalEventKind.REC, fifo_rank=0)
    with pytest.raises(ValueError, match="REFERENCE_RECOVERY_OUTSIDE_HI"):
        from formal_toolchain.reference.executable_semantics import apply_recovery
        apply_recovery(state, event, taskset)


def test_switch_batch_release_ledger_is_release_fixed():
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
        abnormal_hi_releases={
            ("hi", 0),
        },
    )

    for _ in range(4):
        state, _, _ = step_reference(
            state,
            taskset,
        )

    hi = state.released[("hi", 0)]
    lo = state.released[("lo", 0)]

    assert hi.release_class \
        == "HI_ABNORMAL_SWITCH_TRIGGER"
    assert hi.released_mode == "LO"
    assert hi.release_budget == 5

    assert lo.release_class \
        == "LO_PRIMARY_SAME_BATCH_SWITCH_TIME"
    assert lo.released_mode == "LO"
    assert lo.release_budget == 3
