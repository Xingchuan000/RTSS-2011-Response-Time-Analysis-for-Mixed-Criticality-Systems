from types import SimpleNamespace

import pytest

from formal_toolchain.conformance.boot_controller import check_closure_controller_contract
from formal_toolchain.conformance.runtime_evidence import derive_controller_fields


def _target():
    return SimpleNamespace(
        ordered_tasks=(SimpleNamespace(period=10, deadline=10),),
    )


def _controller_binding(*, projection="STUTTER_IF_PRECLOSED"):
    noop = {
        "status": "PASS",
        "timing_projection": "STUTTER",
        "budget_identity": True,
        "running_job_unchanged": True,
        "mode_unchanged": True,
        "controller_time_unchanged": True,
        "effective_event_frontier_unchanged": True,
        "released_job_fields_unchanged": True,
        "explicit_and_fallback_same_timing_semantics": True,
        "plant_progress_separated": True,
    }
    selected = {
        "status": "PASS",
        "timing_projection": projection,
        "requires_preclosed_boundary": True,
        "active_jobs_unchanged": True,
        "ready_jobs_unchanged": True,
        "running_job_unchanged_if_preclosed": True,
        "service_unchanged": True,
        "mode_unchanged": True,
        "released_job_fields_unchanged": True,
        "released_job_snapshot_unchanged": True,
        "released_job_service_unchanged": True,
        "released_job_demand_unchanged": True,
        "released_job_classification_unchanged": True,
        "completion_miss_unchanged": True,
        "effective_event_frontier_unchanged_if_preclosed": True,
        "plant_progression_separated": True,
    }
    return {
        "status": "PASS",
        "explicit_noop_runtime_binding": noop,
        "selected_action_runtime_binding": selected,
    }


def _fields(*, projection="STUTTER_IF_PRECLOSED"):
    return derive_controller_fields(
        target=_target(),
        controller_binding=_controller_binding(projection=projection),
        event_binding={"status": "PASS", "fifo_sequence": "EventQueue._counter"},
        recovery_binding={"status": "PASS"},
        initial_state={"runtime_budgets": {"T0": 5}},
        boot={"initial_runtime_budget_snapshot": {"T0": 5}},
    )


def test_preflight_consumes_conditional_selected_action_stutter():
    fields = _fields()
    assert fields["selected_requires_preclosed_boundary"] is True
    assert fields["selected_running_unchanged_if_preclosed"] is True
    assert fields["selected_effective_event_frontier_unchanged_if_preclosed"] is True
    assert fields["changes_running_if_preclosed"] is False

    result = check_closure_controller_contract(
        phase_edges={"arrival": ["classify"], "classify": []},
        controller_fields=fields,
    )
    assert result["status"] == "PASS"


def test_preflight_does_not_accept_old_unconditional_selected_projection():
    fields = _fields(projection="STUTTER")
    assert fields["selected_timing_stutter_if_preclosed"] is False
    with pytest.raises(ValueError, match="closure/controller facts"):
        check_closure_controller_contract(
            phase_edges={"arrival": ["classify"], "classify": []},
            controller_fields=fields,
        )
