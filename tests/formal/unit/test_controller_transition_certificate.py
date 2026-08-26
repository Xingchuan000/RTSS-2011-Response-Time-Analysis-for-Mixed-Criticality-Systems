from __future__ import annotations

from copy import deepcopy

from formal_toolchain.binding.action_binding import bind_action_runtime
from formal_toolchain.binding.controller_binding import bind_controller_runtime
from formal_toolchain.bridge.controller_transition import build_controller_transition_certificate
from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object


def _bindings() -> tuple[dict, dict, dict]:
    controller = bind_controller_runtime(".")
    action = bind_action_runtime(".", action_dim=25, explicit_noop=True)
    deployed = {"status": "PASS", "binding_hash": sha256_object({"binding": "deployed"})}
    return controller, action, deployed


def test_controller_certificate_has_independent_sync_cases() -> None:
    controller, action, deployed = _bindings()
    certificate = build_controller_transition_certificate(
        controller_binding=controller,
        action_binding=action,
        deployed_policy_binding=deployed,
        controller_postclosure_certificate={
            "obligation_status": "PASS",
            "artifact_hash": sha256_object({"controller_postclosure": "verified"}),
        },
        context_hash="0" * 64,
    )
    assert certificate["obligation_status"] == "PASS"
    assert verify_obligation_certificate(certificate)
    cases = certificate["witness"]["cases"]
    assert cases["CONTROLLER_NO_ACTION"]["source_kind"] == "CONTROLLER_SYNCHRONOUS"
    assert cases["CONTROLLER_NO_ACTION"]["noop_sources"] == (
        "EXPLICIT_NOOP",
        "DEFENSIVE_FALLBACK",
    )
    selected = cases["CONTROLLER_SELECTED_ACTION"]
    assert selected["source_binding"] == "self._engine.apply_budget_updates"
    for field in (
        "payload_prevalidated", "no_partial_mutation", "zero_time",
        "time_unchanged", "mode_unchanged", "active_jobs_unchanged",
        "ready_jobs_unchanged", "running_job_unchanged",
        "released_job_fields_unchanged", "released_job_snapshot_unchanged",
        "released_job_service_unchanged", "released_job_demand_unchanged",
        "released_job_classification_unchanged", "completion_miss_unchanged",
        "service_unchanged", "effective_event_frontier_unchanged",
    ):
        assert selected[field] is True
    assert selected["timing_projection"] == "STUTTER"
    assert selected["plant_progression"] is False


def test_controller_certificate_fails_closed_when_noop_stutter_is_tampered() -> None:
    controller, action, deployed = _bindings()
    controller = deepcopy(controller)
    controller["explicit_noop_runtime_binding"]["time_unchanged"] = False
    certificate = build_controller_transition_certificate(
        controller_binding=controller,
        action_binding=action,
        deployed_policy_binding=deployed,
        controller_postclosure_certificate={
            "obligation_status": "PASS",
            "artifact_hash": sha256_object({"controller_postclosure": "verified"}),
        },
        context_hash="0" * 64,
    )
    assert certificate["obligation_status"] == "UNRESOLVED"


def test_controller_certificate_does_not_accept_queued_budget_update_source() -> None:
    controller, action, deployed = _bindings()
    controller["selected_action_runtime_binding"]["source_binding"] = "EventRuntimeEngine._process_event(BUDGET_UPDATE)"
    certificate = build_controller_transition_certificate(
        controller_binding=controller,
        action_binding=action,
        deployed_policy_binding=deployed,
        controller_postclosure_certificate={
            "obligation_status": "PASS",
            "artifact_hash": sha256_object({"controller_postclosure": "verified"}),
        },
        context_hash="0" * 64,
    )
    assert certificate["obligation_status"] == "UNRESOLVED"
