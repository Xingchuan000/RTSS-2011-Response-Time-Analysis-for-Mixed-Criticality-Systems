"""Canonical synchronous controller transition certificate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import obligation_certificate


CONTROLLER_SELECTED_ACTION = "CONTROLLER_SELECTED_ACTION"
CONTROLLER_NO_ACTION = "CONTROLLER_NO_ACTION"


def _status(value: Mapping[str, Any] | None) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return value.get("status", value.get("obligation_status"))


def build_controller_transition_certificate(
    *,
    controller_binding: Mapping[str, Any],
    action_binding: Mapping[str, Any],
    deployed_policy_binding: Mapping[str, Any],
    controller_postclosure_certificate: Mapping[str, Any],
    context_hash: str,
) -> dict[str, Any]:
    selected_binding = controller_binding.get("selected_action_runtime_binding", {})
    noop_binding = controller_binding.get("explicit_noop_runtime_binding", {})
    explicit_noop = action_binding.get("explicit_noop") is True
    postclosure_ok = _status(controller_postclosure_certificate) == "PASS"
    action_noop = action_binding.get("noop_binding", {})
    selected_required_true_fields = (
        "payload_prevalidated",
        "no_partial_mutation",
        "zero_time",
        "time_unchanged",
        "mode_unchanged",
        "active_jobs_unchanged",
        "ready_jobs_unchanged",
        "released_job_fields_unchanged",
        "released_job_snapshot_unchanged",
        "released_job_service_unchanged",
        "released_job_demand_unchanged",
        "released_job_classification_unchanged",
        "completion_miss_unchanged",
        "service_unchanged",
        "plant_progression_separated",
        "running_job_unchanged_if_preclosed",
        "effective_event_frontier_unchanged_if_preclosed",
    )
    noop_required_true_fields = (
        "budget_identity",
        "running_job_unchanged",
        "effective_event_frontier_unchanged",
        "released_job_fields_unchanged",
        "mode_unchanged",
        "time_unchanged",
        "explicit_and_fallback_same_timing_semantics",
        "plant_progress_separated",
    )
    selected_action_ok = all((
        _status(action_binding) == "PASS",
        action_binding.get("action_profile") == "single25_explicit_noop",
        _status(deployed_policy_binding) == "PASS",
        postclosure_ok,
        _status(selected_binding) == "PASS",
        selected_binding.get("source") == "amc_py/rl/env.py:AmcBudgetEnv.step",
        selected_binding.get("source_kind") == "CONTROLLER_SYNCHRONOUS",
        selected_binding.get("source_binding") == "self._engine.apply_budget_updates",
        selected_binding.get("timing_projection") == "STUTTER_IF_PRECLOSED",
        selected_binding.get("requires_preclosed_boundary") is True,
        all(selected_binding.get(field) is True for field in selected_required_true_fields),
    ))
    noop_runtime_ok = (
        _status(noop_binding) == "PASS"
        and noop_binding.get("timing_projection") == "STUTTER"
        and all(noop_binding.get(field) is True for field in noop_required_true_fields)
    )
    noop_action_ok = (
        explicit_noop
        and action_noop.get("present") is True
        and action_noop.get("action_id") is not None
        and action_noop.get("derived_action_ids") == [action_noop.get("action_id")]
        and action_noop.get("defensive_fallback_same_action_verified") is True
    )
    noop_ok = noop_runtime_ok and noop_action_ok
    controller_ok = _status(controller_binding) == "PASS"
    selected_case = {
        "case_id": CONTROLLER_SELECTED_ACTION,
        "source": selected_binding.get("source"),
        "source_kind": selected_binding.get("source_kind"),
        "source_binding": selected_binding.get("source_binding"),
        "zero_time": selected_binding.get("zero_time") is True,
        "plant_progression": not bool(selected_binding.get("plant_progression_separated")),
        "timing_projection": "STUTTER" if selected_action_ok else "UNRESOLVED",
        "payload_prevalidated": selected_binding.get("payload_prevalidated"),
        "no_partial_mutation": selected_binding.get("no_partial_mutation"),
        "time_unchanged": selected_binding.get("time_unchanged"),
        "mode_unchanged": selected_binding.get("mode_unchanged"),
        "active_jobs_unchanged": selected_binding.get("active_jobs_unchanged"),
        "ready_jobs_unchanged": selected_binding.get("ready_jobs_unchanged"),
        "running_job_unchanged": True if selected_action_ok else False,
        "released_job_fields_unchanged": selected_binding.get("released_job_fields_unchanged"),
        "released_job_snapshot_unchanged": selected_binding.get("released_job_snapshot_unchanged"),
        "released_job_service_unchanged": selected_binding.get("released_job_service_unchanged"),
        "released_job_demand_unchanged": selected_binding.get("released_job_demand_unchanged"),
        "released_job_classification_unchanged": selected_binding.get("released_job_classification_unchanged"),
        "completion_miss_unchanged": selected_binding.get("completion_miss_unchanged"),
        "service_unchanged": selected_binding.get("service_unchanged"),
        "effective_event_frontier_unchanged": True if selected_action_ok else False,
        "budget_effect": "CERTIFIED_ACTION_UPDATE",
        "status": "PASS" if selected_action_ok and controller_ok else "UNRESOLVED",
    }
    noop_case = {
        "case_id": CONTROLLER_NO_ACTION,
        "source_kind": "CONTROLLER_SYNCHRONOUS",
        "source_binding": "EXPLICIT_OR_FALLBACK_NOOP",
        "zero_time": noop_binding.get("time_unchanged") is True,
        "budget_effect": "IDENTITY",
        "timing_projection": noop_binding.get("timing_projection"),
        "plant_progression": False,
        "running_job_unchanged": noop_binding.get("running_job_unchanged"),
        "effective_event_frontier_unchanged": noop_binding.get("effective_event_frontier_unchanged"),
        "released_job_fields_unchanged": noop_binding.get("released_job_fields_unchanged"),
        "mode_unchanged": noop_binding.get("mode_unchanged"),
        "time_unchanged": noop_binding.get("time_unchanged"),
        "plant_separated": noop_binding.get("plant_separated", noop_binding.get("plant_progress_separated")),
        "noop_sources": ("EXPLICIT_NOOP", "DEFENSIVE_FALLBACK"),
        "same_action_id": noop_action_ok,
        "same_budget_semantics": noop_binding.get("budget_identity") is True,
        "same_timing_semantics": noop_binding.get("timing_projection") == "STUTTER",
        "status": "PASS" if noop_ok and controller_ok else "UNRESOLVED",
    }
    status = "PASS" if selected_action_ok and noop_ok and controller_ok else "UNRESOLVED"
    failure = None if status == "PASS" else {"code": "CONTROLLER_TRANSITION_BINDING_UNRESOLVED"}
    deployed_hash = deployed_policy_binding.get("artifact_hash", deployed_policy_binding.get("binding_hash"))
    return obligation_certificate(
        obligation_id="CONTROLLER_TRANSITION",
        status=status,
        context_hash=context_hash,
        inputs={
            "controller_binding_hash": controller_binding.get("binding_hash"),
            "action_binding_hash": action_binding.get("binding_hash"),
            "deployed_policy_binding_hash": deployed_hash,
            "controller_postclosure_hash": controller_postclosure_certificate.get("artifact_hash"),
        },
        witness={
            "case_ids": (CONTROLLER_SELECTED_ACTION, CONTROLLER_NO_ACTION),
            "cases": {
                CONTROLLER_SELECTED_ACTION: selected_case,
                CONTROLLER_NO_ACTION: noop_case,
            },
            "selected_action_ok": selected_action_ok,
            "noop_ok": noop_ok,
            "source_kind": "CONTROLLER_SYNCHRONOUS",
        },
        direct_predecessor_hashes={
            key: value for key, value in {
                "action_binding": action_binding.get("binding_hash"),
                "controller_binding": controller_binding.get("binding_hash"),
                "deployed_policy_binding": deployed_hash,
                "controller_postclosure": controller_postclosure_certificate.get("artifact_hash"),
            }.items() if isinstance(value, str)
        },
        checker_id=__name__,
        checker_version="phase-k-v1",
        failure=failure,
    )
