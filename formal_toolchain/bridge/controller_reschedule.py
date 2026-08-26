"""Certificate for controller force-reschedule stuttering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import obligation_certificate


def _status(value: Mapping[str, Any] | None) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return value.get("status", value.get("obligation_status"))


def compare_controller_state_frame(
    *, before: Mapping[str, Any], after: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the concrete PreClosed/PostClosed frame at one controller time."""

    fields = (
        "current_time", "mode", "active_keys", "ready_keys", "running_key",
        "released_job_snapshot", "released_job_service", "released_job_demand",
        "released_job_classification", "completion_status", "miss_status",
        "service", "effective_event_frontier",
    )
    equal = {field: before.get(field) == after.get(field) for field in fields}
    return {
        "status": "PASS" if all(equal.values()) else "FAIL",
        "fields": equal,
        "current_time_unchanged": equal["current_time"],
        "mode_unchanged": equal["mode"],
        "active_keys_unchanged": equal["active_keys"],
        "ready_keys_unchanged": equal["ready_keys"],
        "running_key_unchanged": equal["running_key"],
        "released_job_snapshot_unchanged": equal["released_job_snapshot"],
        "released_job_service_unchanged": equal["released_job_service"],
        "released_job_demand_unchanged": equal["released_job_demand"],
        "released_job_classification_unchanged": equal["released_job_classification"],
        "completion_miss_unchanged": equal["completion_status"] and equal["miss_status"],
        "service_unchanged": equal["service"],
        "effective_event_frontier_unchanged": equal["effective_event_frontier"],
    }


def build_controller_force_reschedule_certificate(
    *,
    source_binding: Mapping[str, Any],
    scheduler_certificate: Mapping[str, Any],
    effective_frontier_certificate: Mapping[str, Any],
    context_hash: str,
) -> dict[str, Any]:
    """Build the logical-key and effective-frontier force-reschedule proof."""

    source_ok = (
        _status(source_binding) == "PASS"
        and source_binding.get("preclosed_frame_source_stable") is True
        and source_binding.get("scheduler_selection_exact") is True
        and source_binding.get("scheduler_selector_contract") is True
        and source_binding.get("priority_map_contract") is True
        and source_binding.get("logical_removal_source_stable") is True
        and source_binding.get("logical_overrun_source_stable") is True
        and source_binding.get("completion_miss_status_source_stable") is True
        and source_binding.get("released_job_snapshot_source_stable") is True
    )
    scheduler_ok = _status(scheduler_certificate) == "PASS"
    frontier_ok = (
        _status(effective_frontier_certificate) == "PASS"
        and effective_frontier_certificate.get("preserved_if_preclosed") is True
    )
    logical_running_key_preserved_if_preclosed = source_ok and scheduler_ok
    active_keys_unchanged = source_ok
    ready_keys_unchanged = source_ok
    priority_order_unchanged = source_ok and scheduler_ok
    release_snapshots_unchanged = source_binding.get("released_job_fields_source_stable") is True
    released_job_snapshot_unchanged = source_binding.get("released_job_snapshot_source_stable") is True
    released_job_service_unchanged = source_binding.get("released_job_service_source_stable") is True
    released_job_demand_unchanged = source_binding.get("released_job_demand_source_stable") is True
    released_job_classification_unchanged = source_binding.get("released_job_classification_source_stable") is True
    demand_unchanged = released_job_demand_unchanged
    service_unchanged = source_binding.get("time_unchanged") is True
    completion_miss_unchanged = source_binding.get("completion_miss_status_unchanged") is True
    mode_unchanged = source_binding.get("mode_source_stable") is True
    effective_frontier_preserved_if_preclosed = source_ok and frontier_ok
    status = "PASS" if all((
        logical_running_key_preserved_if_preclosed,
        active_keys_unchanged,
        ready_keys_unchanged,
        priority_order_unchanged,
        release_snapshots_unchanged,
        released_job_snapshot_unchanged,
        released_job_service_unchanged,
        released_job_demand_unchanged,
        released_job_classification_unchanged,
        demand_unchanged,
        service_unchanged,
        mode_unchanged,
        completion_miss_unchanged,
        effective_frontier_preserved_if_preclosed,
    )) else "UNRESOLVED"
    return obligation_certificate(
        obligation_id="CONTROLLER_FORCE_RESCHEDULE_STUTTER",
        status=status,
        context_hash=context_hash,
        inputs={
            "source_binding_hash": source_binding.get("binding_hash"),
            "scheduler_certificate_hash": scheduler_certificate.get("binding_hash", scheduler_certificate.get("artifact_hash")),
            "effective_frontier_certificate_hash": effective_frontier_certificate.get("binding_hash", effective_frontier_certificate.get("artifact_hash")),
        },
        witness={
            "source_kind": "CONTROLLER_SYNCHRONOUS",
            "logical_running_key_preserved_if_preclosed": logical_running_key_preserved_if_preclosed,
            "proof_key": "(task_name, release_index)",
            "active_keys_unchanged": active_keys_unchanged,
            "ready_keys_unchanged": ready_keys_unchanged,
            "running_key_unchanged_if_preclosed": logical_running_key_preserved_if_preclosed,
            "priority_order_unchanged": priority_order_unchanged,
            "release_snapshots_unchanged": release_snapshots_unchanged,
            "released_job_snapshot_unchanged": released_job_snapshot_unchanged,
            "released_job_service_unchanged": released_job_service_unchanged,
            "released_job_demand_unchanged": released_job_demand_unchanged,
            "released_job_classification_unchanged": released_job_classification_unchanged,
            "job_demand_unchanged": demand_unchanged,
            "service_unchanged": service_unchanged,
            "mode_unchanged": mode_unchanged,
            "completion_miss_unchanged": completion_miss_unchanged,
            "effective_event_frontier_unchanged_if_preclosed": effective_frontier_preserved_if_preclosed,
            "raw_token_refresh_allowed": source_binding.get("event_frontier_refresh_allowed") is True,
            "token_refresh_formula_proof": dict(
                effective_frontier_certificate.get("token_refresh_formula_proof", {})
            ) if isinstance(effective_frontier_certificate.get("token_refresh_formula_proof"), Mapping) else {},
            "conditional_on_preclosed": True,
        },
        checker_id=__name__,
        checker_version="controller-force-reschedule-v1",
        failure=None if status == "PASS" else {"code": "CONTROLLER_FORCE_RESCHEDULE_STUTTER_UNRESOLVED"},
    )
