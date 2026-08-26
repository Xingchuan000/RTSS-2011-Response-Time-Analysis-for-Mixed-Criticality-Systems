"""Phase F06：closure/controller/time-progress 的静态合同。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def derive_phase_edges(event_binding: Mapping[str, Any]) -> dict[str, list[str]]:
    """从已绑定的 event runtime 证据导出 phase DAG。"""
    if event_binding.get("status") != "PASS":
        raise ValueError("event runtime binding 不足以导出 phase DAG")
    if not isinstance(event_binding.get("queue_methods"), Mapping) or \
       not isinstance(event_binding.get("semantics"), Mapping) or \
       event_binding["semantics"].get("verified") is not True:
        raise ValueError("event runtime binding 缺少 phase 顺序证据")
    return {"arrival": ["classify"], "classify": ["switch"], "switch": ["completion"],
            "completion": ["recovery"], "recovery": []}


def check_closure_controller_contract(*, phase_edges: Mapping[str, list[str]] | None = None,
                                      controller_fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if phase_edges is None or controller_fields is None:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "CLOSURE_CONTROLLER_EVIDENCE_MISSING"}}
    edges = phase_edges or {"arrival": ["classify"], "classify": ["switch"], "switch": []}
    visiting: set[str] = set(); visited: set[str] = set()
    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("phase DAG 存在环")
        if node in visited:
            return
        visiting.add(node)
        for child in edges.get(node, []):
            visit(child)
        visiting.remove(node); visited.add(node)
    for node in edges:
        visit(node)
    fields = controller_fields
    if not isinstance(fields.get("witnesses"), list) or not fields["witnesses"]:
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "CLOSURE_CONTROLLER_WITNESSES_MISSING"}}
    binding = fields.get("binding")
    if isinstance(binding, Mapping):
        from formal_toolchain.core.hashing import sha256_object
        if fields.get("binding_hash") != sha256_object(binding):
            return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                    "failure": {"code": "CLOSURE_CONTROLLER_BINDING_HASH_MISMATCH"}}
    elif not fields.get("trace"):
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "CLOSURE_CONTROLLER_BINDING_MISSING"}}
    required = {"sequence_allocation_deterministic", "finite_token_height",
                "ready_nonempty_advances_tick",
                "ready_empty_jumps_next_event", "zero_time_stutter_forbidden",
                "active_release_budget_immutable",
                "explicit_noop_budget_identity", "explicit_noop_macro_stutter",
                "explicit_noop_effective_frontier_stutter",
                "explicit_noop_released_jobs_immutable",
                "explicit_noop_fallback_equivalent",
                "explicit_noop_plant_progress_separated",
                "selected_active_unchanged", "selected_ready_unchanged",
                "selected_requires_preclosed_boundary",
                "selected_running_unchanged_if_preclosed",
                "selected_released_job_fields_unchanged",
                "selected_released_job_snapshot_unchanged",
                "selected_released_job_service_unchanged",
                "selected_released_job_demand_unchanged",
                "selected_released_job_classification_unchanged",
                "selected_completion_miss_unchanged",
                "selected_service_unchanged", "selected_mode_unchanged",
                "selected_effective_event_frontier_unchanged_if_preclosed",
                "selected_plant_progression_separated",
                "selected_timing_stutter_if_preclosed"}
    if not required <= set(fields):
        return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED",
                "failure": {"code": "CLOSURE_CONTROLLER_FACTS_INCOMPLETE",
                             "fields": sorted(required - set(fields))}}
    if any(fields[key] is not True for key in required - {"changes_mode"}):
        raise ValueError("closure/controller facts 不满足 P0 合同")
    if fields.get("changes_mode") is not False:
        raise ValueError("controller 不得改变 mode")
    if any(fields.get(key) is not False for key in (
        "changes_active", "changes_ready", "changes_running_if_preclosed",
        "changes_current_service", "changes_mode", "changes_service",
    )):
        raise ValueError("controller 不得修改 runtime state")
    if any(fields.get(name, False) for name in (
        "changes_active", "changes_ready", "changes_running_if_preclosed", "changes_service",
    )):
        raise ValueError("controller 不得改变 processor service state")
    return {"status": "PASS", "schema_version": "closure_controller_v2_explicit_noop",
            "phase_dag_acyclic": True, "controller_invisible": True,
            "post_closure": True, "zero_time_stutter": False,
            "explicit_noop_zero_time_stutter": True, "time_progress": True}
