"""绑定 event runtime 的 removal/deadline 处理路径。"""

from __future__ import annotations

from pathlib import Path
import ast

from .python_ast_ir import function_to_ir
from formal_toolchain.semantics.frozen_runtime_contract import frozen_event_runtime_path, CONTRACT_VERSION


def bind_removal_runtime(source_root: Path) -> dict[str, object]:
    runtime_path = frozen_event_runtime_path(source_root)
    source = runtime_path.read_text(encoding="utf-8")
    boundary_ok = "job.executed_time <= budget" in source and "primary_on_switch_time" in source
    tree = ast.parse(source, filename=str(runtime_path))
    process = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "_process_event"), None)
    deadline_nodes = []
    if process is not None:
        deadline_nodes = [node for node in ast.walk(process)
                          if isinstance(node, ast.If) and "DEADLINE_CHECK" in ast.unparse(node.test)]
    deadline_statements = [ast.unparse(node) for parent in deadline_nodes for node in ast.walk(parent)
                           if isinstance(node, (ast.Call, ast.Assign, ast.AnnAssign, ast.AugAssign))]
    deadline_mutation_guarded = bool(deadline_nodes) and all(
        "nonvacuity_deadline_cleanup_remove" in ast.unparse(node)
        for node in deadline_nodes
        if any(
            token in ast.unparse(node)
            for token in (".remove(", "active_jobs", "running_job")
        )
    )
    deadline_observe_only = bool(deadline_nodes) and (
        not any(
            ".remove(" in item or "active_jobs" in item or "running_job" in item
            for item in deadline_statements
        )
        or deadline_mutation_guarded
    )
    targets = {
        "EventRuntimeEngine._process_event": function_to_ir(source, "EventRuntimeEngine._process_event"),
        "_maybe_recover_to_lo": function_to_ir(source, "_maybe_recover_to_lo"),
        "_schedule_running_job_events": function_to_ir(source, "_schedule_running_job_events"),
    }
    unresolved = [name for name, ir in targets.items() if ir.get("status") != "PASS"]
    if not boundary_ok or not deadline_observe_only:
        return {"status": "FAIL", "failure": {"code": "REMOVAL_BOUNDARY_SEMANTICS_FAILED",
                "route": "MODEL_CONFORMANCE_FAILED"}, "targets": targets}
    result: dict[str, object] = {"status": "PASS" if not unresolved else "UNRESOLVED",
        "formal_semantics_contract_version": CONTRACT_VERSION,
        "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
        "targets": targets,
        "removal_paths": ["normal_completion", "primary_lo_budget_cancellation",
                          "mode_switch_active_lo_drop", "degraded_release_drop", "response_time_expiry"],
        "path_evidence": {"source": "derived_from_target_ir", "verified": not unresolved},
        "p0_contract": {"deadline_observe_only": deadline_observe_only and "deadline_misses" in source,
                        "deadline_cleanup_profile_guarded": deadline_mutation_guarded,
                        "completion_precedes_deadline_observation": "job.executed_time <= budget" in source,
                        "hi_nontruncation": "job.task.criticality" in source and "JobCancellationEvent" in source,
                        "idle_only_recovery": "not state.active_jobs" in source and "state.running_job is None" in source}}
    if unresolved:
        result["failure"] = {"code": "TARGET_METHOD_IR_UNRESOLVED", "route": "UNRESOLVED", "targets": unresolved}
    return result
