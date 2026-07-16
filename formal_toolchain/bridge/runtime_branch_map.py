"""从真实 runtime handler 生成 Phase K 完整 transition-path map。

这里不把任意 ``if`` 节点当作 transition。path selector 是有限的人工维护
绑定，但每条 path 必须绑定实际 handler 的源码 hash、guard hash、effect
hash 和 terminal point；源码变化会使旧 map 失效。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.adapters.source_manifest import build_source_manifest
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
from formal_toolchain.binding.removal_binding import bind_removal_runtime
from .p0_case_manifest import p0_case_manifest_hash
from .transition_cases import REQUIRED_P0_CASE_IDS


# selector 只选择完整 handler path；它不直接把某个 if 节点当 transition。
PATH_SPECS = (
    ("boot/preclosed_0", "EventRuntimeEngine.build", "BOOT_TO_PRECLOSED_0", 535, 586, "initial_state_and_queue", ("boot", "budget_frame")),
    ("arrival_batch/no_switch", "EventRuntimeEngine._process_job_arrival_batch", "ARRIVAL_BATCH_NO_SWITCH", 958, 986, "not switched_by_c_amc_sem_batch", ("arrival_batch", "release", "reschedule")),
    ("arrival_batch/switch_s0", "EventRuntimeEngine._maybe_enter_c_amc_sem_hi_mode_at_arrival", "ARRIVAL_BATCH_SWITCH_S0", 779, 816, "abnormal_arrivals and mode_is_LO", ("mode=HI", "mode_switch", "same_batch")),
    ("release/primary_lo", "EventRuntimeEngine._process_single_arrival_in_priority_order", "PRIMARY_LO_RELEASE", 848, 929, "release_mode=LO and task_is_LO", ("build_job", "release_fixed_budget", "active_add", "ready_add")),
    ("release/degraded_lo", "EventRuntimeEngine._process_single_arrival_in_priority_order", "DEGRADED_LO_RELEASE", 848, 929, "release_mode=HI and task_is_LO and c_amc_sem", ("degraded_budget", "actual_cost_clamp", "active_add", "ready_add")),
    ("release/hi", "EventRuntimeEngine._process_single_arrival_in_priority_order", "HI_RELEASE", 848, 929, "task_is_HI", ("build_job", "release_fixed_budget", "active_add", "ready_add")),
    ("dispatch/preempt", "_reschedule", "PREEMPTION_DISPATCH", 438, 469, "selected_is_not_previous_or_force", ("highest_priority_select", "running_update", "preempt_invalidate")),
    ("service/one_tick", "EventRuntimeEngine._advance_time", "ONE_SERVICE_TICK", 676, 704, "event_before_boundary", ("advance_time", "service_accounting", "remaining_update")),
    ("completion/normal", "EventRuntimeEngine._process_event", "NORMAL_COMPLETION", 1047, 1086, "event_is_JOB_COMPLETION and job_is_normal", ("executed_to_actual", "active_remove", "running_clear")),
    ("completion/degraded", "EventRuntimeEngine._process_event", "DEGRADED_COMPLETION", 1047, 1086, "event_is_JOB_COMPLETION and job_is_degraded", ("executed_to_actual", "active_remove", "running_clear")),
    ("completion/hi", "EventRuntimeEngine._process_event", "HI_COMPLETION", 1047, 1086, "event_is_JOB_COMPLETION and job_is_HI", ("executed_to_actual", "active_remove", "running_clear", "hi_complete")),
    ("cancellation/primary_lo", "EventRuntimeEngine._process_event", "PRIMARY_LO_CANCELLATION", 1128, 1195, "lo_budget_overrun and c_amc_sem", ("active_remove", "running_clear", "cancellation_event")),
    ("deadline/no_miss", "EventRuntimeEngine._process_event", "DEADLINE_OBSERVATION_NO_MISS", 1006, 1045, "job_finished", ("deadline_observe_only",)),
    ("deadline/first_hi_miss", "EventRuntimeEngine._process_event", "DEADLINE_OBSERVATION_FIRST_HI_MISS", 1006, 1045, "not job_finished and task_is_HI", ("hi_miss_flag", "deadline_observe_only")),
    ("recovery/idle", "_maybe_recover_to_lo", "IDLE_RECOVERY", 201, 213, "uses_idle_recovery and HI and quiescent", ("mode=LO", "recovery_event")),
    ("controller/no_action", "EventRuntimeEngine._process_event", "CONTROLLER_NO_ACTION", 988, 1001, "event_is_BUDGET_UPDATE and no_updates", ("budget_frame", "event_projection")),
    ("controller/selected_action", "EventRuntimeEngine._process_event", "CONTROLLER_SELECTED_ACTION", 988, 1001, "event_is_BUDGET_UPDATE and updates", ("future_budget_update", "event_projection")),
    ("time/jump_to_next_event", "EventRuntimeEngine.run_until", "JUMP_TO_NEXT_EVENT", 1251, 1276, "ready_empty and next_event_exists", ("next_event_min", "time_jump", "no_service")),
)


def _handler_source(root: Path, qualified: str) -> tuple[str, int, int]:
    module, name = qualified.split(".", 1) if "." in qualified else ("", qualified)
    if qualified.startswith("EventRuntimeEngine."):
        path = root / "amc_py/event_runtime.py"
        class_name, fn_name = qualified.split(".", 1)
    else:
        path = root / "amc_py/event_runtime.py"
        class_name, fn_name = None, qualified
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = None
    for candidate in ast.walk(tree):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) and candidate.name == fn_name:
            if class_name is None or any(isinstance(parent, ast.ClassDef) and parent.name == class_name for parent in []):
                node = candidate
                break
    if node is None and class_name is not None:
        for cls in [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name]:
            node = next((x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == fn_name), None)
    if node is None:
        raise ValueError(f"runtime handler 不存在: {qualified}")
    return ast.unparse(node), int(node.lineno), int(getattr(node, "end_lineno", node.lineno))


def _path_row(root: Path, spec: tuple[Any, ...]) -> dict[str, Any]:
    path_id, handler, case_id, start, end, guard, effects = spec
    source, handler_start, handler_end = _handler_source(root, handler)
    if start < handler_start or end > handler_end:
        raise ValueError(f"path {path_id} 超出 handler 范围")
    source_lines = (root / "amc_py/event_runtime.py").read_text(encoding="utf-8").splitlines()
    path_text = "\n".join(source_lines[start - 1:end])
    evidence = {
        "mode=HI": ("SystemMode.HI",), "mode=LO": ("SystemMode.LO",),
        "mode_switch": ("mode_switches.append",), "same_batch": ("abnormal_arrivals",),
        "arrival_batch": ("pop_all_matching",), "release": ("_process_single_arrival_in_priority_order",),
        "reschedule": ("self._reschedule",), "build_job": ("_build_job",),
        "release_fixed_budget": ("runtime_budget_at_release",),
        "degraded_budget": ("_c_amc_sem_degraded_lo_budget",),
        "actual_cost_clamp": ("actual_cost_override",), "active_add": ("active_jobs.append",),
        "ready_add": ("active_jobs.append",), "highest_priority_select": ("_select_highest_priority_ready_job",),
        "running_update": ("running_job",), "preempt_invalidate": ("_invalidate_job_events",),
        "advance_time": ("_advance_time",), "service_accounting": ("_update_running_progress",),
        "remaining_update": ("executed_time",), "executed_to_actual": ("executed_time = job.actual_cost",),
        "active_remove": ("active_jobs.remove",), "running_clear": ("running_job = None",),
        "hi_complete": ("completed_job_was_hi",), "cancellation_event": ("job_cancellations.append",),
        "deadline_observe_only": ("deadline_misses",), "hi_miss_flag": ("deadline_misses.append",),
        "recovery_event": ("mode_recoveries.append",), "budget_frame": ("budget_state",),
        "event_projection": ("_append_debug_event",), "future_budget_update": ("apply_updates",),
        "next_event_min": ("queue",), "time_jump": ("current_time",), "no_service": ("event.time >= target_time",),
        "boot": ("runtime_budgets",),
    }
    for effect in effects:
        patterns = evidence.get(effect)
        if patterns is None or not any(pattern in path_text for pattern in patterns):
            raise ValueError(f"effect {effect} 无真实源码证据: {path_id}")
    handler_hash = sha256_object({"handler": handler, "source": source})
    path_effect_hash = sha256_object({"handler_hash": handler_hash, "start": start, "end": end,
                                      "guard": guard, "effects": effects})
    return {"path_id": path_id, "handler": handler, "case_id": case_id,
            "source_file": "amc_py/event_runtime.py", "line_start": start, "line_end": end,
            "guard": guard, "guard_hash": sha256_object(guard),
            "effects": list(effects), "path_effect_hash": path_effect_hash,
            "handler_hash": handler_hash, "terminal": f"{handler}:{end}:return_or_closure"}


def build_runtime_branch_map(source_root: str | Path, *, source_hash: str,
                             path_map: Mapping[str, Any] | None = None,
                             case_by_branch: Mapping[str, str] | None = None,
                             schema_ir: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = Path(source_root)
    actual_source_hash = build_source_manifest(root)["semantic_hash"]
    if source_hash != actual_source_hash:
        return {"status": "FAIL", "failure": "SOURCE_HASH_MISMATCH", "expected": actual_source_hash, "provided": source_hash}
    event = bind_event_runtime(root)
    removal = bind_removal_runtime(root)
    if event.get("status") != "PASS" or removal.get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "RUNTIME_AST_BINDING_INCOMPLETE"}
    if path_map is None:
        # 兼容旧 API，但不再把 if-node mapping 当成正式路径证明。
        return {"status": "UNRESOLVED", "failure": "COMPLETE_TRANSITION_PATH_MAP_REQUIRED"}
    if not isinstance(path_map.get("paths"), Mapping):
        return {"status": "UNRESOLVED", "failure": "TRANSITION_PATH_MAP_SCHEMA_INVALID"}
    expected = {spec[0]: spec for spec in PATH_SPECS}
    if set(path_map["paths"]) != set(expected):
        return {"status": "FAIL", "failure": "TRANSITION_PATH_SET_MISMATCH",
                "missing": sorted(set(expected) - set(path_map["paths"])),
                "unknown": sorted(set(path_map["paths"]) - set(expected))}
    rows = []
    for path_id, spec in expected.items():
        actual = _path_row(root, spec)
        supplied = path_map["paths"][path_id]
        for field in ("case_id", "handler", "source_file", "line_start", "line_end", "guard",
                      "guard_hash", "effects", "path_effect_hash", "handler_hash", "terminal"):
            if supplied.get(field) != actual[field]:
                return {"status": "UNRESOLVED", "failure": "TRANSITION_PATH_BINDING_STALE", "path_id": path_id, "field": field}
        rows.append(actual)
    if {row["case_id"] for row in rows} != set(REQUIRED_P0_CASE_IDS):
        return {"status": "UNRESOLVED", "failure": "P0_CASE_PATH_COVERAGE_INCOMPLETE"}
    return {"status": "PASS", "source_hash": source_hash, "path_count": len(rows),
            "paths": rows, "case_manifest_hash": p0_case_manifest_hash(),
            "path_map_hash": sha256_object(rows)}
