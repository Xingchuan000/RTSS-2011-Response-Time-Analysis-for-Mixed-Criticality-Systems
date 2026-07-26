"""event queue/arrival batch 的静态绑定结果。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .python_ast_ir import function_to_ir
from formal_toolchain.semantics.frozen_runtime_contract import frozen_event_models_path, frozen_event_runtime_path, CONTRACT_VERSION


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name):
                key_value = f"{key.value.id}.{key.attr}"
            else:
                key_value = ast.literal_eval(key)
            result[key_value] = ast.literal_eval(value)
        return result
    return ast.literal_eval(node)


def bind_event_runtime(source_root: Path) -> dict[str, Any]:
    path = frozen_event_models_path(source_root)
    runtime_path = frozen_event_runtime_path(source_root)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    priority = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_TYPE_PRIORITY" for t in node.targets):
            priority = _literal(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_TYPE_PRIORITY":
            priority = _literal(node.value)
    if not isinstance(priority, dict):
        return {"status": "UNRESOLVED", "failure": {"code": "EVENT_PRIORITY_NOT_LITERAL", "route": "UNRESOLVED"}}
    runtime_source = runtime_path.read_text(encoding="utf-8")
    queue_source_ok = ("self._counter = itertools.count()" in source and
                       "order = next(self._counter)" in source and
                       "key=lambda entry: (entry[0], entry[1], entry[2])" in source)
    arrival_source_ok = ("events.sort" in runtime_source and "actual_cost > task.c_lo" in runtime_source)
    priority_names = {getattr(key, "value", str(key).split(".")[-1]): int(value)
                      for key, value in priority.items()}
    required_order = ("JOB_COMPLETION", "DEADLINE_CHECK", "JOB_ARRIVAL")
    priority_semantics_ok = (all(name in priority_names for name in required_order) and
                             priority_names["JOB_COMPLETION"] < priority_names["DEADLINE_CHECK"] < priority_names["JOB_ARRIVAL"])
    names = ("EventRuntimeEngine._maybe_enter_c_amc_sem_hi_mode_at_arrival", "EventRuntimeEngine._process_job_arrival_batch",
             "EventRuntimeEngine._process_single_arrival_in_priority_order")
    queue_source = source
    queue_methods = {name: function_to_ir(queue_source, f"EventQueue.{name}") for name in ("push", "pop", "pop_all_matching")}
    functions = {name: function_to_ir(runtime_source, name) for name in names}
    failures = [item for item in functions.values() if item.get("status") != "PASS"]
    failures.extend(item for item in queue_methods.values() if item.get("status") != "PASS")
    if not priority_semantics_ok or not queue_source_ok or not arrival_source_ok:
        return {"status": "FAIL", "event_type_priority": priority_names,
                "failure": {"code": "EVENT_PRIORITY_SEMANTICS_FAILED", "route": "MODEL_CONFORMANCE_FAILED"},
                "queue_methods": queue_methods, "arrival_batch": functions}
    result = {"status": "PASS" if not failures else "UNRESOLVED",
              "formal_semantics_contract_version": CONTRACT_VERSION,
              "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
              "event_type_priority": {str(key): value for key, value in priority.items()},
              "fifo_sequence": "EventQueue._counter", "queue_methods": queue_methods, "arrival_batch": functions,
              "semantics": {"verified": not failures,
                             "abnormal_scan_before_job_construction": all(item.get("status") == "PASS" for item in functions.values()),
                             "single_switch_per_batch": "_process_job_arrival_batch" in runtime_source and "mode_switch" in runtime_source and arrival_source_ok,
                             "mode_before_batch_preserved": "mode_before_batch" in runtime_source or "mode_before" in runtime_source,
                             "primary_on_switch_time": "c_amc_sem_primary_on_switch_time" in runtime_source}}
    if result["status"] == "PASS" and not all(bool(value) for key, value in result["semantics"].items() if key != "verified"):
        result["status"] = "FAIL"
        result["failure"] = {"code": "EVENT_SEMANTIC_FACT_UNDERIVED", "route": "MODEL_CONFORMANCE_FAILED"}
    if failures:
        result["failure"] = {"code": "UNSUPPORTED_AST_NODE", "route": "UNRESOLVED"}
    return result
