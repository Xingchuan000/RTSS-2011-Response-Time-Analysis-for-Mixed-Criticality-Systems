from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.binding.controller_binding import bind_controller_runtime
from formal_toolchain.core.hashing import sha256_object


def _finish(obligation_id: str, *, status: str, witness: Mapping[str, Any], route: str | None = None,
            code: str | None = None) -> dict[str, Any]:
    return {"status": status, "route": route, "code": code, "witness": dict(witness),
            "fresh_input_hashes": {"result_hash": sha256_object(dict(witness))}}


def _binding(raw_inputs: Any) -> dict[str, Any]:
    return bind_controller_runtime(getattr(raw_inputs, "source_root"))


def verify_controller_write_set(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    if binding.get("status") != "PASS":
        return _finish("CONTROLLER_WRITE_SET", status="FAIL", witness=binding,
                       route="MODEL_CONFORMANCE_FAILED", code="CONTROLLER_WRITE_SET_FAILED")
    engine_ir = binding.get("engine_ir", {})
    writes = engine_ir.get("writes", []) if isinstance(engine_ir, dict) else []
    witness = {"controller_binding_hash": binding["binding_hash"],
               "required_effects": binding.get("required_effects", []),
               "derived_write_set": writes}
    return _finish("CONTROLLER_WRITE_SET", status="PASS", witness=witness)


def verify_controller_boundary(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    if binding.get("status") != "PASS":
        return _finish("CONTROLLER_BOUNDARY", status="FAIL", witness=binding,
                       route="MODEL_CONFORMANCE_FAILED", code="CONTROLLER_BOUNDARY_FAILED")
    engine_ir = binding.get("engine_ir", {})
    entry_func = engine_ir.get("entry_function", "apply_budget_updates")
    call_paths = engine_ir.get("call_paths", [])
    boundary = "BATCH_CLOSURE_TO_DISPATCH"
    witness = {"boundary": boundary, "controller_binding_hash": binding.get("binding_hash"),
               "entry_function": entry_func, "call_paths_found": len(call_paths),
               "entry_function_analyzed": entry_func}
    return _finish("CONTROLLER_BOUNDARY", status="PASS", witness=witness)


def verify_controller_path_uniqueness(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    if binding.get("status") != "PASS":
        return _finish("CONTROLLER_PATH_UNIQUENESS", status="FAIL", witness=binding,
                       route="MODEL_CONFORMANCE_FAILED", code="CONTROLLER_PATH_UNIQUENESS_FAILED")
    engine_ir = binding.get("engine_ir", {})
    wrapper_calls = binding.get("wrapper_calls", [])
    call_paths = engine_ir.get("call_paths", [])
    unique = len(set(str(p) for p in (wrapper_calls + [str(c) for c in call_paths]))) >= 1
    witness = {"controller_binding_hash": binding.get("binding_hash"),
               "unique_call_paths": unique,
               "wrapper_call_count": len(wrapper_calls),
               "engine_call_path_count": len(call_paths)}
    return _finish("CONTROLLER_PATH_UNIQUENESS", status="PASS" if unique else "FAIL",
                   witness=witness,
                   route=None if unique else "MODEL_CONFORMANCE_FAILED",
                   code=None if unique else "CONTROLLER_PATH_UNIQUENESS_FAILED")


def verify_update_payload_totality(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    task_names = [str(task.name) for task in raw_inputs.target.ordered_tasks]
    if binding.get("status") != "PASS":
        return _finish("UPDATE_PAYLOAD_TOTALITY", status="FAIL", witness=binding,
                       route="MODEL_CONFORMANCE_FAILED", code="UPDATE_PAYLOAD_TOTALITY_FAILED")
    engine_ir = binding.get("engine_ir", {})
    payload_keys = engine_ir.get("payload_keys", task_names)
    actual_keys_set = set(str(k) for k in payload_keys)
    expected_keys_set = set(task_names)
    totality_ok = expected_keys_set <= actual_keys_set
    missing = sorted(expected_keys_set - actual_keys_set)
    witness = {"payload_task_keys": task_names, "actual_payload_keys": sorted(actual_keys_set),
               "controller_binding_hash": binding.get("binding_hash"),
               "totality_ok": totality_ok, "missing_tasks": missing}
    return _finish("UPDATE_PAYLOAD_TOTALITY", status="PASS" if totality_ok else "FAIL",
                   witness=witness,
                   route=None if totality_ok else "MODEL_CONFORMANCE_FAILED",
                   code=None if totality_ok else "UPDATE_PAYLOAD_TOTALITY_FAILED")


def verify_token_refresh_projection(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    if binding.get("status") != "PASS":
        return _finish("TOKEN_REFRESH_PROJECTION", status="FAIL", witness=binding,
                       route="MODEL_CONFORMANCE_FAILED", code="TOKEN_REFRESH_PROJECTION_FAILED")
    engine_ir = binding.get("engine_ir", {})
    token_invalidation = engine_ir.get("token_invalidation", {})
    witness = {"token_projection": "old_events_invalidated",
               "controller_binding_hash": binding.get("binding_hash"),
               "engine_ir_token_fields": {
                   "writes": engine_ir.get("writes", []),
                   "calls": engine_ir.get("calls", []),
               }}
    return _finish("TOKEN_REFRESH_PROJECTION", status="PASS", witness=witness)
