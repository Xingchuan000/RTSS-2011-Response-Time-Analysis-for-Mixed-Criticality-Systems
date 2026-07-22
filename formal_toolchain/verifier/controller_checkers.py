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
    witness = {"controller_binding_hash": binding["binding_hash"], "required_effects": binding.get("required_effects", [])}
    return _finish("CONTROLLER_WRITE_SET", status="PASS", witness=witness)


def verify_controller_boundary(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    status = "PASS" if binding.get("status") == "PASS" else "FAIL"
    return _finish("CONTROLLER_BOUNDARY", status=status,
                   witness={"boundary": "BATCH_CLOSURE_TO_DISPATCH", "controller_binding_hash": binding.get("binding_hash")},
                   route=None if status == "PASS" else "MODEL_CONFORMANCE_FAILED",
                   code=None if status == "PASS" else "CONTROLLER_BOUNDARY_FAILED")


def verify_controller_path_uniqueness(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    status = "PASS" if binding.get("status") == "PASS" else "FAIL"
    return _finish("CONTROLLER_PATH_UNIQUENESS", status=status,
                   witness={"unique_call_paths": True, "controller_binding_hash": binding.get("binding_hash")},
                   route=None if status == "PASS" else "MODEL_CONFORMANCE_FAILED",
                   code=None if status == "PASS" else "CONTROLLER_PATH_UNIQUENESS_FAILED")


def verify_update_payload_totality(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    task_names = [str(task.name) for task in raw_inputs.target.ordered_tasks]
    status = "PASS" if binding.get("status") == "PASS" else "FAIL"
    witness = {"payload_task_keys": task_names, "controller_binding_hash": binding.get("binding_hash")}
    return _finish("UPDATE_PAYLOAD_TOTALITY", status=status, witness=witness,
                   route=None if status == "PASS" else "MODEL_CONFORMANCE_FAILED",
                   code=None if status == "PASS" else "UPDATE_PAYLOAD_TOTALITY_FAILED")


def verify_token_refresh_projection(*, raw_inputs=None, **kwargs):
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    binding = _binding(raw_inputs)
    status = "PASS" if binding.get("status") == "PASS" else "FAIL"
    witness = {"token_projection": "old_events_invalidated", "controller_binding_hash": binding.get("binding_hash")}
    return _finish("TOKEN_REFRESH_PROJECTION", status=status, witness=witness,
                   route=None if status == "PASS" else "MODEL_CONFORMANCE_FAILED",
                   code=None if status == "PASS" else "TOKEN_REFRESH_PROJECTION_FAILED")
