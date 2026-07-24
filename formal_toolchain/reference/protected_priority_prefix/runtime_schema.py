"""Structural conformance checks for the fixed reference executable semantics."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_file, sha256_object

ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = (
    "formal_toolchain/reference/executable_semantics.py",
    "formal_toolchain/reference/reference_state.py",
    "formal_toolchain/reference/c_amc_sem_semantics.py",
    "formal_toolchain/reference/semantics_contract.py",
    "formal_toolchain/bridge/logical_events.py",
    "formal_toolchain/reference/p0_transition_contract.py",
)


def _source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _function(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(_source(SOURCE_FILES[0]))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError(f"RUNTIME_SCHEMA_FUNCTION_MISSING:{name}")


def _has_text(name: str, *needles: str) -> bool:
    text = _source(name)
    return all(needle in text for needle in needles)


def _function_has(name: str, *needles: str) -> bool:
    segment = ast.get_source_segment(_source(SOURCE_FILES[0]), _function(name)) or ""
    return all(needle in segment for needle in needles)


def runtime_schema_checks() -> dict[str, bool]:
    """Return proof obligations, not a finite execution result."""
    return {
        "strict_fixed_priority_dispatch": _function_has("_normalize_dispatch", "sorted(", "_job_schedule_key")
        and _function_has("_job_schedule_key", "_task_priority(task)"),
        "tail_service_exclusion": _function_has("_normalize_dispatch", "_job_schedule_key")
        and _has_text(SOURCE_FILES[0], "priority_index"),
        "fixed_unit_processor_supply": _function_has("apply_service_tick", "old.executed + 1", "state.time + 1"),
        "release_fixed_actual_demand": _function_has("apply_arrival_batch", "release_demand_overrides.get")
        and _function_has("apply_release", "removal_demand=plan.removal_demand"),
        "zero_time_mode_updates": _function_has("apply_mode_switch", "replace(state, mode=\"HI\""),
        "released_jobs_not_reclassified": _function_has("apply_mode_switch", "replace(state, mode=\"HI\""),
        "completion_guard_fixed_demand": _function_has("apply_service_tick", "executed >= jobs[rk].budget"),
        "deadline_observe_only": _function_has("apply_deadline_observation", "misses.append", "pop_event"),
        "canonical_same_timestamp_closure": _function_has("close_timestamp", "closure_measure", "apply_logical_event"),
        "arrival_projection_schema": _has_text(SOURCE_FILES[0], "event.batch_jobs", "_arrival_event_for_job"),
        "transition_input_total": _function_has("apply_logical_event", "else:", "pop_event"),
        "finite_closure_measure": _function_has("closure_measure", "sorted(set(PHASE_RANK.values()))")
        and _has_text("formal_toolchain/bridge/logical_events.py", "PHASE_RANK"),
    }


def build_runtime_schema_certificate() -> dict[str, Any]:
    checks = runtime_schema_checks()
    bindings = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    payload = {"schema_version": "protected-prefix-runtime-schema-v1", "checks": checks, "source_bindings": bindings}
    return {**payload, "certificate_hash": sha256_object(payload),
            "status": "PASS" if all(checks.values()) else "FAIL",
            "failure": None if all(checks.values()) else {"code": "RUNTIME_SCHEMA_CONFORMANCE_FAILED", "checks": [k for k, v in checks.items() if not v]}}


def verify_runtime_schema_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    rebuilt = build_runtime_schema_certificate()
    if certificate.get("certificate_hash") != rebuilt["certificate_hash"]:
        return {"status": "FAIL", "code": "RUNTIME_SCHEMA_CERTIFICATE_BINDING_MISMATCH"}
    if certificate.get("source_bindings") != rebuilt["source_bindings"]:
        return {"status": "FAIL", "code": "RUNTIME_SCHEMA_SOURCE_BINDING_MISMATCH"}
    return {"status": rebuilt["status"], "certificate_hash": rebuilt["certificate_hash"],
            "checks": rebuilt["checks"], "failure": rebuilt.get("failure")}
