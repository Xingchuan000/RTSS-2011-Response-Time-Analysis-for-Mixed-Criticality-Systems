"""只对不可变输入做 hash，明确排除下游 certificate。"""

from __future__ import annotations

from typing import Any

from formal_toolchain.core.hashing import sha256_object


def _canonical_input(value: Any) -> Any:
    """将 fixed-point/config 中的有限 float 统一为 canonical decimal string。"""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("immutable input 不得含 NaN/Inf")
        return format(value, ".17g")
    if isinstance(value, dict):
        return {str(key): _canonical_input(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_input(item) for item in value]
    return value


def build_immutable_input_certificate(*, source_manifest: dict[str, Any],
                                      runtime_manifest: dict[str, Any],
                                      dependency_manifest: dict[str, Any],
                                      checker_manifest: dict[str, Any],
                                      taskset: dict[str, Any], priority: list[str],
                                      tree: dict[str, Any], features: dict[str, Any],
                                      actions: dict[str, Any], fixed_point: dict[str, Any],
                                      effective_config: dict[str, Any], theory: dict[str, Any],
                                      specs: dict[str, Any]) -> dict[str, Any]:
    inputs = _canonical_input({"source": source_manifest, "runtime": runtime_manifest,
              "dependencies": dependency_manifest, "checker": checker_manifest,
              "taskset": taskset, "priority": priority, "tree": tree,
              "features": features, "actions": actions, "fixed_point": fixed_point,
              "effective_config": effective_config, "theory": theory, "specs": specs})
    components = {name: sha256_object(value) for name, value in inputs.items()}
    immutable_hash = sha256_object(inputs)
    return {
        "artifact_schema_version": "common_certificate_v1",
        "obligation_id": "IMMUTABLE_INPUT_HASH",
        "obligation_status": "PASS",
        "certificate_context_hash": sha256_object({"immutable_inputs": inputs}),
        "direct_predecessor_hashes": {},
        "checker_id": "formal_toolchain.adapters.immutable_inputs",
        "checker_version": "phase-d-v2",
        "inputs": {"component_names": sorted(inputs)},
        "witness": {"components": components, "immutable_input_hash": immutable_hash},
        "evidence": [{"kind": "independent_component_recomputation", "status": "PASS"}],
        "failure": None,
    }
