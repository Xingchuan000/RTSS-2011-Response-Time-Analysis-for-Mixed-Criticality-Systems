"""分层 context 的确定性计算。

每层只包含本层及其显式上游输入，禁止把下游结论塞回 context，避免证书
通过自引用或编译器结果污染 verifier 输入。
"""

from __future__ import annotations

from typing import Any, Mapping
import re

from .hashing import sha256_object

CONTEXT_FIELDS = {
    "bootstrap_context": ("bootstrap",),
    "implementation_context": ("bootstrap_context", "implementation"),
    "semantic_context": ("implementation_context", "semantic"),
    "policy_context": ("semantic_context", "policy"),
    "invariant_context": ("policy_context", "candidate_envelope"),
    "reference_context": ("invariant_context", "certified_envelope"),
    "bridge_context": ("reference_context", "bridge"),
    "composition_context": ("bridge_context", "composition_inputs"),
    "bundle_context": ("composition_context", "bundle_inputs"),
}

# 该表是 obligation 与证明上下文的唯一静态绑定。它故意不提供“未知义务
# 默认使用 semantic_context”的分支：新增义务若没有明确归属，必须在
# registry/context 评审中补齐，否则 verifier 只能返回 UNRESOLVED。
OBLIGATION_CONTEXT_LAYERS: dict[str, str] = {
    # bootstrap
    "REGISTRY_META_SCHEMA": "bootstrap_context", "P0_PROFILE_SCHEMA": "bootstrap_context",
    "THEORY_MANIFEST": "bootstrap_context", "THEORY_LIBRARY_VERSION": "bootstrap_context",
    "ASSURANCE_POLICY": "bootstrap_context", "OBLIGATION_REGISTRY": "bootstrap_context",
    "CLAIM_AGGREGATION": "bootstrap_context", "CONTEXT_SCHEMA": "bootstrap_context",
    "CANONICAL_SERIALIZATION": "bootstrap_context", "INTERFACE_COVERAGE": "bootstrap_context",
    "MIGRATION_MANIFEST": "bootstrap_context", "PROOF_REQUEST": "bootstrap_context",
    # implementation
    "SOURCE_TREE_INTEGRITY": "implementation_context", "RUNTIME_ENVIRONMENT": "implementation_context",
    "DEPENDENCY_LOCK": "implementation_context", "CHECKER_VERSION": "implementation_context",
    "IMMUTABLE_INPUT_HASH": "implementation_context",
    # semantic
    "EFFECTIVE_RUNTIME_CONFIG": "semantic_context", "SCHEDULER_MODEL": "semantic_context",
    "STRICT_PRIORITY_ORDER": "semantic_context", "TIME_DOMAIN": "semantic_context",
    "NO_OVERFLOW": "semantic_context", "OVERHEAD_PROFILE": "semantic_context",
    "INITIAL_QUIESCENCE": "semantic_context", "BOOT_INITIALIZATION": "semantic_context",
    "MODE_SEMANTICS_CONFORMANCE": "semantic_context", "DEMAND_ORACLE_BATCH_CONTRACT": "semantic_context",
    "HI_EXECUTION_CONTRACT": "semantic_context", "REMOVAL_COMPLETENESS": "semantic_context",
    "HI_NONTRUNCATION": "semantic_context", "DEADLINE_OBSERVATION": "semantic_context",
    "EFFECTIVE_EVENT_ORDER": "semantic_context", "SEQUENCE_ALLOCATION": "semantic_context",
    "PHASE_DAG": "semantic_context", "BATCH_CLOSURE": "semantic_context",
    "DEADLINE_BOUNDARY_ORDER": "semantic_context", "CONTROLLER_INVISIBILITY": "semantic_context",
    "CONTROLLER_POSTCLOSURE": "semantic_context", "TIME_PROGRESS": "semantic_context",
    "WINDOW_MODE_NORMALIZATION": "semantic_context", "OBSERVATION_EXTRACTION": "semantic_context",
    "FEATURE_TOTALITY": "semantic_context",
    # policy
    "FEATURE_SCHEMA_CONSISTENCY": "policy_context", "FEATURE_QUANTIZATION": "policy_context",
    "TREE_WELLFORMEDNESS": "policy_context", "LEAF_GUARD_PARTITION": "policy_context",
    "ACTION_TRANSITION": "policy_context", "MASK_FALLBACK": "policy_context",
    "SELECTED_ACTION_REGIONS": "policy_context", "EXECUTABLE_POLICY_SEMANTICS": "policy_context",
    # invariant
    "CANDIDATE_ENVELOPE": "invariant_context", "BUDGET_DOMAIN": "invariant_context",
    "LO_BUDGET_UPPER_INVARIANT": "invariant_context", "HI_BUDGET_LOWER_INVARIANT": "invariant_context",
    "ACTIVE_RELEASE_BUDGET_INVARIANT": "invariant_context",
    "COMMON_TRANSITION_PRESERVATION": "invariant_context", "DEPLOYED_POLICY_PRESERVATION": "invariant_context",
    "CERTIFIED_ENVELOPE": "invariant_context",
    # reference
    "CODE_REFERENCE_UPPER_BOUND_MAPPING": "reference_context", "REFERENCE_TASKSET": "reference_context",
    "ALL_TASK_REFERENCE_RTA_ARITHMETIC": "reference_context",
    "BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION": "reference_context",
    "REFERENCE_SEMANTICS_CONTRACT": "reference_context", "REFERENCE_TRANSITION_SYSTEM_IDENTITY": "reference_context",
    "REFERENCE_MODEL_CONFORMANCE": "reference_context",
    "REFERENCE_TASKSET_SCHEDULABLE": "reference_context",
    "REFERENCE_HI_SUBSET_SAFETY": "reference_context",
    "DISCRETE_TICK_EMBEDDING": "reference_context", "RELEASE_COUNT": "reference_context",
    "DEMAND_DOMINATION": "reference_context", "LO_MODE_RTA": "reference_context",
    "WORST_CASE_START_TIME": "reference_context", "CASE1_INTEGER_DOMAIN": "reference_context",
    "CASE2_INTEGER_DOMAIN": "reference_context", "ZERO_RELATIVE_START": "reference_context",
    "INHERITED_HI_DOMINATION": "reference_context", "PROTECTED_HI_RTA_ARITHMETIC": "reference_context",
    "PER_HI_TASK_INDUCTIVE_WCRT": "reference_context", "PROTECTED_HI_SAFETY_COROLLARY": "reference_context",
    "FINITE_BAD_PREFIX_CONTRADICTION": "composition_context",
    "FINAL_CLAIM_COMPOSITION": "composition_context",
    # bridge
    "RELEASE_FIXED_REMOVAL_MAPPING": "bridge_context", "CLOSED_PREFIX_REFINEMENT": "bridge_context",
    "REFERENCE_PREFIX_EXTENSION": "bridge_context", "HI_BAD_CLOSED_PREFIX_REFLECTION": "bridge_context",
    "EARLY_STOP_CONFIGURATION_GATE": "bridge_context",
    "EFFECTIVE_EVENT_FRONTIER_RELATION": "bridge_context",
    "TOKEN_REFRESH_PROJECTION": "bridge_context",
    "CONTROLLER_WRITE_SET": "semantic_context",
    "CONTROLLER_BOUNDARY": "semantic_context",
    "CONTROLLER_PATH_UNIQUENESS": "semantic_context",
    "UPDATE_PAYLOAD_TOTALITY": "semantic_context",
    # bundle/root mirrors
    "ARTIFACT_MANIFEST": "bundle_context", "COMPONENT_CONTEXT_INTEGRITY": "bundle_context",
    "DIRECT_PREDECESSOR_HASHES": "bundle_context", "STATUS_EVIDENCE": "bundle_context",
    "OUTER_BUNDLE_ROOT": "bundle_context", "INDEPENDENT_BUNDLE_VERIFICATION": "bundle_context",
    "CLAIM_AGGREGATION_RESULT": "bundle_context",
}


def context_layer_for_obligation(obligation_id: str) -> str:
    """返回 obligation 的明确 context 层；未知 ID 直接报错而非猜测。"""

    try:
        return OBLIGATION_CONTEXT_LAYERS[obligation_id]
    except KeyError as exc:
        raise KeyError(f"obligation 未声明 context layer: {obligation_id}") from exc


def expected_context_for_obligation(obligation_id: str,
                                    contexts: Mapping[str, Mapping[str, Any]]) -> str:
    """取出 verifier 对该义务重新计算的 context hash。"""

    layer = context_layer_for_obligation(obligation_id)
    context = contexts.get(layer)
    if not isinstance(context, Mapping) or not isinstance(context.get("hash"), str):
        raise ValueError(f"context layer 不完整: {layer}")
    return str(context["hash"])


def validate_context_contract(inputs: dict[str, Any]) -> None:
    """验证 candidate/certified 边界，防止 certified 直接覆盖 candidate。

    该检查单独提供给严格 bundle verifier；基础 context 构造仍保留对早期
    Phase B 调用方的兼容性。
    """
    candidate = inputs.get("candidate_envelope")
    certified = inputs.get("certified_envelope")
    if not isinstance(candidate, dict) or not isinstance(certified, dict):
        raise ValueError("candidate_envelope 和 certified_envelope 必须是 object")
    required = {"bootstrap", "implementation", "semantic", "policy", "candidate_envelope",
                "certified_envelope", "bridge", "bundle_inputs"}
    has_composition = "composition_inputs" in inputs
    missing = sorted(required - set(inputs))
    if missing:
        raise ValueError(f"context 输入缺失: {missing}")
    if "preservation_certificate_hash" not in certified:
        raise ValueError("certified_envelope 必须引用 preservation certificate")
    if certified.get("candidate_envelope_hash") != sha256_object(candidate):
        raise ValueError("certified_envelope 未引用当前 candidate_envelope")
    if not re.fullmatch(r"[0-9a-f]{64}", str(certified["preservation_certificate_hash"])):
        raise ValueError("preservation_certificate_hash 必须是 SHA-256")
    preservation = certified.get("preservation_certificate")
    if not isinstance(preservation, dict) or preservation.get("obligation_status") != "PASS":
        raise ValueError("certified_envelope 必须携带已验证的 preservation certificate")
    if sha256_object(preservation) != certified["preservation_certificate_hash"]:
        raise ValueError("preservation certificate hash 不匹配")
    forbidden = {"summary", "outer_root", "hash_map", "claim_result", "downstream_certificates"}
    if isinstance(inputs["bundle_inputs"], dict) and forbidden & set(inputs["bundle_inputs"]):
        raise ValueError("bundle_inputs 不得包含下游 bundle 对象")


def finalize_context(schema_version: str, preimage: Mapping[str, Any]) -> dict[str, Any]:
    """对单层 context 使用 ``body → hash`` 两步算法。

    hash 不会被放进自身 preimage；这使得 verifier 可以在不信任 candidate
    的前提下，从同一组输入重算每一层。
    """

    body = {"schema_version": schema_version, "preimage": dict(preimage)}
    return {**body, "hash": sha256_object(body)}


def build_bootstrap_context(**inputs: Any) -> dict[str, Any]:
    return finalize_context("bootstrap_context_v1", inputs)


def build_implementation_context(*, bootstrap_context_hash: str, **inputs: Any) -> dict[str, Any]:
    return finalize_context("implementation_context_v1",
                            {"bootstrap_context_hash": bootstrap_context_hash, **inputs})


def build_semantic_context(*, implementation_context_hash: str, **inputs: Any) -> dict[str, Any]:
    return finalize_context("semantic_context_v1",
                            {"implementation_context_hash": implementation_context_hash, **inputs})


def build_policy_context(*, semantic_context_hash: str, **inputs: Any) -> dict[str, Any]:
    return finalize_context("policy_context_v1",
                            {"semantic_context_hash": semantic_context_hash, **inputs})


def build_invariant_context(*, policy_context_hash: str, **inputs: Any) -> dict[str, Any]:
    return finalize_context("invariant_context_v1",
                            {"policy_context_hash": policy_context_hash, **inputs})


def build_reference_context_layer(*, invariant_context_hash: str, **inputs: Any) -> dict[str, Any]:
    return finalize_context("reference_context_v1",
                            {"invariant_context_hash": invariant_context_hash, **inputs})


def build_bridge_context(*, reference_context_hash: str, **inputs: Any) -> dict[str, Any]:
    return finalize_context("bridge_context_v1",
                            {"reference_context_hash": reference_context_hash, **inputs})


def build_composition_context(*, bridge_context_hash: str, **inputs: Any) -> dict[str, Any]:
    return finalize_context("composition_context_v1",
                            {"bridge_context_hash": bridge_context_hash, **inputs})


def build_bundle_context(*, composition_context_hash: str, **inputs: Any) -> dict[str, Any]:
    return finalize_context("bundle_context_v2",
                            {"composition_context_hash": composition_context_hash, **inputs})


def build_contexts(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "candidate_envelope" not in inputs:
        raise ValueError("invariant_context 必须明确提供 candidate_envelope")
    if "certified_envelope" not in inputs:
        raise ValueError("reference_context 必须明确提供 certified_envelope")
    validate_context_contract(inputs)
    contexts: dict[str, dict[str, Any]] = {}
    for name, fields in CONTEXT_FIELDS.items():
        value: dict[str, Any] = {"schema_version": f"{name}_v1", "preimage": {}}
        for field in fields:
            if field.endswith("_context"):
                value["preimage"][field] = contexts[field]["hash"]
            elif field in inputs:
                value["preimage"][field] = inputs[field]
        value["hash"] = sha256_object(value)
        contexts[name] = value
    return contexts


def context_mutation_scope(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> set[str]:
    """返回 context hash 发生变化的层，供 B05 mutation matrix 使用。"""
    return {name for name in before if before[name].get("hash") != after.get(name, {}).get("hash")}


def require_verified_predecessor(
    *,
    predecessor: Mapping[str, Any],
    predecessor_entry: Mapping[str, Any],
    contexts: Mapping[str, Mapping[str, Any]],
) -> None:
    """跨层前驱验证：前驱的 certificate_context_hash 必须等于其 context layer 的 hash。"""
    from formal_toolchain.core.artifact import verify_obligation_certificate
    expected_hash = contexts[predecessor_entry["context_layer"]]["hash"]
    if predecessor.get("certificate_context_hash") != expected_hash:
        raise ValueError(
            f"PREDECESSOR_CONTEXT_LAYER_MISMATCH: "
            f"expected {expected_hash}, got {predecessor.get('certificate_context_hash')}"
        )
    if not verify_obligation_certificate(predecessor):
        raise ValueError("PREDECESSOR_ARTIFACT_INVALID")
    if predecessor.get("obligation_status") != "PASS":
        raise ValueError("PREDECESSOR_NOT_PASS")


def build_reference_context(*, semantic_context_hash: str,
                            certified_envelope_hash: str,
                            code_taskset_fingerprint: str,
                            priority_order: list[str] | tuple[str, ...],
                            xf: str,
                            effective_runtime_config_hash: str,
                            reference_taskset_fingerprint: str,
                            mapping_schema_version: str = "reference_mapping_v1") -> dict[str, Any]:
    """按 Phase I 冻结的 preimage 计算唯一 reference context。"""
    preimage = {
        "schema_version": "reference_context_v1",
        "mapping_schema_version": mapping_schema_version,
        "semantic_context_hash": semantic_context_hash,
        "certified_envelope_hash": certified_envelope_hash,
        "code_taskset_fingerprint": code_taskset_fingerprint,
        "priority_order": list(priority_order),
        "xf": xf,
        "effective_runtime_config_hash": effective_runtime_config_hash,
        "reference_taskset_fingerprint": reference_taskset_fingerprint,
    }
    preimage["hash"] = sha256_object(preimage)
    return preimage
