"""分层 context 的确定性计算。

每层只包含本层及其显式上游输入，禁止把下游结论塞回 context，避免证书
通过自引用或编译器结果污染 verifier 输入。
"""

from __future__ import annotations

from typing import Any
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
    "bundle_context": ("bridge_context", "bundle_inputs"),
}


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


def build_contexts(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "candidate_envelope" not in inputs:
        raise ValueError("invariant_context 必须明确提供 candidate_envelope")
    if "certified_envelope" not in inputs:
        raise ValueError("reference_context 必须明确提供 certified_envelope")
    validate_context_contract(inputs)
    contexts: dict[str, dict[str, Any]] = {}
    for name, fields in CONTEXT_FIELDS.items():
        value: dict[str, Any] = {}
        for field in fields:
            if field.endswith("_context"):
                value[field] = contexts[field]["hash"]
            elif field in inputs:
                value[field] = inputs[field]
        value["hash"] = sha256_object(value)
        contexts[name] = value
    return contexts


def context_mutation_scope(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> set[str]:
    """返回 context hash 发生变化的层，供 B05 mutation matrix 使用。"""
    return {name for name in before if before[name].get("hash") != after.get(name, {}).get("hash")}


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
