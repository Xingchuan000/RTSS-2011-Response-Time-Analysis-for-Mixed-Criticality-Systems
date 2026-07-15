"""独立重算 immutable-input root 的 verifier。

该模块刻意不调用 ``build_immutable_input_certificate``，避免 generator 和
verifier 共享同一段可能存在缺陷的计算代码。
"""

from __future__ import annotations

import math
from typing import Any

from formal_toolchain.core.hashing import sha256_object


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("immutable input 不得含 NaN/Inf")
        return format(value, ".17g")
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def recompute_immutable_input_root(inputs: dict[str, Any]) -> dict[str, Any]:
    """从原始 component inputs 独立计算 component hashes 和 root。"""
    canonical = _canonical(inputs)
    components = {name: sha256_object(value) for name, value in canonical.items()}
    return {"components": components, "immutable_input_hash": sha256_object(canonical),
            "certificate_context_hash": sha256_object({"immutable_inputs": canonical})}


def verify_immutable_input_certificate(certificate: dict[str, Any], *, inputs: dict[str, Any]) -> dict[str, Any]:
    """比较证书与 verifier 独立重算的所有输入根；不接受证书自报 root。"""
    try:
        expected = recompute_immutable_input_root(inputs)
        witness = certificate.get("witness")
        if not isinstance(witness, dict):
            raise ValueError("immutable certificate 缺少 witness")
        if witness.get("components") != expected["components"]:
            raise ValueError("component hash 不匹配")
        if witness.get("immutable_input_hash") != expected["immutable_input_hash"]:
            raise ValueError("immutable input root 不匹配")
        if certificate.get("certificate_context_hash") != expected["certificate_context_hash"]:
            raise ValueError("immutable context hash 不匹配")
    except (TypeError, ValueError) as exc:
        return {"status": "FAIL", "failure": {"code": "IMMUTABLE_INPUT_RECOMPUTE_FAILED",
                "route": "PROOF_BUNDLE_INVALID", "detail": str(exc)}}
    return {"status": "PASS", "immutable_input_hash": expected["immutable_input_hash"]}
