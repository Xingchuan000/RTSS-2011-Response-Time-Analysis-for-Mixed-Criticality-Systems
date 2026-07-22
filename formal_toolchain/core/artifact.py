"""计划 4.4 的统一 machine-readable obligation certificate envelope。"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .hashing import sha256_object
from .status import OBLIGATION_STATUSES


CERTIFICATE_SCHEMA_VERSION = "certificate_envelope_v2"


def obligation_certificate(*, obligation_id: str, status: str, context_hash: str,
                           inputs: Mapping[str, Any], witness: Mapping[str, Any],
                           checker_id: str, checker_version: str,
                           direct_predecessor_hashes: Mapping[str, str] | None = None,
                           evidence: list[Mapping[str, Any]] | None = None,
                           failure: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """创建不含机器绝对路径、且 status 与 witness 分离的 certificate。"""
    if status not in OBLIGATION_STATUSES:
        raise ValueError(f"obligation status 无效：{status}")
    if not isinstance(context_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", context_hash):
        raise ValueError("certificate context hash 必须是 64 位 SHA-256")
    predecessors = dict(direct_predecessor_hashes or {})
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in predecessors.values()):
        raise ValueError("direct predecessor hash 必须是 64 位 SHA-256")
    if status == "PASS" and failure is not None:
        raise ValueError("PASS certificate 不得包含 failure")
    if status != "PASS" and failure is None:
        raise ValueError("非 PASS certificate 必须包含 failure")
    artifact = {
        "artifact_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "obligation_id": obligation_id,
        "obligation_status": status,
        "certificate_context_hash": context_hash,
        "direct_predecessor_hashes": predecessors,
        "checker_id": checker_id,
        "checker_version": checker_version,
        "inputs": dict(inputs),
        "witness": dict(witness),
        "evidence": list(evidence or []),
        "failure": dict(failure) if failure is not None else None,
    }
    artifact["artifact_hash"] = sha256_object(artifact)
    return artifact


CERTIFICATE_ENVELOPE_KEYS = (
    "artifact_schema_version", "obligation_id", "obligation_status",
    "certificate_context_hash", "direct_predecessor_hashes", "checker_id",
    "checker_version", "inputs", "witness", "evidence", "failure",
)


def verify_obligation_certificate(artifact: Mapping[str, Any]) -> bool:
    """独立重算 envelope hash；缺字段、hash 不匹配都返回 False。"""
    expected = artifact.get("artifact_hash")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        return False
    if any(key not in artifact for key in CERTIFICATE_ENVELOPE_KEYS):
        return False
    payload = {key: artifact[key] for key in CERTIFICATE_ENVELOPE_KEYS}
    return sha256_object(payload) == expected
