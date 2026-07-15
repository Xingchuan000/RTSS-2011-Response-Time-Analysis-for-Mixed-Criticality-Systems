"""证书 envelope 的独立 schema 与 hash 校验入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.schema_loader import load_schema
from formal_toolchain.core.registry import load_registry


def verify_certificate(certificate: dict[str, Any], *, schema_name: str = "common_certificate.schema.json") -> dict[str, Any]:
    """校验标准证书结构，并返回可写入 bundle 的明确状态。"""
    try:
        schema = load_schema(Path(__file__).parents[1] / "specs/certificates" / schema_name)
        # jsonschema 是正式 verifier 的依赖；导入失败必须明确 unresolved。
        from jsonschema import Draft202012Validator, RefResolver
        schema_path = Path(__file__).parents[1] / "specs/certificates" / schema_name
        resolver = RefResolver(schema_path.as_uri(), schema)
        errors = sorted(Draft202012Validator(schema, resolver=resolver).iter_errors(certificate), key=lambda e: list(e.path))
    except Exception as exc:
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_SCHEMA_CHECK_UNAVAILABLE",
                "route": "PROOF_BUNDLE_INVALID", "detail": str(exc)}}
    if errors:
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_SCHEMA_INVALID",
                "route": "PROOF_BUNDLE_INVALID", "paths": [list(error.path) for error in errors]}}
    if certificate.get("obligation_status") in {"PASS", "FAIL", "UNRESOLVED"} and not certificate.get("certificate_context_hash"):
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_CONTEXT_HASH_MISSING",
                "route": "PROOF_BUNDLE_INVALID"}}
    return {"status": "PASS", "certificate_hash": sha256_object(certificate)}


def verify_registry_certificate(certificate: dict[str, Any], *, registry_path: Path,
                               predecessor_certificates: dict[str, dict[str, Any]] | None = None,
                               context_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """按 obligation registry 选择 schema，禁止调用方绕过 category schema。"""
    entries = {str(entry["id"]): entry for entry in load_registry(Path(registry_path))}
    obligation_id = str(certificate.get("obligation_id", ""))
    entry = entries.get(obligation_id)
    if entry is None:
        return {"status": "FAIL", "failure": {"code": "UNKNOWN_OBLIGATION", "route": "PROOF_BUNDLE_INVALID"}}
    schema_name = str(entry["artifact_schema"]).split("/", 1)[-1]
    result = verify_certificate(certificate, schema_name=schema_name)
    if result.get("status") != "PASS":
        return result
    if predecessor_certificates is None or context_inputs is None:
        return {"status": "FAIL", "failure": {"code": "INDEPENDENT_INPUTS_REQUIRED",
                "route": "PROOF_BUNDLE_INVALID"}}
    expected_predecessors = set(str(item) for item in entry.get("depends_on", []))
    actual_predecessors = set(certificate.get("direct_predecessor_hashes", {}))
    if actual_predecessors != expected_predecessors:
        return {"status": "FAIL", "failure": {"code": "DIRECT_PREDECESSOR_SET_MISMATCH",
                "route": "PROOF_BUNDLE_INVALID", "expected": sorted(expected_predecessors),
                "actual": sorted(actual_predecessors)}}
    for predecessor_id in expected_predecessors:
        predecessor = predecessor_certificates.get(predecessor_id)
        if not isinstance(predecessor, dict) or predecessor.get("obligation_status") != "PASS":
            return {"status": "FAIL", "failure": {"code": "PREDECESSOR_NOT_VERIFIED",
                    "route": "PROOF_BUNDLE_INVALID", "obligation_id": predecessor_id}}
        if sha256_object(predecessor) != certificate["direct_predecessor_hashes"][predecessor_id]:
            return {"status": "FAIL", "failure": {"code": "PREDECESSOR_HASH_MISMATCH",
                    "route": "PROOF_BUNDLE_INVALID", "obligation_id": predecessor_id}}
    if certificate.get("certificate_context_hash") != sha256_object(context_inputs):
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_CONTEXT_RECOMPUTE_FAILED",
                "route": "PROOF_BUNDLE_INVALID"}}
    return result
