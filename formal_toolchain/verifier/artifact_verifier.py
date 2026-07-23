"""证书 envelope 的独立 schema 与 hash 校验入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.schema_loader import load_schema
from formal_toolchain.core.registry import load_registry


_CERTIFICATE_SCHEMA_BASE_URI = "https://formal-toolchain/specs/certificates/"


def _schema_retrieval_uri(path: Path) -> str:
    """Return the canonical retrieval URI used for relative certificate refs."""
    return f"{_CERTIFICATE_SCHEMA_BASE_URI}{path.name}"


def _schema_with_retrieval_id(path: Path) -> dict[str, Any]:
    """Load one schema and give legacy no-$id schemas a stable base URI.

    Several certificate schemas intentionally use relative ``$ref`` values but
    predate the explicit ``$id`` convention.  Registering those documents under
    ``file://`` while validating a root document with no base URI makes the
    relative references unresolvable.  Supplying the canonical in-repository
    retrieval URI preserves the schema content while giving every relative ref
    an unambiguous base.
    """
    schema = json.loads(path.read_text(encoding="utf-8"))
    if "$id" not in schema:
        schema = {"$id": _schema_retrieval_uri(path), **schema}
    return schema


def _build_schema_registry(schema_root: Path):
    """Build a referencing.Registry from all certificate schema files."""
    from referencing import Registry, Resource

    registry = Registry()
    for path in sorted(schema_root.glob("*.schema.json")):
        schema = _schema_with_retrieval_id(path)
        declared_uri = str(schema["$id"])
        resource = Resource.from_contents(schema)
        registry = registry.with_resource(declared_uri, resource)
        canonical_uri = _schema_retrieval_uri(path)
        if canonical_uri != declared_uri:
            registry = registry.with_resource(canonical_uri, resource)
    return registry


def verify_certificate(certificate: dict[str, Any], *, schema_name: str = "common_certificate.schema.json") -> dict[str, Any]:
    """校验标准证书结构，并返回可写入 bundle 的明确状态。"""
    try:
        schema_root = Path(__file__).parents[1] / "specs/certificates"
        schema_path = schema_root / schema_name
        # Keep load_schema's existence/JSON validation, then attach a stable
        # retrieval ID for legacy schemas whose relative refs otherwise have no
        # base URI.
        load_schema(schema_path)
        schema = _schema_with_retrieval_id(schema_path)
        from jsonschema import Draft202012Validator
        ref_registry = _build_schema_registry(schema_root)
        validator = Draft202012Validator(schema, registry=ref_registry)
        errors = sorted(validator.iter_errors(certificate), key=lambda e: list(e.path))
    except Exception as exc:
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_SCHEMA_CHECK_UNAVAILABLE",
                "route": "PROOF_BUNDLE_INVALID", "detail": str(exc)}}
    if errors:
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_SCHEMA_INVALID",
                "route": "PROOF_BUNDLE_INVALID", "paths": [list(error.path) for error in errors]}}
    if certificate.get("obligation_status") in {"PASS", "FAIL", "UNRESOLVED"} and not certificate.get("certificate_context_hash"):
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_CONTEXT_HASH_MISSING",
                "route": "PROOF_BUNDLE_INVALID"}}
    from formal_toolchain.core.artifact import CERTIFICATE_ENVELOPE_KEYS
    payload = {key: certificate[key] for key in CERTIFICATE_ENVELOPE_KEYS if key != "artifact_hash"}
    expected = certificate.get("artifact_hash")
    if expected and sha256_object(payload) != expected:
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_HASH_MISMATCH",
                "route": "PROOF_BUNDLE_INVALID"}}
    return {"status": "PASS", "certificate_hash": expected}


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
        if certificate["direct_predecessor_hashes"][predecessor_id] != predecessor.get("artifact_hash"):
            return {"status": "FAIL", "failure": {"code": "PREDECESSOR_HASH_MISMATCH",
                    "route": "PROOF_BUNDLE_INVALID", "obligation_id": predecessor_id}}
    if certificate.get("certificate_context_hash") != sha256_object(context_inputs):
        return {"status": "FAIL", "failure": {"code": "CERTIFICATE_CONTEXT_RECOMPUTE_FAILED",
                "route": "PROOF_BUNDLE_INVALID"}}
    return result
