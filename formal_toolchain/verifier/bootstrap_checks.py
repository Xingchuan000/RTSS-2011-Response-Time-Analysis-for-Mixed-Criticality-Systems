"""bootstrap、migration、source binding 和 interface coverage checker。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.registry import interface_coverage, registry_fingerprint, validate_registry
from formal_toolchain.verifier.registry_graph import verifier_topological_order


def _result(status: str, *, route: str | None = None, code: str | None = None,
            witness: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"status": status, "route": route, "code": code,
            "witness": dict(witness or {})}


def _pass(**witness: Any) -> dict[str, Any]:
    return _result("PASS", witness=witness)


def _fail(code: str, *, route: str = "PROOF_BUNDLE_INVALID", **witness: Any) -> dict[str, Any]:
    return _result("FAIL", route=route, code=code, witness=witness)


def verify_obligation_registry(*, registry: list[dict[str, Any]]) -> dict[str, Any]:
    """现场验证 registry schema、DAG 和 verifier 自己的拓扑序。"""

    try:
        validate_registry(registry)
        order = verifier_topological_order(registry)
    except (TypeError, ValueError) as exc:
        return _fail("OBLIGATION_REGISTRY_INVALID", message=str(exc))
    return _pass(registry_fingerprint=registry_fingerprint(registry), topological_order=order)


def verify_migration_manifest(*, migration: Mapping[str, Any],
                              registry: list[dict[str, Any]],
                              current_schema_version: str) -> dict[str, Any]:
    """用当前 registry 现场重算 fingerprint，拒绝手工复制旧 hash。"""

    current = registry_fingerprint(registry)
    if migration.get("registry_fingerprint") != current:
        return _fail("MIGRATION_MANIFEST_MISMATCH",
                     expected_registry_fingerprint=current,
                     manifest_registry_fingerprint=migration.get("registry_fingerprint"))
    if migration.get("registry_schema_version") != current_schema_version:
        return _fail("MIGRATION_SCHEMA_VERSION_MISMATCH",
                     expected_schema_version=current_schema_version,
                     manifest_schema_version=migration.get("registry_schema_version"))
    if not isinstance(migration.get("migration_id"), str) or not migration.get("migration_id"):
        return _fail("MIGRATION_ID_MISSING")
    return _pass(registry_fingerprint=current,
                 registry_schema_version=current_schema_version,
                 migration_id=migration["migration_id"])


def build_interface_coverage_report(*, registry: list[dict[str, Any]],
                                    spec_root: Path,
                                    checker_catalog: Mapping[str, Any],
                                    structural_ids: set[str]) -> dict[str, Any]:
    """现场计算 active obligation、schema、checker、DAG 和路径覆盖。"""

    raw = interface_coverage(registry, specs_root=spec_root)
    active = {str(item["id"]) for item in registry
              if item.get("activation") == "active" and item.get("required") is True}
    missing_checker = sorted(active - set(checker_catalog) - set(structural_ids))
    orphan_checker = sorted(set(checker_catalog) - {str(item["id"]) for item in registry})
    errors = {key: value for key, value in raw.items() if value}
    if missing_checker:
        errors["missing_verifier_checker"] = missing_checker
    if orphan_checker:
        errors["orphan_checker"] = orphan_checker
    status = "PASS" if not errors else "FAIL"
    return {"status": status, "active_obligations": sorted(active),
            "errors": errors, "registry_fingerprint": registry_fingerprint(registry)}


def verify_json_schema_file(path: Path, *, schema_version: str | None = None) -> dict[str, Any]:
    """只验证 bootstrap 文件存在、可解析和声明版本；不读取下游 summary。"""

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail("BOOTSTRAP_FILE_INVALID", path=str(path), message=str(exc))
    if schema_version is not None and data.get("schema_version") != schema_version:
        return _fail("BOOTSTRAP_SCHEMA_VERSION_MISMATCH", path=str(path),
                     expected=schema_version, actual=data.get("schema_version"))
    return _pass(path=str(path), object_hash=sha256_object(data))


def verify_source_manifest(*, manifest: Mapping[str, Any], source_root: Path) -> dict[str, Any]:
    """重新计算 source manifest，确保源码 binding 不是手工修改的摘要。"""

    from formal_toolchain.adapters.source_manifest import build_source_manifest

    actual = build_source_manifest(Path(source_root))
    if manifest.get("semantic_hash") != actual.get("semantic_hash"):
        return _fail("SOURCE_MANIFEST_MISMATCH", route="MODEL_CONFORMANCE_FAILED",
                     expected=actual.get("semantic_hash"), actual=manifest.get("semantic_hash"))
    return _pass(source_manifest_hash=actual.get("semantic_hash"), bound_files=actual.get("files", {}))
