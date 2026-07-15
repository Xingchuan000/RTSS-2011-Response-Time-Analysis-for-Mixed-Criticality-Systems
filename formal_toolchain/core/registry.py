"""registry DAG、字段覆盖和 migration 合同检查。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
from pathlib import Path
import json
from typing import Any

REQUIRED_FIELDS = frozenset({"id", "profile", "kind", "activation", "required", "depends_on",
                             "artifact", "artifact_schema", "summary_path", "failure_route",
                             "gates_claims", "status_evidence_rule"})
KNOWN_FAILURE_ROUTES = frozenset({"PROOF_BUNDLE_INVALID", "MODEL_CONFORMANCE_FAILED",
                                  "CONCRETE_TIMING_COUNTEREXAMPLE", "POLICY_CONTRACT_VIOLATION",
                                  "REFERENCE_COUNTEREXAMPLE", "REFERENCE_CERTIFICATE_FAILED",
                                  "UNRESOLVED"})

def load_registry(path: Path) -> list[dict[str, Any]]:
    """只读取磁盘 JSON；loader 不增删、不覆盖任何 registry 字段。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [dict(entry) for entry in data.get("entries", data)]


def validate_registry(entries: list[dict[str, Any]]) -> None:
    schema_path = Path(__file__).parents[1] / "specs/registry_meta_schema.json"
    try:
        import jsonschema
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        for entry in entries:
            jsonschema.Draft202012Validator(schema).validate(entry)
    except Exception as exc:
        raise ValueError(f"registry meta-schema 校验失败: {exc}") from exc
    ids = [entry.get("id") for entry in entries]
    if any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("registry obligation id 必须唯一且为字符串")
    known = set(ids)
    graph = {item: set() for item in known}
    for entry in entries:
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"{entry['id']} 缺少字段: {sorted(missing)}")
        unknown = set(entry["depends_on"]) - known
        if unknown:
            raise ValueError(f"{entry['id']} 依赖未知 obligation: {sorted(unknown)}")
        if entry["failure_route"] not in KNOWN_FAILURE_ROUTES:
            raise ValueError(f"{entry['id']} 使用未知 failure route")
        for field in ("artifact", "artifact_schema"):
            path = PurePosixPath(str(entry[field]).replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{entry['id']} 的 {field} 越界")
        graph[entry["id"]] = set(entry["depends_on"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("obligation registry 存在环")
        if node in visited:
            return
        visiting.add(node)
        for dep in graph[node]:
            visit(dep)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def interface_coverage(entries: list[dict[str, Any]], known_artifacts: set[str] | None = None,
                       *, specs_root: Path | None = None) -> dict[str, list[str]]:
    known_artifacts = known_artifacts or set()
    seen_artifacts = {str(e["artifact"]) for e in entries}
    known_routes = KNOWN_FAILURE_ROUTES
    schema_missing = []
    if specs_root is not None:
        for entry in entries:
            schema = specs_root / str(entry["artifact_schema"])
            if not schema.is_file():
                schema_missing.append(str(entry["id"]))
    artifact_duplicates = sorted(path for path in {e["artifact"] for e in entries}
                                 if sum(e["artifact"] == path for e in entries) > 1)
    summary_duplicates = sorted(path for path in {e["summary_path"] for e in entries}
                                if sum(e["summary_path"] == path for e in entries) > 1)
    active_ids = {str(e["id"]) for e in entries if e["activation"] == "active"}
    dependency_activation = sorted(str(e["id"]) for e in entries
                                   if e["activation"] == "active" and any(dep not in active_ids for dep in e["depends_on"]))
    return {"orphan_obligation": [], "orphan_artifact": sorted(known_artifacts - seen_artifacts),
            "orphan_summary_path": [str(e["id"]) for e in entries if not e["summary_path"]],
            "unknown_failure_route": [str(e["id"]) for e in entries if e["failure_route"] not in known_routes],
            "inactive_without_rule": [str(e["id"]) for e in entries if e["activation"] != "active" and not e["status_evidence_rule"]],
            "claim_gate_without_active_entry": [str(e["id"]) for e in entries if e["gates_claims"] and e["activation"] != "active"],
            "missing_artifact_schema": schema_missing,
            "duplicate_artifact_path": artifact_duplicates,
            "duplicate_summary_path": summary_duplicates,
            "active_dependency_not_active": dependency_activation,
            "semantic_dependency_missing": []}


def write_interface_coverage_report(entries: list[dict[str, Any]], output_path, *,
                                    known_artifacts: set[str] | None = None,
                                    specs_root: Path | None = None) -> dict[str, list[str]]:
    import json
    report = interface_coverage(entries, known_artifacts, specs_root=specs_root)
    output_path = __import__("pathlib").Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def registry_fingerprint(entries: list[dict[str, Any]]) -> str:
    from formal_toolchain.core.hashing import sha256_object
    return sha256_object(sorted(entries, key=lambda item: str(item["id"])))


def check_migration(previous_schema_version: str, current_schema_version: str,
                    migration_id: str | None) -> None:
    """schema 变化必须显式携带 migration id，避免静默破坏旧证书。"""
    if previous_schema_version != current_schema_version and not migration_id:
        raise ValueError("breaking schema change 必须提升版本并提供 migration_id")


def check_registry_migration(previous_registry_hash: str, current_registry_hash: str,
                             migration_manifest: dict[str, Any]) -> None:
    if previous_registry_hash != current_registry_hash and not migration_manifest.get("migration_id"):
        raise ValueError("registry dependency 变化必须更新 migration_manifest")
