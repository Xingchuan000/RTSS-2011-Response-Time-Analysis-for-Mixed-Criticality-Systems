"""registry DAG、字段覆盖、ClaimClosure 和 migration 合同检查。"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from pathlib import PurePosixPath
from pathlib import Path
import json
from typing import Any

from formal_toolchain.core.obligation_ids import (
    ALL_TASK_REFERENCE_RTA_ARITHMETIC,
    BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION,
    FINITE_BAD_PREFIX_CONTRADICTION,
    FINAL_CLAIM_COMPOSITION,
    HI_BAD_CLOSED_PREFIX_REFLECTION,
    REFERENCE_HI_SUBSET_SAFETY,
    REFERENCE_MODEL_CONFORMANCE,
    REFERENCE_PREFIX_EXTENSION,
    REFERENCE_TASKSET_SCHEDULABLE,
)

REQUIRED_FIELDS = frozenset({"id", "profile", "kind", "activation", "required", "depends_on",
                             "artifact", "artifact_schema", "summary_path", "failure_route",
                             "gates_claims", "status_evidence_rule",
                             "proof_role", "context_layer"})
KNOWN_FAILURE_ROUTES = frozenset({"PROOF_BUNDLE_INVALID", "MODEL_CONFORMANCE_FAILED",
                                  "CONCRETE_TIMING_COUNTEREXAMPLE", "POLICY_CONTRACT_VIOLATION",
                                  "REFERENCE_COUNTEREXAMPLE", "REFERENCE_CERTIFICATE_FAILED",
                                  "UNRESOLVED"})

REGISTRY_HASH_FIELDS = (
    "id", "profile", "kind", "activation", "required", "depends_on",
    "artifact", "artifact_schema", "summary_path", "failure_route",
    "gates_claims", "status_evidence_rule",
    "proof_role", "context_layer", "producer", "authorization_for",
)

STRUCTURAL_GATE_IDS = frozenset({
    "ARTIFACT_MANIFEST", "COMPONENT_CONTEXT_INTEGRITY", "DIRECT_PREDECESSOR_HASHES",
    "STATUS_EVIDENCE", "OUTER_BUNDLE_ROOT", "INDEPENDENT_BUNDLE_VERIFICATION",
    "CLAIM_AGGREGATION_RESULT",
})


@dataclass(frozen=True)
class ClaimClosure:
    mathematical: frozenset[str]
    authorization: frozenset[str]
    structural: frozenset[str]

    @property
    def candidate_artifacts(self) -> frozenset[str]:
        return self.mathematical | self.authorization

    @property
    def verified_artifacts(self) -> frozenset[str]:
        return self.candidate_artifacts | self.structural


def build_claim_closure(entries: list[dict[str, Any]], claim: str) -> ClaimClosure:
    """从 Registry 精确推导 claim 的闭包分层。
    
    compiler 与 verifier 必须调用同一个实现，确保 candidate/verified closure 一致。
    """
    by_id = {str(e["id"]): e for e in entries}
    math_roots = {str(e["id"]) for e in entries
                  if e["proof_role"] == "mathematical_root" and claim in e.get("gates_claims", [])}
    if not math_roots:
        raise ValueError(f"NO_MATHEMATICAL_ROOT_FOR_CLAIM:{claim}")
    if len(math_roots) > 1:
        raise ValueError(f"MULTIPLE_MATHEMATICAL_ROOTS:{sorted(math_roots)}")

    def dependency_closure(roots: set[str]) -> set[str]:
        result: set[str] = set()
        def visit(oid: str) -> None:
            if oid in result:
                return
            entry = by_id.get(oid)
            if entry is None or entry.get("activation") != "active":
                return
            result.add(oid)
            for dep in entry.get("depends_on", []):
                visit(str(dep))
        for r in roots:
            visit(r)
        return result

    mathematical = dependency_closure(math_roots)
    authorization = {
        str(e["id"]) for e in entries
        if e["proof_role"] == "authorization_gate"
        and e.get("activation") == "active"
        and claim in e.get("authorization_for", [])
    }
    structural = {str(e["id"]) for e in entries
                  if e["proof_role"] == "structural_gate"} | STRUCTURAL_GATE_IDS
    return ClaimClosure(
        mathematical=frozenset(mathematical),
        authorization=frozenset(authorization),
        structural=frozenset(structural),
    )


def exact_math_roots(entries: list[dict[str, Any]], claim: str) -> set[str]:
    """返回 claim 的精确数学根集合。"""
    roots = {str(e["id"]) for e in entries
             if e["proof_role"] == "mathematical_root" and claim in e.get("gates_claims", [])}
    if not roots:
        raise ValueError(f"NO_MATHEMATICAL_ROOT_FOR_CLAIM:{claim}")
    return roots


def load_registry(path: Path) -> list[dict[str, Any]]:
    """只读取磁盘 JSON；loader 不增删、不覆盖任何 registry 字段。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [dict(entry) for entry in data.get("entries", data)]
    validate_registry(entries)
    return entries


def active_obligations_for_claim(entries: list[dict[str, Any]], *, claim: str,
                                  phase_ids: set[str] | None = None) -> list[str]:
    """由 Registry 计算 claim 的 active closure，不接受调用方自行删减。"""
    validate_registry(entries)
    by_id = {str(entry["id"]): entry for entry in entries}
    roots = {item for item, entry in by_id.items()
             if entry.get("activation") == "active" and claim in entry.get("gates_claims", [])}
    if phase_ids is not None:
        roots &= phase_ids
    result: set[str] = set()
    def visit(obligation_id: str) -> None:
        if obligation_id in result:
            return
        result.add(obligation_id)
        for predecessor in by_id[obligation_id].get("depends_on", []):
            if predecessor in by_id and by_id[predecessor].get("activation") == "active":
                visit(str(predecessor))
    for root in sorted(roots):
        visit(root)
    return sorted(result)


def phase_ijk_obligation_closure(entries: list[dict[str, Any]]) -> list[str]:
    """计算 Phase I-K 的 canonical registry 闭包。

    正式入口只能消费 registry 中的依赖关系；本地 bridge witness 不得再
    自己维护另一套“必要前置列表”。
    """
    validate_registry(entries)
    roots = {
        BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION,
        ALL_TASK_REFERENCE_RTA_ARITHMETIC,
        REFERENCE_MODEL_CONFORMANCE,
        REFERENCE_TASKSET_SCHEDULABLE,
        REFERENCE_HI_SUBSET_SAFETY,
        "CLOSED_PREFIX_REFINEMENT",
        REFERENCE_PREFIX_EXTENSION,
        HI_BAD_CLOSED_PREFIX_REFLECTION,
        FINITE_BAD_PREFIX_CONTRADICTION,
        FINAL_CLAIM_COMPOSITION,
    }
    by_id = {str(entry["id"]): entry for entry in entries}
    if not roots <= set(by_id):
        raise ValueError(f"Phase I-K canonical obligation 缺失: {sorted(roots - set(by_id))}")
    result: set[str] = set()
    def visit(obligation_id: str) -> None:
        if obligation_id in result:
            return
        entry = by_id[obligation_id]
        if entry.get("activation") != "active":
            raise ValueError(f"Phase I-K 依赖不是 active: {obligation_id}")
        result.add(obligation_id)
        for dep in entry.get("depends_on", []):
            visit(str(dep))
    for root in sorted(roots):
        visit(root)
    return sorted(result)


def verify_registry_local_closure(entries: list[dict[str, Any]], certificates: dict[str, Any], *,
                                  context_hash: str) -> dict[str, Any]:
    """校验 Phase I-K 闭包中的证书、ID、上下文和 direct predecessors。"""
    try:
        closure = phase_ijk_obligation_closure(entries)
    except ValueError as exc:
        return {"status": "UNRESOLVED", "failure": str(exc)}
    by_id = {str(entry["id"]): entry for entry in entries}
    missing = [item for item in closure if item not in certificates]
    if missing:
        return {"status": "UNRESOLVED", "failure": "REGISTRY_CLOSURE_CERTIFICATE_MISSING", "missing": missing,
                "closure": closure}
    from .artifact import verify_obligation_certificate
    bad = []
    for obligation_id in closure:
        cert = certificates[obligation_id]
        if not isinstance(cert, dict) or cert.get("obligation_id") != obligation_id:
            bad.append((obligation_id, "ID")); continue
        if cert.get("obligation_status") not in {"PASS", "FAIL", "UNRESOLVED"} or cert.get("certificate_context_hash") != context_hash:
            bad.append((obligation_id, "STATUS_OR_CONTEXT")); continue
        if not verify_obligation_certificate(cert):
            bad.append((obligation_id, "HASH")); continue
        expected = set(by_id[obligation_id].get("depends_on", [])) & set(closure)
        actual = set(cert.get("direct_predecessor_hashes", {}))
        if expected != actual:
            bad.append((obligation_id, "DIRECT_PREDECESSOR"))
    return {"status": "PASS" if not bad else "UNRESOLVED", "closure": closure, "bad": bad}


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
        if entry["proof_role"] == "authorization_gate" and not entry.get("authorization_for"):
            raise ValueError(f"{entry['id']} 是 authorization_gate 但 authorization_for 为空")
        for field in ("artifact", "artifact_schema"):
            path = PurePosixPath(str(entry[field]).replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{entry['id']} 的 {field} 越界")
        if entry["proof_role"] not in frozenset({"mathematical_lemma", "mathematical_root",
                                                  "authorization_gate", "structural_gate",
                                                  "diagnostic"}):
            raise ValueError(f"{entry['id']} 使用未知 proof_role: {entry['proof_role']}")
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


def topological_order(entries: list[dict[str, Any]]) -> list[str]:
    """Return the registry's dependency-first order, rejecting every cycle."""
    validate_registry(entries)
    by_id = {str(entry["id"]): entry for entry in entries}
    state: dict[str, int] = {}
    result: list[str] = []

    def visit(node: str) -> None:
        if state.get(node) == 1:
            raise ValueError("obligation registry 存在环")
        if state.get(node) == 2:
            return
        state[node] = 1
        for dependency in by_id[node].get("depends_on", []):
            visit(str(dependency))
        state[node] = 2
        result.append(node)

    for node in sorted(by_id):
        visit(node)
    return result


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
    producer_missing = sorted(
        row["id"] for row in entries
        if row.get("activation") == "active" and "producer" not in row
    )
    artifact_path_not_canonical = sorted(
        row["id"] for row in entries
        if row.get("activation") == "active"
        and row.get("artifact") != f"artifacts/{row['id'].lower()}.json"
    )
    return {"orphan_obligation": [], "orphan_artifact": sorted(known_artifacts - seen_artifacts),
            "orphan_summary_path": [str(e["id"]) for e in entries if not e["summary_path"]],
            "unknown_failure_route": [str(e["id"]) for e in entries if e["failure_route"] not in known_routes],
            "inactive_without_rule": [str(e["id"]) for e in entries if e["activation"] != "active" and not e["status_evidence_rule"]],
            "claim_gate_without_active_entry": [str(e["id"]) for e in entries if e["gates_claims"] and e["activation"] != "active"],
            "missing_artifact_schema": schema_missing,
            "duplicate_artifact_path": artifact_duplicates,
            "duplicate_summary_path": summary_duplicates,
            "active_dependency_not_active": dependency_activation,
            "semantic_dependency_missing": [],
            "producer_missing": producer_missing,
            "artifact_path_not_canonical": artifact_path_not_canonical}


def write_interface_coverage_report(entries: list[dict[str, Any]], output_path, *,
                                    known_artifacts: set[str] | None = None,
                                    specs_root: Path | None = None) -> dict[str, list[str]]:
    import json
    report = interface_coverage(entries, known_artifacts, specs_root=specs_root)
    output_path = __import__("pathlib").Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def artifact_path_for(entry: dict[str, Any], bundle_root: Path) -> Path:
    """从 Registry entry 的 artifact 字段计算规范路径。"""
    relative = Path(str(entry["artifact"]))
    path = (bundle_root / relative).resolve()
    if bundle_root.resolve() not in path.parents and path != bundle_root.resolve():
        raise ValueError(f"REGISTRY_ARTIFACT_PATH_ESCAPE: {entry['id']} -> {path}")
    return path


def registry_fingerprint(entries: list[dict[str, Any]]) -> str:
    from formal_toolchain.core.hashing import sha256_object
    stable = [
        {field: entry.get(field) for field in REGISTRY_HASH_FIELDS}
        for entry in entries
    ]
    return sha256_object(sorted(stable, key=lambda row: str(row.get("id", ""))))


def check_migration(previous_schema_version: str, current_schema_version: str,
                    migration_id: str | None) -> None:
    """schema 变化必须显式携带 migration id，避免静默破坏旧证书。"""
    if previous_schema_version != current_schema_version and not migration_id:
        raise ValueError("breaking schema change 必须提升版本并提供 migration_id")


def check_registry_migration(previous_registry_hash: str, current_registry_hash: str,
                             migration_manifest: dict[str, Any]) -> None:
    if previous_registry_hash != current_registry_hash and not migration_manifest.get("migration_id"):
        raise ValueError("registry dependency 变化必须更新 migration_manifest")
