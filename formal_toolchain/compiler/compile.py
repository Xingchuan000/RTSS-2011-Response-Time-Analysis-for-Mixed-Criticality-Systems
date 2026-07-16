"""Phase L candidate compiler。

compiler 只生成可被 verifier 重新检查的 candidate。它不会生成可信 outer
root，也不会把 candidate summary 当作最终 claim。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.compiler.dag_runner import claim_dependency_closure, topological_order
from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.formal_checks import calculate_raw_evidence, proof_safe
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.registry import load_registry


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _evidence_key(obligation_id: str) -> str:
    groups = {
        "TREE_WELLFORMEDNESS": "TREE", "LEAF_GUARD_PARTITION": "TREE",
        "FEATURE_QUANTIZATION": "QUANTIZATION", "ACTION_TRANSITION": "ACTION",
        "MASK_FALLBACK": "MASK", "EXECUTABLE_POLICY_SEMANTICS": "EXECUTABLE",
        "CANDIDATE_ENVELOPE": "CANDIDATE", "COMMON_TRANSITION_PRESERVATION": "COMMON",
        "DEPLOYED_POLICY_PRESERVATION": "DEPLOYED", "BUDGET_DOMAIN": "DOMAIN",
        "LO_BUDGET_UPPER_INVARIANT": "DEPLOYED", "HI_BUDGET_LOWER_INVARIANT": "DEPLOYED",
        "ACTIVE_RELEASE_BUDGET_INVARIANT": "DEPLOYED", "SELECTED_ACTION_REGIONS": "EXECUTABLE",
    }
    return groups.get(obligation_id, "PREFLIGHT")


def compile_request(request_path: Path, out_dir: Path) -> dict[str, Any]:
    """执行 candidate DAG 并写出完整 candidate bundle。"""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(__file__).parents[1] / "specs/obligation_registry.json"
    registry = load_registry(registry_path)
    active = sorted(claim_dependency_closure(registry, "DEPLOYED_HI_SAFETY"))
    try:
        computed = calculate_raw_evidence(request_path, source_root=Path.cwd(), include_reference=False)
        base_error: dict[str, Any] | None = None
    except Exception as exc:
        computed = None
        base_error = {"route": "MODEL_CONFORMANCE_FAILED", "code": "CANDIDATE_INPUT_REPLAY_FAILED",
                      "message": str(exc)}

    if computed is None:
        context_hash = sha256_object({"request": json.loads(Path(request_path).read_text(encoding="utf-8"))})
        evidence: dict[str, Any] = {}
    else:
        context_hash = str(computed["context_hash"])
        evidence = computed["evidence"]

    by_id = {str(entry["id"]): entry for entry in registry}
    built: dict[str, dict[str, Any]] = {}
    for obligation_id in topological_order(registry):
        if obligation_id not in active:
            continue
        entry = by_id[obligation_id]
        predecessor_ids = [str(item) for item in entry.get("depends_on", []) if str(item) in active]
        predecessor_statuses = [built[item]["obligation_status"] for item in predecessor_ids]
        if base_error is not None:
            status = "FAIL" if obligation_id in {"PROOF_REQUEST", "ARTIFACT_MANIFEST"} else "UNRESOLVED"
            witness: dict[str, Any] = {"replay_error": base_error}
            failure = base_error
        elif any(item == "FAIL" for item in predecessor_statuses):
            status = "UNRESOLVED"; witness = {"not_run": "前驱 FAIL"}; failure = {"route": "UNRESOLVED", "code": "PREDECESSOR_FAILED"}
        elif obligation_id in {"PROTECTED_HI_RTA_ARITHMETIC", "PER_HI_TASK_INDUCTIVE_WCRT",
                                "PROTECTED_HI_SAFETY_COROLLARY", "RELEASE_FIXED_REMOVAL_MAPPING",
                                "CLOSED_PREFIX_REFINEMENT", "REFERENCE_PREFIX_EXTENSION",
                                "HI_BAD_CLOSED_PREFIX_REFLECTION", "REFERENCE_TASKSET",
                                "CODE_REFERENCE_UPPER_BOUND_MAPPING", "LO_MODE_RTA",
                                "WORST_CASE_START_TIME", "CASE1_INTEGER_DOMAIN",
                                "CASE2_INTEGER_DOMAIN", "ZERO_RELATIVE_START",
                                "INHERITED_HI_DOMINATION", "RELEASE_COUNT",
                                "DEMAND_DOMINATION", "CERTIFIED_ENVELOPE"}:
            # compiler 阶段不持有 fresh-process certified envelope；这些节点必须
            # 留在 candidate/UNRESOLVED，不能以预写算术结果越过 verifier 边界。
            status = "UNRESOLVED"; witness = {"not_run": "fresh verifier required"}; failure = {"route": "UNRESOLVED", "code": "FRESH_VERIFIER_REQUIRED"}
        else:
            key = _evidence_key(obligation_id)
            raw = evidence.get(key, evidence.get("PREFLIGHT", {"status": "PASS"}))
            raw_status = raw.get("status", raw.get("obligation_status", "PASS")) if isinstance(raw, Mapping) else "PASS"
            status = raw_status if raw_status in {"PASS", "FAIL", "UNRESOLVED"} else "UNRESOLVED"
            witness = {"evidence_key": key, "evidence": proof_safe(raw)}
            failure = None if status == "PASS" else {"route": raw.get("route", entry.get("failure_route", "UNRESOLVED")) if isinstance(raw, Mapping) else "UNRESOLVED",
                                                        "code": "CANDIDATE_EVIDENCE_FAILED"}
        built[obligation_id] = obligation_certificate(
            obligation_id=obligation_id, status=status, context_hash=context_hash,
            inputs={"profile": "P0", "primary_claim": "DEPLOYED_HI_SAFETY", "candidate": True},
            witness=witness, checker_id="formal_toolchain.compiler.compile",
            checker_version="phase-l-v1",
            direct_predecessor_hashes={item: built[item]["artifact_hash"] for item in predecessor_ids},
            evidence=[{"candidate": True, "fresh_verifier_required": status == "UNRESOLVED"}],
            failure=failure,
        )

    for obligation_id, certificate in built.items():
        _write(out_dir / "artifacts" / f"{obligation_id}.json", certificate)
    if computed is not None:
        _write(out_dir / "candidate_inputs.json", {
            "context_hash": computed["context_hash"],
            "tree_files": computed["inventory"]["files"],
            "semantic_input_hash": sha256_object(computed["context_body"]),
        })
    summary = {
        "schema_version": "proof_summary_v1",
        "workflow_status": "CANDIDATE",
        "result_status": "CANDIDATE",
        "profile": "P0",
        "primary_claim": "DEPLOYED_HI_SAFETY",
        "certificate_context_hash": context_hash,
        "active_obligation_ids": active,
        "obligation_statuses": {key: value["obligation_status"] for key, value in built.items()},
        "real_seed_evaluation": "DEFERRED" if computed and computed["request"].get("fixture_id", "synthetic_p0") == "synthetic_p0" else "NOT_EVALUATED",
    }
    _write(out_dir / "artifact_manifest.json", {"schema_version": "candidate_artifact_manifest_v1",
                                                 "artifacts": {key: value["artifact_hash"] for key, value in built.items()}})
    _write(out_dir / "proof_summary.json", summary)
    return summary
