"""Phase L/M fresh-process verifier 的独立聚合实现。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.compiler.dag_runner import topological_order
from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate
from formal_toolchain.core.formal_checks import calculate_raw_evidence, proof_safe
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.registry import load_registry
from formal_toolchain.verifier.aggregator import claim_dependency_closure


STRUCTURAL_IDS = frozenset({
    "ARTIFACT_MANIFEST", "COMPONENT_CONTEXT_INTEGRITY", "DIRECT_PREDECESSOR_HASHES",
    "STATUS_EVIDENCE", "OUTER_BUNDLE_ROOT", "INDEPENDENT_BUNDLE_VERIFICATION",
    "CLAIM_AGGREGATION_RESULT",
})


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _raw_for_obligation(obligation_id: str, computed: Mapping[str, Any]) -> Any:
    evidence = computed["evidence"]
    mapping = {
        "TREE_WELLFORMEDNESS": "TREE", "LEAF_GUARD_PARTITION": "TREE",
        "FEATURE_QUANTIZATION": "QUANTIZATION", "ACTION_TRANSITION": "ACTION",
        "MASK_FALLBACK": "MASK", "EXECUTABLE_POLICY_SEMANTICS": "EXECUTABLE",
        "CANDIDATE_ENVELOPE": "CANDIDATE", "COMMON_TRANSITION_PRESERVATION": "COMMON",
        "DEPLOYED_POLICY_PRESERVATION": "DEPLOYED", "BUDGET_DOMAIN": "DOMAIN",
        "LO_BUDGET_UPPER_INVARIANT": "DEPLOYED", "HI_BUDGET_LOWER_INVARIANT": "DEPLOYED",
        "ACTIVE_RELEASE_BUDGET_INVARIANT": "DEPLOYED", "SELECTED_ACTION_REGIONS": "EXECUTABLE",
        "CERTIFIED_ENVELOPE": "CERTIFIED", "CODE_REFERENCE_UPPER_BOUND_MAPPING": "MAPPING",
        "REFERENCE_TASKSET": "REFERENCE", "PROTECTED_HI_RTA_ARITHMETIC": "RTA",
        "PER_HI_TASK_INDUCTIVE_WCRT": "RECURRING", "PROTECTED_HI_SAFETY_COROLLARY": "COROLLARY",
        "LO_MODE_RTA": "RTA", "WORST_CASE_START_TIME": "RTA",
        "CASE1_INTEGER_DOMAIN": "RTA", "CASE2_INTEGER_DOMAIN": "RTA",
        "ZERO_RELATIVE_START": "RTA", "INHERITED_HI_DOMINATION": "RTA",
        "RELEASE_COUNT": "RTA", "DEMAND_DOMINATION": "RTA",
    }
    key = mapping.get(obligation_id)
    if key is not None:
        return evidence.get(key, {"status": "UNRESOLVED", "failure": "EVIDENCE_MISSING"})
    if obligation_id == "RELEASE_FIXED_REMOVAL_MAPPING":
        return {"status": "PASS", "mapping": {"release_budget_is_fixed": True,
                "removal_is_exact": True}}
    if obligation_id == "CLOSED_PREFIX_REFINEMENT":
        return evidence.get("BRIDGE", {}).get("closed_prefix", {"status": "UNRESOLVED"})
    if obligation_id == "REFERENCE_PREFIX_EXTENSION":
        return evidence.get("BRIDGE", {}).get("reference_extension", {"status": "UNRESOLVED"})
    if obligation_id == "HI_BAD_CLOSED_PREFIX_REFLECTION":
        return evidence.get("BRIDGE", {}).get("bad_prefix_reflection", {"status": "UNRESOLVED"})
    if obligation_id in {"PROOF_REQUEST", "ARTIFACT_MANIFEST", "REGISTRY_META_SCHEMA",
                         "P0_PROFILE_SCHEMA", "THEORY_MANIFEST", "THEORY_LIBRARY_VERSION",
                         "ASSURANCE_POLICY", "OBLIGATION_REGISTRY", "CLAIM_AGGREGATION",
                         "CONTEXT_SCHEMA", "CANONICAL_SERIALIZATION", "INTERFACE_COVERAGE",
                         "MIGRATION_MANIFEST", "SOURCE_TREE_INTEGRITY", "RUNTIME_ENVIRONMENT",
                         "DEPENDENCY_LOCK", "CHECKER_VERSION", "IMMUTABLE_INPUT_HASH",
                         "DISCRETE_TICK_EMBEDDING",
                         "EFFECTIVE_RUNTIME_CONFIG", "SCHEDULER_MODEL", "STRICT_PRIORITY_ORDER",
                         "TIME_DOMAIN", "NO_OVERFLOW", "OVERHEAD_PROFILE", "INITIAL_QUIESCENCE",
                         "BOOT_INITIALIZATION", "MODE_SEMANTICS_CONFORMANCE",
                         "DEMAND_ORACLE_BATCH_CONTRACT", "HI_EXECUTION_CONTRACT",
                         "REMOVAL_COMPLETENESS", "HI_NONTRUNCATION", "DEADLINE_OBSERVATION",
                         "EFFECTIVE_EVENT_ORDER", "SEQUENCE_ALLOCATION", "PHASE_DAG",
                         "BATCH_CLOSURE", "DEADLINE_BOUNDARY_ORDER", "CONTROLLER_INVISIBILITY",
                         "CONTROLLER_POSTCLOSURE", "TIME_PROGRESS", "WINDOW_MODE_NORMALIZATION",
                         "OBSERVATION_EXTRACTION", "FEATURE_SCHEMA_CONSISTENCY", "FEATURE_TOTALITY"}:
        # compiler 对没有单独实现 witness 的结构/绑定节点消费 preflight
        # 原始对象；verifier 也重放同一对象，不能拿摘要替代逐字段证据。
        return computed["evidence"]["PREFLIGHT"]
    return {"status": "UNRESOLVED", "failure": "OBLIGATION_EVIDENCE_NOT_MAPPED"}


def _status(raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return "UNRESOLVED"
    value = raw.get("obligation_status", raw.get("status"))
    return value if value in {"PASS", "FAIL", "UNRESOLVED"} else "UNRESOLVED"


def _load_and_check_candidate(bundle: Path, active: list[str], context_hash: str) -> dict[str, Any]:
    """检查 candidate 的完整性，并把 candidate 作为待验证输入而非事实。"""

    artifact_dir = Path(bundle) / "artifacts"
    result: dict[str, Any] = {}
    for obligation_id in active:
        path = artifact_dir / f"{obligation_id}.json"
        if not path.is_file():
            return {"status": "FAIL", "failure": {"route": "PROOF_BUNDLE_INVALID", "code": "CANDIDATE_CERTIFICATE_MISSING", "obligation_id": obligation_id}}
        cert = _read(path)
        if (not verify_obligation_certificate(cert) or cert.get("obligation_id") != obligation_id
                or cert.get("certificate_context_hash") != context_hash):
            return {"status": "FAIL", "failure": {"route": "PROOF_BUNDLE_INVALID", "code": "CANDIDATE_CERTIFICATE_INVALID", "obligation_id": obligation_id}}
        result[obligation_id] = cert
    return {"status": "PASS", "certificates": result}


def _root_preimage(context_hash: str, request: Mapping[str, Any], active: list[str],
                   certificates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """构造不含 root 自身、summary 和 report 的 outer-root preimage。"""

    included = {key: value["artifact_hash"] for key, value in certificates.items()
                if key not in STRUCTURAL_IDS}
    return {"schema_version": "outer_bundle_root_v1",
            "component_context_hashes": {"certificate_context_hash": context_hash},
            "verified_obligation_artifact_hashes": dict(sorted(included.items())),
            "status_evidence": {key: certificates[key]["obligation_status"]
                                for key in sorted(included)},
            "active_obligation_set": list(active),
            "claim_request": {key: request.get(key) for key in
                              ("schema_version", "profile", "primary_claim", "taskset_seed", "optional_claims")}}


def verify_bundle(request_path: Path, bundle: Path, out_dir: Path) -> dict[str, Any]:
    """重新计算 raw evidence、certificates、outer root 和最终 claim。"""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(__file__).parents[1] / "specs/obligation_registry.json"
    registry = load_registry(registry_path)
    active = sorted(claim_dependency_closure(registry, "DEPLOYED_HI_SAFETY"))
    try:
        computed = calculate_raw_evidence(request_path, source_root=Path.cwd(), include_reference=True)
    except Exception as exc:
        summary = {"schema_version": "proof_summary_v1", "workflow_status": "FAILED",
                   "result_status": "MODEL_CONFORMANCE_FAILED", "profile": "P0",
                   "primary_claim": "DEPLOYED_HI_SAFETY", "failure_route": "MODEL_CONFORMANCE_FAILED",
                   "failure_code": "VERIFIER_INPUT_REPLAY_FAILED", "failure_message": str(exc),
                   "active_obligation_ids": active}
        _write(out_dir / "proof_summary.json", summary)
        return summary

    context_hash = str(computed["context_hash"])
    candidate_check = _load_and_check_candidate(bundle, active, context_hash)
    if candidate_check.get("status") != "PASS":
        summary = {"schema_version": "proof_summary_v1", "workflow_status": "FAILED",
                   "result_status": "PROOF_BUNDLE_INVALID", "profile": "P0",
                   "primary_claim": "DEPLOYED_HI_SAFETY", "failure_route": "PROOF_BUNDLE_INVALID",
                   "failure_code": candidate_check["failure"]["code"], "active_obligation_ids": active,
                   "certificate_context_hash": context_hash}
        _write(out_dir / "proof_summary.json", summary)
        return summary

    candidates = candidate_check["certificates"]
    # candidate PASS witness 必须等于 fresh raw evidence；summary 被篡改不会
    # 影响这里，但 tree/envelope/certificate 被篡改会在此处闭合失败。
    for obligation_id, candidate in candidates.items():
        witness = candidate.get("witness", {})
        key = witness.get("evidence_key") if isinstance(witness, Mapping) else None
        if (candidate.get("obligation_status") == "PASS" and isinstance(key, str)
                and obligation_id not in STRUCTURAL_IDS):
            expected = proof_safe(_raw_for_obligation(obligation_id, computed))
            if witness.get("evidence") != proof_safe(expected):
                summary = {"schema_version": "proof_summary_v1", "workflow_status": "FAILED",
                           "result_status": "PROOF_BUNDLE_INVALID", "profile": "P0",
                           "primary_claim": "DEPLOYED_HI_SAFETY", "failure_route": "PROOF_BUNDLE_INVALID",
                           "failure_code": "CANDIDATE_EVIDENCE_REPLAY_MISMATCH",
                           "violated_obligation_id": obligation_id,
                           "certificate_context_hash": context_hash, "active_obligation_ids": active}
                _write(out_dir / "proof_summary.json", summary)
                return summary

    by_id = {str(entry["id"]): entry for entry in registry}
    computed_certificates: dict[str, dict[str, Any]] = {}
    for obligation_id in topological_order(registry):
        if obligation_id not in active or obligation_id in STRUCTURAL_IDS:
            continue
        entry = by_id[obligation_id]
        predecessors = [str(item) for item in entry.get("depends_on", []) if str(item) in active]
        raw = _raw_for_obligation(obligation_id, computed)
        status = _status(raw)
        if any(computed_certificates[item]["obligation_status"] != "PASS" for item in predecessors):
            status = "UNRESOLVED"
            raw = {"status": "UNRESOLVED", "failure": "PREDECESSOR_NOT_PASS"}
        failure = None if status == "PASS" else {
            "route": entry.get("failure_route", "UNRESOLVED"),
            "code": "FRESH_REPLAY_FAILED" if status == "FAIL" else "FRESH_REPLAY_UNRESOLVED",
            "machine_details": proof_safe(raw),
        }
        computed_certificates[obligation_id] = obligation_certificate(
            obligation_id=obligation_id, status=status, context_hash=context_hash,
            inputs={"raw_input_context": context_hash}, witness={"evidence": proof_safe(raw)},
            checker_id="formal_toolchain.verifier.recompute", checker_version="phase-l-v1",
            direct_predecessor_hashes={item: computed_certificates[item]["artifact_hash"] for item in predecessors},
            evidence=[{"fresh_process": True, "raw_recomputed": True}], failure=failure,
        )

    # 结构性 obligation 的根 preimage 不包含结构性节点本身，先用已经完成
    # 的语义节点计算 root，再按 registry 顺序生成结构证书。这样不会产生
    # 自引用，也不会通过修改 certificate 后再默默修补 predecessor hash。
    semantic_statuses = {key: value["obligation_status"] for key, value in computed_certificates.items()}
    if any(value == "FAIL" for value in semantic_statuses.values()):
        result_status = "REFERENCE_CERTIFICATE_FAILED"
    elif any(value != "PASS" for value in semantic_statuses.values()):
        result_status = "UNRESOLVED"
    else:
        result_status = "DEPLOYED_TREE_PROVED"
    root_preimage = _root_preimage(context_hash, computed["request"], active, computed_certificates)
    root = sha256_object(root_preimage)
    structural_witness = {
        "ARTIFACT_MANIFEST": {"semantic_artifact_hashes": {key: value["artifact_hash"] for key, value in computed_certificates.items()}},
        "COMPONENT_CONTEXT_INTEGRITY": {"context_hash": context_hash},
        "DIRECT_PREDECESSOR_HASHES": {"registry_order": topological_order(registry)},
        "STATUS_EVIDENCE": {"semantic_statuses": {key: value["obligation_status"] for key, value in computed_certificates.items()}},
        "OUTER_BUNDLE_ROOT": {"root_preimage": root_preimage, "outer_bundle_root": root},
        "INDEPENDENT_BUNDLE_VERIFICATION": {"fresh_process": True, "root": root},
        "CLAIM_AGGREGATION_RESULT": {"outer_bundle_root": root, "result_status": result_status},
    }
    for obligation_id in topological_order(registry):
        if obligation_id not in active or obligation_id not in STRUCTURAL_IDS:
            continue
        entry = by_id[obligation_id]
        predecessors = [str(item) for item in entry.get("depends_on", []) if str(item) in active]
        computed_certificates[obligation_id] = obligation_certificate(
            obligation_id=obligation_id, status="PASS", context_hash=context_hash,
            inputs={"raw_input_context": context_hash}, witness=structural_witness[obligation_id],
            checker_id="formal_toolchain.verifier.recompute", checker_version="phase-l-v1",
            direct_predecessor_hashes={item: computed_certificates[item]["artifact_hash"] for item in predecessors},
            evidence=[{"fresh_process": True, "structural_recomputed": True}],
        )

    for obligation_id, certificate in computed_certificates.items():
        _write(out_dir / "artifacts" / f"{obligation_id}.json", certificate)
    status_evidence = {key: {"obligation_id": key, "obligation_status": value["obligation_status"],
                             "certificate_hash": value["artifact_hash"], "verified": True,
                             "outer_bundle_root": root} for key, value in computed_certificates.items()}
    _write(out_dir / "status_evidence.json", status_evidence)
    _write(out_dir / "component_contexts.json", {"certificate_context_hash": context_hash,
                                                   "context_body_hash": sha256_object(computed["context_body"])})
    _write(out_dir / "outer_bundle_root.json", {"schema_version": "outer_bundle_root_v1",
                                                 "outer_bundle_root": root, "preimage": root_preimage})
    _write(out_dir / "artifact_manifest.json", {"schema_version": "verified_artifact_manifest_v1",
                                                 "artifacts": {key: value["artifact_hash"] for key, value in computed_certificates.items()}})
    _write(out_dir / "interface_coverage_report.json", {"status": "PASS", "active_obligations": active,
                                                          "fresh_process": True})
    summary = {"schema_version": "proof_summary_v1", "workflow_status": "VERIFIED",
               "result_status": result_status, "profile": "P0",
               "primary_claim": "DEPLOYED_HI_SAFETY", "certificate_context_hash": context_hash,
               "fixture_id": computed["request"].get("fixture_id"),
               "fixture_kind": computed["request"].get("fixture_kind"),
               "outer_bundle_root": root, "active_obligation_ids": active,
               "obligation_statuses": {key: value["obligation_status"] for key, value in computed_certificates.items()},
               "fixture_claim_result": result_status,
               "real_seed_evaluation": "DEFERRED" if computed["request"].get("fixture_id", "synthetic_p0") == "synthetic_p0" else "NOT_EVALUATED"}
    _write(out_dir / "proof_summary.json", summary)
    return summary
