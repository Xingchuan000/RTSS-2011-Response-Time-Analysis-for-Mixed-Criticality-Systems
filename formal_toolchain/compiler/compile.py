"""Phase L candidate compiler。

compiler 只生成可被 verifier 重新检查的 candidate。它不会生成可信 outer
root，也不会把 candidate summary 当作最终 claim。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.compiler.dag_runner import claim_dependency_closure, topological_order
from formal_toolchain.compiler.evidence_catalog import evidence_key_for
from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.formal_checks import calculate_raw_evidence, proof_safe
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.contexts import expected_context_for_obligation
from formal_toolchain.core.registry import load_registry


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _compile_phase_k_candidate(*, computed: Mapping[str, Any], built: Mapping[str, Mapping[str, Any]],
                               request_path: Path) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    """Phase K 由 fresh verifier 统一生成；compiler 只保留 fail-closed 诊断。"""

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    workspace = Path(request_path).resolve().parent.parent
    formal_inputs = workspace / str(request.get("formal_inputs_dir", ""))
    case_map_path = formal_inputs / "phase_k_case_map.json"
    if not case_map_path.is_file():
        return {}, "PHASE_K_CASE_MAP_MISSING"
    reference_raw = computed.get("evidence", {}).get("REFERENCE")
    if not isinstance(reference_raw, Mapping) or not isinstance(reference_raw.get("taskset"), Mapping):
        return {}, "REFERENCE_TASKSET_CANDIDATE_MISSING"
    release_mapping = computed.get("evidence", {}).get("RELEASE_FIXED_REMOVAL_MAPPING")
    if not isinstance(release_mapping, Mapping):
        return {}, "RELEASE_MAPPING_CANDIDATE_MISSING"
    contexts = computed.get("contexts", {})
    bridge_context_hash = contexts.get("bridge_context", {}).get("hash")
    source_manifest_hash = computed.get("context_body", {}).get("source_manifest", {}).get("semantic_hash")
    if not isinstance(bridge_context_hash, str) or not isinstance(source_manifest_hash, str):
        return {}, "BRIDGE_CONTEXT_INPUT_MISSING"
    # 不再在 compiler 中拼装 bridge proof object；这里只保留“有资格进入
    # fresh verifier”的输入诊断，实际 proof object 由 verifier 现场生成。
    return {}, "FRESH_VERIFIER_REQUIRED"


def compile_request(request_path: Path, out_dir: Path) -> dict[str, Any]:
    """执行 candidate DAG 并写出完整 candidate bundle。"""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(__file__).parents[1] / "specs/obligation_registry.json"
    registry = load_registry(registry_path)
    active = sorted(claim_dependency_closure(registry, "DEPLOYED_HI_SAFETY"))
    try:
        # candidate 可以生成 production/reference/RTA 的不受信任对象；它们
        # 仍必须在 fresh verifier 中重新 replay，不能把 candidate status 当
        # 成最终授权。Phase K bridge 则必须有独立 proof object，缺失时保留
        # UNRESOLVED。
        computed = calculate_raw_evidence(request_path, source_root=Path.cwd(), include_reference=True)
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

    candidate_contexts = computed.get("contexts", {}) if computed is not None else {}

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
        elif obligation_id in {"CLOSED_PREFIX_REFINEMENT", "REFERENCE_PREFIX_EXTENSION",
                                "HI_BAD_CLOSED_PREFIX_REFLECTION"}:
            # compiler 阶段不持有 fresh-process certified envelope；这些节点必须
            # 留在 candidate/UNRESOLVED，不能以预写算术结果越过 verifier 边界。
            status = "UNRESOLVED"; witness = {"not_run": "fresh verifier required"}; failure = {"route": "UNRESOLVED", "code": "FRESH_VERIFIER_REQUIRED"}
        else:
            key = evidence_key_for(obligation_id)
            raw = evidence.get(key) if key is not None else None
            # 没有唯一 evidence key 的 obligation 不能继承 PREFLIGHT，也不能
            # 通过空字典默认 PASS；candidate 只记录当前已知的事实。
            raw_status = raw.get("status", raw.get("obligation_status")) if isinstance(raw, Mapping) else None
            status = raw_status if raw_status in {"PASS", "FAIL", "UNRESOLVED"} else "UNRESOLVED"
            witness = {"evidence_key": key, "evidence": proof_safe(raw) if raw is not None else None}
            failure = None if status == "PASS" else {"route": raw.get("route", entry.get("failure_route", "UNRESOLVED")) if isinstance(raw, Mapping) else "UNRESOLVED",
                                                        "code": "CANDIDATE_EVIDENCE_FAILED" if raw is not None else "CANDIDATE_EVIDENCE_MISSING"}
        certificate_context_hash = (expected_context_for_obligation(obligation_id, candidate_contexts)
                                    if candidate_contexts and obligation_id in __import__("formal_toolchain.core.contexts", fromlist=["OBLIGATION_CONTEXT_LAYERS"]).OBLIGATION_CONTEXT_LAYERS
                                    else context_hash)
        built[obligation_id] = obligation_certificate(
            obligation_id=obligation_id, status=status, context_hash=certificate_context_hash,
            inputs={"profile": "P0", "primary_claim": "DEPLOYED_HI_SAFETY", "candidate": True},
            witness=witness, checker_id="formal_toolchain.compiler.compile",
            checker_version="phase-l-v1",
            direct_predecessor_hashes={item: built[item]["artifact_hash"] for item in predecessor_ids},
            evidence=[{"candidate": True, "fresh_verifier_required": status == "UNRESOLVED"}],
            failure=failure,
        )

    phase_k_failure: str | None = None
    if computed is not None:
        phase_k_failure = "FRESH_VERIFIER_REQUIRED"

    for obligation_id, certificate in built.items():
        _write(out_dir / "artifacts" / f"{obligation_id}.json", certificate)
    if computed is not None:
        _write(out_dir / "candidate_inputs.json", {
            "context_hash": computed["context_hash"],
            "tree_files": computed["inventory"]["files"],
            "semantic_input_hash": sha256_object(computed["context_body"]),
        })
        _write(out_dir / "component_contexts.json", computed["contexts"])
    summary = {
        "schema_version": "proof_summary_v1",
        "workflow_status": "CANDIDATE",
        "result_status": "CANDIDATE",
        "profile": "P0",
        "primary_claim": "DEPLOYED_HI_SAFETY",
        "certificate_context_hash": context_hash,
        "active_obligation_ids": active,
        "obligation_statuses": {key: value["obligation_status"] for key, value in built.items()},
        "real_seed_evaluation": "NOT_APPLICABLE" if computed and computed["request"].get("target_kind", "SYNTHETIC_P0") == "SYNTHETIC_P0" else "COMPLETED",
        "phase_k_candidate_status": "PASS" if phase_k_failure is None and computed is not None else "UNRESOLVED",
        "phase_k_candidate_failure": phase_k_failure,
    }
    _write(out_dir / "artifact_manifest.json", {"schema_version": "candidate_artifact_manifest_v1",
                                                 "artifacts": {key: value["artifact_hash"] for key, value in built.items()}})
    _write(out_dir / "proof_summary.json", summary)
    return summary
