"""fresh-process verifier 的主实现。

本模块有意不导入 compiler 和 ``core.formal_checks``。candidate 只提供待验
证的 proof object；输入、源码 binding、RTA replay、结构检查和 claim
aggregation 均在本进程重新执行。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.contexts import expected_context_for_obligation, context_layer_for_obligation
from formal_toolchain.core.registry import load_registry
from formal_toolchain.verifier.aggregator import aggregate_for_claim, claim_dependency_closure
from formal_toolchain.verifier.bootstrap_checks import (
    build_interface_coverage_report,
    verify_migration_manifest,
    verify_obligation_registry,
)
from formal_toolchain.verifier.checker_catalog import VERIFIER_CHECKERS, checker_for
from formal_toolchain.verifier import independent_arithmetic
from formal_toolchain.verifier.bridge_proof_checker import (
    verify_bad_prefix_proof_object,
    verify_closed_prefix_proof_object,
    verify_prefix_extension_proof_object,
)
from formal_toolchain.verifier.release_mapping_checker import verify_release_mapping
from formal_toolchain.verifier.replay_inputs import candidate_evidence, load_verifier_inputs
from formal_toolchain.verifier.registry_graph import verifier_topological_order
from formal_toolchain.verifier.structural_checks import (
    StructuralCheckResult,
    verify_artifact_manifest,
    verify_claim_aggregation_result,
    verify_component_contexts,
    verify_independent_bundle,
    verify_predecessor_hashes,
    verify_status_evidence,
)


STRUCTURAL_IDS = frozenset({
    "ARTIFACT_MANIFEST", "COMPONENT_CONTEXT_INTEGRITY", "DIRECT_PREDECESSOR_HASHES",
    "STATUS_EVIDENCE", "OUTER_BUNDLE_ROOT", "INDEPENDENT_BUNDLE_VERIFICATION",
    "CLAIM_AGGREGATION_RESULT",
})
BRIDGE_OBLIGATION_IDS = frozenset({
    "CLOSED_PREFIX_REFINEMENT", "REFERENCE_PREFIX_EXTENSION",
    "HI_BAD_CLOSED_PREFIX_REFLECTION",
})
ROUTED_FAILURES = frozenset({
    "PROOF_BUNDLE_INVALID", "MODEL_CONFORMANCE_FAILED", "POLICY_CONTRACT_VIOLATION",
    "REFERENCE_CERTIFICATE_FAILED", "REFERENCE_COUNTEREXAMPLE",
    "CONCRETE_TIMING_COUNTEREXAMPLE", "UNRESOLVED",
})


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fail_summary(*, active: list[str], status: str, code: str,
                  message: str | None = None, **extra: Any) -> dict[str, Any]:
    result = {"schema_version": "proof_summary_v1", "workflow_status": "FAILED",
              "result_status": status, "profile": "P0",
              "primary_claim": "DEPLOYED_HI_SAFETY", "failure_route": status,
              "failure_code": code, "active_obligation_ids": active, **extra}
    if message is not None:
        result["failure_message"] = message
    return result


def _load_candidate(bundle: Path, active: list[str]) -> tuple[dict[str, Mapping[str, Any]] | None, dict[str, dict[str, Any]], dict[str, Any] | None]:
    """加载并校验 candidate envelope，缺失任一 active artifact 立即拒绝。"""

    artifact_dir = Path(bundle) / "artifacts"
    candidates: dict[str, dict[str, Any]] = {}
    contexts_path = Path(bundle) / "component_contexts.json"
    candidate_contexts = _read(contexts_path) if contexts_path.is_file() else None
    if not isinstance(candidate_contexts, Mapping):
        return None, {}, {"code": "CANDIDATE_COMPONENT_CONTEXTS_MISSING"}
    for obligation_id in active:
        path = artifact_dir / f"{obligation_id}.json"
        if not path.is_file():
            return None, {}, {"code": "CANDIDATE_CERTIFICATE_MISSING", "obligation_id": obligation_id}
        certificate = _read(path)
        if not isinstance(certificate, dict) or not verify_obligation_certificate(certificate):
            return None, {}, {"code": "CANDIDATE_CERTIFICATE_INVALID", "obligation_id": obligation_id}
        if certificate.get("obligation_id") != obligation_id:
            return None, {}, {"code": "CANDIDATE_CERTIFICATE_ID_MISMATCH", "obligation_id": obligation_id}
        try:
            layer = context_layer_for_obligation(obligation_id)
        except KeyError:
            return None, {}, {"code": "CANDIDATE_CONTEXT_LAYER_UNDECLARED", "obligation_id": obligation_id}
        expected = candidate_contexts.get(layer)
        if not isinstance(expected, Mapping) or certificate.get("certificate_context_hash") != expected.get("hash"):
            return None, {}, {"code": "CANDIDATE_CERTIFICATE_CONTEXT_LAYER_MISMATCH", "obligation_id": obligation_id}
        candidates[obligation_id] = certificate
    manifest_path = Path(bundle) / "artifact_manifest.json"
    manifest = _read(manifest_path) if manifest_path.is_file() else None
    if manifest is None:
        return candidate_contexts, candidates, {"code": "CANDIDATE_ARTIFACT_MANIFEST_MISSING"}
    return candidate_contexts, candidates, None


def _source_binding(source_root: Path) -> dict[str, Any]:
    """重算 P0 removal binding；源码边界失败必须走模型不符合路由。"""

    from formal_toolchain.binding.removal_binding import bind_removal_runtime

    binding = bind_removal_runtime(source_root)
    if binding.get("status") != "PASS":
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                "code": "REMOVAL_RUNTIME_BINDING_FAILED", "witness": binding}
    contract = binding.get("p0_contract", {})
    required = {
        "completion_precedes_deadline_observation": True,
        "hi_nontruncation": True,
    }
    if any(contract.get(key) is not expected for key, expected in required.items()):
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                "code": "RELEASE_MAPPING_SOURCE_CONTRACT_FAILED", "witness": binding}
    return {"status": "PASS", "witness": binding}


def _fresh_reference_taskset(inputs: Any, certified_envelope: Mapping[str, Any]) -> Any:
    """用 fresh certified envelope 重建 reference taskset。

    candidate 中的 reference object 只用于后续一致性比较；这里的任务顺序、
    code cost 和 budget provenance 全部来自 verifier 重新加载的 target。
    """

    from formal_toolchain.adapters.runtime_config import export_formal_target_config
    from formal_toolchain.reference.task_mapping import build_reference_taskset

    envelope_hash = sha256_object(dict(certified_envelope))
    budget_by_task = {
        str(name): {**dict(row), "b_bar": int(certified_envelope["upper"][name]),
                    "certified_envelope_hash": envelope_hash}
        for name, row in inputs.target.provenance["budget_by_task"].items()
    }
    return build_reference_taskset(
        inputs.target.ordered_tasks, budget_by_task,
        xf=inputs.target.runtime_config.c_amc_sem_lo_degradation_ratio,
        certified_envelope=certified_envelope,
        semantic_context_hash=str(inputs.contexts["semantic_context"]["hash"]),
        effective_runtime_config_hash=sha256_object(export_formal_target_config(inputs.target)),
    )


def _rta_replay(*, inputs: Any, certified_envelope: Mapping[str, Any],
                candidate: Mapping[str, Mapping[str, Any]],
                fresh_reference: Any | None = None) -> dict[str, Any]:
    """现场生成 production，再用 verifier 的独立整数 replay 重放。

    candidate 的 reference/RTA witness 只做对象一致性诊断，绝不提供 fresh
    replay 的 taskset 或 production 输入。
    """

    try:
        from formal_toolchain.reference.rta_production import protected_hi_rta
        taskset = fresh_reference or _fresh_reference_taskset(inputs, certified_envelope)
        production = protected_hi_rta(taskset)
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "PROOF_BUNDLE_INVALID", "route": "PROOF_BUNDLE_INVALID",
                "code": "FRESH_REFERENCE_TASKSET_INVALID", "message": str(exc)}

    # 通过模块属性调用，确保测试或部署环境替换独立 replay 实现时，fresh
    # verifier 不会继续使用 import 时缓存的旧函数引用。
    replay = independent_arithmetic.replay_protected_hi_rta(taskset, production)
    if replay.get("status") == "FAIL":
        return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                "code": "RTA_REPLAY_MISMATCH", "replay": replay,
                "fresh_reference": taskset.to_dict()}
    if replay.get("status") != "PASS":
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "RTA_REPLAY_UNRESOLVED", "replay": replay}

    candidate_reference = candidate_evidence(candidate.get("REFERENCE_TASKSET", {})) or {}
    candidate_taskset = candidate_reference.get("taskset")
    if isinstance(candidate_taskset, Mapping) and candidate_taskset.get("tasks") != taskset.to_dict().get("tasks"):
        return {"status": "PROOF_BUNDLE_INVALID", "route": "PROOF_BUNDLE_INVALID",
                "code": "CANDIDATE_REFERENCE_TASKSET_MISMATCH"}
    return {"status": "PASS", "replay": replay,
            "replay_hash": sha256_object(replay),
            "fresh_reference": taskset.to_dict(),
            "fresh_production_hash": sha256_object(production),
            "replay_status": "PASS"}


def _fresh_bridge_proofs(*, inputs: Any, fresh_certificates: Mapping[str, Mapping[str, Any]],
                         fresh_reference: Any | None) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any] | None]:
    """由 fresh verifier 在本进程中重新生成 closed-prefix / prefix-extension proof objects。"""

    if fresh_reference is None:
        return {}, {"route": "REFERENCE_CERTIFICATE_FAILED", "code": "FRESH_REFERENCE_TASKSET_MISSING"}
    bridge_context_hash = str(inputs.contexts["bridge_context"]["hash"])
    case_map_path = Path(inputs.workspace) / "request" / "inputs" / "formal_inputs" / "phase_k_case_map.json"
    if not case_map_path.is_file():
        return {}, {"route": "UNRESOLVED", "code": "PHASE_K_CASE_MAP_MISSING"}
    from formal_toolchain.bridge.compile_bridge import compile_phase_k
    from formal_toolchain.bridge.model_bounds import derive_p0_model_bounds
    from formal_toolchain.bridge.phase_k_runtime_states import build_preclosed_runtime_states
    from formal_toolchain.bridge.runtime_branch_map import build_runtime_branch_map

    case_map = json.loads(case_map_path.read_text(encoding="utf-8"))
    branch_map = build_runtime_branch_map(
        Path.cwd(), source_hash=str(inputs.source_manifest.get("semantic_hash", "")),
        path_map=case_map)
    if branch_map.get("status") != "PASS":
        return {}, {"route": "UNRESOLVED", "code": "PHASE_K_BRANCH_MAP_UNRESOLVED"}
    upstream_names = (
        "SCHEDULER_MODEL", "MODE_SEMANTICS_CONFORMANCE", "DEMAND_ORACLE_BATCH_CONTRACT",
        "HI_EXECUTION_CONTRACT", "REMOVAL_COMPLETENESS", "HI_NONTRUNCATION",
        "DEADLINE_OBSERVATION", "EFFECTIVE_EVENT_ORDER", "BATCH_CLOSURE",
        "CONTROLLER_POSTCLOSURE", "TIME_PROGRESS", "WINDOW_MODE_NORMALIZATION",
        "CERTIFIED_ENVELOPE",
    )
    upstream = {name: fresh_certificates[name] for name in upstream_names
                if fresh_certificates.get(name, {}).get("obligation_status") == "PASS"}
    if len(upstream) != len(upstream_names):
        return {}, {"route": "UNRESOLVED", "code": "PHASE_K_UPSTREAM_CLOSURE_UNRESOLVED"}
    release_mapping = fresh_certificates.get("RELEASE_FIXED_REMOVAL_MAPPING")
    if not isinstance(release_mapping, Mapping) or release_mapping.get("obligation_status") != "PASS":
        return {}, {"route": "UNRESOLVED", "code": "RELEASE_MAPPING_CANDIDATE_MISSING"}
    reference_taskset = fresh_reference.to_dict()
    concrete_base, reference_base = build_preclosed_runtime_states(inputs.target, reference_taskset)
    model_bounds = derive_p0_model_bounds(reference_taskset)
    bridge = compile_phase_k(
        source_root=Path.cwd(), branch_map=branch_map, reference_taskset=reference_taskset,
        bridge_context_hash=bridge_context_hash, model_bounds=model_bounds,
        concrete_base=concrete_base, reference_base=reference_base,
        upstream_certificates=upstream, release_mapping_certificate=release_mapping,
        protected_hi_certificate=None,
    )
    if bridge.get("status") != "PASS":
        return {}, {"route": "UNRESOLVED", "code": str(bridge.get("failure", "PHASE_K_UNRESOLVED"))}
    return {
        "CLOSED_PREFIX_REFINEMENT": bridge["closed_prefix"],
        "REFERENCE_PREFIX_EXTENSION": bridge["reference_extension"],
    }, None


def _fresh_bad_prefix_proof(*, inputs: Any, fresh_certificates: Mapping[str, Mapping[str, Any]],
                            fresh_reference: Any | None) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any] | None]:
    """在 corollary PASS 后，重新生成 HI bad-prefix proof object。"""

    if fresh_reference is None:
        return {}, {"route": "REFERENCE_CERTIFICATE_FAILED", "code": "FRESH_REFERENCE_TASKSET_MISSING"}
    bridge_context_hash = str(inputs.contexts["bridge_context"]["hash"])
    case_map_path = Path(inputs.workspace) / "request" / "inputs" / "formal_inputs" / "phase_k_case_map.json"
    if not case_map_path.is_file():
        return {}, {"route": "UNRESOLVED", "code": "PHASE_K_CASE_MAP_MISSING"}
    protected_hi = fresh_certificates.get("PROTECTED_HI_SAFETY_COROLLARY")
    if not isinstance(protected_hi, Mapping) or protected_hi.get("obligation_status") != "PASS":
        return {}, {"route": "UNRESOLVED", "code": "PHASE_K_PROTECTED_HI_UNRESOLVED"}
    release_mapping = fresh_certificates.get("RELEASE_FIXED_REMOVAL_MAPPING")
    if not isinstance(release_mapping, Mapping) or release_mapping.get("obligation_status") != "PASS":
        return {}, {"route": "UNRESOLVED", "code": "RELEASE_MAPPING_CANDIDATE_MISSING"}
    from formal_toolchain.bridge.compile_bridge import compile_phase_k
    from formal_toolchain.bridge.model_bounds import derive_p0_model_bounds
    from formal_toolchain.bridge.phase_k_runtime_states import build_preclosed_runtime_states
    from formal_toolchain.bridge.runtime_branch_map import build_runtime_branch_map

    case_map = json.loads(case_map_path.read_text(encoding="utf-8"))
    branch_map = build_runtime_branch_map(
        Path.cwd(), source_hash=str(inputs.source_manifest.get("semantic_hash", "")),
        path_map=case_map)
    if branch_map.get("status") != "PASS":
        return {}, {"route": "UNRESOLVED", "code": "PHASE_K_BRANCH_MAP_UNRESOLVED"}
    reference_taskset = fresh_reference.to_dict()
    concrete_base, reference_base = build_preclosed_runtime_states(inputs.target, reference_taskset)
    model_bounds = derive_p0_model_bounds(reference_taskset)
    bridge = compile_phase_k(
        source_root=Path.cwd(), branch_map=branch_map, reference_taskset=reference_taskset,
        bridge_context_hash=bridge_context_hash, model_bounds=model_bounds,
        concrete_base=concrete_base, reference_base=reference_base,
        upstream_certificates={name: fresh_certificates[name] for name in (
            "SCHEDULER_MODEL", "MODE_SEMANTICS_CONFORMANCE", "DEMAND_ORACLE_BATCH_CONTRACT",
            "HI_EXECUTION_CONTRACT", "REMOVAL_COMPLETENESS", "HI_NONTRUNCATION",
            "DEADLINE_OBSERVATION", "EFFECTIVE_EVENT_ORDER", "BATCH_CLOSURE",
            "CONTROLLER_POSTCLOSURE", "TIME_PROGRESS", "WINDOW_MODE_NORMALIZATION",
            "CERTIFIED_ENVELOPE",
        )},
        release_mapping_certificate=release_mapping,
        protected_hi_certificate=protected_hi,
    )
    if bridge.get("status") != "PASS" or "bad_prefix_reflection" not in bridge:
        return {}, {"route": "UNRESOLVED", "code": str(bridge.get("failure", "PHASE_K_UNRESOLVED"))}
    return {"HI_BAD_CLOSED_PREFIX_REFLECTION": bridge["bad_prefix_reflection"]}, None


def _semantic_certificate(*, obligation_id: str, candidate: Mapping[str, Any],
                          status: str, context_hash: str,
                          predecessors: Mapping[str, Mapping[str, Any]],
                          failure: Mapping[str, Any] | None = None,
                          witness: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把 fresh checker 结果封装成标准证书，并绑定真实前驱 hash。"""

    return obligation_certificate(
        obligation_id=obligation_id, status=status, context_hash=context_hash,
        inputs={"candidate_artifact_hash": candidate.get("artifact_hash"),
                "fresh_process": True},
        witness=dict(witness or {"candidate_witness": candidate.get("witness", {})}),
        checker_id=f"formal_toolchain.verifier.checker_catalog.{obligation_id}",
        checker_version="r10-verifier-v1",
        direct_predecessor_hashes={key: value["artifact_hash"] for key, value in predecessors.items()},
        evidence=[{"fresh_process": True, "candidate_replayed": True}],
        failure=dict(failure) if failure is not None else None,
    )


def _root_preimage(*, contexts: Mapping[str, Any], certificates: Mapping[str, Mapping[str, Any]],
                   status_evidence_hashes: Mapping[str, str], active: list[str],
                   request: Mapping[str, Any], independent_verification_payload_hash: str | None = None) -> dict[str, Any]:
    """唯一 outer-root v3 preimage；不包含 root、summary、report 和日志。"""

    # STATUS_EVIDENCE 自身在 root 生成前已经冻结，因而可以作为普通叶子
    # 纳入 root；唯一不能纳入 preimage 的是引用 root 的 OUTER mirror。
    excluded = {"OUTER_BUNDLE_ROOT", "CLAIM_AGGREGATION_RESULT"}
    return {
        "schema_version": "outer_bundle_root_v3",
        "component_context_hashes": {str(key): value.get("hash")
                                      for key, value in contexts.items()},
        "verified_obligation_artifact_hashes": {
            key: certificates[key]["artifact_hash"]
            for key in sorted(certificates) if key not in excluded
        },
        "status_evidence_hashes": dict(sorted(status_evidence_hashes.items())),
        "independent_verification_payload_hash": independent_verification_payload_hash or sha256_object({"certificate_count": len(certificates)}),
        "active_obligation_set": list(active),
        "claim_request": {key: request.get(key) for key in (
            "schema_version", "profile", "primary_claim", "target_id",
            "target_kind", "taskset_seed", "tree_variant", "optional_claims")},
    }


def _first_failed_obligation(*, order: list[str], certificates: Mapping[str, Mapping[str, Any]]) -> tuple[str | None, str | None, str | None, str | None]:
    """按 verifier 拓扑顺序提取最先失败的义务及其审计详情。"""

    for obligation_id in order:
        certificate = certificates.get(obligation_id)
        if not isinstance(certificate, Mapping):
            continue
        if certificate.get("obligation_status") == "PASS":
            continue
        failure = certificate.get("failure") if isinstance(certificate.get("failure"), Mapping) else {}
        route = str(failure.get("route") or certificate.get("failure_route") or "UNRESOLVED")
        code = str(failure.get("code") or certificate.get("failure_code") or "OBLIGATION_FAILED")
        message = failure.get("message") if isinstance(failure.get("message"), str) else None
        return obligation_id, route, code, message
    return None, None, None, None


def verify_bundle(request_path: Path, bundle: Path, out_dir: Path) -> dict[str, Any]:
    """从原始输入开始 fresh replay，最后只调用 canonical aggregator。"""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(__file__).parents[1] / "specs/obligation_registry.json"
    registry = load_registry(registry_path)
    try:
        active = sorted(claim_dependency_closure(registry, "DEPLOYED_HI_SAFETY"))
        order = verifier_topological_order(registry)
    except ValueError as exc:
        summary = _fail_summary(active=[], status="PROOF_BUNDLE_INVALID",
                                code="REGISTRY_GRAPH_INVALID", message=str(exc))
        _write(out_dir / "proof_summary.json", summary)
        return summary

    registry_check = verify_obligation_registry(registry=registry)
    if registry_check["status"] != "PASS":
        summary = _fail_summary(active=active, status="PROOF_BUNDLE_INVALID",
                                code="OBLIGATION_REGISTRY_INVALID", witness=registry_check)
        _write(out_dir / "proof_summary.json", summary)
        return summary
    migration = _read(Path(__file__).parents[1] / "specs/migration_manifest.json")
    migration_check = verify_migration_manifest(
        migration=migration, registry=registry,
        current_schema_version="obligation_registry_v3")
    if migration_check["status"] != "PASS":
        summary = _fail_summary(active=active, status="PROOF_BUNDLE_INVALID",
                                code=migration_check["code"] or "MIGRATION_MANIFEST_MISMATCH",
                                witness=migration_check)
        _write(out_dir / "proof_summary.json", summary)
        return summary

    try:
        inputs = load_verifier_inputs(request_path, source_root=Path.cwd())
    except Exception as exc:
        summary = _fail_summary(active=active, status="MODEL_CONFORMANCE_FAILED",
                                code="VERIFIER_INPUT_REPLAY_FAILED", message=str(exc))
        _write(out_dir / "proof_summary.json", summary)
        return summary
    if inputs.preflight.get("obligation_status") != "PASS":
        summary = _fail_summary(active=active, status="MODEL_CONFORMANCE_FAILED",
                                code="TARGET_PREFLIGHT_FAILED", witness=dict(inputs.preflight))
        _write(out_dir / "proof_summary.json", summary)
        return summary
    source_check = _source_binding(inputs.source_root)
    if source_check["status"] != "PASS":
        summary = _fail_summary(active=active, status=source_check["route"],
                                code=source_check["code"], witness=source_check,
                                # removal binding 失败时，明确指出最先受影响
                                # 的 P0 义务，避免 summary 只给出笼统 route。
                                violated_obligation_id="REMOVAL_COMPLETENESS")
        _write(out_dir / "proof_summary.json", summary)
        return summary
    phase_k_map = Path(inputs.workspace) / "request" / "inputs" / "formal_inputs" / "phase_k_case_map.json"
    if not phase_k_map.is_file():
        summary = _fail_summary(active=active, status="UNRESOLVED",
                                code="PHASE_K_CASE_MAP_MISSING")
        _write(out_dir / "proof_summary.json", summary)
        return summary

    candidate_contexts, candidates, candidate_error = _load_candidate(bundle, active)
    if candidate_error is not None:
        summary = _fail_summary(active=active, status="PROOF_BUNDLE_INVALID",
                                code=candidate_error["code"],
                                violated_obligation_id=candidate_error.get("obligation_id"))
        _write(out_dir / "proof_summary.json", summary)
        return summary
    assert candidate_contexts is not None
    context_check = verify_component_contexts(contexts=candidate_contexts,
                                               expected_contexts=inputs.contexts)
    if context_check.status != "PASS":
        summary = _fail_summary(active=active, status="PROOF_BUNDLE_INVALID",
                                code=context_check.code or "COMPONENT_CONTEXT_INVALID",
                                witness=context_check.witness)
        _write(out_dir / "proof_summary.json", summary)
        return summary
    context_hash = str(inputs.contexts["semantic_context"]["hash"])
    candidate_manifest = _read(Path(bundle) / "artifact_manifest.json")
    candidate_manifest_check = verify_artifact_manifest(
        registry=registry, certificates=candidates, manifest=candidate_manifest)
    if candidate_manifest_check.status != "PASS":
        summary = _fail_summary(active=active, status="PROOF_BUNDLE_INVALID",
                                code=candidate_manifest_check.code or "CANDIDATE_ARTIFACT_MANIFEST_INVALID")
        _write(out_dir / "proof_summary.json", summary)
        return summary

    from formal_toolchain.verifier.envelope_checker import independently_verify_envelope
    candidate_envelope = candidate_evidence(candidates.get("CANDIDATE_ENVELOPE", {})) or {}
    common_candidate = candidate_evidence(candidates.get("COMMON_TRANSITION_PRESERVATION", {})) or {}
    deployed_candidate = candidate_evidence(candidates.get("DEPLOYED_POLICY_PRESERVATION", {})) or {}
    envelope_state = independently_verify_envelope(
        candidate_envelope=candidate_envelope, common_preservation=common_candidate,
        deployed_preservation=deployed_candidate, raw_inputs=inputs,
        invariant_context_hash=str(inputs.contexts["invariant_context"]["hash"]),
    )
    fresh_reference = None
    if envelope_state.certified_envelope is not None:
        try:
            fresh_reference = _fresh_reference_taskset(inputs, envelope_state.certified_envelope)
        except (KeyError, TypeError, ValueError):
            # 具体失败由 REFERENCE_TASKSET 的 fresh 结果给出；不能在这里
            # 用 candidate taskset 或默认值补齐 verifier 输入。
            fresh_reference = None

    by_id = {str(entry["id"]): entry for entry in registry}
    fresh: dict[str, dict[str, Any]] = {}
    bridge_generation_cache: dict[str, Mapping[str, Any]] | None = None
    bridge_generation_failure: dict[str, Any] | None = None
    bad_bridge_generation_cache: dict[str, Mapping[str, Any]] | None = None
    bad_bridge_generation_failure: dict[str, Any] | None = None
    for obligation_id in order:
        if obligation_id not in active or obligation_id in STRUCTURAL_IDS:
            continue
        candidate = candidates[obligation_id]
        predecessor_ids = [str(item) for item in by_id[obligation_id].get("depends_on", [])
                           if str(item) in fresh]
        predecessors = {item: fresh[item] for item in predecessor_ids}
        if any(item["obligation_status"] != "PASS" for item in predecessors.values()):
            fresh[obligation_id] = _semantic_certificate(
                obligation_id=obligation_id, candidate=candidate, status="UNRESOLVED",
                context_hash=expected_context_for_obligation(obligation_id, inputs.contexts), predecessors=predecessors,
                failure={"route": "UNRESOLVED", "code": "PREDECESSOR_NOT_PASS"})
            continue
        if obligation_id in BRIDGE_OBLIGATION_IDS:
            if obligation_id == "HI_BAD_CLOSED_PREFIX_REFLECTION":
                if bad_bridge_generation_cache is None and bad_bridge_generation_failure is None:
                    bad_bridge_generation_cache, bad_bridge_generation_failure = _fresh_bad_prefix_proof(
                        inputs=inputs, fresh_certificates=fresh,
                        fresh_reference=fresh_reference)
                if bad_bridge_generation_failure is not None or bad_bridge_generation_cache is None:
                    fresh[obligation_id] = _semantic_certificate(
                        obligation_id=obligation_id, candidate=candidate, status="UNRESOLVED",
                        context_hash=expected_context_for_obligation(obligation_id, inputs.contexts),
                        predecessors=predecessors,
                        failure=bad_bridge_generation_failure or {"route": "UNRESOLVED", "code": "PHASE_K_UNRESOLVED"},
                        witness={"bridge_generation": bad_bridge_generation_failure or {"route": "UNRESOLVED", "code": "PHASE_K_UNRESOLVED"}})
                    continue
                checked = verify_bad_prefix_proof_object(
                    candidate=bad_bridge_generation_cache[obligation_id],
                    bridge_context_hash=inputs.contexts["bridge_context"]["hash"],
                    raw_inputs=inputs,
                    reference_taskset=(fresh_reference.to_dict() if fresh_reference is not None else {}),
                    certified_envelope=envelope_state.certified_envelope,
                )
                status = checked.get("status", "UNRESOLVED")
                failure = None if status == "PASS" else {
                    "route": checked.get("route", "UNRESOLVED"),
                    "code": checked.get("code", "BRIDGE_PROOF_CHECK_FAILED"),
                }
                witness = checked.get("witness")
                fresh[obligation_id] = _semantic_certificate(
                    obligation_id=obligation_id, candidate=bad_bridge_generation_cache[obligation_id],
                    status=status, context_hash=expected_context_for_obligation(obligation_id, inputs.contexts),
                    predecessors=predecessors, failure=failure, witness=witness)
                continue
            if bridge_generation_cache is None and bridge_generation_failure is None:
                bridge_generation_cache, bridge_generation_failure = _fresh_bridge_proofs(
                    inputs=inputs, fresh_certificates=fresh,
                    fresh_reference=fresh_reference)
            if bridge_generation_failure is not None or bridge_generation_cache is None:
                fresh[obligation_id] = _semantic_certificate(
                    obligation_id=obligation_id, candidate=candidate, status="UNRESOLVED",
                    context_hash=expected_context_for_obligation(obligation_id, inputs.contexts),
                    predecessors=predecessors,
                    failure=bridge_generation_failure or {"route": "UNRESOLVED", "code": "PHASE_K_UNRESOLVED"},
                    witness={"bridge_generation": bridge_generation_failure or {"route": "UNRESOLVED", "code": "PHASE_K_UNRESOLVED"}})
                continue
            checked = {
                "CLOSED_PREFIX_REFINEMENT": verify_closed_prefix_proof_object,
                "REFERENCE_PREFIX_EXTENSION": verify_prefix_extension_proof_object,
                "HI_BAD_CLOSED_PREFIX_REFLECTION": verify_bad_prefix_proof_object,
            }[obligation_id](
                candidate=bridge_generation_cache[obligation_id],
                bridge_context_hash=inputs.contexts["bridge_context"]["hash"],
                raw_inputs=inputs,
                reference_taskset=(fresh_reference.to_dict() if fresh_reference is not None else {}),
                certified_envelope=envelope_state.certified_envelope,
            )
            status = checked.get("status", "UNRESOLVED")
            failure = None if status == "PASS" else {
                "route": checked.get("route", "UNRESOLVED"),
                "code": checked.get("code", "BRIDGE_PROOF_CHECK_FAILED"),
            }
            witness = checked.get("witness")
            fresh[obligation_id] = _semantic_certificate(
                obligation_id=obligation_id, candidate=bridge_generation_cache[obligation_id],
                status=status, context_hash=expected_context_for_obligation(obligation_id, inputs.contexts),
                predecessors=predecessors, failure=failure, witness=witness)
            continue
        # candidate status 只作为 checker 的比较对象，不能决定 fresh status。
        # 即使 candidate 主动写 FAIL/UNRESOLVED，也必须继续走 verifier checker。
        status = "UNRESOLVED"
        failure = candidate.get("failure") if isinstance(candidate.get("failure"), Mapping) else None
        witness: Mapping[str, Any] | None = None
        if obligation_id == "CANDIDATE_ENVELOPE":
            status = envelope_state.candidate_status
            failure = None if status == "PASS" else {"route": "UNRESOLVED", "code": "CANDIDATE_ENVELOPE_INVALID"}
            witness = {"candidate_replayed": True}
        elif obligation_id == "COMMON_TRANSITION_PRESERVATION":
            status = envelope_state.common_status
            failure = None if status == "PASS" else {"route": "UNRESOLVED", "code": "COMMON_PRESERVATION_INVALID"}
            witness = {"candidate_replayed": True}
        elif obligation_id == "DEPLOYED_POLICY_PRESERVATION":
            status = envelope_state.deployed_status
            candidate_failure = (candidate.get("failure")
                                 if isinstance(candidate.get("failure"), Mapping) else {})
            failure = None if status == "PASS" else {
                "route": str(candidate_failure.get("route", "POLICY_CONTRACT_VIOLATION")),
                "code": str(candidate_failure.get("code", "DEPLOYED_PRESERVATION_INVALID")),
            }
            witness = {"candidate_replayed": True, "candidate_failure": candidate_failure}
        elif obligation_id == "CERTIFIED_ENVELOPE":
            status = "PASS" if envelope_state.certified_envelope is not None else "UNRESOLVED"
            failure = None if status == "PASS" else {"route": "UNRESOLVED", "code": "ENVELOPE_NOT_CERTIFIED"}
            witness = envelope_state.certified_envelope
        checker = checker_for(obligation_id)
        if checker is not None and obligation_id not in {
                "CANDIDATE_ENVELOPE", "COMMON_TRANSITION_PRESERVATION",
                "CERTIFIED_ENVELOPE"}:
            candidate_witness = candidate.get("witness", {})
            evidence_key = candidate_witness.get("evidence_key") if isinstance(candidate_witness, Mapping) else None
            raw_evidence = candidate_witness.get("evidence") if isinstance(candidate_witness, Mapping) else None
            checked = checker(
                candidate_certificate=candidate,
                candidate_evidence=raw_evidence,
                raw_inputs=inputs,
                verified_predecessors=predecessors,
                expected_context_hash=expected_context_for_obligation(obligation_id, inputs.contexts),
                certified_envelope=(envelope_state.certified_envelope
                                   if obligation_id in {"CODE_REFERENCE_UPPER_BOUND_MAPPING",
                                                        "REFERENCE_TASKSET",
                                                        "PROTECTED_HI_RTA_ARITHMETIC",
                                                        "PER_HI_TASK_INDUCTIVE_WCRT",
                                                        "PROTECTED_HI_SAFETY_COROLLARY"}
                                   else None),
                fresh_reference=(fresh_reference
                                 if obligation_id in {"CODE_REFERENCE_UPPER_BOUND_MAPPING",
                                                      "REFERENCE_TASKSET",
                                                      "PROTECTED_HI_RTA_ARITHMETIC",
                                                      "PER_HI_TASK_INDUCTIVE_WCRT",
                                                      "PROTECTED_HI_SAFETY_COROLLARY"}
                                 else None),
            )
            # fresh verifier 的状态必须完全取自独立 checker 的结果。
            # 不能只在 checker 失败时覆盖初始的 UNRESOLVED；否则 checker
            # 明确返回 PASS 时，外层仍会把该义务错误地收敛成
            # CANDIDATE_OBLIGATION_NOT_PASS，造成真实 s185 的假阴性。
            status = checked.get("status", "UNRESOLVED")
            if status != "PASS":
                failure = {"route": checked.get("route", "UNRESOLVED"),
                           "code": checked.get("code", "VERIFIER_CHECK_FAILED")}
                witness = checked.get("witness")
            else:
                # PASS 结果不应继续携带候选证书中的 failure，避免后续
                # 聚合逻辑把一个已经独立复核通过的义务误判为失败。
                failure = None
                witness = checked.get("witness")
        if obligation_id == "PROTECTED_HI_RTA_ARITHMETIC":
            replay = (_rta_replay(
                inputs=inputs, certified_envelope=envelope_state.certified_envelope,
                candidate=candidates, fresh_reference=fresh_reference)
                      if envelope_state.certified_envelope is not None else
                      {"status": "UNRESOLVED", "route": "UNRESOLVED",
                       "code": "CERTIFIED_ENVELOPE_REQUIRED"})
            status = replay["status"]
            failure = None if status == "PASS" else {
                "route": replay.get("route", "UNRESOLVED"),
                "code": replay.get("code", "RTA_REPLAY_UNRESOLVED"),
            }
            witness = replay
        elif status == "PASS" and obligation_id == "RELEASE_FIXED_REMOVAL_MAPPING":
            evidence = candidate_evidence(candidate)
            checked = verify_release_mapping(
                candidate_certificate=evidence or {}, source_root=inputs.source_root,
                bridge_context_hash=inputs.contexts["bridge_context"]["hash"])
            if checked.get("status") != "PASS":
                status = checked.get("status", "UNRESOLVED")
                failure = {"route": checked.get("route", "UNRESOLVED"),
                           "code": checked.get("code", "RELEASE_MAPPING_CHECK_FAILED")}
                witness = checked.get("witness")
        if status not in {"PASS", "FAIL", "UNRESOLVED"}:
            status = "UNRESOLVED"
            failure = {"route": "UNRESOLVED", "code": "INVALID_CANDIDATE_STATUS"}
        if status != "PASS" and failure is None:
            failure = {"route": by_id[obligation_id].get("failure_route", "UNRESOLVED"),
                       "code": "CANDIDATE_OBLIGATION_NOT_PASS"}
        fresh[obligation_id] = _semantic_certificate(
            obligation_id=obligation_id, candidate=candidate, status=status,
            context_hash=expected_context_for_obligation(obligation_id, inputs.contexts), predecessors=predecessors,
            failure=failure, witness=witness)

    # 结构证书只在对应结构 checker 真实返回 PASS 时生成；它们不改变
    # semantic status，也不参加自己的授权循环。
    structural: dict[str, dict[str, Any]] = {}
    all_before_structure = {**fresh}
    structural_checks: dict[str, StructuralCheckResult] = {}
    structural_checks["ARTIFACT_MANIFEST"] = candidate_manifest_check
    structural_checks["COMPONENT_CONTEXT_INTEGRITY"] = StructuralCheckResult(
        "PASS", None, None, {"certificate_context_hash": context_hash})
    structural_checks["DIRECT_PREDECESSOR_HASHES"] = verify_predecessor_hashes(
        registry=registry, certificates=all_before_structure)
    deferred_structural = {"STATUS_EVIDENCE", "OUTER_BUNDLE_ROOT",
                           "INDEPENDENT_BUNDLE_VERIFICATION"}
    for obligation_id in order:
        if obligation_id not in active or obligation_id not in STRUCTURAL_IDS \
                or obligation_id in deferred_structural:
            continue
        if obligation_id == "CLAIM_AGGREGATION_RESULT":
            continue
        check = structural_checks.get(obligation_id)
        if check is None:
            check = StructuralCheckResult("UNRESOLVED", "UNRESOLVED",
                                          "STRUCTURAL_CHECK_NOT_IMPLEMENTED", {})
        predecessors = {str(dep): {**fresh, **structural}[str(dep)]
                        for dep in by_id[obligation_id].get("depends_on", [])
                        if str(dep) in {**fresh, **structural}}
        status = check.status
        failure = None if status == "PASS" else {
            "route": check.route or by_id[obligation_id].get("failure_route", "UNRESOLVED"),
            "code": check.code or "STRUCTURAL_CHECK_FAILED",
            "witness": check.witness,
        }
        structural[obligation_id] = _semantic_certificate(
            obligation_id=obligation_id, candidate={"artifact_hash": None, "witness": {}},
            status=status, context_hash=expected_context_for_obligation(obligation_id, inputs.contexts), predecessors=predecessors,
            failure=failure, witness=check.witness)

    certificates = {**fresh, **structural}
    predecessor_check = verify_predecessor_hashes(registry=registry, certificates=certificates)
    if predecessor_check.status != "PASS":
        # 这里是结构失配，必须覆盖语义状态，不能被 aggregator 降级成普通
        # reference failure。
        summary = _fail_summary(active=active, status="PROOF_BUNDLE_INVALID",
                                code=predecessor_check.code or "PREDECESSOR_HASH_MISMATCH")
        _write(out_dir / "proof_summary.json", summary)
        return summary

    # 先冻结 STATUS_EVIDENCE，再生成其 registry 前驱要求的 independent
    # certificate；两者随后都作为 root 的最终叶子，避免旧实现中 root
    # 先生成、independent/status 后追加的闭包缺口。
    status_entries = {
        key: {"obligation_id": key, "obligation_status": value["obligation_status"],
              "certificate_hash": value["artifact_hash"]}
        for key, value in sorted(certificates.items())
    }
    structural["STATUS_EVIDENCE"] = _semantic_certificate(
        obligation_id="STATUS_EVIDENCE", candidate={"artifact_hash": None, "witness": {}},
        status="PASS", context_hash=expected_context_for_obligation("STATUS_EVIDENCE", inputs.contexts),
        predecessors={}, witness={"status_entries_hash": sha256_object(status_entries)})
    certificates = {**fresh, **structural}
    independent_predecessors = {
        str(dep): certificates[str(dep)]
        for dep in by_id["INDEPENDENT_BUNDLE_VERIFICATION"].get("depends_on", [])
        if str(dep) in certificates
    }
    independent_payload = {
        "schema_version": "independent_verification_payload_v1",
        "certificate_hashes": {key: value["artifact_hash"]
                               for key, value in sorted(certificates.items())},
        "status_entries_hash": sha256_object(status_entries),
    }
    independent_check = verify_independent_bundle(certificates=certificates, registry=registry)
    structural["INDEPENDENT_BUNDLE_VERIFICATION"] = _semantic_certificate(
        obligation_id="INDEPENDENT_BUNDLE_VERIFICATION", candidate={"artifact_hash": None, "witness": {}},
        status=independent_check.status,
        context_hash=expected_context_for_obligation("INDEPENDENT_BUNDLE_VERIFICATION", inputs.contexts),
        predecessors=independent_predecessors,
        failure=None if independent_check.status == "PASS" else {
            "route": independent_check.route or "PROOF_BUNDLE_INVALID",
            "code": independent_check.code or "INDEPENDENT_CERTIFICATE_INVALID"},
        witness={"independent_payload": independent_payload, **independent_check.witness})
    certificates = {**fresh, **structural}
    status_evidence_hashes = {
        key: sha256_object({"obligation_id": key,
                            "obligation_status": value["obligation_status"],
                            "certificate_hash": value["artifact_hash"]})
        for key, value in sorted(certificates.items())
        if key != "OUTER_BUNDLE_ROOT"
    }
    contexts = dict(inputs.contexts)
    root_preimage = _root_preimage(
        contexts=contexts, certificates=certificates,
        status_evidence_hashes=status_evidence_hashes, active=active,
        request=inputs.request,
        independent_verification_payload_hash=sha256_object(independent_payload),
    )
    root = sha256_object(root_preimage)
    structural["OUTER_BUNDLE_ROOT"] = _semantic_certificate(
        obligation_id="OUTER_BUNDLE_ROOT", candidate={"artifact_hash": None, "witness": {}},
        status="PASS", context_hash=expected_context_for_obligation("OUTER_BUNDLE_ROOT", inputs.contexts),
        predecessors={}, witness={"root_preimage": root_preimage, "outer_bundle_root": root})
    certificates = {**fresh, **structural}
    status_evidence = {
        key: {"obligation_id": key, "obligation_status": value["obligation_status"],
              "certificate_hash": value["artifact_hash"], "verified": True,
              "outer_bundle_root": root}
        for key, value in sorted(certificates.items())
    }
    status_check = verify_status_evidence(status_evidence=status_evidence,
                                          certificates=certificates, outer_root=root)
    final_predecessor_check = verify_predecessor_hashes(registry=registry, certificates=certificates)
    if status_check.status != "PASS" or final_predecessor_check.status != "PASS":
        summary = _fail_summary(
            active=active, status="PROOF_BUNDLE_INVALID",
            code=(status_check.code or final_predecessor_check.code
                  or "FINAL_CERTIFICATE_CLOSURE_INVALID"),
        )
        _write(out_dir / "proof_summary.json", summary)
        return summary
    aggregation_status = aggregate_for_claim(
        claim="DEPLOYED_HI_SAFETY", registry=registry,
        verified_certificates=certificates,
        verified_status_evidence=status_evidence,
        verified_outer_root=root,
        aggregation_spec=_read(Path(__file__).parents[1] / "specs/claim_aggregation.json"),
    )
    if aggregation_status not in ROUTED_FAILURES and aggregation_status != "DEPLOYED_TREE_PROVED":
        aggregation_status = "PROOF_BUNDLE_INVALID"
    for obligation_id, certificate in certificates.items():
        _write(out_dir / "artifacts" / f"{obligation_id}.json", certificate)
    _write(out_dir / "status_evidence.json", status_evidence)
    _write(out_dir / "component_contexts.json", contexts)
    _write(out_dir / "outer_bundle_root.json", {
        "schema_version": "outer_bundle_root_v3", "outer_bundle_root": root,
        "preimage": root_preimage})
    _write(out_dir / "artifact_manifest.json", {
        "schema_version": "verified_artifact_manifest_v2",
        "artifacts": {key: value["artifact_hash"] for key, value in certificates.items()}})
    coverage = build_interface_coverage_report(
        registry=registry, spec_root=Path(__file__).parents[1] / "specs",
        checker_catalog=VERIFIER_CHECKERS, structural_ids=set(STRUCTURAL_IDS))
    _write(out_dir / "interface_coverage_report.json", coverage)
    result = {"result_status": aggregation_status, "outer_bundle_root": root,
              "claim_aggregation_source": "canonical_claim_aggregation"}
    aggregation_check = verify_claim_aggregation_result(
        result=result, aggregated_status=aggregation_status, outer_root=root)
    if aggregation_check.status != "PASS":
        aggregation_status = "PROOF_BUNDLE_INVALID"
    violated, failure_route, failure_code, failure_message = _first_failed_obligation(
        order=order, certificates=certificates)
    if violated is None and aggregation_status != "DEPLOYED_TREE_PROVED":
        failure_route = aggregation_status if aggregation_status in ROUTED_FAILURES else "PROOF_BUNDLE_INVALID"
        failure_code = "CLAIM_AGGREGATION_FAILED"
    summary = {"schema_version": "proof_summary_v1", "workflow_status": "VERIFIED",
               "result_status": aggregation_status, "profile": "P0",
               "primary_claim": "DEPLOYED_HI_SAFETY", "certificate_context_hash": context_hash,
               "fixture_id": inputs.request.get("target_id"),
               "fixture_kind": inputs.request.get("target_kind"),
               "target_id": inputs.request.get("target_id"),
               "target_kind": inputs.request.get("target_kind"),
               "taskset_seed": inputs.request.get("taskset_seed"),
               "tree_variant": inputs.request.get("tree_variant"),
               "outer_bundle_root": root, "active_obligation_ids": active,
               "failure_route": failure_route,
               "failure_code": failure_code,
               "obligation_statuses": {key: value["obligation_status"] for key, value in certificates.items()},
               "fixture_claim_result": aggregation_status,
               "violated_obligation_id": violated,
               "failure_message": failure_message,
               "claim_aggregation_source": "canonical_claim_aggregation",
               "rta_replay_verified": certificates.get("PROTECTED_HI_RTA_ARITHMETIC", {}).get("obligation_status") == "PASS"
                                      and certificates.get("PROTECTED_HI_RTA_ARITHMETIC", {}).get("witness", {}).get("replay_status", "PASS") == "PASS",
               "certified_envelope_verified": certificates.get("CERTIFIED_ENVELOPE", {}).get("obligation_status") == "PASS"
                                      and certificates.get("CERTIFIED_ENVELOPE", {}).get("witness", {}).get("verified_by") == "fresh_verifier",
               "bridge_proof_verified": all(certificates.get(key, {}).get("obligation_status") == "PASS"
                                             for key in ("CLOSED_PREFIX_REFINEMENT",
                                                         "REFERENCE_PREFIX_EXTENSION",
                                                         "HI_BAD_CLOSED_PREFIX_REFLECTION")),
               "real_seed_evaluation": "DEFERRED" if inputs.request.get("target_kind") == "SYNTHETIC_P0"
               else "COMPLETED" if inputs.request.get("target_kind") is not None else "UNRESOLVED"}
    _write(out_dir / "proof_summary.json", summary)
    return summary
