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
from formal_toolchain.core.contexts import (expected_context_for_obligation, context_layer_for_obligation,
                                             build_terminal_route_context, build_composition_context,
                                             build_bundle_context)
from formal_toolchain.core.registry import load_registry, artifact_path_for, registry_fingerprint
from formal_toolchain.verifier.aggregator import aggregate_for_claim, _check_proof_role_invariants
from formal_toolchain.verifier.checker_context import FreshVerifierState, CheckerContext
from formal_toolchain.verifier.bootstrap_checks import (
    build_interface_coverage_report,
    verify_migration_manifest,
    verify_obligation_registry,
)
from formal_toolchain.verifier.checker_catalog import VERIFIER_CHECKERS, checker_for
from formal_toolchain.routes.resolver import resolve_route
from formal_toolchain.verifier import independent_arithmetic
from formal_toolchain.core.registry import load_registry, build_claim_closure
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


def recompute_controller_transition_certificate(
    *,
    source_root: Path,
    verified_action_binding: Mapping[str, Any],
    verified_policy_binding: Mapping[str, Any],
    verified_controller_postclosure: Mapping[str, Any],
    context_hash: str,
) -> dict[str, Any]:
    """Rebuild the controller certificate from current source in the verifier."""
    from formal_toolchain.binding.action_binding import bind_action_runtime
    from formal_toolchain.binding.controller_binding import bind_controller_runtime
    from formal_toolchain.bridge.controller_transition import (
        build_controller_transition_certificate,
    )

    action_dim = int(verified_action_binding["action_dim"])
    explicit_noop = bool(verified_action_binding["explicit_noop"])
    action_space_type = str(verified_action_binding["action_space_type"])
    action_binding = bind_action_runtime(
        Path(source_root),
        action_space_type=action_space_type,
        action_dim=action_dim,
        explicit_noop=explicit_noop,
    )
    controller_binding = bind_controller_runtime(Path(source_root))
    return build_controller_transition_certificate(
        controller_binding=controller_binding,
        action_binding=action_binding,
        deployed_policy_binding=verified_policy_binding,
        controller_postclosure_certificate=verified_controller_postclosure,
        context_hash=context_hash,
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

# These obligations can be replayed solely from the sealed request, source
# overlay, and fresh runtime adapter.  Replaying them before candidate-envelope
# loading prevents a compiler-side candidate_failure.json from hiding the
# scientifically relevant first failing model/policy obligation.
EARLY_DECISIVE_OBLIGATIONS = frozenset({
    "OVERHEAD_PROFILE",
    "MODE_SEMANTICS_CONFORMANCE",
    "DEMAND_ORACLE_BATCH_CONTRACT",
    "HI_NONTRUNCATION",
    "DEADLINE_OBSERVATION",
    "EFFECTIVE_EVENT_ORDER",
    "CONTROLLER_POSTCLOSURE",
    "ACTIVE_RELEASE_BUDGET_INVARIANT",
})


def _replay_early_decisive_failure(*, inputs: Any, active: list[str],
                                    order: list[str]) -> dict[str, Any] | None:
    """Return the first independently reproduced decisive failure, if any.

    This is deliberately fail-closed: only a checker result whose status is
    exactly ``FAIL`` is returned.  Missing evidence, checker exceptions, and
    ``UNRESOLVED`` results do not convert a malformed candidate into a semantic
    negative; normal candidate structural verification remains responsible for
    those cases.
    """

    route_strategy = resolve_route(inputs.proof_route)
    for obligation_id in order:
        if obligation_id not in active or obligation_id not in EARLY_DECISIVE_OBLIGATIONS:
            continue
        checker = checker_for(obligation_id, route_strategy=route_strategy)
        if checker is None:
            continue
        try:
            checked = checker(
                raw_inputs=inputs,
                candidate_evidence=None,
                expected_context_hash=expected_context_for_obligation(
                    obligation_id, inputs.contexts
                ),
                verified_predecessors={},
            )
        except (KeyError, TypeError, ValueError):
            continue
        if checked.get("status") != "FAIL":
            continue
        failure = checked.get("failure") if isinstance(checked.get("failure"), Mapping) else {}
        route = str(checked.get("route") or failure.get("route") or "UNRESOLVED")
        code = str(checked.get("code") or failure.get("code") or f"{obligation_id}_FAILED")
        return {
            "obligation_id": obligation_id,
            "route": route,
            "code": code,
            "witness": checked.get("witness", checked),
        }
    return None


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


def _load_candidate(bundle: Path, active: list[str], registry: list[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]] | None, dict[str, dict[str, Any]], dict[str, Any] | None]:
    """加载并校验 candidate envelope，缺失任一 active artifact 立即拒绝。"""

    from formal_toolchain.core.registry import artifact_path_for
    from formal_toolchain.verifier.artifact_verifier import verify_certificate
    by_id = {str(entry["id"]): entry for entry in registry}

    artifact_dir = Path(bundle) / "artifacts"
    candidates: dict[str, dict[str, Any]] = {}
    candidate_failure_path = Path(bundle) / "candidate_failure.json"
    if candidate_failure_path.is_file():
        candidate_failure = _read(candidate_failure_path)
        failure = candidate_failure.get("failure") if isinstance(candidate_failure, Mapping) else None
        if isinstance(failure, Mapping):
            return None, {}, {
                "code": str(failure.get("code", "CANDIDATE_INPUT_REPLAY_FAILED")),
                "candidate_failure": dict(failure),
            }
        return None, {}, {"code": "CANDIDATE_INPUT_REPLAY_FAILED"}
    contexts_path = Path(bundle) / "component_contexts.json"
    candidate_contexts = _read(contexts_path) if contexts_path.is_file() else None
    if not isinstance(candidate_contexts, Mapping):
        return None, {}, {"code": "CANDIDATE_COMPONENT_CONTEXTS_MISSING"}
    for obligation_id in active:
        entry = by_id.get(obligation_id)
        # Structural obligations are produced by this fresh verifier from the
        # bundle as a whole.  They are active claim gates, but they are not
        # candidate certificates and therefore must not be required under
        # ``bundle/artifacts`` before their checks have run.
        if obligation_id in STRUCTURAL_IDS or (
            entry is not None
            and str(entry.get("producer", {}).get("kind", "")) == "structural_verifier"
        ):
            continue
        if entry is not None:
            path = artifact_path_for(entry, bundle)
        else:
            path = artifact_dir / f"{obligation_id}.json"
        if not path.is_file():
            return None, {}, {"code": "CANDIDATE_CERTIFICATE_MISSING", "obligation_id": obligation_id}
        certificate = _read(path)
        if not isinstance(certificate, dict) or not verify_obligation_certificate(certificate):
            return None, {}, {"code": "CANDIDATE_CERTIFICATE_INVALID", "obligation_id": obligation_id}
        if certificate.get("obligation_id") != obligation_id:
            return None, {}, {"code": "CANDIDATE_CERTIFICATE_ID_MISMATCH", "obligation_id": obligation_id}
        # PASS candidate 必须满足 obligation 的专用 witness schema。FAIL/
        # UNRESOLVED candidate 只是一份 fail-closed envelope；强迫它伪造
        # PASS-only witness 字段会把正常的数学失败误报成 bundle schema 错误。
        schema_name = "common_certificate.schema.json"
        if certificate.get("obligation_status") == "PASS" and entry is not None:
            schema_name = str(entry.get("artifact_schema", schema_name)).split("/")[-1]
        schema_result = verify_certificate(certificate, schema_name=schema_name)
        if schema_result.get("status") != "PASS":
            return None, {}, {"code": "CANDIDATE_CERTIFICATE_SCHEMA_INVALID",
                              "obligation_id": obligation_id, "schema_error": schema_result.get("failure")}
        registry_layer = str(entry.get("context_layer", "")) if entry is not None else ""
        if registry_layer:
            layer = registry_layer
        else:
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


def registry_predecessors_or_fail(
    obligation_id: str,
    by_id: Mapping[str, Any],
    certificates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """精确前驱验证：缺失前驱直接报错。"""
    ids = [str(x) for x in by_id[obligation_id].get("depends_on", [])]
    missing = [x for x in ids if x not in certificates]
    if missing:
        raise ValueError(
            f"VERIFIED_PREDECESSOR_MISSING:{obligation_id}:{missing}"
        )
    return {x: certificates[x] for x in ids}


def _select_route_reference_envelope(envelope_state: Any) -> Mapping[str, Any] | None:
    """Use the certified envelope on successful proofs; candidate view is failure-only fallback."""
    return (
        envelope_state.certified_envelope
        or envelope_state.candidate_reference_envelope
    )


def build_fresh_verifier_state(
    inputs: Any,
    envelope_state: Any,
    fresh_reference: Any | None,
) -> FreshVerifierState | None:
    """一次性构造所有 fresh verifier 状态对象。"""
    if fresh_reference is None:
        return None
    route_strategy = resolve_route(inputs.proof_route)
    try:
        prepared_route = route_strategy.prepare_analysis(
            full_reference_taskset=fresh_reference,
            reference_context_hash=str(inputs.contexts["reference_context"]["hash"]),
        )
        route_registry = inputs.resolved_registry
        terminal_context = build_terminal_route_context(
            reference_context_hash=str(inputs.contexts["reference_context"]["hash"]),
            route_id=route_strategy.route_id,
            route_config_schema_version=inputs.proof_route.schema_version,
            route_registry_fragment_fingerprint=route_registry.route_fingerprint,
            route_implementation_version=prepared_route.route_implementation_version,
            analysis_taskset_fingerprint=prepared_route.analysis_taskset.to_dict()["fingerprint"],
        )
        route_certs = route_strategy.build_construction_certificates(
            prepared=prepared_route, terminal_context_hash=terminal_context["hash"])
        inputs.contexts["terminal_route_context"] = terminal_context
        inputs.contexts["composition_context"] = build_composition_context(
            bridge_context_hash=str(inputs.contexts["bridge_context"]["hash"]),
            terminal_route_context_hash=terminal_context["hash"],
            mathematical_root_id="FINAL_CLAIM_COMPOSITION", claim="DEPLOYED_HI_SAFETY")
        inputs.contexts["bundle_context"] = build_bundle_context(
            composition_context_hash=str(inputs.contexts["composition_context"]["hash"]),
            target_id=inputs.request.get("target_id"), claim="DEPLOYED_HI_SAFETY",
            resolved_registry_fingerprint=route_registry.resolved_fingerprint)
        analysis_taskset = prepared_route.analysis_taskset
        from formal_toolchain.reference.rta_production import (
            all_task_reference_rta, all_task_protected_prefix_rta,
            all_task_raw_protected_prefix_rta,
        )
        from formal_toolchain.reference.rta_replay import replay_all_task_rta
        if route_strategy.route_id == "protected_prefix":
            selected_id = "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC"
            rta_production = all_task_protected_prefix_rta(
                analysis_taskset, certificate_context_hash=terminal_context["hash"])
        elif route_strategy.route_id == "raw_protected_prefix":
            selected_id = "RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC"
            rta_production = all_task_raw_protected_prefix_rta(
                analysis_taskset, certificate_context_hash=terminal_context["hash"])
        else:
            selected_id = "ALL_TASK_REFERENCE_RTA_ARITHMETIC"
            rta_production = all_task_reference_rta(
                analysis_taskset, certificate_context_hash=terminal_context["hash"])
        # Route context is the certificate boundary; taskset source context
        # remains the full reference derivation context.
        rta_production = dict(rta_production)
        rta_replay = replay_all_task_rta(
            analysis_taskset, rta_production,
            expected_obligation_id=selected_id,
            expected_route_id=route_strategy.route_id)
    except Exception:
        prepared_route = None
        terminal_context = None
        route_certs = {}
        analysis_taskset = fresh_reference
        selected_id = None
        rta_production = {}
        rta_replay = {"status": "UNRESOLVED", "message": "RTA generation failed"}

    # N4/N5 boot state is authoritative from the frozen C-AMC-sem/P0 model.
    # The mutable experiment runtime is deliberately not executed here; q-AMC
    # or later runtime extensions may only appear in a non-blocking audit hash.
    concrete_preclosed_engine = None
    concrete_runtime_snapshot = None
    reference_preclosed_state = None
    reference_runtime_snapshot = None
    try:
        from formal_toolchain.semantics.frozen_preclosed_state import (
            build_frozen_preclosed_bundle,
        )
        from formal_toolchain.reference.runtime_snapshot import (
            build_p0_reference_runtime_snapshot,
        )

        paired_concrete, reference_preclosed_state, concrete_runtime_snapshot = (
            build_frozen_preclosed_bundle(
                inputs.target,
                fresh_reference.to_dict(),
            )
        )
        reference_runtime_snapshot = build_p0_reference_runtime_snapshot(
            reference_preclosed_state,
        )
        concrete_preclosed_engine = {
            "kind": "FROZEN_FORMAL_PRECLOSED_STATE",
            "time": int(paired_concrete.time),
            "mode": str(paired_concrete.mode),
        }
    except Exception:
        concrete_preclosed_engine = None
        concrete_runtime_snapshot = None
        reference_preclosed_state = None
        reference_runtime_snapshot = None

    return FreshVerifierState(
        inputs=inputs,
        certified_envelope=envelope_state.certified_envelope,
        fresh_reference_taskset=fresh_reference,
        fresh_rta_production=rta_production,
        fresh_rta_replay=rta_replay,
        concrete_preclosed_engine=concrete_preclosed_engine,
        concrete_runtime_snapshot=concrete_runtime_snapshot,
        reference_preclosed_state=reference_preclosed_state,
        reference_runtime_snapshot=reference_runtime_snapshot,
        phase_k_objects={},
        route_strategy=route_strategy,
        prepared_route=prepared_route,
        full_reference_taskset=fresh_reference,
        analysis_taskset=analysis_taskset,
        terminal_route_context=terminal_context,
        route_construction_certificates=route_certs,
        selected_rta_obligation_id=selected_id,
        selected_route_id=route_strategy.route_id,
    )


def _fresh_reference_taskset(inputs: Any, certified_envelope: Mapping[str, Any] | None) -> Any:
    """用 fresh certified envelope 重建 reference taskset。

    candidate 中的 reference object 只用于后续一致性比较；这里的任务顺序、
    code cost 和 budget provenance 全部来自 verifier 重新加载的 target。
    """

    if not isinstance(certified_envelope, Mapping):
        raise TypeError("FRESH_REFERENCE_TASKSET_REQUIRES_CERTIFIED_ENVELOPE")

    from formal_toolchain.adapters.runtime_config import export_formal_target_config
    from formal_toolchain.reference.task_mapping import build_reference_taskset

    envelope_hash = sha256_object(dict(certified_envelope))
    budget_by_task = {
        str(name): {**dict(row), "b_bar": int(certified_envelope["upper"][name]),
                    "certified_envelope_hash": envelope_hash}
        for name, row in inputs.target.provenance["budget_by_task"].items()
    }
    is_candidate_view = bool(certified_envelope.get("trust_level") == "CANDIDATE_UNVERIFIED")
    return build_reference_taskset(
        inputs.target.ordered_tasks, budget_by_task,
        xf=inputs.target.runtime_config.c_amc_sem_lo_degradation_ratio,
        certified_envelope=certified_envelope,
        semantic_context_hash=str(inputs.contexts["semantic_context"]["hash"]),
        effective_runtime_config_hash=sha256_object(export_formal_target_config(inputs.target)),
        allow_unverified_candidate=is_candidate_view,
    )


def _rta_replay(*, inputs: Any, certified_envelope: Mapping[str, Any] | None,
                candidate: Mapping[str, Mapping[str, Any]],
                fresh_reference: Any | None = None,
                fresh_state: FreshVerifierState | None = None) -> dict[str, Any]:
    """现场生成 production，再用 verifier 的独立整数 replay 重放。

    candidate 的 reference/RTA witness 只做对象一致性诊断，绝不提供 fresh
    replay 的 taskset 或 production 输入。
    """

    try:
        from formal_toolchain.reference.rta_production import (
            all_task_reference_rta, all_task_protected_prefix_rta,
            all_task_raw_protected_prefix_rta,
        )
        taskset = (fresh_state.analysis_taskset if fresh_state is not None
                   else fresh_reference or _fresh_reference_taskset(inputs, certified_envelope))
        route_id = fresh_state.selected_route_id if fresh_state is not None else "strict_full"
        obligation_id = (fresh_state.selected_rta_obligation_id if fresh_state is not None
                         else "ALL_TASK_REFERENCE_RTA_ARITHMETIC")
        context_hash = (fresh_state.terminal_route_context.get("hash")
                        if fresh_state is not None and fresh_state.terminal_route_context else None)
        if route_id == "protected_prefix":
            production = all_task_protected_prefix_rta(taskset, certificate_context_hash=context_hash)
        elif route_id == "raw_protected_prefix":
            production = all_task_raw_protected_prefix_rta(taskset, certificate_context_hash=context_hash)
        else:
            production = all_task_reference_rta(taskset, certificate_context_hash=context_hash)
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "PROOF_BUNDLE_INVALID", "route": "PROOF_BUNDLE_INVALID",
                "code": "FRESH_REFERENCE_TASKSET_INVALID", "message": str(exc)}

    # 通过模块属性调用，确保测试或部署环境替换独立 replay 实现时，fresh
    # verifier 不会继续使用 import 时缓存的旧函数引用。
    from formal_toolchain.reference.rta_replay import replay_all_task_rta
    replay = replay_all_task_rta(taskset, production,
                                 expected_obligation_id=obligation_id,
                                 expected_route_id=route_id)
    if replay.get("status") == "FAIL":
        return {"status": "FAIL", "route": "REFERENCE_CERTIFICATE_FAILED",
                "code": "RTA_REPLAY_MISMATCH", "replay": replay,
                "fresh_reference": taskset.to_dict()}
    if replay.get("status") != "PASS":
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "code": "RTA_REPLAY_UNRESOLVED", "replay": replay}

    candidate_reference = candidate_evidence(candidate.get("REFERENCE_TASKSET", {})) or {}
    candidate_taskset = candidate_reference.get("taskset")
    # REFERENCE_TASKSET always denotes the full concrete-to-reference target.
    # Under protected_prefix, ``taskset`` is the transformed analysis prefix, so
    # comparing it directly with the candidate full reference is a category
    # error and makes the default route fail even when both objects are valid.
    expected_full_reference = (fresh_state.full_reference_taskset
                               if fresh_state is not None
                               else fresh_reference or taskset)
    if (isinstance(candidate_taskset, Mapping)
            and candidate_taskset.get("tasks")
            != expected_full_reference.to_dict().get("tasks")):
        return {"status": "PROOF_BUNDLE_INVALID", "route": "PROOF_BUNDLE_INVALID",
                "code": "CANDIDATE_REFERENCE_TASKSET_MISMATCH"}
    if (fresh_state is not None
            and taskset.to_dict().get("fingerprint")
            != fresh_state.analysis_taskset.to_dict().get("fingerprint")):
        return {"status": "PROOF_BUNDLE_INVALID", "route": "PROOF_BUNDLE_INVALID",
                "code": "ROUTE_ANALYSIS_TASKSET_MISMATCH"}
    from formal_toolchain.reference.rta_soundness import derive_all_task_rta_soundness
    soundness = derive_all_task_rta_soundness(
        replay=replay,
        taskset=taskset,
        theorem_id=(
            "PREFIX_ALL_TASK_RTA_SOUNDNESS" if route_id == "protected_prefix" else
            "RAW_PREFIX_ALL_TASK_RTA_SOUNDNESS" if route_id == "raw_protected_prefix" else
            "REFERENCE_ALL_TASK_RTA_SOUNDNESS"
        ),
    )
    if soundness.get("status") != "PASS":
        return {
            "status": "UNRESOLVED",
            "route": "UNRESOLVED",
            "code": "RTA_SOUNDNESS_RECEIPT_UNRESOLVED",
            "replay": replay,
            "soundness_receipt": soundness.get("soundness_receipt", {}),
        }
    return {"status": "PASS", "replay": replay,
            "replay_hash": sha256_object(replay),
            "fresh_reference": taskset.to_dict(),
            "analysis_taskset": taskset.to_dict(),
            "selected_rta_obligation_id": obligation_id,
            "route_id": route_id,
            "fresh_production_hash": sha256_object(production),
            "replay_status": "PASS",
            "soundness_receipt": soundness["soundness_receipt"]}


def _fresh_source_root(inputs: Any) -> Path:
    return Path(inputs.source_root).resolve()


def _fresh_reference_prefix_backend() -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    from formal_toolchain.theory.loader import TCB_BACKENDS, load_verified_theory_statement
    theory_dir = Path(__file__).resolve().parents[1] / "theory"
    try:
        theorem = load_verified_theory_statement(theory_dir, "REFERENCE_PREFIX_EXTENSION")
    except (OSError, ValueError) as exc:
        # A stale or unavailable proof backend is an explicit unresolved proof
        # obligation, not a verifier-process crash.  This distinction also
        # preserves an earlier mathematical failure such as reference RTA FAIL.
        return None, None, {
            "route": "UNRESOLVED",
            "code": "REFERENCE_PREFIX_THEORY_LOAD_FAILED",
            "message": str(exc),
        }
    proof_object = theorem.get("proof_object", {})
    backend = TCB_BACKENDS.get(proof_object.get("backend"))
    if backend is None:
        return None, None, {"route": "UNRESOLVED", "code": "REFERENCE_PREFIX_BACKEND_MISSING"}
    proof_path = (theory_dir / proof_object.get("path", "")).resolve()
    receipt = backend.verify(proof_path, theorem=theorem)
    if receipt.get("status") != "PASS":
        return theorem, receipt, {
            "route": "UNRESOLVED", "code": "REFERENCE_PREFIX_BACKEND_REJECTED",
            "backend_result": receipt,
        }
    return theorem, receipt, None



def _fresh_reference_prefix_extension_object(
    *, inputs: Any, fresh_certificates: Mapping[str, Mapping[str, Any]],
    fresh_reference: Any | None,
) -> tuple[Mapping[str, Any] | None, dict[str, Any] | None]:
    """Build N5 prefix extension before transition identity/closed-prefix N5.

    REFERENCE_PREFIX_EXTENSION has only REFERENCE_TASKSET, TIME_PROGRESS, and
    EFFECTIVE_EVENT_ORDER as registry predecessors.  Building it through the
    monolithic Phase-K compiler introduced a verifier-only cycle because that
    compiler also builds CLOSED_PREFIX_REFINEMENT and therefore requires
    REFERENCE_TRANSITION_SYSTEM_IDENTITY, which is topologically later.
    """

    if fresh_reference is None:
        return None, {"route": "REFERENCE_CERTIFICATE_FAILED",
                      "code": "FRESH_REFERENCE_TASKSET_MISSING"}
    theorem, receipt, backend_error = _fresh_reference_prefix_backend()
    if backend_error:
        return None, backend_error
    required = ("REFERENCE_TASKSET", "TIME_PROGRESS", "EFFECTIVE_EVENT_ORDER")
    missing = [name for name in required
               if fresh_certificates.get(name, {}).get("obligation_status") != "PASS"]
    if missing:
        return None, {"route": "UNRESOLVED",
                      "code": "REFERENCE_PREFIX_UPSTREAM_CERTIFICATE_MISSING",
                      "missing": missing}
    from formal_toolchain.bridge.prefix_extension import (
        build_parameterized_prefix_extension_certificate,
    )
    try:
        certificate = build_parameterized_prefix_extension_certificate(
            reference_taskset=fresh_reference.to_dict(),
            reference_taskset_certificate=fresh_certificates["REFERENCE_TASKSET"],
            time_progress_certificate=fresh_certificates["TIME_PROGRESS"],
            event_order_certificate=fresh_certificates["EFFECTIVE_EVENT_ORDER"],
            contexts=inputs.contexts,
            context_hash=str(inputs.contexts["bridge_context"]["hash"]),
            theorem_statement=theorem,
            theorem_proof_receipt=receipt,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"route": "UNRESOLVED",
                      "code": "REFERENCE_PREFIX_EXTENSION_BUILD_FAILED",
                      "message": str(exc)}
    if certificate.get("obligation_status") != "PASS":
        return None, {"route": "UNRESOLVED",
                      "code": "REFERENCE_PREFIX_EXTENSION_BUILD_UNRESOLVED"}
    return certificate, None


def _fresh_bridge_proofs(*, inputs: Any, fresh_certificates: Mapping[str, Mapping[str, Any]],
                        fresh_reference: Any | None) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any] | None]:
    """由 fresh verifier 在本进程中重新生成 closed-prefix / prefix-extension proof objects。"""

    if fresh_reference is None:
        return {}, {"route": "REFERENCE_CERTIFICATE_FAILED", "code": "FRESH_REFERENCE_TASKSET_MISSING"}
    reference_prefix_theorem, reference_prefix_receipt, backend_error = _fresh_reference_prefix_backend()
    if backend_error:
        return {}, backend_error
    bridge_context_hash = str(inputs.contexts["bridge_context"]["hash"])
    case_map_path = Path(inputs.workspace) / "request" / "inputs" / "formal_inputs" / "phase_k_case_map.json"
    if not case_map_path.is_file():
        return {}, {"route": "UNRESOLVED", "code": "PHASE_K_CASE_MAP_MISSING"}
    from formal_toolchain.bridge.compile_bridge import compile_phase_k
    from formal_toolchain.bridge.model_bounds import derive_p0_model_bounds
    from formal_toolchain.bridge.phase_k_runtime_states import build_preclosed_runtime_states
    from formal_toolchain.bridge.runtime_branch_map import build_runtime_branch_map

    case_map = json.loads(case_map_path.read_text(encoding="utf-8"))
    src_root = _fresh_source_root(inputs)
    branch_map = build_runtime_branch_map(
        src_root, source_hash=str(inputs.source_manifest.get("semantic_hash", "")),
        path_map=case_map)
    if branch_map.get("status") != "PASS":
        return {}, {
            "route": "UNRESOLVED",
            "code": "PHASE_K_BRANCH_MAP_UNRESOLVED",
            "failure_detail": dict(branch_map),
        }
    reference_taskset = fresh_reference.to_dict()
    concrete_base, reference_base = build_preclosed_runtime_states(inputs.target, reference_taskset)
    model_bounds = derive_p0_model_bounds(reference_taskset)
    required_upstream = (
        "SCHEDULER_MODEL", "MODE_SEMANTICS_CONFORMANCE", "DEMAND_ORACLE_BATCH_CONTRACT",
        "HI_EXECUTION_CONTRACT", "REMOVAL_COMPLETENESS", "HI_NONTRUNCATION",
        "DEADLINE_OBSERVATION", "EFFECTIVE_EVENT_ORDER", "BATCH_CLOSURE",
        "CONTROLLER_POSTCLOSURE", "TIME_PROGRESS", "WINDOW_MODE_NORMALIZATION",
        "CERTIFIED_ENVELOPE", "DEPLOYED_POLICY_PRESERVATION",
        "REFERENCE_TASKSET", "REFERENCE_TRANSITION_SYSTEM_IDENTITY",
        "EFFECTIVE_EVENT_FRONTIER_RELATION",
    )
    missing_upstream = [name for name in required_upstream if name not in fresh_certificates]
    if missing_upstream:
        return {}, {"route": "UNRESOLVED", "code": "BRIDGE_UPSTREAM_CERTIFICATE_MISSING",
                    "missing": missing_upstream}
    bridge = compile_phase_k(
        source_root=src_root, branch_map=branch_map, reference_taskset=reference_taskset,
        bridge_context_hash=bridge_context_hash, model_bounds=model_bounds,
        contexts=inputs.contexts, reference_prefix_theorem=reference_prefix_theorem,
        reference_prefix_proof_receipt=reference_prefix_receipt,
        concrete_base=concrete_base, reference_base=reference_base,
        upstream_certificates={name: fresh_certificates[name] for name in required_upstream},
        release_mapping_certificate=fresh_certificates.get("RELEASE_FIXED_REMOVAL_MAPPING"),
        closure_completion_certificate=None, runtime_config=inputs.target.runtime_config,
    )
    if bridge.get("status") != "PASS":
        return {}, {
            "route": "UNRESOLVED",
            "code": str(bridge.get("failure", "PHASE_K_UNRESOLVED")),
            "failure_detail": bridge.get("failure_detail"),
            "phase_k_bridge": bridge,
        }
    return {
        "CLOSED_PREFIX_REFINEMENT": bridge["closed_prefix"],
        "REFERENCE_PREFIX_EXTENSION": bridge["reference_extension"],
        "HI_BAD_CLOSED_PREFIX_REFLECTION": bridge["bad_prefix_reflection"],
    }, None


def _semantic_certificate(*, obligation_id: str, candidate: Mapping[str, Any],
                          status: str, context_hash: str,
                          predecessors: Mapping[str, Mapping[str, Any]],
                          failure: Mapping[str, Any] | None = None,
                          witness: Mapping[str, Any] | None = None,
                          verified_inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把 fresh checker 结果封装成标准证书，并绑定真实前驱 hash。

    ``verified_inputs`` is reserved for semantic identities freshly rebuilt and
    checked in this verifier process.  It must not be populated from an
    unverified candidate placeholder.  Downstream proof builders may consume
    these identities from the verified predecessor certificate.
    """

    certificate_inputs = {
        "candidate_artifact_hash": candidate.get("artifact_hash"),
        "fresh_process": True,
    }
    if verified_inputs is not None:
        certificate_inputs.update(dict(verified_inputs))

    return obligation_certificate(
        obligation_id=obligation_id, status=status, context_hash=context_hash,
        inputs=certificate_inputs,
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


def verify_bundle(request_path: Path, bundle: Path, out_dir: Path, *, source_root: Path | None = None) -> dict[str, Any]:
    """从原始输入开始 fresh replay，最后只调用 canonical aggregator。"""

    source_root = Path(source_root).resolve(strict=True) if source_root is not None else Path(request_path).resolve().parent.parent.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        inputs = load_verifier_inputs(request_path, source_root=source_root)
        registry = list(inputs.resolved_registry.entries)
    except Exception as exc:
        summary = _fail_summary(active=[], status="MODEL_CONFORMANCE_FAILED",
                                code="VERIFIER_INPUT_REPLAY_FAILED", message=str(exc))
        _write(out_dir / "proof_summary.json", summary)
        return summary
    try:
        closure = build_claim_closure(registry, "DEPLOYED_HI_SAFETY")
        active = sorted(closure.verified_artifacts)
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
    route_migration = dict(migration)
    # The published migration manifest is for the global source registry.  A
    # resolved route is independently hashed and validated here; rebind only
    # this static manifest field so route switching does not become a bundle
    # invalidation unrelated to the selected DAG.
    route_migration["registry_fingerprint"] = registry_fingerprint(registry)
    migration_check = verify_migration_manifest(
        migration=route_migration, registry=registry,
        current_schema_version="obligation_registry_v5")
    if migration_check["status"] != "PASS":
        summary = _fail_summary(active=active, status="PROOF_BUNDLE_INVALID",
                                code=migration_check["code"] or "MIGRATION_MANIFEST_MISMATCH",
                                witness=migration_check)
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

    early_failure = _replay_early_decisive_failure(
        inputs=inputs, active=active, order=order
    )
    if early_failure is not None:
        summary = _fail_summary(
            active=active,
            status=str(early_failure["route"]),
            code=str(early_failure["code"]),
            violated_obligation_id=str(early_failure["obligation_id"]),
            witness=early_failure.get("witness"),
        )
        _write(out_dir / "proof_summary.json", summary)
        return summary

    candidate_contexts, candidates, candidate_error = _load_candidate(bundle, active, registry)
    if candidate_error is not None:
        summary = _fail_summary(active=active, status="PROOF_BUNDLE_INVALID",
                                code=candidate_error["code"],
                                violated_obligation_id=candidate_error.get("obligation_id"))
        _write(out_dir / "proof_summary.json", summary)
        return summary
    assert candidate_contexts is not None
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
    # Route-dependent candidate contexts are compiled from an explicitly
    # untrusted candidate-envelope view.  Rebuild that same view even when
    # deployed preservation fails, so component-context integrity remains a
    # structural check and does not hide the policy-contract failure.
    route_reference_envelope = _select_route_reference_envelope(envelope_state)
    if route_reference_envelope is not None:
        try:
            fresh_reference = _fresh_reference_taskset(inputs, route_reference_envelope)
        except (KeyError, TypeError, ValueError):
            fresh_reference = None

    fresh_state = build_fresh_verifier_state(inputs, envelope_state, fresh_reference)

    context_check = verify_component_contexts(contexts=candidate_contexts,
                                               expected_contexts=inputs.contexts)
    if context_check.status != "PASS":
        summary = _fail_summary(active=active, status="PROOF_BUNDLE_INVALID",
                                code=context_check.code or "COMPONENT_CONTEXT_INVALID",
                                witness=context_check.witness)
        _write(out_dir / "proof_summary.json", summary)
        return summary

    by_id = {str(entry["id"]): entry for entry in registry}
    fresh: dict[str, dict[str, Any]] = {}
    bridge_generation_cache: dict[str, Mapping[str, Any]] | None = None
    bridge_generation_failure: dict[str, Any] | None = None
    for obligation_id in order:
        if obligation_id not in active or obligation_id in closure.structural:
            continue
        candidate = candidates[obligation_id]
        try:
            predecessors = registry_predecessors_or_fail(
                obligation_id, by_id, fresh,
            )
        except ValueError as exc:
            fresh[obligation_id] = _semantic_certificate(
                obligation_id=obligation_id, candidate=candidate, status="UNRESOLVED",
                context_hash=expected_context_for_obligation(obligation_id, inputs.contexts), predecessors={},
                failure={"route": "PROOF_BUNDLE_INVALID", "code": "VERIFIED_PREDECESSOR_MISSING",
                         "message": str(exc)})
            continue
        if any(item["obligation_status"] != "PASS" for item in predecessors.values()):
            fresh[obligation_id] = _semantic_certificate(
                obligation_id=obligation_id, candidate=candidate, status="UNRESOLVED",
                context_hash=expected_context_for_obligation(obligation_id, inputs.contexts), predecessors=predecessors,
                failure={"route": "UNRESOLVED", "code": "PREDECESSOR_NOT_PASS"})
            continue
        if obligation_id in BRIDGE_OBLIGATION_IDS:
            if obligation_id == "HI_BAD_CLOSED_PREFIX_REFLECTION":
                checked = verify_bad_prefix_proof_object(
                    candidate=candidate,

                    bridge_context_hash=
                        inputs.contexts[
                            "bridge_context"
                        ]["hash"],

                    contexts=inputs.contexts,

                    predecessors=predecessors,
                )

                status = checked.get("status", "UNRESOLVED")

                failure = (
                    None
                    if status == "PASS"
                    else {
                        "route":
                            checked.get(
                                "route",
                                "UNRESOLVED",
                            ),
                        "code":
                            checked.get(
                                "code",
                                "N6_CHECK_FAILED",
                            ),
                    }
                )

                fresh[obligation_id] = (
                    _semantic_certificate(
                        obligation_id=obligation_id,
                        candidate=candidate,
                        status=status,

                        context_hash=
                            expected_context_for_obligation(
                                obligation_id,
                                inputs.contexts,
                            ),

                        predecessors=predecessors,
                        failure=failure,
                        witness=checked.get("witness"),
                    )
                )

                continue
            if obligation_id == "REFERENCE_PREFIX_EXTENSION":
                extension_object, extension_failure = _fresh_reference_prefix_extension_object(
                    inputs=inputs, fresh_certificates=fresh, fresh_reference=fresh_reference,
                )
                if extension_failure is not None or extension_object is None:
                    fresh[obligation_id] = _semantic_certificate(
                        obligation_id=obligation_id, candidate=candidate, status="UNRESOLVED",
                        context_hash=expected_context_for_obligation(obligation_id, inputs.contexts),
                        predecessors=predecessors,
                        failure=extension_failure or {"route": "UNRESOLVED",
                                                      "code": "REFERENCE_PREFIX_EXTENSION_UNRESOLVED"},
                        witness={"extension_generation": extension_failure},
                    )
                    continue
                checked = verify_prefix_extension_proof_object(
                    candidate=extension_object,
                    bridge_context_hash=inputs.contexts["bridge_context"]["hash"],
                    contexts=inputs.contexts,
                    predecessors=predecessors,
                    raw_inputs=inputs,
                    reference_taskset=fresh_reference.to_dict(),
                    certified_envelope=envelope_state.certified_envelope,
                )
                extension_status = checked.get("status", "UNRESOLVED")
                fresh[obligation_id] = _semantic_certificate(
                    obligation_id=obligation_id, candidate=extension_object,
                    status=extension_status,
                    context_hash=expected_context_for_obligation(obligation_id, inputs.contexts),
                    predecessors=predecessors,
                    failure=(None if extension_status == "PASS" else {
                        "route": checked.get("route", "UNRESOLVED"),
                        "code": checked.get("code", "REFERENCE_PREFIX_EXTENSION_CHECK_FAILED"),
                    }),
                    witness=checked.get("witness"),
                    verified_inputs=(
                        extension_object.get("inputs", {})
                        if extension_status == "PASS"
                        else None
                    ),
                )
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
                contexts=inputs.contexts,
                predecessors=predecessors,
                raw_inputs=inputs,
                reference_taskset=(fresh_reference.to_dict() if fresh_reference is not None else {}),
                certified_envelope=envelope_state.certified_envelope,
                verified_deployed_policy=fresh.get("DEPLOYED_POLICY_PRESERVATION"),
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
        checker = checker_for(obligation_id, route_strategy=(fresh_state.route_strategy if fresh_state else None))
        if checker is not None and obligation_id not in {
                "CANDIDATE_ENVELOPE", "COMMON_TRANSITION_PRESERVATION",
                "CERTIFIED_ENVELOPE"}:
            candidate_witness = candidate.get("witness", {})
            evidence_key = candidate_witness.get("evidence_key") if isinstance(candidate_witness, Mapping) else None
            raw_evidence = candidate_witness.get("evidence") if isinstance(candidate_witness, Mapping) else None
            ctx = CheckerContext(
                obligation_id=obligation_id,
                candidate_certificate=candidate,
                candidate_evidence=raw_evidence,
                verified_predecessors=predecessors,
                expected_context_hash=expected_context_for_obligation(obligation_id, inputs.contexts),
                raw_inputs=inputs,
                fresh_state=fresh_state,
            )
            try:
                checked = checker(
                    context=ctx,
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
                                                            "REFERENCE_HI_SUBSET_SAFETY",
                                                            "ALL_TASK_REFERENCE_RTA_ARITHMETIC",
                                                            "BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION",
                                                            "REFERENCE_SEMANTICS_CONTRACT",
                                                            "REFERENCE_MODEL_CONFORMANCE",
                                                            "REFERENCE_TASKSET_SCHEDULABLE",
                                                            "EFFECTIVE_EVENT_FRONTIER_RELATION",
                                                            "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
                                                            "SELECTED_REFERENCE_HI_SAFETY"}
                                       else None),
                    fresh_reference=(fresh_reference
                                     if obligation_id in {"CODE_REFERENCE_UPPER_BOUND_MAPPING",
                                                           "REFERENCE_TASKSET",
                                                           "PROTECTED_HI_RTA_ARITHMETIC",
                                                           "PER_HI_TASK_INDUCTIVE_WCRT",
                                                           "REFERENCE_HI_SUBSET_SAFETY",
                                                           "ALL_TASK_REFERENCE_RTA_ARITHMETIC",
                                                           "BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION",
                                                           "REFERENCE_SEMANTICS_CONTRACT",
                                                           "REFERENCE_TRANSITION_SYSTEM_IDENTITY",
                                                           "REFERENCE_MODEL_CONFORMANCE",
                                                           "REFERENCE_TASKSET_SCHEDULABLE",
                                                           "EFFECTIVE_EVENT_FRONTIER_RELATION",
                                                           "CLOSED_PREFIX_REFINEMENT",
                                                           "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
                                                           "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE",
                                                           "SELECTED_REFERENCE_HI_SAFETY"}
                                     else None),
                fresh_runtime_snapshot=(fresh_state.concrete_runtime_snapshot
                                        if obligation_id == "EFFECTIVE_EVENT_FRONTIER_RELATION"
                                        and fresh_state is not None else None),
                fresh_reference_snapshot=(fresh_state.reference_runtime_snapshot
                                          if obligation_id == "EFFECTIVE_EVENT_FRONTIER_RELATION"
                                          and fresh_state is not None else None),
                )
            except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as exc:
                checked = {
                    "status": "FAIL",
                    "route": "PROOF_BUNDLE_INVALID",
                    "code": "CHECKER_EXECUTION_ERROR",
                    "failure": {
                        "obligation_id": obligation_id,
                        "exception_type": type(exc).__name__,
                        "detail": str(exc),
                    },
                }
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
        if obligation_id in ("PROTECTED_HI_RTA_ARITHMETIC", "ALL_TASK_REFERENCE_RTA_ARITHMETIC",
                             "PROTECTED_PREFIX_ALL_TASK_RTA_ARITHMETIC",
                             "RAW_PREFIX_ALL_TASK_RTA_ARITHMETIC"):
            replay = (_rta_replay(
                inputs=inputs, certified_envelope=envelope_state.certified_envelope,
                candidate=candidates, fresh_reference=fresh_reference, fresh_state=fresh_state)
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
        if obligation_id not in active or obligation_id not in closure.structural \
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
        entry = by_id.get(obligation_id)
        if entry is not None:
            out_path = artifact_path_for(entry, out_dir)
        else:
            out_path = out_dir / "artifacts" / f"{obligation_id}.json"
        _write(out_path, certificate)
    _write(out_dir / "status_evidence.json", status_evidence)
    _write(out_dir / "component_contexts.json", contexts)
    _write(out_dir / "outer_bundle_root.json", {
        "schema_version": "outer_bundle_root_v3", "outer_bundle_root": root,
        "preimage": root_preimage})
    _write(out_dir / "artifact_manifest.json", {
        "schema_version": "verified_artifact_manifest_v2",
        "artifacts": {key: value["artifact_hash"] for key, value in certificates.items()}})
    combined_checker_catalog = {
        **VERIFIER_CHECKERS,
        **(fresh_state.route_strategy.checker_catalog() if fresh_state else {}),
    }
    resolved_ids = {str(item["id"]) for item in registry}
    coverage = build_interface_coverage_report(
        registry=registry, spec_root=Path(__file__).parents[1] / "specs",
        checker_catalog={key: value for key, value in combined_checker_catalog.items()
                         if key in resolved_ids},
        structural_ids=set(closure.structural))
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
    auth_gates = {k: v for k, v in certificates.items()
                  if k in closure.authorization}
    auth_all_pass = bool(auth_gates) and all(
        v.get("obligation_status") == "PASS" for v in auth_gates.values())
    math_root_pass = certificates.get("FINAL_CLAIM_COMPOSITION", {}).get("obligation_status") == "PASS"
    structural_all_pass = all(
        certificates.get(k, {}).get("obligation_status") == "PASS"
        for k in closure.structural if k in certificates)
    summary = {"schema_version": "proof_summary_v2",
               "workflow_status": "VERIFIED",
               "result_status": aggregation_status,
               "profile": "P0",
               "primary_claim": "DEPLOYED_HI_SAFETY",
               "proof_route": inputs.proof_route.route.value,
               "proof_route_schema_version": inputs.proof_route.schema_version,
               "common_registry_fingerprint": inputs.resolved_registry.common_fingerprint,
               "route_registry_fingerprint": inputs.resolved_registry.route_fingerprint,
               "resolved_registry_fingerprint": inputs.resolved_registry.resolved_fingerprint,
               "certificate_context_hash": context_hash,
               "fixture_id": inputs.request.get("target_id"),
               "fixture_kind": inputs.request.get("target_kind"),
               "target_id": inputs.request.get("target_id"),
               "target_kind": inputs.request.get("target_kind"),
               "taskset_seed": inputs.request.get("taskset_seed"),
               "tree_variant": inputs.request.get("tree_variant"),
               "outer_bundle_root": root,
               "active_obligation_ids": active,
               "failure_route": failure_route,
               "failure_code": failure_code,
               "obligation_statuses": {key: value["obligation_status"] for key, value in certificates.items()},
               "fixture_claim_result": aggregation_status,
               "violated_obligation_id": violated,
               "failure_message": failure_message,
               "claim_aggregation_source": "canonical_claim_aggregation",
               "layered_status": {
                   "environment_status": "PASS" if (
                       certificates.get("DEPENDENCY_LOCK", {}).get("obligation_status") == "PASS"
                       and certificates.get("RUNTIME_ENVIRONMENT", {}).get("obligation_status") == "PASS"
                   ) else "FAIL",
                   "pipeline_integrity_status": "PASS" if structural_all_pass and auth_all_pass else "FAIL",
                   "mathematical_proof_status": "PASS" if math_root_pass else (
                       "UNRESOLVED" if aggregation_status == "UNRESOLVED" else "FAIL"),
               },
               "rta_replay_verified": certificates.get(
                   fresh_state.selected_rta_obligation_id
                   if fresh_state is not None
                   else "ALL_TASK_REFERENCE_RTA_ARITHMETIC", {}
               ).get("obligation_status") == "PASS",
               "certified_envelope_verified": certificates.get("CERTIFIED_ENVELOPE", {}).get("obligation_status") == "PASS"
                                       and certificates.get("CERTIFIED_ENVELOPE", {}).get("witness", {}).get("verified_by") == "fresh_verifier",
               "bridge_proof_verified": all(certificates.get(key, {}).get("obligation_status") == "PASS"
                                             for key in ("CLOSED_PREFIX_REFINEMENT",
                                                         "REFERENCE_PREFIX_EXTENSION",
                                                         "HI_BAD_CLOSED_PREFIX_REFLECTION")),
               "diagnostic_mode": False,
               "claim_eligible": aggregation_status == "DEPLOYED_TREE_PROVED",
               "real_seed_evaluation": "DEFERRED" if inputs.request.get("target_kind") == "SYNTHETIC_P0"
               else "COMPLETED" if inputs.request.get("target_kind") is not None else "UNRESOLVED"}
    if fresh_state is not None:
        summary.update({
            "analysis_taskset_kind": fresh_state.prepared_route.analysis_taskset_kind if fresh_state.prepared_route else None,
            "full_reference_taskset_fingerprint": fresh_state.full_reference_taskset.to_dict()["fingerprint"] if fresh_state.full_reference_taskset else None,
            "analysis_taskset_fingerprint": fresh_state.analysis_taskset.to_dict()["fingerprint"] if fresh_state.analysis_taskset else None,
            "selected_rta_obligation_id": fresh_state.selected_rta_obligation_id,
            **(dict(fresh_state.prepared_route.route_metadata) if fresh_state.prepared_route else {}),
            "route_terminal_status": fresh_state.fresh_rta_replay.get("status", "UNRESOLVED"),
        })
    _write(out_dir / "proof_summary.json", summary)
    return summary
