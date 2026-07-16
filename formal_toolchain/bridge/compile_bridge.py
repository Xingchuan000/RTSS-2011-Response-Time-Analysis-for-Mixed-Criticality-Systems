"""Phase K 正向编排器：从真实 branch map 现场生成全部局部证明对象。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
from formal_toolchain.binding.removal_binding import bind_removal_runtime
from .closure_cases import (
    build_bridge_prerequisite_certificates, build_deadline_observation_certificate,
    build_event_order_certificate, build_hi_nontruncation_certificate,
)
from .p0_case_manifest import p0_case_manifest_hash
from .p0_case_manifest import require_case
from .prefix_extension import build_parameterized_prefix_extension_certificate
from .bad_prefix import build_hi_bad_prefix_reflection_certificate
from .prefix_refinement import closed_prefix_certificate
from .transition_cases import TransitionCaseProof
from .transition_compiler import compile_and_prove_all_transition_cases
from .state_relation import P0ConcreteState, P0ReferenceState
from .state_relation import p0_state_relation_schema_hash
from .model_bounds import P0ModelBounds, derive_p0_model_bounds


def _theory(theorem_id: str) -> dict[str, str]:
    return json.loads((Path(__file__).resolve().parents[1] / "theory" / "hashes.json").read_text(encoding="utf-8"))["statements"][theorem_id]


def compile_phase_k(*, source_root: str | Path, branch_map: Mapping[str, Any],
                    reference_taskset: Mapping[str, Any], bridge_context_hash: str,
                    model_bounds: P0ModelBounds | None = None,
                    concrete_base: P0ConcreteState | None = None,
                    reference_base: P0ReferenceState | None = None,
                    upstream_certificates: Mapping[str, Mapping[str, Any]] | None = None,
                    release_mapping_certificate: Mapping[str, Any] | None = None,
                    protected_hi_certificate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """不读取外部 proof PASS 对象，只消费已绑定源码与参考 taskset。"""
    bounds = model_bounds or derive_p0_model_bounds(reference_taskset)
    compiled = compile_and_prove_all_transition_cases(
        branch_map, bridge_context_hash=bridge_context_hash, bounds=bounds)
    if compiled.get("status") != "PASS":
        return {"status": compiled.get("status", "UNRESOLVED"), "failure": "TRANSITION_COMPILATION_INCOMPLETE",
                "transition_cases": compiled}
    if branch_map.get("coverage", {}).get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "NORMAL_RUNTIME_PATH_COVERAGE_REQUIRED",
                "transition_cases": compiled, "coverage": branch_map.get("coverage")}
    if (any(proof.get("concrete_delta_source") != "EFFECT_IR" for proof in compiled["proofs"])
            or compiled.get("state_relation_schema_hash") != p0_state_relation_schema_hash(bounds)
            or compiled.get("model_bounds_hash") != bounds.fingerprint):
        return {"status": "UNRESOLVED", "failure": "BOUND_PATH_OR_SCHEMA_GATE_FAILED",
                "transition_cases": compiled}
    by_case = {proof["case_id"]: proof for proof in compiled["proofs"]}
    all_three = all(proof.get("concrete_feasibility") == "SAT"
                    and proof.get("reference_totality") == "PASS"
                    and proof.get("relation_preservation") == "PASS"
                    for proof in compiled["proofs"])
    semantic_gate = (all_three
        and by_case["HI_RELEASE"].get("demand_semantics") == "ACTUAL_COST"
        and by_case["DEGRADED_LO_RELEASE"].get("demand_semantics") == "MIN_ACTUAL_DEGRADED"
        and by_case["ARRIVAL_BATCH_NO_SWITCH"].get("job_count_delta") == 0
        and by_case["JUMP_TO_NEXT_EVENT"].get("idle_precondition_bound") is True
        and all((require_case(p["case_id"])["required_relation_components"].count("job_key") == 0
                 or p.get("affected_job_identity_bound") is True)
                and p.get("frame_predicates_bound") is True for p in compiled["proofs"]))
    if not semantic_gate:
        return {"status": "UNRESOLVED", "failure": "K5_SEMANTIC_ACCEPTANCE_GATE_FAILED",
                "transition_cases": compiled}
    root = Path(source_root)
    event_binding = bind_event_runtime(root)
    removal_binding = bind_removal_runtime(root)
    from formal_toolchain.binding.controller_binding import bind_controller_runtime
    from .handler_decomposition import build_handler_decomposition_certificate
    controller_binding = bind_controller_runtime(root)
    decomposition = build_handler_decomposition_certificate(
        root, context_hash=bridge_context_hash,
        transition_case_certificates=compiled["proofs"])
    if controller_binding.get("status") != "PASS" or decomposition.get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "COMPOSITE_HANDLER_DECOMPOSITION_REQUIRED",
                "transition_cases": compiled, "controller_binding": controller_binding,
                "decomposition": decomposition}
    if (not isinstance(upstream_certificates, Mapping)
            or any(upstream_certificates.get(name, {}).get("obligation_status") != "PASS"
                   for name in ("SCHEDULER_MODEL", "MODE_SEMANTICS_CONFORMANCE",
                                "DEMAND_ORACLE_BATCH_CONTRACT", "HI_EXECUTION_CONTRACT",
                                "REMOVAL_COMPLETENESS", "HI_NONTRUNCATION", "DEADLINE_OBSERVATION",
                                "EFFECTIVE_EVENT_ORDER", "BATCH_CLOSURE", "CONTROLLER_POSTCLOSURE",
                                "TIME_PROGRESS", "WINDOW_MODE_NORMALIZATION", "CERTIFIED_ENVELOPE"))
            or not isinstance(release_mapping_certificate, Mapping)
            or release_mapping_certificate.get("obligation_id") != "RELEASE_FIXED_REMOVAL_MAPPING"
            or release_mapping_certificate.get("obligation_status") != "PASS"):
        return {"status": "UNRESOLVED", "failure": "REGISTRY_UPSTREAM_CLOSURE_REQUIRED",
                "transition_cases": compiled}
    if concrete_base is None or reference_base is None:
        return {"status": "UNRESOLVED", "failure": "PRECLOSED_BASE_STATE_REQUIRED",
                "transition_cases": compiled}
    event_order = build_event_order_certificate(context_hash=bridge_context_hash, binding=event_binding)
    from .base_relation import build_preclosed0_base_certificate, empty_boot_states
    empty_concrete, empty_reference = empty_boot_states(reference_taskset=reference_taskset)
    proof_by_case = {row["case_id"]: row for row in compiled["proofs"]}
    certificate_by_case = {str(item.get("inputs", {}).get("case_id")): item
                          for item in compiled["proof_certificates"]}
    base_relation = build_preclosed0_base_certificate(
        context_hash=bridge_context_hash,
        reference_taskset=reference_taskset,
        demand_oracle_certificate=upstream_certificates["DEMAND_ORACLE_BATCH_CONTRACT"],
        boot_path_certificate=certificate_by_case["BOOT_TO_PRECLOSED_0"],
        arrival_batch_decomposition_certificate=decomposition,
        handler_decomposition_certificate=decomposition,
        transition_case_certificates=compiled["proof_certificates"],
        event_order_certificate=event_order,
        concrete_boot=empty_concrete, reference_boot=empty_reference)
    prereqs = build_bridge_prerequisite_certificates(
        context_hash=bridge_context_hash,
        case_proofs=compiled["proof_certificates"], event_order_certificate=event_order,
        concrete_base=concrete_base, reference_base=reference_base,
        bounds=bounds,
        coverage_certificate=branch_map["coverage"], decomposition_certificate=decomposition,
        controller_binding=controller_binding)
    # 旧的样例 base 只用于 debug 输入；正式 closed-prefix 只消费参数化
    # PreClosed(0) certificate。
    prereqs["base_relation"] = base_relation
    proofs = [TransitionCaseProof(**row) for row in compiled["proofs"]]
    closed = closed_prefix_certificate(
        base_relation_certificate=prereqs["base_relation"],
        cases=proofs, source_hash=str(branch_map["source_hash"]), bridge_context_hash=bridge_context_hash,
        branch_map=branch_map, prerequisite_certificates=prereqs,
        theorem_hash=_theory("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT")["statement_hash"],
        upstream_certificates=upstream_certificates,
        release_mapping_certificate=release_mapping_certificate,
        transition_case_certificates=compiled["proof_certificates"])
    if closed.get("obligation_status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "CLOSED_PREFIX_INCOMPLETE", "transition_cases": compiled,
                "prerequisites": prereqs, "closed_prefix": closed}
    extension = build_parameterized_prefix_extension_certificate(
        reference_taskset=reference_taskset, time_progress_certificate=prereqs["positive_time"],
        event_order_certificate=event_order, context_hash=bridge_context_hash,
        theorem_manifest={**_theory("REFERENCE_PREFIX_EXTENSION"), "theorem_id": "REFERENCE_PREFIX_EXTENSION"})
    if extension.get("inputs", {}).get("theorem_id") != "REFERENCE_PREFIX_EXTENSION":
        return {"status": "UNRESOLVED", "failure": "REFERENCE_PREFIX_EXTENSION_THEOREM_REQUIRED",
                "transition_cases": compiled, "closed_prefix": closed, "reference_extension": extension}
    deadline = build_deadline_observation_certificate(context_hash=bridge_context_hash,
                                                      removal_binding=removal_binding)
    nontruncation = build_hi_nontruncation_certificate(context_hash=bridge_context_hash,
                                                       removal_binding=removal_binding)
    bad_prefix = build_hi_bad_prefix_reflection_certificate(
        closed_prefix_certificate=closed, prefix_extension_certificate=extension,
        deadline_observation_certificate=deadline, hi_nontruncation_certificate=nontruncation,
        event_projection_certificate=prereqs["event_projection"],
        state_relation_schema=p0_state_relation_schema_hash(bounds), context_hash=bridge_context_hash,
        theorem_manifest=_theory("FINITE_HI_BAD_PREFIX_REFLECTION"),
        protected_hi_certificate=protected_hi_certificate,
        release_mapping_certificate=release_mapping_certificate)
    if bad_prefix.get("obligation_status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "BAD_PREFIX_REFLECTION_INCOMPLETE",
                "transition_cases": compiled, "prerequisites": prereqs,
                "closed_prefix": closed, "reference_extension": extension,
                "bad_prefix_reflection": bad_prefix}
    return {"status": "PASS", "manifest_hash": p0_case_manifest_hash(),
            "model_bounds_status": "PASS", "model_bounds": bounds.to_dict(),
            "transition_cases": compiled, "prerequisites": prereqs,
            "coverage": branch_map["coverage"], "controller_binding": controller_binding,
            "decomposition": decomposition,
            "closed_prefix": closed, "reference_extension": extension,
            "deadline_observation": deadline, "hi_nontruncation": nontruncation,
            "bad_prefix_reflection": bad_prefix}
