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
from .prefix_extension import build_parameterized_prefix_extension_certificate
from .bad_prefix import build_hi_bad_prefix_reflection_certificate
from .prefix_refinement import closed_prefix_certificate
from .transition_cases import TransitionCaseProof
from .transition_compiler import compile_and_prove_all_transition_cases
from .state_relation import P0ConcreteState, P0ReferenceState
from .state_relation import p0_state_relation_schema_hash


def _theory(theorem_id: str) -> dict[str, str]:
    return json.loads((Path(__file__).resolve().parents[1] / "theory" / "hashes.json").read_text(encoding="utf-8"))["statements"][theorem_id]


def compile_phase_k(*, source_root: str | Path, branch_map: Mapping[str, Any],
                    reference_taskset: Mapping[str, Any], bridge_context_hash: str) -> dict[str, Any]:
    """不读取外部 proof PASS 对象，只消费已绑定源码与参考 taskset。"""
    compiled = compile_and_prove_all_transition_cases(branch_map, bridge_context_hash=bridge_context_hash)
    if compiled.get("status") != "PASS":
        return {"status": compiled.get("status", "UNRESOLVED"), "failure": "TRANSITION_COMPILATION_INCOMPLETE",
                "transition_cases": compiled}
    if (any(proof.get("concrete_delta_source") != "BOUND_PATH_IR" for proof in compiled["proofs"])
            or compiled.get("state_relation_schema_hash") != p0_state_relation_schema_hash()):
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
        and all(p.get("affected_job_identity_bound") is True and
                p.get("frame_predicates_bound") is True for p in compiled["proofs"]))
    if not semantic_gate:
        return {"status": "UNRESOLVED", "failure": "K5_SEMANTIC_ACCEPTANCE_GATE_FAILED",
                "transition_cases": compiled}
    root = Path(source_root)
    event_binding = bind_event_runtime(root)
    removal_binding = bind_removal_runtime(root)
    event_order = build_event_order_certificate(context_hash=bridge_context_hash, binding=event_binding)
    prereqs = build_bridge_prerequisite_certificates(
        context_hash=bridge_context_hash,
        case_proofs=compiled["proof_certificates"], event_order_certificate=event_order)
    proofs = [TransitionCaseProof(**row) for row in compiled["proofs"]]
    closed = closed_prefix_certificate(
        P0ConcreteState(0, "LO"), P0ReferenceState(0, "LO"),
        proofs, source_hash=str(branch_map["source_hash"]), bridge_context_hash=bridge_context_hash,
        branch_map=branch_map, prerequisite_certificates=prereqs,
        theorem_hash=_theory("CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT")["statement_hash"])
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
        state_relation_schema=p0_state_relation_schema_hash(), context_hash=bridge_context_hash,
        theorem_manifest=_theory("FINITE_HI_BAD_PREFIX_REFLECTION"))
    return {"status": "PASS", "manifest_hash": p0_case_manifest_hash(),
            "transition_cases": compiled, "prerequisites": prereqs,
            "closed_prefix": closed, "reference_extension": extension,
            "deadline_observation": deadline, "hi_nontruncation": nontruncation,
            "bad_prefix_reflection": bad_prefix}
