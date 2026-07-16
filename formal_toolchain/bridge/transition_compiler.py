"""把源码绑定的完整 transition path 编译为可检查的局部证明。"""

from __future__ import annotations

from typing import Any, Mapping

from .case_templates import compile_bound_path_effect, compile_case_template
from .state_relation import p0_state_relation_schema_hash
from .transition_cases import TransitionCaseProof, prove_smt2_case
from formal_toolchain.core.artifact import obligation_certificate


def compile_and_prove_all_transition_cases(branch_map: Mapping[str, Any], *,
                                           bridge_context_hash: str) -> dict[str, Any]:
    if branch_map.get("status") != "PASS":
        return {"status": "UNRESOLVED", "failure": "BRANCH_MAP_REQUIRED"}
    if not isinstance(bridge_context_hash, str) or len(bridge_context_hash) != 64:
        return {"status": "UNRESOLVED", "failure": "BRIDGE_CONTEXT_REQUIRED"}
    rows = branch_map.get("paths")
    if not isinstance(rows, list):
        return {"status": "UNRESOLVED", "failure": "TRANSITION_PATHS_REQUIRED"}
    proofs: list[TransitionCaseProof] = []
    for row in rows:
        try:
            template = compile_case_template(row["case_id"])
            concrete_delta = compile_bound_path_effect(row)
            proof = prove_smt2_case(
                case_id=row["case_id"], source_branch_id=row["path_id"],
                declarations=template.declarations, precondition=template.precondition,
                preservation=template.preservation, concrete_delta=concrete_delta,
                projected_reference_delta=template.reference_delta,
                bound_source_hash=str(branch_map["source_hash"]),
            )
            proof = TransitionCaseProof(**{**proof.to_dict(),
                "source_branch_id": row["path_id"],
                "branch_subtree_hash": row["path_effect_hash"],
                "bridge_context_hash": bridge_context_hash,
                "case_template_hash": template.template_hash,
                "concrete_delta_source": "BOUND_PATH_IR",
                "path_id": row["path_id"],
                "path_effect_hash": row["path_effect_hash"],
                "demand_semantics": {"PRIMARY_LO_RELEASE": "MIN_ACTUAL_B_PLUS_ONE",
                    "DEGRADED_LO_RELEASE": "MIN_ACTUAL_DEGRADED",
                    "HI_RELEASE": "ACTUAL_COST"}.get(row["case_id"], "NOT_APPLICABLE"),
                "job_count_delta": 0 if row["case_id"] == "ARRIVAL_BATCH_NO_SWITCH" else
                    (1 if row["case_id"] in {"PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE"} else 0),
                "idle_precondition_bound": row["case_id"] == "JUMP_TO_NEXT_EVENT",
                "affected_job_identity_bound": True,
                "frame_predicates_bound": True,
            })
            proofs.append(proof)
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "UNRESOLVED", "failure": "CASE_COMPILATION_FAILED", "message": str(exc)}
    from .transition_cases import check_handler_coverage
    coverage = check_handler_coverage([row["path_id"] for row in rows], proofs,
                                      require_all_p0_cases=True)
    status = "PASS" if coverage["status"] == "PASS" else ("UNRESOLVED" if coverage["unresolved_cases"] else "FAIL")
    proof_certificates = [obligation_certificate(
        obligation_id="P0_TRANSITION_CASE", status="PASS" if proof.z3_proof_result == "PASS" else "UNRESOLVED",
        context_hash=bridge_context_hash,
        inputs={"case_id": proof.case_id, "path_id": proof.path_id,
                "source_hash": proof.bound_source_hash, "template_hash": proof.case_template_hash,
                "concrete_delta_source": proof.concrete_delta_source},
        witness=proof.to_dict(), checker_id=__name__, checker_version="phase-k-v2",
        failure=None if proof.z3_proof_result == "PASS" else {"code": "Z3_CASE_UNRESOLVED"})
        for proof in proofs]
    return {"status": status, "proofs": [proof.to_dict() for proof in proofs],
            "proof_certificates": proof_certificates, "coverage": coverage,
            "branch_map_hash": branch_map["path_map_hash"], "source_hash": branch_map["source_hash"],
            "bridge_context_hash": bridge_context_hash,
            "state_relation_schema_hash": p0_state_relation_schema_hash()}
