"""Phase K05/K06：P0 transition proof object 与 branch coverage。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
from formal_toolchain.core.hashing import sha256_object
from .state_relation import p0_state_relation_schema_hash
from .state_relation import p0_smt_relation_fields


REQUIRED_P0_CASE_IDS = (
    "BOOT_TO_PRECLOSED_0", "ARRIVAL_BATCH_NO_SWITCH", "ARRIVAL_BATCH_SWITCH_S0",
    "PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE", "PREEMPTION_DISPATCH",
    "ONE_SERVICE_TICK", "NORMAL_COMPLETION", "PRIMARY_LO_CANCELLATION",
    "DEGRADED_COMPLETION", "HI_COMPLETION", "DEADLINE_OBSERVATION_NO_MISS",
    "DEADLINE_OBSERVATION_FIRST_HI_MISS", "IDLE_RECOVERY", "CONTROLLER_NO_ACTION",
    "CONTROLLER_SELECTED_ACTION", "JUMP_TO_NEXT_EVENT",
)


@dataclass(frozen=True, slots=True)
class TransitionCaseProof:
    case_id: str
    precondition_formula: str
    concrete_delta: str
    projected_reference_delta: str
    relation_preservation_formula: str
    z3_proof_result: str
    bound_source_hash: str
    source_branch_id: str
    verified_by_checker: bool = False
    concrete_delta_hash: str = ""
    projected_reference_delta_hash: str = ""
    state_relation_schema_hash: str = ""
    branch_subtree_hash: str = ""
    bridge_context_hash: str = ""
    case_template_hash: str = ""
    concrete_delta_source: str = ""
    path_id: str = ""
    path_effect_hash: str = ""
    concrete_feasibility: str = "UNRESOLVED"
    reference_totality: str = "UNRESOLVED"
    relation_preservation: str = "UNRESOLVED"
    demand_semantics: str = "NOT_APPLICABLE"
    job_count_delta: int = 0
    idle_precondition_bound: bool = False
    affected_job_identity_bound: bool = False
    frame_predicates_bound: bool = False

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def required_p0_case_ids() -> tuple[str, ...]:
    """返回计划 K05 固定要求的全部 case ID。"""
    return REQUIRED_P0_CASE_IDS


def prove_smt2_case(*, case_id: str, source_branch_id: str,
                    declarations: str, precondition: str,
                    preservation: str, concrete_delta: str,
                    projected_reference_delta: str,
                    bound_source_hash: str) -> TransitionCaseProof:
    """检查 concrete 可行性及 ``Concrete => exists Reference`` simulation。

    输入是带声明的 SMT-LIB2 片段，而不是调用方传入的 PASS 标志。若 Z3
    未安装或片段无法解析，结果明确为 UNRESOLVED，绝不降级为 PASS。
    """
    base = {
        "case_id": case_id, "precondition_formula": precondition,
        "concrete_delta": concrete_delta,
        "projected_reference_delta": projected_reference_delta,
        "relation_preservation_formula": preservation,
        "bound_source_hash": bound_source_hash, "source_branch_id": source_branch_id,
        "concrete_delta_hash": sha256_object(concrete_delta),
        "projected_reference_delta_hash": sha256_object(projected_reference_delta),
        "state_relation_schema_hash": p0_state_relation_schema_hash(),
    }
    try:
        import z3
    except ImportError as exc:
        unresolved = dict(base)
        unresolved["relation_preservation_formula"] = f"{preservation}\n; error={exc}"
        return TransitionCaseProof(z3_proof_result="UNRESOLVED", verified_by_checker=True, **unresolved)
    try:
        feasibility = z3.Solver()
        feasibility.from_string(declarations + f"\n(assert {precondition})\n(assert {concrete_delta})\n")
        feasible_result = feasibility.check()
        concrete_feasibility = "SAT" if feasible_result == z3.sat else ("UNSAT" if feasible_result == z3.unsat else "UNRESOLVED")
        post_vars = " ".join(f"(r_{field}_post Int)" for field in p0_smt_relation_fields())
        def counterexample(body: str) -> object:
            solver = z3.Solver()
            solver.from_string(declarations + f"\n(assert {precondition})\n(assert {concrete_delta})\n(assert (not (exists ({post_vars}) {body})))\n")
            return solver.check()
        totality_result = counterexample(projected_reference_delta)
        preservation_result = counterexample(f"(and {projected_reference_delta} {preservation})")
        reference_totality = "PASS" if totality_result == z3.unsat else ("FAIL" if totality_result == z3.sat else "UNRESOLVED")
        relation_preservation = "PASS" if preservation_result == z3.unsat else ("FAIL" if preservation_result == z3.sat else "UNRESOLVED")
        proof_result = "PASS" if (concrete_feasibility == "SAT" and reference_totality == "PASS" and relation_preservation == "PASS") else ("UNRESOLVED" if "UNRESOLVED" in {concrete_feasibility, reference_totality, relation_preservation} or concrete_feasibility == "UNSAT" else "FAIL")
        return TransitionCaseProof(z3_proof_result=proof_result, verified_by_checker=True,
                                   concrete_feasibility=concrete_feasibility,
                                   reference_totality=reference_totality,
                                   relation_preservation=relation_preservation, **base)
    except (ValueError, z3.Z3Exception) as exc:
        unresolved = dict(base)
        unresolved["relation_preservation_formula"] = f"{preservation}\n; error={exc}"
        return TransitionCaseProof(z3_proof_result="UNRESOLVED", verified_by_checker=True, **unresolved)


def check_handler_coverage(source_branch_ids: Iterable[str], cases: Iterable[TransitionCaseProof],
                           *, unreachable_branch_ids: Iterable[str] = (),
                           require_all_p0_cases: bool = False,
                           unreachable_proved: bool = False) -> dict[str, object]:
    """要求每个 source branch 恰有一个 case，未覆盖则失败。"""
    branches = set(source_branch_ids)
    case_list = tuple(cases)
    covered = [case.source_branch_id for case in case_list]
    covered_set = set(covered)
    unreachable = set(unreachable_branch_ids)
    duplicates = sorted(branch for branch in covered if covered.count(branch) > 1)
    missing = sorted(branches - covered_set - unreachable)
    unknown = sorted(covered_set - branches)
    case_ids = {case.case_id for case in case_list}
    required_missing = sorted(set(REQUIRED_P0_CASE_IDS) - case_ids) if require_all_p0_cases else []
    unresolved = sorted(case.case_id for case in case_list
                        if case.z3_proof_result != "PASS" or not case.verified_by_checker)
    unreachable_unresolved = bool(unreachable and not unreachable_proved)
    # 多个 source branch 映射到同一个语义 case 是计划允许的 many-to-one
    # 映射；真正非法的是同一 source branch 被重复证明。
    status = "PASS" if not missing and not unknown and not duplicates and not required_missing \
        and not unresolved and not unreachable_unresolved else "FAIL"
    return {"status": status, "missing": missing, "unknown": unknown,
            "duplicate": sorted(set(duplicates)), "duplicate_case_ids": [],
            "unreachable": sorted(unreachable),
            "required_missing": required_missing, "unresolved_cases": unresolved,
            "unreachable_proof_missing": unreachable_unresolved}
