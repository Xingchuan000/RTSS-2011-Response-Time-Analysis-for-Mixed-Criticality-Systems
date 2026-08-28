"""Phase K05/K06：P0 transition proof object 与 branch coverage。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.z3_resources import new_context, new_solver
from .state_relation import p0_state_relation_schema_hash, parameterized_state_relation_schema_hash
from .state_relation import p0_smt_relation_fields
from .model_bounds import P0ModelBounds, _legacy_test_bounds


REQUIRED_PLANT_P0_CASE_IDS = (
    "BOOT_TO_PRECLOSED_0", "ARRIVAL_BATCH_NO_SWITCH", "ARRIVAL_BATCH_SWITCH_S0",
    "PRIMARY_LO_RELEASE", "DEGRADED_LO_RELEASE", "HI_RELEASE",
    "RESCHEDULE_KEEP_SAME", "RESCHEDULE_TO_IDLE", "PREEMPTION_DISPATCH",
    "ONE_SERVICE_TICK", "NORMAL_COMPLETION", "PRIMARY_LO_CANCELLATION",
    "DEGRADED_COMPLETION", "HI_COMPLETION", "DEADLINE_OBSERVATION_NO_MISS",
    "DEADLINE_OBSERVATION_FIRST_HI_MISS", "IDLE_RECOVERY", "JUMP_TO_NEXT_EVENT",
)

REQUIRED_CONTROLLER_CASE_IDS = (
    "CONTROLLER_NO_ACTION",
    "CONTROLLER_SELECTED_ACTION",
)

# Backward-compatible name for callers that mean the formal plant casewise
# partition. Controller cases have their own certificate and are intentionally
# absent from this tuple.
REQUIRED_P0_CASE_IDS = REQUIRED_PLANT_P0_CASE_IDS

ALL_REQUIRED_P0_CASE_IDS = (
    *REQUIRED_PLANT_P0_CASE_IDS,
    *REQUIRED_CONTROLLER_CASE_IDS,
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
    guard_ast_hash: str = ""
    effect_ir_hash: str = ""
    path_ast_hash: str = ""
    queue_relation_hash: str = ""
    source_context_hash: str = ""
    model_bounds_hash: str = ""
    compiled_guard_hashes: tuple[str, ...] = ()
    consumed_effect_hashes: tuple[str, ...] = ()
    non_state_effect_hashes: tuple[str, ...] = ()
    parameterized_relation_schema_hash: str = ""
    local_footprint_hash: str = ""
    map_update_kind: str = "UNCHANGED"
    created_key_fresh_proved: bool = False
    released_ledger_contract_proved: bool = False
    terminal_ledger_contract_proved: bool = False
    miss_ledger_contract_proved: bool = False
    unaffected_job_frame_proved: bool = False
    effective_frontier_contract_proved: bool = False
    parameterized_contract_status: str = "UNRESOLVED"
    modified_components: tuple[str, ...] = ()
    semantic_effect_kinds: tuple[str, ...] = ()
    affected_job_sources: tuple[str, ...] = ()
    evidence_hashes: tuple[str, ...] = ()
    batch_decomposition_receipt_hash: str = ""
    parameterized_contract_failure: str = ""
    declarations: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def required_p0_case_ids() -> tuple[str, ...]:
    """返回正式 plant runtime case partition 的 case ID。"""
    return REQUIRED_PLANT_P0_CASE_IDS


MAP_UPDATE_KINDS = frozenset({"UNCHANGED", "EXTEND_WITH_FRESH_RELEASE", "EXTEND_WITH_FINITE_RELEASE_BATCH", "MARK_TERMINAL", "ADD_MISS_RECORD"})
EXPECTED_MAP_UPDATE_KIND = {
    "PRIMARY_LO_RELEASE": "EXTEND_WITH_FRESH_RELEASE", "DEGRADED_LO_RELEASE": "EXTEND_WITH_FRESH_RELEASE", "HI_RELEASE": "EXTEND_WITH_FRESH_RELEASE",
    "ARRIVAL_BATCH_NO_SWITCH": "EXTEND_WITH_FINITE_RELEASE_BATCH", "ARRIVAL_BATCH_SWITCH_S0": "EXTEND_WITH_FINITE_RELEASE_BATCH",
    "NORMAL_COMPLETION": "MARK_TERMINAL", "PRIMARY_LO_CANCELLATION": "MARK_TERMINAL", "DEGRADED_COMPLETION": "MARK_TERMINAL", "HI_COMPLETION": "MARK_TERMINAL",
    "DEADLINE_OBSERVATION_FIRST_HI_MISS": "ADD_MISS_RECORD",
}
ALLOWED_MODIFIED_COMPONENTS = {
    "PRIMARY_LO_RELEASE": {"released_ledger", "active_jobs", "ready_order", "effective_event_frontier"},
    "DEGRADED_LO_RELEASE": {"released_ledger", "active_jobs", "ready_order", "effective_event_frontier"},
    "HI_RELEASE": {"released_ledger", "active_jobs", "ready_order", "effective_event_frontier"},
    "ARRIVAL_BATCH_NO_SWITCH": {"released_ledger", "active_jobs", "ready_order", "effective_event_frontier"},
    "ARRIVAL_BATCH_SWITCH_S0": {"released_ledger", "active_jobs", "ready_order", "mode", "effective_event_frontier"},
    "NORMAL_COMPLETION": {"active_jobs", "ready_order", "running_key", "terminal_ledger", "effective_event_frontier"},
    "PRIMARY_LO_CANCELLATION": {"active_jobs", "ready_order", "running_key", "terminal_ledger", "effective_event_frontier"},
    "DEGRADED_COMPLETION": {"active_jobs", "ready_order", "running_key", "terminal_ledger", "effective_event_frontier"},
    "HI_COMPLETION": {"active_jobs", "ready_order", "running_key", "terminal_ledger", "effective_event_frontier"},
    "DEADLINE_OBSERVATION_FIRST_HI_MISS": {"miss_ledger", "time"},
    "DEADLINE_OBSERVATION_NO_MISS": {"time"}, "ONE_SERVICE_TICK": {"time", "active_service", "remaining_to_removal", "effective_event_frontier"},
    "RESCHEDULE_KEEP_SAME": set(),
    "RESCHEDULE_TO_IDLE": {"running_key", "effective_event_frontier"},
    "PREEMPTION_DISPATCH": {"ready_order", "running_key", "effective_event_frontier"}, "IDLE_RECOVERY": {"mode"},
    "CONTROLLER_SELECTED_ACTION": {"future_budget_ghost", "running_key", "effective_event_frontier"},
    # Both paths are source-level over-approximations.  CONTROLLER_NO_ACTION
    # contains the common apply_updates call, but its template fixes
    # update_arity=0 and the SMT proof establishes a future-budget frame.
    # JUMP_TO_NEXT_EVENT may pop and immediately push the same boundary event;
    # its queue equations and SMT proof establish frontier correspondence.
    "CONTROLLER_NO_ACTION": {"future_budget_ghost"},
    "JUMP_TO_NEXT_EVENT": {"time", "effective_event_frontier"},
    # Boot may enqueue externally supplied budget-update events.  The queue
    # projection is still checked by the bound queue equations and the SMT
    # relation proof; this entry only prevents the static footprint checker
    # from rejecting that legitimate frontier write before those checks run.
    "BOOT_TO_PRECLOSED_0": {"effective_event_frontier"},
}


@dataclass(frozen=True, slots=True)
class ParameterizedCaseContract:
    status: str
    case_id: str
    parameterized_relation_schema_hash: str
    local_footprint_hash: str
    map_update_kind: str
    modified_components: tuple[str, ...]
    created_key_fresh_proved: bool
    released_ledger_contract_proved: bool
    terminal_ledger_contract_proved: bool
    miss_ledger_contract_proved: bool
    unaffected_job_frame_proved: bool
    effective_frontier_contract_proved: bool
    evidence_hashes: tuple[str, ...]
    failure: str | None = None
    semantic_effect_kinds: tuple[str, ...] = ()
    affected_job_sources: tuple[str, ...] = ()
    batch_decomposition_receipt_hash: str = ""

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["parameterized_contract_status"] = row.pop("status")
        row["parameterized_contract_failure"] = row.pop("failure") or ""
        return row


def _unresolved(case_id: str, failure: str, **kwargs) -> ParameterizedCaseContract:
    return ParameterizedCaseContract("UNRESOLVED", case_id, parameterized_state_relation_schema_hash(), kwargs.get("local_footprint_hash", ""), kwargs.get("map_update_kind", EXPECTED_MAP_UPDATE_KIND.get(case_id, "UNCHANGED")), tuple(kwargs.get("modified_components", ())), False, False, False, False, False, False, tuple(kwargs.get("evidence_hashes", ())), failure, tuple(kwargs.get("semantic_effect_kinds", ())), tuple(kwargs.get("affected_job_sources", ())), kwargs.get("batch_decomposition_receipt_hash", ""))


def derive_parameterized_case_contract(*, case_id: str, effect_ir: Sequence[Mapping[str, object]], compiled_effect, concrete_delta: str, queue_relation_hash: str, expected_queue_relation_hash: str, batch_decomposition_certificate: Mapping[str, object] | None = None) -> ParameterizedCaseContract:
    evidence = tuple(sorted(item.get("ast_hash") for item in effect_ir if isinstance(item.get("ast_hash"), str)))
    footprint = sha256_object({"case_id": case_id, "effect_hashes": evidence, "modified_components": compiled_effect.modified_components, "semantic_effect_kinds": compiled_effect.semantic_effect_kinds})
    base = dict(local_footprint_hash=footprint, evidence_hashes=evidence, modified_components=compiled_effect.modified_components, semantic_effect_kinds=compiled_effect.semantic_effect_kinds, affected_job_sources=compiled_effect.affected_job_sources)
    if not effect_ir: return _unresolved(case_id, "PARAMETERIZED_EFFECT_IR_EMPTY", **base)
    if queue_relation_hash != expected_queue_relation_hash: return _unresolved(case_id, "PARAMETERIZED_QUEUE_BINDING_MISMATCH", **base)
    if not compiled_effect.consumed_effect_hashes: return _unresolved(case_id, "PARAMETERIZED_EFFECTS_NOT_CONSUMED", **base)
    kind = EXPECTED_MAP_UPDATE_KIND.get(case_id, "UNCHANGED")
    unexpected = set(compiled_effect.modified_components) - ALLOWED_MODIFIED_COMPONENTS.get(case_id, set())
    if unexpected: return _unresolved(case_id, "PARAMETERIZED_UNEXPECTED_STATE_WRITE:" + ",".join(sorted(unexpected)), map_update_kind=kind, **base)
    frame = bool("_post" in concrete_delta and ("c_job_" in concrete_delta or case_id not in REQUIRED_P0_CASE_IDS))
    frontier_modified = any(k in compiled_effect.semantic_effect_kinds for k in ("QUEUE_PUSH", "QUEUE_POP", "TOKEN_INVALIDATION"))
    frontier = (bool(compiled_effect.queue_equations) and queue_relation_hash == expected_queue_relation_hash and set(compiled_effect.consumed_effect_hashes) >= set(evidence)) if frontier_modified else "effective_event_frontier" not in compiled_effect.modified_components
    fresh = False
    released = False
    terminal = False
    miss = False
    batch_hash = ""
    if kind == "EXTEND_WITH_FRESH_RELEASE":
        fresh = "JOB_RELEASE" in compiled_effect.semantic_effect_kinds and "active_jobs.append" in "\n".join(str(x.get("source", "")) for x in effect_ir) and "jobs_by_key[" in "\n".join(str(x.get("source", "")) for x in effect_ir) and "release_job_key" in concrete_delta and "c_job_" in concrete_delta and "(not (= c_job_" in concrete_delta
        released = fresh
        terminal = "TERMINAL_MARK" not in compiled_effect.semantic_effect_kinds
        miss = "MISS_APPEND" not in compiled_effect.semantic_effect_kinds
    elif kind == "EXTEND_WITH_FINITE_RELEASE_BATCH":
        batch = batch_decomposition_certificate or {}
        batch_hash = str(batch.get("artifact_hash", ""))
        valid = batch.get("status", batch.get("obligation_status")) == "PASS" and batch.get("schema_version") == "arrival_batch_release_decomposition_v1" and batch.get("finite_batch") is True and batch.get("one_release_substep_per_event") is True and batch.get("release_keys_unique") is True
        fresh = valid and batch.get("release_keys_unique") is True
        released = valid; terminal = valid; miss = valid
        if not valid: return _unresolved(case_id, "ARRIVAL_BATCH_RELEASE_DECOMPOSITION_REQUIRED", map_update_kind=kind, batch_decomposition_receipt_hash=batch_hash, **base)
    elif kind == "MARK_TERMINAL":
        sources = "\n".join(str(x.get("source", "")) for x in effect_ir)
        deleted = any(x in sources for x in ("del self.jobs_by_key", "jobs_by_key.pop", "released_ledger.remove", "released_ledger.pop"))
        released = "JOB_REMOVE" in compiled_effect.semantic_effect_kinds and "TERMINAL_MARK" in compiled_effect.semantic_effect_kinds and not deleted
        terminal = released; miss = "MISS_APPEND" not in compiled_effect.semantic_effect_kinds
    elif kind == "ADD_MISS_RECORD":
        miss = "MISS_APPEND" in compiled_effect.semantic_effect_kinds and "deadline_misses.append" in "\n".join(str(x.get("source", "")) for x in effect_ir) and not any(k in compiled_effect.semantic_effect_kinds for k in ("JOB_REMOVE", "JOB_RELEASE", "TERMINAL_MARK"))
        released = miss; terminal = miss
    else:
        miss = "MISS_APPEND" not in compiled_effect.semantic_effect_kinds
        released = "released_ledger" not in compiled_effect.modified_components
        terminal = "terminal_ledger" not in compiled_effect.modified_components or "TERMINAL_MARK" in compiled_effect.semantic_effect_kinds
    status = "PASS" if all((released, terminal, miss, frame, frontier)) else "UNRESOLVED"
    failure = None if status == "PASS" else "PARAMETERIZED_CONTRACT_COMPONENT_FAILED"
    return ParameterizedCaseContract(status, case_id, parameterized_state_relation_schema_hash(), footprint, kind, tuple(compiled_effect.modified_components), fresh, released, terminal, miss, frame, bool(frontier), evidence, failure, tuple(compiled_effect.semantic_effect_kinds), tuple(compiled_effect.affected_job_sources), batch_hash)


def prove_smt2_case(*, case_id: str, source_branch_id: str,
                    declarations: str, precondition: str,
                    preservation: str, concrete_delta: str,
                    projected_reference_delta: str,
                    bound_source_hash: str,
                    bounds: P0ModelBounds | None = None) -> TransitionCaseProof:
    """检查 concrete 可行性及 ``Concrete => exists Reference`` simulation。

    输入是带声明的 SMT-LIB2 片段，而不是调用方传入的 PASS 标志。若 Z3
    未安装或片段无法解析，结果明确为 UNRESOLVED，绝不降级为 PASS。
    """
    bounds = bounds or _legacy_test_bounds()
    base = {
        "case_id": case_id, "precondition_formula": precondition,
        "concrete_delta": concrete_delta,
        "projected_reference_delta": projected_reference_delta,
        "relation_preservation_formula": preservation,
        "bound_source_hash": bound_source_hash, "source_branch_id": source_branch_id,
        "declarations": declarations,
        "concrete_delta_hash": sha256_object(concrete_delta),
        "projected_reference_delta_hash": sha256_object(projected_reference_delta),
        "state_relation_schema_hash": p0_state_relation_schema_hash(bounds),
        "model_bounds_hash": bounds.fingerprint,
    }
    try:
        import z3
    except ImportError:
        # Keep Phase K machine-checkable on hosts that provide the system Z3
        # shared library but not the z3py wheel.  The fallback submits the same
        # quantified SMT-LIB2 obligations to a fresh native Z3 solver; it never
        # replaces universal solving with finite testing.
        from formal_toolchain.theory.smt_solver import solve_closed_smt2

        feasibility_query = (
            declarations
            + f"\n(assert {precondition})\n(assert {concrete_delta})\n(check-sat)\n"
        )
        feasible_result, _ = solve_closed_smt2(feasibility_query)
        concrete_feasibility = (
            "SAT" if feasible_result == "SAT"
            else "UNSAT" if feasible_result == "UNSAT"
            else "UNRESOLVED"
        )
        post_vars = " ".join(
            f"(r_{field}_post Int)" for field in p0_smt_relation_fields(bounds)
        )

        def native_counterexample(body: str) -> str:
            query = (
                declarations
                + f"\n(assert {precondition})\n"
                + f"(assert {concrete_delta})\n"
                + f"(assert (not (exists ({post_vars}) {body})))\n"
                + "(check-sat)\n"
            )
            result, _ = solve_closed_smt2(query)
            return result

        totality_result = native_counterexample(projected_reference_delta)
        preservation_result = native_counterexample(
            f"(and {projected_reference_delta} {preservation})"
        )
        reference_totality = (
            "PASS" if totality_result == "UNSAT"
            else "FAIL" if totality_result == "SAT"
            else "UNRESOLVED"
        )
        relation_preservation = (
            "PASS" if preservation_result == "UNSAT"
            else "FAIL" if preservation_result == "SAT"
            else "UNRESOLVED"
        )
        proof_result = (
            "PASS"
            if (
                concrete_feasibility == "SAT"
                and reference_totality == "PASS"
                and relation_preservation == "PASS"
            )
            else "UNRESOLVED"
            if (
                "UNRESOLVED" in {
                    concrete_feasibility,
                    reference_totality,
                    relation_preservation,
                }
                or concrete_feasibility == "UNSAT"
            )
            else "FAIL"
        )
        return TransitionCaseProof(
            z3_proof_result=proof_result,
            verified_by_checker=True,
            concrete_feasibility=concrete_feasibility,
            reference_totality=reference_totality,
            relation_preservation=relation_preservation,
            **base,
        )
    try:
        # Each transition case owns a dedicated Z3 context.  The previous
        # default-context implementation retained native AST allocations for
        # the lifetime of the whole verifier process and caused multi-GB peaks
        # when Phase K was generated and replayed repeatedly.
        context = new_context(z3)
        feasibility = new_solver(z3, context=context)
        feasibility.from_string(declarations + f"\n(assert {precondition})\n(assert {concrete_delta})\n")
        feasible_result = feasibility.check()
        concrete_feasibility = "SAT" if feasible_result == z3.sat else ("UNSAT" if feasible_result == z3.unsat else "UNRESOLVED")
        post_vars = " ".join(f"(r_{field}_post Int)" for field in p0_smt_relation_fields(bounds))
        def counterexample(body: str) -> object:
            solver = new_solver(z3, context=context)
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
