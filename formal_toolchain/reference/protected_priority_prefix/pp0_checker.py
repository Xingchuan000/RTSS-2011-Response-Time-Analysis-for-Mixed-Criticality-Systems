"""PP0 runtime schema checker — validates primitive transition equations.

Fresh regenerates the PP0 Transition IR and SMT2 queries for every check;
never trusts candidate-provided SMT or solver results.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

from .transition_schema import (
    CANONICAL_CASES,
    canonical_case_ids,
    transition_obligations,
)
from .pp0_transition_ir import build_pp0_transition_ir, ir_for_case
from .pp0_queries import generate_all_queries, is_trivial_query_source


#  Obligation × Case applicability table.  A True entry means the obligation
#  applies to the case; False means it is structurally inapplicable (trivial).
#  This table is the single source of truth for triviality — no text inference.
CASE_OBLIGATION_APPLICABILITY: dict[str, dict[str, bool]] = {
    "REM_COMPLETION": {
        "FIXED_DEMAND_NOT_MODIFIED": True,
        "PROTECTED_KEY_NOT_MODIFIED": True,
        "MODE_ONLY_NOT_MODIFY_PROTECTED": False,
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": False,
        "DDL_READ_ONLY_DEADLINE_COMPLETION": False,
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": True,
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": False,
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": False,
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": False,
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": True,
    },
    "RECOVERY": {
        "FIXED_DEMAND_NOT_MODIFIED": True,
        "PROTECTED_KEY_NOT_MODIFIED": True,
        "MODE_ONLY_NOT_MODIFY_PROTECTED": True,
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": False,
        "DDL_READ_ONLY_DEADLINE_COMPLETION": False,
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": False,
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": False,
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": False,
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": False,
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": True,
    },
    "DDL_OBSERVE": {
        "FIXED_DEMAND_NOT_MODIFIED": True,
        "PROTECTED_KEY_NOT_MODIFIED": True,
        "MODE_ONLY_NOT_MODIFY_PROTECTED": False,
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": False,
        "DDL_READ_ONLY_DEADLINE_COMPLETION": True,
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": False,
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": False,
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": False,
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": False,
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": True,
    },
    "ARRIVAL_BATCH_OPEN": {
        "FIXED_DEMAND_NOT_MODIFIED": True,
        "PROTECTED_KEY_NOT_MODIFIED": True,
        "MODE_ONLY_NOT_MODIFY_PROTECTED": False,
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": False,
        "DDL_READ_ONLY_DEADLINE_COMPLETION": False,
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": False,
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": False,
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": False,
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": True,
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": True,
    },
    "MODE_SWITCH": {
        "FIXED_DEMAND_NOT_MODIFIED": True,
        "PROTECTED_KEY_NOT_MODIFIED": True,
        "MODE_ONLY_NOT_MODIFY_PROTECTED": True,
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": False,
        "DDL_READ_ONLY_DEADLINE_COMPLETION": False,
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": False,
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": False,
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": False,
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": False,
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": True,
    },
    "RELEASE": {
        "FIXED_DEMAND_NOT_MODIFIED": False,
        "PROTECTED_KEY_NOT_MODIFIED": True,
        "MODE_ONLY_NOT_MODIFY_PROTECTED": False,
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": False,
        "DDL_READ_ONLY_DEADLINE_COMPLETION": False,
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": False,
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": False,
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": False,
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": False,
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": True,
    },
    "FINAL_DISPATCH": {
        "FIXED_DEMAND_NOT_MODIFIED": True,
        "PROTECTED_KEY_NOT_MODIFIED": True,
        "MODE_ONLY_NOT_MODIFY_PROTECTED": False,
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": False,
        "DDL_READ_ONLY_DEADLINE_COMPLETION": False,
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": False,
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": True,
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": False,
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": False,
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": True,
    },
    "SERVICE_UNIT": {
        "FIXED_DEMAND_NOT_MODIFIED": True,
        "PROTECTED_KEY_NOT_MODIFIED": True,
        "MODE_ONLY_NOT_MODIFY_PROTECTED": False,
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": False,
        "DDL_READ_ONLY_DEADLINE_COMPLETION": False,
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": False,
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": False,
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": True,
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": False,
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": True,
    },
    "TAIL_ONLY_SERVICE": {
        "FIXED_DEMAND_NOT_MODIFIED": True,
        "PROTECTED_KEY_NOT_MODIFIED": True,
        "MODE_ONLY_NOT_MODIFY_PROTECTED": False,
        "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": True,
        "DDL_READ_ONLY_DEADLINE_COMPLETION": False,
        "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": False,
        "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": False,
        "SERVICE_UNIT_SINGLE_DISCRETE_RATE": False,
        "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": False,
        "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": True,
    },
}


def _is_case_obligation_trivial(case_id: str, obligation_key: str) -> bool:
    """Determine triviality from the applicability table, not text content."""
    case_applicability = CASE_OBLIGATION_APPLICABILITY.get(case_id, {})
    return case_applicability.get(obligation_key, True) is False


def check_pp0_transition_queries() -> dict[str, Any]:
    """Fresh generate IR, SMT queries, and return a validation report.

    Never trusts candidate-provided SMT or solver results.
    Triviality is determined by CASE_OBLIGATION_APPLICABILITY, not text comments.

    If compiled IR (from executable_transition_compiler) is available and
    COMPILED, the results include source_function_ast_hash and compiled_ir_hash
    in the obligation results, per the fresh verifier output format.
    """
    ir_list = build_pp0_transition_ir()
    ir_map = {ir.case_id: ir for ir in ir_list}
    queries = generate_all_queries()
    obligations = transition_obligations()

    try:
        from .executable_transition_compiler import compiled_ir_map as _compiled_ir_map
        compiled_map = _compiled_ir_map()
    except (ImportError, ValueError, TypeError):
        compiled_map = {}

    case_results: list[dict[str, Any]] = []
    all_resolved = True
    all_pass = True

    for case in CANONICAL_CASES:
        ir = ir_map.get(case.case_id)
        compiled = compiled_map.get(case.case_id)
        case_entry = {
            "case_id": case.case_id,
            "guard_fields": list(case.guard_fields),
            "read_fields": list(case.read_fields),
            "write_fields": list(case.write_fields),
            "protected_frame_fields": list(case.protected_frame_fields),
            "time_delta": case.time_delta,
            "ir_hash": ir.ir_hash if ir else None,
            "source_function": ir.source_function if ir else None,
            "source_binding_hash": ir.source_binding if ir else None,
            "compiled_ir_hash": compiled.ir_hash() if compiled and compiled.is_compiled() else None,
            "compiled_source_function_ast_hash": (
                compiled.source_function_ast_hash
                if compiled and compiled.is_compiled() else None
            ),
            "compilation_status": compiled.compilation_status if compiled else "NOT_COMPILED",
            "obligation_results": {},
        }

        for obligation_key in obligations:
            query_id = f"{case.case_id}_{obligation_key}"
            query_info = queries.get(query_id, {})
            smt2 = query_info.get("smt2_source", "")

            is_trivial = _is_case_obligation_trivial(case.case_id, obligation_key)
            code_bound = query_info.get("transition_equations_bound") is True
            compiled_available = compiled is not None and compiled.is_compiled()

            if is_trivial:
                status = "PASS"
                detail = "trivial — obligation not applicable to this case"
            elif not code_bound:
                status = "UNRESOLVED"
                detail = (
                    "The query contains free pre/post variables but no equations "
                    "binding them to the executable transition relation.  Solver "
                    "UNSAT would therefore not prove PP0 conformance."
                )
                if compiled_available:
                    detail += (
                        f" Compiled IR for case {case.case_id} exists but SMT query "
                        f"generation did not produce code-bound output."
                    )
                all_resolved = False
            else:
                status = "UNRESOLVED"
                detail = (
                    "Code-bound SMT2 query generated but not yet verified by an "
                    "integrated solver.  Solver must prove UNSAT for PASS."
                )
                all_resolved = False

            case_entry["obligation_results"][obligation_key] = {
                "query_id": query_id,
                "smt2_hash": query_info.get("smt2_hash"),
                "ir_hash": query_info.get("ir_hash"),
                "compiled_ir_hash": case_entry.get("compiled_ir_hash"),
                "source_function_ast_hash": (
                    case_entry.get("compiled_source_function_ast_hash")
                    if compiled_available else query_info.get("source_binding_hash")
                ),
                "solver_result": None,
                "transition_equations_bound": code_bound,
                "proof_scope": query_info.get("proof_scope"),
                "status": status,
                "detail": detail,
            }
            if status != "PASS":
                all_pass = False

        case_results.append(case_entry)

    payload = {
        "schema_version": "pp0_transition_schema_v2",
        "canonical_cases": canonical_case_ids(),
        "obligations": {k: v for k, v in obligations.items()},
        "case_results": case_results,
        "query_count": len(queries),
        "pass_count": sum(
            1 for cr in case_results
            for orv in cr["obligation_results"].values()
            if orv["status"] == "PASS"
        ),
        "code_bound_query_count": sum(
            1 for info in queries.values()
            if info.get("transition_equations_bound") is True
        ),
        "unresolved_count": sum(
            1 for cr in case_results
            for orv in cr["obligation_results"].values()
            if orv["status"] == "UNRESOLVED"
        ),
    }

    if all_pass:
        status = "PASS"
        code = None
    elif all_resolved:
        status = "FAIL"
        code = "PP0_TRANSITION_QUERY_FAILED"
    else:
        status = "UNRESOLVED"
        code = "PP0_TRANSITION_RELATION_NOT_BOUND" if not any(
            info.get("transition_equations_bound") is True for info in queries.values()
        ) else "PP0_TRANSITION_SOLVER_UNAVAILABLE"

    return {
        **payload,
        "status": status,
        "code": code,
        "certificate_hash": sha256_object(payload),
        "failure": None if status == "PASS" else {
            "code": code or "PP0_SCHEMA_UNRESOLVED",
            "reason": (
                "At least one primitive transition query has not been verified "
                "by an SMT solver.  AST/source-shape checks are insufficient."
            ),
        },
    }


def build_pp0_transition_certificate() -> dict[str, Any]:
    return check_pp0_transition_queries()
