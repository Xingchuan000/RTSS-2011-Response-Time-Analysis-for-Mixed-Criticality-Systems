from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def _solve_code_bound_smt2(smt2: str) -> tuple[str, str | None]:
    try:
        import z3
    except Exception as exc:
        return "UNRESOLVED", f"PP0_Z3_UNAVAILABLE:{type(exc).__name__}:{exc}"
    try:
        assertions = z3.parse_smt2_string(smt2)
        solver = z3.Solver()
        solver.add(assertions)
        result = solver.check()
    except Exception as exc:
        return "UNRESOLVED", f"PP0_SMT_PARSE_OR_SOLVE_ERROR:{type(exc).__name__}:{exc}"
    if result == z3.unsat:
        return "UNSAT", None
    if result == z3.sat:
        return "SAT", "PP0_NEGATED_OBLIGATION_SATISFIABLE"
    return "UNKNOWN", "PP0_Z3_RETURNED_UNKNOWN"


from .pp0_smt_encoder import (
    generate_code_bound_queries,
    is_trivial_query_source,
    RELATIONAL_PP0_RECEIPTS,
)


def _receipt_applicable(smt2: str) -> bool:
    return "(declare-const" in smt2 and "(assert" in smt2 and "(check-sat)" in smt2


def check_pp0_transition_queries() -> dict[str, Any]:
    queries = generate_code_bound_queries()

    receipt_results: list[dict[str, Any]] = []
    all_pass = True

    for receipt_info in RELATIONAL_PP0_RECEIPTS:
        receipt_id = receipt_info["receipt_id"]
        case_id = receipt_info["case_id"]
        query_info = queries.get(receipt_id, {})
        smt2 = query_info.get("smt2_source", "")
        code_bound = query_info.get("transition_equations_bound") is True
        applicable = _receipt_applicable(smt2)

        if not applicable:
            status = "UNRESOLVED"
            detail = "SMT2 query missing domain, assertions, or check-sat"
        elif not code_bound:
            status = "UNRESOLVED"
            detail = "Query contains free variables without code-bound transition equations"
        else:
            solver_result, solver_error = _solve_code_bound_smt2(smt2)
            if solver_result == "UNSAT":
                status = "PASS"
                detail = "Z3 proved relational PP0 obligation UNSAT"
            else:
                status = "FAIL" if solver_result in {"SAT", "UNKNOWN"} else "UNRESOLVED"
                detail = solver_error or "PP0 solver did not prove UNSAT"

        if status != "PASS":
            all_pass = False

        receipt_results.append({
            "receipt_id": receipt_id,
            "case_id": case_id,
            "smt2_hash": query_info.get("smt2_hash"),
            "ir_hash": query_info.get("ir_hash"),
            "source_function": query_info.get("source_function"),
            "source_ast_hash": query_info.get("source_ast_hash"),
            "relation_schema_hash": query_info.get("relation_schema_hash"),
            "solver_result": solver_result if code_bound and applicable else None,
            "transition_equations_bound": code_bound,
            "proof_scope": query_info.get("proof_scope"),
            "status": status,
            "detail": detail,
        })

    pass_count = sum(1 for r in receipt_results if r["status"] == "PASS")
    unresolved_count = sum(1 for r in receipt_results if r["status"] != "PASS" and r["status"] != "FAIL")
    fail_count = sum(1 for r in receipt_results if r["status"] == "FAIL")
    code_bound_query_count = sum(
        1 for r in receipt_results if r["transition_equations_bound"] is True
    )

    payload = {
        "schema_version": "pp0_relational_schema_v1",
        "receipt_count": len(receipt_results),
        "pass_count": pass_count,
        "unresolved_count": unresolved_count,
        "fail_count": fail_count,
        "code_bound_query_count": code_bound_query_count,
        "receipt_results": receipt_results,
    }

    status = "PASS" if all_pass else ("FAIL" if fail_count > 0 else "UNRESOLVED")
    code = None if status == "PASS" else (
        "PP0_RELATIONAL_QUERY_FAILED" if fail_count > 0 else "PP0_RELATIONAL_NOT_BOUND"
    )

    return {
        **payload,
        "status": status,
        "code": code,
        "certificate_hash": sha256_object(payload),
        "failure": None if status == "PASS" else {
            "code": code or "PP0_SCHEMA_UNRESOLVED",
            "reason": "At least one relational PP0 receipt has not been verified by Z3",
        },
    }


def build_pp0_transition_certificate() -> dict[str, Any]:
    return check_pp0_transition_queries()
