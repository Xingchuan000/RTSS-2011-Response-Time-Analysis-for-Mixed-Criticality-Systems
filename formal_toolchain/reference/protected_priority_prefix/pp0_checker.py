"""PP0 runtime schema checker — validates primitive transition equations.

Replaces the old AST/source-shape checks with parameterized queries that
verify actual transition equations over all reachable full/prefix states.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

from .transition_schema import (
    CANONICAL_CASES,
    canonical_case_ids,
    transition_obligations,
)
from .pp0_queries import generate_all_queries, is_trivial_query_source


def check_pp0_transition_queries() -> dict[str, Any]:
    """Generate all primitive transition queries and return a validation report.

    Each query must be independently solvable (UNSAT for safety properties)
    to prove the PP0 runtime schema obligations.

    Returns UNRESOLVED when solver results are not yet available (Z3 not
    integrated or solver verification pending).
    """
    queries = generate_all_queries()
    obligations = transition_obligations()

    case_results: list[dict[str, Any]] = []
    all_resolved = True
    all_pass = True

    for case in CANONICAL_CASES:
        case_entry = {
            "case_id": case.case_id,
            "guard_fields": list(case.guard_fields),
            "read_fields": list(case.read_fields),
            "write_fields": list(case.write_fields),
            "protected_frame_fields": list(case.protected_frame_fields),
            "time_delta": case.time_delta,
            "obligation_results": {},
        }

        for obligation_key in obligations:
            query_id = f"{case.case_id}_{obligation_key}"
            query_info = queries.get(query_id, {})
            smt2 = query_info.get("smt2_source", "")
            is_trivial = is_trivial_query_source(smt2)

            code_bound = query_info.get("transition_equations_bound") is True
            if is_trivial:
                status = "PASS"
                detail = "trivial — not in transition scope"
            elif not code_bound:
                status = "UNRESOLVED"
                detail = (
                    "The query contains free pre/post variables but no equations "
                    "binding them to the executable transition relation.  Solver "
                    "UNSAT would therefore not prove PP0 conformance."
                )
                all_resolved = False
            else:
                status = "UNRESOLVED"
                detail = (
                    "Code-bound SMT2 query generated but not yet verified by an "
                    "integrated solver."
                )
                all_resolved = False

            case_entry["obligation_results"][obligation_key] = {
                "query_id": query_id,
                "smt2_hash": query_info.get("smt2_hash"),
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
        "schema_version": "pp0_transition_schema_v1",
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
    """Build a PP0 runtime schema conformance certificate.

    This replaces build_runtime_schema_certificate() from runtime_schema.py.
    The old AST-based checks are no longer accepted as proof of PP0 conformance.
    """
    return check_pp0_transition_queries()
