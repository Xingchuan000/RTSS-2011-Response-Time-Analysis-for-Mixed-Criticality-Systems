from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def _solve_with_z3_ctypes(smt2: str) -> tuple[str, str | None]:
    """Use the system Z3 shared library when z3py is unavailable.

    The fallback is deliberately narrow: it loads the same SMT-LIB2 query into
    a fresh solver and returns only SAT/UNSAT/UNKNOWN.  It does not weaken the
    proof obligation or replace universal solving with finite enumeration.
    """
    import ctypes
    import ctypes.util

    library_name = ctypes.util.find_library("z3")
    if not library_name:
        return "UNRESOLVED", "PP0_Z3_SHARED_LIBRARY_UNAVAILABLE"
    try:
        lib = ctypes.CDLL(library_name)
        void_p = ctypes.c_void_p
        lib.Z3_mk_config.restype = void_p
        lib.Z3_del_config.argtypes = [void_p]
        lib.Z3_mk_context_rc.argtypes = [void_p]
        lib.Z3_mk_context_rc.restype = void_p
        lib.Z3_del_context.argtypes = [void_p]
        lib.Z3_mk_solver.argtypes = [void_p]
        lib.Z3_mk_solver.restype = void_p
        lib.Z3_solver_inc_ref.argtypes = [void_p, void_p]
        lib.Z3_solver_dec_ref.argtypes = [void_p, void_p]
        lib.Z3_solver_from_string.argtypes = [void_p, void_p, ctypes.c_char_p]
        lib.Z3_solver_check.argtypes = [void_p, void_p]
        lib.Z3_solver_check.restype = ctypes.c_int
        lib.Z3_get_error_code.argtypes = [void_p]
        lib.Z3_get_error_code.restype = ctypes.c_uint
        lib.Z3_get_error_msg.argtypes = [void_p, ctypes.c_uint]
        lib.Z3_get_error_msg.restype = ctypes.c_char_p

        config = lib.Z3_mk_config()
        context = lib.Z3_mk_context_rc(config)
        lib.Z3_del_config(config)
        # Z3's default C error handler terminates the process on malformed
        # SMT-LIB2.  Install a no-op callback so parse errors are reported via
        # Z3_get_error_code and remain fail-closed as UNRESOLVED instead of
        # aborting the verifier process.
        error_handler_type = ctypes.CFUNCTYPE(None, void_p, ctypes.c_uint)
        lib.Z3_set_error_handler.argtypes = [void_p, error_handler_type]
        error_handler = error_handler_type(lambda _ctx, _code: None)
        lib.Z3_set_error_handler(context, error_handler)
        solver = lib.Z3_mk_solver(context)
        lib.Z3_solver_inc_ref(context, solver)
        try:
            # Solver-from-string accepts declarations/assertions.  The command
            # is removed because the API itself performs the check.
            source = "\n".join(
                line for line in smt2.splitlines()
                if line.strip() != "(check-sat)"
            )
            lib.Z3_solver_from_string(context, solver, source.encode("utf-8"))
            error_code = int(lib.Z3_get_error_code(context))
            if error_code != 0:
                message = lib.Z3_get_error_msg(context, error_code)
                detail = message.decode("utf-8", "replace") if message else str(error_code)
                return "UNRESOLVED", f"PP0_Z3_CTYPE_PARSE_ERROR:{detail}"
            result = int(lib.Z3_solver_check(context, solver))
            if result == -1:
                return "UNSAT", None
            if result == 1:
                return "SAT", "PP0_NEGATED_OBLIGATION_SATISFIABLE"
            return "UNKNOWN", "PP0_Z3_RETURNED_UNKNOWN"
        finally:
            lib.Z3_solver_dec_ref(context, solver)
            lib.Z3_del_context(context)
    except Exception as exc:
        return "UNRESOLVED", f"PP0_Z3_CTYPE_ERROR:{type(exc).__name__}:{exc}"


def _solve_code_bound_smt2(smt2: str) -> tuple[str, str | None]:
    try:
        import z3
    except Exception:
        return _solve_with_z3_ctypes(smt2)
    try:
        source = "\n".join(
            line for line in smt2.splitlines()
            if line.strip() != "(check-sat)"
        )
        assertions = z3.parse_smt2_string(source)
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
            "semantic_effect_hash": query_info.get("semantic_effect_hash"),
            "required_assumption_ids": query_info.get("required_assumption_ids", []),
            "projection_derivation_complete": query_info.get("projection_derivation_complete"),
            "full_ir_hash": query_info.get("full_ir_hash"),
            "prefix_ir_hash": query_info.get("prefix_ir_hash"),
            "pairing_kind": query_info.get("pairing_kind"),
            "all_paths_covered": query_info.get("all_paths_covered"),
            "direct_executable_encoding": query_info.get("direct_executable_encoding"),
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
