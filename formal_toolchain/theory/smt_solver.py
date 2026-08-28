"""Narrow shared SMT-LIB2 solver used by source-bound theory backends.

This module is route-neutral.  It prefers z3py and falls back to the system
libz3 shared library when Python bindings are unavailable.  It never weakens a
closed SMT obligation into finite testing.
"""
from __future__ import annotations


def _solve_with_z3_ctypes(smt2: str) -> tuple[str, str | None]:
    import ctypes
    import ctypes.util

    library_name = ctypes.util.find_library("z3")
    if not library_name:
        return "UNRESOLVED", "Z3_SHARED_LIBRARY_UNAVAILABLE"
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
        error_handler_type = ctypes.CFUNCTYPE(None, void_p, ctypes.c_uint)
        lib.Z3_set_error_handler.argtypes = [void_p, error_handler_type]
        error_handler = error_handler_type(lambda _ctx, _code: None)
        lib.Z3_set_error_handler(context, error_handler)
        solver = lib.Z3_mk_solver(context)
        lib.Z3_solver_inc_ref(context, solver)
        try:
            source = "\n".join(
                line for line in smt2.splitlines()
                if line.strip() != "(check-sat)"
            )
            lib.Z3_solver_from_string(context, solver, source.encode("utf-8"))
            error_code = int(lib.Z3_get_error_code(context))
            if error_code != 0:
                message = lib.Z3_get_error_msg(context, error_code)
                detail = message.decode("utf-8", "replace") if message else str(error_code)
                return "UNRESOLVED", f"Z3_CTYPE_PARSE_ERROR:{detail}"
            result = int(lib.Z3_solver_check(context, solver))
            if result == -1:
                return "UNSAT", None
            if result == 1:
                return "SAT", "NEGATED_OBLIGATION_SATISFIABLE"
            return "UNKNOWN", "Z3_RETURNED_UNKNOWN"
        finally:
            lib.Z3_solver_dec_ref(context, solver)
            lib.Z3_del_context(context)
    except Exception as exc:  # fail-closed backend boundary
        return "UNRESOLVED", f"Z3_CTYPE_ERROR:{type(exc).__name__}:{exc}"


def solve_closed_smt2(smt2: str) -> tuple[str, str | None]:
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
        return "UNRESOLVED", f"SMT_PARSE_OR_SOLVE_ERROR:{type(exc).__name__}:{exc}"
    if result == z3.unsat:
        return "UNSAT", None
    if result == z3.sat:
        return "SAT", "NEGATED_OBLIGATION_SATISFIABLE"
    return "UNKNOWN", "Z3_RETURNED_UNKNOWN"


__all__ = ["solve_closed_smt2"]
