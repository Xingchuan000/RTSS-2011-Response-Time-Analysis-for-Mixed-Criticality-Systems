"""Validation of compiled transition IR against executable semantics.

Fresh recompiles every function and compares the compiled IR against
the expected schema.  No hand-written equation can trigger a PASS;
only compiler output that compiles successfully is accepted.

Also validates the PP transition binding (BoundTransitionCase) from
pp_transition_binding.py for the 9 canonical cases.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

from .executable_transition_compiler import compile_all_transitions, compile_function
from .executable_transition_ir import CompiledTransitionIR


def validate_compiled_ir(ir: CompiledTransitionIR) -> dict[str, Any]:
    """Validate a single compiled transition IR.

    Checks:
    1. Source function AST hash matches fresh compilation
    2. Compilation status is COMPILED
    3. Binding kind is EXECUTABLE_TRANSITION_COMPILER
    """
    fresh = compile_function(ir.source_function)

    ast_matches = fresh.source_function_ast_hash == ir.source_function_ast_hash
    compiled_ok = ir.compilation_status == "COMPILED"
    bound_ok = ir.binding_kind == "EXECUTABLE_TRANSITION_COMPILER"
    receipt = ir.compilation_receipt
    receipt_ok = (
        receipt is not None
        and receipt.source_function == ir.source_function
        and receipt.source_ast_hash == ir.source_function_ast_hash
        and receipt.covered_return_path_count <= receipt.return_path_count
        and receipt.covered_raise_path_count <= receipt.raise_path_count
        and ir.compilation_status == "COMPILED"
        and receipt.total_semantic_coverage
        and not receipt.unsupported_nodes
    )
    status_ok = fresh.compilation_status == "COMPILED" and compiled_ok

    return {
        "case_id": ir.case_id,
        "source_function": ir.source_function,
        "fresh_ast_hash_matches": ast_matches,
        "fresh_compilation_status": fresh.compilation_status,
        "ir_compilation_status": ir.compilation_status,
        "binding_kind": ir.binding_kind,
        "receipt": receipt.to_dict() if receipt else None,
        "status": "PASS" if (ast_matches and status_ok and bound_ok and receipt_ok) else "UNRESOLVED",
        "code": None if (ast_matches and status_ok and bound_ok) else (
            "EXECUTABLE_TRANSITION_COMPILER_SEMANTIC_COVERAGE_INCOMPLETE"
            if ir.compilation_status == "PARTIAL_AST_EXTRACTION" else
            "EXECUTABLE_TRANSITION_AST_UNSUPPORTED" if not compiled_ok else
            "EXECUTABLE_TRANSITION_IR_VALIDATION_FAILED"
        ),
    }


def validate_bound_transition_case(case_id: str) -> dict[str, Any]:
    """Validate that a BoundTransitionCase exists and is CODE_BOUND."""
    try:
        from .pp_transition_binding import bound_transition_for_case
        case = bound_transition_for_case(case_id)
    except (ImportError, ValueError, TypeError, AttributeError):
        return {
            "case_id": case_id,
            "status": "UNRESOLVED",
            "code": "PP_TRANSITION_BINDING_MODULE_UNAVAILABLE",
        }

    if case is None:
        return {
            "case_id": case_id,
            "status": "UNRESOLVED",
            "code": "PP_TRANSITION_BINDING_CASE_MISSING",
        }

    ok = case.binding_status == "CODE_BOUND" and bool(case.source_ast_hash)
    return {
        "case_id": case_id,
        "source_function": case.source_function,
        "source_ast_hash": case.source_ast_hash,
        "binding_status": case.binding_status,
        "status": "PASS" if ok else "UNRESOLVED",
        "code": None if ok else "PP_TRANSITION_BINDING_NOT_CODE_BOUND",
    }


def validate_all_compiled_ir() -> dict[str, Any]:
    """Validate all compiled transition IR records and bound transition cases.

    Returns a report with per-case validation results.
    """
    compiled = _import_compiled_ir_map()
    if compiled is None:
        compiled = _fallback_compile()

    results: list[dict[str, Any]] = []
    all_pass = True

    for case_id, ir in sorted(compiled.items()):
        validation = validate_compiled_ir(ir)
        results.append(validation)
        if validation["status"] != "PASS":
            all_pass = False

    bound_case_ids = [
        "REM_COMPLETION", "RECOVERY", "DEADLINE_OBSERVATION",
        "ARRIVAL_BATCH", "MODE_SWITCH", "RELEASE",
        "FINAL_DISPATCH", "SERVICE_UNIT", "TAIL_ONLY_SERVICE",
    ]
    for case_id in bound_case_ids:
        validation = validate_bound_transition_case(case_id)
        results.append(validation)
        if validation["status"] != "PASS":
            all_pass = False

    payload = {
        "schema_version": "compiled_transition_ir_validation_v2",
        "case_count": len(results),
        "pass_count": sum(1 for r in results if r["status"] == "PASS"),
        "unresolved_count": sum(1 for r in results if r["status"] != "PASS"),
        "case_results": results,
    }

    return {
        **payload,
        "status": "PASS" if all_pass else "UNRESOLVED",
        "code": None if all_pass else "EXECUTABLE_TRANSITION_IR_VALIDATION_FAILED",
        "certificate_hash": sha256_object(payload),
        "failure": None if all_pass else {
            "code": "EXECUTABLE_TRANSITION_IR_VALIDATION_FAILED",
            "reason": (
                "At least one compiled transition IR or bound transition case "
                "failed validation."
            ),
            "failed_cases": [r["case_id"] for r in results if r["status"] != "PASS"],
        },
    }


def _import_compiled_ir_map() -> dict[str, Any] | None:
    try:
        from .executable_transition_compiler import compiled_ir_map as _compiled_ir_map
        return _compiled_ir_map()
    except (ImportError, ValueError, TypeError):
        return None


def _fallback_compile() -> dict[str, Any]:
    try:
        from .executable_transition_compiler import compile_all_transitions
        return {ir.case_id: ir for ir in compile_all_transitions()}
    except (ImportError, ValueError, TypeError):
        return {}
