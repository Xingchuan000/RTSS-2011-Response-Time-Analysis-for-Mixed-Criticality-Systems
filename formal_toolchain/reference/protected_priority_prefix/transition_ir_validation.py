"""Fresh translation validation for executable transition IR and PP0 binding."""

from __future__ import annotations

from typing import Any

from formal_toolchain.core.hashing import sha256_object

from .executable_transition_compiler import compile_all_transitions, compile_function
from .executable_transition_ir import CompiledTransitionIR


def validate_compiled_ir(ir: CompiledTransitionIR) -> dict[str, Any]:
    """Recompile the same case and compare the complete proof-oriented IR."""
    try:
        fresh = compile_function(ir.source_function, case_id=ir.case_id, fresh=True)
    except (ValueError, TypeError, OSError, SyntaxError) as exc:
        return {
            "case_id": ir.case_id,
            "source_function": ir.source_function,
            "status": "UNRESOLVED",
            "code": f"EXECUTABLE_TRANSITION_FRESH_RECOMPILE_FAILED:{type(exc).__name__}:{exc}",
        }

    effect_hash = ir.semantic_effect.effect_hash() if ir.semantic_effect else None
    fresh_effect_hash = fresh.semantic_effect.effect_hash() if fresh.semantic_effect else None
    checks = {
        "source_ast_hash": fresh.source_function_ast_hash == ir.source_function_ast_hash,
        "compiled_status": ir.is_compiled() and fresh.is_compiled(),
        "complete_ir_hash": fresh.ir_hash() == ir.ir_hash(),
        "path_hashes": tuple(path.path_hash() for path in fresh.paths)
        == tuple(path.path_hash() for path in ir.paths),
        "helper_summary_hashes": fresh.helper_summary_hashes == ir.helper_summary_hashes,
        "semantic_effect_hash": fresh_effect_hash == effect_hash and effect_hash is not None,
        "projection_derivation_complete": bool(
            ir.semantic_effect and fresh.semantic_effect
            and ir.semantic_effect.derivation_complete
            and fresh.semantic_effect.derivation_complete
            and ir.semantic_effect.concrete_write_targets
            == ir.semantic_effect.covered_concrete_write_targets
            and fresh.semantic_effect.concrete_write_targets
            == fresh.semantic_effect.covered_concrete_write_targets
        ),
        "return_paths": ir.covered_return_path_count == ir.return_path_count,
        "raise_paths": ir.covered_raise_path_count == ir.raise_path_count,
        "binding_kind": ir.binding_kind == "EXECUTABLE_TRANSITION_COMPILER",
    }
    passed = all(checks.values())
    return {
        "case_id": ir.case_id,
        "source_function": ir.source_function,
        "source_ast_hash": ir.source_function_ast_hash,
        "fresh_ir_hash": fresh.ir_hash(),
        "ir_hash": ir.ir_hash(),
        "semantic_effect_hash": effect_hash,
        "checks": checks,
        "receipt": ir.compilation_receipt.to_dict() if ir.compilation_receipt else None,
        "status": "PASS" if passed else "UNRESOLVED",
        "code": None if passed else "EXECUTABLE_TRANSITION_TRANSLATION_VALIDATION_FAILED",
    }


def validate_bound_transition_case(case_id: str) -> dict[str, Any]:
    try:
        from .pp_transition_binding import bound_transition_for_case
        case = bound_transition_for_case(case_id)
    except (ImportError, ValueError, TypeError, AttributeError) as exc:
        return {
            "case_id": case_id,
            "status": "UNRESOLVED",
            "code": f"PP_TRANSITION_BINDING_MODULE_UNAVAILABLE:{type(exc).__name__}:{exc}",
        }
    if case is None:
        return {
            "case_id": case_id,
            "status": "UNRESOLVED",
            "code": "PP_TRANSITION_BINDING_CASE_MISSING",
        }
    compiled = {item.case_id: item for item in compile_all_transitions()}.get(case_id)
    ok = bool(
        compiled is not None
        and compiled.is_compiled()
        and case.binding_status == "CODE_BOUND"
        and case.source_ast_hash == compiled.source_function_ast_hash
        and case.compiled_ir_hash == compiled.ir_hash()
        and case.semantic_effect_hash
        == (compiled.semantic_effect.effect_hash() if compiled.semantic_effect else None)
        and case.path_hashes == tuple(path.path_hash() for path in compiled.paths)
        and case.required_assumption_ids
        == (compiled.semantic_effect.required_assumption_ids if compiled.semantic_effect else ())
        and case.projection_derivation_complete
        and case.total_semantic_coverage
    )
    return {
        "case_id": case_id,
        "source_function": case.source_function,
        "source_ast_hash": case.source_ast_hash,
        "binding_status": case.binding_status,
        "compiled_ir_hash": case.compiled_ir_hash,
        "semantic_effect_hash": case.semantic_effect_hash,
        "path_count": len(case.path_hashes),
        "required_assumption_ids": list(case.required_assumption_ids),
        "projection_derivation_complete": case.projection_derivation_complete,
        "status": "PASS" if ok else "UNRESOLVED",
        "code": None if ok else "PP_TRANSITION_BINDING_NOT_CODE_BOUND",
    }


def validate_all_compiled_ir() -> dict[str, Any]:
    compiled = tuple(compile_all_transitions())
    results = [validate_compiled_ir(item) for item in compiled]
    results.extend(validate_bound_transition_case(item.case_id) for item in compiled)
    all_pass = bool(results) and all(item.get("status") == "PASS" for item in results)
    payload = {
        "schema_version": "compiled_transition_ir_validation_v3",
        "executable_case_count": len(compiled),
        "case_count": len(results),
        "pass_count": sum(item.get("status") == "PASS" for item in results),
        "unresolved_count": sum(item.get("status") != "PASS" for item in results),
        "case_results": results,
    }
    return {
        **payload,
        "status": "PASS" if all_pass else "UNRESOLVED",
        "code": None if all_pass else "EXECUTABLE_TRANSITION_IR_VALIDATION_FAILED",
        "certificate_hash": sha256_object(payload),
        "failure": None if all_pass else {
            "code": "EXECUTABLE_TRANSITION_IR_VALIDATION_FAILED",
            "failed_cases": [item.get("case_id") for item in results if item.get("status") != "PASS"],
        },
    }
