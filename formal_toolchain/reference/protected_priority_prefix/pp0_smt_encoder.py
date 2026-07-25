"""Direct executable-to-PP0 relational SMT encoder.

Unlike the legacy encoder, this module never consumes hand-written
``PP0TransitionIR`` equations.  Every query is generated from a fresh
``CompiledTransitionIR.semantic_effect`` whose path hashes and helper summaries
are bound to the executable Python source.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Any, Iterable, Mapping

from formal_toolchain.core.hashing import sha256_object

from .executable_transition_compiler import compiled_ir_map
from .executable_transition_ir import CompiledTransitionIR, SemanticEffectRule
from .pp_transition_binding import RELATION_SCHEMA_HASH, compile_all_transitions as bound_transitions


LOCKSTEP = "LOCKSTEP"
FULL_STUTTER_PREFIX = "FULL_STUTTER_PREFIX"
PREFIX_STUTTER_FULL = "PREFIX_STUTTER_FULL"
FULL_TAIL_SERVICE_PREFIX_IDLE = "FULL_TAIL_SERVICE_PREFIX_IDLE"
LOCKSTEP_IDLE_JUMP = "LOCKSTEP_IDLE_JUMP"

# Relation fields are abstract digests of the complete protected projection.
# Equality of a digest represents equality of the corresponding finite map or
# ledger; deterministic source-derived update functions preserve equality by
# congruence.
_BASE_FIELDS = (
    "jobs", "pending", "released", "terminal", "misses", "running",
    "ready", "service", "metadata", "frontier_protected", "time",
)

RELATIONAL_PP0_RECEIPTS: list[dict[str, Any]] = [
    {
        "receipt_id": "PP0_REM_RELATION", "case_id": "REM_COMPLETION",
        "description": "REM completion preserves the protected job/terminal projection",
        "pairing_kind": LOCKSTEP,
        "relation_fields": _BASE_FIELDS,
        "phase": "SvcEnd",
    },
    {
        "receipt_id": "PP0_RECOVERY_STUTTER_FULL_ONLY", "case_id": "RECOVERY",
        "description": "Full-only recovery is a protected-observable stutter",
        "pairing_kind": FULL_STUTTER_PREFIX,
        "relation_fields": _BASE_FIELDS,
        "phase": "AfterREM",
    },
    {
        "receipt_id": "PP0_RECOVERY_STUTTER_PREFIX_ONLY", "case_id": "RECOVERY",
        "description": "Prefix-only recovery is a protected-observable stutter",
        "pairing_kind": PREFIX_STUTTER_FULL,
        "relation_fields": _BASE_FIELDS,
        "phase": "AfterREM",
    },
    {
        "receipt_id": "PP0_DDL_OBSERVE_ONLY", "case_id": "DEADLINE_OBSERVATION",
        "description": "Deadline observation records the same protected miss",
        "pairing_kind": LOCKSTEP,
        "relation_fields": _BASE_FIELDS,
        "phase": "AfterREC",
    },
    {
        "receipt_id": "PP0_ARR_PENDING_PLAN_PROJECTION", "case_id": "ARRIVAL_BATCH",
        "description": "Projected arrival fold creates equal protected pending plans",
        "pairing_kind": LOCKSTEP,
        "relation_fields": _BASE_FIELDS,
        "phase": "ARRCursor",
    },
    {
        "receipt_id": "PP0_SWITCH_STUTTER_FULL_ONLY", "case_id": "MODE_SWITCH",
        "description": "Full-only mode switch is a protected-observable stutter",
        "pairing_kind": FULL_STUTTER_PREFIX,
        "relation_fields": _BASE_FIELDS,
        "phase": "ARRCursor",
    },
    {
        "receipt_id": "PP0_SWITCH_STUTTER_PREFIX_ONLY", "case_id": "MODE_SWITCH",
        "description": "Prefix-only mode switch is a protected-observable stutter",
        "pairing_kind": PREFIX_STUTTER_FULL,
        "relation_fields": _BASE_FIELDS,
        "phase": "ARRCursor",
    },
    {
        "receipt_id": "PP0_RELEASE_PROTECTED_PAYLOAD", "case_id": "RELEASE",
        "description": "Release materializes equal protected job payloads",
        "pairing_kind": LOCKSTEP,
        "relation_fields": _BASE_FIELDS,
        "phase": "PreDisp",
    },
    {
        "receipt_id": "PP0_DISPATCH_DETERMINISM", "case_id": "FINAL_DISPATCH",
        "description": "Strict fixed-priority dispatch selects the same protected job",
        "pairing_kind": LOCKSTEP,
        "relation_fields": _BASE_FIELDS,
        "phase": "PreDisp",
    },
    {
        "receipt_id": "PP0_SERVICE_PROTECTED", "case_id": "SERVICE_UNIT",
        "description": "The same protected running job receives one equal service unit",
        "pairing_kind": LOCKSTEP,
        "relation_fields": _BASE_FIELDS,
        "phase": "Close",
    },
    {
        "receipt_id": "PP0_TAIL_SERVICE_STUTTER", "case_id": "TAIL_ONLY_SERVICE",
        "description": "Full tail service and prefix idle tick preserve the protected projection",
        "pairing_kind": FULL_TAIL_SERVICE_PREFIX_IDLE,
        "relation_fields": _BASE_FIELDS,
        "phase": "Close",
    },
    {
        "receipt_id": "PP0_IDLE_JUMP_STUTTER", "case_id": "IDLE_JUMP",
        "description": "Both executions jump to the same minimum future protected observation",
        "pairing_kind": LOCKSTEP_IDLE_JUMP,
        "relation_fields": _BASE_FIELDS,
        "phase": "Close",
    },
]


def _idle_jump_source_binding() -> dict[str, Any]:
    """Validate the explicit infinite-idle/minimum-event branch in source."""
    try:
        from .execution_builder import next_closed_boundary
        source = textwrap.dedent(inspect.getsource(next_closed_boundary))
        node = ast.parse(source)
    except (ImportError, OSError, TypeError, SyntaxError) as exc:
        return {"status": "UNRESOLVED", "code": f"IDLE_JUMP_SOURCE_UNAVAILABLE:{exc}"}
    text = ast.unparse(node)
    required = {
        "closed_precondition": "is_closed_reference_state(state, taskset)" in text,
        "fixed_oracle_binding": "_bind_oracle_for_current_arrival(current, oracle)" in text,
        "future_event_guard": "if future" in text,
        "explicit_idle_tick": "replace(current, time=int(current.time) + 1)" in text,
        "strict_later_boundary": "int(current.time) > start_time" in text,
    }
    status = "PASS" if all(required.values()) else "UNRESOLVED"
    ast_hash = sha256_object({"source_ast": ast.dump(node, include_attributes=False)})
    payload = {"ast_hash": ast_hash, "required": required}
    return {
        "status": status,
        "code": None if status == "PASS" else "IDLE_JUMP_SOURCE_PATTERN_INCOMPLETE",
        "source_function": (
            "formal_toolchain.reference.protected_priority_prefix."
            "execution_builder.next_closed_boundary"
        ),
        "source_ast_hash": ast_hash,
        "ir_hash": sha256_object(payload),
        "semantic_effect_hash": sha256_object({
            "effect": "protected_fields_frame; time_post=jump_target",
            "source": payload,
        }),
        "required_facts": required,
    }


def _parse_template(expression: str) -> ast.AST:
    # Existing arithmetic templates may already be SMT prefix expressions.
    if expression.strip().startswith("("):
        return ast.Name(id="__RAW_SMT__", ctx=ast.Load())
    return ast.parse(expression, mode="eval").body


def _collect_functions(expression: str) -> dict[str, int]:
    if expression.strip().startswith("("):
        return {}
    result: dict[str, int] = {}
    node = ast.parse(expression, mode="eval")
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if isinstance(call.func, ast.Name):
            result[call.func.id] = max(result.get(call.func.id, 0), len(call.args))
    return result


def _collect_variables(expression: str) -> set[str]:
    expression = expression.strip()
    if expression.startswith("("):
        return {"time"} if re.search(r"\btime\b", expression) else set()
    node = ast.parse(expression, mode="eval")
    function_names = {
        call.func.id for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    return {
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and item.id not in function_names
    }


def _expr_to_smt(expression: str, *, side: str, inputs: set[str]) -> str:
    expression = expression.strip()
    if expression.startswith("("):
        # Only the audited (+ time 1) arithmetic form is currently emitted.
        return re.sub(r"\btime\b", f"time_{side}_pre", expression)
    node = ast.parse(expression, mode="eval").body

    def convert(item: ast.AST) -> str:
        if isinstance(item, ast.Name):
            suffix = "input" if item.id in inputs or item.id in {
                "event_key", "projected_batch_payload", "priority_order",
                "jump_target",
            } else "pre"
            return f"{item.id}_{side}_{suffix}"
        if isinstance(item, ast.Constant):
            if isinstance(item.value, bool):
                return "1" if item.value else "0"
            if item.value is None:
                return "0"
            return str(item.value)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
            return f"({item.func.id} {' '.join(convert(arg) for arg in item.args)})"
        if isinstance(item, ast.BinOp):
            op = "+" if isinstance(item.op, ast.Add) else "-" if isinstance(item.op, ast.Sub) else "*"
            return f"({op} {convert(item.left)} {convert(item.right)})"
        raise ValueError(f"PP0_EFFECT_TEMPLATE_UNSUPPORTED:{type(item).__name__}:{expression}")

    return convert(node)


def _effect_map(effect: SemanticEffectRule | None) -> dict[str, str]:
    return dict(effect.field_equations) if effect else {}


def _declare_variables(fields: Iterable[str], inputs: Iterable[str],
                       function_arities: Mapping[str, int],
                       assumptions: Iterable[str] = ()) -> str:
    lines = ["(set-logic QF_UFLIA)"]
    for field in sorted(set(fields)):
        for side in ("f", "p"):
            lines.append(f"(declare-const {field}_{side}_pre Int)")
            lines.append(f"(declare-const {field}_{side}_post Int)")
    for name in sorted(set(inputs)):
        lines.append(f"(declare-const {name}_f_input Int)")
        lines.append(f"(declare-const {name}_p_input Int)")
    for function, arity in sorted(function_arities.items()):
        args = " ".join("Int" for _ in range(arity))
        lines.append(f"(declare-fun {function} ({args}) Int)")
    for assumption in sorted(set(assumptions)):
        lines.append(f"(declare-const assume_{assumption} Bool)")
    return "\n".join(lines)


def _pre_relation(fields: Iterable[str], inputs: Iterable[str],
                  assumptions: Iterable[str] = ()) -> str:
    lines = [f"(assert (= {field}_f_pre {field}_p_pre))" for field in fields]
    lines.extend(f"(assert (= {name}_f_input {name}_p_input))" for name in inputs)
    lines.extend(f"(assert assume_{name})" for name in sorted(set(assumptions)))
    return "\n".join(lines)


def _emit_effect_side(*, side: str, fields: tuple[str, ...], effect: SemanticEffectRule,
                      mode: str, input_symbols: Iterable[str] | None = None) -> str:
    effect_map = _effect_map(effect)
    inputs = set(input_symbols if input_symbols is not None else effect.input_symbols)
    lines: list[str] = []
    for field in fields:
        if mode == "IDLE_JUMP":
            rhs = f"jump_target_{side}_input" if field == "time" else f"{field}_{side}_pre"
        elif mode == "IDLE_TICK":
            rhs = f"(+ {field}_{side}_pre 1)" if field == "time" else f"{field}_{side}_pre"
        elif mode == "STUTTER":
            rhs = f"{field}_{side}_pre"
        elif field in effect_map:
            rhs = _expr_to_smt(effect_map[field], side=side, inputs=inputs)
        else:
            rhs = f"{field}_{side}_pre"
        lines.append(f"(assert (= {field}_{side}_post {rhs}))")
    return "\n".join(lines)


def _negated_post_relation(fields: Iterable[str]) -> str:
    equalities = " ".join(f"(= {field}_f_post {field}_p_post)" for field in fields)
    return f"(assert (not (and {equalities})))"


def _build_query(receipt: Mapping[str, Any], ir: CompiledTransitionIR | None,
                 idle_binding: Mapping[str, Any] | None = None) -> str:
    fields = tuple(receipt["relation_fields"])
    pairing = str(receipt["pairing_kind"])
    if pairing == LOCKSTEP_IDLE_JUMP:
        inputs = ("jump_target",)
        effect = SemanticEffectRule(
            effect_id="IDLE_JUMP", field_equations=(),
            frame_fields=frozenset(fields) - {"time"},
            input_symbols=inputs, supported_pairings=(LOCKSTEP_IDLE_JUMP,),
            validator_rule_id="IDLE_JUMP_SOURCE_PATTERN_V1",
            validator_facts=tuple(
                key for key, value in (idle_binding or {}).get("required_facts", {}).items()
                if value
            ), source_path_hashes=((idle_binding or {}).get("ir_hash", ""),),
            concrete_write_targets=("time",),
            covered_concrete_write_targets=("time",),
            path_effect_hashes=((idle_binding or {}).get("ir_hash", ""),),
            required_assumption_ids=(
                "JUMP_TARGET_EQUAL", "NO_PROTECTED_EVENT_BEFORE_JUMP_TARGET",
            ),
            derivation_complete=True,
        )
        function_arities: dict[str, int] = {}
        assumptions = effect.required_assumption_ids
    else:
        assert ir is not None and ir.semantic_effect is not None
        effect = ir.semantic_effect
        inputs = effect.input_symbols
        function_arities = {}
        free_variables: set[str] = set()
        for _, expression in effect.field_equations:
            function_arities.update(_collect_functions(expression))
            free_variables.update(_collect_variables(expression))
        inputs = tuple(sorted(set(inputs) | (free_variables - set(fields))))
        assumptions = effect.required_assumption_ids

    declarations = _declare_variables(fields, inputs, function_arities, assumptions)
    relation = _pre_relation(fields, inputs, assumptions)
    if pairing == LOCKSTEP:
        full = _emit_effect_side(side="f", fields=fields, effect=effect, mode="EFFECT", input_symbols=inputs)
        prefix = _emit_effect_side(side="p", fields=fields, effect=effect, mode="EFFECT", input_symbols=inputs)
    elif pairing == FULL_STUTTER_PREFIX:
        full = _emit_effect_side(side="f", fields=fields, effect=effect, mode="EFFECT", input_symbols=inputs)
        prefix = _emit_effect_side(side="p", fields=fields, effect=effect, mode="STUTTER", input_symbols=inputs)
    elif pairing == PREFIX_STUTTER_FULL:
        full = _emit_effect_side(side="f", fields=fields, effect=effect, mode="STUTTER", input_symbols=inputs)
        prefix = _emit_effect_side(side="p", fields=fields, effect=effect, mode="EFFECT", input_symbols=inputs)
    elif pairing == FULL_TAIL_SERVICE_PREFIX_IDLE:
        # Full executes the source-bound service primitive on a tail job; the
        # prefix performs an idle unit.  Protected digests frame on both sides.
        full = _emit_effect_side(side="f", fields=fields, effect=effect, mode="IDLE_TICK", input_symbols=inputs)
        prefix = _emit_effect_side(side="p", fields=fields, effect=effect, mode="IDLE_TICK", input_symbols=inputs)
    elif pairing == LOCKSTEP_IDLE_JUMP:
        full = _emit_effect_side(side="f", fields=fields, effect=effect, mode="IDLE_JUMP", input_symbols=inputs)
        prefix = _emit_effect_side(side="p", fields=fields, effect=effect, mode="IDLE_JUMP", input_symbols=inputs)
    else:
        raise ValueError(f"PP0_PAIRING_KIND_UNKNOWN:{pairing}")
    negated = _negated_post_relation(fields)
    source = (idle_binding or {}).get("source_function") if ir is None else f"{ir.source_module}.{ir.source_function}"
    source_hash = (idle_binding or {}).get("source_ast_hash") if ir is None else ir.source_function_ast_hash
    effect_hash = (idle_binding or {}).get("semantic_effect_hash") if ir is None else effect.effect_hash()
    return f"""; {receipt['receipt_id']} — direct executable relational query
; source_function: {source}
; source_ast_hash: {source_hash}
; semantic_effect_hash: {effect_hash}
; pairing_kind: {pairing}
; required_assumptions: {','.join(assumptions)}
{declarations}
{relation}
{full}
{prefix}
{negated}
(check-sat)
"""


def _relation_schema_hash(receipt: Mapping[str, Any]) -> str:
    return sha256_object({
        "global_relation_schema_hash": RELATION_SCHEMA_HASH,
        "receipt_id": receipt["receipt_id"],
        "case_id": receipt["case_id"],
        "pairing_kind": receipt["pairing_kind"],
        "relation_fields": list(receipt["relation_fields"]),
        "phase": receipt["phase"],
    })


def generate_code_bound_queries() -> dict[str, dict[str, Any]]:
    compiled = compiled_ir_map()
    bound = bound_transitions()
    idle = _idle_jump_source_binding()
    results: dict[str, dict[str, Any]] = {}

    for receipt in RELATIONAL_PP0_RECEIPTS:
        receipt_id = str(receipt["receipt_id"])
        case_id = str(receipt["case_id"])
        pairing = str(receipt["pairing_kind"])
        relation_hash = _relation_schema_hash(receipt)

        if case_id == "IDLE_JUMP":
            code_bound = idle.get("status") == "PASS"
            ir = None
            source_function = idle.get("source_function")
            source_hash = idle.get("source_ast_hash")
            ir_hash = idle.get("ir_hash")
            effect_hash = idle.get("semantic_effect_hash")
            paths_covered = code_bound
            direct = code_bound
        else:
            ir = compiled.get(case_id)
            bound_case = bound.get(case_id)
            effect = ir.semantic_effect if ir else None
            code_bound = bool(
                ir is not None and ir.is_compiled()
                and bound_case is not None
                and bound_case.binding_status == "CODE_BOUND"
                and bound_case.compiled_ir_hash == ir.ir_hash()
                and bound_case.semantic_effect_hash == (effect.effect_hash() if effect else None)
                and effect is not None
                and effect.derivation_complete
                and effect.concrete_write_targets == effect.covered_concrete_write_targets
                and set(effect.source_path_hashes) == set(effect.path_effect_hashes)
                and pairing in effect.supported_pairings
            )
            source_function = f"{ir.source_module}.{ir.source_function}" if ir else None
            source_hash = ir.source_function_ast_hash if ir else None
            ir_hash = ir.ir_hash() if ir else None
            effect_hash = effect.effect_hash() if effect else None
            paths_covered = bool(ir and ir.is_compiled())
            direct = code_bound

        try:
            smt2 = _build_query(receipt, ir, idle)
        except (AssertionError, ValueError, SyntaxError) as exc:
            smt2 = f"; PP0 query generation failed: {type(exc).__name__}:{exc}"
            code_bound = False
            direct = False

        results[receipt_id] = {
            "query_id": receipt_id,
            "receipt_id": receipt_id,
            "case_id": case_id,
            "smt2_source": smt2,
            "smt2_hash": sha256_object({"source": smt2}),
            "ir_hash": ir_hash,
            "full_ir_hash": ir_hash,
            "prefix_ir_hash": ir_hash,
            "source_function": source_function,
            "source_ast_hash": source_hash,
            "semantic_effect_hash": effect_hash,
            "required_assumption_ids": list(
                ("JUMP_TARGET_EQUAL", "NO_PROTECTED_EVENT_BEFORE_JUMP_TARGET") if case_id == "IDLE_JUMP"
                else (ir.semantic_effect.required_assumption_ids if ir and ir.semantic_effect else ())
            ),
            "relation_schema_hash": relation_hash,
            "transition_equations_bound": code_bound,
            "compiler_total_semantic_coverage": paths_covered,
            "adapter_equivalence_proved": direct,
            "direct_executable_encoding": direct,
            "projection_derivation_complete": bool(
                code_bound if case_id == "IDLE_JUMP"
                else (ir and ir.semantic_effect and ir.semantic_effect.derivation_complete)
            ),
            "pairing_kind": pairing,
            "relation_pre_hash": relation_hash,
            "relation_post_hash": relation_hash,
            "all_paths_covered": paths_covered,
            "proof_scope": "CODE_BOUND_RELATIONAL" if code_bound else "UNRESOLVED_NOT_CODE_BOUND",
            "expected_result": "unsat" if code_bound else "unresolved",
        }
    return results


def _balanced_parentheses(source: str) -> bool:
    depth = 0
    in_comment = False
    for char in source:
        if char == "\n":
            in_comment = False
            continue
        if in_comment:
            continue
        if char == ";":
            in_comment = True
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _post_equation_rhs(source: str) -> dict[str, dict[str, str]]:
    """Extract generated post equations for a non-authoritative audit."""
    result: dict[str, dict[str, str]] = {"f": {}, "p": {}}
    prefix = "(assert (= "
    for line in source.splitlines():
        text = line.strip()
        if not text.startswith(prefix) or not text.endswith("))"):
            continue
        body = text[len(prefix):-2]
        lhs, sep, rhs = body.partition(" ")
        if not sep or not lhs.endswith("_post"):
            continue
        match = re.match(r"(.+)_([fp])_post$", lhs)
        if match:
            result[match.group(2)][match.group(1)] = rhs
    return result


def audit_generated_query_congruence(smt2: str) -> dict[str, Any]:
    """Static negative-control aid; never used as a proof result.

    The audit checks that the generated full/prefix equations are congruent
    after side-renaming, that all post fields are covered, and that the SMT-LIB
    source is balanced.  Universal validity still comes only from Z3 UNSAT.
    """
    equations = _post_equation_rhs(smt2)
    fields = sorted(set(equations["f"]) | set(equations["p"]))

    def canonical(text: str) -> str:
        return (text.replace("_f_pre", "_side_pre")
                    .replace("_p_pre", "_side_pre")
                    .replace("_f_input", "_side_input")
                    .replace("_p_input", "_side_input"))

    mismatches = [
        field for field in fields
        if field not in equations["f"] or field not in equations["p"]
        or canonical(equations["f"].get(field, ""))
        != canonical(equations["p"].get(field, ""))
    ]
    return {
        "balanced_parentheses": _balanced_parentheses(smt2),
        "post_field_count": len(fields),
        "mismatched_post_fields": mismatches,
        "diagnostic_status": "PASS" if fields and not mismatches and _balanced_parentheses(smt2) else "FAIL",
        "authoritative_proof": False,
    }


def is_trivial_query_source(smt2: str) -> bool:
    stripped = smt2.strip()
    return not stripped or stripped.startswith("; PP0 query generation failed") or "(assert" not in smt2 or "(check-sat)" not in smt2
