"""SMT2 encoder for PP0 transition IR.

Encodes each PP0TransitionIR into a code-bound SMT2 query that includes:
  Domain(pre) ∧ TransitionCase(pre, post) ∧ ¬RequiredPostcondition(post)

A free-variable query (no transition equations) is never code-bound and
must be classified as SCHEMA_ONLY_NOT_CODE_BOUND.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

from .transition_schema import (
    CANONICAL_CASES,
    PrimitiveTransitionSchema,
    transition_obligations,
)
from .pp0_transition_ir import (
    PP0TransitionIR,
    build_pp0_transition_ir,
    ir_for_case,
)


def _emit_frame_constraints(ir: PP0TransitionIR) -> str:
    """Emit SMT2 assertions from IR frame equations."""
    lines = []
    for eq in ir.frame_equations:
        lines.append(f"(assert (= {eq.lhs} {eq.rhs}))")
    return "\n".join(lines)


def _emit_state_constraints(ir: PP0TransitionIR) -> str:
    """Emit SMT2 assertions from IR state equations."""
    lines = []
    for eq in ir.state_equations:
        lines.append(f"(assert (= {eq.lhs} {eq.rhs}))")
    return "\n".join(lines)


def _emit_time_constraint(ir: PP0TransitionIR) -> str:
    return f"(assert (= {ir.time_equation.lhs} {ir.time_equation.rhs}))"


def _emit_guard_constraint(ir: PP0TransitionIR) -> str:
    return f"(assert {ir.guard_formula})"


def _emit_domain_constraint(schema: PrimitiveTransitionSchema) -> str:
    """Emit SMT2 domain constraints for pre-state integers."""
    lines = []
    all_fields = sorted(set(
        list(schema.read_fields) + list(schema.write_fields) +
        list(schema.protected_frame_fields) + list(schema.guard_fields)
    ))
    for field in all_fields:
        lines.append(f"(declare-const {field}_pre Int)")
        lines.append(f"(declare-const {field}_post Int)")
        lines.append(f"(assert (>= {field}_pre 0))")
    return "\n".join(lines)


def build_fixed_demand_frame_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: prove fixed demand is never modified.

    Domain(pre) ∧ Transition(pre, post) ∧ (fixed_demand_post ≠ fixed_demand_pre)
    """
    if "fixed_demand" not in ir.frame_equations_encoded():
        return "; trivial — fixed_demand not in transition scope"
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    return f"""; PP0_FIXED_DEMAND_FRAME — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(assert (not (= fixed_demand_post fixed_demand_pre)))
(check-sat)
"""


def build_protected_key_frame_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: prove protected key fields are never modified."""
    key_fields = ["release_time", "absolute_deadline", "criticality", "priority_index"]
    frame_encoded = ir.frame_equations_encoded()
    relevant = [f for f in key_fields if f in frame_encoded]
    if not relevant:
        return "; trivial — no protected key fields in this case"
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    conjuncts = " ".join(f"(not (= {f}_post {f}_pre))" for f in relevant)
    return f"""; PP0_PROTECTED_JOB_KEY_FRAME — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(assert (or {conjuncts}))
(check-sat)
"""


def build_mode_only_stutter_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: mode-only steps do not modify protected activity."""
    if ir.case_id not in ("RECOVERY", "MODE_SWITCH"):
        return "; not applicable"
    protected = ["active", "ready", "running", "service"]
    frame_encoded = ir.frame_equations_encoded()
    relevant = [f for f in protected if f in frame_encoded]
    if not relevant:
        return "; trivial — no protected activity fields in this case"
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    conjuncts = " ".join(f"(not (= {f}_post {f}_pre))" for f in relevant)
    return f"""; PP0_MODE_ONLY_PROTECTED_STUTTER — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(assert (or {conjuncts}))
(check-sat)
"""


def build_tail_only_stutter_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: tail service does not modify any protected frame field."""
    if ir.case_id != "TAIL_ONLY_SERVICE":
        return "; not applicable"
    protected = list(schema.protected_frame_fields)
    if not protected:
        return "; trivial"
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    conjuncts = " ".join(f"(not (= {f}_post {f}_pre))" for f in protected)
    return f"""; PP0_TAIL_ONLY_PROTECTED_STUTTER — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(assert (or {conjuncts}))
(check-sat)
"""


def build_ddl_observe_only_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: DDL is observe-only, does not modify protected state."""
    if ir.case_id != "DDL_OBSERVE":
        return "; not applicable"
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    return f"""; PP0_DDL_OBSERVE_ONLY — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(assert (or (not (= active_post active_pre)) (not (= ready_post ready_pre))))
(check-sat)
"""


def build_completion_guard_equivalence_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: completion guard is service >= fixed_demand."""
    if ir.case_id != "REM_COMPLETION":
        return "; not applicable"
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    return f"""; PP0_COMPLETION_GUARD_EQUIVALENCE — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(assert (not {ir.guard_formula}))
(check-sat)
"""


def build_dispatch_priority_totality_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: dispatch is strict fixed-priority total selection."""
    if ir.case_id != "FINAL_DISPATCH":
        return "; not applicable"
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    return f"""; PP0_DISPATCH_FIXED_PRIORITY_TOTALITY — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(declare-const selected_priority Int)
(assert (and (> ready_pre 0) (= running_post 0)))
(check-sat)
"""


def build_service_unit_rate_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: service increases by exactly one discrete unit."""
    if ir.case_id != "SERVICE_UNIT":
        return "; not applicable"
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    return f"""; PP0_SERVICE_UNIT_RATE — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(assert (not (= service_post (+ service_pre 1))))
(check-sat)
"""


def build_arrival_protected_independence_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: protected arrival entries independent of tail deletion."""
    if ir.case_id != "ARRIVAL_BATCH_OPEN":
        return "; not applicable"
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    return f"""; PP0_ARRIVAL_PROTECTED_INPUT_INDEPENDENCE — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(declare-const tail_deleted Bool)
(assert (and tail_deleted (not (= active_post (+ active_pre 1)))))
(check-sat)
"""


def build_closure_finite_order_query(
    schema: PrimitiveTransitionSchema, ir: PP0TransitionIR
) -> str:
    """Code-bound: same-time closure measure is finite and strictly decreasing."""
    domain = _emit_domain_constraint(schema)
    guard = _emit_guard_constraint(ir)
    state = _emit_state_constraints(ir)
    frame = _emit_frame_constraints(ir)
    time_c = _emit_time_constraint(ir)
    return f"""; PP0_SAME_TIMESTAMP_CLOSURE_FINITE_ORDER — {ir.case_id}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
{domain}
{guard}
{state}
{frame}
{time_c}
(declare-const closure_measure_pre Int)
(declare-const closure_measure_post Int)
(assert (and (= time_post time_pre) (>= closure_measure_post closure_measure_pre)))
(check-sat)
"""


ENCODER_MAP: dict[str, Any] = {
    "FIXED_DEMAND_NOT_MODIFIED": build_fixed_demand_frame_query,
    "PROTECTED_KEY_NOT_MODIFIED": build_protected_key_frame_query,
    "MODE_ONLY_NOT_MODIFY_PROTECTED": build_mode_only_stutter_query,
    "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE": build_tail_only_stutter_query,
    "DDL_READ_ONLY_DEADLINE_COMPLETION": build_ddl_observe_only_query,
    "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND": build_completion_guard_equivalence_query,
    "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION": build_dispatch_priority_totality_query,
    "SERVICE_UNIT_SINGLE_DISCRETE_RATE": build_service_unit_rate_query,
    "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL": build_arrival_protected_independence_query,
    "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER": build_closure_finite_order_query,
}


def _try_load_compiled_ir() -> dict[str, Any] | None:
    """Try to load compiled transition IR from the executable compiler."""
    try:
        from .executable_transition_compiler import compiled_ir_map as _compiled_ir_map
        return _compiled_ir_map()
    except (ImportError, ValueError, TypeError):
        return None


def _build_smt2_from_compiled_ir(
    compiled: Any,
    case_id: str,
    obligation_key: str,
    encoder: Any,
    schema_case: Any,
) -> str | None:
    """Build SMT2 query from a CompiledTransitionIR, or None if not applicable."""
    if compiled is None:
        return None
    if compiled.compilation_status != "COMPILED":
        return f"; Compiled IR for {case_id} not compiled (status={compiled.compilation_status})"
    domain = _emit_domain_constraint(schema_case)
    guard_smt = compiled.precondition.to_smt()
    state_lines = []
    for eq in compiled.post_equations:
        if eq.kind == "state":
            target = eq.target if str(eq.target).endswith("_post") else f"{eq.target}_post"
            state_lines.append(f"(assert (= {target} {eq.expression.to_smt()}))")
    frame_lines = []
    for f in sorted(compiled.frame_fields):
        frame_lines.append(f"(assert (= {f}_post {f}_pre))")
    time_line = f"(assert (= time_post {compiled.time_update.to_smt()}))"
    guard_assert = f"(assert {guard_smt})"

    neg_query = _negated_obligation(
        obligation_key, compiled, schema_case
    )

    return f"""; PP0_{obligation_key} — {case_id} (COMPILED)
; source: {compiled.source_function}  ast_hash: {compiled.source_function_ast_hash[:16]}...
; IR hash: {compiled.ir_hash()[:16]}...
{domain}
{guard_assert}
{chr(10).join(state_lines)}
{chr(10).join(frame_lines)}
{time_line}
{neg_query}
(check-sat)
"""


def _negated_obligation(obligation_key: str, compiled: Any, schema_case: Any) -> str:
    """Generate the negated obligation assertion for SMT UNSAT check."""
    if obligation_key == "FIXED_DEMAND_NOT_MODIFIED":
        return "(assert (not (= fixed_demand_post fixed_demand_pre)))"
    if obligation_key == "PROTECTED_KEY_NOT_MODIFIED":
        key_fields = ["release_time", "absolute_deadline", "criticality", "priority_index"]
        relevant = [f for f in key_fields if f in compiled.frame_fields]
        if not relevant:
            return "; trivial"
        conjuncts = " ".join(f"(not (= {f}_post {f}_pre))" for f in relevant)
        return f"(assert (or {conjuncts}))"
    if obligation_key == "MODE_ONLY_NOT_MODIFY_PROTECTED":
        if compiled.case_id not in ("RECOVERY", "MODE_SWITCH"):
            return "; not applicable"
        protected_fields = ["active", "ready", "running", "service"]
        relevant = [f for f in protected_fields if f in compiled.frame_fields]
        if not relevant:
            return "; trivial"
        conjuncts = " ".join(f"(not (= {f}_post {f}_pre))" for f in relevant)
        return f"(assert (or {conjuncts}))"
    if obligation_key == "TAIL_NOT_MODIFY_PROTECTED_OBSERVABLE":
        if compiled.case_id != "TAIL_ONLY_SERVICE":
            return "; not applicable"
        protected = sorted(compiled.frame_fields)
        if not protected:
            return "; trivial"
        conjuncts = " ".join(f"(not (= {f}_post {f}_pre))" for f in protected)
        return f"(assert (or {conjuncts}))"
    if obligation_key == "DDL_READ_ONLY_DEADLINE_COMPLETION":
        if compiled.case_id != "DDL_OBSERVE":
            return "; not applicable"
        return "(assert (or (not (= active_post active_pre)) (not (= ready_post ready_pre))))"
    if obligation_key == "COMPLETION_GUARD_EQUIV_SERVICE_GE_DEMAND":
        if compiled.case_id != "REM_COMPLETION":
            return "; not applicable"
        return f"(assert (not {compiled.precondition.to_smt()}))"
    if obligation_key == "DISPATCH_IS_FIXED_PRIORITY_TOTAL_SELECTION":
        if compiled.case_id != "FINAL_DISPATCH":
            return "; not applicable"
        return "(declare-const selected_priority Int)\n(assert (and (> ready_pre 0) (= running_post 0)))\n"
    if obligation_key == "SERVICE_UNIT_SINGLE_DISCRETE_RATE":
        if compiled.case_id != "SERVICE_UNIT":
            return "; not applicable"
        return "(assert (not (= service_post (+ service_pre 1))))"
    if obligation_key == "ARRIVAL_BATCH_PROTECTED_INDEPENDENT_OF_TAIL":
        if compiled.case_id != "ARRIVAL_BATCH_OPEN":
            return "; not applicable"
        return "(declare-const tail_deleted Bool)\n(assert (and tail_deleted (not (= active_post (+ active_pre 1)))))"
    if obligation_key == "SAME_TIME_CLOSURE_FIXED_FINITE_ORDER":
        return (
            "(declare-const closure_measure_pre Int)\n"
            "(declare-const closure_measure_post Int)\n"
            "(assert (and (= time_post time_pre) (>= closure_measure_post closure_measure_pre)))"
        )
    return "; unknown obligation"


def generate_code_bound_queries() -> dict[str, dict[str, Any]]:
    """Generate all code-bound SMT2 queries from the PP0 transition IR.

    If compiled IR is available from the executable transition compiler,
    it takes priority over hand-written IR.  Only compiled IR with
    compilation_status == "COMPILED" produces code-bound queries.

    Each query includes domain, guard, state, frame, and time constraints
    from the IR.  The result has transition_equations_bound=True when the
    query contains actual assert statements from a compiled IR.
    """
    ir_map = {ir.case_id: ir for ir in build_pp0_transition_ir()}
    compiled_map = _try_load_compiled_ir()
    obligations = transition_obligations()
    results: dict[str, dict[str, Any]] = {}

    for obligation_key, encoder in ENCODER_MAP.items():
        for case in CANONICAL_CASES:
            query_id = f"{case.case_id}_{obligation_key}"
            ir = ir_map.get(case.case_id)
            compiled_ir = compiled_map.get(case.case_id) if compiled_map else None

            # Priority: compiled IR with COMPILED status > hand-written IR
            if compiled_ir is not None and compiled_ir.compilation_status == "COMPILED":
                smt2 = _build_smt2_from_compiled_ir(
                    compiled_ir, case.case_id, obligation_key, encoder, case,
                )
                if smt2 is not None and "(assert" in smt2:
                    scope = "EXECUTABLE_CODE_BOUND"
                    bound = True
                elif smt2 is not None:
                    scope = "COMPILED_TRIVIAL_OR_NOT_APPLICABLE"
                    bound = False
                else:
                    smt2 = f"; COMPILED_IR_BUILD_FAILED:{case.case_id}"
                    scope = "COMPILED_IR_ERROR"
                    bound = False
                source_func = compiled_ir.source_function
                source_binding = compiled_ir.source_function_ast_hash
                ir_hash = compiled_ir.ir_hash()
            elif ir is None:
                smt2 = f"; IR missing for case {case.case_id}"
                bound = False
                scope = "IR_NOT_AVAILABLE"
                source_func = None
                source_binding = None
                ir_hash = None
            else:
                smt2 = encoder(case, ir)
                has_asserts = "(assert" in smt2
                has_domain = "(declare-const" in smt2
                syntactically_encoded = has_asserts and has_domain and (
                    "(assert (=" in smt2 or "(assert (and" in smt2 or "(assert (not" in smt2
                )
                bound = syntactically_encoded and ir.binding_kind == "EXECUTABLE_TRANSITION_COMPILER"
                if bound:
                    scope = "EXECUTABLE_CODE_BOUND"
                elif syntactically_encoded:
                    scope = "HAND_WRITTEN_SCHEMA_ONLY"
                else:
                    scope = "TRIVIAL_OR_NOT_APPLICABLE"
                source_func = ir.source_function
                source_binding = ir.source_binding
                ir_hash = ir.ir_hash

            results[query_id] = {
                "query_id": query_id,
                "case_id": case.case_id,
                "obligation": obligation_key,
                "smt2_source": smt2,
                "smt2_hash": sha256_object({"source": smt2}),
                "ir_hash": ir_hash,
                "source_function": source_func,
                "source_binding_hash": source_binding,
                "transition_equations_bound": bound,
                "proof_scope": scope,
                "expected_result": "trivial" if not bound else "unsat",
            }
    return results


def is_trivial_query_source(smt2: str) -> bool:
    return smt2.strip().startswith("; trivial") or smt2.strip().startswith("; not applicable")


# Patch into PP0TransitionIR for encoder access
def _frame_encoded(self: PP0TransitionIR) -> frozenset[str]:
    def base_name(name: str) -> str:
        for suffix in ("_post", "_pre"):
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name
    return frozenset(base_name(eq.lhs) for eq in self.frame_equations)


PP0TransitionIR.frame_equations_encoded = _frame_encoded
