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


RELATIONAL_PP0_RECEIPTS: list[dict[str, Any]] = [
    {
        "receipt_id": "PP0_REM_RELATION",
        "case_id": "REM_COMPLETION",
        "description": "REM completion preserves protected observable across full and prefix",
        "prefix_skips": False,
        "phase": "SvcEnd",
    },
    {
        "receipt_id": "PP0_RECOVERY_STUTTER_FULL_ONLY",
        "case_id": "RECOVERY",
        "description": "Recovery is full-only; prefix stutters",
        "prefix_skips": True,
        "phase": "AfterREM",
    },
    {
        "receipt_id": "PP0_RECOVERY_STUTTER_PREFIX_ONLY",
        "case_id": "RECOVERY",
        "description": "Recovery is prefix-only; full stutters",
        "prefix_skips": False,
        "phase": "AfterREM",
    },
    {
        "receipt_id": "PP0_DDL_OBSERVE_ONLY",
        "case_id": "DEADLINE_OBSERVATION",
        "description": "Deadline observation preserves protected observable across full and prefix",
        "prefix_skips": False,
        "phase": "AfterREC",
    },
    {
        "receipt_id": "PP0_ARR_PENDING_PLAN_PROJECTION",
        "case_id": "ARRIVAL_BATCH",
        "description": "Arrival batch pending plan fields equal across full and prefix",
        "prefix_skips": False,
        "phase": "DDLCursor",
    },
    {
        "receipt_id": "PP0_SWITCH_STUTTER_FULL_ONLY",
        "case_id": "MODE_SWITCH",
        "description": "Mode switch is full-only; prefix stutters",
        "prefix_skips": True,
        "phase": "ARRCursor",
    },
    {
        "receipt_id": "PP0_SWITCH_STUTTER_PREFIX_ONLY",
        "case_id": "MODE_SWITCH",
        "description": "Mode switch is prefix-only; full stutters",
        "prefix_skips": False,
        "phase": "ARRCursor",
    },
    {
        "receipt_id": "PP0_RELEASE_PROTECTED_PAYLOAD",
        "case_id": "RELEASE",
        "description": "Release payload fields equal across full and prefix",
        "prefix_skips": False,
        "phase": "PreDisp",
    },
    {
        "receipt_id": "PP0_DISPATCH_DETERMINISM",
        "case_id": "FINAL_DISPATCH",
        "description": "Dispatch selects same protected job in both views",
        "prefix_skips": False,
        "phase": "PreDisp",
    },
    {
        "receipt_id": "PP0_SERVICE_PROTECTED",
        "case_id": "SERVICE_UNIT",
        "description": "Same running protected job gets same service unit in both views",
        "prefix_skips": False,
        "phase": "Close",
    },
    {
        "receipt_id": "PP0_TAIL_SERVICE_STUTTER",
        "case_id": "TAIL_ONLY_SERVICE",
        "description": "Tail service does not change protected observable; both stutter",
        "prefix_skips": True,
        "phase": "Close",
    },
    {
        "receipt_id": "PP0_IDLE_JUMP_STUTTER",
        "case_id": "TAIL_ONLY_SERVICE",
        "description": "Idle jump advances time; protected observable unchanged in both views",
        "prefix_skips": True,
        "phase": "Close",
    },
]


def _emit_domain_constraint(schema: PrimitiveTransitionSchema) -> str:
    lines = []
    all_fields = sorted(set(
        list(schema.read_fields) + list(schema.write_fields) +
        list(schema.protected_frame_fields) + list(schema.guard_fields) + [
            "time", "active", "ready", "running", "running_job_key",
            "job_key", "miss", "miss_ledger", "completed", "removed",
            "event_kind", "batch_size", "pending_releases", "mode",
            "mode_is_hi", "mode_is_lo", "HI", "LO", "active_job_count",
            "running_present", "pending_release_count", "pending_abnormal_trigger",
            "protected_ready", "protected_ready_empty", "state_time",
            "tail_ready", "active_job_set", "DEADLINE", "ARR_BATCH", "SW",
            "RELEASE", "SERVICE", "REM", "REC", "DSP", "event_kind",
        ]
    ))
    for field in all_fields:
        for side in ("_f_pre", "_f_post", "_p_pre", "_p_post"):
            lines.append(f"(declare-const {field}{side} Int)")
        lines.append(f"(assert (>= {field}_f_pre 0))")
        lines.append(f"(assert (>= {field}_p_pre 0))")
    # Formula atoms such as HI, RELEASE and batch_size are parameters of the
    # executable guard rather than state fields; bind them explicitly.
    for atom in ("LO", "HI", "DEADLINE", "ARR_BATCH", "SW", "RELEASE",
                 "SERVICE", "REM", "REC", "DSP", "batch_size",
                 "active_job_set"):
        lines.append(f"(declare-const {atom} Int)")
    for atom in ("running_present", "priority_index"):
        lines.append(f"(declare-const {atom} Int)")
    for atom in ("release_time", "job_key", "running_f_present_f_pre",
                 "running_f_present_p_pre", "running_p_present_p_pre"):
        lines.append(f"(declare-const {atom} Int)")
    return "\n".join(lines)


def _emit_relation_constraint(schema: PrimitiveTransitionSchema) -> str:
    lines = []
    for field in schema.protected_frame_fields:
        lines.append(f"(assert (= {field}_f_pre {field}_p_pre))")
    if schema.guard_fields:
        for field in schema.guard_fields:
            lines.append(f"(assert (= {field}_f_pre {field}_p_pre))")
    return "\n".join(lines)


def _emit_next_relation_constraint(schema: PrimitiveTransitionSchema) -> str:
    conjuncts = " ".join(
        f"(= {field}_f_post {field}_p_post)"
        for field in schema.protected_frame_fields
    )
    if not conjuncts:
        return "(assert false)"
    return f"(assert (not (and {conjuncts})))"


import re


def _suffix_vars(expr: str, suffix: str) -> str:
    """Add suffix to bare field names in an expression.

    Uses word-boundary matching to avoid substring corruption
    (e.g., ``active`` inside ``active_job_count``).

    Handles both bare names (``service``) and already-suffixed names
    (``running_pre``) by replacing ``_pre``/``_post`` with the new suffix.
    """
    mapped = expr
    for old_suffix in ("_pre", "_post"):
        mapped = mapped.replace(old_suffix, suffix)
    pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    def _replacer(m: re.Match) -> str:
        name = m.group(1)
        if name.isupper():
            return name
        if name in ("true", "false", "and", "or", "not", "ite",
                     "declare-const", "assert", "check-sat", "Int", "Bool"):
            return name
        if name.endswith(("_f_pre", "_f_post", "_p_pre", "_p_post")):
            return name
        return f"{name}{suffix}"
    result = pattern.sub(_replacer, mapped)
    # The schema models flags as integer 0/1 variables; normalize boolean
    # negation emitted by the hand-audited formula into an integer predicate.
    result = re.sub(r"\(not ([A-Za-z_][A-Za-z0-9_]*)\)", r"(= \1 0)", result)
    return result


def _emit_full_transition(ir: PP0TransitionIR, schema: PrimitiveTransitionSchema | None = None) -> str:
    lines = []
    for eq in ir.state_equations:
        target = eq.lhs.replace("_post", "_f_post")
        rhs = eq.rhs
        for side in ("_pre", "_post"):
            rhs = rhs.replace(side, f"_f{side}")
        rhs = re.sub(r"\(not ([A-Za-z_][A-Za-z0-9_]*)\)", r"(= \1 0)", rhs)
        rhs = re.sub(r"\(select_min_priority_index [^)]*\)", "0", rhs)
        lines.append(f"(assert (= {target} {rhs}))")
    for eq in ir.frame_equations:
        target = eq.lhs.replace("_post", "_f_post")
        rhs = eq.rhs
        for side in ("_pre", "_post"):
            rhs = rhs.replace(side, f"_f{side}")
        rhs = re.sub(r"\(not ([A-Za-z_][A-Za-z0-9_]*)\)", r"(= \1 0)", rhs)
        rhs = re.sub(r"\(select_min_priority_index [^)]*\)", "0", rhs)
        lines.append(f"(assert (= {target} {rhs}))")
    time_rhs = ir.time_equation.rhs
    for side in ("_pre", "_post"):
        time_rhs = time_rhs.replace(side, f"_f{side}")
    lines.append(f"(assert (= time_f_post {time_rhs}))")
    if schema is not None:
        for field in schema.protected_frame_fields:
            lines.append(f"(assert (= {field}_f_post {field}_f_pre))")
    guard = _suffix_vars(ir.guard_formula, "_f_pre")
    lines.append(f"(assert {guard})")
    return "\n".join(lines)


def _emit_prefix_skip(schema: PrimitiveTransitionSchema) -> str:
    lines = []
    all_fields = sorted(set(
        list(schema.read_fields) + list(schema.write_fields) +
        list(schema.protected_frame_fields) + list(schema.guard_fields)
    ))
    for field in all_fields:
        lines.append(f"(assert (= {field}_p_post {field}_p_pre))")
    lines.append("(assert (= time_p_post time_p_pre))")
    return "\n".join(lines)


def _emit_prefix_transition(ir: PP0TransitionIR, schema: PrimitiveTransitionSchema | None = None) -> str:
    lines = []
    for eq in ir.state_equations:
        target = eq.lhs.replace("_post", "_p_post")
        rhs = eq.rhs
        for side in ("_pre", "_post"):
            rhs = rhs.replace(side, f"_p{side}")
        rhs = re.sub(r"\(not ([A-Za-z_][A-Za-z0-9_]*)\)", r"(= \1 0)", rhs)
        rhs = re.sub(r"\(select_min_priority_index [^)]*\)", "0", rhs)
        lines.append(f"(assert (= {target} {rhs}))")
    for eq in ir.frame_equations:
        target = eq.lhs.replace("_post", "_p_post")
        rhs = eq.rhs
        for side in ("_pre", "_post"):
            rhs = rhs.replace(side, f"_p{side}")
        rhs = re.sub(r"\(not ([A-Za-z_][A-Za-z0-9_]*)\)", r"(= \1 0)", rhs)
        rhs = re.sub(r"\(select_min_priority_index [^)]*\)", "0", rhs)
        lines.append(f"(assert (= {target} {rhs}))")
    time_rhs = ir.time_equation.rhs
    for side in ("_pre", "_post"):
        time_rhs = time_rhs.replace(side, f"_p{side}")
    lines.append(f"(assert (= time_p_post {time_rhs}))")
    if schema is not None:
        for field in schema.protected_frame_fields:
            lines.append(f"(assert (= {field}_p_post {field}_p_pre))")
    guard = _suffix_vars(ir.guard_formula, "_p_pre")
    lines.append(f"(assert {guard})")
    return "\n".join(lines)


def _build_relational_smt2(receipt: dict[str, Any],
                            schema: PrimitiveTransitionSchema,
                            ir: PP0TransitionIR) -> str:
    domain = _emit_domain_constraint(schema)
    relation = _emit_relation_constraint(schema)
    full_trans = _emit_full_transition(ir, schema)
    if receipt.get("prefix_skips"):
        prefix_trans = _emit_prefix_skip(schema)
    else:
        prefix_trans = _emit_prefix_transition(ir, schema)
    neg_next = _emit_next_relation_constraint(schema)

    return f"""; {receipt['receipt_id']} — {receipt['case_id']}
; source: {ir.source_function}  hash: {ir.source_binding[:16]}...
; IR hash: {ir.ir_hash[:16]}...
; Relation: {receipt['description']}
; Query: Domain(pre_f,pre_p) ∧ Rel_phase(pre,pre) ∧ T_f(pre_f,post_f) ∧ T_p/Skip(pre_p,post_p) ∧ ¬Rel_next(post_f,post_p)
{domain}
{relation}
{full_trans}
{prefix_trans}
{neg_next}
(check-sat)
"""


def _try_load_compiled_ir() -> dict[str, Any] | None:
    try:
        from .executable_transition_compiler import compiled_ir_map as _compiled_ir_map
        return _compiled_ir_map()
    except (ImportError, ValueError, TypeError):
        return None


def _try_load_bound_transitions() -> dict[str, Any] | None:
    try:
        from .pp_transition_binding import compile_all_transitions as _bound_transitions
        return _bound_transitions()
    except (ImportError, ValueError, TypeError, AttributeError):
        return None


PP0_SOURCE_HASH_CACHE: dict[str, str] = {}


def _compute_relation_schema_hash(receipt: dict[str, Any]) -> str:
    return sha256_object({
        "receipt_id": receipt["receipt_id"],
        "case_id": receipt["case_id"],
        "description": receipt["description"],
        "prefix_skips": receipt["prefix_skips"],
        "phase": receipt["phase"],
    })


def generate_code_bound_queries() -> dict[str, dict[str, Any]]:
    ir_map = {ir.case_id: ir for ir in build_pp0_transition_ir()}
    schema_map = {case.case_id: case for case in CANONICAL_CASES}
    bound_map = _try_load_bound_transitions() or {}
    compiled_map = _try_load_compiled_ir() or {}
    results: dict[str, dict[str, Any]] = {}

    for receipt in RELATIONAL_PP0_RECEIPTS:
        case_id = receipt["case_id"]
        receipt_id = receipt["receipt_id"]
        schema = schema_map.get(case_id)
        ir = ir_map.get(case_id)
        bound = bound_map.get(case_id)
        compiled = compiled_map.get(case_id)

        if schema is None or ir is None:
            results[receipt_id] = {
                "query_id": receipt_id,
                "case_id": case_id,
                "receipt_id": receipt_id,
                "smt2_source": f"; Schema or IR missing for case {case_id}",
                "smt2_hash": sha256_object({"source": f"; missing {case_id}"}),
                "ir_hash": None,
                "source_function": None,
                "source_ast_hash": None,
                "relation_schema_hash": _compute_relation_schema_hash(receipt),
                "transition_equations_bound": False,
                "proof_scope": "MISSING_SCHEMA_OR_IR",
                "expected_result": "unresolved",
            }
            continue

        smt2 = _build_relational_smt2(receipt, schema, ir)

        source_func = bound.source_function if bound else ir.source_function
        source_hash = bound.source_ast_hash if bound else ir.source_binding

        # The equations below are emitted from the audited PP0TransitionIR.
        # Matching a schema case proves only that the intended mathematical
        # case was selected; it does not prove semantic equivalence to the
        # executable transition.  A relational query is therefore code-bound
        # only after the executable compiler has total path/update/frame
        # coverage and a separate adapter-equivalence proof binds the audited
        # IR equations to that compiled transition.  Until that proof exists,
        # fail closed instead of allowing Z3 to prove a theorem about a
        # handwritten adapter and label it as executable-code-bound.
        compiler_total = bool(
            compiled is not None
            and getattr(compiled, "compilation_status", None) == "COMPILED"
            and getattr(compiled, "binding_kind", None)
                == "EXECUTABLE_TRANSITION_COMPILER"
            and getattr(compiled, "source_function_ast_hash", None) == source_hash
            and getattr(compiled, "compilation_receipt", None) is not None
            and compiled.compilation_receipt.total_semantic_coverage
            and not compiled.compilation_receipt.unsupported_nodes
        )
        adapter_receipt = getattr(compiled, "adapter_equivalence_receipt", None)
        adapter_equivalence_proved = bool(
            compiler_total
            and isinstance(adapter_receipt, Mapping)
            and adapter_receipt.get("status") == "PASS"
            and adapter_receipt.get("theorem_id") == "EXECUTABLE_TO_PP0_ADAPTER_EQUIVALENCE"
            and adapter_receipt.get("compiled_ir_hash") == compiled.ir_hash()
            and adapter_receipt.get("audited_pp0_ir_hash") == ir.ir_hash
            and adapter_receipt.get("all_paths_related") is True
            and adapter_receipt.get("all_updates_related") is True
            and adapter_receipt.get("all_frames_related") is True
        )
        equations_bound = compiler_total and adapter_equivalence_proved

        results[receipt_id] = {
            "query_id": receipt_id,
            "case_id": case_id,
            "receipt_id": receipt_id,
            "smt2_source": smt2,
            "smt2_hash": sha256_object({"source": smt2}),
            "ir_hash": ir.ir_hash,
            "source_function": source_func,
            "source_ast_hash": source_hash,
            "relation_schema_hash": _compute_relation_schema_hash(receipt),
            "transition_equations_bound": equations_bound,
            "compiler_total_semantic_coverage": compiler_total,
            "adapter_equivalence_proved": adapter_equivalence_proved,
            "full_ir_hash": ir.ir_hash,
            "prefix_ir_hash": ir.ir_hash,
            "pairing_kind": "FULL_STUTTER_PREFIX" if receipt.get("prefix_skips") else "LOCKSTEP",
            "relation_pre_hash": _compute_relation_schema_hash(receipt),
            "relation_post_hash": _compute_relation_schema_hash(receipt),
            "all_paths_covered": bool(compiled and compiled.total_semantic_coverage),
            "proof_scope": (
                "CODE_BOUND_RELATIONAL" if equations_bound
                else "HAND_WRITTEN_SCHEMA_NOT_CODE_BOUND"
            ),
            "expected_result": "unsat" if equations_bound else "unresolved",
        }

    return results


def is_trivial_query_source(smt2: str) -> bool:
    return smt2.strip().startswith("; trivial") or smt2.strip().startswith("; not applicable") or "(assert" not in smt2


def _frame_encoded(self: PP0TransitionIR) -> frozenset[str]:
    def base_name(name: str) -> str:
        for suffix in ("_post", "_pre"):
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name
    return frozenset(base_name(eq.lhs) for eq in self.frame_equations)

PP0TransitionIR.frame_equations_encoded = _frame_encoded
