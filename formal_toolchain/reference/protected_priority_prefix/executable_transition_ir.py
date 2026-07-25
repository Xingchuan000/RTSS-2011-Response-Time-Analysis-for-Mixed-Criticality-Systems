"""Source-bound executable transition intermediate representation.

The IR is intentionally proof-oriented.  It keeps a path-sensitive structural
trace of the real Python transition, source-bound helper summaries, and a
validated abstract effect used by the PP0 relational encoder.  A transition is
``COMPILED`` only when all terminal paths and helper calls are covered and the
case-specific semantic validator succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True, slots=True)
class BoolExpr:
    kind: str
    op: str | None = None
    left: str | None = None
    right: str | None = None
    children: tuple["BoolExpr", ...] = ()

    def to_smt(self) -> str:
        if self.kind == "atomic":
            return self.left or "true"
        if self.kind == "and":
            return f"(and {' '.join(c.to_smt() for c in self.children)})"
        if self.kind == "or":
            return f"(or {' '.join(c.to_smt() for c in self.children)})"
        if self.kind == "not":
            return f"(not {self.children[0].to_smt()})" if self.children else "false"
        if self.kind == "cmp":
            op_map = {"==": "=", "=": "=", "!=": "distinct"}
            op = op_map.get(self.op or "=", self.op or "=")
            if op == "distinct":
                return f"(distinct {self.left} {self.right})"
            return f"({op} {self.left} {self.right})"
        return self.left or "true"


@dataclass(frozen=True, slots=True)
class IntExpr:
    kind: str
    var: str | None = None
    value: int | None = None
    left: str | None = None
    right: str | None = None
    condition: BoolExpr | None = None
    then_val: str | None = None
    else_val: str | None = None

    def to_smt(self) -> str:
        if self.kind in {"raw", "var"}:
            return self.var or "0"
        if self.kind == "const":
            return str(0 if self.value is None else self.value)
        if self.kind == "add":
            return f"(+ {self.left or '0'} {self.right or '0'})"
        if self.kind == "ite":
            cond = self.condition.to_smt() if self.condition else "true"
            return f"(ite {cond} {self.then_val or '0'} {self.else_val or '0'})"
        return self.var or "0"


@dataclass(frozen=True, slots=True)
class Assignment:
    target: str
    expression: IntExpr
    kind: str = "state"


@dataclass(frozen=True, slots=True)
class GeneratedEventRule:
    event_kind: str
    time_expr: IntExpr
    condition: BoolExpr | None = None
    job_key_expr: str | None = None


@dataclass(frozen=True, slots=True)
class FoldIR:
    sequence_symbol: str
    cursor_symbol: str
    invariant_schema_hash: str
    step_summary_hash: str
    termination_measure: str
    mutated_locals: tuple[str, ...] = ()
    exceptional_path_hashes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SymbolicOperation:
    """One source-derived operation in a terminal path or loop summary."""

    kind: str
    target: str | None
    expression: str | None
    source_line: int
    source_col: int

    def canonical(self) -> tuple[Any, ...]:
        return (self.kind, self.target, self.expression, self.source_line, self.source_col)


@dataclass(frozen=True, slots=True)
class SemanticEffectRule:
    """Validated abstract effect consumed directly by the PP0 encoder.

    ``field_equations`` use a small symbolic vocabulary (uninterpreted
    deterministic functions plus arithmetic).  The validator proves that the
    real source contains the operations required by ``validator_facts``.
    """

    effect_id: str
    field_equations: tuple[tuple[str, str], ...]
    frame_fields: frozenset[str]
    input_symbols: tuple[str, ...]
    supported_pairings: tuple[str, ...]
    validator_rule_id: str
    validator_facts: tuple[str, ...]
    source_path_hashes: tuple[str, ...]
    # The projection compiler is part of the proof TCB.  These fields make its
    # coverage explicit: every concrete state write on every terminal path is
    # either represented by an abstract equation or discharged by a named
    # projection/frame lemma.
    concrete_write_targets: tuple[str, ...] = ()
    covered_concrete_write_targets: tuple[str, ...] = ()
    path_effect_hashes: tuple[str, ...] = ()
    required_assumption_ids: tuple[str, ...] = ()
    derivation_complete: bool = False

    def effect_hash(self) -> str:
        return sha256_object({
            "effect_id": self.effect_id,
            "field_equations": list(self.field_equations),
            "frame_fields": sorted(self.frame_fields),
            "input_symbols": list(self.input_symbols),
            "supported_pairings": list(self.supported_pairings),
            "validator_rule_id": self.validator_rule_id,
            "validator_facts": list(self.validator_facts),
            "source_path_hashes": list(self.source_path_hashes),
            "concrete_write_targets": list(self.concrete_write_targets),
            "covered_concrete_write_targets": list(self.covered_concrete_write_targets),
            "path_effect_hashes": list(self.path_effect_hashes),
            "required_assumption_ids": list(self.required_assumption_ids),
            "derivation_complete": self.derivation_complete,
        })


@dataclass(frozen=True, slots=True)
class CompiledPathIR:
    path_id: str
    path_condition: BoolExpr
    updates: tuple[Assignment, ...]
    frame_fields: frozenset[str]
    generated_events: tuple[GeneratedEventRule, ...]
    terminator: str
    exception_type: str | None = None
    operations: tuple[SymbolicOperation, ...] = ()
    folds: tuple[FoldIR, ...] = ()
    helper_summary_hashes: tuple[str, ...] = ()
    return_expression: str | None = None

    def path_hash(self) -> str:
        return sha256_object({
            "path_id": self.path_id,
            "condition": self.path_condition.to_smt(),
            "updates": [(a.target, a.expression.to_smt(), a.kind) for a in self.updates],
            "frame_fields": sorted(self.frame_fields),
            "events": [(e.event_kind, e.time_expr.to_smt(), e.job_key_expr) for e in self.generated_events],
            "terminator": self.terminator,
            "exception_type": self.exception_type,
            "operations": [op.canonical() for op in self.operations],
            "folds": [
                (f.sequence_symbol, f.cursor_symbol, f.invariant_schema_hash,
                 f.step_summary_hash, f.termination_measure,
                 f.mutated_locals, f.exceptional_path_hashes)
                for f in self.folds
            ],
            "return_expression": self.return_expression,
        })


@dataclass(frozen=True, slots=True)
class HelperSummaryReceipt:
    helper_name: str
    source_ast_hash: str
    precondition: BoolExpr
    paths: tuple[CompiledPathIR, ...]
    total_semantic_coverage: bool
    summary_hash: str
    source_function: str | None = None
    operation_trace_hash: str | None = None
    called_helpers: tuple[str, ...] = ()
    unsupported_nodes: tuple[str, ...] = ()
    binding_kind: str = "SOURCE_BOUND_STRUCTURAL_SUMMARY"


@dataclass(frozen=True, slots=True)
class TransitionCompilationReceipt:
    source_function: str
    source_ast_hash: str
    normalized_cfg_hash: str
    return_path_count: int
    covered_return_path_count: int
    raise_path_count: int
    covered_raise_path_count: int
    helper_calls: tuple[str, ...] = ()
    helper_summaries_bound: tuple[str, ...] = ()
    state_fields_read: tuple[str, ...] = ()
    state_fields_written: tuple[str, ...] = ()
    frame_fields: tuple[str, ...] = ()
    unsupported_nodes: tuple[str, ...] = ()
    total_semantic_coverage: bool = False
    domain_equations: tuple[str, ...] = ()
    guard_equations: tuple[str, ...] = ()
    update_equations: tuple[str, ...] = ()
    exceptional_paths: tuple[str, ...] = ()
    helper_summary_hashes: tuple[str, ...] = ()
    control_flow_coverage: Mapping[str, Any] = field(default_factory=dict)
    semantic_validator_rule_id: str | None = None
    semantic_validator_facts: tuple[str, ...] = ()
    semantic_validator_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_function": self.source_function,
            "source_ast_hash": self.source_ast_hash,
            "normalized_cfg_hash": self.normalized_cfg_hash,
            "return_path_count": self.return_path_count,
            "covered_return_path_count": self.covered_return_path_count,
            "raise_path_count": self.raise_path_count,
            "covered_raise_path_count": self.covered_raise_path_count,
            "helper_calls": list(self.helper_calls),
            "helper_summaries_bound": list(self.helper_summaries_bound),
            "state_fields_read": list(self.state_fields_read),
            "state_fields_written": list(self.state_fields_written),
            "frame_fields": list(self.frame_fields),
            "unsupported_nodes": list(self.unsupported_nodes),
            "total_semantic_coverage": self.total_semantic_coverage,
            "domain_equations": list(self.domain_equations),
            "guard_equations": list(self.guard_equations),
            "update_equations": list(self.update_equations),
            "exceptional_paths": list(self.exceptional_paths),
            "helper_summary_hashes": list(self.helper_summary_hashes),
            "control_flow_coverage": dict(self.control_flow_coverage),
            "semantic_validator_rule_id": self.semantic_validator_rule_id,
            "semantic_validator_facts": list(self.semantic_validator_facts),
            "semantic_validator_passed": self.semantic_validator_passed,
        }


@dataclass(frozen=True, slots=True)
class CompiledTransitionIR:
    case_id: str
    source_module: str
    source_function: str
    source_function_ast_hash: str
    precondition: BoolExpr
    post_equations: tuple[Assignment, ...]
    frame_fields: frozenset[str]
    generated_events: tuple[GeneratedEventRule, ...]
    time_update: IntExpr
    compilation_receipt_hash: str
    compilation_status: str = "UNRESOLVED"
    binding_kind: str = "EXECUTABLE_TRANSITION_COMPILER"
    compilation_receipt: TransitionCompilationReceipt | None = None
    paths: tuple[CompiledPathIR, ...] = ()
    helper_summary_hashes: tuple[str, ...] = ()
    return_path_count: int = 0
    covered_return_path_count: int = 0
    raise_path_count: int = 0
    covered_raise_path_count: int = 0
    total_semantic_coverage: bool = False
    semantic_effect: SemanticEffectRule | None = None
    folds: tuple[FoldIR, ...] = ()

    def ir_hash(self) -> str:
        return sha256_object({
            "case_id": self.case_id,
            "source_module": self.source_module,
            "source_function": self.source_function,
            "source_function_ast_hash": self.source_function_ast_hash,
            "precondition": self.precondition.to_smt(),
            "post_equations": [(a.target, a.expression.to_smt(), a.kind) for a in self.post_equations],
            "frame_fields": sorted(self.frame_fields),
            "events": [(e.event_kind, e.time_expr.to_smt(), e.job_key_expr) for e in self.generated_events],
            "time_update": self.time_update.to_smt(),
            "compilation_status": self.compilation_status,
            "binding_kind": self.binding_kind,
            "compilation_receipt": self.compilation_receipt.to_dict() if self.compilation_receipt else None,
            "paths": [p.path_hash() for p in self.paths],
            "helper_summary_hashes": list(self.helper_summary_hashes),
            "folds": [
                (f.sequence_symbol, f.cursor_symbol, f.invariant_schema_hash,
                 f.step_summary_hash, f.termination_measure,
                 f.mutated_locals, f.exceptional_path_hashes)
                for f in self.folds
            ],
            "semantic_effect_hash": self.semantic_effect.effect_hash() if self.semantic_effect else None,
        })

    def is_compiled(self) -> bool:
        receipt = self.compilation_receipt
        return bool(
            self.compilation_status == "COMPILED"
            and self.binding_kind == "EXECUTABLE_TRANSITION_COMPILER"
            and receipt is not None
            and receipt.total_semantic_coverage
            and receipt.semantic_validator_passed
            and self.total_semantic_coverage
            and self.semantic_effect is not None
            and self.semantic_effect.derivation_complete
            and self.semantic_effect.concrete_write_targets
            == self.semantic_effect.covered_concrete_write_targets
            and set(self.semantic_effect.source_path_hashes)
            == set(self.semantic_effect.path_effect_hashes)
            and not receipt.unsupported_nodes
            and self.covered_return_path_count == self.return_path_count
            and self.covered_raise_path_count == self.raise_path_count
            and len(self.paths) == self.return_path_count + self.raise_path_count
        )


def empty_bool_expr() -> BoolExpr:
    return BoolExpr(kind="atomic", left="true")


def atomic_bool(condition: str) -> BoolExpr:
    return BoolExpr(kind="atomic", left=condition)


def cmp_expr(op: str, left: str, right: str) -> BoolExpr:
    return BoolExpr(kind="cmp", op=op, left=left, right=right)


def var_expr(name: str) -> IntExpr:
    return IntExpr(kind="var", var=name)


def raw_expr(source: str) -> IntExpr:
    return IntExpr(kind="raw", var=source)


def const_expr(value: int) -> IntExpr:
    return IntExpr(kind="const", value=value)


def add_expr(left: str, right: str) -> IntExpr:
    return IntExpr(kind="add", left=left, right=right)


def ite_expr(condition: BoolExpr, then_val: str, else_val: str) -> IntExpr:
    return IntExpr(kind="ite", condition=condition, then_val=then_val, else_val=else_val)


def state_assignment(target: str, expr: IntExpr) -> Assignment:
    return Assignment(target=target, expression=expr, kind="state")


def frame_assignment(target: str, expr: IntExpr) -> Assignment:
    return Assignment(target=target, expression=expr, kind="frame")


def time_assignment(target: str, expr: IntExpr) -> Assignment:
    return Assignment(target=target, expression=expr, kind="time")
