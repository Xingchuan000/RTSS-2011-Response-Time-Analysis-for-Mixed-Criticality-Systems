"""Compiled executable transition intermediate representation.

Defines the CompiledTransitionIR dataclass with supporting types for
compiler-derived transition equations.  Only compiler output may set
``binding_kind = "EXECUTABLE_TRANSITION_COMPILER"``.  Hand-written
IR in pp0_transition_ir.py serves as schema/obligation catalogue only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True, slots=True)
class BoolExpr:
    """A restricted SMT-like boolean expression over pre-state variables."""
    kind: str  # "atomic", "and", "or", "not", "cmp"
    op: str | None = None
    left: str | None = None
    right: str | None = None
    children: tuple[BoolExpr, ...] = ()

    def to_smt(self) -> str:
        if self.kind == "atomic":
            return self.left or "true"
        if self.kind == "and":
            parts = " ".join(c.to_smt() for c in self.children)
            return f"(and {parts})"
        if self.kind == "or":
            parts = " ".join(c.to_smt() for c in self.children)
            return f"(or {parts})"
        if self.kind == "not":
            return f"(not {self.children[0].to_smt()})" if self.children else "(not true)"
        if self.kind == "cmp":
            op_map = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "=": "=", "==": "="}
            smt_op = op_map.get(self.op, "=")
            return f"({smt_op} {self.left} {self.right})"
        return "true"


@dataclass(frozen=True, slots=True)
class IntExpr:
    kind: str  # "var", "const", "add", "ite"
    var: str | None = None
    value: int | None = None
    left: str | None = None
    right: str | None = None
    condition: BoolExpr | None = None
    then_val: str | None = None
    else_val: str | None = None

    def to_smt(self) -> str:
        if self.kind == "raw":
            return self.var or "0"
        if self.kind == "var":
            return self.var or "0"
        if self.kind == "const":
            return str(self.value or 0)
        if self.kind == "add":
            return f"(+ {self.left or '0'} {self.right or '0'})"
        if self.kind == "ite":
            cond = self.condition.to_smt() if self.condition else "true"
            return f"(ite {cond} {self.then_val or '0'} {self.else_val or '0'})"
        return "0"


@dataclass(frozen=True, slots=True)
class Assignment:
    target: str
    expression: IntExpr
    kind: str = "state"  # "state", "frame", "time"


@dataclass(frozen=True, slots=True)
class GeneratedEventRule:
    event_kind: str
    time_expr: IntExpr
    condition: BoolExpr | None = None
    job_key_expr: str | None = None


@dataclass(frozen=True, slots=True)
class FoldIR:
    """Finite-sequence fold emitted for a data-dependent event loop."""
    sequence_symbol: str
    cursor_symbol: str
    invariant_schema_hash: str
    step_summary_hash: str
    termination_measure: str


@dataclass(frozen=True, slots=True)
class CompiledPathIR:
    path_id: str
    path_condition: BoolExpr
    updates: tuple[Assignment, ...]
    frame_fields: frozenset[str]
    generated_events: tuple[GeneratedEventRule, ...]
    terminator: str
    exception_type: str | None = None


@dataclass(frozen=True, slots=True)
class HelperSummaryReceipt:
    helper_name: str
    source_ast_hash: str
    precondition: BoolExpr
    paths: tuple[CompiledPathIR, ...]
    total_semantic_coverage: bool
    summary_hash: str


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
    compilation_status: str = "COMPILED"  # COMPILED only after total semantic coverage; otherwise PARTIAL_AST_EXTRACTION / AST_UNSUPPORTED / UNRESOLVED
    binding_kind: str = "EXECUTABLE_TRANSITION_COMPILER"
    compilation_receipt: TransitionCompilationReceipt | None = None
    paths: tuple[CompiledPathIR, ...] = ()
    helper_summary_hashes: tuple[str, ...] = ()
    return_path_count: int = 0
    covered_return_path_count: int = 0
    raise_path_count: int = 0
    covered_raise_path_count: int = 0
    total_semantic_coverage: bool = False

    def ir_hash(self) -> str:
        payload = {
            "case_id": self.case_id,
            "source_function": self.source_function,
            "source_function_ast_hash": self.source_function_ast_hash,
            "precondition": self.precondition.to_smt(),
            "post_equations": [(a.target, a.expression.to_smt(), a.kind) for a in self.post_equations],
            "frame_fields": sorted(self.frame_fields),
            "time_update": self.time_update.to_smt(),
            "compilation_status": self.compilation_status,
            "binding_kind": self.binding_kind,
            "compilation_receipt": self.compilation_receipt.to_dict() if self.compilation_receipt else None,
            "paths": [(p.path_id, p.path_condition.to_smt(), p.terminator,
                       p.exception_type, sorted(p.frame_fields)) for p in self.paths],
            "helper_summary_hashes": list(self.helper_summary_hashes),
        }
        return sha256_object(payload)

    def is_compiled(self) -> bool:
        return (self.compilation_status == "COMPILED"
                and self.compilation_receipt is not None
                and self.compilation_receipt.total_semantic_coverage
                and self.total_semantic_coverage
                and not self.compilation_receipt.unsupported_nodes)


def empty_bool_expr() -> BoolExpr:
    return BoolExpr(kind="atomic", left="true")


def atomic_bool(condition: str) -> BoolExpr:
    return BoolExpr(kind="atomic", left=condition)


def cmp_expr(op: str, left: str, right: str) -> BoolExpr:
    return BoolExpr(kind="cmp", op=op, left=left, right=right)


def var_expr(name: str) -> IntExpr:
    return IntExpr(kind="var", var=name)


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
