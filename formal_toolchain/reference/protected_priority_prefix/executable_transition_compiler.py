"""Compile executable transition functions into CompiledTransitionIR.

Reads the actual Python source of each primitive transition function,
parses the AST, and extracts guard/state/frame/time equations.  Only
the restricted subset of Python constructs listed in the PP0 proof
plan is supported; unsupported AST nodes produce UNRESOLVED.

Compilation targets (from executable_semantics.py):
    _normalize_dispatch  -> FINAL_DISPATCH
    apply_removal        -> REM_COMPLETION
    apply_recovery       -> RECOVERY
    apply_deadline_observation -> DDL_OBSERVE
    apply_arrival_batch  -> ARRIVAL_BATCH_OPEN
    apply_mode_switch    -> MODE_SWITCH
    apply_release        -> RELEASE
    apply_service_tick   -> SERVICE_UNIT
    close_timestamp closure -> TAIL_ONLY_SERVICE
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference import executable_semantics

from .executable_transition_ir import (
    Assignment,
    BoolExpr,
    CompiledTransitionIR,
    GeneratedEventRule,
    IntExpr,
    add_expr,
    atomic_bool,
    cmp_expr,
    const_expr,
    empty_bool_expr,
    frame_assignment,
    ite_expr,
    state_assignment,
    time_assignment,
    var_expr,
    TransitionCompilationReceipt,
)

_SOURCE_MODULE = "formal_toolchain.reference.executable_semantics"

COMPILATION_MAP: dict[str, dict[str, Any]] = {
    "_normalize_dispatch": {
        "case_id": "FINAL_DISPATCH",
        "description": "Dispatch: strict FP total order, smaller priority_index first.",
    },
    "apply_removal": {
        "case_id": "REM_COMPLETION",
        "description": "Removal after service >= fixed_demand; may generate recovery.",
    },
    "apply_recovery": {
        "case_id": "RECOVERY",
        "description": "HI->LO recovery when quiescent.",
    },
    "apply_deadline_observation": {
        "case_id": "DDL_OBSERVE",
        "description": "Deadline observation: non-completed job misses.",
    },
    "apply_arrival_batch": {
        "case_id": "ARRIVAL_BATCH_OPEN",
        "description": "Arrival batch: creates pending releases, does NOT create active/ready.",
    },
    "apply_mode_switch": {
        "case_id": "MODE_SWITCH",
        "description": "LO->HI mode switch triggered by abnormal HI arrival.",
    },
    "apply_release": {
        "case_id": "RELEASE",
        "description": "Release: moves job from pending to released/jobs (active/ready).",
    },
    "apply_service_tick": {
        "case_id": "SERVICE_UNIT",
        "description": "Service tick: increments service by 1, may generate removal.",
    },
}


def _get_source(obj: Any) -> tuple[str, str]:
    """Return (source_code, source_hash) for a callable."""
    try:
        source = textwrap.dedent(inspect.getsource(obj))
    except (TypeError, OSError) as exc:
        raise ValueError(f"COMPILER_CANNOT_READ_SOURCE:{exc}") from exc
    source_hash = sha256_object({"source": source})
    return source, source_hash


def _parse_function(source: str) -> ast.FunctionDef:
    """Parse a single function definition from source code."""
    module = ast.parse(source)
    for node in ast.iter_child_nodes(module):
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("COMPILER_NO_FUNCTION_DEF_FOUND")


_COMPILE_RESULT_KEYWORDS = frozenset({
    "running", "time", "jobs", "terminal", "mode", "mode_switches",
    "pending_releases", "released", "misses", "ready_order",
    "frontier", "primary_on_switch_time", "abnormal_hi_releases",
    "release_demand_overrides", "ghost_future_budgets",
})


def _is_state_field(name: str) -> bool:
    return name in _COMPILE_RESULT_KEYWORDS


def _expr_to_smt(node: ast.expr) -> IntExpr:
    """Convert a Python AST expression node to an IntExpr.

    Handles: constants, names, binary ops (comparison, addition),
    if-expressions, function calls to int().
    Unsupported constructs raise ValueError with AST_UNSUPPORTED.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            if isinstance(node.value, (str, type(None))):
                raise ValueError(
                    f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:non_int_constant:{type(node.value).__name__}"
                )
            raise ValueError(
                "EXECUTABLE_TRANSITION_AST_UNSUPPORTED:non_int_constant"
            )
        return const_expr(node.value)

    if isinstance(node, ast.Name):
        return var_expr(node.id)

    if isinstance(node, ast.BinOp):
        left = _expr_to_smt(node.left)
        right = _expr_to_smt(node.right)
        if isinstance(node.op, ast.Add):
            return add_expr(left.to_smt(), right.to_smt())
        if isinstance(node.op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)):
            op = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "=", ast.NotEq: "!="}[type(node.op)]
            raise ValueError(
                f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:comparison_as_expr:{op}"
            )

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "int":
            if node.args:
                return _expr_to_smt(node.args[0])
            return const_expr(0)
        raise ValueError(
            f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:call:{node.func.id if isinstance(node.func, ast.Name) else 'complex'}"
        )

    if isinstance(node, ast.IfExp):
        cond = _bool_to_smt(node.test)
        then_val = _expr_to_smt(node.body).to_smt()
        else_val = _expr_to_smt(node.orelse).to_smt()
        return ite_expr(cond, then_val, else_val)

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            inner = _expr_to_smt(node.operand)
            return IntExpr(kind="add", left="0", right=f"(- {inner.to_smt()})")
        raise ValueError(
            "EXECUTABLE_TRANSITION_AST_UNSUPPORTED:unary_op"
        )

    if isinstance(node, ast.Subscript):
        raise ValueError(
            "EXECUTABLE_TRANSITION_AST_UNSUPPORTED:subscript"
        )

    if isinstance(node, ast.Attribute):
        raise ValueError(
            f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:attribute:{node.attr}"
        )

    raise ValueError(
        f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:expr:{type(node).__name__}"
    )


def _bool_to_smt(node: ast.expr) -> BoolExpr:
    """Convert a Python AST expression to a BoolExpr.

    Handles: boolean constants, comparisons, 'and'/'or'/'not', 'is'/'is not',
    function calls, method calls, member tests.
    """
    if isinstance(node, ast.Constant):
        if node.value is True:
            return BoolExpr(kind="atomic", left="true")
        if node.value is False:
            return BoolExpr(kind="atomic", left="false")
        raise ValueError(
            f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:bool_constant:{node.value}"
        )

    if isinstance(node, ast.Name):
        return BoolExpr(kind="atomic", left=node.id)

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError(
                "EXECUTABLE_TRANSITION_AST_UNSUPPORTED:chain_comparison"
            )
        left = _expr_to_smt(node.left).to_smt()
        right = _expr_to_smt(node.comparators[0]).to_smt()
        op = node.ops[0]
        if isinstance(op, ast.Gt):
            return cmp_expr(">", left, right)
        if isinstance(op, ast.GtE):
            return cmp_expr(">=", left, right)
        if isinstance(op, ast.Lt):
            return cmp_expr("<", left, right)
        if isinstance(op, ast.LtE):
            return cmp_expr("<=", left, right)
        if isinstance(op, ast.Eq):
            return cmp_expr("=", left, right)
        if isinstance(op, ast.NotEq):
            return cmp_expr("!=", left, right)
        if isinstance(op, ast.In):
            return BoolExpr(kind="atomic", left=f"(in {left} {right})")
        if isinstance(op, ast.NotIn):
            return BoolExpr(kind="not", children=(BoolExpr(kind="atomic", left=f"(in {left} {right})"),))
        if isinstance(op, ast.Is):
            return cmp_expr("=", left, right)
        if isinstance(op, ast.IsNot):
            return cmp_expr("!=", left, right)
        raise ValueError(
            f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:compare_op:{type(op).__name__}"
        )

    if isinstance(node, ast.BoolOp):
        children = tuple(_bool_to_smt(v) for v in node.values)
        if isinstance(node.op, ast.And):
            return BoolExpr(kind="and", children=children)
        if isinstance(node.op, ast.Or):
            return BoolExpr(kind="or", children=children)
        raise ValueError(
            f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:bool_op:{type(node.op).__name__}"
        )

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _bool_to_smt(node.operand)
        return BoolExpr(kind="not", children=(inner,))

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "recovery_is_legal":
            return BoolExpr(
                kind="and",
                children=(
                    BoolExpr(kind="atomic", left="(= mode HI)"),
                    BoolExpr(kind="atomic", left="(= active_job_count 0)"),
                    BoolExpr(kind="atomic", left="(= running_present 0)"),
                    BoolExpr(kind="atomic", left="(= pending_release_count 0)"),
                ),
            )
        raise ValueError(
            f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:call:{node.func.id if isinstance(node.func, ast.Name) else 'complex'}"
        )

    if isinstance(node, ast.Attribute):
        raise ValueError(
            f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:attribute_as_bool:{node.attr}"
        )

    if isinstance(node, ast.Subscript):
        raise ValueError(
            "EXECUTABLE_TRANSITION_AST_UNSUPPORTED:subscript_as_bool"
        )

    raise ValueError(
        f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:bool:{type(node).__name__}"
    )


def _extract_guard(body: list[ast.stmt]) -> BoolExpr:
    """Extract guard condition from function body.

    Looks for leading if/raise patterns and the first if/else block
    that branches to a non-raise result.
    """
    for stmt in body:
        if isinstance(stmt, ast.Raise):
            # raise inside body - this is an exclusion, skip it
            continue
        if isinstance(stmt, ast.If):
            test = _bool_to_smt(stmt.test)
            return test
    return empty_bool_expr()


def _extract_state_assignments(body: list[ast.stmt]) -> tuple[Assignment, ...]:
    """Extract state assignments from function body.

    Looks for:
    - replace(state, field=value) patterns
    - dict[key] = value insertions
    - dict assign/delete
    """
    results: list[Assignment] = []
    for stmt in body:
        results.extend(_extract_from_stmt(stmt))
    return tuple(results)


def _extract_from_stmt(stmt: ast.stmt) -> list[Assignment]:
    """Extract assignments from a single statement."""
    results: list[Assignment] = []

    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                try:
                    val = _expr_to_smt(stmt.value)
                    results.append(state_assignment(f"{target.id}_post", val))
                except ValueError:
                    pass

    if isinstance(stmt, ast.AugAssign):
        if isinstance(stmt.target, ast.Name):
            try:
                val = _expr_to_smt(stmt.value)
                left = var_expr(stmt.target.id).to_smt()
                right = val.to_smt()
                op_node = stmt.op
                if isinstance(op_node, ast.Add):
                    results.append(state_assignment(f"{stmt.target.id}_post", add_expr(left, right)))
                elif isinstance(op_node, ast.Sub):
                    results.append(state_assignment(f"{stmt.target.id}_post", add_expr(left, f"(- {right})")))
            except ValueError:
                pass

    if isinstance(stmt, ast.If):
        # Recurse into if/else body
        try:
            results.extend(_extract_state_assignments(stmt.body))
        except ValueError:
            pass
        try:
            results.extend(_extract_state_assignments(stmt.orelse))
        except ValueError:
            pass

    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Call):
            results.extend(_extract_from_call(stmt.value))

    return results


def _extract_from_call(node: ast.Call) -> list[Assignment]:
    """Extract assignments from a function call like replace()."""
    results: list[Assignment] = []
    if isinstance(node.func, ast.Name):
        if node.func.id == "replace":
            for kw in node.keywords:
                if kw.arg and kw.arg != "running" and kw.arg != "ready_order":
                    try:
                        val = _expr_to_smt(kw.value)
                        results.append(state_assignment(f"{kw.arg}_post", val))
                    except ValueError:
                        pass
    return results


def _extract_frame_fields(body: list[ast.stmt]) -> frozenset[str]:
    """Extract protected frame fields - fields NOT modified by the function.

    Inspects the body for field names that appear on the left side of
    assignments or as 'replace' keyword arguments, and excludes them.
    """
    written: set[str] = _collect_modified_fields(body)
    all_protected = {
        "release_time", "absolute_deadline", "criticality", "fixed_demand",
        "priority_index", "task_name", "hi_class", "job_key", "service",
    }
    return frozenset(all_protected - written)


def _collect_modified_fields(body: list[ast.stmt]) -> set[str]:
    """Collect field names that are modified in the function body."""
    modified: set[str] = set()
    for stmt in body:
        _collect_from_stmt(stmt, modified)
    return modified


def _collect_from_stmt(stmt: ast.stmt, modified: set[str]) -> None:
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                modified.add(target.id)
    if isinstance(stmt, ast.AugAssign):
        if isinstance(stmt.target, ast.Name):
            modified.add(stmt.target.id)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "replace":
            for kw in stmt.value.keywords:
                if kw.arg:
                    modified.add(kw.arg)
    if isinstance(stmt, ast.If):
        for s in stmt.body:
            _collect_from_stmt(s, modified)
        for s in stmt.orelse:
            _collect_from_stmt(s, modified)


def _extract_time_update(body: list[ast.stmt]) -> IntExpr:
    """Extract time update expression from function body.

    Looks for patterns like replace(state, time=state.time+1) or
    explicit time assignments.  Default: time unchanged.
    """
    for stmt in body:
        if isinstance(stmt, ast.Return) and stmt.value:
            time_expr = _collect_time_from_expr(stmt.value)
            if time_expr is not None:
                return time_expr
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            time_expr = _collect_time_from_call(stmt.value)
            if time_expr is not None:
                return time_expr
        if isinstance(stmt, ast.If):
            for branch in stmt.body + stmt.orelse:
                if isinstance(branch, ast.Return) and branch.value:
                    time_expr = _collect_time_from_expr(branch.value)
                    if time_expr is not None:
                        return time_expr
                if isinstance(branch, ast.Expr) and isinstance(branch.value, ast.Call):
                    time_expr = _collect_time_from_call(branch.value)
                    if time_expr is not None:
                        return time_expr
    return var_expr("time_pre")


def _collect_time_from_expr(node: ast.expr) -> IntExpr | None:
    """Look for time=... in replace() calls."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "replace":
        for kw in node.keywords:
            if kw.arg == "time":
                try:
                    return _expr_to_smt(kw.value)
                except ValueError:
                    return None
        # Nested replace
        if node.args:
            return _collect_time_from_expr(node.args[0])
    return None


def _collect_time_from_call(node: ast.Call) -> IntExpr | None:
    if isinstance(node.func, ast.Name) and node.func.id == "replace":
        for kw in node.keywords:
            if kw.arg == "time":
                try:
                    return _expr_to_smt(kw.value)
                except ValueError:
                    return None
        if node.args:
            return _collect_time_from_expr(node.args[0])
    return None


def _extract_generated_events(body: list[ast.stmt]) -> tuple[GeneratedEventRule, ...]:
    """Extract generated event rules from function body.

    Looks for _append_generated_event calls and LogicalEvent constructors.
    """
    events: list[GeneratedEventRule] = []
    for stmt in body:
        _collect_events_from_stmt(stmt, events)
    return tuple(events)


def _collect_events_from_stmt(stmt: ast.stmt, events: list[GeneratedEventRule]) -> None:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        _collect_events_from_call(stmt.value, events)
    if isinstance(stmt, ast.If):
        for s in stmt.body + stmt.orelse:
            _collect_events_from_stmt(s, events)
    if isinstance(stmt, ast.Assign):
        # Look for LogicalEvent(...) constructors
        if isinstance(stmt.value, ast.Call):
            _collect_events_from_call(stmt.value, events)


def _collect_events_from_call(node: ast.Call, events: list[GeneratedEventRule]) -> None:
    if isinstance(node.func, ast.Name):
        if node.func.id == "_append_generated_event":
            kw = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            gen = kw.get("generated_event")
            if gen and isinstance(gen, ast.Call):
                kind_kw = {kw.arg: kw.value for kw in gen.keywords if kw.arg}
                time_val = kind_kw.get("time")
                kind_val = kind_kw.get("kind")
                jk_val = kind_kw.get("job_key")
                if time_val and kind_val:
                    try:
                        time_expr = _expr_to_smt(time_val)
                        kind_name = _resolve_event_kind(kind_val)
                        jk_smt = None
                        if jk_val:
                            try:
                                jk_smt = _expr_to_smt(jk_val).to_smt()
                            except ValueError:
                                jk_smt = str(jk_val)
                        events.append(GeneratedEventRule(
                            event_kind=kind_name,
                            time_expr=time_expr,
                            job_key_expr=jk_smt,
                        ))
                    except ValueError:
                        return
        if node.func.id == "LogicalEvent":
            kw = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            time_val = kw.get("time")
            kind_val = kw.get("kind")
            jk_val = kw.get("job_key")
            if time_val and kind_val:
                try:
                    time_expr = _expr_to_smt(time_val)
                    kind_name = _resolve_event_kind(kind_val)
                    jk_smt = None
                    if jk_val:
                        try:
                            jk_smt = _expr_to_smt(jk_val).to_smt()
                        except ValueError:
                            jk_smt = str(jk_val)
                    events.append(GeneratedEventRule(
                        event_kind=kind_name,
                        time_expr=time_expr,
                        job_key_expr=jk_smt,
                    ))
                except ValueError:
                    return


def _resolve_event_kind(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return "UNKNOWN_EVENT_KIND"
    return "UNKNOWN_EVENT_KIND"


def _compute_ast_hash(source: str) -> str:
    return sha256_object({"source_ast": str(ast.dump(ast.parse(source)))})


def _compilation_receipt(
    *, function_name: str, source: str, ast_hash: str, node: ast.FunctionDef,
    unsupported: tuple[str, ...], covered: bool,
    state_fields: tuple[str, ...] = (), frame_fields: frozenset[str] = frozenset(),
) -> TransitionCompilationReceipt:
    returns = tuple(n for n in ast.walk(node) if isinstance(n, ast.Return))
    raises = tuple(n for n in ast.walk(node) if isinstance(n, ast.Raise))
    helpers = sorted({n.func.id for n in ast.walk(node)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)})
    receipt = TransitionCompilationReceipt(
        source_function=function_name,
        source_ast_hash=ast_hash,
        normalized_cfg_hash=sha256_object({"ast": ast.dump(node, annotate_fields=True)}),
        return_path_count=len(returns),
        covered_return_path_count=len(returns) if covered else 0,
        raise_path_count=len(raises),
        covered_raise_path_count=len(raises) if covered else 0,
        helper_calls=tuple(helpers),
        helper_summaries_bound=tuple(h for h in helpers if h in {
            "replace", "pop_event", "_append_generated_event", "_canonical_frontier",
        }),
        state_fields_read=tuple(sorted(state_fields)),
        state_fields_written=tuple(sorted(state_fields)),
        frame_fields=tuple(sorted(frame_fields)),
        unsupported_nodes=tuple(sorted(set(unsupported))),
        total_semantic_coverage=bool(covered and not unsupported),
    )
    return receipt


def compile_function(function_name: str) -> CompiledTransitionIR:
    """Compile a single transition function into CompiledTransitionIR.

    Reads the source code, parses the AST, and extracts:
    - precondition (guard)
    - post_equations (state updates)
    - frame_fields (unchanged fields)
    - time_update
    - generated_events

    The current extractor is deliberately diagnostic-only: it does not yet
    prove that every return path, container mutation, helper call, exception
    guard, and frame equation has been translated.  Therefore a syntactically
    parsed function is reported as ``PARTIAL_AST_EXTRACTION`` rather than
    ``COMPILED``.  Only a future total compiler with an independently checked
    semantic-coverage receipt may emit ``COMPILED``.
    """
    config = COMPILATION_MAP.get(function_name)
    if config is None:
        raise ValueError(f"COMPILER_UNKNOWN_FUNCTION:{function_name}")

    func = getattr(executable_semantics, function_name, None)
    if func is None:
        raise ValueError(f"COMPILER_FUNCTION_NOT_FOUND:{function_name}")

    source, source_hash = _get_source(func)
    ast_hash = _compute_ast_hash(source)

    try:
        func_node = _parse_function(source)
    except (ValueError, SyntaxError) as exc:
        receipt = TransitionCompilationReceipt(
            source_function=function_name, source_ast_hash=ast_hash,
            normalized_cfg_hash=sha256_object({"error": str(exc)}),
            return_path_count=0, covered_return_path_count=0,
            raise_path_count=0, covered_raise_path_count=0,
            unsupported_nodes=(str(exc),), total_semantic_coverage=False,
        )
        return CompiledTransitionIR(
            case_id=config["case_id"],
            source_module=_SOURCE_MODULE,
            source_function=function_name,
            source_function_ast_hash=ast_hash,
            precondition=empty_bool_expr(),
            post_equations=(),
            frame_fields=frozenset(),
            generated_events=(),
            time_update=var_expr("time_pre"),
            compilation_receipt_hash=sha256_object({"error": str(exc)}),
            compilation_status="AST_UNSUPPORTED",
            binding_kind="EXECUTABLE_TRANSITION_COMPILER",
            compilation_receipt=receipt,
        )

    try:
        guard = _extract_guard(func_node.body)
    except ValueError as exc:
        receipt = TransitionCompilationReceipt(
            source_function=function_name, source_ast_hash=ast_hash,
            normalized_cfg_hash=sha256_object({"stage": "guard", "error": str(exc)}),
            return_path_count=sum(isinstance(n, ast.Return) for n in ast.walk(func_node)),
            covered_return_path_count=0,
            raise_path_count=sum(isinstance(n, ast.Raise) for n in ast.walk(func_node)),
            covered_raise_path_count=0,
            unsupported_nodes=(str(exc),), total_semantic_coverage=False,
        )
        return CompiledTransitionIR(
            case_id=config["case_id"],
            source_module=_SOURCE_MODULE,
            source_function=function_name,
            source_function_ast_hash=ast_hash,
            precondition=empty_bool_expr(),
            post_equations=(),
            frame_fields=frozenset(),
            generated_events=(),
            time_update=var_expr("time_pre"),
            compilation_receipt_hash=sha256_object({"error": str(exc), "stage": "guard"}),
            compilation_status="AST_UNSUPPORTED",
            binding_kind="EXECUTABLE_TRANSITION_COMPILER",
            compilation_receipt=receipt,
        )

    try:
        state_eqs = _extract_state_assignments(func_node.body)
    except ValueError:
        state_eqs = ()

    try:
        frame_fields = _extract_frame_fields(func_node.body)
    except ValueError:
        frame_fields = frozenset()

    try:
        time_update = _extract_time_update(func_node.body)
    except ValueError:
        time_update = var_expr("time_pre")

    try:
        gen_events = _extract_generated_events(func_node.body)
    except ValueError:
        gen_events = ()

    unsupported = []
    if not state_eqs:
        unsupported.append("state_update_equations")
    if not gen_events and function_name in {"apply_arrival_batch", "apply_service_tick"}:
        unsupported.append("generated_event_rules")
    receipt = _compilation_receipt(
        function_name=function_name, source=source, ast_hash=ast_hash,
        node=func_node, unsupported=tuple(unsupported), covered=False,
        state_fields=_COMPILE_RESULT_KEYWORDS, frame_fields=frame_fields,
    )
    receipt_hash = sha256_object({
        "source_hash": source_hash,
        "ast_hash": ast_hash,
        "guard": guard.to_smt(),
        "state_eqs": [(a.target, a.expression.to_smt(), a.kind) for a in state_eqs],
        "frame_fields": sorted(frame_fields),
        "time_update": time_update.to_smt(),
        "gen_events": [(e.event_kind, e.time_expr.to_smt()) for e in gen_events],
        "compilation_receipt": receipt.to_dict(),
    })

    return CompiledTransitionIR(
        case_id=config["case_id"],
        source_module=_SOURCE_MODULE,
        source_function=function_name,
        source_function_ast_hash=ast_hash,
        precondition=guard,
        post_equations=state_eqs,
        frame_fields=frame_fields,
        generated_events=gen_events,
        time_update=time_update,
        compilation_receipt_hash=receipt_hash,
        compilation_status="PARTIAL_AST_EXTRACTION",
        binding_kind="EXECUTABLE_TRANSITION_COMPILER",
        compilation_receipt=receipt,
    )


def compile_all_transitions() -> tuple[CompiledTransitionIR, ...]:
    """Compile all primitive transition functions.

    Returns a tuple of (successful, unsuccessful) tuples.
    """
    results: list[CompiledTransitionIR] = []
    for func_name in COMPILATION_MAP:
        try:
            ir = compile_function(func_name)
        except (ValueError, TypeError, AttributeError) as exc:
            config = COMPILATION_MAP[func_name]
            ir = CompiledTransitionIR(
                case_id=config["case_id"],
                source_module=_SOURCE_MODULE,
                source_function=func_name,
                source_function_ast_hash=sha256_object({"error": str(exc)}),
                precondition=empty_bool_expr(),
                post_equations=(),
                frame_fields=frozenset(),
                generated_events=(),
                time_update=var_expr("time_pre"),
                compilation_receipt_hash=sha256_object({"error": str(exc)}),
                compilation_status="UNRESOLVED",
                binding_kind="EXECUTABLE_TRANSITION_COMPILER",
            )
        results.append(ir)
    return tuple(results)


def compiled_ir_map() -> dict[str, CompiledTransitionIR]:
    """Return a map from case_id to CompiledTransitionIR."""
    return {ir.case_id: ir for ir in compile_all_transitions()}


def compiled_ir_for_case(case_id: str) -> CompiledTransitionIR | None:
    return compiled_ir_map().get(case_id)
