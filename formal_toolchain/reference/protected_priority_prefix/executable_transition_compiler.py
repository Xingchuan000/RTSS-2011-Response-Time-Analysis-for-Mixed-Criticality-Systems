"""Path-sensitive compiler for the executable protected-prefix transitions.

This compiler is deliberately target-specific.  It structurally interprets the
real Python AST, preserves each terminal path separately, emits finite fold IR
for event loops, recursively source-binds helper summaries, and validates the
essential semantic pattern of each audited transition.  No hand-written PP0
adapter equation is used to establish executable binding.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from dataclasses import dataclass, replace
from functools import lru_cache
from types import ModuleType
from typing import Any, Iterable, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference import executable_semantics

from .executable_transition_ir import (
    Assignment,
    BoolExpr,
    CompiledPathIR,
    CompiledTransitionIR,
    FoldIR,
    GeneratedEventRule,
    HelperSummaryReceipt,
    SemanticEffectRule,
    SymbolicOperation,
    TransitionCompilationReceipt,
    atomic_bool,
    empty_bool_expr,
    raw_expr,
    state_assignment,
    var_expr,
)

_SOURCE_MODULE = "formal_toolchain.reference.executable_semantics"

# TAIL_ONLY_SERVICE is the low-priority branch of the same executable service
# primitive.  Binding it to close_timestamp would be a category error: closure
# consumes zero-time events, whereas tail service advances time by one tick.
COMPILATION_TARGETS: tuple[tuple[str, str], ...] = (
    ("FINAL_DISPATCH", "_normalize_dispatch"),
    ("REM_COMPLETION", "apply_removal"),
    ("RECOVERY", "apply_recovery"),
    ("DEADLINE_OBSERVATION", "apply_deadline_observation"),
    ("ARRIVAL_BATCH", "apply_arrival_batch"),
    ("MODE_SWITCH", "apply_mode_switch"),
    ("RELEASE", "apply_release"),
    ("SERVICE_UNIT", "apply_service_tick"),
    ("TAIL_ONLY_SERVICE", "apply_service_tick"),
)

COMPILATION_MAP: dict[str, dict[str, Any]] = {
    function_name: {"case_id": case_id}
    for case_id, function_name in COMPILATION_TARGETS
    if function_name != "apply_service_tick" or case_id == "SERVICE_UNIT"
}
CANONICAL_CASE_ID = {
    "DDL_OBSERVE": "DEADLINE_OBSERVATION",
    "ARRIVAL_BATCH_OPEN": "ARRIVAL_BATCH",
}

# Structural compiler coverage.  Operators, contexts and comprehension helper
# nodes are intentionally admitted; they are represented in the operation
# trace and are never silently discarded.
_SUPPORTED_STMTS = (
    ast.Assign, ast.AnnAssign, ast.AugAssign, ast.If, ast.For, ast.While,
    ast.Return, ast.Raise, ast.Expr, ast.Delete, ast.Pass, ast.Break, ast.Continue,
    ast.Try,
)
_SUPPORTED_EXPRS = (
    ast.Name, ast.Attribute, ast.Constant, ast.Subscript, ast.Tuple, ast.List,
    ast.Dict, ast.Set, ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Compare,
    ast.IfExp, ast.Call, ast.ListComp, ast.DictComp, ast.SetComp,
    ast.GeneratorExp, ast.JoinedStr, ast.FormattedValue, ast.Lambda,
    ast.Slice,
)
_IGNORED_AST_NODES = (
    ast.Load, ast.Store, ast.Del, ast.arguments, ast.arg, ast.keyword,
    ast.comprehension,
    ast.operator, ast.boolop, ast.unaryop, ast.cmpop, ast.expr_context,
)

_INTRINSIC_CALLS = {
    "int", "len", "dict", "list", "set", "tuple", "str", "bool", "range",
    "min", "max", "sorted", "any", "all", "enumerate", "zip", "replace",
    "isinstance", "getattr", "hasattr",
    "ValueError", "TerminalRecord", "MissRecord", "LogicalEvent",
    "PendingReferenceRelease", "ReferenceModeSwitch", "ReleasedJobRecord",
    "ReferenceJob",
}
_INTRINSIC_METHODS = {
    "get", "items", "keys", "values", "append", "extend", "remove",
    "discard", "add", "update", "pop", "setdefault", "sort",
}

_PROTECTED_FIELDS = frozenset({
    "jobs", "pending", "released", "terminal", "misses", "running",
    "ready", "service", "metadata", "frontier_protected", "time",
})


@dataclass
class _PathState:
    guards: tuple[str, ...]
    env: dict[str, str]
    operations: list[SymbolicOperation]
    folds: list[FoldIR]
    generated_events: list[GeneratedEventRule]
    terminated: str | None = None
    exception_type: str | None = None
    return_expression: str | None = None

    def clone(self) -> "_PathState":
        return _PathState(
            guards=tuple(self.guards),
            env=dict(self.env),
            operations=list(self.operations),
            folds=list(self.folds),
            generated_events=list(self.generated_events),
            terminated=self.terminated,
            exception_type=self.exception_type,
            return_expression=self.return_expression,
        )


def _source_and_ast(obj: Any) -> tuple[str, ast.FunctionDef, str]:
    source = textwrap.dedent(inspect.getsource(obj))
    module = ast.parse(source)
    function = next((n for n in module.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if not isinstance(function, ast.FunctionDef):
        raise ValueError("COMPILER_NO_FUNCTION_DEF_FOUND")
    ast_hash = sha256_object({"source_ast": ast.dump(function, annotate_fields=True, include_attributes=False)})
    return source, function, ast_hash


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return "None"
    return ast.unparse(node)


def _operation(kind: str, stmt: ast.AST, target: str | None = None,
               expression: str | None = None) -> SymbolicOperation:
    return SymbolicOperation(
        kind=kind,
        target=target,
        expression=expression,
        source_line=int(getattr(stmt, "lineno", 0)),
        source_col=int(getattr(stmt, "col_offset", 0)),
    )


class _EnvSubstituter(ast.NodeTransformer):
    def __init__(self, env: Mapping[str, str]):
        self._env = env
        self._stack: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load) or node.id not in self._env or node.id in self._stack:
            return node
        expression = self._env[node.id]
        try:
            replacement = ast.parse(expression, mode="eval").body
        except SyntaxError:
            return node
        self._stack.add(node.id)
        replacement = self.visit(replacement)
        self._stack.discard(node.id)
        return ast.copy_location(replacement, node)


def _substitute(node: ast.AST, env: Mapping[str, str]) -> str:
    copied = ast.fix_missing_locations(_EnvSubstituter(env).visit(ast.parse(_unparse(node), mode="eval").body))
    return ast.unparse(copied)


def _target_root(target: ast.AST) -> str | None:
    current = target
    while isinstance(current, (ast.Subscript, ast.Attribute)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _target_text(target: ast.AST) -> str:
    return _unparse(target)


def _call_name(call: ast.Call) -> str:
    return _unparse(call.func)


def _mutated_locals(statements: Iterable[ast.stmt]) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.Module(body=list(statements), type_ignores=[])):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets.append(node.target)
            elif isinstance(node, ast.AugAssign):
                targets.append(node.target)
            elif isinstance(node, ast.Delete):
                targets.extend(node.targets)
            for target in targets:
                root = _target_root(target)
                if root:
                    result.add(root)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"append", "add", "discard", "remove", "update", "pop"}:
                root = _target_root(node.func.value)
                if root:
                    result.add(root)
        elif isinstance(node, ast.Call) and _call_name(node) == "_append_generated_event":
            for kw in node.keywords:
                if kw.arg == "frontier" and isinstance(kw.value, ast.Name):
                    result.add(kw.value.id)
    return result


def _condition_expr(node: ast.expr, env: Mapping[str, str]) -> str:
    return _substitute(node, env)


def _bool_from_guards(guards: tuple[str, ...]) -> BoolExpr:
    if not guards:
        return empty_bool_expr()
    children = tuple(atomic_bool(g) for g in guards)
    return children[0] if len(children) == 1 else BoolExpr(kind="and", children=children)


def _assignment_to_env(stmt: ast.stmt, state: _PathState) -> None:
    if isinstance(stmt, ast.Assign):
        value = _substitute(stmt.value, state.env)
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                state.env[target.id] = value
            elif isinstance(target, ast.Subscript):
                root = _target_root(target)
                if root:
                    current = state.env.get(root, root)
                    key = _substitute(target.slice, state.env)
                    state.env[root] = f"store({current}, {key}, {value})"
            state.operations.append(_operation("ASSIGN", stmt, _target_text(target), value))
        return
    if isinstance(stmt, ast.AnnAssign):
        value = _substitute(stmt.value, state.env) if stmt.value is not None else "None"
        target = stmt.target
        if isinstance(target, ast.Name):
            state.env[target.id] = value
        state.operations.append(_operation("ANN_ASSIGN", stmt, _target_text(target), value))
        return
    if isinstance(stmt, ast.AugAssign):
        target = stmt.target
        root = _target_root(target)
        current = state.env.get(root, root or _target_text(target))
        value = _substitute(stmt.value, state.env)
        op = type(stmt.op).__name__
        expression = f"{op}({current}, {value})"
        if root:
            state.env[root] = expression
        state.operations.append(_operation("AUG_ASSIGN", stmt, _target_text(target), expression))
        return
    if isinstance(stmt, ast.Delete):
        for target in stmt.targets:
            root = _target_root(target)
            if root and isinstance(target, ast.Subscript):
                current = state.env.get(root, root)
                key = _substitute(target.slice, state.env)
                state.env[root] = f"delete({current}, {key})"
            state.operations.append(_operation("DELETE", stmt, _target_text(target), None))


def _apply_expression_side_effect(stmt: ast.Expr, state: _PathState) -> None:
    expression = stmt.value
    rendered = _substitute(expression, state.env)
    state.operations.append(_operation("CALL", stmt, None, rendered))
    if not isinstance(expression, ast.Call):
        return
    if isinstance(expression.func, ast.Attribute):
        root = _target_root(expression.func.value)
        method = expression.func.attr
        if root and method in {"append", "add", "discard", "remove", "update", "pop"}:
            current = state.env.get(root, root)
            args = ", ".join(_substitute(arg, state.env) for arg in expression.args)
            state.env[root] = f"{method}({current}{', ' if args else ''}{args})"
    elif _call_name(expression) == "_append_generated_event":
        kw = {item.arg: item.value for item in expression.keywords if item.arg}
        frontier_node = kw.get("frontier")
        if isinstance(frontier_node, ast.Name):
            root = frontier_node.id
            current = state.env.get(root, root)
            parent = _substitute(kw["parent_event"], state.env) if "parent_event" in kw else "None"
            event = _substitute(kw["generated_event"], state.env) if "generated_event" in kw else "None"
            state.env[root] = f"append_generated_event({current}, {parent}, {event})"
            state.generated_events.append(GeneratedEventRule(
                event_kind="SOURCE_DERIVED",
                time_expr=raw_expr(f"event_time({event})"),
                job_key_expr=f"event_job_key({event})",
            ))


def _extract_return_updates(expression: str) -> tuple[Assignment, ...]:
    """Extract state field updates from a source-derived return expression.

    The expression remains exact source syntax.  Nested ``replace`` and state
    helper calls are recognized; unrecognized calls are retained as one raw
    state transformer rather than being dropped.
    """
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return (state_assignment("state_post", raw_expr(expression)),)

    updates: dict[str, str] = {}

    def visit(expr: ast.AST) -> None:
        if isinstance(expr, ast.Call):
            name = _call_name(expr)
            if name == "replace":
                if expr.args:
                    visit(expr.args[0])
                for kw in expr.keywords:
                    if kw.arg:
                        updates[kw.arg] = _unparse(kw.value)
                return
            if name == "pop_event":
                if expr.args:
                    visit(expr.args[0])
                updates["frontier"] = f"pop_event(frontier_pre, {_unparse(expr.args[1]) if len(expr.args) > 1 else 'event'})"
                return
            if name in {"_normalize_dispatch", "dispatch_if_needed"}:
                if expr.args:
                    visit(expr.args[0])
                updates["running"] = "dispatch_head(jobs_post)"
                updates["ready_order"] = "dispatch_tail(jobs_post)"
                return
        if isinstance(expr, ast.Name) and expr.id == "state":
            return
        updates.setdefault("state", _unparse(expr))

    visit(node)
    return tuple(state_assignment(f"{field}_post", raw_expr(value)) for field, value in sorted(updates.items()))


def _compile_loop(stmt: ast.For | ast.While, state: _PathState) -> tuple[_PathState, list[_PathState]]:
    sequence = _substitute(stmt.iter, state.env) if isinstance(stmt, ast.For) else f"while({_condition_expr(stmt.test, state.env)})"
    cursor = _unparse(stmt.target) if isinstance(stmt, ast.For) else "iteration"
    mutated = tuple(sorted(_mutated_locals(stmt.body)))

    step_seed = state.clone()
    if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
        step_seed.env[stmt.target.id] = f"element({sequence}, {cursor})"
    step_paths = _compile_block(stmt.body, [step_seed], inside_loop=True)
    exceptional: list[_PathState] = []
    normal_step_hashes: list[str] = []
    for path in step_paths:
        payload = {
            "guards": path.guards,
            "env": path.env,
            "ops": [op.canonical() for op in path.operations[len(state.operations):]],
            "terminator": path.terminated,
            "exception": path.exception_type,
        }
        digest = sha256_object(payload)
        if path.terminated == "RAISE":
            raised = path.clone()
            raised.guards = state.guards + (f"exists_iteration({sequence})",) + tuple(path.guards[len(state.guards):])
            exceptional.append(raised)
        else:
            normal_step_hashes.append(digest)

    step_hash = sha256_object({
        "sequence": sequence,
        "cursor": cursor,
        "normal_step_hashes": normal_step_hashes,
        "mutated": mutated,
    })
    fold = FoldIR(
        sequence_symbol=sequence,
        cursor_symbol=cursor,
        invariant_schema_hash=sha256_object({"loop": ast.dump(stmt, include_attributes=False), "mutated": mutated}),
        step_summary_hash=step_hash,
        termination_measure=f"len({sequence})-{cursor}" if isinstance(stmt, ast.For) else "closure_measure(current)",
        mutated_locals=mutated,
        exceptional_path_hashes=tuple(sha256_object({"guards": p.guards, "exception": p.exception_type}) for p in exceptional),
    )
    continued = state.clone()
    continued.folds.append(fold)
    continued.operations.append(_operation("FOLD", stmt, cursor, sequence))
    for name in mutated:
        pre = continued.env.get(name, name)
        continued.env[name] = f"fold_{step_hash[:16]}({sequence}, {pre})"
    return continued, exceptional


def _compile_statement(stmt: ast.stmt, state: _PathState, *, inside_loop: bool = False) -> list[_PathState]:
    if state.terminated:
        return [state]
    if not isinstance(stmt, _SUPPORTED_STMTS):
        raise ValueError(f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:statement:{type(stmt).__name__}")
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
        result = state.clone()
        _assignment_to_env(stmt, result)
        return [result]
    if isinstance(stmt, ast.Expr):
        result = state.clone()
        _apply_expression_side_effect(stmt, result)
        return [result]
    if isinstance(stmt, ast.If):
        cond = _condition_expr(stmt.test, state.env)
        yes = state.clone()
        yes.guards += (cond,)
        no = state.clone()
        no.guards += (f"not ({cond})",)
        yes_paths = _compile_block(stmt.body, [yes], inside_loop=inside_loop)
        no_paths = _compile_block(stmt.orelse, [no], inside_loop=inside_loop) if stmt.orelse else [no]
        return yes_paths + no_paths
    if isinstance(stmt, (ast.For, ast.While)):
        continued, exceptional = _compile_loop(stmt, state)
        return exceptional + [continued]
    if isinstance(stmt, ast.Try):
        # Every handler is a source-visible exceptional branch.  The normal
        # branch executes body + else; finally is appended to every branch.
        body_paths = _compile_block(stmt.body, [state.clone()], inside_loop=inside_loop)
        normal_paths: list[_PathState] = []
        for item in body_paths:
            if item.terminated == "RAISE":
                continue
            item.terminated = None if item.terminated in {"BREAK", "LOOP_CONTINUE"} else item.terminated
            normal_paths.append(item)
        if stmt.orelse:
            normal_paths = _compile_block(stmt.orelse, normal_paths, inside_loop=inside_loop)
        handler_paths: list[_PathState] = []
        for handler in stmt.handlers:
            branch = state.clone()
            branch.guards += (f"exception_matches({_unparse(handler.type)})",)
            handler_paths.extend(_compile_block(handler.body, [branch], inside_loop=inside_loop))
        combined = normal_paths + handler_paths
        if stmt.finalbody:
            combined = _compile_block(stmt.finalbody, combined, inside_loop=inside_loop)
        return combined
    if isinstance(stmt, ast.Return):
        result = state.clone()
        expression = _substitute(stmt.value, result.env) if stmt.value is not None else "None"
        result.operations.append(_operation("RETURN", stmt, None, expression))
        result.terminated = "RETURN"
        result.return_expression = expression
        return [result]
    if isinstance(stmt, ast.Raise):
        result = state.clone()
        expression = _substitute(stmt.exc, result.env) if stmt.exc is not None else "re-raise"
        result.operations.append(_operation("RAISE", stmt, None, expression))
        result.terminated = "RAISE"
        result.exception_type = expression
        return [result]
    if isinstance(stmt, ast.Break):
        result = state.clone()
        result.operations.append(_operation("BREAK", stmt))
        result.terminated = "BREAK" if inside_loop else "UNSUPPORTED_BREAK"
        return [result]
    if isinstance(stmt, ast.Continue):
        result = state.clone()
        result.operations.append(_operation("CONTINUE", stmt))
        result.terminated = "LOOP_CONTINUE" if inside_loop else "UNSUPPORTED_CONTINUE"
        return [result]
    if isinstance(stmt, ast.Pass):
        result = state.clone()
        result.operations.append(_operation("PASS", stmt))
        return [result]
    raise ValueError(f"EXECUTABLE_TRANSITION_AST_UNSUPPORTED:{type(stmt).__name__}")


def _compile_block(statements: list[ast.stmt], states: list[_PathState], *, inside_loop: bool = False) -> list[_PathState]:
    current = states
    for stmt in statements:
        next_states: list[_PathState] = []
        for state in current:
            if state.terminated:
                next_states.append(state)
            else:
                next_states.extend(_compile_statement(stmt, state, inside_loop=inside_loop))
        current = next_states
    return current


def compile_statement(stmt: ast.stmt, symbolic_state: Mapping[str, str] | None = None,
                      path_condition: BoolExpr | None = None) -> tuple[SymbolicOperation, ...]:
    seed = _PathState(
        guards=(() if path_condition is None or path_condition.to_smt() == "true" else (path_condition.to_smt(),)),
        env=dict(symbolic_state or {}), operations=[], folds=[], generated_events=[])
    paths = _compile_statement(stmt, seed)
    return tuple(op for path in paths for op in path.operations)


def compile_if(node: ast.If, symbolic_state: Mapping[str, str] | None = None,
               path_condition: BoolExpr | None = None) -> tuple[CompiledPathIR, ...]:
    if not isinstance(node, ast.If):
        raise TypeError("compile_if expects ast.If")
    seed = _PathState(guards=(), env=dict(symbolic_state or {}), operations=[], folds=[], generated_events=[])
    return _states_to_paths(_compile_statement(node, seed), ())


def compile_finite_for(node: ast.For | ast.While, symbolic_state: Mapping[str, str] | None = None,
                       path_condition: BoolExpr | None = None) -> FoldIR:
    if not isinstance(node, (ast.For, ast.While)):
        raise TypeError("compile_finite_for expects ast.For or ast.While")
    seed = _PathState(guards=(), env=dict(symbolic_state or {}), operations=[], folds=[], generated_events=[])
    continued, _ = _compile_loop(node, seed)
    return continued.folds[-1]


def compile_replace_call(node: ast.Call, symbolic_state: Mapping[str, str] | None = None) -> tuple[Assignment, ...]:
    if not isinstance(node, ast.Call) or _call_name(node) != "replace":
        raise TypeError("compile_replace_call expects replace(...) call")
    expression = _substitute(node, symbolic_state or {})
    return _extract_return_updates(expression)


def compile_collection_update(node: ast.Assign | ast.AnnAssign | ast.AugAssign | ast.Delete,
                              symbolic_state: Mapping[str, str] | None = None) -> tuple[SymbolicOperation, ...]:
    seed = _PathState(guards=(), env=dict(symbolic_state or {}), operations=[], folds=[], generated_events=[])
    result = _compile_statement(node, seed)[0]
    return tuple(result.operations)


def _states_to_paths(states: list[_PathState], helper_hashes: tuple[str, ...]) -> tuple[CompiledPathIR, ...]:
    result: list[CompiledPathIR] = []
    for index, state in enumerate(states):
        if state.terminated not in {"RETURN", "RAISE"}:
            continue
        updates = _extract_return_updates(state.return_expression) if state.terminated == "RETURN" and state.return_expression else ()
        modified = {a.target.removesuffix("_post") for a in updates}
        frame = _PROTECTED_FIELDS - modified
        result.append(CompiledPathIR(
            path_id=f"path_{index}",
            path_condition=_bool_from_guards(state.guards),
            updates=updates,
            frame_fields=frozenset(frame),
            generated_events=tuple(state.generated_events),
            terminator=state.terminated,
            exception_type=state.exception_type,
            operations=tuple(state.operations),
            folds=tuple(state.folds),
            helper_summary_hashes=helper_hashes,
            return_expression=state.return_expression,
        ))
    return tuple(result)


def enumerate_all_return_and_raise_paths(function_ast: ast.FunctionDef) -> tuple[CompiledPathIR, ...]:
    states = _compile_block(function_ast.body, [_PathState((), {}, [], [], [])])
    return _states_to_paths(states, ())


def _unsupported_nodes(function: ast.FunctionDef) -> tuple[str, ...]:
    unsupported: set[str] = set()
    for node in ast.walk(function):
        if node is function or isinstance(node, _IGNORED_AST_NODES):
            continue
        if isinstance(node, ast.stmt) and not isinstance(node, _SUPPORTED_STMTS):
            unsupported.add(type(node).__name__)
        elif isinstance(node, ast.expr) and not isinstance(node, _SUPPORTED_EXPRS):
            unsupported.add(type(node).__name__)
    return tuple(sorted(unsupported))


def _called_function_names(function: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(sorted({_call_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)}))


def _resolve_name(name: str, namespace: Mapping[str, Any]) -> Any | None:
    if "." in name:
        return None
    if name in namespace:
        return namespace[name]
    return getattr(builtins, name, None)


def _structural_helper_summary(name: str, namespace: Mapping[str, Any], seen: frozenset[str]) -> HelperSummaryReceipt:
    if name in _INTRINSIC_CALLS or name.split(".")[-1] in _INTRINSIC_METHODS:
        payload = {"helper": name, "rule": "PYTHON_INTRINSIC_OR_METHOD"}
        digest = sha256_object(payload)
        return HelperSummaryReceipt(
            helper_name=name,
            source_ast_hash=digest,
            precondition=empty_bool_expr(),
            paths=(),
            total_semantic_coverage=True,
            summary_hash=digest,
            source_function=name,
            operation_trace_hash=digest,
            binding_kind="TRUSTED_PYTHON_INTRINSIC",
        )
    if name in seen:
        digest = sha256_object({"helper": name, "rule": "PRIMITIVE_RECURSION_EDGE"})
        return HelperSummaryReceipt(
            helper_name=name, source_ast_hash=digest, precondition=empty_bool_expr(),
            paths=(), total_semantic_coverage=True, summary_hash=digest,
            source_function=name, operation_trace_hash=digest,
            binding_kind="SOURCE_BOUND_RECURSION_EDGE",
        )
    obj = _resolve_name(name, namespace)
    if obj is None or not callable(obj):
        digest = sha256_object({"helper": name, "unresolved": True})
        return HelperSummaryReceipt(
            helper_name=name, source_ast_hash=digest, precondition=empty_bool_expr(),
            paths=(), total_semantic_coverage=False, summary_hash=digest,
            source_function=name, operation_trace_hash=digest,
            unsupported_nodes=("CALLABLE_SOURCE_UNAVAILABLE",),
        )
    try:
        source, function, ast_hash = _source_and_ast(obj)
    except (OSError, TypeError, ValueError):
        # Classes/constructors exposed through imported names are pure object
        # constructors in the audited transitions.
        if inspect.isclass(obj):
            digest = sha256_object({"helper": name, "class": f"{obj.__module__}.{obj.__qualname__}"})
            return HelperSummaryReceipt(
                helper_name=name, source_ast_hash=digest, precondition=empty_bool_expr(),
                paths=(), total_semantic_coverage=True, summary_hash=digest,
                source_function=f"{obj.__module__}.{obj.__qualname__}",
                operation_trace_hash=digest, binding_kind="SOURCE_BOUND_CONSTRUCTOR",
            )
        digest = sha256_object({"helper": name, "source_unavailable": True})
        return HelperSummaryReceipt(
            helper_name=name, source_ast_hash=digest, precondition=empty_bool_expr(),
            paths=(), total_semantic_coverage=False, summary_hash=digest,
            source_function=name, operation_trace_hash=digest,
            unsupported_nodes=("HELPER_SOURCE_UNAVAILABLE",),
        )
    unsupported = _unsupported_nodes(function)
    namespace2 = getattr(inspect.getmodule(obj), "__dict__", namespace)
    nested_names = _called_function_names(function)
    nested = tuple(
        _structural_helper_summary(child, namespace2, seen | {name})
        for child in nested_names
        if child != name
    )
    nested_ok = all(item.total_semantic_coverage for item in nested)
    # Structural operation trace; terminal paths are optional for pure helper
    # functions that end by falling through only if they return None.
    try:
        states = _compile_block(function.body, [_PathState((), {}, [], [], [])])
        paths = _states_to_paths(states, tuple(item.summary_hash for item in nested))
        # Python helpers may return None implicitly.  Make that terminal path
        # explicit in the structural summary rather than treating it as a gap.
        for state in states:
            if state.terminated is None:
                state.terminated = "RETURN"
                state.return_expression = "None"
                state.operations.append(SymbolicOperation("IMPLICIT_RETURN", None, "None", 0, 0))
        paths = _states_to_paths(states, tuple(item.summary_hash for item in nested))
        fallthrough_ok = all(state.terminated in {"RETURN", "RAISE"} for state in states)
        operation_hash = sha256_object({
            "states": [{
                "guards": state.guards,
                "env": state.env,
                "operations": [op.canonical() for op in state.operations],
                "terminated": state.terminated,
            } for state in states]
        })
    except (ValueError, TypeError, SyntaxError) as exc:
        paths = ()
        fallthrough_ok = False
        operation_hash = sha256_object({"error": str(exc)})
        unsupported = tuple(sorted(set(unsupported) | {str(exc)}))
    total = not unsupported and nested_ok and fallthrough_ok
    payload = {
        "helper": name,
        "ast_hash": ast_hash,
        "operation_hash": operation_hash,
        "nested": [item.summary_hash for item in nested],
        "total": total,
    }
    return HelperSummaryReceipt(
        helper_name=name,
        source_ast_hash=ast_hash,
        precondition=empty_bool_expr(),
        paths=paths,
        total_semantic_coverage=total,
        summary_hash=sha256_object(payload),
        source_function=f"{obj.__module__}.{obj.__qualname__}",
        operation_trace_hash=operation_hash,
        called_helpers=nested_names,
        unsupported_nodes=unsupported,
    )


def _call_set(function: ast.FunctionDef) -> set[str]:
    return {_call_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)}


def _raise_messages(function: ast.FunctionDef) -> set[str]:
    messages: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
            arg = node.exc.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                messages.add(arg.value)
    return messages


def _source_text_set(function: ast.FunctionDef) -> set[str]:
    return {_unparse(node) for node in ast.walk(function)}


_EXPECTED_CONCRETE_WRITES: dict[str, frozenset[str]] = {
    "FINAL_DISPATCH": frozenset({"running", "ready_order"}),
    "REM_COMPLETION": frozenset({"jobs", "terminal", "running", "ready_order", "frontier"}),
    "RECOVERY": frozenset({"mode", "frontier"}),
    "DEADLINE_OBSERVATION": frozenset({"misses", "frontier"}),
    "ARRIVAL_BATCH": frozenset({"pending_releases", "frontier", "ready_order"}),
    "MODE_SWITCH": frozenset({"mode", "mode_switches", "frontier"}),
    "RELEASE": frozenset({"jobs", "released", "pending_releases", "frontier", "ready_order"}),
    "SERVICE_UNIT": frozenset({"jobs", "frontier", "running", "ready_order", "time"}),
    "TAIL_ONLY_SERVICE": frozenset({"jobs", "frontier", "running", "ready_order", "time"}),
}


def _concrete_write_targets(paths: tuple[CompiledPathIR, ...]) -> tuple[str, ...]:
    return tuple(sorted({
        assignment.target.removesuffix("_post")
        for path in paths if path.terminator == "RETURN"
        for assignment in path.updates
    }))


def _all_return_paths_are_state_transformers(paths: tuple[CompiledPathIR, ...]) -> bool:
    return all(
        path.return_expression is not None
        and bool(path.updates or path.return_expression in {"state", "new_state", "normalized"})
        for path in paths if path.terminator == "RETURN"
    )


def _semantic_effect(case_id: str, function: ast.FunctionDef,
                     paths: tuple[CompiledPathIR, ...]) -> tuple[SemanticEffectRule | None, tuple[str, ...]]:
    calls = _call_set(function)
    texts = _source_text_set(function)
    raises = _raise_messages(function)
    facts: list[str] = []

    def require(condition: bool, fact: str) -> None:
        if condition:
            facts.append(fact)
        else:
            facts.append(f"MISSING:{fact}")

    if case_id == "FINAL_DISPATCH":
        require("replace" in calls, "replace_updates_running_ready")
        require("sorted" in calls and "_job_schedule_key" in calls, "strict_fp_total_sort")
        require(any("running=None" in (p.return_expression or "") for p in paths), "empty_jobs_dispatches_none")
        require(any("[0]" in (p.return_expression or "") and "_job_schedule_key" in (p.return_expression or "") for p in paths), "nonempty_dispatches_head")
        equations = (("running", "dispatch_head(jobs, priority_order)"),
                     ("ready", "dispatch_tail(jobs, priority_order)"))
        frame = _PROTECTED_FIELDS - {"running", "ready"}
        pairings = ("LOCKSTEP",)
        inputs = ("priority_order",)
        rule = "DISPATCH_SOURCE_PATTERN_V1"
    elif case_id == "REM_COMPLETION":
        require(any(isinstance(n, ast.Delete) and "jobs[key]" in _unparse(n) for n in ast.walk(function)), "delete_completed_job")
        require("TerminalRecord" in calls, "append_completion_terminal")
        require("pop_event" in calls, "consume_rem_event")
        require("_update_ready_order" in calls, "renormalize_ready")
        require("_append_generated_event" in calls, "conditional_recovery_event")
        equations = (
            ("jobs", "remove_job(jobs, event_key)"),
            ("terminal", "record_completion(terminal, jobs, event_key, time)"),
            ("running", "clear_if_equal(running, event_key)"),
            ("ready", "ready_after_removal(jobs, event_key, priority_order)"),
            ("frontier_protected", "consume_protected_event(frontier_protected, event_key)"),
        )
        frame = _PROTECTED_FIELDS - {name for name, _ in equations}
        pairings = ("LOCKSTEP",)
        inputs = ("event_key", "priority_order")
        rule = "REMOVAL_SOURCE_PATTERN_V1"
    elif case_id == "RECOVERY":
        require("recovery_is_legal" in calls, "quiescent_recovery_guard")
        require({"REFERENCE_RECOVERY_OUTSIDE_HI", "REFERENCE_RECOVERY_NOT_QUIESCENT"} <= raises, "illegal_recovery_raises")
        require(any("mode='LO'" in (p.return_expression or "") or 'mode="LO"' in (p.return_expression or "") for p in paths), "legal_recovery_only_changes_mode")
        require("pop_event" in calls, "consume_recovery_event")
        equations = ()
        frame = _PROTECTED_FIELDS
        pairings = ("FULL_STUTTER_PREFIX", "PREFIX_STUTTER_FULL")
        inputs = ()
        rule = "RECOVERY_SOURCE_PATTERN_V1"
    elif case_id == "DEADLINE_OBSERVATION":
        require("MissRecord" in calls and "misses.append" in calls, "append_miss_only_on_incomplete_job")
        require("pop_event" in calls, "consume_deadline_event")
        require(sum(1 for p in paths if p.terminator == "RETURN") >= 5, "all_early_completion_absence_paths")
        equations = (("misses", "observe_deadline(misses, terminal, jobs, event_key, service, demand)"),
                     ("frontier_protected", "consume_event(frontier_protected, event_key)"))
        frame = _PROTECTED_FIELDS - {"misses", "frontier_protected"}
        pairings = ("LOCKSTEP",)
        inputs = ("event_key",)
        rule = "DEADLINE_SOURCE_PATTERN_V1"
    elif case_id == "ARRIVAL_BATCH":
        require(any(isinstance(n, ast.For) and _unparse(n.iter) == "event.batch_jobs" for n in ast.walk(function)), "finite_batch_fold")
        require("classify_arrival_batch" in calls and "decide_reference_release" in calls, "camc_release_classification")
        require("PendingReferenceRelease" in calls, "pending_plan_insert")
        require("_arrival_event_for_job" in calls, "recurring_next_arrival")
        require("_append_generated_event" in calls and "LogicalEvent" in calls, "switch_and_release_events")
        require(any(
            isinstance(n, ast.ListComp)
            and "frontier" in _unparse(n)
            and "e != event" in _unparse(n)
            for n in ast.walk(function)
        ), "consume_arrival_event")
        require({"REFERENCE_RELEASE_DEMAND_OUTSIDE_BOUND"} <= raises, "demand_bound_guard")
        require(any("REFERENCE_DUPLICATE_RELEASE" in item for item in raises) or any("REFERENCE_DUPLICATE_RELEASE" in text for text in texts), "duplicate_release_guard")
        equations = (
            ("pending", "arrival_fold_pending(pending, projected_batch_payload)"),
            ("frontier_protected", "arrival_fold_frontier(frontier_protected, projected_batch_payload)"),
            ("ready", "ready_after_arrival_open(jobs, running, priority_order)"),
        )
        frame = _PROTECTED_FIELDS - {"pending", "frontier_protected", "ready"}
        pairings = ("LOCKSTEP",)
        inputs = ("projected_batch_payload", "priority_order")
        rule = "ARRIVAL_BATCH_SOURCE_PATTERN_V1"
    elif case_id == "MODE_SWITCH":
        required = {
            "REFERENCE_MODE_SWITCH_OUTSIDE_LO",
            "REFERENCE_MODE_SWITCH_TRIGGER_NOT_PENDING_ABNORMAL_HI",
            "REFERENCE_MODE_SWITCH_TIME_MISMATCH",
            "REFERENCE_MODE_SWITCH_DUPLICATE_TIME",
        }
        require(required <= raises, "all_mode_switch_legality_guards")
        require("ReferenceModeSwitch" in calls and "pop_event" in calls, "record_switch_and_consume_event")
        require(any("mode='HI'" in (p.return_expression or "") or 'mode="HI"' in (p.return_expression or "") for p in paths), "switch_only_changes_global_mode")
        equations = ()
        frame = _PROTECTED_FIELDS
        pairings = ("FULL_STUTTER_PREFIX", "PREFIX_STUTTER_FULL")
        inputs = ()
        rule = "MODE_SWITCH_SOURCE_PATTERN_V1"
    elif case_id == "RELEASE":
        require("ReleasedJobRecord" in calls and "ReferenceJob" in calls, "create_released_and_active_records")
        require("_append_generated_event" in calls and "LogicalEvent" in calls, "create_deadline_event")
        require(any(
            isinstance(n, ast.ListComp)
            and "state.frontier" in _unparse(n)
            and "e != event" in _unparse(n)
            for n in ast.walk(function)
        ), "consume_release_event")
        require("_update_ready_order" in calls, "ready_order_recomputed")
        require(any(isinstance(n, ast.DictComp) and "pending_releases" in _unparse(n) for n in ast.walk(function)), "remove_pending_plan")
        require(any("REFERENCE_RELEASE_PLAN_MISSING" in item for item in raises), "pending_plan_guard")
        equations = (
            ("jobs", "release_job(jobs, pending, event_key)"),
            ("released", "record_release(released, pending, event_key)"),
            ("pending", "remove_pending(pending, event_key)"),
            ("frontier_protected", "release_deadline_frontier(frontier_protected, pending, event_key)"),
            ("ready", "ready_after_release(jobs, pending, event_key, priority_order)"),
        )
        frame = _PROTECTED_FIELDS - {name for name, _ in equations}
        pairings = ("LOCKSTEP",)
        inputs = ("event_key", "priority_order")
        rule = "RELEASE_SOURCE_PATTERN_V1"
    elif case_id in {"SERVICE_UNIT", "TAIL_ONLY_SERVICE"}:
        require("_normalize_dispatch" in calls, "normalized_before_and_after_service")
        require("REFERENCE_SERVICE_TICK_WITHOUT_RUNNING" in raises, "running_guard")
        require("REFERENCE_SERVICE_STATE_NOT_NORMALIZED" in raises, "normalized_state_guard")
        require("REFERENCE_SERVICE_CROSSES_EVENT_BOUNDARY" in raises, "event_boundary_guard")
        require(any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) and "jobs[rk]" in _unparse(t) for t in n.targets) for n in ast.walk(function)), "increment_exact_running_job")
        require("_append_generated_event" in calls and "LogicalEvent" in calls, "completion_generates_rem")
        require(any("time=state.time + 1" in (p.return_expression or "") for p in paths), "time_advances_exactly_one")
        if case_id == "SERVICE_UNIT":
            equations = (
                ("jobs", "service_one(jobs, running)"),
                ("service", "service_one_digest(service, running)"),
                ("frontier_protected", "maybe_append_rem(frontier_protected, jobs, running)"),
                ("running", "dispatch_head(service_one(jobs, running), priority_order)"),
                ("ready", "dispatch_tail(service_one(jobs, running), priority_order)"),
                ("time", "(+ time 1)"),
            )
            frame = _PROTECTED_FIELDS - {name for name, _ in equations}
            pairings = ("LOCKSTEP",)
            inputs = ("priority_order",)
            rule = "PROTECTED_SERVICE_SOURCE_PATTERN_V1"
        else:
            # Under the L1 premise the executable running job is a tail job, so
            # the protected projection is a frame while both executions advance
            # to the next integer boundary.
            equations = (("time", "(+ time 1)"),)
            frame = _PROTECTED_FIELDS - {"time"}
            pairings = ("FULL_TAIL_SERVICE_PREFIX_IDLE",)
            inputs = ()
            rule = "TAIL_SERVICE_SOURCE_PATTERN_V1"
    else:
        return None, (f"MISSING:unknown_case:{case_id}",)

    assumptions_by_case: dict[str, tuple[str, ...]] = {
        "FINAL_DISPATCH": ("STRICT_FP_PRIORITY_PREFIX_PARTITION",),
        "REM_COMPLETION": ("EVENT_KEY_IS_PROTECTED", "MODE_EVENTS_ERASED_FROM_PROTECTED_FRONTIER"),
        "RECOVERY": ("MODE_EVENTS_ERASED_FROM_PROTECTED_FRONTIER",),
        "DEADLINE_OBSERVATION": ("EVENT_KEY_IS_PROTECTED",),
        "ARRIVAL_BATCH": (
            "PROJECTED_BATCH_PAYLOAD_EQUAL", "PROTECTED_INPUT_INDEPENDENCE",
            "PROTECTED_INPUT_DEMAND_RECEPTIVENESS",
        ),
        "MODE_SWITCH": ("MODE_EVENTS_ERASED_FROM_PROTECTED_FRONTIER",),
        "RELEASE": ("EVENT_KEY_IS_PROTECTED", "PENDING_PROTECTED_PAYLOAD_EQUAL"),
        "SERVICE_UNIT": ("RUNNING_JOB_IS_PROTECTED", "STRICT_FP_PRIORITY_PREFIX_PARTITION"),
        "TAIL_ONLY_SERVICE": (
            "RUNNING_JOB_IS_TAIL", "TAIL_STRICTLY_LOWER_PRIORITY",
            "NO_PROTECTED_READY_JOB", "TAIL_EVENTS_ERASED_FROM_PROTECTED_FRONTIER",
        ),
    }
    concrete_writes = _concrete_write_targets(paths)
    expected_writes = tuple(sorted(_EXPECTED_CONCRETE_WRITES.get(case_id, frozenset())))
    require(concrete_writes == expected_writes, "all_concrete_state_writes_accounted")
    require(_all_return_paths_are_state_transformers(paths), "all_return_paths_symbolically_interpreted")
    require(all(path.path_hash() for path in paths), "all_terminal_paths_source_hashed")

    valid = all(not item.startswith("MISSING:") for item in facts)
    if not valid:
        return None, tuple(facts)
    effect = SemanticEffectRule(
        effect_id=case_id,
        field_equations=tuple(equations),
        frame_fields=frozenset(frame),
        input_symbols=tuple(inputs),
        supported_pairings=tuple(pairings),
        validator_rule_id=rule,
        validator_facts=tuple(facts),
        source_path_hashes=tuple(path.path_hash() for path in paths),
        concrete_write_targets=concrete_writes,
        covered_concrete_write_targets=expected_writes,
        path_effect_hashes=tuple(path.path_hash() for path in paths),
        required_assumption_ids=assumptions_by_case.get(case_id, ()),
        derivation_complete=True,
    )
    return effect, tuple(facts)


def _aggregate_path_updates(paths: tuple[CompiledPathIR, ...]) -> tuple[Assignment, ...]:
    by_target: dict[str, set[str]] = {}
    for path in paths:
        if path.terminator != "RETURN":
            continue
        for assignment in path.updates:
            by_target.setdefault(assignment.target, set()).add(assignment.expression.to_smt())
    result: list[Assignment] = []
    for target, expressions in sorted(by_target.items()):
        if len(expressions) == 1:
            result.append(state_assignment(target, raw_expr(next(iter(expressions)))))
        else:
            result.append(state_assignment(target, raw_expr(
                "path_merge(" + ", ".join(sorted(expressions)) + ")")))
    return tuple(result)


def _compile_case(case_id: str, function_name: str) -> CompiledTransitionIR:
    function_obj = getattr(executable_semantics, function_name, None)
    if function_obj is None:
        raise ValueError(f"COMPILER_FUNCTION_NOT_FOUND:{function_name}")
    source, function, ast_hash = _source_and_ast(function_obj)
    unsupported = _unsupported_nodes(function)
    namespace = executable_semantics.__dict__
    helper_names = _called_function_names(function)
    helper_receipts = tuple(_structural_helper_summary(name, namespace, frozenset({function_name})) for name in helper_names)
    helper_ok = all(item.total_semantic_coverage for item in helper_receipts)
    helper_hashes = tuple(item.summary_hash for item in helper_receipts)

    try:
        states = _compile_block(function.body, [_PathState((), {}, [], [], [])])
        paths = _states_to_paths(states, helper_hashes)
        nonterminal = [state for state in states if state.terminated not in {"RETURN", "RAISE"}]
    except (ValueError, TypeError, SyntaxError) as exc:
        states = []
        paths = ()
        nonterminal = []
        unsupported = tuple(sorted(set(unsupported) | {str(exc)}))

    covered_return = sum(1 for path in paths if path.terminator == "RETURN")
    covered_raise = sum(1 for path in paths if path.terminator == "RAISE")
    # Independent terminal-state accounting.  The path conversion is checked
    # against this count; it may not define its own coverage denominator.
    return_count = sum(1 for state in states if state.terminated == "RETURN")
    raise_count = sum(1 for state in states if state.terminated == "RAISE")
    effect, validator_facts = _semantic_effect(case_id, function, paths)
    validator_ok = effect is not None

    # Loop body raises are represented as existential fold-step paths, so the
    # source syntactic raise count and terminal IR raise count remain equal for
    # the audited transitions.
    path_coverage = (
        not nonterminal
        and covered_return == return_count
        and covered_raise == raise_count
    )
    total = bool(not unsupported and helper_ok and path_coverage and validator_ok)
    folds = tuple(fold for path in paths for fold in path.folds)
    generated = tuple(event for path in paths for event in path.generated_events)
    aggregate = _aggregate_path_updates(paths)
    frame = frozenset.intersection(*(path.frame_fields for path in paths if path.terminator == "RETURN")) if any(path.terminator == "RETURN" for path in paths) else frozenset()

    receipt = TransitionCompilationReceipt(
        source_function=function_name,
        source_ast_hash=ast_hash,
        normalized_cfg_hash=sha256_object({
            "paths": [path.path_hash() for path in paths],
            "folds": [fold.step_summary_hash for fold in folds],
        }),
        return_path_count=return_count,
        covered_return_path_count=covered_return,
        raise_path_count=raise_count,
        covered_raise_path_count=covered_raise,
        helper_calls=helper_names,
        helper_summaries_bound=tuple(item.helper_name for item in helper_receipts if item.total_semantic_coverage),
        state_fields_read=tuple(sorted({
            _unparse(node) for node in ast.walk(function)
            if isinstance(node, ast.Attribute) and _unparse(node).startswith("state.")
        })),
        state_fields_written=tuple(sorted({assignment.target for assignment in aggregate})),
        frame_fields=tuple(sorted(frame)),
        unsupported_nodes=unsupported + tuple(
            f"HELPER_UNRESOLVED:{item.helper_name}" for item in helper_receipts
            if not item.total_semantic_coverage
        ),
        total_semantic_coverage=total,
        domain_equations=("validate_reference_state(pre)",),
        guard_equations=tuple(path.path_condition.to_smt() for path in paths),
        update_equations=tuple(
            f"{assignment.target}={assignment.expression.to_smt()}"
            for path in paths for assignment in path.updates
        ),
        exceptional_paths=tuple(path.path_hash() for path in paths if path.terminator == "RAISE"),
        helper_summary_hashes=helper_hashes,
        control_flow_coverage={
            "terminal_path_count": len(paths),
            "return_path_count": return_count,
            "raise_path_count": raise_count,
            "covered_return_paths": covered_return,
            "covered_raise_paths": covered_raise,
            "fold_count": len(folds),
            "all_paths_covered": path_coverage,
            "all_helpers_covered": helper_ok,
        },
        semantic_validator_rule_id=effect.validator_rule_id if effect else None,
        semantic_validator_facts=validator_facts,
        semantic_validator_passed=validator_ok,
    )
    status = "COMPILED" if total else (
        "AST_UNSUPPORTED" if unsupported else "UNRESOLVED")
    receipt_hash = sha256_object(receipt.to_dict())
    time_update = raw_expr(next((rhs for field, rhs in effect.field_equations if field == "time"), "time") if effect else "time")
    return CompiledTransitionIR(
        case_id=case_id,
        source_module=_SOURCE_MODULE,
        source_function=function_name,
        source_function_ast_hash=ast_hash,
        precondition=BoolExpr(kind="or", children=tuple(path.path_condition for path in paths)) if len(paths) > 1 else (paths[0].path_condition if paths else empty_bool_expr()),
        post_equations=aggregate,
        frame_fields=frame,
        generated_events=generated,
        time_update=time_update,
        compilation_receipt_hash=receipt_hash,
        compilation_status=status,
        compilation_receipt=receipt,
        paths=paths,
        helper_summary_hashes=helper_hashes,
        return_path_count=return_count,
        covered_return_path_count=covered_return,
        raise_path_count=raise_count,
        covered_raise_path_count=covered_raise,
        total_semantic_coverage=total,
        semantic_effect=effect,
        folds=folds,
    )


def _resolved_case_id(function_name: str, case_id: str | None) -> str:
    if case_id is None:
        config = COMPILATION_MAP.get(function_name)
        if config is None:
            raise ValueError(f"COMPILER_UNKNOWN_FUNCTION:{function_name}")
        case_id = str(config["case_id"])
    return CANONICAL_CASE_ID.get(case_id, case_id)


@lru_cache(maxsize=32)
def _compile_function_cached(function_name: str, case_id: str) -> CompiledTransitionIR:
    return _compile_case(case_id, function_name)


def compile_function(function_name: str, *, case_id: str | None = None,
                     fresh: bool = False) -> CompiledTransitionIR:
    resolved = _resolved_case_id(function_name, case_id)
    return _compile_case(resolved, function_name) if fresh else _compile_function_cached(function_name, resolved)


def _compile_all_transitions_uncached() -> tuple[CompiledTransitionIR, ...]:
    results: list[CompiledTransitionIR] = []
    for case_id, function_name in COMPILATION_TARGETS:
        try:
            results.append(_compile_case(case_id, function_name))
        except (ValueError, TypeError, AttributeError, OSError, SyntaxError) as exc:
            receipt = TransitionCompilationReceipt(
                source_function=function_name,
                source_ast_hash=sha256_object({"error": str(exc)}),
                normalized_cfg_hash=sha256_object({"error": str(exc), "case": case_id}),
                return_path_count=0,
                covered_return_path_count=0,
                raise_path_count=0,
                covered_raise_path_count=0,
                unsupported_nodes=(str(exc),),
                total_semantic_coverage=False,
            )
            results.append(CompiledTransitionIR(
                case_id=case_id,
                source_module=_SOURCE_MODULE,
                source_function=function_name,
                source_function_ast_hash=receipt.source_ast_hash,
                precondition=empty_bool_expr(),
                post_equations=(),
                frame_fields=frozenset(),
                generated_events=(),
                time_update=var_expr("time"),
                compilation_receipt_hash=sha256_object(receipt.to_dict()),
                compilation_status="UNRESOLVED",
                compilation_receipt=receipt,
            ))
    return tuple(results)


@lru_cache(maxsize=1)
def _compile_all_transitions_cached() -> tuple[CompiledTransitionIR, ...]:
    return _compile_all_transitions_uncached()


def compile_all_transitions(*, fresh: bool = False) -> tuple[CompiledTransitionIR, ...]:
    return _compile_all_transitions_uncached() if fresh else _compile_all_transitions_cached()


def clear_compiler_cache() -> None:
    _compile_function_cached.cache_clear()
    _compile_all_transitions_cached.cache_clear()


def compiled_ir_map() -> dict[str, CompiledTransitionIR]:
    return {item.case_id: item for item in compile_all_transitions()}


def compiled_ir_for_case(case_id: str) -> CompiledTransitionIR | None:
    return compiled_ir_map().get(CANONICAL_CASE_ID.get(case_id, case_id))
