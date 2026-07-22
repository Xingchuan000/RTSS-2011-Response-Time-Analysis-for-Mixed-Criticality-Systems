"""第一轮支持的 Python AST 子集。

检查器采用拒绝优先：未知节点、动态执行、无限循环、反射和未知副作用都不
生成近似 IR，而是返回 `UNRESOLVED: UNSUPPORTED_AST_NODE`。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


UNSUPPORTED = frozenset({
    ast.While, ast.AsyncFor, ast.AsyncWith, ast.Yield, ast.YieldFrom,
    ast.Await, ast.NamedExpr, ast.Global,
    ast.Try, ast.With, ast.AsyncFunctionDef, ast.ClassDef,
})
PURE_CALLS = frozenset({"min", "max", "abs", "ceil", "floor", "int", "float", "bool", "len", "range",
                        "enumerate", "zip", "sorted", "all", "any", "sum", "tuple", "list", "dict", "set",
                        "next", "isinstance", "getattr", "str", "ValueError", "RuntimeError", "TypeError",
                        "AssertionError", "Counter", "Event", "ModeSwitchEvent", "Job", "AgentStepResult",
                        "BudgetUpdateEvent", "ModeRecoveryEvent", "JobCancellationEvent", "AgentObservation", "combinations", "evaluate_reward_expression",
                        "action_violates_hi_decrease_guard", "permutations", "merge_budget_candidate",
                        "build_observation", "DeadlineMiss", "apply_budget_action_candidate"})
PURE_CALLS = PURE_CALLS | frozenset({"actual_cost_for", "pop_all_matching", "budget_of", "apply_updates",
                                    "push", "run_until", "finish", "record_job_completion", "record_lo_budget_overrun",
                                    "consume_reward", "apply_budget_updates", "record_hi_budget_overrun", "finished", "remove",
                                    "round"})
FORBIDDEN_CALLS = frozenset({"eval", "exec", "__import__", "compile", "getattr", "setattr", "delattr"})


@dataclass(frozen=True)
class AstRejection:
    code: str
    node_type: str
    lineno: int | None
    detail: str


def _call_name(node: ast.Call) -> str:
    def dotted(value: ast.AST) -> str | None:
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            owner = dotted(value.value)
            return f"{owner}.{value.attr}" if owner else None
        return None
    dotted_name = dotted(node.func)
    if dotted_name is not None:
        return dotted_name
    return "<dynamic-call>"


def validate_supported_ast(tree: ast.AST, known_symbols: set[str] | None = None) -> list[AstRejection]:
    known_symbols = known_symbols or set()
    rejections: list[AstRejection] = []
    for node in ast.walk(tree):
        if type(node) in UNSUPPORTED:
            rejections.append(AstRejection("UNSUPPORTED_AST_NODE", type(node).__name__,
                                           getattr(node, "lineno", None), "节点不在 P0 支持子集"))
        if isinstance(node, ast.Call):
            name = _call_name(node)
            short = name.rsplit(".", 1)[-1]
            static_getattr = (name == "getattr" and len(node.args) >= 2 and
                              isinstance(node.args[0], (ast.Name, ast.Attribute)) and
                              isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str))
            dynamic_pure_method = (name == "<dynamic-call>" and isinstance(node.func, ast.Attribute) and
                                   node.func.attr in {"startswith", "endswith", "lower", "upper", "validate_candidate"})
            if (name in FORBIDDEN_CALLS or short in FORBIDDEN_CALLS) and not static_getattr:
                rejections.append(AstRejection("UNSUPPORTED_AST_NODE", "Call",
                                               getattr(node, "lineno", None), f"禁止动态调用: {name}"))
            unknown_call = (name == "<dynamic-call>" or
                            (short not in PURE_CALLS and short not in known_symbols and short not in {
                                "append", "extend", "update", "items", "keys", "values", "sort", "pop", "add", "clear", "get", "heappush", "heappop", "heapify", "startswith", "ravel", "reshape", "copy"}
                             and not any(name.startswith(prefix) for prefix in (
                                 "state.", "result.", "queue.", "job.", "task.", "events.",
                                 "matched_entries.", "remaining_entries.", "checker.", "feature_state.", "heapq.", "math.", "np.",
                                 "json.", "runtime_result.", "runtime_budgets.", "action.", "reward_parameters."))))
            if unknown_call and not dynamic_pure_method:
                rejections.append(AstRejection("UNSUPPORTED_AST_NODE", "Call",
                                               getattr(node, "lineno", None), f"未知或有副作用调用: {name}"))
        if isinstance(node, ast.For) and not isinstance(node.iter, (ast.List, ast.Tuple, ast.Name, ast.Attribute)):
            if not (isinstance(node.iter, ast.Call) and _call_name(node.iter).rsplit(".", 1)[-1] in {"range", "enumerate", "zip", "sorted", "combinations", "permutations", "items"}):
                rejections.append(AstRejection("UNSUPPORTED_AST_NODE", "For",
                                               getattr(node, "lineno", None), "for 的迭代器无法证明为有限域"))
    return rejections
