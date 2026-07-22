"""第一轮 single/24 action、mask 和 step 目标函数绑定。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .python_ast_ir import function_to_ir


def _function_node(source: str, qualified_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    tree = ast.parse(source)
    if "." in qualified_name:
        class_name, method_name = qualified_name.rsplit(".", 1)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        return item
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == qualified_name:
            return node
    return None


def _has_none_base_return(source: str, qualified_name: str) -> bool:
    node = _function_node(source, qualified_name)
    if node is None:
        return False
    for item in ast.walk(node):
        if isinstance(item, ast.Return) and isinstance(item.value, ast.Tuple) and len(item.value.elts) == 2:
            left, right = item.value.elts
            if isinstance(left, ast.Constant) and left.value is None and isinstance(right, ast.Name) and right.id == "base":
                return True
    return False


def _top1_or_noop_branch_returns_none_base(source: str) -> bool:
    node = _function_node(source, "IntegerTreeBudgetPolicy.select_action_id")
    if node is None:
        return False
    for item in ast.walk(node):
        if not isinstance(item, ast.If):
            continue
        has_target_compare = False
        for sub in ast.walk(item.test):
            if (
                isinstance(sub, ast.Compare)
                and isinstance(sub.left, ast.Name)
                and sub.left.id == "selection_semantics"
                and len(sub.ops) == 1
                and isinstance(sub.ops[0], ast.Eq)
                and len(sub.comparators) == 1
                and isinstance(sub.comparators[0], ast.Constant)
                and sub.comparators[0].value == "top1_or_noop"
            ):
                has_target_compare = True
                break
        if not has_target_compare:
            continue
        for sub in ast.walk(item):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple) and len(sub.value.elts) == 2:
                left, right = sub.value.elts
                if isinstance(left, ast.Constant) and left.value is None and isinstance(right, ast.Name) and right.id == "base":
                    return True
        return False
    return False


def bind_action_runtime(source_root: Path, *, action_space_type: str = "single",
                        action_dim: int = 24, explicit_noop: bool = False) -> dict[str, Any]:
    if (action_space_type, action_dim, explicit_noop) != ("single", 24, False):
        return {"status": "UNRESOLVED", "failure": {"code": "UNSUPPORTED_ACTION_SCOPE", "route": "MODEL_CONFORMANCE_FAILED"}}
    path = Path(source_root) / "amc_py/rl/env.py"
    source = path.read_text(encoding="utf-8")
    names = (
        "build_budget_action_space",
        "apply_budget_action_candidate",
        "AmcBudgetEnv.formal_valid_action_mask",
        "AmcBudgetEnv.evaluate_budget_candidate",
        "AmcBudgetEnv.valid_action_mask",
        "AmcBudgetEnv.step",
        "AmcBudgetEnv._budget_floor_violation",
        "AmcBudgetEnv._deploy_cap_increase_reject_reason",
    )
    functions = {}
    for name in names:
        target_source = source if name.startswith("AmcBudgetEnv.") else (Path(source_root) / "amc_py/rl/actions.py").read_text(encoding="utf-8")
        lookup = name if name.startswith("AmcBudgetEnv.") else name
        functions[name] = function_to_ir(target_source, lookup)
    policy_source = (Path(source_root) / "amc_py/viper/tree_policy.py").read_text(encoding="utf-8")
    functions["IntegerTreeBudgetPolicy.select_action_id"] = function_to_ir(
        policy_source, "IntegerTreeBudgetPolicy.select_action_id")
    unresolved = [name for name, value in functions.items() if value.get("status") != "PASS"]
    if unresolved:
        return {"status": "UNRESOLVED", "failure": {"code": "TARGET_METHOD_IR_UNRESOLVED", "route": "UNRESOLVED", "functions": unresolved}, "functions": functions}
    fallback_semantics_ok = (
        _top1_or_noop_branch_returns_none_base(policy_source)
        and _has_none_base_return(policy_source, "IntegerTreeBudgetPolicy.select_action_id")
        and "tree_no_valid_action" in policy_source
    )
    if not fallback_semantics_ok:
        return {"status": "FAIL", "failure": {"code": "ACTION_FALLBACK_SEMANTICS_FAILED",
                "route": "MODEL_CONFORMANCE_FAILED"}, "functions": functions}
    action_source = (Path(source_root) / "amc_py/rl/actions.py").read_text(encoding="utf-8")
    canonical_tokens = (
        "action_id", "increase_ratio", "decrease_ratio", "math.ceil",
        "math.floor", "forbid_decreasing_hi_budgets", "budget_floor",
        "valid_action_mask", "formal_valid_action_mask",
        "evaluate_budget_candidate", "apply_budget_action_candidate",
    )
    missing_tokens = [token for token in canonical_tokens if token not in action_source + source]
    exact_indexing = "self._actions[action_id]" in source
    shared_candidate_evaluator = (
        exact_indexing
        and "self.evaluate_budget_candidate(" in source
        and "def formal_valid_action_mask" in source
        and "def evaluate_budget_candidate" in source
        and "def valid_action_mask" in source
        and "def step" in source
    )
    if missing_tokens or not shared_candidate_evaluator:
        return {"status": "FAIL", "failure": {"code": "ACTION_ORDER_OR_MASK_SEMANTICS_FAILED",
                "route": "MODEL_CONFORMANCE_FAILED", "missing": missing_tokens}, "functions": functions}
    return {"status": "PASS", "action_space_type": action_space_type, "action_dim": action_dim,
            "explicit_noop": explicit_noop, "functions": functions,
            "order_evidence": {"source": "derived_from_canonical_action_source", "verified": exact_indexing,
                                "action_table_source_hash": __import__("hashlib").sha256(action_source.encode()).hexdigest(),
                                "mask_and_step_share_candidate_evaluator": shared_candidate_evaluator,
                                "candidate_evaluator_name": "AmcBudgetEnv.evaluate_budget_candidate",
                                "fallback": "implicit_none_when_no_valid_action",
                                "guards": ["ceil", "floor", "HI_decrease_guard", "LO_floor_guard"]}}
