"""Bind the deployed tree fallback to frozen C-AMC-sem/P0 action semantics.

The mutable RL environment and q-AMC experiment code are deliberately treated
as non-blocking audit inputs.  The proof route consumes only the frozen
single/24 action contract plus the deployed integer-tree fallback policy.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from .python_ast_ir import function_to_ir
from formal_toolchain.semantics.frozen_runtime_contract import (
    CONTRACT_VERSION,
    frozen_action_runtime_path,
)


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


def _audit_hashes(source_root: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for relative in ("amc_py/rl/env.py", "amc_py/rl/actions.py", "amc_py/rl/safety.py"):
        path = source_root / relative
        if not path.is_file():
            continue
        payload = path.read_bytes()
        records.append({
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        })
    return {
        "binding": "NON_BLOCKING_AUDIT_ONLY",
        "files": records,
    }


def bind_action_runtime(source_root: Path, *, action_space_type: str = "single",
                        action_dim: int = 24, explicit_noop: bool = False) -> dict[str, Any]:
    if (action_space_type, action_dim, explicit_noop) != ("single", 24, False):
        return {"status": "UNRESOLVED", "failure": {"code": "UNSUPPORTED_ACTION_SCOPE", "route": "MODEL_CONFORMANCE_FAILED"}}

    root = Path(source_root)
    frozen_path = frozen_action_runtime_path(root)
    frozen_source = frozen_path.read_text(encoding="utf-8")
    frozen_names = (
        "build_budget_action_space",
        "apply_budget_action_candidate",
        "formal_valid_action_mask",
        "evaluate_budget_candidate",
        "valid_action_mask",
        "step",
        "budget_floor_violation",
        "deploy_cap_increase_reject_reason",
    )
    functions = {name: function_to_ir(frozen_source, name) for name in frozen_names}

    policy_path = root / "amc_py/viper/tree_policy.py"
    policy_source = policy_path.read_text(encoding="utf-8")
    functions["IntegerTreeBudgetPolicy.select_action_id"] = function_to_ir(
        policy_source, "IntegerTreeBudgetPolicy.select_action_id")

    unresolved = [name for name, value in functions.items() if value.get("status") != "PASS"]
    if unresolved:
        return {
            "status": "UNRESOLVED",
            "failure": {
                "code": "TARGET_METHOD_IR_UNRESOLVED",
                "route": "UNRESOLVED",
                "functions": unresolved,
            },
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "functions": functions,
            "implementation_audit": _audit_hashes(root),
        }

    fallback_semantics_ok = (
        _top1_or_noop_branch_returns_none_base(policy_source)
        and _has_none_base_return(policy_source, "IntegerTreeBudgetPolicy.select_action_id")
        and "tree_no_valid_action" in policy_source
    )
    if not fallback_semantics_ok:
        return {
            "status": "FAIL",
            "failure": {
                "code": "ACTION_FALLBACK_SEMANTICS_FAILED",
                "route": "MODEL_CONFORMANCE_FAILED",
            },
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "functions": functions,
            "implementation_audit": _audit_hashes(root),
        }

    canonical_tokens = (
        "action_id",
        "increase_ratio",
        "decrease_ratio",
        "math.ceil",
        "math.floor",
        "forbid_decreasing_hi_budgets",
        "budget_floor_ratio",
        "formal_valid_action_mask",
        "evaluate_budget_candidate",
        "apply_budget_action_candidate",
        'inc_value = math.ceil(raw_inc) if rounding_mode == "ceil_floor" else int(round(raw_inc))',
        'dec_value = math.floor(raw_dec) if rounding_mode == "ceil_floor" else int(round(raw_dec))',
        "candidate.update(updates)",
        "action = actions[action_id]",
    )
    missing_tokens = [token for token in canonical_tokens if token not in frozen_source]
    exact_indexing = "action = actions[action_id]" in frozen_source
    shared_candidate_evaluator = (
        exact_indexing
        and "evaluation = evaluate_budget_candidate(" in frozen_source
        and "def formal_valid_action_mask" in frozen_source
        and "def evaluate_budget_candidate" in frozen_source
        and "def valid_action_mask" in frozen_source
        and "def step" in frozen_source
    )
    if missing_tokens or not shared_candidate_evaluator:
        return {
            "status": "FAIL",
            "failure": {
                "code": "ACTION_ORDER_OR_MASK_SEMANTICS_FAILED",
                "route": "MODEL_CONFORMANCE_FAILED",
                "missing": missing_tokens,
            },
            "formal_semantics_contract_version": CONTRACT_VERSION,
            "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
            "functions": functions,
            "implementation_audit": _audit_hashes(root),
        }

    frozen_hash = hashlib.sha256(frozen_source.encode("utf-8")).hexdigest()
    return {
        "status": "PASS",
        "action_space_type": action_space_type,
        "action_dim": action_dim,
        "explicit_noop": explicit_noop,
        "formal_semantics_contract_version": CONTRACT_VERSION,
        "formal_action_semantics_source": frozen_path.relative_to(root).as_posix(),
        "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
        "functions": functions,
        "order_evidence": {
            "source": "derived_from_frozen_c_amc_sem_action_contract",
            "verified": exact_indexing,
            "action_table_source_hash": frozen_hash,
            "mask_and_step_share_candidate_evaluator": shared_candidate_evaluator,
            "candidate_evaluator_name": "evaluate_budget_candidate",
            "fallback": "implicit_none_when_no_valid_action",
            "guards": ["ceil", "floor", "HI_decrease_guard", "LO_floor_guard"],
        },
        "implementation_audit": _audit_hashes(root),
    }
