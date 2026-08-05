"""Build coherent B4 guard-removal patches from current source."""

from __future__ import annotations

from pathlib import Path

from ..python_binding import bind_symbol
from ...canonical import python_symbol_hash


def _patch(root: Path, *, role: str, relative: str, symbol: str,
           before: str, after: str) -> dict:
    source = (root / relative).read_text(encoding="utf-8")
    bound = bind_symbol(source, symbol)
    if bound.source.count(before) != 1:
        raise ValueError(f"B4_GUARD_BINDING_NOT_UNIQUE:{relative}:{symbol}")
    return {
        "role": role,
        "target_file": relative,
        "target_symbol": symbol,
        "before_ast_hash": python_symbol_hash(source, symbol),
        "before_snippet": before,
        "after_snippet": after,
        "occurrence": 1,
    }


def build_guard_catalog(root: Path) -> dict[str, tuple[dict, ...]]:
    env = "amc_py/rl/env.py"
    frozen = "formal_toolchain/semantics/frozen_c_amc_sem_action_runtime.py"
    return {
        "decrease_hi_forbidden": (
            _patch(
                root, role="DEPLOYED_GUARD", relative=env,
                symbol="AmcBudgetEnv.evaluate_budget_candidate",
                before="            True\n            and action_violates_hi_decrease_guard(\n",
                after="            False\n            and action_violates_hi_decrease_guard(\n",
            ),
            _patch(
                root, role="FROZEN_GUARD", relative=frozen,
                symbol="evaluate_budget_candidate",
                before="    if forbid_decreasing_hi_budgets:\n",
                after="    if False and forbid_decreasing_hi_budgets:\n",
            ),
        ),
        "budget_floor_violation": (
            _patch(
                root, role="DEPLOYED_GUARD", relative=env,
                symbol="AmcBudgetEnv.evaluate_budget_candidate",
                before="        if floor_reject_reason is not None:\n",
                after="        if False and floor_reject_reason is not None:\n",
            ),
            _patch(
                root, role="FROZEN_GUARD", relative=frozen,
                symbol="evaluate_budget_candidate",
                before="    if floor_reason is not None:\n",
                after="    if False and floor_reason is not None:\n",
            ),
        ),
        "deploy_cap": (
            _patch(
                root, role="DEPLOYED_GUARD", relative=env,
                symbol="AmcBudgetEnv.evaluate_budget_candidate",
                before="        if cap_reason is not None:\n",
                after="        if False and cap_reason is not None:\n",
            ),
            _patch(
                root, role="FROZEN_GUARD", relative=frozen,
                symbol="evaluate_budget_candidate",
                before="    if cap_reason is not None:\n",
                after="    if False and cap_reason is not None:\n",
            ),
        ),
        "safety_checker": (
            _patch(
                root, role="DEPLOYED_GUARD", relative=env,
                symbol="AmcBudgetEnv.evaluate_budget_candidate",
                before="        if self.check_safety:\n",
                after="        if False and self.check_safety:\n",
            ),
            _patch(
                root, role="FROZEN_GUARD",
                relative="formal_toolchain/adapters/s185_target.py",
                symbol="build_target",
                before="            disabled_guards=(),\n",
                after='            disabled_guards=("safety_checker",),\n',
            ),
        ),
    }
