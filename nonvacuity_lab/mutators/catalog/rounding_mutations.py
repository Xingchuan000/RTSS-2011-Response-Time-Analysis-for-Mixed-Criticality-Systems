"""Current-source bindings for C2 rounding-to-nearest mutation."""

from __future__ import annotations

from pathlib import Path

from ..python_binding import bind_symbol
from ...canonical import python_symbol_hash


def _patch(root: Path, *, role: str, relative: str, symbol: str,
           before: str, after: str, occurrence: int = 1) -> dict:
    source = (root / relative).read_text(encoding="utf-8")
    bound = bind_symbol(source, symbol)
    actual = bound.source.count(before)
    if actual != occurrence:
        raise ValueError(
            f"C2_ROUND_BINDING_NOT_UNIQUE:{relative}:{symbol}:"
            f"expected={occurrence}:actual={actual}"
        )
    return {
        "role": role,
        "target_file": relative,
        "target_symbol": symbol,
        "before_ast_hash": python_symbol_hash(source, symbol),
        "before_snippet": before,
        "after_snippet": after,
        "occurrence": occurrence,
    }


def build_rounding_catalog(root: Path) -> tuple[dict, ...]:
    deployed_before = '        if self.budget_rounding_mode not in {"ceil_floor", "nearest"}:\n'
    deployed_after = '        self.budget_rounding_mode = "nearest"\n' + deployed_before
    return (
        _patch(
            root,
            role="DEPLOYED_APPLY",
            relative="amc_py/rl/env.py",
            symbol="AmcBudgetEnv.__post_init__",
            before=deployed_before,
            after=deployed_after,
        ),
        _patch(
            root,
            role="FORMAL_SEMANTIC_MIRROR",
            relative="formal_toolchain/adapters/s185_target.py",
            symbol="build_target",
            before='"ceil_floor"',
            after='"nearest"',
            occurrence=3,
        ),
    )
