"""Frozen atomic budget-update contract for the C-AMC-sem/P0 proof."""

from __future__ import annotations

from collections.abc import Mapping


class BudgetState:
    """Minimal budget state needed by the frozen controller proof."""

    def __init__(self, budgets: dict[str, int]) -> None:
        self.budgets = budgets

    def apply_updates(self, updates: Mapping[str, int]) -> None:
        normalized = {
            str(name): int(value)
            for name, value in updates.items()
        }
        for name, value in normalized.items():
            if value <= 0:
                raise ValueError("runtime budget must be > 0")
            if name not in self.budgets:
                raise KeyError(f"missing runtime budget for task {name!r}")
        self.budgets.update(normalized)

