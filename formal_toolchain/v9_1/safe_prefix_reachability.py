"""Independent reachability checks for SAT first-bad-window witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .symbolic_state import BoundModel


@dataclass(frozen=True, slots=True)
class ReachabilityResult:
    status: str
    code: str
    reason: str | None = None


def check_witness_boot_safe_prefix(witness: Mapping[str, Any], model: BoundModel) -> ReachabilityResult:
    """Check explicit boot-reachability evidence; never trust a status flag."""

    if witness.get("z0_reachable_from_boot") is not True:
        return ReachabilityResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
                                  "boot reachability was not independently established")
    if witness.get("z0_no_prior_hi_miss") is not True:
        return ReachabilityResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
                                  "witness z0 does not carry an independent M=0 proof")
    budgets = witness.get("z0_budgets")
    if not isinstance(budgets, Mapping) or set(budgets) != {task.name for task in model.tasks}:
        return ReachabilityResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
                                  "witness does not expose the complete z0 budget vector")
    for task in model.tasks:
        value = budgets.get(task.name)
        if not isinstance(value, int) or not (task.budget_floor <= value <= task.budget_upper):
            return ReachabilityResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
                                      "witness z0 budget is outside the bound model")
    prefix = witness.get("safe_prefix_trace")
    if not isinstance(prefix, list) or not prefix:
        return ReachabilityResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
                                  "safe-prefix trace is missing")
    return ReachabilityResult("PASS", "Z0_SAFE_PREFIX_REACHABLE")


__all__ = ["ReachabilityResult", "check_witness_boot_safe_prefix"]
