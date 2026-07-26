"""Fail-closed constraints for the certified q-AMC RL entry path."""

from __future__ import annotations

from collections.abc import Iterable

from amc_py.runtime_models import RuntimeSemantics


QAMC_CERTIFIED_ACTION_SPACES = frozenset({"single"})


def validate_qamc_rl_semantics(
    *,
    semantics: RuntimeSemantics,
    action_space: str,
    check_safety: bool,
    step_guard_semantics: str,
    nonvacuity_disabled_guards: Iterable[str],
    budget_rounding_mode: str,
    min_budget_delta: int,
) -> None:
    if semantics is not RuntimeSemantics.Q_AMC:
        return
    if not check_safety:
        raise ValueError("QAMC_REQUIRES_CHECK_SAFETY")
    if step_guard_semantics != "checked":
        raise ValueError("QAMC_UNCHECKED_STEP_GUARD_FORBIDDEN")
    if tuple(nonvacuity_disabled_guards):
        raise ValueError("QAMC_NONVACUITY_GUARD_DISABLE_FORBIDDEN")
    if budget_rounding_mode != "ceil_floor":
        raise ValueError("QAMC_UNCERTIFIED_BUDGET_ROUNDING_MODE")
    if min_budget_delta != 1:
        raise ValueError("QAMC_UNCERTIFIED_MIN_BUDGET_DELTA")
    if action_space not in QAMC_CERTIFIED_ACTION_SPACES:
        raise ValueError(f"QAMC_ACTION_SPACE_NOT_CERTIFIED:{action_space}")


__all__ = ["QAMC_CERTIFIED_ACTION_SPACES", "validate_qamc_rl_semantics"]
