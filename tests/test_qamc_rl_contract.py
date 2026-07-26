from __future__ import annotations

import pytest

from amc_py.qamc.rl_contract import validate_qamc_rl_semantics
from amc_py.runtime_models import RuntimeSemantics


def _validate(**overrides: object) -> None:
    values = {
        "semantics": RuntimeSemantics.Q_AMC,
        "action_space": "single",
        "check_safety": True,
        "step_guard_semantics": "checked",
        "nonvacuity_disabled_guards": (),
        "budget_rounding_mode": "ceil_floor",
        "min_budget_delta": 1,
    }
    values.update(overrides)
    validate_qamc_rl_semantics(**values)  # type: ignore[arg-type]


def test_qamc_certified_contract_passes() -> None:
    _validate()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"check_safety": False}, "QAMC_REQUIRES_CHECK_SAFETY"),
        ({"step_guard_semantics": "unchecked_apply"}, "UNCHECKED_STEP_GUARD"),
        ({"nonvacuity_disabled_guards": ("budget_floor",)}, "GUARD_DISABLE"),
        ({"action_space": "pair"}, "ACTION_SPACE_NOT_CERTIFIED"),
        ({"budget_rounding_mode": "nearest"}, "BUDGET_ROUNDING"),
        ({"min_budget_delta": 2}, "MIN_BUDGET_DELTA"),
    ],
)
def test_qamc_rejects_uncertified_rl_settings(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(**overrides)
