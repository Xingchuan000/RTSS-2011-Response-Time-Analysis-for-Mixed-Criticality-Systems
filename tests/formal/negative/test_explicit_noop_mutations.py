from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from amc_py.models import Criticality, Task
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.observation_metadata import build_action_definitions
from formal_toolchain.core.formal_checks import _build_formal_actions
from formal_toolchain.policy.mask_fallback import build_parametric_mask_fallback_certificate


class _Adapter:
    def __init__(self, *, explicit: bool = True, noop_id: int = 24) -> None:
        self.explicit = explicit
        self.noop_id = noop_id

    def export_mask_contract(self):
        return {"explicit_noop": self.explicit,
                "explicit_noop_action_ids": [self.noop_id] if self.explicit else [],
                "explicit_noop_always_valid": self.explicit}


def _target():
    tasks = tuple(Task(name=f"T{index:02d}", period=100, deadline=100,
                       c_lo=10, c_hi=10, criticality=Criticality.LO)
                  for index in range(12))
    actions = build_budget_action_space(tasks, action_space="single",
                                        include_explicit_noop=True)
    return SimpleNamespace(
        ordered_tasks=tasks,
        runtime_config=SimpleNamespace(action_space="single", budget_increase_ratio=0.1,
                                       budget_decrease_ratio=0.05),
        runtime_adapter=_Adapter(),
        action_definitions=tuple(build_action_definitions(actions)),
    )


def test_mutated_noop_flag_is_rejected() -> None:
    target = _target()
    definitions = list(deepcopy(target.action_definitions))
    definitions[24]["is_noop"] = False
    target.action_definitions = tuple(definitions)
    with pytest.raises(ValueError, match="TARGET_FORMAL_ACTION_SCHEMA_MISMATCH"):
        _build_formal_actions(target)


def test_adapter_claiming_implicit_only_cannot_hide_action_24() -> None:
    target = _target()
    target.runtime_adapter = _Adapter(explicit=False)
    with pytest.raises(ValueError, match="TARGET_FORMAL_ACTION_DIMENSION_MISMATCH"):
        _build_formal_actions(target)


def test_mutated_noop_id_is_rejected() -> None:
    target = _target()
    target.runtime_adapter = _Adapter(noop_id=23)
    with pytest.raises(ValueError, match="TARGET_EXPLICIT_NOOP_ID_MISMATCH"):
        _build_formal_actions(target)


def test_ranking_missing_action_24_fails_closed() -> None:
    ranking = tuple(range(24)) + (23,)
    result = build_parametric_mask_fallback_certificate(
        rankings={0: ranking}, action_dim=25,
        mask_contract={"shared_with_step": True, "selection": "ranked_first_valid",
                       "explicit_noop": True, "explicit_noop_action_ids": [24],
                       "explicit_noop_always_valid": True},
    )
    assert result["status"] == "FAIL"
    assert result["failure"]["code"] == "RANKING_NOT_COMPLETE"
