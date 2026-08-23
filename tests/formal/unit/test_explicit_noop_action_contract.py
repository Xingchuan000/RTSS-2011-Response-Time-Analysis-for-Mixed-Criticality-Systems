from __future__ import annotations

import pytest

from formal_toolchain.policy.mask_fallback import (
    build_parametric_mask_fallback_certificate,
    select_first_valid,
)
from formal_toolchain.policy.selected_regions import selected_action_regions_v2
from formal_toolchain.semantics.frozen_c_amc_sem_action_runtime import (
    apply_budget_action_candidate,
    build_budget_action_space,
    evaluate_budget_candidate,
    formal_valid_action_mask,
)


TASKS = [f"T{index:02d}" for index in range(12)]


def _evaluator_kwargs() -> dict[str, object]:
    return {
        "task_names": TASKS,
        "task_criticalities": ["HI"] * 6 + ["LO"] * 6,
        "task_deadlines": [100] * 12,
        "task_c_hi": [80] * 12,
        "initial_budgets": {name: 50 for name in TASKS},
        "rounding_mode": "ceil_floor",
        "min_budget_delta": 1,
        "forbid_decreasing_hi_budgets": True,
        "budget_floor_ratio": 0.9,
        "deploy_cap_enabled": True,
        "deploy_cap_ratio": 4.0,
        "deploy_cap_criticality": "LO",
    }


def test_single24_and_single25_layout() -> None:
    legacy = build_budget_action_space(TASKS)
    explicit = build_budget_action_space(TASKS, include_explicit_noop=True)
    assert len(legacy) == 24
    assert len(explicit) == 25
    assert [row["action_id"] for row in explicit if row["is_noop"]] == [24]


def test_explicit_noop_is_total_identity_and_always_valid() -> None:
    actions = build_budget_action_space(TASKS, include_explicit_noop=True)
    noop = actions[24]
    for value in range(1, 101):
        budgets = {name: value for name in TASKS}
        assert apply_budget_action_candidate(
            action=noop,
            budgets=budgets,
            task_names=TASKS,
            task_criticalities=["HI"] * 6 + ["LO"] * 6,
            task_deadlines=[100] * 12,
            task_c_hi=[80] * 12,
        ) == {}
        result = evaluate_budget_candidate(action=noop, budgets=budgets, **_evaluator_kwargs())
        assert result["accepted"] is True
        assert result["candidate_budgets"] == budgets
        assert result["is_explicit_noop"] is True
        assert formal_valid_action_mask(actions=actions, budgets=budgets, **_evaluator_kwargs())[24]


def test_tampered_noop_targets_are_rejected() -> None:
    noop = dict(build_budget_action_space(TASKS, include_explicit_noop=True)[24])
    noop["increase_idx"] = 0
    with pytest.raises(ValueError, match="NOOP_HAS_INCREASE_TARGET"):
        apply_budget_action_candidate(
            action=noop, budgets={name: 50 for name in TASKS}, task_names=TASKS,
            task_criticalities=["HI"] * 6 + ["LO"] * 6,
            task_deadlines=[100] * 12, task_c_hi=[80] * 12,
        )


def test_explicit_noop_selection_is_action_24_not_implicit_none() -> None:
    ranking = tuple(range(25))
    mask = (False,) * 24 + (True,)
    assert select_first_valid(ranking, mask, action_dim=25) == 24
    contract = {
        "shared_with_step": True,
        "selection": "ranked_first_valid",
        "explicit_noop": True,
        "explicit_noop_action_ids": [24],
        "explicit_noop_always_valid": True,
    }
    certificate = build_parametric_mask_fallback_certificate(
        rankings={1: ranking}, action_dim=25, mask_contract=contract
    )
    assert certificate["status"] == "PASS"
    assert certificate["implicit_noop"] is False
    assert certificate["leaves"][0]["regions"][-1]["selected_action"] == 24
    regions = selected_action_regions_v2(
        {1: []}, {1: ranking}, explicit_noop_action_id=24
    )["regions"]
    assert regions[-1]["action_id"] == 24
    assert all(row["action_id"] is not None for row in regions)


def test_invalid_explicit_noop_contract_fails_closed() -> None:
    result = build_parametric_mask_fallback_certificate(
        rankings={1: tuple(range(25))},
        action_dim=25,
        mask_contract={
            "shared_with_step": True,
            "selection": "ranked_first_valid",
            "explicit_noop": True,
            "explicit_noop_action_ids": [24],
            "explicit_noop_always_valid": False,
        },
    )
    assert result["status"] == "FAIL"
    assert result["failure"]["code"] == "EXPLICIT_NOOP_IDENTITY_UNRESOLVED"
