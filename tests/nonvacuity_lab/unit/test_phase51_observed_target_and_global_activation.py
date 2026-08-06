from __future__ import annotations

import pytest

from nonvacuity_lab.activation.hout_activation import evaluate_hout_activation
from nonvacuity_lab.config_resolver import _bind_observed_raw_target


def test_b2_binding_requires_positive_invalid_and_lower_rank_fallback():
    mutation = {}
    rows = [
        {
            "leaf_id": 1,
            "action_ranking": [3, 4],
            "raw_top1_invalid_count": 10,
            "fallback_count": 0,
            "selected_rank_histogram": {"0": 20},
        },
        {
            "leaf_id": 2,
            "action_ranking": [5, 6],
            "raw_top1_invalid_count": 8,
            "fallback_count": 8,
            "selected_rank_histogram": {"1": 8},
        },
    ]
    _bind_observed_raw_target(
        mutation,
        rows,
        evidence_field="raw_top1_invalid_count",
        require_fallback=True,
    )
    assert mutation["resolved_target"]["leaf_id"] == 2
    assert mutation["resolved_target"]["baseline_fallback_count"] == 8


def test_b3_binding_rejects_zero_evidence():
    with pytest.raises(ValueError, match="OBSERVED_TARGET_EVIDENCE_MISSING"):
        _bind_observed_raw_target(
            {},
            [{"leaf_id": 1, "action_ranking": [0], "all_invalid_count": 0}],
            evidence_field="all_invalid_count",
        )


def test_global_difference_scope_detects_earlier_global_selection_change():
    base = [
        {
            "scenario_id": 1,
            "controller_decision_index": 0,
            "leaf_id": 9,
            "raw_top1_action_id": 3,
            "raw_top1_invalid": True,
            "selected_action_id": 4,
            "selected_rank": 1,
            "all_invalid": False,
        },
        {
            "scenario_id": 1,
            "controller_decision_index": 1,
            "leaf_id": 10,
            "raw_top1_action_id": 5,
            "raw_top1_invalid": False,
            "selected_action_id": 5,
            "selected_rank": 0,
            "all_invalid": False,
        },
    ]
    mutated = [
        {
            "scenario_id": 1,
            "controller_decision_index": 0,
            "leaf_id": 9,
            "raw_top1_action_id": 3,
            "raw_top1_invalid": True,
            "selected_action_id": None,
            "selected_rank": None,
            "all_invalid": False,
        },
        {
            "scenario_id": 1,
            "controller_decision_index": 1,
            "leaf_id": 11,
            "raw_top1_action_id": 7,
            "raw_top1_invalid": False,
            "selected_action_id": 7,
            "selected_rank": 0,
            "all_invalid": False,
        },
    ]
    result = evaluate_hout_activation(
        mutation_id="B2",
        base_events=base,
        mutated_events=mutated,
        rule={
            "required_leaf_id": 10,
            "required_action_id": 5,
            "require_selection_difference": True,
            "difference_scope": "global",
        },
    )
    # The exact audit anchor is no longer visited after the earlier global
    # divergence.  The global paired trace still proves that the semantics was
    # activated.
    assert result.details["global_differences"]["selection_difference_count"] == 2


def test_refresh_rejects_stale_b3_target_before_long_run(tmp_path, monkeypatch):
    from nonvacuity_lab.config_resolver import refresh_resolved_runtime_bindings

    monkeypatch.setattr(
        "nonvacuity_lab.mutators.catalog.selection_mutations.build_selection_catalog",
        lambda source_root: {"B3": ()},
    )
    config = {
        "mutations": [
            {
                "mutation_id": "B3_demo",
                "resolved_target": {
                    "tree_path": str(tmp_path / "tree.json"),
                    "leaf_id": 1,
                    "action_id": 0,
                    "activation_evidence_field": "all_invalid_count",
                    "activation_evidence_count": 0,
                },
                "mutator": {"parameters": {}},
                "activation": {},
            }
        ]
    }
    refreshed = refresh_resolved_runtime_bindings(
        config,
        source_root=tmp_path,
        mutation_ids={"B3_demo"},
    )
    mutation = refreshed["mutations"][0]
    assert mutation["enabled"] is False
    assert mutation["resolution_status"] == "BLOCKED_NO_WITNESS"
    assert "B3_ALL_INVALID_WITNESS_UNAVAILABLE" in mutation["resolution_error"]
    assert mutation["metadata"]["runtime_block"] == {
        "status": "BLOCKED_NO_WITNESS",
        "reason": "B3_ALL_INVALID_WITNESS_UNAVAILABLE",
        "evidence_field": "all_invalid_count",
        "evidence_count": 0,
    }
