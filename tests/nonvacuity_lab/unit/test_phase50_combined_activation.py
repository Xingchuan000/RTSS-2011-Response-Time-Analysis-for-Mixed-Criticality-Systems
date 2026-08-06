from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nonvacuity_lab.activation.hout_activation import evaluate_hout_activation
from nonvacuity_lab.activation.symbolic_model_builder import _normalize_action
from nonvacuity_lab.runners import campaign as campaign_runner
from nonvacuity_lab.schema import ActivationStatus, MutationManifest


def test_hout_activation_can_require_selection_difference():
    common = {
        "scenario_seed": 1,
        "controller_decision_index": 0,
        "leaf_id": 7,
        "raw_top1_action_id": 13,
        "raw_top1_invalid": True,
    }
    result = evaluate_hout_activation(
        mutation_id="B2",
        base_events=[{**common, "selected_action_id": 2, "selected_rank": 1}],
        mutated_events=[{**common, "selected_action_id": None, "selected_rank": None}],
        rule={
            "required_leaf_id": 7,
            "required_action_id": 13,
            "require_baseline_reject": True,
            "require_selection_difference": True,
        },
    )
    assert result.status is ActivationStatus.ACTIVATED
    assert result.details["selection_difference_count"] == 1


def test_combined_activation_uses_hout_when_symbolic_model_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    base_events = tmp_path / "base.jsonl"
    mutated_events = tmp_path / "mutated.jsonl"
    common = {
        "scenario_seed": 1,
        "controller_decision_index": 0,
        "leaf_id": 7,
        "raw_top1_action_id": 13,
        "raw_top1_invalid": True,
    }
    base_events.write_text(json.dumps({**common, "selected_action_id": 2}) + "\n", encoding="utf-8")
    mutated_events.write_text(json.dumps({**common, "selected_action_id": None}) + "\n", encoding="utf-8")

    def unsupported_symbolic(**kwargs):
        del kwargs
        raise ValueError("UNSUPPORTED_DYNAMIC_ACTION_SLOT:safe_transfer")

    monkeypatch.setattr(campaign_runner, "run_auto_symbolic_activation", unsupported_symbolic)
    manifest = MutationManifest.from_mapping({
        "schema_version": "nonvacuity_mutation_v1",
        "mutation_id": "B2_demo",
        "mutation_class": "NO_FIRST_VALID",
        "enabled": True,
        "seed_dir": str(tmp_path),
        "tree_variant": "best_overall",
        "mutator": {"kind": "coherent_source_patch", "parameters": {}},
        "activation": {
            "mode": "symbolic_auto_or_hout",
            "required_leaf_id": 7,
            "required_action_id": 13,
            "require_baseline_reject": True,
            "require_selection_difference": True,
        },
        "expected": {"allowed_result_statuses": ["DEPLOYED_TREE_PROVED"]},
        "metadata": {
            "resolved_target": {"tree_path": str(tmp_path / "tree.json"), "leaf_id": 7, "action_id": 13}
        },
    }, base_dir=tmp_path)
    activation_dir = tmp_path / "activation"
    activation_dir.mkdir()
    workspace = SimpleNamespace(
        source_overlay=tmp_path / "overlay",
        activation_output=activation_dir,
    )
    payload = campaign_runner._run_activation(
        manifest,
        workspace,
        {
            "setup_valid": True,
            "base_events": str(base_events),
            "mutated_events": str(mutated_events),
        },
        source_root=tmp_path,
    )
    assert payload["status"] == "ACTIVATED"
    assert {item["status"] for item in payload["all_evidence"]} == {
        "ACTIVATED",
        "ACTIVATION_SETUP_INVALID",
    }


def test_dynamic_action_slot_is_not_silently_treated_as_noop():
    with pytest.raises(ValueError, match="UNSUPPORTED_DYNAMIC_ACTION_SLOT"):
        _normalize_action(
            {
                "action_id": 4,
                "is_residual_ranked": True,
                "residual_action_type": "safe_transfer_global_low_to_lo_risk",
            },
            {"default_action_ratio": "1/50"},
        )
