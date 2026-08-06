from __future__ import annotations

import json
import shutil
from pathlib import Path

from nonvacuity_lab.mutators.action_config import ActionConfigMutation, ActionStepMutation
from nonvacuity_lab.mutators.base import MutationContext


def test_inc_only_action_change_updates_declared_entries_and_hash(tmp_path: Path):
    seed = tmp_path / "seed"
    (seed / "tree").mkdir(parents=True)
    actions = seed / "tree" / "action_definitions.json"
    actions.write_text(
        json.dumps(
            {
                "actions": [
                    {"increase_ratio": 0.02, "decrease_ratio": 0.02},
                    {"increase_ratio": 0.02, "decrease_ratio": 0.02},
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = seed / "tree" / "artifact_manifest.json"
    manifest.write_text(
        json.dumps({"file_hashes": {"action_definitions.json": "old"}}),
        encoding="utf-8",
    )
    mutator = ActionConfigMutation()
    result = mutator.apply(
        MutationContext(
            mutation_id="C1_INC_ONLY",
            source_root=tmp_path,
            mutated_seed=seed,
            source_overlay=None,
            parameters={
                "target_file": "tree/action_definitions.json",
                "semantic_group": "increase_ratio_2_to_5",
                "patches": [
                    {
                        "json_pointer": "/actions/0/increase_ratio",
                        "expected_before": 0.02,
                        "value": 0.05,
                    },
                    {
                        "json_pointer": "/actions/1/increase_ratio",
                        "expected_before": 0.02,
                        "value": 0.05,
                    },
                ],
                "hash_updates": [
                    {
                        "target_file": "tree/artifact_manifest.json",
                        "json_pointer": "/file_hashes/action_definitions.json",
                        "hash_kind": "file_sha256",
                    }
                ],
            },
        )
    )
    after = json.loads(actions.read_text(encoding="utf-8"))
    assert result.status == "PASS"
    assert result.semantic_change_count == 1
    assert after["actions"][0]["increase_ratio"] == 0.05
    assert after["actions"][0]["decrease_ratio"] == 0.02
    assert result.artifact_manifest_validation == "PASS"


def test_c1_inc_only_updates_copied_recipe_and_artifact(tmp_path: Path):
    seed = tmp_path / "seed"
    variant = seed / "best_overall"
    formal_inputs = seed / "formal_inputs"
    variant.mkdir(parents=True)
    formal_inputs.mkdir()
    actions = [
        {
            "action_id": 0,
            "increase_task": "HI",
            "decrease_tasks": [],
            "increase_ratio": 0.02,
            "decrease_ratio": 0.02,
        },
        {
            "action_id": 1,
            "increase_task": None,
            "decrease_tasks": ["LO"],
            "increase_ratio": 0.02,
            "decrease_ratio": 0.02,
        },
    ]
    (variant / "action_definitions.json").write_text(
        json.dumps({"actions": actions}), encoding="utf-8"
    )
    (variant / "artifact_manifest.json").write_text(
        json.dumps({"files": {"action_definitions.json": "old"}}),
        encoding="utf-8",
    )
    (formal_inputs / "target_recipe.json").write_text(
        json.dumps(
            {
                "factory": "example:build",
                "kwargs": {
                    "expected_action_definitions": actions,
                    "runtime_args": {
                        "budget_increase_ratio": 0.02,
                        "budget_decrease_ratio": 0.02,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    mutator = ActionStepMutation()
    result = mutator.apply(
        MutationContext(
            mutation_id="C1",
            source_root=tmp_path,
            mutated_seed=seed,
            source_overlay=None,
            parameters={
                "tree_variant": "best_overall",
                "direction": "inc_only",
                "before_ratio": 0.02,
                "after_ratio": 0.05,
            },
        )
    )
    updated_actions = json.loads(
        (variant / "action_definitions.json").read_text(encoding="utf-8")
    )["actions"]
    recipe = json.loads(
        (formal_inputs / "target_recipe.json").read_text(encoding="utf-8")
    )
    assert mutator.verify_single_change(result).status == "PASS"
    assert updated_actions[0]["increase_ratio"] == 0.05
    assert updated_actions[0]["decrease_ratio"] == 0.02
    assert updated_actions[1]["increase_ratio"] == 0.02
    assert updated_actions[1]["decrease_ratio"] == 0.02
    assert recipe["kwargs"]["runtime_args"]["budget_increase_ratio"] == 0.05
    assert recipe["kwargs"]["runtime_args"]["budget_decrease_ratio"] == 0.02


def test_action_step_matches_recipe_rows_by_action_id_not_position(tmp_path: Path):
    seed = tmp_path / "seed"
    variant = seed / "best_overall"
    formal_inputs = seed / "formal_inputs"
    variant.mkdir(parents=True)
    formal_inputs.mkdir(parents=True)
    actions = [
        {"action_id": 0, "increase_task": "lo0", "increase_ratio": 0.02, "decrease_tasks": []},
        {"action_id": 1, "increase_task": None, "increase_ratio": 0.02, "decrease_tasks": ["lo0"], "decrease_ratio": 0.02},
    ]
    (variant / "action_definitions.json").write_text(json.dumps({"actions": actions}), encoding="utf-8")
    (variant / "artifact_manifest.json").write_text(
        json.dumps({"files": {"action_definitions.json": "old"}}), encoding="utf-8"
    )
    (formal_inputs / "target_recipe.json").write_text(
        json.dumps({
            "factory": "example:build",
            "kwargs": {
                "expected_action_definitions": list(reversed(actions)),
                "runtime_args": {
                    "budget_increase_ratio": 0.02,
                    "budget_decrease_ratio": 0.02,
                },
            },
        }),
        encoding="utf-8",
    )
    result = ActionStepMutation().apply(
        MutationContext(
            mutation_id="C1",
            source_root=tmp_path,
            mutated_seed=seed,
            source_overlay=None,
            parameters={
                "tree_variant": "best_overall",
                "direction": "inc_only",
                "before_ratio": 0.02,
                "after_ratio": 0.05,
            },
        )
    )
    assert result.status == "PASS"
    artifact = json.loads((variant / "action_definitions.json").read_text(encoding="utf-8"))["actions"]
    recipe = json.loads((formal_inputs / "target_recipe.json").read_text(encoding="utf-8"))
    expected = recipe["kwargs"]["expected_action_definitions"]
    assert artifact == expected
    assert expected[0]["action_id"] == 0
    assert expected[0]["increase_ratio"] == 0.05
