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


def test_c1_inc_only_updates_real_copied_recipe_and_artifact(tmp_path: Path):
    root = Path(__file__).resolve().parents[3]
    seed = tmp_path / "seed"
    (seed / "best_overall").mkdir(parents=True)
    (seed / "formal_inputs").mkdir()
    for name in ("action_definitions.json", "artifact_manifest.json"):
        shutil.copy2(root / "s185" / "best_overall" / name, seed / "best_overall" / name)
    shutil.copy2(
        root / "s185" / "formal_inputs" / "target_recipe.json",
        seed / "formal_inputs" / "target_recipe.json",
    )
    mutator = ActionStepMutation()
    result = mutator.apply(
        MutationContext(
            mutation_id="C1",
            source_root=root,
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
    actions = json.loads(
        (seed / "best_overall" / "action_definitions.json").read_text(encoding="utf-8")
    )
    recipe = json.loads(
        (seed / "formal_inputs" / "target_recipe.json").read_text(encoding="utf-8")
    )
    assert mutator.verify_single_change(result).status == "PASS"
    assert all(
        row["increase_ratio"] == 0.05
        for row in actions
        if row["increase_task"] is not None
    )
    assert all(
        row["decrease_ratio"] == 0.02
        for row in actions
        if row["decrease_tasks"]
    )
    assert recipe["kwargs"]["runtime_args"]["budget_increase_ratio"] == 0.05
    assert recipe["kwargs"]["runtime_args"]["budget_decrease_ratio"] == 0.02
