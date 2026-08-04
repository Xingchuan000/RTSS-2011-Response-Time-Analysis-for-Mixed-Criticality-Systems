from __future__ import annotations

import json
import shutil
from pathlib import Path

from amc_py.viper.schema import INTEGER_TREE_SCHEMA_VERSION
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from nonvacuity_lab.mutators.base import MutationContext
from nonvacuity_lab.mutators.tree_ranking import DangerousTop1Mutation


def test_tree_ranking_mutation_changes_one_leaf_and_updates_manifest(tmp_path: Path):
    root = Path(__file__).resolve().parents[3]
    seed = tmp_path / "seed"
    shutil.copytree(
        root / "tests" / "formal" / "fixtures" / "synthetic_p0" / "best_overall",
        seed / "best_overall",
    )
    tree_path = seed / "best_overall" / "integer_tree.json"
    before = json.loads(tree_path.read_text(encoding="utf-8"))
    before["schema_version"] = INTEGER_TREE_SCHEMA_VERSION
    tree_path.write_text(json.dumps(before, separators=(",", ":")) + "\n", encoding="utf-8")
    target_leaf = before["leaves"][0]
    original_ranking = list(target_leaf["action_ranking"])
    action_id = original_ranking[-1]
    mutator = DangerousTop1Mutation()
    result = mutator.apply(
        MutationContext(
            mutation_id="A1",
            source_root=root,
            mutated_seed=seed,
            source_overlay=None,
            parameters={
                "tree_variant": "best_overall",
                "leaf_id": target_leaf["node_id"],
                "action_id": action_id,
            },
        )
    )
    assert mutator.verify_single_change(result).status == "PASS"
    after = json.loads(tree_path.read_text(encoding="utf-8"))
    changed_leaf = after["leaves"][0]
    assert changed_leaf["action_ranking"][0] == action_id
    assert changed_leaf["raw_action_id"] == action_id
    assert changed_leaf["full_action_counts"] == target_leaf["full_action_counts"]
    assert after["leaves"][1:] == before["leaves"][1:]
    inspect_tree_artifact(
        seed / "best_overall",
        expected_state_dim=38,
        expected_action_dim=6,
        expected_seed=None,
    )
