import json

import pytest

from scripts.bootstrap_formal_target_recipe import main


def _fixture(tmp_path, *, noop_id=24):
    seed_dir = tmp_path / "r0_s1775"
    artifact = seed_dir / "trees/viper/best_overall"
    artifact.mkdir(parents=True)
    (artifact / "feature_names.json").write_text(
        json.dumps({"feature_names": ["x"]}), encoding="utf-8"
    )
    actions = [{"action_id": index, "is_noop": index == noop_id} for index in range(25)]
    (artifact / "action_definitions.json").write_text(
        json.dumps({"action_definitions": actions}), encoding="utf-8"
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "workload_args": {"fixed_taskset_seed": 1775},
        "runtime_args": {"include_explicit_noop": True},
        "feature_config": {},
    }), encoding="utf-8")
    argv = ["--seed-dir", str(seed_dir), "--seed", "1775",
            "--tree-variant", "trees/viper/best_overall", "--config", str(config)]
    return seed_dir, argv


def test_bootstrap_recipe_binds_dynamic_factory_and_artifact_schema(tmp_path):
    seed_dir, argv = _fixture(tmp_path)
    assert main(argv) == 0
    recipe = json.loads((seed_dir / "formal_inputs/target_recipe.json").read_text())
    assert recipe["factory"].endswith("mc_stratified_dynamic_target:build_target")
    assert recipe["kwargs"]["runtime_args"]["include_explicit_noop"] is True
    assert recipe["kwargs"]["expected_action_definitions"][-1]["action_id"] == 24


def test_bootstrap_recipe_rejects_wrong_noop_slot(tmp_path):
    _, argv = _fixture(tmp_path, noop_id=23)
    with pytest.raises(ValueError, match="EXPLICIT_NOOP_LAYOUT"):
        main(argv)
