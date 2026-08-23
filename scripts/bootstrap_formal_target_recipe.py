"""Create a seed-local formal target recipe from a VIPER tree artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap_list(value: Any, *keys: str) -> list[Any]:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError(f"ARTIFACT_LIST_REQUIRED:{','.join(keys)}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--tree-variant", default="best_overall")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    seed_dir = args.seed_dir.resolve()
    artifact = seed_dir / args.tree_variant
    config = _read(args.config.resolve())
    workload_args = dict(config["workload_args"])
    if int(workload_args.get("fixed_taskset_seed", args.seed)) != args.seed:
        raise ValueError("FORMAL_RECIPE_FIXED_TASKSET_SEED_MISMATCH")

    features = [str(value) for value in _unwrap_list(
        _read(artifact / "feature_names.json"), "feature_names"
    )]
    actions = [dict(value) for value in _unwrap_list(
        _read(artifact / "action_definitions.json"),
        "action_definitions", "actions",
    )]
    if len(actions) != 25:
        raise ValueError("FORMAL_RECIPE_EXPECTED_SINGLE25_ACTIONS")
    noops = [row for row in actions if bool(row.get("is_noop", False))]
    if len(noops) != 1 or int(noops[0].get("action_id", -1)) != 24:
        raise ValueError("FORMAL_RECIPE_EXPLICIT_NOOP_LAYOUT_MISMATCH")

    recipe = {
        "schema_version": "real_viper_seed_target_recipe_v2",
        "factory": "formal_toolchain.adapters.mc_stratified_dynamic_target:build_target",
        "kwargs": {
            "seed": args.seed,
            "workload_args": workload_args,
            "runtime_args": dict(config["runtime_args"]),
            "feature_config": dict(config["feature_config"]),
            "expected_feature_names": features,
            "expected_action_definitions": actions,
            "original_reward_mode": config.get("original_reward_mode"),
            "formal_reward_mode": config.get("formal_reward_mode", "mendes"),
        },
    }
    output = seed_dir / "formal_inputs" / "target_recipe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
