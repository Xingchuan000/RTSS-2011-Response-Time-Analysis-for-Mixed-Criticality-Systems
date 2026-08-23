"""导出真实 seed 的 authoritative formal_inputs。

脚本只从实际 workload provider、runtime 配置和 tree artifact 导出 canonical
文件；不会从 HOUT、模型输出或排序后的副本反推输入。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_toolchain.adapters.amc_taskset import export_taskset
from formal_toolchain.adapters.target_factory import build_target
from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.core.hashing import sha256_file


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_seed(seed_dir: Path, *, seed: int = 185, variant: str = "best_overall") -> None:
    seed_dir = Path(seed_dir).resolve()
    artifact = seed_dir / variant
    recipe = json.loads((seed_dir / "formal_inputs/target_recipe.json").read_text(encoding="utf-8"))
    factory = recipe.get("factory")
    if not isinstance(factory, str) or ":" not in factory:
        raise ValueError("TARGET_RECIPE_FACTORY_INVALID")
    recipe_kwargs = dict(recipe.get("kwargs", {}))
    if int(recipe_kwargs.get("seed", seed)) != int(seed):
        raise ValueError("TARGET_RECIPE_SEED_MISMATCH")
    target = build_target(factory, recipe_kwargs)
    taskset = export_taskset(target.ordered_tasks, target.provenance["budget_by_task"])
    feature_names = list(target.feature_names)
    actions = list(target.action_definitions)
    inputs = seed_dir / "formal_inputs"
    _write(inputs / "code_taskset_canonical.json", taskset)
    _write(inputs / "priority_order.json", {"schema_version": "priority_order_v1", "priority_order": taskset["priority_order"]})
    _write(inputs / "effective_runtime_config.json", export_formal_target_config(target))
    _write(inputs / "action_definitions_canonical.json", {"schema_version": "action_definitions_canonical_v1", "action_definitions": actions})
    _write(inputs / "feature_schema_canonical.json", {"schema_version": "feature_schema_canonical_v1", "feature_names": feature_names})
    _write(inputs / "source_tree_manifest.json", {"schema_version": "source_tree_manifest_v1", "note": "由 verifier 按 code_root fresh 重算"})
    _write(inputs / "runtime_environment_manifest.json", {"schema_version": "runtime_environment_manifest_v1", "python": __import__("platform").python_version()})
    _write(inputs / "dependency_manifest.json", {"schema_version": "dependency_manifest_v1", "source": "current-python-environment"})
    noop_ids = list(target.runtime_adapter.export_mask_contract().get("explicit_noop_action_ids", ()))
    _write(inputs / "provenance.json", {"schema_version": "real_seed_provenance_v1", "seed": seed, "variant": variant,
                                          "taskset_seed": int(target.provenance["taskset_seed"]), "scenario_seed": int(target.provenance["scenario_seed"]),
                                          "taskset_fingerprint": taskset["fingerprint"],
                                          "target_factory": factory,
                                          "workload_family": target.provenance.get("workload_family"),
                                          "action_dim": len(target.action_definitions),
                                          "explicit_noop": bool(noop_ids),
                                          "explicit_noop_action_id": noop_ids[0] if len(noop_ids) == 1 else None,
                                          "tree_integer_sha256": sha256_file(artifact / "integer_tree.json")})
    manifest = {"schema_version": "formal_target_manifest_v1", "target_id": f"s{seed}",
                "target_kind": "REAL_VIPER_SEED", "taskset_seed": seed,
                "tree_variants": [variant], "authoritative_input_mode": "FROZEN_FORMAL_INPUTS",
                "formal_inputs_version": "real_seed_p0_v3_explicit_noop"}
    _write(seed_dir / "formal_target_manifest.json", manifest)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=Path, default=Path("s185"))
    parser.add_argument("--seed", type=int, default=185)
    parser.add_argument("--tree-variant", default="best_overall")
    args = parser.parse_args(argv)
    export_seed(args.seed_dir, seed=args.seed, variant=args.tree_variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
