"""导出真实 seed 的 authoritative formal_inputs。

脚本只从实际 workload provider、runtime 配置和 tree artifact 导出 canonical
文件；不会从 HOUT、模型输出或排序后的副本反推输入。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

from amc_py.dqn.experiment import build_env_from_experiment_config, build_mc_fairgen_experiment_config, resolve_experiment_bundle
from amc_py.runtime_models import RuntimeSemantics
from amc_py.rl.feature_config import FeatureConfig
from formal_toolchain.adapters.amc_taskset import export_taskset
from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.core.hashing import sha256_file, sha256_object


class RuntimeConfig(SimpleNamespace):
    """与真实 target factory 同名的 authoritative 配置视图。"""


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_seed(seed_dir: Path, *, seed: int = 185, variant: str = "best_overall") -> None:
    seed_dir = Path(seed_dir).resolve()
    artifact = seed_dir / variant
    recipe = json.loads((seed_dir / "formal_inputs/target_recipe.json").read_text(encoding="utf-8"))
    config = build_mc_fairgen_experiment_config(**recipe["workload_args"])
    bundle = resolve_experiment_bundle(config, seed)
    runtime_args = recipe["runtime_args"]
    environment = build_env_from_experiment_config(
        config, seed=seed, end_time=int(runtime_args["end_time"]), agent_period=int(runtime_args["agent_period"]),
        semantics=RuntimeSemantics.C_AMC_SEM, action_space=runtime_args["action_space"],
        budget_increase_ratio=float(runtime_args["budget_increase_ratio"]),
        budget_decrease_ratio=float(runtime_args["budget_decrease_ratio"]),
        budget_floor_ratio=float(runtime_args["budget_floor_ratio"]),
        forbid_decreasing_hi_budgets=bool(runtime_args["forbid_decreasing_hi_budgets"]),
        mask_detail_mode=runtime_args["mask_detail_mode"],
        enable_deploy_cap_mask=bool(runtime_args["enable_deploy_cap_mask"]),
        deploy_cap_mask_ratio=float(runtime_args["deploy_cap_mask_ratio"]),
        deploy_cap_mask_criticality=runtime_args["deploy_cap_mask_criticality"],
        c_amc_sem_xf=float(runtime_args["c_amc_sem_xf"]),
        feature_config=FeatureConfig(observation_mode=runtime_args["observation_mode"]),
    )
    metadata = bundle.metadata["task_meta"]
    budget = {row.name: {"initial_runtime_budget": int(row.initial_budget),
                         "budget_floor": int(row.initial_budget * 0.9),
                         "budget_cap": int(row.base_c_hi if row.criticality.value == "HI" else row.base_c_lo)} for row in metadata}
    taskset = export_taskset(bundle.ordered_tasks, budget)
    feature_names = json.loads((artifact / "feature_names.json").read_text(encoding="utf-8"))
    actions = json.loads((artifact / "action_definitions.json").read_text(encoding="utf-8"))
    visible = SimpleNamespace(**{name: getattr(environment.runtime_config, name) for name in (
        "semantics", "drop_lo_jobs_on_hi_switch", "c_amc_sem_lo_degradation_ratio",
        "c_amc_sem_primary_on_switch_time", "stop_at_first_miss", "capture_trace",
        "capture_debug_events")}, **{name: runtime_args[name] for name in (
        "agent_period", "action_space", "budget_increase_ratio", "budget_decrease_ratio",
        "budget_floor_ratio", "forbid_decreasing_hi_budgets", "mask_detail_mode",
        "enable_deploy_cap_mask", "deploy_cap_mask_ratio", "deploy_cap_mask_criticality")},
        observation_mode="v11_full_10d")
    if is_dataclass(environment.runtime_config):
        runtime_values = {field.name: getattr(environment.runtime_config, field.name)
                          for field in fields(environment.runtime_config)}
    else:
        runtime_values = {name: getattr(environment.runtime_config, name)
                          for name in dir(environment.runtime_config)
                          if not name.startswith("_") and not callable(getattr(environment.runtime_config, name))}
    runtime_values["processor_overhead"] = int(runtime_args["processor_overhead"])
    target = SimpleNamespace(ordered_tasks=bundle.ordered_tasks,
                            runtime_config=RuntimeConfig(**runtime_values),
                            environment=visible, provenance={"budget_by_task": budget}, feature_names=feature_names)
    inputs = seed_dir / "formal_inputs"
    _write(inputs / "code_taskset_canonical.json", taskset)
    _write(inputs / "priority_order.json", {"schema_version": "priority_order_v1", "priority_order": taskset["priority_order"]})
    _write(inputs / "effective_runtime_config.json", export_formal_target_config(target))
    _write(inputs / "action_definitions_canonical.json", {"schema_version": "action_definitions_canonical_v1", "action_definitions": actions})
    _write(inputs / "feature_schema_canonical.json", {"schema_version": "feature_schema_canonical_v1", "feature_names": feature_names})
    _write(inputs / "source_tree_manifest.json", {"schema_version": "source_tree_manifest_v1", "note": "由 verifier 按 code_root fresh 重算"})
    _write(inputs / "runtime_environment_manifest.json", {"schema_version": "runtime_environment_manifest_v1", "python": __import__("platform").python_version()})
    _write(inputs / "dependency_manifest.json", {"schema_version": "dependency_manifest_v1", "source": "current-python-environment"})
    _write(inputs / "provenance.json", {"schema_version": "real_seed_provenance_v1", "seed": seed, "variant": variant,
                                          "taskset_seed": int(bundle.taskset_seed), "scenario_seed": int(bundle.scenario_seed),
                                          "taskset_fingerprint": taskset["fingerprint"],
                                          "tree_integer_sha256": sha256_file(artifact / "integer_tree.json")})
    manifest = {"schema_version": "formal_target_manifest_v1", "target_id": f"s{seed}",
                "target_kind": "REAL_VIPER_SEED", "taskset_seed": seed,
                "tree_variants": [variant], "authoritative_input_mode": "FROZEN_FORMAL_INPUTS",
                "formal_inputs_version": "s185_p0_v2"}
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
