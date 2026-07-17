"""真实 s185 target factory。

任务集参数、priority 顺序和 runtime 参数来自仓库内冻结的 formal_inputs；
factory 不从 HOUT 或 tree artifact 推断 taskset，也不自动排序 feature/action。
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

from formal_toolchain.adapters.amc_real_runtime_adapter import AMCRealRuntimeAdapter
from formal_toolchain.adapters.target_factory import FormalTarget
from amc_py.rl.feature_config import FeatureConfig


class RuntimeConfig(SimpleNamespace):
    """真实 provider 配置的形式化只读视图。

    provider 的 slots dataclass 没有 action/mask 扩展字段；该视图只把
    authoritative recipe 的字段显式合并进同一配置对象，保持导出的来源
    类型仍为 RuntimeConfig，避免把 wrapper 类型变化伪装成配置变化。
    """


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_target(*, seed: int = 185, tree_variant: str = "best_overall", **_kwargs):
    root = Path.cwd()
    inputs = root / f"s{seed}" / "formal_inputs"
    artifact = root / f"s{seed}" / tree_variant
    if not inputs.is_dir() or not artifact.is_dir():
        raise RuntimeError("AUTHORITATIVE_TARGET_MISSING")
    from amc_py.dqn.experiment import build_env_from_experiment_config, build_mc_fairgen_experiment_config, resolve_experiment_bundle
    recipe = _read(inputs / "target_recipe.json")
    bundle = resolve_experiment_bundle(build_mc_fairgen_experiment_config(**recipe["workload_args"]), seed)
    feature_names = tuple(_read(artifact / "feature_names.json"))
    action_definitions = tuple(_read(artifact / "action_definitions.json"))
    priority = tuple(_read(inputs / "priority_order.json")["priority_order"])
    actual = tuple(task.name for task in bundle.ordered_tasks)
    if actual != priority or tuple(item.split(".")[1] for item in feature_names[:10 * len(actual):10]) != priority:
        raise RuntimeError("FEATURE_TASK_PRIORITY_ORDER_MISMATCH")
    env = build_env_from_experiment_config(
        build_mc_fairgen_experiment_config(**recipe["workload_args"]), seed=seed,
        end_time=int(recipe["runtime_args"].get("end_time", 8000000)),
        agent_period=int(recipe["runtime_args"]["agent_period"]),
        semantics=__import__("amc_py.runtime_models", fromlist=["RuntimeSemantics"]).RuntimeSemantics.C_AMC_SEM,
        action_space=recipe["runtime_args"]["action_space"],
        budget_increase_ratio=float(recipe["runtime_args"]["budget_increase_ratio"]),
        budget_decrease_ratio=float(recipe["runtime_args"]["budget_decrease_ratio"]),
        budget_floor_ratio=float(recipe["runtime_args"]["budget_floor_ratio"]),
        forbid_decreasing_hi_budgets=bool(recipe["runtime_args"]["forbid_decreasing_hi_budgets"]),
        mask_detail_mode=recipe["runtime_args"]["mask_detail_mode"],
        enable_deploy_cap_mask=bool(recipe["runtime_args"]["enable_deploy_cap_mask"]),
        deploy_cap_mask_ratio=float(recipe["runtime_args"]["deploy_cap_mask_ratio"]),
        deploy_cap_mask_criticality=recipe["runtime_args"]["deploy_cap_mask_criticality"],
        c_amc_sem_xf=float(recipe["runtime_args"]["c_amc_sem_xf"]),
        feature_config=FeatureConfig(observation_mode=recipe["runtime_args"]["observation_mode"]),
    )
    if tuple(dict(row).get("action_id") for row in action_definitions) != tuple(action.action_id for action in env._actions):
        raise RuntimeError("ACTION_ORDER_MISMATCH")
    metadata = _read(inputs / "code_taskset_canonical.json")
    budget_by_task = {row["name"]: {"initial_runtime_budget": int(row["initial_runtime_budget"]),
                                     "budget_floor": int(row["budget_floor"]),
                                     "budget_cap": int(row["budget_cap"])}
                      for row in metadata["ordered_tasks"]}
    # provider 的 RuntimeConfig 只保存运行内核字段，形式化流水线还需要
    # recipe 中明确冻结的 action/mask 参数。这里建立一个合并后的只读
    # 配置视图；数值仍逐项来自实际 env 或 authoritative recipe，不补默认值。
    if is_dataclass(env.runtime_config):
        runtime_values = {field.name: getattr(env.runtime_config, field.name)
                          for field in fields(env.runtime_config)}
    else:
        runtime_values = {name: getattr(env.runtime_config, name)
                          for name in dir(env.runtime_config)
                          if not name.startswith("_") and not callable(getattr(env.runtime_config, name))}
    runtime_values["processor_overhead"] = int(recipe["runtime_args"]["processor_overhead"])
    runtime_values.update({
        name: recipe["runtime_args"][name] for name in (
            "agent_period", "action_space", "budget_increase_ratio", "budget_decrease_ratio",
            "budget_floor_ratio", "forbid_decreasing_hi_budgets", "mask_detail_mode",
            "enable_deploy_cap_mask", "deploy_cap_mask_ratio", "deploy_cap_mask_criticality")})
    runtime_config = RuntimeConfig(**runtime_values)
    visible = SimpleNamespace(**{name: getattr(env.runtime_config, name) for name in (
        "semantics", "drop_lo_jobs_on_hi_switch", "c_amc_sem_lo_degradation_ratio",
        "c_amc_sem_primary_on_switch_time", "stop_at_first_miss", "capture_trace",
        "capture_debug_events")}, **{name: recipe["runtime_args"][name] for name in (
        "agent_period", "action_space", "budget_increase_ratio", "budget_decrease_ratio",
        "budget_floor_ratio", "forbid_decreasing_hi_budgets", "mask_detail_mode",
        "enable_deploy_cap_mask", "deploy_cap_mask_ratio", "deploy_cap_mask_criticality")},
        observation_mode="v11_full_10d")
    return FormalTarget(
        ordered_tasks=tuple(bundle.ordered_tasks), runtime_config=runtime_config,
        environment=visible, policy=SimpleNamespace(name="s185_best_overall_integer_tree"),
        scenario=bundle.scenario, action_definitions=action_definitions,
        feature_names=feature_names,
        provenance={"taskset_seed": seed, "adapter_kind": "REAL_AMC_RUNTIME",
                    "budget_by_task": budget_by_task, "tree_variant": tree_variant},
        runtime_adapter=AMCRealRuntimeAdapter(env, action_space=env._actions),
    )
