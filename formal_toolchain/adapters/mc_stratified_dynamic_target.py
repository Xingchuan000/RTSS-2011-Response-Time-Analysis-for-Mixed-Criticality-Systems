"""Authoritative target factory for ``mc_stratified_dynamic`` VIPER artifacts."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from amc_py.dqn.experiment import (
    build_env_from_experiment_config,
    build_mc_stratified_dynamic_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from formal_toolchain.adapters.amc_real_runtime_adapter import AMCRealRuntimeAdapter
from formal_toolchain.adapters.s185_target import FormalScenarioContract
from formal_toolchain.adapters.target_factory import FormalTarget


_REQUIRED_RUNTIME = {
    "runtime_semantics": "C_AMC_SEM",
    "end_time": 8_000_000,
    "agent_period": 25_000,
    "action_space": "single",
    "include_explicit_noop": True,
    "budget_increase_ratio": 0.02,
    "budget_decrease_ratio": 0.02,
    "budget_floor_ratio": 0.9,
    "forbid_decreasing_hi_budgets": True,
    "mask_detail_mode": "full",
    "enable_deploy_cap_mask": True,
    "deploy_cap_mask_ratio": 4.0,
    "deploy_cap_mask_criticality": "lo",
    "capture_trace": True,
    "capture_debug_events": False,
    "processor_overhead": 0,
    "c_amc_sem_xf": 0.5,
}

_REQUIRED_WORKLOAD = {
    "num_tasks": 12,
    "hi_ratio": 0.5,
    "period_family": "seed_paired",
    "period_scale": 500,
    "scenario_seed_offset": 100000,
    "require_schedulable": True,
}


def _require_exact(values: Mapping[str, Any], required: Mapping[str, Any], code: str) -> None:
    mismatches = {
        key: {"expected": expected, "actual": values.get(key)}
        for key, expected in required.items()
        if values.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{code}:{mismatches}")


def _budget_metadata(bundle: Any, floor_ratio: float) -> dict[str, dict[str, int]]:
    meta_by_name = {str(row.name): row for row in bundle.metadata["task_meta"]}
    result: dict[str, dict[str, int]] = {}
    for task in bundle.ordered_tasks:
        row = meta_by_name[str(task.name)]
        initial = int(row.initial_budget)
        result[str(task.name)] = {
            "initial_runtime_budget": initial,
            "budget_floor": max(1, math.ceil(initial * float(floor_ratio))),
            "action_hard_upper": (
                int(task.c_hi) if row.criticality.value == "HI" else int(task.deadline)
            ),
            "source_base_budget": int(
                row.base_demand_hi if row.criticality.value == "HI" else row.base_demand_lo
            ),
        }
    return result


def build_target(
    *,
    seed: int,
    workload_args: Mapping[str, Any],
    runtime_args: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    expected_feature_names: Sequence[str],
    expected_action_definitions: Sequence[Mapping[str, Any]],
    original_reward_mode: str | None = None,
    formal_reward_mode: str = "mendes",
) -> FormalTarget:
    workload_values = dict(workload_args)
    runtime_values_in = dict(runtime_args)
    _require_exact(workload_values, _REQUIRED_WORKLOAD, "DYNAMIC_WORKLOAD_CONFIG_MISMATCH")
    _require_exact(runtime_values_in, _REQUIRED_RUNTIME, "DYNAMIC_RUNTIME_CONFIG_MISMATCH")
    if int(workload_values.get("fixed_taskset_seed", -1)) != int(seed):
        raise ValueError("DYNAMIC_FIXED_TASKSET_SEED_MISMATCH")

    config = build_mc_stratified_dynamic_experiment_config(**workload_values)
    bundle = resolve_experiment_bundle(config, int(seed))
    if int(bundle.taskset_seed) != int(seed):
        raise ValueError("DYNAMIC_BUNDLE_TASKSET_SEED_MISMATCH")

    environment = build_env_from_experiment_config(
        config,
        seed=int(seed),
        end_time=int(runtime_values_in["end_time"]),
        agent_period=int(runtime_values_in["agent_period"]),
        semantics=RuntimeSemantics(str(runtime_values_in["runtime_semantics"])),
        reward_mode=str(formal_reward_mode),
        action_space=str(runtime_values_in["action_space"]),
        include_explicit_noop=bool(runtime_values_in["include_explicit_noop"]),
        budget_increase_ratio=float(runtime_values_in["budget_increase_ratio"]),
        budget_decrease_ratio=float(runtime_values_in["budget_decrease_ratio"]),
        budget_floor_ratio=float(runtime_values_in["budget_floor_ratio"]),
        forbid_decreasing_hi_budgets=bool(runtime_values_in["forbid_decreasing_hi_budgets"]),
        budget_rounding_mode="ceil_floor",
        min_budget_delta=1,
        mask_detail_mode=str(runtime_values_in["mask_detail_mode"]),
        enable_deploy_cap_mask=bool(runtime_values_in["enable_deploy_cap_mask"]),
        deploy_cap_mask_ratio=float(runtime_values_in["deploy_cap_mask_ratio"]),
        deploy_cap_mask_criticality=str(runtime_values_in["deploy_cap_mask_criticality"]),
        capture_trace=bool(runtime_values_in["capture_trace"]),
        capture_debug_events=bool(runtime_values_in["capture_debug_events"]),
        c_amc_sem_xf=float(runtime_values_in["c_amc_sem_xf"]),
        feature_config=FeatureConfig(**dict(feature_config)),
    )
    feature_names = tuple(environment.get_observation_feature_names())
    action_definitions = tuple(environment.get_action_definitions())
    expected_features = tuple(str(value) for value in expected_feature_names)
    expected_actions = [dict(value) for value in expected_action_definitions]
    if feature_names != expected_features:
        raise ValueError("REAL_TARGET_FEATURE_SCHEMA_MISMATCH")
    if list(action_definitions) != expected_actions:
        raise ValueError("REAL_TARGET_ACTION_SCHEMA_MISMATCH")
    if tuple(task.name for task in environment.ordered_tasks) != tuple(task.name for task in bundle.ordered_tasks):
        raise ValueError("REAL_TARGET_TASK_ORDER_MISMATCH")
    if len(environment.ordered_tasks) != 12 or len(feature_names) != 128 or len(action_definitions) != 25:
        raise ValueError("DYNAMIC_TARGET_DIMENSION_MISMATCH")
    noop_rows = [row for row in action_definitions if bool(row.get("is_noop", False))]
    if len(noop_rows) != 1 or int(noop_rows[0]["action_id"]) != 24:
        raise ValueError("DYNAMIC_TARGET_NOOP_LAYOUT_MISMATCH")

    effective = {
        "end_time": int(environment.runtime_config.end_time),
        "jobs_per_task": int(environment.runtime_config.jobs_per_task),
        "hyperperiod_limit": int(environment.runtime_config.hyperperiod_limit),
        "capture_trace": bool(environment.runtime_config.capture_trace),
        "capture_debug_events": bool(environment.runtime_config.capture_debug_events),
        "stop_at_first_miss": bool(environment.runtime_config.stop_at_first_miss),
        "drop_lo_jobs_on_hi_switch": bool(environment.runtime_config.drop_lo_jobs_on_hi_switch),
        "semantics": environment.runtime_config.semantics,
        "record_dropped_lo_releases": bool(environment.runtime_config.record_dropped_lo_releases),
        "c_amc_sem_lo_degradation_ratio": float(environment.runtime_config.c_amc_sem_lo_degradation_ratio),
        "c_amc_sem_primary_on_switch_time": bool(environment.runtime_config.c_amc_sem_primary_on_switch_time),
        "processor_overhead": int(runtime_values_in["processor_overhead"]),
        "agent_period": int(environment.agent_period),
        "action_space": str(environment.action_space),
        "include_explicit_noop": bool(environment.include_explicit_noop),
        "budget_increase_ratio": float(environment.budget_increase_ratio),
        "budget_decrease_ratio": float(environment.budget_decrease_ratio),
        "budget_floor_ratio": float(environment.budget_floor_ratio),
        "forbid_decreasing_hi_budgets": bool(environment.forbid_decreasing_hi_budgets),
        "mask_detail_mode": str(environment.mask_detail_mode),
        "enable_deploy_cap_mask": bool(environment.enable_deploy_cap_mask),
        "deploy_cap_mask_ratio": float(environment.deploy_cap_mask_ratio),
        "deploy_cap_mask_criticality": str(environment.deploy_cap_mask_criticality),
        "check_safety": bool(environment.check_safety),
        "observation_mode": str(environment.feature_config.observation_mode),
        "budget_rounding_mode": "ceil_floor",
        "min_budget_delta": 1,
    }
    runtime_view = SimpleNamespace(**effective)
    environment_view = SimpleNamespace(**effective)
    budget = _budget_metadata(bundle, float(runtime_values_in["budget_floor_ratio"]))
    adapter = AMCRealRuntimeAdapter(
        environment,
        action_space=tuple(environment._actions),
        selection_semantics="ranked_first_valid",
        disabled_guards=(),
        rounding_mode="ceil_floor",
        min_budget_delta=1,
    )
    return FormalTarget(
        ordered_tasks=tuple(bundle.ordered_tasks),
        runtime_config=runtime_view,
        environment=environment_view,
        policy=None,
        scenario=FormalScenarioContract(bundle.scenario),
        action_definitions=action_definitions,
        feature_names=feature_names,
        provenance={
            "adapter_kind": "REAL_AMC_RUNTIME",
            "workload_family": "mc_stratified_dynamic",
            "taskset_seed": int(bundle.taskset_seed),
            "scenario_seed": int(bundle.scenario_seed),
            "taskset_attempts": int(bundle.taskset_attempts),
            "taskset_fingerprint_short": str(bundle.taskset_fingerprint),
            "budget_by_task": budget,
            "feature_metadata": {"feature_names": list(feature_names)},
            "original_reward_mode": original_reward_mode,
            "formal_reward_mode": formal_reward_mode,
        },
        runtime_adapter=adapter,
    )


__all__ = ["build_target"]
