"""Authoritative real VIPER seed target factory used by frozen formal_inputs."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Mapping

from amc_py.dqn.experiment import (
    build_env_from_experiment_config,
    build_mc_fairgen_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics
from formal_toolchain.adapters.amc_real_runtime_adapter import AMCRealRuntimeAdapter
from formal_toolchain.adapters.target_factory import FormalTarget


class FormalScenarioContract:
    """Expose the actual scenario plus the P0 contract consumed by the toolchain."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.name = getattr(delegate, "name", type(delegate).__name__)

    def actual_cost_for(self, task: Any, release_index: int) -> int:
        return int(self.delegate.actual_cost_for(task, release_index))

    def export_formal_contract(self) -> dict[str, Any]:
        # These properties are enforced jointly by ExecutionScenario validation,
        # the seeded provider, and the current event-runtime binding.
        return {
            "schema_version": "real_seed_scenario_contract_v1",
            "total": True,
            "positive_integer_codomain": True,
            "non_anticipating": True,
            "batch_entry_frozen": True,
            "key_stable_repeated_read": True,
            "projection_order_idempotent": True,
            "hi_upper_bound": True,
            "normal_abnormal_boundary": True,
            "abnormal_hi_arrival_only_switch": True,
            "same_batch_lo_classification": True,
            "hi_mode_persists_until_idle": True,
            "idle_recovery_iff_quiescent": True,
            "entry_mode_boundary_identified": True,
            "delegate_type": type(self.delegate).__qualname__,
        }


def _budget_metadata(bundle: Any, floor_ratio: float) -> dict[str, dict[str, int]]:
    meta_by_name = {str(row.name): row for row in bundle.metadata["task_meta"]}
    result: dict[str, dict[str, int]] = {}
    for task in bundle.ordered_tasks:
        row = meta_by_name[str(task.name)]
        initial = int(row.initial_budget)
        floor = max(1, math.ceil(initial * float(floor_ratio)))
        cap = int(row.base_c_hi if row.criticality.value == "HI" else row.base_c_lo)
        result[str(task.name)] = {
            "initial_runtime_budget": initial,
            "budget_floor": floor,
            "budget_cap": cap,
        }
    return result


def build_target(
    *,
    seed: int,
    workload_args: Mapping[str, Any],
    runtime_args: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    expected_feature_names: list[str] | tuple[str, ...],
    expected_action_definitions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    original_reward_mode: str | None = None,
    formal_reward_mode: str = "mendes",
) -> FormalTarget:
    config = build_mc_fairgen_experiment_config(**dict(workload_args))
    bundle = resolve_experiment_bundle(config, int(seed))
    semantics = RuntimeSemantics(str(runtime_args["runtime_semantics"]))
    environment = build_env_from_experiment_config(
        config,
        seed=int(seed),
        end_time=int(runtime_args["end_time"]),
        agent_period=int(runtime_args["agent_period"]),
        semantics=semantics,
        reward_mode=str(formal_reward_mode),
        action_space=str(runtime_args["action_space"]),
        budget_increase_ratio=float(runtime_args["budget_increase_ratio"]),
        budget_decrease_ratio=float(runtime_args["budget_decrease_ratio"]),
        include_explicit_noop=bool(runtime_args.get("include_explicit_noop", False)),
        budget_floor_ratio=float(runtime_args["budget_floor_ratio"]),
        forbid_decreasing_hi_budgets=bool(runtime_args["forbid_decreasing_hi_budgets"]),
        mask_detail_mode=str(runtime_args["mask_detail_mode"]),
        enable_deploy_cap_mask=bool(runtime_args["enable_deploy_cap_mask"]),
        deploy_cap_mask_ratio=float(runtime_args["deploy_cap_mask_ratio"]),
        deploy_cap_mask_criticality=str(runtime_args["deploy_cap_mask_criticality"]),
        capture_trace=bool(runtime_args.get("capture_trace", True)),
        capture_debug_events=bool(runtime_args.get("capture_debug_events", False)),
        c_amc_sem_xf=float(runtime_args["c_amc_sem_xf"]),
        feature_config=FeatureConfig(**dict(feature_config)),
    )
    feature_names = tuple(environment.get_observation_feature_names())
    action_definitions = tuple(environment.get_action_definitions())
    if feature_names != tuple(str(value) for value in expected_feature_names):
        raise ValueError("REAL_TARGET_FEATURE_SCHEMA_MISMATCH")
    if list(action_definitions) != [dict(value) for value in expected_action_definitions]:
        raise ValueError("REAL_TARGET_ACTION_SCHEMA_MISMATCH")
    if tuple(task.name for task in environment.ordered_tasks) != tuple(task.name for task in bundle.ordered_tasks):
        raise ValueError("REAL_TARGET_TASK_ORDER_MISMATCH")

    runtime_values = {
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
        "processor_overhead": int(runtime_args.get("processor_overhead", 0)),
        "agent_period": int(environment.agent_period),
        "action_space": str(environment.action_space),
        "budget_increase_ratio": float(environment.budget_increase_ratio),
        "budget_decrease_ratio": float(environment.budget_decrease_ratio),
        "budget_floor_ratio": float(environment.budget_floor_ratio),
        "forbid_decreasing_hi_budgets": bool(environment.forbid_decreasing_hi_budgets),
        "mask_detail_mode": str(environment.mask_detail_mode),
        "enable_deploy_cap_mask": bool(environment.enable_deploy_cap_mask),
        "deploy_cap_mask_ratio": float(environment.deploy_cap_mask_ratio),
        "deploy_cap_mask_criticality": str(environment.deploy_cap_mask_criticality),
        "observation_mode": str(environment.feature_config.observation_mode),
    }
    runtime_view = SimpleNamespace(**runtime_values)
    visible_names = (
        "agent_period", "action_space", "budget_increase_ratio", "budget_decrease_ratio",
        "budget_floor_ratio", "forbid_decreasing_hi_budgets", "mask_detail_mode",
        "enable_deploy_cap_mask", "deploy_cap_mask_ratio", "deploy_cap_mask_criticality",
    )
    visible = {name: getattr(environment, name) for name in visible_names}
    visible.update({
        "semantics": environment.runtime_config.semantics,
        "drop_lo_jobs_on_hi_switch": environment.runtime_config.drop_lo_jobs_on_hi_switch,
        "c_amc_sem_lo_degradation_ratio": environment.runtime_config.c_amc_sem_lo_degradation_ratio,
        "c_amc_sem_primary_on_switch_time": environment.runtime_config.c_amc_sem_primary_on_switch_time,
        "stop_at_first_miss": environment.runtime_config.stop_at_first_miss,
        "capture_trace": environment.runtime_config.capture_trace,
        "capture_debug_events": environment.runtime_config.capture_debug_events,
        "observation_mode": environment.feature_config.observation_mode,
    })
    budget = _budget_metadata(bundle, float(runtime_args["budget_floor_ratio"]))
    return FormalTarget(
        ordered_tasks=tuple(bundle.ordered_tasks),
        runtime_config=runtime_view,
        environment=SimpleNamespace(**visible),
        policy=None,
        scenario=FormalScenarioContract(bundle.scenario),
        action_definitions=action_definitions,
        feature_names=feature_names,
        provenance={
            "adapter_kind": "REAL_AMC_RUNTIME",
            "taskset_seed": int(bundle.taskset_seed),
            "scenario_seed": int(bundle.scenario_seed),
            "taskset_attempts": int(bundle.taskset_attempts),
            "taskset_fingerprint_short": str(bundle.taskset_fingerprint),
            "budget_by_task": budget,
            "feature_metadata": {"feature_names": list(feature_names)},
            "original_reward_mode": original_reward_mode,
            "formal_reward_mode": formal_reward_mode,
        },
        runtime_adapter=AMCRealRuntimeAdapter(
            environment,
            action_space=tuple(environment._actions),
        ),
    )


__all__ = ["build_target"]
