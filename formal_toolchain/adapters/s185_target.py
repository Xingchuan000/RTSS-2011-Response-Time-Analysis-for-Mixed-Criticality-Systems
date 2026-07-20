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
from amc_py.nonvacuity import resolve_nonvacuity_settings


class FormalScenarioContract:
    """Expose the actual scenario plus the P0 contract consumed by the toolchain."""

    def __init__(self, delegate: Any, *, nonvacuity_settings: Any = None) -> None:
        self.delegate = delegate
        self.name = getattr(delegate, "name", type(delegate).__name__)
        self.nonvacuity_settings = nonvacuity_settings
        self._read_counts: dict[tuple[str, int], int] = {}

    def actual_cost_for(self, task: Any, release_index: int) -> int:
        value = int(self.delegate.actual_cost_for(task, release_index))
        if bool(getattr(self.nonvacuity_settings, "unstable_demand_reads", False)):
            key = (str(task.name), int(release_index))
            count = self._read_counts.get(key, 0)
            self._read_counts[key] = count + 1
            if count % 2 == 1:
                return max(1, value - 1)
        return value

    def export_formal_contract(self) -> dict[str, Any]:
        # These properties are enforced jointly by ExecutionScenario validation,
        # the seeded provider, and the current event-runtime binding.
        settings = self.nonvacuity_settings
        unstable = bool(getattr(settings, "unstable_demand_reads", False))
        bad_recovery = bool(getattr(settings, "recover_without_quiescence", False))
        return {
            "schema_version": "real_seed_scenario_contract_v1",
            "total": True,
            "positive_integer_codomain": True,
            "non_anticipating": not unstable,
            "batch_entry_frozen": not unstable,
            "key_stable_repeated_read": not unstable,
            "projection_order_idempotent": not unstable,
            "hi_upper_bound": True,
            "normal_abnormal_boundary": True,
            "abnormal_hi_arrival_only_switch": True,
            "same_batch_lo_classification": True,
            "hi_mode_persists_until_idle": not bad_recovery,
            "idle_recovery_iff_quiescent": not bad_recovery,
            "entry_mode_boundary_identified": True,
            "nonvacuity_profile": getattr(settings, "profile", "off"),
            "delegate_type": type(self.delegate).__qualname__,
        }


def _budget_metadata(bundle: Any, floor_ratio: float) -> dict[str, dict[str, int]]:
    meta_by_name = {str(row.name): row for row in bundle.metadata["task_meta"]}
    result: dict[str, dict[str, int]] = {}
    for task in bundle.ordered_tasks:
        row = meta_by_name[str(task.name)]
        initial = int(row.initial_budget)
        floor = max(1, math.ceil(initial * float(floor_ratio)))
        action_hard_upper = (
            int(task.c_hi)
            if row.criticality.value == "HI"
            else int(task.deadline)
        )
        result[str(task.name)] = {
            "initial_runtime_budget": initial,
            "budget_floor": floor,
            "action_hard_upper": action_hard_upper,
            "source_base_budget": int(
                row.base_c_hi if row.criticality.value == "HI" else row.base_c_lo
            ),
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
    nonvacuity_profile: str = "off",
    nonvacuity_params: Mapping[str, Any] | None = None,
) -> FormalTarget:
    settings = resolve_nonvacuity_settings(nonvacuity_profile, nonvacuity_params)
    effective_runtime_args = dict(runtime_args)
    if settings.action_ratio_override is not None:
        effective_runtime_args["budget_increase_ratio"] = float(settings.action_ratio_override)
        effective_runtime_args["budget_decrease_ratio"] = float(settings.action_ratio_override)
    config = build_mc_fairgen_experiment_config(**dict(workload_args))
    bundle = resolve_experiment_bundle(config, int(seed))
    semantics = RuntimeSemantics(str(effective_runtime_args["runtime_semantics"]))
    environment = build_env_from_experiment_config(
        config,
        seed=int(seed),
        end_time=int(effective_runtime_args["end_time"]),
        agent_period=int(effective_runtime_args["agent_period"]),
        semantics=semantics,
        reward_mode=str(formal_reward_mode),
        action_space=str(effective_runtime_args["action_space"]),
        budget_increase_ratio=float(effective_runtime_args["budget_increase_ratio"]),
        budget_decrease_ratio=float(effective_runtime_args["budget_decrease_ratio"]),
        include_explicit_noop=bool(effective_runtime_args.get("include_explicit_noop", False)),
        budget_floor_ratio=float(effective_runtime_args["budget_floor_ratio"]),
        forbid_decreasing_hi_budgets=bool(effective_runtime_args["forbid_decreasing_hi_budgets"]),
        policy_selection_semantics=settings.selection_semantics,
        step_guard_semantics=settings.step_guard_semantics,
        nonvacuity_profile=settings.profile,
        nonvacuity_disabled_guards=settings.disabled_guards,
        budget_rounding_mode=settings.rounding_mode,
        min_budget_delta=settings.min_budget_delta,
        nonvacuity_deadline_cleanup_remove=settings.deadline_cleanup_remove,
        nonvacuity_hi_budget_cap_truncate=settings.hi_budget_cap_truncate,
        nonvacuity_arrival_before_deadline=settings.arrival_before_deadline,
        nonvacuity_controller_overhead_ticks=settings.controller_overhead_ticks,
        nonvacuity_recover_without_quiescence=settings.recover_without_quiescence,
        nonvacuity_unstable_demand_reads=settings.unstable_demand_reads,
        mask_detail_mode=str(effective_runtime_args["mask_detail_mode"]),
        enable_deploy_cap_mask=bool(effective_runtime_args["enable_deploy_cap_mask"]),
        deploy_cap_mask_ratio=float(effective_runtime_args["deploy_cap_mask_ratio"]),
        deploy_cap_mask_criticality=str(effective_runtime_args["deploy_cap_mask_criticality"]),
        capture_trace=bool(effective_runtime_args.get("capture_trace", True)),
        capture_debug_events=bool(effective_runtime_args.get("capture_debug_events", False)),
        c_amc_sem_xf=float(effective_runtime_args["c_amc_sem_xf"]),
        feature_config=FeatureConfig(**dict(feature_config)),
    )
    feature_names = tuple(environment.get_observation_feature_names())
    action_definitions = tuple(environment.get_action_definitions())
    expected_actions_effective = [dict(value) for value in expected_action_definitions]
    if settings.action_ratio_override is not None:
        for row in expected_actions_effective:
            row["increase_ratio"] = float(settings.action_ratio_override)
            row["decrease_ratio"] = float(settings.action_ratio_override)
    if feature_names != tuple(str(value) for value in expected_feature_names):
        raise ValueError("REAL_TARGET_FEATURE_SCHEMA_MISMATCH")
    if list(action_definitions) != expected_actions_effective:
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
        "processor_overhead": (
            int(settings.controller_overhead_ticks)
            if settings.enabled else int(effective_runtime_args.get("processor_overhead", 0))
        ),
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
        "check_safety": bool(environment.check_safety),
        "observation_mode": str(environment.feature_config.observation_mode),
    }
    if settings.enabled:
        runtime_values.update({
            "policy_selection_semantics": str(environment.policy_selection_semantics),
            "step_guard_semantics": str(environment.step_guard_semantics),
            "nonvacuity_profile": settings.profile,
            "nonvacuity_enabled": True,
            "nonvacuity_disabled_guards": list(settings.disabled_guards),
            "budget_rounding_mode": settings.rounding_mode,
            "min_budget_delta": int(settings.min_budget_delta),
            "nonvacuity_deadline_cleanup_remove": settings.deadline_cleanup_remove,
            "nonvacuity_hi_budget_cap_truncate": settings.hi_budget_cap_truncate,
            "nonvacuity_arrival_before_deadline": settings.arrival_before_deadline,
            "nonvacuity_recover_without_quiescence": settings.recover_without_quiescence,
            "nonvacuity_unstable_demand_reads": settings.unstable_demand_reads,
        })
    runtime_view = SimpleNamespace(**runtime_values)
    visible_names = (
        "agent_period", "action_space", "budget_increase_ratio", "budget_decrease_ratio",
        "budget_floor_ratio", "forbid_decreasing_hi_budgets", "mask_detail_mode",
        "enable_deploy_cap_mask", "deploy_cap_mask_ratio", "deploy_cap_mask_criticality",
        "check_safety",
    )
    if settings.enabled:
        visible_names = visible_names + (
            "policy_selection_semantics", "step_guard_semantics",
            "nonvacuity_profile", "nonvacuity_disabled_guards",
            "budget_rounding_mode", "min_budget_delta",
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
    budget = _budget_metadata(bundle, float(effective_runtime_args["budget_floor_ratio"]))
    return FormalTarget(
        ordered_tasks=tuple(bundle.ordered_tasks),
        runtime_config=runtime_view,
        environment=SimpleNamespace(**visible),
        policy=None,
        scenario=FormalScenarioContract(bundle.scenario, nonvacuity_settings=settings),
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
            "nonvacuity": settings.to_dict(),
        },
        runtime_adapter=AMCRealRuntimeAdapter(
            environment,
            action_space=tuple(environment._actions),
            selection_semantics=str(environment.policy_selection_semantics),
            step_guard_semantics=str(environment.step_guard_semantics),
            disabled_guards=tuple(environment.nonvacuity_disabled_guards),
            rounding_mode=str(environment.budget_rounding_mode),
            min_budget_delta=int(environment.min_budget_delta),
        ),
    )


__all__ = ["build_target"]
