"""Mask-aware C-AMC-sem dynamic baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from amc_py.dqn import build_env_from_experiment_config
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics, SimulationResult


@dataclass(frozen=True, slots=True)
class DynamicBaselineRun:
    """Per-seed result from the shared mask-aware environment loop."""

    runtime_result: SimulationResult
    total_reward: float
    step_count: int
    selected_action_count: int
    accepted_actions: int
    rejected_actions: int
    noop_actions: int
    explicit_noop_actions: int
    debug_statistics: dict[str, object]
    action_log: list[dict[str, object]]
    mask_log: list[dict[str, object]]


def run_mask_aware_baseline(
    *,
    selector,
    experiment_config,
    seed: int,
    end_time: int,
    agent_period: int,
    reward_mode: str,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    enable_deploy_cap_mask: bool,
    deploy_cap_mask_ratio: float,
    deploy_cap_mask_criticality: str,
    feature_config: FeatureConfig,
    c_amc_sem_xf: float,
) -> DynamicBaselineRun:
    """Run a selector through the same env path used by formal DQN evaluation."""

    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
        end_time=end_time,
        agent_period=agent_period,
        semantics=RuntimeSemantics.C_AMC_SEM,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode=mask_detail_mode,
        enable_deploy_cap_mask=enable_deploy_cap_mask,
        deploy_cap_mask_ratio=deploy_cap_mask_ratio,
        deploy_cap_mask_criticality=deploy_cap_mask_criticality,
        capture_trace=False,
        capture_debug_events=False,
        record_dropped_lo_releases=True,
        c_amc_sem_xf=c_amc_sem_xf,
        feature_config=feature_config,
    )
    obs = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step_count = 0
    selected_action_count = 0
    accepted_actions = 0
    rejected_actions = 0
    noop_actions = 0
    explicit_noop_actions = 0

    while not done:
        step_count += 1
        mask = env.valid_action_mask()
        action_id = selector.select_action_id(
            observation=obs,
            valid_action_mask=mask,
            actions=env.actions,
        )
        selected_action_count += int(action_id is not None)

        result = env.step(action_id)
        total_reward += float(result.reward)

        if bool(result.info.get("is_noop", False)):
            noop_actions += 1
            explicit_noop_actions += int(
                bool(result.info.get("is_explicit_noop_action", False))
            )

        if action_id is not None:
            if bool(result.info.get("accepted", False)):
                accepted_actions += 1
            else:
                rejected_actions += 1

        obs = result.observation
        done = result.done

    runtime_result = env._engine.finish() if env._engine is not None else SimulationResult()
    return DynamicBaselineRun(
        runtime_result=runtime_result,
        total_reward=total_reward,
        step_count=step_count,
        selected_action_count=selected_action_count,
        accepted_actions=accepted_actions,
        rejected_actions=rejected_actions,
        noop_actions=noop_actions,
        explicit_noop_actions=explicit_noop_actions,
        debug_statistics=dict(env.debug_statistics()),
        action_log=[dict(row) for row in env.action_log],
        mask_log=[dict(row) for row in env.mask_log],
    )


def run_random_valid_baseline(*, selector_seed: int, **kwargs) -> DynamicBaselineRun:
    """Run the formal random-valid selector."""

    from amc_py.rl.baseline_policies import RandomValidSelector

    return run_mask_aware_baseline(
        selector=RandomValidSelector(seed=selector_seed),
        **kwargs,
    )


def run_pressure_threshold_baseline(
    *,
    ordered_tasks,
    u_low: float,
    u_high: float,
    **kwargs,
) -> DynamicBaselineRun:
    """Run the formal pressure-threshold selector."""

    from amc_py.rl.baseline_policies import PressureThresholdValidSelector

    selector = PressureThresholdValidSelector(
        ordered_tasks=tuple(ordered_tasks),
        u_low=float(u_low),
        u_high=float(u_high),
    )
    return run_mask_aware_baseline(selector=selector, **kwargs)


def load_pressure_heuristic_selection(path: Path) -> dict[str, object]:
    """Load a frozen pressure-threshold selection JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pressure heuristic selection must be a JSON object")
    return dict(payload)
