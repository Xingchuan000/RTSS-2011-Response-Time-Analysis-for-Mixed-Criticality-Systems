"""单动作空间 full recovery reward 的回归测试。"""

from __future__ import annotations

import json
import math

from amc_py.dqn.experiment import build_env_from_experiment_config, build_small_nominal_experiment_config
from amc_py.rl.reward_config import evaluate_reward_expression, load_reward_mode_config


def _pick_valid_action(env) -> int | None:
    """选择一个可执行动作，优先挑非 noop，便于覆盖真实预算动作路径。"""

    mask = env.valid_action_mask()
    for action_id, is_valid in enumerate(mask):
        if is_valid and not bool(env._actions[action_id].is_noop):  # noqa: SLF001
            return action_id
    for action_id, is_valid in enumerate(mask):
        if is_valid:
            return action_id
    return None


def _dummy_recovery_reward_variables() -> dict[str, float | bool | str]:
    """构造 single recovery reward 表达式求值所需的完整 dummy 变量表。"""

    return {
        "paper_reward": 0.0,
        "noop_bonus_if_noop": 0.0,
        "budget_change_penalty": 0.0,
        "budget_change_norm": 0.0,
        "budget_drift_penalty": 0.0,
        "budget_drift_mean": 0.0,
        "budget_under_drift_mean": 0.0,
        "budget_over_drift_mean": 0.0,
        "budget_over_drift_deadzone_mean": 0.0,
        "budget_abs_drift_mean": 0.0,
        "budget_abs_drift_deadzone": 0.05,
        "budget_abs_drift_deadzone_mean": 0.0,
        "budget_abs_drift_penalty": 0.0,
        "budget_abs_drift_penalty_value": 0.0,
        "over_budget_dwell_penalty": 0.0,
        "over_increase_penalty": 0.0,
        "over_increase_deadzone": 0.05,
        "over_increase_excess": 0.0,
        "is_over_increase_action": 0.0,
        "budget_soft_cap_ratio": 0.0,
        "budget_soft_cap_penalty": 0.0,
        "budget_soft_cap_increase_excess": 0.0,
        "budget_soft_cap_penalty_value": 0.0,
        "is_soft_cap_increase_action": 0.0,
        "safe_recovery_decrease": 0.0,
        "recovery_decrease_target_count": 0.0,
        "recovery_decrease_excess_before_mean": 0.0,
        "unsafe_decrease_penalty": 0.0,
        "unsafe_decrease_full": 0.0,
        "pingpong_penalty": 0.0,
        "pingpong_action": 0.0,
        "concentration_penalty": 0.0,
        "concentration_window": 3.0,
        "increase_concentration_excess": 0.0,
        "consecutive_increase_count_for_target": 0.0,
        "lo_pressure_mean": 0.0,
        "lo_pressure_max": 0.0,
        "lo_near_cancel_rate": 0.0,
        "hi_mode_pressure_mean": 0.0,
        "lo_pressure_penalty": 0.0,
        "lo_pressure_max_penalty": 0.0,
        "lo_near_cancel_penalty": 0.0,
        "hi_mode_pressure_penalty": 0.0,
        "lo_pressure_threshold": 0.8,
        "lo_near_cancel_threshold": 0.9,
        "hi_mode_pressure_threshold": 0.8,
        "lo_pressure_penalty_value": 0.0,
        "lo_pressure_max_penalty_value": 0.0,
        "lo_near_cancel_penalty_value": 0.0,
        "hi_mode_pressure_penalty_value": 0.0,
        "is_explicit_noop_action": False,
        "event_job_start_reward": 0.0,
        "event_lo_overrun_reward": 0.0,
        "event_hi_overrun_reward": 0.0,
        "delta_job_start": 0.0,
        "delta_lo_overrun": 0.0,
        "delta_hi_overrun": 0.0,
        "delta_mode_changes": 0.0,
        "delta_lo_cancellations": 0.0,
        "delta_deadline_misses": 0.0,
        "interval_time": 1.0,
        "delta_total_jobs": 1.0,
        "lo_overrun_rate": 0.0,
        "hi_overrun_rate": 0.0,
        "mode_change_rate": 0.0,
        "mode_change_per_job": 0.0,
        "lo_cancellation_rate": 0.0,
        "deadline_miss_rate": 0.0,
        "invalid_action": 0.0,
        "is_budget_action": 1.0,
        "is_increase_action": 0.0,
        "is_decrease_action": 0.0,
        "is_transfer_action": 0.0,
        "decrease_hits_hi": 0.0,
        "decrease_hits_lo": 0.0,
        "decrease_task_count": 0.0,
        "unsafe_decrease": 0.0,
        "current_budget_action_direction": "noop",
        "current_budget_action_task": "",
        "last_budget_action_direction": "",
        "last_budget_action_task": "",
    }


def test_single_recovery_reward_mode_loads_and_evaluates() -> None:
    """新 reward mode 必须可加载，并且表达式可在 dummy 变量表上求值。"""

    config = load_reward_mode_config("interval_qos_v2_single_recovery_full")
    variables = _dummy_recovery_reward_variables()
    variables.update(config.reward_parameters)
    reward = evaluate_reward_expression(config.step_reward_formula, variables)
    assert math.isfinite(reward)


def test_single_recovery_reward_env_step_exposes_new_fields() -> None:
    """新 reward mode 在 single 动作空间下应可正常 reset/step，并输出新增 info 字段。"""

    env = build_env_from_experiment_config(
        build_small_nominal_experiment_config(),
        seed=0,
        end_time=50,
        agent_period=10,
        reward_mode="interval_qos_v2_single_recovery_full",
        action_space="single",
        include_explicit_noop=True,
    )
    obs = env.reset(seed=0)
    assert len(obs.state_vector) > 0

    action_id = _pick_valid_action(env)
    result = env.step(action_id)

    assert math.isfinite(float(result.reward))
    for key in (
        "budget_over_drift_deadzone_mean",
        "over_increase_deadzone",
        "over_increase_excess",
        "is_over_increase_action",
        "budget_soft_cap_ratio",
        "budget_soft_cap_penalty",
        "budget_soft_cap_increase_excess",
        "budget_soft_cap_penalty_value",
        "is_soft_cap_increase_action",
        "safe_recovery_decrease",
        "recovery_decrease_target_count",
        "recovery_decrease_excess_before_mean",
        "unsafe_decrease_full",
        "pingpong_action",
        "increase_concentration_excess",
        "consecutive_increase_count_for_target",
        "final_budget_ratio_by_task_json",
        "max_budget_ratio_by_task_json",
        "min_budget_ratio_by_task_json",
        "increase_count_by_task_json",
        "decrease_count_by_task_json",
        "recovery_decrease_count_by_task_json",
        "over_increase_count_by_task_json",
        "consecutive_increase_max_by_task_json",
        "over_budget_dwell_steps_by_task_json",
    ):
        assert key in result.info

    final_ratios = json.loads(str(result.info["final_budget_ratio_by_task_json"]))
    max_ratios = json.loads(str(result.info["max_budget_ratio_by_task_json"]))
    min_ratios = json.loads(str(result.info["min_budget_ratio_by_task_json"]))
    for task_name, final_ratio in final_ratios.items():
        assert float(min_ratios[task_name]) <= float(final_ratio) + 1e-9
        assert float(final_ratio) <= float(max_ratios[task_name]) + 1e-9


def test_soft_cap_reward_expression_penalizes_cap_excess() -> None:
    """soft cap reward 配置必须能使用新增变量完成表达式求值。"""

    config = load_reward_mode_config(
        "interval_qos_v2_single_recovery_full_C5_overinc016_abs005_softcap3_p002"
    )
    variables = _dummy_recovery_reward_variables()
    variables.update(
        {
            "budget_soft_cap_ratio": 3.0,
            "budget_soft_cap_penalty": 0.002,
            "budget_soft_cap_increase_excess": 1.5,
            "is_soft_cap_increase_action": 1.0,
        }
    )
    variables.update(config.reward_parameters)
    reward = evaluate_reward_expression(config.step_reward_formula, variables)
    assert math.isfinite(reward)
