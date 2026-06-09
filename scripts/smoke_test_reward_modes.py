"""奖励模式表达式 smoke test。

该脚本用于在不跑完整训练的前提下，快速验证 reward mode 配置是否可被发现、加载、
并且 step_reward_formula 中引用的变量都存在。
"""

from __future__ import annotations

import argparse
import sys

from amc_py.rl.reward_config import (
    available_reward_modes,
    evaluate_reward_expression,
    load_reward_mode_config,
)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="Smoke test reward mode expressions")
    parser.add_argument(
        "--modes",
        type=str,
        default="",
        help="逗号分隔的 reward mode 列表；为空时默认测试全部可用 mode",
    )
    return parser


def build_dummy_variables() -> dict[str, float | bool]:
    """构造与 env.py 同口径的 dummy 变量表。

    这些值不追求业务含义，只用于确保表达式可求值且变量名完整。
    """

    return {
        "paper_reward": 0.0,
        "noop_bonus_if_noop": 0.0,
        "budget_change_penalty": 0.0,
        "budget_change_norm": 0.1,
        "budget_drift_penalty": 0.0,
        "budget_drift_mean": 0.1,
        "budget_under_drift_mean": 0.1,
        "budget_over_drift_mean": 0.0,
        "budget_over_drift_deadzone_mean": 0.0,
        "budget_abs_drift_mean": 0.1,
        "budget_abs_drift_deadzone": 0.05,
        "budget_abs_drift_deadzone_mean": 0.05,
        "budget_abs_drift_penalty": 0.0,
        "budget_abs_drift_penalty_value": 0.0,
        "over_budget_dwell_penalty": 0.0,
        "over_increase_penalty": 0.0,
        "over_increase_deadzone": 0.05,
        "over_increase_excess": 0.0,
        "is_over_increase_action": 0.0,
        # soft upper cap reward 新增变量：
        # 这里只补齐表达式求值所需的 dummy 变量，不改变 smoke test 的执行逻辑。
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
        "lo_pressure_mean": 0.1,
        "lo_pressure_max": 0.2,
        "lo_near_cancel_rate": 0.0,
        "hi_mode_pressure_mean": 0.1,
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
        "delta_job_start": 10.0,
        "delta_lo_overrun": 1.0,
        "delta_hi_overrun": 0.0,
        "delta_mode_changes": 1.0,
        "delta_lo_cancellations": 2.0,
        "delta_deadline_misses": 0.0,
        "interval_time": 50000.0,
        "delta_total_jobs": 10.0,
        "lo_overrun_rate": 0.1,
        "hi_overrun_rate": 0.0,
        "mode_change_rate": 0.00002,
        "mode_change_per_job": 0.1,
        "lo_cancellation_rate": 0.2,
        "deadline_miss_rate": 0.0,
        "invalid_action": 0.0,
        # interval_qos_pareto_v1 新增动作语义变量（与 env.py 同名同口径）。
        "is_budget_action": 1.0,
        "is_increase_action": 0.0,
        "is_decrease_action": 1.0,
        "is_transfer_action": 0.0,
        "decrease_hits_hi": 1.0,
        "decrease_hits_lo": 0.0,
        "decrease_task_count": 1.0,
        "unsafe_decrease": 1.0,
        "current_budget_action_direction": "decrease",
        "current_budget_action_task": "t0",
        "last_budget_action_direction": "increase",
        "last_budget_action_task": "t0",
        "budget_under_drift_mean": 0.1,
        "budget_over_drift_mean": 0.0,
        "budget_over_drift_deadzone_mean": 0.0,
    }


def parse_modes(raw_modes: str) -> list[str]:
    """解析模式列表。"""

    if raw_modes.strip() == "":
        return list(available_reward_modes())
    return [mode.strip() for mode in raw_modes.split(",") if mode.strip()]


def main() -> int:
    """执行 smoke test，失败返回非零退出码。"""

    args = build_parser().parse_args()
    modes = parse_modes(args.modes)
    if not modes:
        print("未找到需要测试的 reward mode", file=sys.stderr)
        return 2

    available = set(available_reward_modes())
    exit_code = 0
    for mode in modes:
        if mode not in available:
            print(f"[FAIL] {mode}: 不在 available_reward_modes() 中", file=sys.stderr)
            exit_code = 1
            continue
        try:
            config = load_reward_mode_config(mode)
            variables = build_dummy_variables()
            # 与 env.py 一致：reward_parameters 需要并入变量表。
            variables.update(config.reward_parameters)
            step_reward = evaluate_reward_expression(config.step_reward_formula, variables)
            print(f"[OK] {mode}: step_reward={step_reward:.6f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {mode}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
