"""诊断 residual 系列动作槽位在当前环境中的解析结果。"""

from __future__ import annotations

import argparse

from amc_py.models import Criticality, Task
from amc_py.rl.actions import describe_budget_action
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario


def _build_demo_tasks() -> list[Task]:
    """构造一个小规模演示任务集，便于本地快速验证 residual 槽位解析。"""

    return [
        Task("h", 12, 12, 2, 3, Criticality.HI),
        Task("l1", 14, 14, 2, 2, Criticality.LO),
        # 兼容 residual_anchor_mc_lo_2：诊断脚本任务集中保留固定锚点任务名。
        Task("mc_lo_2", 16, 16, 2, 2, Criticality.LO),
        Task("l3", 20, 20, 3, 3, Criticality.LO),
        Task("l4", 24, 24, 3, 3, Criticality.LO),
    ]


def main() -> None:
    """执行 residual 动作空间诊断并打印摘要表。"""

    parser = argparse.ArgumentParser(description="诊断 residual 动作空间的 mask/解析结果")
    parser.add_argument(
        "--action-space",
        choices=[
            "residual_ranked",
            "residual_safe_ranked",
            "residual_anchor_mc_lo_2",
            "residual_safe_adjust_15a",
        ],
        default="residual_ranked",
        help="要诊断的 residual 动作空间名称",
    )
    parser.add_argument(
        "--forbid-decreasing-hi-budgets",
        action="store_true",
        help="启用后禁止降低 HI 任务预算，用于对齐训练配置。",
    )
    args = parser.parse_args()

    env = AmcBudgetEnv(
        ordered_tasks=_build_demo_tasks(),
        scenario=make_nominal_scenario(),
        runtime_config=RuntimeConfig(end_time=50, semantics=RuntimeSemantics.AMC_PLUS),
        agent_period=10,
        action_space=args.action_space,
        forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
        mask_detail_mode="full",
    )
    env.reset(seed=0)
    mask = env.valid_action_mask()

    actions = env._actions
    details = env._last_mask_details

    transfer_action_count = sum(1 for action in actions if "transfer" in (action.residual_action_type or ""))
    guarded_decrease_action_count = sum(
        1
        for action in actions
        if (action.residual_action_type or "").startswith("safe_decrease_")
    )
    safe_adjust_increase_action_count = sum(
        1
        for action in actions
        if (action.residual_action_type or "").startswith("safe_increase_")
    )

    print(f"action_space={args.action_space}")
    print(f"action_count={len(actions)}")
    print(f"safe_adjust_increase_action_count={safe_adjust_increase_action_count}")
    print(f"guarded_decrease_action_count={guarded_decrease_action_count}")
    print(f"transfer_action_count={transfer_action_count}")
    print(f"noop_count={sum(1 for action in actions if action.is_noop)}")
    print(f"all_action_ids_contiguous={ [a.action_id for a in actions] == list(range(len(actions))) }")
    print()
    print(
        "action_id\taction_name\tmask_valid\treject_reason\tresolved_increase_task\t"
        "resolved_decrease_tasks\tincrease_idx\tdecrease_indices\tsafe_candidate\tsafe_reject_reason"
    )
    for action, row, is_valid in zip(actions, details, mask, strict=True):
        dec_tasks = ",".join(row.get("resolved_decrease_tasks", ()))
        dec_indices = ",".join(str(v) for v in row.get("decrease_indices", ()))
        print(
            f"{action.action_id}\t"
            f"{describe_budget_action(action)}\t"
            f"{is_valid}\t"
            f"{row.get('reject_reason')}\t"
            f"{row.get('resolved_increase_task')}\t"
            f"{dec_tasks}\t"
            f"{row.get('increase_idx')}\t"
            f"{dec_indices}\t"
            f"{row.get('safe_candidate')}\t"
            f"{row.get('safe_reject_reason')}"
        )


if __name__ == "__main__":
    main()
