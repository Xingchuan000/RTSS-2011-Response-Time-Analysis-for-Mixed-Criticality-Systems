"""不依赖真实 seed 的 P0 合成 target。

夹具故意提供完整的 effective-config 来源链、task budget 元数据、feature/action
顺序和六个微场景所需的最小运行时契约，供 Phase A-E 做契约验收。
"""

from types import SimpleNamespace

from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeSemantics
from formal_toolchain.adapters.target_factory import FormalTarget


class SyntheticEnvironment(SimpleNamespace):
    """显式列出计划要求的 wrapper 最终字段，不依赖隐式默认值。"""


def build_target(**_kwargs):
    tasks = (Task("SYN_HI", 10, 10, 2, 3, Criticality.HI),
             Task("SYN_LO", 15, 15, 2, 2, Criticality.LO))
    feature_names = tuple(
        f"T{slot:02d}.{task.name}.{name}"
        for slot, task in enumerate(tasks)
        for name in (
            "budget_norm", "recent_cost_norm", "ema_cost_norm", "max_cost_k_norm",
            "overrun_ema", "risk", "surplus", "criticality", "priority_norm", "util_budget",
        )
    ) + tuple(f"G.{name}" for name in (
        "total_budget_util", "hi_budget_util", "lo_budget_util",
        "recent_mode_change_rate", "recent_lo_cancel_rate", "recent_hi_overrun_rate",
        "recent_lo_overrun_rate", "safety_margin_min",
    ))
    action_definitions = tuple(
        {"action_id": index, "target_task": task.name, "is_noop": False}
        for index, task in enumerate(tasks)
    ) + tuple({"action_id": index, "target_task": "SYN_HI", "is_noop": False}
              for index in range(2, 24))
    config = SimpleNamespace(
        semantics=RuntimeSemantics.C_AMC_SEM,
        drop_lo_jobs_on_hi_switch=False,
        c_amc_sem_lo_degradation_ratio=0.5,
        c_amc_sem_primary_on_switch_time=True,
        stop_at_first_miss=False, capture_trace=False, capture_debug_events=False,
        agent_period=10, action_space="single", budget_increase_ratio=0.1,
        budget_decrease_ratio=0.1, budget_floor_ratio=0.5,
        forbid_decreasing_hi_budgets=True, mask_detail_mode="minimal",
        enable_deploy_cap_mask=False, deploy_cap_mask_ratio=1.0,
        deploy_cap_mask_criticality="lo", observation_mode="synthetic_v1",
    )
    environment = SyntheticEnvironment(**{name: getattr(config, name) for name in (
        "semantics", "drop_lo_jobs_on_hi_switch", "c_amc_sem_lo_degradation_ratio",
        "c_amc_sem_primary_on_switch_time", "stop_at_first_miss", "capture_trace",
        "capture_debug_events", "agent_period", "action_space", "budget_increase_ratio",
        "budget_decrease_ratio", "budget_floor_ratio", "forbid_decreasing_hi_budgets",
        "mask_detail_mode", "enable_deploy_cap_mask", "deploy_cap_mask_ratio",
        "deploy_cap_mask_criticality", "observation_mode")})
    return FormalTarget(
        ordered_tasks=tasks, runtime_config=config, environment=environment,
        policy=SimpleNamespace(name="synthetic_integer_tree"),
        scenario=SimpleNamespace(name="synthetic_p0"),
        action_definitions=action_definitions, feature_names=feature_names,
        provenance={"fixture": "synthetic_p0_v1", "taskset_seed": None,
                    "budget_by_task": {"SYN_HI": {"initial_runtime_budget": 2, "budget_floor": 1, "budget_cap": 3},
                                       "SYN_LO": {"initial_runtime_budget": 2, "budget_floor": 1, "budget_cap": 2}}},
    )
