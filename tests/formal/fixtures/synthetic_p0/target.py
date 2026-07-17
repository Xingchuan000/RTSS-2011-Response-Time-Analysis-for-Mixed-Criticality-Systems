"""完全自包含的 synthetic P0 target factory。

该文件只提供测试夹具，不从 ``examples`` 或真实 Seed 导入对象，保证 fixture
在 editable install 和源码目录两种方式下具有相同的 canonical target。
"""

from types import SimpleNamespace

from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeSemantics
from formal_toolchain.adapters.synthetic_runtime_adapter import SyntheticP0RuntimeAdapter
from formal_toolchain.adapters.target_factory import FormalTarget


class SyntheticEnvironment(SimpleNamespace):
    """显式保存 wrapper 最终可观察配置。"""


def build_target(**_kwargs):
    tasks = (Task("SYN_HI", 10, 10, 2, 3, Criticality.HI),
             Task("SYN_LO", 15, 15, 2, 2, Criticality.LO))
    per_task = ("budget_norm", "recent_cost_norm", "ema_cost_norm", "max_cost_k_norm",
                "overrun_ema", "risk", "surplus", "criticality", "priority_norm", "util_budget")
    feature_names = tuple(f"T{slot:02d}.{task.name}.{name}" for slot, task in enumerate(tasks) for name in per_task)
    feature_names += tuple(f"G.{name}" for name in ("total_budget_util", "hi_budget_util", "lo_budget_util",
        "recent_mode_change_rate", "recent_lo_cancel_rate", "recent_hi_overrun_rate",
        "recent_lo_overrun_rate", "safety_margin_min"))
    action_definitions = (
        {"action_id": 0, "target_task": "SYN_HI", "direction": "increase", "is_noop": False},
        {"action_id": 1, "target_task": "SYN_LO", "direction": "increase", "is_noop": False},
        {"action_id": 2, "target_task": "SYN_HI", "direction": "decrease", "is_noop": False},
        {"action_id": 3, "target_task": "SYN_LO", "direction": "decrease", "is_noop": False},
    )
    config = SimpleNamespace(
        semantics=RuntimeSemantics.C_AMC_SEM, drop_lo_jobs_on_hi_switch=False,
        c_amc_sem_lo_degradation_ratio=0.5, c_amc_sem_primary_on_switch_time=True,
        stop_at_first_miss=False, capture_trace=False, capture_debug_events=False,
        agent_period=10, action_space="single", budget_increase_ratio=0.1,
        budget_decrease_ratio=0.1, budget_floor_ratio=0.5,
        forbid_decreasing_hi_budgets=True, mask_detail_mode="minimal",
        enable_deploy_cap_mask=False, deploy_cap_mask_ratio=1.0,
        deploy_cap_mask_criticality="lo", observation_mode="synthetic_v1",
        processor_overhead=0,
    )
    visible = ("semantics", "drop_lo_jobs_on_hi_switch", "c_amc_sem_lo_degradation_ratio",
        "c_amc_sem_primary_on_switch_time", "stop_at_first_miss", "capture_trace",
        "capture_debug_events", "agent_period", "action_space", "budget_increase_ratio",
        "budget_decrease_ratio", "budget_floor_ratio", "forbid_decreasing_hi_budgets",
        "mask_detail_mode", "enable_deploy_cap_mask", "deploy_cap_mask_ratio",
        "deploy_cap_mask_criticality", "observation_mode")
    environment = SyntheticEnvironment(**{name: getattr(config, name) for name in visible})
    target = FormalTarget(
        ordered_tasks=tasks, runtime_config=config, environment=environment,
        policy=SimpleNamespace(name="synthetic_integer_tree"), scenario=SimpleNamespace(name="synthetic_p0"),
        action_definitions=action_definitions, feature_names=feature_names,
        provenance={"fixture": "synthetic_p0_v1", "taskset_seed": None,
                    "budget_by_task": {
                        "SYN_HI": {
                            "initial_runtime_budget": 2,
                            "budget_floor": 1,
                            "action_hard_upper": 3,
                            "source_base_budget": 2,
                        },
                        "SYN_LO": {
                            "initial_runtime_budget": 2,
                            "budget_floor": 1,
                            "action_hard_upper": 15,
                            "source_base_budget": 1,
                        },
                    }})
    object.__setattr__(target, "runtime_adapter", SyntheticP0RuntimeAdapter(target))
    return target


__all__ = ["build_target"]
