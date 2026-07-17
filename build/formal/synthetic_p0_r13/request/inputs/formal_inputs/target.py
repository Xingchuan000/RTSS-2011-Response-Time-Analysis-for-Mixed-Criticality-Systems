"""canonical synthetic P0 的唯一 target factory。

该 fixture 使用三个严格 priority 的独立任务：最高优先级 HI task 形成
W=0 分支，最低优先级 HI task 在 LO task 干扰下形成 W>0 分支。所有参数
都直接写入 fixture 的 canonical input，不从默认配置推断。
"""

from types import SimpleNamespace

from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeSemantics
from formal_toolchain.adapters.target_factory import FormalTarget


def build_target(**_kwargs):
    tasks = (
        Task("SYN_HI_0", 20, 20, 1, 2, Criticality.HI),
        Task("SYN_LO", 30, 2, 1, 1, Criticality.LO),
        Task("SYN_HI_1", 50, 50, 2, 3, Criticality.HI),
    )
    per_task = ("budget_norm", "recent_cost_norm", "ema_cost_norm", "max_cost_k_norm",
                "overrun_ema", "risk", "surplus", "criticality", "priority_norm", "util_budget")
    feature_names = tuple(
        f"T{slot:02d}.{task.name}.{name}"
        for slot, task in enumerate(tasks)
        for name in per_task
    ) + tuple(f"G.{name}" for name in (
        "total_budget_util", "hi_budget_util", "lo_budget_util", "recent_mode_change_rate",
        "recent_lo_cancel_rate", "recent_hi_overrun_rate", "recent_lo_overrun_rate",
        "safety_margin_min"))
    action_definitions = (
        {"action_id": 0, "target_task": "SYN_HI_0", "direction": "increase", "is_noop": False},
        {"action_id": 1, "target_task": "SYN_LO", "direction": "increase", "is_noop": False},
        {"action_id": 2, "target_task": "SYN_HI_1", "direction": "increase", "is_noop": False},
        {"action_id": 3, "target_task": "SYN_HI_0", "direction": "decrease", "is_noop": False},
        {"action_id": 4, "target_task": "SYN_LO", "direction": "decrease", "is_noop": False},
        {"action_id": 5, "target_task": "SYN_HI_1", "direction": "decrease", "is_noop": False},
    )
    config = SimpleNamespace(
        semantics=RuntimeSemantics.C_AMC_SEM,
        drop_lo_jobs_on_hi_switch=False,
        c_amc_sem_lo_degradation_ratio=0.5,
        c_amc_sem_primary_on_switch_time=True,
        stop_at_first_miss=False,
        capture_trace=False,
        capture_debug_events=False,
        agent_period=10,
        action_space="single",
        budget_increase_ratio=0.1,
        budget_decrease_ratio=0.1,
        budget_floor_ratio=0.5,
        forbid_decreasing_hi_budgets=True,
        mask_detail_mode="minimal",
        enable_deploy_cap_mask=False,
        deploy_cap_mask_ratio=1.0,
        deploy_cap_mask_criticality="lo",
        observation_mode="synthetic_v1",
        processor_overhead=0,
    )
    visible = (
        "semantics", "drop_lo_jobs_on_hi_switch", "c_amc_sem_lo_degradation_ratio",
        "c_amc_sem_primary_on_switch_time", "stop_at_first_miss", "capture_trace",
        "capture_debug_events", "agent_period", "action_space", "budget_increase_ratio",
        "budget_decrease_ratio", "budget_floor_ratio", "forbid_decreasing_hi_budgets",
        "mask_detail_mode", "enable_deploy_cap_mask", "deploy_cap_mask_ratio",
        "deploy_cap_mask_criticality", "observation_mode", "processor_overhead",
    )
    environment = SimpleNamespace(**{name: getattr(config, name) for name in visible})
    # synthetic fixture 的 scenario 直接提供正式合同，供 candidate/compiler
    # 与 fresh verifier 在同一份输入上消费。这里不做任何动态推断，只把
    # 这组合成场景固定为计划要求的 P0 合同。
    def export_formal_contract():
        return {
            "abnormal_hi_arrival_only_switch": True,
            "same_batch_lo_classification": True,
            "hi_mode_persists_until_idle": True,
            "idle_recovery_iff_quiescent": True,
            "entry_mode_boundary_identified": True,
            "total": True,
            "positive_integer_codomain": True,
            "non_anticipating": True,
            "batch_entry_frozen": True,
            "key_stable_repeated_read": True,
            "projection_order_idempotent": True,
            "hi_upper_bound": True,
            "normal_abnormal_boundary": True,
        }
    return FormalTarget(
        ordered_tasks=tasks,
        runtime_config=config,
        environment=environment,
        policy=SimpleNamespace(name="synthetic_p0_integer_tree"),
        scenario=SimpleNamespace(name="synthetic_p0", export_formal_contract=export_formal_contract),
        action_definitions=action_definitions,
        feature_names=feature_names,
        provenance={
            "fixture": "synthetic_p0",
            "taskset_seed": None,
            "budget_by_task": {
                "SYN_HI_0": {"initial_runtime_budget": 1, "budget_floor": 1, "budget_cap": 2},
                "SYN_LO": {"initial_runtime_budget": 2, "budget_floor": 2, "budget_cap": 2},
                "SYN_HI_1": {"initial_runtime_budget": 2, "budget_floor": 2, "budget_cap": 3},
            },
        },
    )


__all__ = ["build_target"]
