"""AMC Python reproduction package.

当前对外 API 包含两层能力：
1. 核心数据模型（Task / Criticality / SchedulabilityResult 等）；
2. 运行时仿真 API（scenario、runtime config、集成入口等）。
"""

from .models import (
    Criticality,
    PriorityAssignmentResult,
    SchedulabilityResult,
    Task,
    TaskSet,
)
from .budget_runtime import BudgetState, BudgetUpdate
from .runtime import (
    compare_static_and_runtime,
    simulate_ordered_taskset,
    simulate_taskset_with_policy,
)
from .runtime_models import Job, RuntimeConfig, RuntimeSemantics, SimulationResult, SystemMode
from .runtime_scenarios import (
    ExecutionScenario,
    make_all_hi_jobs_hi_budget_scenario,
    make_nominal_scenario,
    make_single_hi_overrun_scenario,
    make_single_lo_overrun_scenario,
    make_table_scenario,
)

__all__ = [
    # --- 基础数据模型 ---
    "Criticality",
    "Task",
    "TaskSet",
    "SchedulabilityResult",
    "PriorityAssignmentResult",
    "BudgetState",
    "BudgetUpdate",
    # --- runtime 数据模型 ---
    "RuntimeConfig",
    "RuntimeSemantics",
    "SystemMode",
    "Job",
    "SimulationResult",
    # --- runtime scenario ---
    "ExecutionScenario",
    "make_nominal_scenario",
    "make_single_hi_overrun_scenario",
    "make_single_lo_overrun_scenario",
    "make_all_hi_jobs_hi_budget_scenario",
    "make_table_scenario",
    # --- runtime 仿真与集成入口 ---
    "simulate_ordered_taskset",
    "simulate_taskset_with_policy",
    "compare_static_and_runtime",
]
