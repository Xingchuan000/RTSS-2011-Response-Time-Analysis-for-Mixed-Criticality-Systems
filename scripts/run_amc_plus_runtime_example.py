"""AMC+ 与标准 AMC 运行时语义对比示例（阶段 9）。

本脚本展示三组最小案例：
1. AMC_PLUS + LO overrun：应产生 LO cancellation，不产生 mode switch；
2. AMC + LO overrun：应产生 mode switch；
3. AMC_PLUS + dynamic budget：time=0 提升 budget 后可避免原本的 overrun 事件。
"""

from __future__ import annotations

from pathlib import Path
import sys

# 允许直接用 `python scripts/run_amc_plus_runtime_example.py` 运行脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.budget_runtime import BudgetUpdate
from amc_py.models import Criticality, Task
from amc_py.runtime import simulate_ordered_taskset
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_single_lo_overrun_scenario


def _hi(name: str, period: int, c_lo: int, c_hi: int) -> Task:
    """构造 HI 任务。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_hi,
        criticality=Criticality.HI,
    )


def _lo(name: str, period: int, c_lo: int) -> Task:
    """构造 LO 任务。"""

    return Task(
        name=name,
        period=period,
        deadline=period,
        c_lo=c_lo,
        c_hi=c_lo,
        criticality=Criticality.LO,
    )


def _print_case(case: str, semantics: RuntimeSemantics, result) -> None:
    """统一输出单个案例摘要行。"""

    print(
        ",".join(
            [
                case,
                semantics.value,
                f"mode_changes={result.mode_change_count()}",
                f"lo_cancellations={result.lo_job_cancellation_count()}",
                f"recoveries={result.mode_recovery_count()}",
                f"deadline_misses={len(result.deadline_misses)}",
                f"final_mode={result.final_mode.value}",
            ]
        )
    )


def main() -> None:
    """运行并输出阶段 9 要求的三组对比。"""

    print("case,semantics,mode_changes,lo_cancellations,recoveries,deadline_misses,final_mode")

    # 案例 1：AMC+ 下 LO overrun 仅取消该 job。
    tasks_case_1 = [_hi("h", period=10, c_lo=2, c_hi=4), _lo("l", period=10, c_lo=3)]
    scenario_case_1 = make_single_lo_overrun_scenario("l", release_index=0, actual_cost=4)
    result_case_1 = simulate_ordered_taskset(
        tasks_case_1,
        scenario_case_1,
        RuntimeConfig(end_time=12, semantics=RuntimeSemantics.AMC_PLUS),
    )
    _print_case("case1_lo_overrun", RuntimeSemantics.AMC_PLUS, result_case_1)

    # 案例 2：标准 AMC 下同样的 LO overrun 会触发模式切换。
    result_case_2 = simulate_ordered_taskset(
        tasks_case_1,
        scenario_case_1,
        RuntimeConfig(end_time=12, semantics=RuntimeSemantics.AMC),
    )
    _print_case("case2_lo_overrun", RuntimeSemantics.AMC, result_case_2)

    # 案例 3：AMC+ + dynamic budget。
    # 3a. 不更新 budget（l 的默认 budget=3），actual_cost=4 会触发取消。
    tasks_case_3 = [_lo("l", period=10, c_lo=3)]
    scenario_case_3 = make_single_lo_overrun_scenario("l", release_index=0, actual_cost=4)
    result_case_3a = simulate_ordered_taskset(
        tasks_case_3,
        scenario_case_3,
        RuntimeConfig(end_time=8, semantics=RuntimeSemantics.AMC_PLUS),
    )
    _print_case("case3_dynamic_budget_no_update", RuntimeSemantics.AMC_PLUS, result_case_3a)

    # 3b. 在 time=0 提升 l 的 budget 到 4，避免取消。
    result_case_3b = simulate_ordered_taskset(
        tasks_case_3,
        scenario_case_3,
        RuntimeConfig(end_time=8, semantics=RuntimeSemantics.AMC_PLUS),
        budget_updates=[BudgetUpdate(time=0, updates={"l": 4})],
    )
    _print_case("case3_dynamic_budget_updated", RuntimeSemantics.AMC_PLUS, result_case_3b)


if __name__ == "__main__":
    main()
