"""AMC-RA / AMC-RH 运行时语义对比示例。

本脚本严格围绕实现计划中的最小验证案例：
1. 使用同一个小任务集同时运行 AMC_PLUS / AMC_RA / AMC_RH；
2. 观察 response-expiry 触发的 degraded mode 进入；
3. 对比 RA 的 idle recovery 与 RH 的 response-aware recovery 差异；
4. 打印 dropped LO release、JNE 与 TID 等基线统计。
"""

from __future__ import annotations

from pathlib import Path
import sys

# 允许直接用 `python scripts/run_amc_ra_rh_runtime_example.py` 运行脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import compute_runtime_degradation_metrics
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_table_scenario


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


def _print_case(semantics: RuntimeSemantics) -> None:
    """运行单个语义并打印计划要求的关键统计。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _lo("L1", period=2, c_lo=1),
        _hi("H2", period=20, c_lo=5, c_hi=5),
    ]
    scenario = make_table_scenario({("H1", 0): 3}, default_hi="c_lo", default_lo="c_lo")
    result = simulate_ordered_taskset_event_driven(
        ordered_tasks=tasks,
        scenario=scenario,
        config=RuntimeConfig(
            end_time=9,
            semantics=semantics,
            record_dropped_lo_releases=True,
        ),
    )
    degradation = compute_runtime_degradation_metrics(result)
    dropped_lo_jobs = sum(
        1
        for job in result.jobs
        if job.task.criticality is Criticality.LO and job.dropped
    )
    switch_times = [event.switch_time for event in result.mode_switches]
    recovery_times = [event.recovery_time for event in result.mode_recoveries]
    print(
        ",".join(
            [
                semantics.value,
                f"mode_changes={result.mode_change_count()}",
                f"recoveries={result.mode_recovery_count()}",
                f"dropped_lo_jobs={dropped_lo_jobs}",
                f"jne={degradation.jne}",
                f"tid={degradation.tid}",
                f"final_mode={result.final_mode.value}",
                f"mode_switch_times={switch_times}",
                f"mode_recovery_times={recovery_times}",
            ]
        )
    )


def main() -> None:
    """运行 AMC_PLUS / AMC_RA / AMC_RH 的最小对照。"""

    print(
        "semantics,mode_changes,recoveries,dropped_lo_jobs,jne,tid,final_mode,"
        "mode_switch_times,mode_recovery_times"
    )
    _print_case(RuntimeSemantics.AMC_PLUS)
    _print_case(RuntimeSemantics.AMC_RA)
    _print_case(RuntimeSemantics.AMC_RH)


if __name__ == "__main__":
    main()
