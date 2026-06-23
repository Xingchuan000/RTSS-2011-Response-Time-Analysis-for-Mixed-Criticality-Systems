"""AMC-RA / AMC-RH 事件驱动语义测试。"""

from __future__ import annotations

import pytest

from amc_py.amc import build_design_r_lo_map
from amc_py.event_runtime import simulate_ordered_taskset_event_driven, simulate_taskset_with_policy_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario, make_table_scenario


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


@pytest.mark.parametrize("semantics", [RuntimeSemantics.AMC_RA, RuntimeSemantics.AMC_RH])
def test_ra_rh_nominal_case_has_no_degraded_mode(semantics: RuntimeSemantics) -> None:
    """标称场景下，RA/RH 都不应进入 degraded mode。"""

    tasks = [_hi("h", period=10, c_lo=2, c_hi=4), _lo("l", period=10, c_lo=2)]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_nominal_scenario(),
        config=RuntimeConfig(end_time=20, semantics=semantics),
    )
    assert result.mode_change_count() == 0
    assert result.mode_recovery_count() == 0
    assert len(result.deadline_misses) == 0


def test_ra_rh_switches_due_to_response_expiry_instead_of_hi_budget_overrun() -> None:
    """RA/RH 的切换原因必须是 response expiry。"""

    hi = _hi("h", period=10, c_lo=1, c_hi=3)
    lo = _lo("l", period=10, c_lo=1)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("h", 0): 3}),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_RA),
    )
    assert result.mode_change_count() == 1
    assert result.mode_switches[0].reason == "hi_response_time_expiry"
    assert result.mode_switches[0].switch_time == 1


def test_hi_job_completion_at_expiry_does_not_switch_to_hi() -> None:
    """若 HI job 恰好在 expiry 时完成，JOB_COMPLETION 必须先于 expiry 处理。"""

    hi = _hi("h", period=10, c_lo=2, c_hi=2)
    lo = _lo("l", period=10, c_lo=1)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_nominal_scenario(),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_RA),
    )
    assert result.mode_change_count() == 0


def test_ra_and_rh_have_different_recovery_conditions() -> None:
    """RA 等待 idle，RH 在 HI completion 后若无 expired active HI job 可提前恢复。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _lo("L1", period=2, c_lo=1),
        _hi("H2", period=20, c_lo=5, c_hi=5),
    ]
    scenario = make_table_scenario({("H1", 0): 3}, default_hi="c_lo", default_lo="c_lo")
    rh_result = simulate_ordered_taskset_event_driven(
        tasks,
        scenario,
        config=RuntimeConfig(
            end_time=9,
            semantics=RuntimeSemantics.AMC_RH,
            record_dropped_lo_releases=True,
        ),
    )
    ra_result = simulate_ordered_taskset_event_driven(
        tasks,
        scenario,
        config=RuntimeConfig(
            end_time=9,
            semantics=RuntimeSemantics.AMC_RA,
            record_dropped_lo_releases=True,
        ),
    )
    assert rh_result.mode_recoveries[0].recovery_time == 3
    assert rh_result.mode_recoveries[0].reason == "rh_no_expired_hi_job"
    assert ra_result.mode_recoveries[0].recovery_time > 3
    assert ra_result.mode_recoveries[0].reason == "idle"


def test_ra_rh_can_record_dropped_lo_releases_in_degraded_mode() -> None:
    """打开记录开关后，被 degraded mode 直接 dropped 的 LO release 应进入结果。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _lo("L1", period=2, c_lo=1),
        _hi("H2", period=20, c_lo=5, c_hi=5),
    ]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario({("H1", 0): 3}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(
            end_time=8,
            semantics=RuntimeSemantics.AMC_RA,
            record_dropped_lo_releases=True,
        ),
    )
    lo_dropped = [
        job for job in result.jobs
        if job.task.criticality is Criticality.LO and job.dropped
    ]
    assert len(lo_dropped) > 0
    assert all(job.executed_time == 0 for job in lo_dropped if job.drop_time == job.release_time)


def test_mode_switch_active_lo_drops_contribute_to_jne() -> None:
    """进入 degraded mode 时被直接丢弃的 active LO job 应计入 JNE。"""

    from amc_py.metrics import compute_runtime_degradation_metrics

    hi = _hi("H", period=10, c_lo=1, c_hi=3)
    lo = _lo("L", period=10, c_lo=2)
    result = simulate_ordered_taskset_event_driven(
        [hi, lo],
        make_table_scenario({("H", 0): 3}, default_hi="c_lo", default_lo="c_lo"),
        config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_RA),
    )
    assert any(job.task.name == "L" and job.dropped for job in result.jobs)
    assert compute_runtime_degradation_metrics(result).jne >= 1


def test_simultaneous_arrivals_inherit_busy_period_start_by_priority() -> None:
    """同刻释放时，低优先级 HI job 应继承高优先级 busy period 起点。"""

    tasks = [
        _hi("H1", period=20, c_lo=1, c_hi=3),
        _hi("H2", period=20, c_lo=5, c_hi=5),
        _lo("L1", period=20, c_lo=1),
    ]
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_nominal_scenario(),
        config=RuntimeConfig(end_time=2, semantics=RuntimeSemantics.AMC_RA),
    )
    design_r_lo = build_design_r_lo_map(tasks)
    hi_job = result.jobs_of("H2")[0]
    assert hi_job.busy_period_start == 0
    assert hi_job.response_time_expiry == design_r_lo["H2"]


def test_amc_max_is_rejected_for_ra_rh_bridge() -> None:
    """RA/RH 只能和 amc_rtb 分析口径搭配。"""

    tasks = [_hi("h", period=10, c_lo=1, c_hi=3), _lo("l", period=10, c_lo=1)]
    with pytest.raises(ValueError, match="must be used with method='amc_rtb'"):
        simulate_taskset_with_policy_event_driven(
            tasks=tasks,
            method="amc_max",
            priority_policy="dm",
            scenario=make_nominal_scenario(),
            config=RuntimeConfig(end_time=10, semantics=RuntimeSemantics.AMC_RH),
        )
