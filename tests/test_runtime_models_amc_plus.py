"""AMC+ 运行时结果模型测试（阶段 1）。"""

from __future__ import annotations

from amc_py.runtime_models import JobCancellationEvent, ModeSwitchEvent, SimulationResult


def test_simulation_result_default_counts_are_zero() -> None:
    """默认结果对象不包含任何模式切换或 LO 取消事件。"""

    result = SimulationResult()
    assert result.mode_change_count() == 0
    assert result.lo_job_cancellation_count() == 0
    assert result.mode_switched() is False
    assert result.mode_switch is None


def test_mode_switches_are_counted_and_compatible_property_kept() -> None:
    """新增 mode_switches 列表后，旧接口 mode_switch 仍可访问第一条事件。"""

    switch = ModeSwitchEvent(
        switch_time=11,
        triggering_task="tau_hi",
        triggering_release_index=0,
        executed_at_switch=4,
        budget_at_switch=3,
    )
    result = SimulationResult(mode_switches=[switch])

    assert result.mode_switched() is True
    assert result.mode_change_count() == 1
    assert result.mode_switch is switch


def test_job_cancellation_events_are_counted() -> None:
    """LO job 取消事件应可累计统计。"""

    cancellation = JobCancellationEvent(
        cancel_time=7,
        task="tau_lo",
        release_index=1,
        executed_at_cancel=4,
        budget_at_cancel=3,
    )
    result = SimulationResult(job_cancellations=[cancellation])
    assert result.lo_job_cancellation_count() == 1

