"""RuntimeMonitor 单元测试。"""

from __future__ import annotations

from amc_py.rl.monitor import RuntimeMonitor


def test_consume_reward_returns_and_clears_accumulator() -> None:
    """consume_reward 应返回累计值并清零。"""

    monitor = RuntimeMonitor()
    monitor.record_job_start("t1")
    monitor.record_lo_budget_overrun("t1", 3)
    reward = monitor.consume_reward()
    assert reward == -1.5
    assert monitor.consume_reward() == 0.0


def test_completion_updates_recent_execution() -> None:
    """job 完成时应记录最近执行量。"""

    monitor = RuntimeMonitor()
    monitor.ensure_tasks(["t1"])
    monitor.record_job_completion("t1", 4)
    assert monitor.recent_execution["t1"] == 4


def test_overrun_penalties_are_distinct() -> None:
    """LO/HI overrun 奖励惩罚应符合定义。"""

    monitor = RuntimeMonitor()
    monitor.record_lo_budget_overrun("lo", 2)
    monitor.record_hi_budget_overrun("hi", 5)
    assert monitor.lo_overrun_count == 1
    assert monitor.hi_overrun_count == 1
    assert monitor.consume_reward() == -3.5
