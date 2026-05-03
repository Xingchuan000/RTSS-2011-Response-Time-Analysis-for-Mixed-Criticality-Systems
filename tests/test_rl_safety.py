"""运行时预算安全过滤器测试。"""

from __future__ import annotations

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task
from amc_py.rl.safety import RuntimeBudgetSafetyChecker, merge_budget_candidate


def _task(name: str, period: int, deadline: int, c_lo: int, c_hi: int, crit: Criticality) -> Task:
    """便捷构造任务。"""

    return Task(name=name, period=period, deadline=deadline, c_lo=c_lo, c_hi=c_hi, criticality=crit)


def _tasks() -> list[Task]:
    """构造一组可用于安全检查的任务集。"""

    return [
        _task("h1", 10, 10, 2, 3, Criticality.HI),
        _task("h2", 20, 20, 2, 4, Criticality.HI),
        _task("l1", 25, 25, 2, 2, Criticality.LO),
    ]


def test_valid_candidate_is_accepted() -> None:
    """满足约束的候选预算应被接受。"""

    tasks = _tasks()
    checker = RuntimeBudgetSafetyChecker(tasks, design_r_lo={"h1": 6, "h2": 12}, check_lo_tasks=True)
    report = checker.validate_candidate({"h1": 2, "h2": 2, "l1": 2})
    assert report.accepted is True


def test_missing_budget_is_rejected() -> None:
    """缺少任务预算时应拒绝。"""

    tasks = _tasks()
    checker = RuntimeBudgetSafetyChecker(tasks, design_r_lo={"h1": 6, "h2": 12})
    report = checker.validate_candidate({"h1": 2, "h2": 2})
    assert report.accepted is False
    assert "missing_budget" in report.reason


def test_non_positive_budget_is_rejected() -> None:
    """预算小于等于 0 时应拒绝。"""

    tasks = _tasks()
    checker = RuntimeBudgetSafetyChecker(tasks, design_r_lo={"h1": 6, "h2": 12})
    report = checker.validate_candidate({"h1": 2, "h2": 0, "l1": 2})
    assert report.accepted is False
    assert "non_positive_budget" in report.reason


def test_hi_lo_mode_violation_is_rejected() -> None:
    """违反 HI 的 LO-mode 保守检查时应拒绝。"""

    tasks = _tasks()
    checker = RuntimeBudgetSafetyChecker(tasks, design_r_lo={"h1": 5, "h2": 6})
    report = checker.validate_candidate({"h1": 5, "h2": 5, "l1": 2})
    assert report.accepted is False
    assert "hi_lo_mode_violation" in report.reason


def test_hi_mode_switch_violation_is_rejected() -> None:
    """违反 HI mode-switch 检查时应拒绝。"""

    tasks = [
        _task("h1", 10, 10, 2, 3, Criticality.HI),
        _task("h2", 20, 5, 2, 4, Criticality.HI),
    ]
    checker = RuntimeBudgetSafetyChecker(tasks, design_r_lo={"h1": 7, "h2": 8})
    report = checker.validate_candidate({"h1": 2, "h2": 2})
    assert report.accepted is False
    assert "hi_mode_switch_violation" in report.reason


def test_lo_check_rejects_when_enabled() -> None:
    """开启 LO 检查时，违反 LO 约束应拒绝。"""

    tasks = _tasks()
    checker = RuntimeBudgetSafetyChecker(tasks, design_r_lo={"h1": 6, "h2": 12}, check_lo_tasks=True)
    report = checker.validate_candidate({"h1": 10, "h2": 10, "l1": 2})
    assert report.accepted is False
    assert "lo_mode_violation" in report.reason


def test_lo_check_skipped_when_disabled() -> None:
    """关闭 LO 检查时不应执行 LO 约束拒绝。"""

    tasks = _tasks()
    checker = RuntimeBudgetSafetyChecker(tasks, design_r_lo={"h1": 6, "h2": 12}, check_lo_tasks=False)
    report = checker.validate_candidate({"h1": 2, "h2": 2, "l1": 100})
    assert report.accepted is True


def test_merge_budget_candidate_returns_full_vector() -> None:
    """merge_budget_candidate 应返回完整预算向量。"""

    tasks = _tasks()
    current = BudgetState.from_tasks(tasks)
    merged = merge_budget_candidate(current, {"h1": 3})
    assert merged["h1"] == 3
    assert set(merged.keys()) == {"h1", "h2", "l1"}


def test_checker_requires_hi_design_r_lo() -> None:
    """缺少 HI 任务设计时 R_LO 时必须显式报错。"""

    tasks = _tasks()
    try:
        RuntimeBudgetSafetyChecker(tasks, design_r_lo={"h1": 6})
        assert False, "应抛出 ValueError"
    except ValueError as exc:
        assert "design_r_lo" in str(exc)

