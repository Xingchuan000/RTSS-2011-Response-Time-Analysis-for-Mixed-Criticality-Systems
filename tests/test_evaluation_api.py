"""评估入口 API 与方法-策略兼容矩阵测试。"""

from __future__ import annotations

import pytest

from amc_py.experiments import evaluate_taskset, resolve_ordering
from amc_py.models import Criticality, Task


def _reference_taskset() -> list[Task]:
    """构造一个可重复的小型参考任务集。"""

    return [
        Task("tau_1", period=7, deadline=7, c_lo=1, c_hi=2, criticality=Criticality.LO),
        Task("tau_2", period=12, deadline=12, c_lo=2, c_hi=4, criticality=Criticality.HI),
        Task("tau_3", period=25, deadline=25, c_lo=2, c_hi=3, criticality=Criticality.LO),
    ]


def test_crmpo_baseline_can_run_as_independent_method() -> None:
    """阶段 2.1：CrMPO baseline 可作为独立 method 调用。"""

    result = evaluate_taskset(_reference_taskset(), method="crmpo_baseline", priority_policy="crmpo")
    assert result.method == "crmpo_baseline"
    assert "priority_policy=crmpo" in result.details


def test_invalid_method_policy_combination_raises_clear_error() -> None:
    """阶段 2.2：非法组合需要给出清晰错误信息。"""

    with pytest.raises(ValueError, match="method=crmpo_baseline 不支持 priority_policy=dm"):
        evaluate_taskset(_reference_taskset(), method="crmpo_baseline", priority_policy="dm")

    with pytest.raises(ValueError, match="method=ub_hl 不支持 priority_policy=opa"):
        evaluate_taskset(_reference_taskset(), method="ub_hl", priority_policy="opa")


def test_duplicate_task_names_are_rejected() -> None:
    """阶段 1.3：入口参数校验应拦截重复任务名。"""

    duplicated = [
        Task("dup", period=5, deadline=5, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("dup", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
    ]
    with pytest.raises(ValueError, match="重复任务名"):
        evaluate_taskset(duplicated, method="smc", priority_policy="dm")


# ---------------------------------------------------------------------------
# resolve_ordering() 公共接口直接测试（阶段运行时模拟改造，第 1 轮）
# ---------------------------------------------------------------------------
#
# 目的：
# - 锁定 DM / CrMPO / OPA 三条路径在“公共函数”层面的行为；
# - 避免 runtime 模块在复用优先级解析逻辑时被默默破坏；
# - 同时验证 OPA 失败时会抛出 RuntimeError，便于 evaluate_taskset 的
#   try/except 分支（把失败转成 SchedulabilityResult）保持稳定。


def test_resolve_ordering_dm_returns_deadline_monotonic_order() -> None:
    """DM 策略下，结果必须严格按 deadline 升序排列。"""

    tasks = _reference_taskset()
    ordered = resolve_ordering(tasks, priority_policy="dm", method="amc_rtb")

    # 任务名用于稳定断言；参考任务集的 deadline 分别为 7 / 12 / 25，
    # 因此 DM 期望顺序是 tau_1 -> tau_2 -> tau_3。
    assert [task.name for task in ordered] == ["tau_1", "tau_2", "tau_3"]

    # 显式再检查一次 deadline 是否单调不减，避免以后改参考任务集时出现偏差。
    deadlines = [task.deadline for task in ordered]
    assert deadlines == sorted(deadlines)

    # resolve_ordering 的语义是“返回新列表”，不应修改原始输入顺序。
    assert [task.name for task in tasks] == ["tau_1", "tau_2", "tau_3"]


def test_resolve_ordering_crmpo_puts_hi_tasks_ahead_of_lo_tasks() -> None:
    """CrMPO 策略下，HI 任务必须整体排在 LO 任务之前，同类内按 deadline 升序。"""

    tasks = _reference_taskset()
    ordered = resolve_ordering(tasks, priority_policy="crmpo", method="amc_rtb")

    # 参考任务集中只有 tau_2 是 HI；CrMPO 应当把它放在最前面。
    assert ordered[0].name == "tau_2"
    assert ordered[0].criticality is Criticality.HI

    # 其余两个 LO 任务按 deadline 升序（7 早于 25）：tau_1 在 tau_3 之前。
    lo_order = [task.name for task in ordered[1:]]
    assert lo_order == ["tau_1", "tau_3"]

    # 再显式校验一次：HI 前缀、LO 后缀的整体结构。
    criticalities = [task.criticality for task in ordered]
    assert criticalities == [Criticality.HI, Criticality.LO, Criticality.LO]


def test_resolve_ordering_opa_returns_permutation_of_input() -> None:
    """OPA 策略在可调度任务集上应给出一个完整的任务排列（不增不漏）。"""

    tasks = _reference_taskset()
    ordered = resolve_ordering(tasks, priority_policy="opa", method="amc_rtb")

    # OPA 具体顺序取决于分析细节，这里只做“必要条件”断言：
    # 1. 长度一致；2. 任务集合一致；3. 无重复。
    assert len(ordered) == len(tasks)
    assert {task.name for task in ordered} == {task.name for task in tasks}
    assert len({id(task) for task in ordered}) == len(tasks)


def test_resolve_ordering_opa_failure_raises_runtime_error() -> None:
    """OPA 对过载任务集无法分配优先级时，必须以 RuntimeError 形式显式报错。"""

    # 该任务集在 RTSS'11 语义下明显过载，OPA + amc_rtb 必然失败；
    # 与 tests/test_opa.py::test_audsley_opa_fail_case 使用同一组任务，
    # 避免因参数漂移导致未来测试互相冲突。
    overloaded = [
        Task("x1", period=4, deadline=4, c_lo=3, c_hi=3, criticality=Criticality.LO),
        Task("x2", period=5, deadline=5, c_lo=3, c_hi=4, criticality=Criticality.HI),
        Task("x3", period=6, deadline=6, c_lo=4, c_hi=5, criticality=Criticality.HI),
    ]

    with pytest.raises(RuntimeError, match="OPA 分配失败"):
        resolve_ordering(overloaded, priority_policy="opa", method="amc_rtb")


def test_resolve_ordering_rejects_unknown_priority_policy() -> None:
    """非法 priority_policy 必须抛出 ValueError，避免静默回退到默认策略。"""

    with pytest.raises(ValueError, match="不支持的 priority_policy"):
        resolve_ordering(_reference_taskset(), priority_policy="bogus", method="amc_rtb")
