"""评估入口 API 与方法-策略兼容矩阵测试。"""

from __future__ import annotations

import pytest

from amc_py.experiments import evaluate_taskset
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
