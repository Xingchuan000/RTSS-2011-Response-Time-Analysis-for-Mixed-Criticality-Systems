"""阶段 E：趋势回归测试。

本文件用于验证 AMC 相关分析器的相对关系是否与论文/参考实现趋势一致：
- AMC-max >= AMC-rtb >= SMC >= SMC-no
- UB-H&L 通常更“宽松”（即在部分场景可通过，而某些响应时间分析器可能失败）

注意：
- 这里的“>=”表示“在可调度判定上的不弱于”（若右侧可调度，则左侧也应可调度）。
- 由于优先级分配策略影响显著，本测试统一使用 OPA 对响应时间分析器分配优先级；
  对 UB-H&L 按其常见做法使用 DM。
"""

from __future__ import annotations

from amc_py.experiments import evaluate_taskset
from amc_py.generator import generate_taskset
from amc_py.models import Criticality, Task



def _analysis_chain(taskset: list[Task]) -> tuple[bool, bool, bool, bool]:
    """返回同一任务集在四个分析器下的可调度性布尔值。

    返回顺序：
    1. smc_no
    2. smc
    3. amc_rtb
    4. amc_max
    """

    smc_no_sched = evaluate_taskset(taskset, method="smc_no", priority_policy="opa").schedulable
    smc_sched = evaluate_taskset(taskset, method="smc", priority_policy="opa").schedulable
    amc_rtb_sched = evaluate_taskset(taskset, method="amc_rtb", priority_policy="opa").schedulable
    amc_max_sched = evaluate_taskset(taskset, method="amc_max", priority_policy="opa").schedulable
    return smc_no_sched, smc_sched, amc_rtb_sched, amc_max_sched



def test_trend_chain_on_generated_tasksets() -> None:
    """在一批可复现随机任务集上回归趋势关系。

    设计说明：
    - 通过固定 seed 生成多组任务集，覆盖不同参数扰动。
    - 对每组任务集检查“可调度性蕴含关系”：
      - 若 SMC-no 可调度，则 SMC 必须可调度。
      - 若 SMC 可调度，则 AMC-rtb 必须可调度。
      - 若 AMC-rtb 可调度，则 AMC-max 必须可调度。
    """

    for seed in range(1, 31):
        taskset = generate_taskset(
            num_tasks=6,
            total_util=0.75,
            min_period=10,
            max_period=200,
            cf=2.0,
            cp=0.5,
            seed=seed,
        )

        smc_no_sched, smc_sched, amc_rtb_sched, amc_max_sched = _analysis_chain(taskset)

        if smc_no_sched:
            assert smc_sched is True
        if smc_sched:
            assert amc_rtb_sched is True
        if amc_rtb_sched:
            assert amc_max_sched is True



def test_ub_hl_can_be_looser_than_smc_no_example() -> None:
    """验证一个“UB-H&L 通过而 SMC-no 失败”的示例。

    该示例来自固定参数生成（seed=7）并固化为显式任务，
    用于说明“UB-H&L 通常更宽松”这一经验结论。
    """

    taskset = [
        Task("tau_1", period=11, deadline=11, c_lo=2, c_hi=2, criticality=Criticality.LO),
        Task("tau_2", period=29, deadline=29, c_lo=6, c_hi=12, criticality=Criticality.HI),
        Task("tau_3", period=12, deadline=12, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("tau_4", period=13, deadline=13, c_lo=3, c_hi=6, criticality=Criticality.HI),
        Task("tau_5", period=29, deadline=29, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("tau_6", period=78, deadline=78, c_lo=3, c_hi=6, criticality=Criticality.HI),
    ]

    ub_hl_sched = evaluate_taskset(taskset, method="ub_hl", priority_policy="dm").schedulable
    smc_no_sched = evaluate_taskset(taskset, method="smc_no", priority_policy="opa").schedulable

    assert ub_hl_sched is True
    assert smc_no_sched is False
