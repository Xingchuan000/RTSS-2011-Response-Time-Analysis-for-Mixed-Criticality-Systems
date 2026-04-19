from amc_py.amc import compute_amc_rtb_response_time
from amc_py.experiments import evaluate_taskset
from amc_py.models import Criticality, Task
from amc_py.priorities import audsley_opa
from amc_py.rta import compute_r_lo
from amc_py.smc import compute_smc_response_time, smc_sched_test


def _smc_lowest_priority_test(trial_order: list[Task], lowest_priority_idx: int) -> bool:
    task = trial_order[lowest_priority_idx]
    hp_tasks = trial_order[:lowest_priority_idx]
    return compute_smc_response_time(task, hp_tasks) is not None


def _amc_rtb_lowest_priority_test(trial_order: list[Task], lowest_priority_idx: int) -> bool:
    task = trial_order[lowest_priority_idx]
    hp_tasks = trial_order[:lowest_priority_idx]
    r_lo = compute_r_lo(task, hp_tasks)
    if r_lo is None:
        return False
    return compute_amc_rtb_response_time(task, hp_tasks, {task.name: r_lo}) is not None


def test_audsley_opa_returns_valid_priorities() -> None:
    tasks = [
        Task("tau3", period=20, deadline=20, c_lo=2, c_hi=4, criticality=Criticality.HI),
        Task("tau1", period=5, deadline=5, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("tau2", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
    ]

    result = audsley_opa(tasks, _smc_lowest_priority_test)

    assert result.success is True
    assert set(result.priorities.keys()) == {"tau1", "tau2", "tau3"}
    assert sorted(result.priorities.values()) == [0, 1, 2]


def test_audsley_opa_small_manual_taskset() -> None:
    """小规模人工任务集：验证 OPA 能找到可行分配并返回完整优先级映射。"""

    tasks = [
        Task("tau_a", period=9, deadline=9, c_lo=2, c_hi=4, criticality=Criticality.HI),
        Task("tau_b", period=12, deadline=10, c_lo=1, c_hi=3, criticality=Criticality.HI),
        Task("tau_c", period=20, deadline=20, c_lo=2, c_hi=2, criticality=Criticality.LO),
    ]

    result = audsley_opa(tasks, _smc_lowest_priority_test)
    assert result.success is True
    assert set(result.priorities) == {"tau_a", "tau_b", "tau_c"}
    assert sorted(result.priorities.values()) == [0, 1, 2]


def test_audsley_opa_fail_case() -> None:
    # 该任务集明显过载，OPA 应返回失败。
    tasks = [
        Task("x1", period=4, deadline=4, c_lo=3, c_hi=3, criticality=Criticality.LO),
        Task("x2", period=5, deadline=5, c_lo=3, c_hi=4, criticality=Criticality.HI),
        Task("x3", period=6, deadline=6, c_lo=4, c_hi=5, criticality=Criticality.HI),
    ]

    result = audsley_opa(tasks, _amc_rtb_lowest_priority_test)
    assert result.success is False


def test_opa_candidate_semantics_not_full_order_semantics() -> None:
    """候选最低优先级语义回归：旧的“整序判定”会失败，新语义应成功。"""

    tasks = [
        Task("tau_1", period=250, deadline=250, c_lo=49, c_hi=122, criticality=Criticality.LO),
        Task("tau_2", period=120, deadline=120, c_lo=34, c_hi=85, criticality=Criticality.HI),
        Task("tau_3", period=120, deadline=120, c_lo=7, c_hi=18, criticality=Criticality.HI),
        Task("tau_4", period=240, deadline=240, c_lo=58, c_hi=145, criticality=Criticality.LO),
        Task("tau_5", period=560, deadline=560, c_lo=11, c_hi=28, criticality=Criticality.HI),
    ]

    full_order_result = audsley_opa(tasks, smc_sched_test)
    candidate_result = audsley_opa(tasks, _smc_lowest_priority_test)

    assert full_order_result.success is False
    assert candidate_result.success is True


def test_dm_unschedulable_but_opa_schedulable() -> None:
    """对比例子：DM 不可调度，但 OPA 可以找到可调度优先级。"""

    tasks = [
        Task("t0", period=10, deadline=10, c_lo=1, c_hi=7, criticality=Criticality.LO),
        Task("t1", period=15, deadline=15, c_lo=5, c_hi=10, criticality=Criticality.HI),
        Task("t2", period=78, deadline=51, c_lo=2, c_hi=16, criticality=Criticality.LO),
        Task("t3", period=42, deadline=23, c_lo=2, c_hi=14, criticality=Criticality.LO),
    ]

    dm_result = evaluate_taskset(tasks, method="smc_no", priority_policy="dm")
    opa_result = evaluate_taskset(tasks, method="smc_no", priority_policy="opa")

    assert dm_result.schedulable is False
    assert opa_result.schedulable is True


def test_opa_consistent_under_input_permutation() -> None:
    """输入顺序打乱后，OPA 的可调度结论应保持一致。"""

    tasks = [
        Task("t0", period=35, deadline=30, c_lo=3, c_hi=8, criticality=Criticality.HI),
        Task("t1", period=49, deadline=29, c_lo=8, c_hi=10, criticality=Criticality.HI),
        Task("t2", period=27, deadline=24, c_lo=1, c_hi=13, criticality=Criticality.LO),
    ]
    shuffled = [tasks[2], tasks[0], tasks[1]]

    result_a = evaluate_taskset(tasks, method="smc_no", priority_policy="opa")
    result_b = evaluate_taskset(shuffled, method="smc_no", priority_policy="opa")

    assert result_a.schedulable is result_b.schedulable


def test_opa_success_order_passes_full_method_test() -> None:
    tasks = [
        Task("tau3", period=20, deadline=20, c_lo=2, c_hi=4, criticality=Criticality.HI),
        Task("tau1", period=5, deadline=5, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("tau2", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
    ]

    opa = audsley_opa(tasks, _smc_lowest_priority_test)
    assert opa.success is True

    ordered = sorted(tasks, key=lambda task: opa.priorities[task.name])
    full = smc_sched_test(ordered)
    assert full.schedulable is True


def test_evaluate_taskset_with_opa() -> None:
    tasks = [
        Task("tau3", period=20, deadline=20, c_lo=2, c_hi=4, criticality=Criticality.HI),
        Task("tau1", period=5, deadline=5, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("tau2", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
    ]

    result = evaluate_taskset(tasks, method="amc_rtb", priority_policy="opa")

    assert result.method == "amc_rtb"
    assert "priority_policy=opa" in result.details
    assert isinstance(result.schedulable, bool)
