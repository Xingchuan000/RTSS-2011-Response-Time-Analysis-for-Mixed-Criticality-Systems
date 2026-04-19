from amc_py.bounds import ub_h_test, ub_hl_test, ub_l_test
from amc_py.models import Criticality, Task


def test_ub_hl_pass_case() -> None:
    tasks = [
        Task("tau1", period=5, deadline=5, c_lo=1, c_hi=2, criticality=Criticality.HI),
        Task("tau2", period=12, deadline=12, c_lo=2, c_hi=3, criticality=Criticality.HI),
        Task("tau3", period=20, deadline=20, c_lo=2, c_hi=2, criticality=Criticality.LO),
    ]

    assert ub_l_test(tasks) is True
    assert ub_h_test(tasks) is True
    assert ub_hl_test(tasks) is True


def test_ub_hl_fail_case() -> None:
    tasks = [
        Task("tau1", period=5, deadline=5, c_lo=2, c_hi=4, criticality=Criticality.HI),
        Task("tau2", period=8, deadline=8, c_lo=3, c_hi=5, criticality=Criticality.HI),
    ]

    # 该任务集在 LO/HI 模式下都较紧，综合测试应失败。
    assert ub_hl_test(tasks) is False
