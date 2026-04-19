from amc_py.models import Criticality, Task, TaskSet


def test_task_creation_hi() -> None:
    task = Task(
        name="tau1",
        period=20,
        deadline=20,
        c_lo=3,
        c_hi=5,
        criticality=Criticality.HI,
    )

    assert task.name == "tau1"
    assert task.criticality is Criticality.HI


def test_taskset_len() -> None:
    task_set = TaskSet()
    task_set.add(
        Task(
            name="tau_lo",
            period=10,
            deadline=10,
            c_lo=1,
            c_hi=1,
            criticality=Criticality.LO,
        )
    )

    assert len(task_set) == 1
