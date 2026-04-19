from amc_py.models import Criticality, Task
from amc_py.priorities import (
    reindex_priorities,
    sort_by_criticality_monotonic,
    sort_by_crmpo,
    sort_by_deadline_monotonic,
)


def _sample_tasks() -> list[Task]:
    return [
        Task("lo_d8", period=10, deadline=8, c_lo=1, c_hi=1, criticality=Criticality.LO),
        Task("hi_d9", period=12, deadline=9, c_lo=1, c_hi=2, criticality=Criticality.HI),
        Task("hi_d7", period=20, deadline=7, c_lo=2, c_hi=4, criticality=Criticality.HI),
    ]


def test_sort_by_deadline_monotonic() -> None:
    ordered = sort_by_deadline_monotonic(_sample_tasks())
    assert [task.name for task in ordered] == ["hi_d7", "lo_d8", "hi_d9"]


def test_sort_by_criticality_monotonic() -> None:
    ordered = sort_by_criticality_monotonic(_sample_tasks())
    assert [task.name for task in ordered] == ["hi_d9", "hi_d7", "lo_d8"]


def test_sort_by_crmpo_and_reindex() -> None:
    ordered = sort_by_crmpo(_sample_tasks())
    assert [task.name for task in ordered] == ["hi_d7", "hi_d9", "lo_d8"]

    priorities = reindex_priorities(ordered)
    assert priorities == {"hi_d7": 0, "hi_d9": 1, "lo_d8": 2}
