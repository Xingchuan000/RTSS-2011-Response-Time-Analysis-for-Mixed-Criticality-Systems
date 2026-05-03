"""事件模型与事件队列测试（阶段一）。"""

from __future__ import annotations

import pytest

from amc_py.event_models import Event, EventQueue, EventType


def test_event_queue_orders_by_time() -> None:
    """事件队列应先按 time 升序弹出。"""

    queue = EventQueue()
    queue.push(Event(time=5, event_type=EventType.JOB_ARRIVAL))
    queue.push(Event(time=1, event_type=EventType.JOB_ARRIVAL))
    queue.push(Event(time=3, event_type=EventType.JOB_ARRIVAL))

    assert queue.pop().time == 1
    assert queue.pop().time == 3
    assert queue.pop().time == 5


def test_event_queue_orders_by_event_type_at_same_time() -> None:
    """同一时刻下，事件类型排序应满足文档定义的优先级。"""

    queue = EventQueue()
    queue.push(Event(time=10, event_type=EventType.JOB_ARRIVAL))
    queue.push(Event(time=10, event_type=EventType.BUDGET_OVERRUN))
    queue.push(Event(time=10, event_type=EventType.JOB_COMPLETION))
    queue.push(Event(time=10, event_type=EventType.BUDGET_UPDATE))

    assert queue.pop().event_type is EventType.BUDGET_UPDATE
    assert queue.pop().event_type is EventType.JOB_COMPLETION
    assert queue.pop().event_type is EventType.BUDGET_OVERRUN
    assert queue.pop().event_type is EventType.JOB_ARRIVAL


def test_event_queue_keeps_fifo_for_identical_keys() -> None:
    """当 time 与 event_type 都相同，队列必须保持 FIFO 稳定性。"""

    queue = EventQueue()
    first = Event(time=7, event_type=EventType.JOB_ARRIVAL, task_name="t1")
    second = Event(time=7, event_type=EventType.JOB_ARRIVAL, task_name="t2")
    third = Event(time=7, event_type=EventType.JOB_ARRIVAL, task_name="t3")
    queue.push(first)
    queue.push(second)
    queue.push(third)

    assert queue.pop().task_name == "t1"
    assert queue.pop().task_name == "t2"
    assert queue.pop().task_name == "t3"


def test_event_rejects_negative_time_if_you_add_validation() -> None:
    """若实现了负时间校验，负时间事件应被拒绝。"""

    with pytest.raises(ValueError, match="非负整数"):
        Event(time=-1, event_type=EventType.JOB_ARRIVAL)
