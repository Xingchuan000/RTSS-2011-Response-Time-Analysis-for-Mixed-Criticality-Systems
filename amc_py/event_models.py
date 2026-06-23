"""事件驱动运行时的基础事件模型与事件队列。

本模块对应实现计划的阶段一，负责两件事：
1. 定义事件类型与事件数据结构；
2. 提供稳定、可预测排序的事件优先队列。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
import itertools
from typing import Any


class EventType(str, Enum):
    """事件类型枚举。

    排序优先级（同一时刻）由 EventQueue 内部维护，不依赖枚举声明顺序。
    """

    BUDGET_UPDATE = "BUDGET_UPDATE"
    JOB_COMPLETION = "JOB_COMPLETION"
    RESPONSE_TIME_EXPIRY = "RESPONSE_TIME_EXPIRY"
    BUDGET_OVERRUN = "BUDGET_OVERRUN"
    DEADLINE_CHECK = "DEADLINE_CHECK"
    JOB_ARRIVAL = "JOB_ARRIVAL"


@dataclass(frozen=True, slots=True)
class Event:
    """事件对象。

    字段说明：
    - time: 事件发生时刻（非负整数）；
    - event_type: 事件类型；
    - task_name/release_index: 关联 job 的主键信息；
    - token: 事件失效机制预留字段（阶段三开始使用）；
    - payload: 事件扩展载荷。
    """

    time: int
    event_type: EventType
    task_name: str | None = None
    release_index: int | None = None
    token: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 统一前置校验，避免负时刻事件进入队列污染调度顺序。
        if self.time < 0:
            raise ValueError("event.time 必须为非负整数")


class EventQueue:
    """稳定排序事件队列。

    排序规则：
    1. `time` 小的优先；
    2. 同时刻下按事件优先级：
       BUDGET_UPDATE < JOB_COMPLETION < RESPONSE_TIME_EXPIRY < BUDGET_OVERRUN
       < DEADLINE_CHECK < JOB_ARRIVAL；
    3. 若仍相同，按插入顺序 FIFO。
    """

    _TYPE_PRIORITY: dict[EventType, int] = {
        EventType.BUDGET_UPDATE: 0,
        EventType.JOB_COMPLETION: 1,
        EventType.RESPONSE_TIME_EXPIRY: 2,
        EventType.BUDGET_OVERRUN: 3,
        EventType.DEADLINE_CHECK: 4,
        EventType.JOB_ARRIVAL: 5,
    }

    def __init__(self) -> None:
        """初始化空队列与全局递增序号。"""

        self._heap: list[tuple[int, int, int, Event]] = []
        self._counter = itertools.count()

    def push(self, event: Event) -> None:
        """压入一个新事件。"""

        order = next(self._counter)
        type_priority = self._TYPE_PRIORITY[event.event_type]
        heapq.heappush(self._heap, (event.time, type_priority, order, event))

    def pop(self) -> Event:
        """弹出当前最早应处理的事件。"""

        _, _, _, event = heapq.heappop(self._heap)
        return event

    def peek(self) -> Event | None:
        """查看堆顶事件但不弹出；空队列时返回 None。"""

        if not self._heap:
            return None
        return self._heap[0][3]

    def pop_all_matching(self, *, time: int, event_type: EventType) -> list[Event]:
        """弹出所有“同一时刻 + 同一类型”的事件，便于批处理 simultaneous arrivals。"""

        matched_entries: list[tuple[int, int, int, Event]] = []
        remaining_entries: list[tuple[int, int, int, Event]] = []
        for entry in self._heap:
            event = entry[3]
            if event.time == time and event.event_type is event_type:
                matched_entries.append(entry)
            else:
                remaining_entries.append(entry)
        self._heap = remaining_entries
        heapq.heapify(self._heap)
        matched_entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        matched = [entry[3] for entry in matched_entries]
        return matched

    def empty(self) -> bool:
        """队列是否为空。"""

        return len(self._heap) == 0

    def __len__(self) -> int:
        """返回队列当前事件数。"""

        return len(self._heap)


__all__ = ["EventType", "Event", "EventQueue"]
