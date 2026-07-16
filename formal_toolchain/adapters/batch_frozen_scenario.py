"""Phase F03：同时到达任务的 demand 原子冻结适配器。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


class BatchFrozenExecutionScenario:
    """把 keyed demand oracle 包装为按 arrival batch 一次性读取的 oracle。"""

    def __init__(self, ordered_tasks: Sequence[Any], delegate: Any,
                 *, release_time: Callable[[Any, int], int] | None = None,
                 code_hi_by_task: Mapping[str, int] | None = None,
                 code_lo_by_task: Mapping[str, int] | None = None) -> None:
        self.ordered_tasks = tuple(ordered_tasks)
        self.delegate = delegate
        self.release_time = release_time or (lambda task, index: int(task.period) * index)
        self.code_hi_by_task = dict(code_hi_by_task or {str(task.name): int(task.c_hi) for task in ordered_tasks})
        self.code_lo_by_task = dict(code_lo_by_task or {str(task.name): int(task.c_lo) for task in ordered_tasks})
        self._batches: dict[int, dict[tuple[str, int], int]] = {}

    def _read_delegate(self, task: Any, index: int) -> int:
        key = (str(task.name), int(index))
        if hasattr(self.delegate, "demand"):
            value = self.delegate.demand(task, index)
        elif callable(self.delegate):
            value = self.delegate(task, index)
        elif isinstance(self.delegate, Mapping):
            value = self.delegate[key]
        else:
            raise TypeError("delegate 必须提供 demand(task,index)、可调用对象或 mapping")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"demand {key} 必须是正整数")
        if key[0] not in self.code_hi_by_task or value > self.code_hi_by_task[key[0]]:
            raise ValueError(f"demand {key} 超过 code_c_hi")
        return value

    def demand(self, task: Any, release_index: int) -> int:
        """首次访问某时刻时冻结该时刻所有 simultaneous arrivals。"""
        time = int(self.release_time(task, int(release_index)))
        batch = self._batches.get(time)
        if batch is None:
            batch = {}
            for candidate in self.ordered_tasks:
                for index in range(0, int(time // candidate.period) + 1):
                    if int(self.release_time(candidate, index)) == time:
                        batch[(str(candidate.name), index)] = self._read_delegate(candidate, index)
            self._batches[time] = batch
        key = (str(task.name), int(release_index))
        if key not in batch:
            raise KeyError(f"release 不属于已冻结 batch: {key}")
        return batch[key]

    def frozen_batches(self) -> dict[int, dict[tuple[str, int], int]]:
        return {time: dict(values) for time, values in self._batches.items()}

    def classify(self, task: Any, release_index: int) -> str:
        """按冻结后的 A_J 与 code C_LO 唯一决定 normal/abnormal。"""
        demand = self.demand(task, release_index)
        return "abnormal" if demand > self.code_lo_by_task[str(task.name)] else "normal"
