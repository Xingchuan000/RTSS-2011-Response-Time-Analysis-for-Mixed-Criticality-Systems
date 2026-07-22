"""Phase F03：同时到达任务的 actual_cost 原子冻结适配器。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FrozenDemand:
    task_name: str
    release_index: int
    release_time: int
    actual_cost: int


class BatchFrozenExecutionScenario:
    """把 keyed oracle 包装为按 arrival batch 一次性读取的 actual_cost oracle。"""

    def __init__(
        self,
        ordered_tasks: Sequence[Any],
        delegate: Any,
        *,
        release_time: Callable[[Any, int], int] | None = None,
        code_hi_by_task: Mapping[str, int] | None = None,
        code_lo_by_task: Mapping[str, int] | None = None,
    ) -> None:
        self.name = f"batch_frozen[{getattr(delegate, 'name', type(delegate).__name__)}]"
        self.ordered_tasks = tuple(ordered_tasks)
        self.delegate = delegate
        self.release_time = release_time or (
            lambda task, index: int(getattr(task, "offset", 0)) + int(task.period) * int(index)
        )
        self.code_hi_by_task = dict(
            code_hi_by_task or {str(task.name): int(task.c_hi) for task in ordered_tasks}
        )
        self.code_lo_by_task = dict(
            code_lo_by_task or {str(task.name): int(task.c_lo) for task in ordered_tasks}
        )
        self._batches: dict[int, dict[tuple[str, int], FrozenDemand]] = {}

    def _delegate_cost(self, task: Any, release_index: int) -> int:
        if hasattr(self.delegate, "actual_cost_for"):
            value = self.delegate.actual_cost_for(task, release_index)
        elif hasattr(self.delegate, "demand"):
            value = self.delegate.demand(task, release_index)
        elif callable(self.delegate):
            value = self.delegate(task, release_index)
        elif isinstance(self.delegate, Mapping):
            value = self.delegate[(str(task.name), int(release_index))]
        else:
            raise TypeError("delegate 必须实现 actual_cost_for(task, release_index)")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("actual_cost 必须是正整数")
        hi_cap = self.code_hi_by_task.get(str(task.name))
        if hi_cap is not None and value > hi_cap:
            raise ValueError(f"actual_cost 超过 code C_HI: {task.name}")
        return int(value)

    def _freeze_batch(self, release_time: int) -> None:
        frozen: dict[tuple[str, int], FrozenDemand] = {}
        for task in self.ordered_tasks:
            offset = int(getattr(task, "offset", 0))
            if release_time < offset:
                continue
            delta = release_time - offset
            period = int(task.period)
            if period <= 0 or delta % period != 0:
                continue
            index = delta // period
            key = (str(task.name), int(index))
            frozen[key] = FrozenDemand(
                task_name=key[0],
                release_index=key[1],
                release_time=release_time,
                actual_cost=self._delegate_cost(task, index),
            )
        self._batches[release_time] = frozen

    def actual_cost_for(self, task: Any, release_index: int) -> int:
        time = int(self.release_time(task, int(release_index)))
        if time not in self._batches:
            self._freeze_batch(time)
        key = (str(task.name), int(release_index))
        row = self._batches[time].get(key)
        if row is None:
            raise KeyError(f"release 不属于 batch: {key}@{time}")
        return row.actual_cost

    # 仅为旧内部调用保留；不得作为 runtime 主协议。
    def demand(self, task: Any, release_index: int) -> int:
        return self.actual_cost_for(task, release_index)

    def frozen_batches(self) -> dict[int, list[dict[str, Any]]]:
        return {
            time: [
                {
                    "task_name": row.task_name,
                    "release_index": row.release_index,
                    "release_time": row.release_time,
                    "actual_cost": row.actual_cost,
                }
                for _, row in sorted(batch.items())
            ]
            for time, batch in sorted(self._batches.items())
        }

    def frozen_batches_json(self) -> list[dict[str, Any]]:
        return [
            {
                "release_time": time,
                "jobs": [
                    {
                        "task_name": row.task_name,
                        "release_index": row.release_index,
                        "actual_cost": row.actual_cost,
                    }
                    for _, row in sorted(batch.items())
                ],
            }
            for time, batch in sorted(self._batches.items())
        ]

    def classify(self, task: Any, release_index: int) -> str:
        """按冻结后的 A_J 与 code C_LO 唯一决定 normal/abnormal。"""

        demand = self.actual_cost_for(task, release_index)
        return "abnormal" if demand > self.code_lo_by_task[str(task.name)] else "normal"
