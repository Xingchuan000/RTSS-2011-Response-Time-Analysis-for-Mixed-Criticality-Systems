"""Phase K 的 P0 有限模型容量推导。

容量必须从 Phase I 生成的 canonical reference taskset 推导，不能使用
固定的 task/job/queue 数量。该对象同时作为 SMT schema 和 proof context
的输入，保证真实 Seed 扩展任务数量时不会静默截断证明状态。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True, slots=True)
class P0ModelBounds:
    """由 reference taskset 唯一派生的 P0 有限状态容量。"""

    task_slots: int
    job_slots: int
    queue_slots: int
    max_preemptions_per_job: int
    derivation: str = "P0_CONSTRAINED_DEADLINE_PERIODIC_V1"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "derivation":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须为正整数")

    @property
    def fingerprint(self) -> str:
        return sha256_object({"schema_version": "p0_model_bounds_v1", **asdict(self)})

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "p0_model_bounds_v1", **asdict(self),
                "fingerprint": self.fingerprint}


def _require_int(task: Mapping[str, Any], field: str) -> int:
    value = task.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"reference task {task.get('name')} 的 {field} 必须为正整数")
    return value


def derive_p0_model_bounds(reference_taskset: Mapping[str, Any]) -> P0ModelBounds:
    """根据 canonical reference taskset 计算不会遗漏正常路径的容量。"""

    tasks = reference_taskset.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("reference taskset 缺少 tasks")

    task_slots = len(tasks)
    jobs_per_task = [
        max(1, ceil(_require_int(task, "deadline") / _require_int(task, "period")))
        for task in tasks
    ]
    job_slots = sum(jobs_per_task)

    max_preemptions_per_job = 1
    for index, task in enumerate(tasks):
        deadline = _require_int(task, "deadline")
        interference = sum(
            ceil(deadline / _require_int(hp, "period"))
            for hp in tasks[:index]
        )
        max_preemptions_per_job = max(max_preemptions_per_job, interference)

    release_events = task_slots
    deadline_events = job_slots
    running_and_stale_events = 2 * job_slots * (1 + max_preemptions_per_job)
    queue_slots = release_events + deadline_events + running_and_stale_events + 1

    return P0ModelBounds(task_slots, job_slots, queue_slots, max_preemptions_per_job)


def _legacy_test_bounds() -> P0ModelBounds:
    """仅供未升级的旧单元测试调用；正式 Phase K 必须显式传入 bounds。"""
    return P0ModelBounds(task_slots=4, job_slots=4, queue_slots=8,
                         max_preemptions_per_job=1)
