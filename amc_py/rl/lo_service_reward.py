"""增量统计 interval LO service / equivalent JNE reward。"""

from __future__ import annotations

from dataclasses import dataclass, field

from amc_py.metrics import lo_job_service_quality
from amc_py.models import Criticality
from amc_py.runtime_models import Job, SimulationResult


@dataclass(frozen=True, slots=True)
class IntervalLoServiceDelta:
    """单个 agent interval 内结算的 LO 服务质量增量。"""

    released_jobs: int
    finalized_jobs: int
    service_quality_sum: float
    equiv_jne: float
    zero_service_jobs: int
    partial_service_jobs: int

    @property
    def service_quality_per_finalized_job(self) -> float:
        if self.finalized_jobs <= 0:
            return 0.0
        return self.service_quality_sum / float(self.finalized_jobs)

    @property
    def equiv_jne_per_finalized_job(self) -> float:
        if self.finalized_jobs <= 0:
            return 0.0
        return self.equiv_jne / float(self.finalized_jobs)


@dataclass(slots=True)
class LoServiceRewardTracker:
    """增量消费新 job、deadline miss 与尚未结算的 LO job。"""

    _seen_job_count: int = 0
    _seen_deadline_miss_count: int = 0
    _pending_lo_jobs: dict[tuple[str, int], Job] = field(default_factory=dict)
    _accounted_lo_jobs: set[tuple[str, int]] = field(default_factory=set)
    _lo_task_names: set[str] = field(default_factory=set)
    _lo_deadline_miss_keys: set[tuple[str, int]] = field(default_factory=set)

    cumulative_released_jobs: int = 0
    cumulative_finalized_jobs: int = 0
    cumulative_service_quality_sum: float = 0.0
    cumulative_equiv_jne: float = 0.0

    def _consume_new_deadline_misses(self, result: SimulationResult) -> None:
        """只消费自上次调用后新追加的 deadline miss。"""

        new_misses = result.deadline_misses[self._seen_deadline_miss_count :]
        self._seen_deadline_miss_count = len(result.deadline_misses)
        for miss in new_misses:
            if miss.task in self._lo_task_names:
                self._lo_deadline_miss_keys.add((miss.task, miss.release_index))

    def prime(self, result: SimulationResult) -> None:
        """注册 time=0 已释放 job，不产生 interval reward。"""

        self._seen_job_count = len(result.jobs)
        self._seen_deadline_miss_count = 0
        self._pending_lo_jobs.clear()
        self._accounted_lo_jobs.clear()
        self._lo_task_names = {
            job.task.name
            for job in result.jobs
            if job.task.criticality is Criticality.LO
        }
        self._lo_deadline_miss_keys.clear()
        self.cumulative_released_jobs = 0
        self.cumulative_finalized_jobs = 0
        self.cumulative_service_quality_sum = 0.0
        self.cumulative_equiv_jne = 0.0
        self._consume_new_deadline_misses(result)

        for job in result.jobs:
            if job.task.criticality is not Criticality.LO:
                continue
            key = (job.task.name, job.release_index)
            if key in self._accounted_lo_jobs:
                continue
            self.cumulative_released_jobs += 1
            service = lo_job_service_quality(
                job,
                lo_deadline_miss_keys=self._lo_deadline_miss_keys,
                terminal=False,
            )
            if service is None:
                self._pending_lo_jobs[key] = job
                continue

            # time=0 已终结是非常规边界；prime 不返回 reward，但累计量必须
            # 与最终 metrics 保持一致，避免 released/finalized 永久失配。
            self._accounted_lo_jobs.add(key)
            self.cumulative_finalized_jobs += 1
            self.cumulative_service_quality_sum += service
            self.cumulative_equiv_jne += 1.0 - service

    def consume(
        self,
        result: SimulationResult,
        *,
        terminal: bool,
    ) -> IntervalLoServiceDelta:
        """消费当前结果中的新增/终结 LO job，并返回本 interval 增量。"""

        released_jobs = 0
        new_jobs = result.jobs[self._seen_job_count :]
        self._seen_job_count = len(result.jobs)
        for job in new_jobs:
            if job.task.criticality is not Criticality.LO:
                continue
            self._lo_task_names.add(job.task.name)
            released_jobs += 1
            self.cumulative_released_jobs += 1
            key = (job.task.name, job.release_index)
            if key not in self._accounted_lo_jobs:
                self._pending_lo_jobs[key] = job

        # deadline miss 列表按运行时追加；只读取新增尾部，避免每个 agent step
        # 重扫全部历史 jobs / misses 造成长时域下的二次型开销。
        self._consume_new_deadline_misses(result)

        finalized_jobs = 0
        service_quality_sum = 0.0
        equiv_jne = 0.0
        zero_service_jobs = 0
        partial_service_jobs = 0
        for key, job in list(self._pending_lo_jobs.items()):
            service = lo_job_service_quality(
                job,
                lo_deadline_miss_keys=self._lo_deadline_miss_keys,
                terminal=terminal,
            )
            if service is None:
                continue
            finalized_jobs += 1
            service_quality_sum += service
            equiv_jne += 1.0 - service
            zero_service_jobs += int(service == 0.0)
            partial_service_jobs += int(0.0 < service < 1.0)
            self._accounted_lo_jobs.add(key)
            del self._pending_lo_jobs[key]

        self.cumulative_finalized_jobs += finalized_jobs
        self.cumulative_service_quality_sum += service_quality_sum
        self.cumulative_equiv_jne += equiv_jne
        return IntervalLoServiceDelta(
            released_jobs=released_jobs,
            finalized_jobs=finalized_jobs,
            service_quality_sum=service_quality_sum,
            equiv_jne=equiv_jne,
            zero_service_jobs=zero_service_jobs,
            partial_service_jobs=partial_service_jobs,
        )
