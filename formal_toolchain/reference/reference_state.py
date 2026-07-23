from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from formal_toolchain.bridge.logical_events import LogicalEvent
from formal_toolchain.adapters.formal_runtime_snapshot import (
    ReleasedJobRecord, TerminalRecord, MissRecord,
)

JobKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class PendingReferenceRelease:
    job_key: JobKey
    release_time: int
    absolute_deadline: int
    criticality: str
    priority_index: int
    abnormal_hi: bool
    effective_release_mode: str
    release_class: str
    release_budget: int
    removal_demand: int


@dataclass(frozen=True, slots=True)
class ReferenceModeSwitch:
    switch_time: int
    triggering_job_key: JobKey
    reason: str


@dataclass(frozen=True, slots=True)
class ReferenceJob:
    job_key: JobKey
    release_time: int
    absolute_deadline: int
    criticality: str
    released_mode: str
    release_class: str
    budget: int
    removal_demand: int
    executed: int = 0


@dataclass(frozen=True, slots=True)
class ReferenceState:
    time: int
    mode: str
    jobs: Mapping[JobKey, ReferenceJob]
    released: Mapping[JobKey, ReleasedJobRecord]
    terminal: Mapping[JobKey, TerminalRecord]
    misses: tuple[MissRecord, ...]
    ready_order: tuple[JobKey, ...]
    running: JobKey | None
    frontier: tuple[LogicalEvent, ...]
    pending_releases: Mapping[JobKey, PendingReferenceRelease] = field(default_factory=dict)
    release_demand_overrides: Mapping[JobKey, int] = field(default_factory=dict)
    ghost_future_budgets: Mapping[str, int] = field(default_factory=dict)
    mode_switches: tuple[ReferenceModeSwitch, ...] = ()
    abnormal_hi_releases: frozenset[JobKey] = frozenset()
    primary_on_switch_time: bool = True
