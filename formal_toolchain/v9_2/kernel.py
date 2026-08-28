"""Normative data model for the V9.2 Policy-Timing Kernel.

This module deliberately contains only semantics that can be stated exactly without
reusing any retired RTA/reference terminal route.  The symbolic transition encoder is a
separate proof-producing component.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Tuple


class Phase(IntEnum):
    SETTLE_SERVICE_AND_REMOVAL = 0
    IDLE_RECOVERY = 1
    DEADLINE_OBSERVE = 2
    ARRIVAL_BATCH_FREEZE = 3
    MODE_SWITCH = 4
    CONTROLLER = 5
    FINAL_DISPATCH = 6
    TIME_ADVANCE_AND_SERVICE = 7


@dataclass(frozen=True, slots=True)
class ReleaseEligibility:
    task: str
    eta: int
    period: int

    def __post_init__(self) -> None:
        if self.period <= 0 or not (0 <= self.eta <= self.period):
            raise ValueError("invalid release-eligibility state")

    @property
    def eligible(self) -> bool:
        return self.eta == self.period


@dataclass(frozen=True, slots=True)
class KernelJob:
    job_key: tuple[str, int]
    task: str
    release_time: int
    absolute_deadline: int
    priority: int
    tie_break: tuple[int, int]
    criticality: str
    release_entry_mode: str
    classification: str
    budget_at_release: int
    actual_demand: int
    effective_demand: int
    executed_service: int
    removed: bool
    ready: bool

    @property
    def remaining(self) -> int:
        return max(0, self.effective_demand - self.executed_service)


@dataclass(frozen=True, slots=True)
class KernelState:
    t: int
    p: Phase
    m: str
    B: Mapping[str, int]
    eta: Tuple[ReleaseEligibility, ...]
    J: Tuple[KernelJob, ...]
    F: tuple[tuple[str, int], ...]
    M: int
    chi: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.t < 0 or self.m not in {"LO", "HI"} or self.M < 0:
            raise ValueError("invalid Policy-Timing Kernel state")

    @property
    def no_prior_hi_miss(self) -> bool:
        return self.M == 0


def closure_rank(state: KernelState) -> int:
    return 7 - int(state.p)


def next_zero_time_phase(phase: Phase) -> Phase:
    if phase == Phase.TIME_ADVANCE_AND_SERVICE:
        raise ValueError("P7 is the unique time-advancing phase")
    return Phase(int(phase) + 1)


def effective_demand_hi(actual_demand: int) -> int:
    if actual_demand <= 0:
        raise ValueError("actual demand must be positive")
    return actual_demand


def effective_demand_lo_primary(actual_demand: int, budget_at_release: int) -> int:
    if actual_demand <= 0 or budget_at_release <= 0:
        raise ValueError("LO primary demand inputs must be positive")
    return min(actual_demand, budget_at_release + 1)


def effective_demand_lo_degraded(actual_demand: int, degraded_cost: int) -> int:
    if actual_demand <= 0 or degraded_cost <= 0:
        raise ValueError("LO degraded demand inputs must be positive")
    return min(actual_demand, degraded_cost)


def classify_hi_release(actual_demand: int, c_lo: int, c_hi: int) -> str:
    if not (1 <= actual_demand <= c_hi) or not (1 <= c_lo <= c_hi):
        raise ValueError("HI release demand is outside the admissible domain")
    return "NORMAL" if actual_demand <= c_lo else "ABNORMAL"
