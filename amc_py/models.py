"""Core data models for AMC schedulability analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Criticality(str, Enum):
    """Task criticality level in a dual-criticality system."""

    LO = "LO"
    HI = "HI"


@dataclass(frozen=True, slots=True)
class Task:
    """Mixed-criticality periodic task.

    Parameters mirror mceval semantics:
    - period: task period T_i
    - deadline: relative deadline D_i
    - c_lo: WCET in LO mode C_i(LO)
    - c_hi: WCET in HI mode C_i(HI), only relevant for HI tasks
    """

    name: str
    period: int
    deadline: int
    c_lo: int
    c_hi: int
    criticality: Criticality

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be > 0")
        if self.deadline <= 0:
            raise ValueError("deadline must be > 0")
        if self.deadline > self.period:
            raise ValueError("deadline must be <= period")
        if self.c_lo <= 0:
            raise ValueError("c_lo must be > 0")
        if self.c_hi < self.c_lo:
            raise ValueError("c_hi must be >= c_lo")
        if self.c_hi <= 0:
            raise ValueError("c_hi must be > 0")


@dataclass(slots=True)
class TaskSet:
    """Collection wrapper for task operations used by analyses."""

    tasks: list[Task] = field(default_factory=list)

    @classmethod
    def from_iterable(cls, tasks: Iterable[Task]) -> "TaskSet":
        return cls(list(tasks))

    def add(self, task: Task) -> None:
        self.tasks.append(task)

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)


@dataclass(frozen=True, slots=True)
class SchedulabilityResult:
    """Result container for a schedulability/feasibility test."""

    schedulable: bool
    method: str
    response_times: dict[str, int] = field(default_factory=dict)
    details: str = ""


@dataclass(frozen=True, slots=True)
class PriorityAssignmentResult:
    """Result container for priority assignment algorithms."""

    success: bool
    method: str
    priorities: dict[str, int] = field(default_factory=dict)
    details: str = ""
