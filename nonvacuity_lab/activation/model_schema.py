from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class IntDomain:
    name: str
    lower: int
    upper: int

    def validate(self):
        if self.lower > self.upper:
            raise ValueError(f"invalid domain for {self.name}")


@dataclass(frozen=True)
class SymbolicTaskBudget:
    task_id: str
    criticality: str
    current_budget: IntDomain
    minimum_budget: int
    certified_upper_bound: int
    reference_budget: int


@dataclass(frozen=True)
class SymbolicAction:
    action_id: int
    task_id: str | None
    operation: str
    ratio: Fraction | None
    minimum_increment: int
    rounding_mode: str


@dataclass(frozen=True)
class PathPredicate:
    feature_index: int
    comparator: str
    threshold: int
    node_id: int
