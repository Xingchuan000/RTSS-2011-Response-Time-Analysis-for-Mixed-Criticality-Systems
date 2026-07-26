"""Immutable q-AMC profile models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QAmcQualityLevel:
    runtime_level: int
    raw_rank: int
    normalized_quality: float
    isolated_work_ratio: float
    isolated_wcet: int

    def __post_init__(self) -> None:
        if self.runtime_level < 0 or self.raw_rank <= 0 or self.isolated_wcet <= 0:
            raise ValueError("QAMC_INVALID_QUALITY_LEVEL")


@dataclass(frozen=True, slots=True)
class QAmcTaskProfile:
    task_name: str
    design_c_lo: int
    full_quality_isolated_wcet: int
    design_lo_interference_budget: int
    levels: tuple[QAmcQualityLevel, ...]
    initial_runtime_level: int
    threshold_runtime_level: int
    degradable: bool

    def __post_init__(self) -> None:
        if not self.task_name or self.design_c_lo <= 0:
            raise ValueError("QAMC_INVALID_TASK_PROFILE")
        if not self.levels:
            raise ValueError("QAMC_PROFILE_HAS_NO_LEVELS")
        if self.full_quality_isolated_wcet + self.design_lo_interference_budget != self.design_c_lo:
            raise ValueError("QAMC_W_MAX_PLUS_I_DESIGN_MUST_EQUAL_C_LO")
        if [level.runtime_level for level in self.levels] != list(range(len(self.levels))):
            raise ValueError("QAMC_RUNTIME_LEVELS_NOT_CONTIGUOUS")
        if self.initial_runtime_level != len(self.levels) - 1:
            raise ValueError("QAMC_INITIAL_LEVEL_MUST_BE_HIGHEST")
        if self.threshold_runtime_level != 0:
            raise ValueError("QAMC_THRESHOLD_LEVEL_MUST_BE_LOWEST")
        if not self.degradable and len(self.levels) != 1:
            raise ValueError("QAMC_NON_DEGRADABLE_PROFILE_MUST_HAVE_ONE_LEVEL")
        previous: QAmcQualityLevel | None = None
        for level in self.levels:
            if previous is not None:
                if level.raw_rank < previous.raw_rank:
                    raise ValueError("QAMC_RAW_RANK_NOT_MONOTONIC")
                if level.normalized_quality < previous.normalized_quality:
                    raise ValueError("QAMC_NORMALIZED_QUALITY_NOT_MONOTONIC")
                if level.isolated_wcet < previous.isolated_wcet:
                    raise ValueError("QAMC_ISOLATED_WCET_NOT_MONOTONIC")
            previous = level
        if self.levels[-1].normalized_quality != 1.0:
            raise ValueError("QAMC_MAX_LEVEL_QUALITY_MUST_BE_ONE")
        if self.levels[-1].isolated_wcet != self.full_quality_isolated_wcet:
            raise ValueError("QAMC_MAX_LEVEL_WCET_MISMATCH")

    def level(self, runtime_level: int) -> QAmcQualityLevel:
        try:
            return self.levels[runtime_level]
        except (IndexError, TypeError) as exc:
            raise ValueError(f"QAMC_UNKNOWN_RUNTIME_LEVEL:{self.task_name}:{runtime_level}") from exc

    def can_degrade(self, runtime_level: int) -> bool:
        return self.degradable and runtime_level > self.threshold_runtime_level

    def next_lower_level(self, runtime_level: int) -> int:
        if not self.can_degrade(runtime_level):
            raise ValueError(f"QAMC_MIN_QUALITY_EXHAUSTED:{self.task_name}")
        return runtime_level - 1

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "design_c_lo": self.design_c_lo,
            "full_quality_isolated_wcet": self.full_quality_isolated_wcet,
            "design_lo_interference_budget": self.design_lo_interference_budget,
            "levels": [
                {
                    "runtime_level": level.runtime_level,
                    "raw_rank": level.raw_rank,
                    "normalized_quality": level.normalized_quality,
                    "isolated_work_ratio": level.isolated_work_ratio,
                    "isolated_wcet": level.isolated_wcet,
                }
                for level in self.levels
            ],
            "initial_runtime_level": self.initial_runtime_level,
            "threshold_runtime_level": self.threshold_runtime_level,
            "degradable": self.degradable,
        }


@dataclass(frozen=True, slots=True)
class QAmcProfileBundle:
    schema_version: str
    semantic_version: str
    taskset_fingerprint: str
    spec_fingerprint: str
    profiles: dict[str, QAmcTaskProfile]
    fingerprint: str
    ratio_semantics: str = "isolated_work_to_interference"
    integer_partition_rule: str = "minimum_ratio_error_then_lower_w"
    demand_mapping_version: str = "wcet_capped_component_split_v1"

    def __post_init__(self) -> None:
        if not self.taskset_fingerprint or not self.spec_fingerprint:
            raise ValueError("QAMC_PROFILE_FINGERPRINT_FIELDS_REQUIRED")
        if set(self.profiles) != {name for name in self.profiles}:
            raise ValueError("QAMC_PROFILE_TASK_NAMES_INVALID")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_version": self.semantic_version,
            "taskset_fingerprint": self.taskset_fingerprint,
            "spec_fingerprint": self.spec_fingerprint,
            "ratio_semantics": self.ratio_semantics,
            "integer_partition_rule": self.integer_partition_rule,
            "demand_mapping_version": self.demand_mapping_version,
            "profiles": {name: profile.to_jsonable() for name, profile in sorted(self.profiles.items())},
            "fingerprint": self.fingerprint,
        }


__all__ = ["QAmcProfileBundle", "QAmcQualityLevel", "QAmcTaskProfile"]
