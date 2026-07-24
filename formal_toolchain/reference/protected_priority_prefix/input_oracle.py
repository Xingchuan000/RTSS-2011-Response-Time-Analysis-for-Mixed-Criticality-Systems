"""Complete recurring protected input oracle.

The oracle is a deterministic projection of a full-reference input oracle.
It validates every queried recurring job before caching it so malformed or
mode-dependent inputs cannot silently become proof witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from .types import ProtectedPrefixBuildResult

JobKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class ProtectedJobInput:
    job_key: JobKey
    release_time: int
    actual_demand: int
    hi_class: Literal["NORMAL", "ABNORMAL"] | None


class FullReferenceInputOracle(Protocol):
    """Protocol for a full-reference release input oracle."""

    def input_for(self, task_name: str, release_index: int) -> dict[str, Any]: ...


class ProtectedInputOracle:
    """Validated projection of full-reference inputs to the protected prefix.

    This object is an input contract, not a proof of complete-execution
    existence.  It guarantees only that each queried protected job has a
    stable, release-fixed and WCET-legal input record.
    """

    def __init__(
        self,
        full_oracle: FullReferenceInputOracle,
        protected_task_names: frozenset[str],
        construction: ProtectedPrefixBuildResult,
    ) -> None:
        expected = frozenset(construction.protected_task_names)
        if protected_task_names != expected:
            raise ValueError("PROTECTED_INPUT_ORACLE_PARTITION_MISMATCH")
        self._full = full_oracle
        self._protected = protected_task_names
        self._construction = construction
        self._tasks = {str(task.name): task for task in construction.prefix_taskset.tasks}
        self._cache: dict[JobKey, ProtectedJobInput] = {}

    def input_for(self, task_name: str, release_index: int) -> ProtectedJobInput:
        if task_name not in self._protected or task_name not in self._tasks:
            raise ValueError(f"PROTECTED_INPUT_TASK_OUTSIDE_PREFIX:{task_name}")
        if isinstance(release_index, bool) or not isinstance(release_index, int) or release_index < 0:
            raise ValueError("PROTECTED_INPUT_RELEASE_INDEX_INVALID")

        key: JobKey = (task_name, release_index)
        if key in self._cache:
            return self._cache[key]

        full_input = self._full.input_for(task_name, release_index)
        if not isinstance(full_input, Mapping):
            raise ValueError("FULL_INPUT_ORACLE_RECORD_INVALID")
        declared_key = full_input.get("job_key")
        if declared_key is not None and tuple(declared_key) != key:
            raise ValueError("FULL_INPUT_ORACLE_JOB_KEY_MISMATCH")

        release_time = full_input.get("release_time")
        demand = full_input.get("actual_demand")
        hi_class = full_input.get("hi_class")
        if isinstance(release_time, bool) or not isinstance(release_time, int) or release_time < 0:
            raise ValueError("FULL_INPUT_ORACLE_RELEASE_TIME_INVALID")
        if isinstance(demand, bool) or not isinstance(demand, int) or demand <= 0:
            raise ValueError("FULL_INPUT_ORACLE_DEMAND_INVALID")

        task = self._tasks[task_name]
        expected_release_time = int(task.offset) + release_index * int(task.period)
        if release_time != expected_release_time:
            raise ValueError("FULL_INPUT_ORACLE_RELEASE_TIME_NOT_PERIODIC")
        if task.criticality == "LO":
            if hi_class is not None:
                raise ValueError("LO_INPUT_MUST_NOT_HAVE_HI_CLASS")
            bound = int(task.c_lo)  # saturated prefix has C_LO == C_HI for LO tasks
        else:
            if hi_class not in {"NORMAL", "ABNORMAL"}:
                raise ValueError("HI_INPUT_CLASS_INVALID")
            if hi_class == "NORMAL":
                bound = int(task.c_lo)
            else:
                if demand <= int(task.c_lo):
                    raise ValueError("ABNORMAL_HI_DEMAND_NOT_ABOVE_C_LO")
                bound = int(task.c_hi)
        if demand > bound:
            raise ValueError("PROTECTED_INPUT_DEMAND_EXCEEDS_REFERENCE_BOUND")

        result = ProtectedJobInput(
            job_key=key,
            release_time=release_time,
            actual_demand=demand,
            hi_class=hi_class,
        )
        self._cache[key] = result
        return result


def project_full_oracle_to_prefix(
    full_oracle: FullReferenceInputOracle,
    protected_task_names: frozenset[str],
    construction: ProtectedPrefixBuildResult,
) -> ProtectedInputOracle:
    """Project a full-reference input oracle to a protected-prefix oracle."""
    return ProtectedInputOracle(full_oracle, protected_task_names, construction)
