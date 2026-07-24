from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.reference.task_mapping import ReferenceTaskset


@dataclass(frozen=True, slots=True)
class ProtectedPrefixBuildResult:
    full_taskset_fingerprint: str
    prefix_taskset: ReferenceTaskset
    cutoff_task_name: str
    cutoff_priority_index: int
    protected_task_names: tuple[str, ...]
    tail_task_names: tuple[str, ...]
    partition_witness: Mapping[str, Any]
    saturation_witness: Mapping[str, Any]
