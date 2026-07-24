"""Close-boundary relation for full and saturated protected-prefix states."""

from __future__ import annotations

from typing import Any

from .observable import ProtectedStateObservable, project_protected_state
from .types import ProtectedPrefixBuildResult


def rel_pp_close(
    full_state: Any,
    prefix_state: Any,
    *,
    construction: ProtectedPrefixBuildResult,
    full_taskset: object,
    prefix_taskset: object,
) -> bool:
    return project_protected_state(
        full_state, protected_task_names=frozenset(construction.protected_task_names),
        taskset=full_taskset,
    ) == project_protected_state(
        prefix_state, protected_task_names=frozenset(construction.protected_task_names),
        taskset=prefix_taskset,
    )


def relation_difference(
    full_state: Any,
    prefix_state: Any,
    *,
    construction: ProtectedPrefixBuildResult,
    full_taskset: object,
    prefix_taskset: object,
) -> dict[str, Any]:
    full = project_protected_state(full_state, protected_task_names=frozenset(construction.protected_task_names), taskset=full_taskset)
    prefix = project_protected_state(prefix_state, protected_task_names=frozenset(construction.protected_task_names), taskset=prefix_taskset)
    return {"equal": full == prefix, "full": full, "prefix": prefix}
