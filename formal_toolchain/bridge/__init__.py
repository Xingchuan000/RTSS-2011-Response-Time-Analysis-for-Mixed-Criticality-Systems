"""Phase K：release-fixed concrete-to-reference bridge。"""

from typing import Any

__all__ = ["P0Job", "P0ConcreteState", "P0ReferenceState", "P0Event", "relation_holds",
           "build_runtime_branch_map"]


def __getattr__(name: str) -> Any:
    if name == "build_runtime_branch_map":
        from .runtime_branch_map import build_runtime_branch_map
        return build_runtime_branch_map
    if name in {"P0Job", "P0ConcreteState", "P0ReferenceState", "P0Event", "relation_holds"}:
        from . import state_relation
        return getattr(state_relation, name)
    raise AttributeError(name)
