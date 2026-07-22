from __future__ import annotations

from copy import copy
from dataclasses import MISSING, fields, replace
from typing import Any

from amc_py.runtime_models import RuntimeConfig


def copy_runtime_config(runtime_config: Any, **updates: Any) -> Any:
    """复制 RuntimeConfig 并按需覆盖字段。"""

    try:
        return replace(runtime_config, **updates)
    except TypeError:
        values: dict[str, Any] = {}
        for field in fields(RuntimeConfig):
            if hasattr(runtime_config, field.name):
                values[field.name] = getattr(runtime_config, field.name)
            elif field.default is not MISSING:
                values[field.name] = field.default
            elif field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                values[field.name] = field.default_factory()  # type: ignore[misc]
            else:
                raise TypeError(f"runtime_config 缺少字段: {field.name}")
        values.update(updates)
        return RuntimeConfig(**values)
