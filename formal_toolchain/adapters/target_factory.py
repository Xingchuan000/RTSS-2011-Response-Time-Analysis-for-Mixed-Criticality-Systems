"""权威 target factory 的导入和只读构造接口。"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FormalTarget:
    """实际部署 target 的只读形式化边界对象。"""

    ordered_tasks: tuple[Any, ...]
    runtime_config: Any
    environment: Any
    policy: Any
    scenario: Any
    action_definitions: tuple[dict[str, object], ...]
    feature_names: tuple[str, ...]
    provenance: dict[str, object]


def load_target_factory(spec: str):
    """按 ``module:attribute`` 加载 factory，不捕获其内部异常。"""
    if ":" not in spec:
        raise ValueError("target factory 必须使用 module:attribute 格式")
    module_name, attr_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attr_name)
    if not callable(factory):
        raise TypeError("target factory 必须可调用")
    return factory


def build_target(spec: str, kwargs: dict[str, Any] | None = None) -> Any:
    return load_target_factory(spec)(**(kwargs or {}))
