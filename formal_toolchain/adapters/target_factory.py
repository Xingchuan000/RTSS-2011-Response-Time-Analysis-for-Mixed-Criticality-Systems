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
    # runtime_adapter 由具体 factory 显式提供；这里不再根据 provenance 自动补 synthetic。
    runtime_adapter: Any = None


def load_target_factory(spec: str):
    """按 ``module:attribute`` 加载 factory，不捕获其内部异常。"""
    if ":" not in spec:
        raise ValueError("target factory 必须使用 module:attribute 格式")
    module_name, attr_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attr_name)
    if not callable(factory):
        raise TypeError("target factory 必须可调用")
    return factory


def build_target(spec: str, kwargs: dict[str, Any] | None = None) -> FormalTarget:
    """调用权威 factory，并严格检查返回对象的形式化边界类型。"""

    target = load_target_factory(spec)(**(kwargs or {}))
    if not isinstance(target, FormalTarget):
        raise TypeError("target factory 必须返回 FormalTarget")
    if not target.ordered_tasks:
        raise ValueError("FormalTarget.ordered_tasks 不能为空")
    if not target.feature_names:
        raise ValueError("FormalTarget.feature_names 不能为空")
    if not target.action_definitions:
        raise ValueError("FormalTarget.action_definitions 不能为空")
    return target
