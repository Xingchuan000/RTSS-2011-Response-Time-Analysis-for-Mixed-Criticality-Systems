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
    # 第一轮 synthetic 旧 target 暂时可以为 None，但真实 target 若没有
    # runtime adapter 必须在 preflight 阶段 UNRESOLVED，不能借用 synthetic。
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
    # 旧 synthetic fixture 没有显式新增 adapter 字段时，只允许按 target
    # 身份补接计划中规定的 synthetic wrapper。真实 target 绝不走这条分支，
    # 缺少真实 adapter 必须在其自身 preflight 中保持 UNRESOLVED。
    if target.runtime_adapter is None and (
        str(target.provenance.get("adapter_kind", "")).upper() == "SYNTHETIC_P0"
        or str(target.provenance.get("fixture", "")).startswith("synthetic_p0")
    ):
        from formal_toolchain.adapters.synthetic_runtime_adapter import SyntheticP0RuntimeAdapter
        object.__setattr__(target, "runtime_adapter", SyntheticP0RuntimeAdapter(target))
    return target
