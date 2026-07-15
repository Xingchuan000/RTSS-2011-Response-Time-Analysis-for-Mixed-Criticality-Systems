"""实现 proof object 使用的确定性 JSON 编码。

证明对象禁止浮点数，因为不同 Python/平台的浮点打印会破坏 hash 稳定性。
"""

from __future__ import annotations

import json
import math
from pathlib import PurePosixPath
from typing import Any


def _validate(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON 禁止 NaN 和 Inf")
        raise TypeError("proof object 禁止 JSON float，请使用整数或 canonical decimal string")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object key 必须是字符串")
            _validate(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate(item)


def canonical_dumps(value: Any) -> str:
    """返回 UTF-8/LF/升序 key/单末尾换行的 canonical JSON 文本。"""
    _validate(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def canonical_path(path: str) -> str:
    """把 Windows 或 Unix 路径转换为相对 POSIX 路径，并拒绝越界。"""
    normalized = path.replace("\\", "/")
    result = PurePosixPath(normalized)
    if result.is_absolute() or ".." in result.parts:
        raise ValueError(f"路径必须是 workspace-relative 且不可越界: {path}")
    return result.as_posix()
