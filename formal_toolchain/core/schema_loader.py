"""加载并校验 JSON Schema；不把校验失败降级为警告。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"schema 必须是 JSON object: {path}")
    return data


def validate(instance: Any, schema: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - 安装 extras 后走正常路径
        raise RuntimeError("需要安装 formal extra 才能执行 JSON Schema 校验") from exc
    jsonschema.Draft202012Validator(schema).validate(instance)
