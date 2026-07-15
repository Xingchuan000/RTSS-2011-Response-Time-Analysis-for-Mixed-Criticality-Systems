"""对真实 fixed_point.py 做定点量化绑定和独立重放。

这里不调用 production ``quantize_value``。绑定器只从源码抽取合同文字，
重放器使用 Decimal 重新执行 clip、str、scale 和 half-up，避免同源调用把
生产实现的错误再次当成正确答案。
"""

from __future__ import annotations

import ast
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
from typing import Any


def independent_quantize(value: float, config: dict[str, Any]) -> tuple[int, dict[str, str]]:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("不能量化 NaN 或 Inf")
    clipped = min(max(numeric, float(config["input_min"])), float(config["input_max"]))
    decimal_value = Decimal(str(clipped))
    scaled = decimal_value * Decimal(int(config["scale"]))
    rounded = scaled.to_integral_value(rounding=ROUND_HALF_UP)
    result = int(min(max(int(rounded), int(config["output_min"])), int(config["output_max"])))
    return result, {"float": repr(numeric), "float_hex": numeric.hex(),
                    "string": str(clipped), "decimal": str(decimal_value),
                    "scaled": str(scaled), "integer": str(result)}


def _literal_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("config", data)
    if not isinstance(config, dict):
        raise ValueError("fixed_point_config 必须是 object")
    required = {"scale", "input_min", "input_max", "output_min", "output_max", "rounding_mode"}
    if not required <= set(config):
        raise ValueError("fixed_point_config 缺少量化字段")
    return config


def bind_quantization_runtime(source_root: Path, config_artifact: Path) -> dict[str, Any]:
    source_path = Path(source_root) / "amc_py/viper/fixed_point.py"
    try:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == "quantize_value"), None)
        config = _literal_config(Path(config_artifact))
    except (OSError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "UNRESOLVED", "failure": {"code": "QUANTIZATION_INPUT_INVALID",
                "route": "UNRESOLVED", "detail": str(exc)}}
    if function is None:
        return {"status": "UNRESOLVED", "failure": {"code": "QUANTIZE_FUNCTION_NOT_FOUND",
                "route": "UNRESOLVED"}}
    required_tokens = ("math.isfinite", "min(max", "Decimal(str", "ROUND_HALF_UP",
                      "config.scale", "config.output_min", "config.output_max")
    missing = [token for token in required_tokens if token not in source]
    if missing or config.get("rounding_mode") != "half_up_nonnegative":
        return {"status": "FAIL", "failure": {"code": "QUANTIZATION_SEMANTICS_MISMATCH",
                "route": "POLICY_CONTRACT_VIOLATION", "missing": missing}}
    vectors = [0.0, 1.0, 0.5, 0.5005, 0.4995, 0.0005, 0.0015,
               0.123456789, 0.9999999, -1.0, 2.0]
    vectors += [i / 37.0 for i in range(9)]
    traces = []
    for value in vectors:
        result, trace = independent_quantize(value, config)
        traces.append({"input": value, "result": result, "trace": trace})
    return {"status": "PASS", "function": "quantize_value", "vectors": len(vectors),
            "config": config, "replay": "independent_decimal_half_up", "traces": traces}
