"""Phase G01：与生产量化器隔离的 fixed-point 重放。"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Any, Callable, Iterable


def replay_quantize(value: float, config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """严格执行 P0 固定顺序：finite、clip、str(float)、Decimal、half-up、clamp。"""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("不能量化 NaN 或 Inf")
    clipped = min(max(numeric, float(config["input_min"])), float(config["input_max"]))
    scaled = Decimal(str(clipped)) * Decimal(int(config["scale"]))
    rounded = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    result = min(max(rounded, int(config["output_min"])), int(config["output_max"]))
    return int(result), {"input": numeric, "input_hex": numeric.hex(), "clipped": str(clipped),
                         "scaled": str(scaled), "rounded": rounded, "output": int(result)}


def replay_quantization_vectors(values: Iterable[float], config: dict[str, Any]) -> dict[str, Any]:
    traces = []
    for value in values:
        result, trace = replay_quantize(value, config)
        traces.append({"value": value, "result": result, "trace": trace})
    if len(traces) < 10_000:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "QUANTIZATION_SAMPLE_COUNT_TOO_SMALL", "vectors": len(traces)}}
    return {"status": "PASS", "schema_version": "quantization_replay_v1",
            "vectors": len(traces), "traces": traces, "independent": True}


def deterministic_samples(count: int = 10_000) -> tuple[float, ...]:
    """生成固定、可重放且覆盖 clip/边界的 binary64 输入集合。"""
    if count < 10_000:
        raise ValueError("Phase G01 至少需要 10,000 个 deterministic samples")
    seeds = (0.0, 1.0, -1.0, 2.0, 0.5, 0.5000000000000001,
             0.49999999999999994, float("nan"), float("inf"), -float("inf"))
    values = list(seeds)
    for index in range(count - len(values)):
        values.append(((index * 2654435761) % 2_000_003 - 1_000_001) / 1_000_001.0)
    return tuple(values)


def verify_against_production(values: Iterable[float], config: dict[str, Any],
                              production_quantizer: Callable[[float, Any], int]) -> dict[str, Any]:
    """独立重放并与显式传入的 production callable 对照；不从本模块导入生产实现。"""
    checked = 0
    for value in values:
        try:
            expected, _ = replay_quantize(value, config)
            actual = production_quantizer(value, config)
        except (ValueError, TypeError):
            try:
                production_quantizer(value, config)
            except (ValueError, TypeError):
                checked += 1
                continue
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "QUANTIZATION_REJECTION_MISMATCH", "value": repr(value)}}
        if int(actual) != expected:
            return {"status": "FAIL", "route": "POLICY_CONTRACT_VIOLATION",
                    "failure": {"code": "QUANTIZATION_VALUE_MISMATCH", "value": repr(value),
                                "expected": expected, "actual": actual}}
        checked += 1
    if checked < 10_000:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED",
                "failure": {"code": "QUANTIZATION_SAMPLE_COUNT_TOO_SMALL", "checked": checked}}
    return {"status": "PASS", "schema_version": "quantization_production_differential_v1",
            "checked": checked, "independent": True}
