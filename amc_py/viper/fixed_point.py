"""VIPER student observation 的确定性定点编码。

编码只支持非负、半单位向上取整的稳定语义。输入先裁剪到配置范围，
因此范围外输入不会产生超出部署域的整数；NaN/Inf 则代表调用方状态错误，
必须立即报错。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import math
from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class FixedPointConfig:
    scale: int = 1_000_000
    min_int: int = 0
    max_int: int = 1_000_000
    rounding_mode: str = "half_up_nonnegative"
    input_min: float = 0.0
    input_max: float = 1.0
    schema_version: str = "fixed_point_v1"

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, int) or self.scale <= 0:
            raise ValueError("scale 必须是正整数")
        if isinstance(self.min_int, bool) or isinstance(self.max_int, bool):
            raise ValueError("min_int/max_int 必须是整数")
        if self.min_int > self.max_int or self.max_int < 0:
            raise ValueError("定点整数范围非法")
        if self.max_int != self.scale:
            raise ValueError("当前 fixed-point 语义要求 max_int 与 scale 一致")
        if self.rounding_mode != "half_up_nonnegative":
            raise ValueError("不支持的 rounding_mode")
        if not math.isfinite(self.input_min) or not math.isfinite(self.input_max):
            raise ValueError("输入范围必须有限")
        if self.input_min > self.input_max:
            raise ValueError("输入范围非法")


def quantize_unit_float(value: float, config: FixedPointConfig) -> int:
    """把一个值确定性量化为整数。

    边界行为固定为：先 clip 到 ``[input_min,input_max]``，再以 Decimal
    的 ``ROUND_HALF_UP`` 计算；在默认配置下 0.0 -> 0、1.0 -> 1_000_000。
    不使用 Python ``round`` 或 NumPy 的平台相关舍入默认值。
    """
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("定点量化输入不能是 NaN 或 Inf")
    clipped = min(max(numeric, config.input_min), config.input_max)
    try:
        scaled = Decimal(str(clipped)) * Decimal(config.scale)
        result = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("无法确定性量化输入") from exc
    return int(min(max(result, config.min_int), config.max_int))


def quantize_state_vector(values: Sequence[float], config: FixedPointConfig) -> tuple[int, ...]:
    return tuple(quantize_unit_float(value, config) for value in values)


def dequantize_value(value: int, config: FixedPointConfig) -> float:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("定点值必须是 Python int")
    if not config.min_int <= value <= config.max_int:
        raise ValueError("定点值超出配置范围")
    return float(Decimal(value) / Decimal(config.scale))


def fixed_point_config_to_dict(config: FixedPointConfig) -> dict[str, object]:
    return asdict(config)


def fixed_point_config_from_dict(data: dict[str, object]) -> FixedPointConfig:
    return FixedPointConfig(**data)  # type: ignore[arg-type]


def fixed_point_config_hash(config: FixedPointConfig) -> str:
    canonical = json.dumps(fixed_point_config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

