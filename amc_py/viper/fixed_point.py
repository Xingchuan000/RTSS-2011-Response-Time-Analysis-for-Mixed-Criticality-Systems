"""VIPER student 使用的确定性定点编码。

该模块只处理显式输入，不读取环境状态，也不改变环境 observation 的生成方式。
量化采用非负输入上的 half-up 规则，避免 Python ``round`` 的 ties-to-even 行为。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import math
from collections.abc import Sequence

from amc_py.viper.schema import FIXED_POINT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FixedPointConfig:
    """定点编码的完整配置；字段写入 artifact 后不可隐式变更。"""

    scale: int = 1_000_000
    input_min: float = 0.0
    input_max: float = 1.0
    output_min: int = 0
    output_max: int = 1_000_000
    rounding_mode: str = "half_up_nonnegative"
    schema_version: str = FIXED_POINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, int) or self.scale <= 0:
            raise ValueError("fixed-point scale 必须大于 0")
        if isinstance(self.output_min, bool) or not isinstance(self.output_min, int):
            raise ValueError("fixed-point output_min 必须是 int")
        if isinstance(self.output_max, bool) or not isinstance(self.output_max, int):
            raise ValueError("fixed-point output_max 必须是 int")
        if self.input_min >= self.input_max:
            raise ValueError("fixed-point input_min 必须小于 input_max")
        if self.output_min > self.output_max:
            raise ValueError("fixed-point output_min 不能大于 output_max")
        if not isinstance(self.schema_version, str):
            raise ValueError("fixed-point schema_version 必须是 str")
        if not isinstance(self.rounding_mode, str):
            raise ValueError("fixed-point rounding_mode 必须是 str")
        if not math.isfinite(self.input_min) or not math.isfinite(self.input_max):
            raise ValueError("fixed-point input_min/input_max 必须是有限值")
        if self.rounding_mode == "half_up_nonnegative" and self.input_min < 0:
            raise ValueError("half_up_nonnegative 模式要求 input_min >= 0")
        if self.rounding_mode != "half_up_nonnegative":
            raise ValueError(f"不支持的 rounding_mode: {self.rounding_mode}")
        if self.output_min > 0 or self.output_max < 0:
            raise ValueError("输出范围必须能容纳量化后的零值")
        if abs(self.output_min) > 2**24:
            raise ValueError("output_min 不得超过 2**24，以保证 float32 临时转换精确")
        if self.output_max > 2**24:
            raise ValueError("output_max 不得超过 2**24，以保证 float32 临时转换精确")
        expected_max = self.input_max * self.scale
        if expected_max > self.output_max:
            raise ValueError("output_max 不能容纳 input_max 的量化结果")
        if self.input_min * self.scale < self.output_min:
            raise ValueError("output_min 不能容纳 input_min 的量化结果")


def quantize_value(value: float, config: FixedPointConfig) -> int:
    """将一个浮点值 clip 后按 half-up 量化为 Python ``int``。"""

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("不能量化 NaN 或 Inf")
    clipped = min(max(numeric, config.input_min), config.input_max)
    try:
        scaled = Decimal(str(clipped)) * Decimal(config.scale)
        quantized = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("定点量化失败") from exc
    return int(min(max(quantized, config.output_min), config.output_max))


def quantize_state_vector(values: Sequence[float], config: FixedPointConfig) -> tuple[int, ...]:
    """按固定顺序量化 observation，并保持向量长度。"""

    return tuple(quantize_value(value, config) for value in values)


def dequantize_value(value: int, config: FixedPointConfig) -> float:
    """将整数编码恢复为浮点值；输入必须是整数而非 bool。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("定点反量化输入必须是 int")
    if value < config.output_min or value > config.output_max:
        raise ValueError("定点整数超出配置输出范围")
    return float(Decimal(value) / Decimal(config.scale))


def fixed_point_config_to_dict(config: FixedPointConfig) -> dict[str, object]:
    """返回字段完整、可 JSON 序列化的配置字典。"""

    return dict(asdict(config))


def fixed_point_config_from_dict(data: dict[str, object]) -> FixedPointConfig:
    """严格按配置字段构造配置，未知 schema 直接拒绝。"""

    config = FixedPointConfig(**data)  # type: ignore[arg-type]
    if config.schema_version != FIXED_POINT_SCHEMA_VERSION:
        raise ValueError(f"未知 fixed-point schema: {config.schema_version}")
    return config


def fixed_point_config_hash(config: FixedPointConfig) -> str:
    """对排序字段后的规范 JSON 计算 SHA-256。"""

    payload = json.dumps(
        fixed_point_config_to_dict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
