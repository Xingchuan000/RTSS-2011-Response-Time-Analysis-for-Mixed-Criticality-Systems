import math

import pytest

from amc_py.viper.fixed_point import (
    FixedPointConfig,
    fixed_point_config_from_dict,
    fixed_point_config_hash,
    fixed_point_config_to_dict,
    quantize_state_vector,
    quantize_value,
)


def test_default_quantization_and_half_up_are_deterministic() -> None:
    config = FixedPointConfig()
    assert quantize_value(0.0, config) == 0
    assert quantize_value(1.0, config) == 1_000_000
    assert quantize_value(0.5, config) == 500_000
    assert quantize_value(0.0000005, config) == 1
    assert quantize_state_vector((0.0, 0.5, 1.0), config) == (0, 500_000, 1_000_000)
    assert quantize_state_vector((0.25, 0.25), config) == quantize_state_vector((0.25, 0.25), config)


def test_quantization_clips_and_rejects_nonfinite_values() -> None:
    config = FixedPointConfig()
    assert quantize_value(-1.0, config) == 0
    assert quantize_value(2.0, config) == 1_000_000
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            quantize_value(value, config)


def test_config_round_trip_and_hash() -> None:
    config = FixedPointConfig()
    restored = fixed_point_config_from_dict(fixed_point_config_to_dict(config))
    assert restored == config
    assert fixed_point_config_hash(restored) == fixed_point_config_hash(config)
    assert config.output_max <= 2**24


def test_config_rejects_output_min_beyond_float32_safe_range() -> None:
    with pytest.raises(ValueError, match="output_min 不得超过 2\\*\\*24"):
        FixedPointConfig(output_min=-(2**24 + 1), output_max=2**24 + 2)
