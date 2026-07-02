"""VIPER split 校验测试。"""

from __future__ import annotations

import pytest

from amc_py.viper.splits import parse_seed_spec, validate_viper_split_config


def test_parse_seed_spec_supports_ranges_and_csv() -> None:
    assert parse_seed_spec("1:3,8") == {1, 2, 3, 8}


def test_validate_viper_split_config_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        validate_viper_split_config(
            {
                "dqn_training_episode_seeds": "0:10",
                "viper_train_seeds": "10:20",
                "viper_validation_seeds": "30:40",
                "strict_final_hout_seeds": "50:60",
            }
        )


def test_validate_viper_split_config_reports_legacy_warnings_without_failing() -> None:
    result = validate_viper_split_config(
        {
            "dqn_training_episode_seeds": "0:1349",
            "old_validation_seeds": "200:209",
            "legacy_hout_seeds": "200:249",
            "viper_train_seeds": "1350:1399",
            "viper_validation_seeds": "1400:1419",
            "strict_final_hout_seeds": "1500:1549",
        }
    )
    warnings = result["warnings"]
    assert len(warnings) >= 3
    assert "legacy_only: old_validation_seeds 与 legacy_hout_seeds 有重叠" in warnings
    assert "legacy_only: old_validation_seeds 与 dqn_training_episode_seeds 有重叠" in warnings
    assert "legacy_only: legacy_hout_seeds 与 dqn_training_episode_seeds 有重叠" in warnings
