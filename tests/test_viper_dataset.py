"""VIPER dataset IO 测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from amc_py.viper.fixed_point import FixedPointConfig
from amc_py.viper.dataset import ViperSample, read_viper_dataset, samples_to_xyw, write_viper_dataset


def _sample() -> ViperSample:
    return ViperSample(
        teacher_id="t0",
        taskset_seed=None,
        scenario_seed=0,
        scenario_split="train",
        horizon=100,
        decision_index=0,
        time=0,
        state_vector=(0.1, 0.2),
        valid_action_mask=(True, False, True),
        teacher_action_id=2,
        teacher_action_valid=True,
        raw_q_values=(1.0, 0.0, 2.0),
        q_best=2.0,
        q_second_best=1.0,
        q_worst=1.0,
        q_margin_second=1.0,
        viper_weight=1.0,
        behavior_policy="oracle",
        behavior_action_id=2,
        tree_iteration=None,
        raw_budgets_json="{}",
        raw_recent_costs_json="{}",
        mask_reject_reasons_json="{}",
    )


def test_dataset_roundtrip_and_xyw(tmp_path: Path) -> None:
    write_viper_dataset(tmp_path, [_sample()], {"dataset_id": "d0"})
    samples, manifest = read_viper_dataset(tmp_path)
    assert manifest["dataset_id"] == "d0"
    assert samples[0].state_vector == (0.1, 0.2)
    x, y, w = samples_to_xyw(samples, weight_mode="viper_q_span")
    assert x.shape == (1, 2)
    assert y.tolist() == [2]
    assert float(w[0]) == 1.0


def test_fixed_point_dataset_rejects_non_int_student_state() -> None:
    sample = _sample()
    invalid = replace(sample, student_state_vector_int=(1, True))
    with pytest.raises(ValueError, match="student_state_vector_int 必须全部是 int"):
        samples_to_xyw(
            [invalid],
            weight_mode="uniform",
            student_encoding="fixed_point_int",
            fixed_point_config=FixedPointConfig(scale=1, output_max=10),
        )
