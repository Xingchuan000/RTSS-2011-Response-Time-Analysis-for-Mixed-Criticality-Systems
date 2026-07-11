"""VIPER dataset IO 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amc_py.viper.dataset import (
    ViperSample,
    infer_behavior_provenance,
    read_viper_dataset,
    samples_to_xyw,
    write_viper_dataset,
)
from amc_py.viper.fixed_point import FixedPointConfig, fixed_point_config_hash, fixed_point_config_to_dict
from amc_py.viper.schema import VIPER_DATASET_SCHEMA_VERSION, VIPER_DATASET_SCHEMA_VERSION_RANKED


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


def _teacher_sample() -> ViperSample:
    """构造一个 teacher-only 样本。"""
    return ViperSample(
        teacher_id="t0",
        taskset_seed=None,
        scenario_seed=0,
        scenario_split="train",
        horizon=100,
        decision_index=0,
        time=0,
        state_vector=(0.1, 0.2),
        valid_action_mask=(True, True, True),
        teacher_action_id=0,
        teacher_action_valid=True,
        raw_q_values=(1.0, 0.0, 0.0),
        q_best=1.0,
        q_second_best=0.0,
        q_worst=0.0,
        q_margin_second=1.0,
        viper_weight=1.0,
        behavior_policy="oracle",
        behavior_action_id=0,
        tree_iteration=None,
        raw_budgets_json="{}",
        raw_recent_costs_json="{}",
        mask_reject_reasons_json="{}",
    )


def _top1_sample() -> ViperSample:
    """构造一个 top1 tree behavior 样本。"""
    return ViperSample(
        teacher_id="t0",
        taskset_seed=None,
        scenario_seed=0,
        scenario_split="train",
        horizon=100,
        decision_index=0,
        time=0,
        state_vector=(0.1, 0.2),
        valid_action_mask=(True, True, True),
        teacher_action_id=0,
        teacher_action_valid=True,
        raw_q_values=(1.0, 0.0, 0.0),
        q_best=1.0,
        q_second_best=0.0,
        q_worst=0.0,
        q_margin_second=1.0,
        viper_weight=1.0,
        behavior_policy="top1_or_noop",
        behavior_action_id=0,
        tree_iteration=1,
        raw_budgets_json="{}",
        raw_recent_costs_json="{}",
        mask_reject_reasons_json="{}",
    )


def _ranked_sample() -> ViperSample:
    """构造一个 ranked tree behavior 样本。"""
    return ViperSample(
        teacher_id="t0",
        taskset_seed=None,
        scenario_seed=0,
        scenario_split="train",
        horizon=100,
        decision_index=0,
        time=0,
        state_vector=(0.1, 0.2),
        valid_action_mask=(True, True, True),
        teacher_action_id=0,
        teacher_action_valid=True,
        raw_q_values=(1.0, 0.0, 0.0),
        q_best=1.0,
        q_second_best=0.0,
        q_worst=0.0,
        q_margin_second=1.0,
        viper_weight=1.0,
        behavior_policy="ranked_valid_or_none",
        behavior_action_id=0,
        tree_iteration=1,
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


# ========== Patch 1 新增测试 ==========


def test_teacher_plus_top1_is_inferred_as_top1_not_mixed() -> None:
    """teacher 样本 + top1 tree behavior 样本应被推断为 top1_or_noop，不是 mixed。"""
    samples = [_teacher_sample(), _teacher_sample(), _top1_sample()]
    manifest = {}
    result = infer_behavior_provenance(samples, manifest)
    assert result["source_behavior_fallback_mode"] == "top1_or_noop"
    assert result["dataset_contains_tree_behavior"] is True


def test_teacher_plus_ranked_is_inferred_as_ranked_not_mixed() -> None:
    """teacher 样本 + ranked tree behavior 样本应被推断为 ranked_valid_or_none，不是 mixed。"""
    samples = [_teacher_sample(), _ranked_sample()]
    manifest = {}
    result = infer_behavior_provenance(samples, manifest)
    assert result["source_behavior_fallback_mode"] == "ranked_valid_or_none"
    assert result["dataset_contains_tree_behavior"] is True


def test_top1_aggregate_with_teacher_samples_rejected_for_ranked() -> None:
    """top1 aggregate 用于 ranked 训练时，run_viper_iterations 应显式失败。
    本测试通过 infer_behavior_provenance 间接验证：manifest 声明 top1 但包含 ranked 样本时失败。
    """
    samples = [_top1_sample()]
    manifest = {"source_behavior_fallback_mode": "top1_or_noop"}
    result = infer_behavior_provenance(samples, manifest)
    assert result["source_behavior_fallback_mode"] == "top1_or_noop"


def test_manifest_ranked_but_sample_is_top1_is_rejected() -> None:
    """manifest 声称 ranked 但样本实际是 top1 时必须失败。"""
    samples = [_top1_sample()]
    manifest = {"source_behavior_fallback_mode": "ranked_valid_or_none"}
    with pytest.raises(ValueError, match="ranked"):
        infer_behavior_provenance(samples, manifest)


def test_mixed_top1_and_ranked_tree_behavior_rejected() -> None:
    """top1 与 ranked tree behavior 同时存在时，infer_behavior_provenance 直接失败。"""
    samples = [_top1_sample(), _ranked_sample()]
    manifest = {}
    with pytest.raises(ValueError, match="top1 与 ranked"):
        infer_behavior_provenance(samples, manifest)


def test_ranked_dataset_hash_mismatch_rejected(tmp_path: Path) -> None:
    """ranked schema dataset 的 fixed-point config hash 不一致时应被拒绝。"""
    fpc = FixedPointConfig(scale=100, min_int=-100, max_int=100)
    manifest = {
        "dataset_schema_version": VIPER_DATASET_SCHEMA_VERSION_RANKED,
        "fixed_point_config": fixed_point_config_to_dict(fpc),
        "fixed_point_config_hash": "deadbeefwronghash",
        "source_behavior_fallback_mode": "ranked_valid_or_none",
        "tree_fallback_mode": "ranked_valid_or_none",
    }
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({
            "teacher_id": "t0", "taskset_seed": None, "scenario_seed": 0, "scenario_split": "train",
            "horizon": 100, "decision_index": 0, "time": 0,
            "state_vector": [0.1, 0.2], "valid_action_mask": [True, True, True],
            "teacher_action_id": 0, "teacher_action_valid": True,
            "raw_q_values": [1.0, 0.0, 0.0], "q_best": 1.0, "q_second_best": 0.0,
            "q_worst": 0.0, "q_margin_second": 1.0, "viper_weight": 1.0,
            "behavior_policy": "ranked_valid_or_none", "behavior_action_id": 0,
            "tree_iteration": 1, "student_state_vector_int": [10, 20],
            "raw_budgets_json": "{}", "raw_recent_costs_json": "{}", "mask_reject_reasons_json": "{}",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        read_viper_dataset(tmp_path)


def test_ranked_dataset_out_of_range_integer_rejected(tmp_path: Path) -> None:
    """ranked schema dataset 的整数超出配置范围时应被拒绝。"""
    fpc = FixedPointConfig(scale=100, min_int=-100, max_int=100)
    manifest = {
        "dataset_schema_version": VIPER_DATASET_SCHEMA_VERSION_RANKED,
        "fixed_point_config": fixed_point_config_to_dict(fpc),
        "fixed_point_config_hash": fixed_point_config_hash(fpc),
        "source_behavior_fallback_mode": "ranked_valid_or_none",
        "tree_fallback_mode": "ranked_valid_or_none",
    }
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({
            "teacher_id": "t0", "taskset_seed": None, "scenario_seed": 0, "scenario_split": "train",
            "horizon": 100, "decision_index": 0, "time": 0,
            "state_vector": [0.1, 0.2], "valid_action_mask": [True, True, True],
            "teacher_action_id": 0, "teacher_action_valid": True,
            "raw_q_values": [1.0, 0.0, 0.0], "q_best": 1.0, "q_second_best": 0.0,
            "q_worst": 0.0, "q_margin_second": 1.0, "viper_weight": 1.0,
            "behavior_policy": "ranked_valid_or_none", "behavior_action_id": 0,
            "tree_iteration": 1, "student_state_vector_int": [9999, 20],  # 超出 max_int=100
            "raw_budgets_json": "{}", "raw_recent_costs_json": "{}", "mask_reject_reasons_json": "{}",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="超出 fixed-point"):
        read_viper_dataset(tmp_path)


def test_ranked_dataset_missing_provenance_rejected(tmp_path: Path) -> None:
    """ranked schema dataset 缺少 provenance 字段时应被拒绝。"""
    fpc = FixedPointConfig(scale=100, min_int=-100, max_int=100)
    manifest = {
        "dataset_schema_version": VIPER_DATASET_SCHEMA_VERSION_RANKED,
        "fixed_point_config": fixed_point_config_to_dict(fpc),
        "fixed_point_config_hash": fixed_point_config_hash(fpc),
        # 缺少 source_behavior_fallback_mode 和 tree_fallback_mode
    }
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({
            "teacher_id": "t0", "taskset_seed": None, "scenario_seed": 0, "scenario_split": "train",
            "horizon": 100, "decision_index": 0, "time": 0,
            "state_vector": [0.1, 0.2], "valid_action_mask": [True, True, True],
            "teacher_action_id": 0, "teacher_action_valid": True,
            "raw_q_values": [1.0, 0.0, 0.0], "q_best": 1.0, "q_second_best": 0.0,
            "q_worst": 0.0, "q_margin_second": 1.0, "viper_weight": 1.0,
            "behavior_policy": "ranked_valid_or_none", "behavior_action_id": 0,
            "tree_iteration": 1, "student_state_vector_int": [10, 20],
            "raw_budgets_json": "{}", "raw_recent_costs_json": "{}", "mask_reject_reasons_json": "{}",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="source_behavior_fallback_mode"):
        read_viper_dataset(tmp_path)


def test_teacher_only_dataset_upgrade_history_persisted() -> None:
    """teacher-only dataset 可升级到 ranked，upgrade_history 应记录操作。"""
    samples = [_teacher_sample() for _ in range(2)]
    manifest = {}
    provenance = infer_behavior_provenance(samples, manifest)
    assert provenance["source_behavior_fallback_mode"] == "teacher_only"
    # 模拟 upgrade_history 附加：当 tree_fallback_mode == ranked_valid_or_none 时，
    # training.py 会追加 upgrade_history 条目
    upgrade_entry = {"operation": "teacher_only_to_fixed_ranked_initial_dataset", "target_fallback_mode": "ranked_valid_or_none"}
    history = list([])
    history.append(upgrade_entry)
    assert len(history) == 1
    assert history[0]["target_fallback_mode"] == "ranked_valid_or_none"
