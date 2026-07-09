"""Leaf-level audit 测试。

测试 _build_leaf_audit_fields 的输出字段，不启动完整 HOUT。
包含 summarizer 测试。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
sklearn = pytest.importorskip("sklearn.tree")
DecisionTreeClassifier = sklearn.DecisionTreeClassifier

from amc_py.viper.tree_policy import TreeBudgetPolicy
from amc_py.viper.metrics import _build_leaf_audit_fields


def _make_policy() -> TreeBudgetPolicy:
    x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.asarray([0, 1, 2], dtype=np.int64)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(x, y)
    return TreeBudgetPolicy(
        classifier=clf,
        metadata={"state_dim": 1, "action_dim": 3, "method": "bc", "tree_id": "t0"},
        feature_names=("f0",),
        action_definitions=[
            {"action_id": 0, "action_name": "noop"},
            {"action_id": 1, "action_name": "increase_0"},
            {"action_id": 2, "action_name": "decrease_0"},
        ],
    )


def test_build_leaf_audit_fields_basic() -> None:
    """_build_leaf_audit_fields 应返回基本 leaf audit 字段。"""

    policy = _make_policy()
    state_vector = (2.0,)
    _, tree_info = policy.select_action_id(
        state_vector, (True, True, True), include_decision_trace=True
    )

    fields = _build_leaf_audit_fields(
        step_index=0,
        state_vector=state_vector,
        feature_names=policy.feature_names,
        tree_policy=policy,
        tree_info=tree_info,
        selected_action_id=2,
        valid_action_mask=(True, True, True),
        teacher_diag=None,
        leaf_audit_state_mode="split",
        leaf_audit_top_k_actions=3,
    )

    assert "tree_leaf_id" in fields
    assert "tree_path_depth" in fields
    assert "tree_path_predicates_json" in fields
    assert "tree_path_node_ids_json" in fields
    assert "tree_leaf_n_node_samples" in fields
    assert "tree_leaf_impurity" in fields
    assert "tree_leaf_predicted_class_id" in fields
    assert "tree_raw_top1_action_id" in fields
    assert "tree_selected_action_id" in fields
    assert "tree_raw_top1_invalid" in fields
    assert "tree_fallback_used" in fields
    assert "tree_topk_action_ids_json" in fields
    assert "tree_valid_action_count" in fields
    assert "tree_masked_action_count" in fields

    # split 模式下应有 path feature values
    assert "tree_path_feature_values_json" in fields
    split_features = json.loads(str(fields["tree_path_feature_values_json"]))
    assert isinstance(split_features, dict)

    # all 模式
    fields_all = _build_leaf_audit_fields(
        step_index=0,
        state_vector=state_vector,
        feature_names=policy.feature_names,
        tree_policy=policy,
        tree_info=tree_info,
        selected_action_id=2,
        valid_action_mask=(True, True, True),
        teacher_diag=None,
        leaf_audit_state_mode="all",
        leaf_audit_top_k_actions=3,
    )
    assert "state_vector_json" in fields_all

    # none 模式
    fields_none = _build_leaf_audit_fields(
        step_index=0,
        state_vector=state_vector,
        feature_names=policy.feature_names,
        tree_policy=policy,
        tree_info=tree_info,
        selected_action_id=2,
        valid_action_mask=(True, True, True),
        teacher_diag=None,
        leaf_audit_state_mode="none",
        leaf_audit_top_k_actions=3,
    )
    assert "tree_path_feature_values_json" not in fields_none
    assert "state_vector_json" not in fields_none


def test_build_leaf_audit_fields_with_teacher() -> None:
    """有 teacher_diag 时，应输出 teacher 对比字段。"""

    policy = _make_policy()
    state_vector = (1.0,)
    _, tree_info = policy.select_action_id(
        state_vector, (True, True, True), include_decision_trace=True
    )

    teacher_diag = {
        "best_action_id": 1,
        "q_best": 10.0,
        "raw_q_values": [5.0, 10.0, 3.0],
    }

    fields = _build_leaf_audit_fields(
        step_index=0,
        state_vector=state_vector,
        feature_names=policy.feature_names,
        tree_policy=policy,
        tree_info=tree_info,
        selected_action_id=0,
        valid_action_mask=(True, True, True),
        teacher_diag=teacher_diag,
        leaf_audit_state_mode="split",
        leaf_audit_top_k_actions=3,
    )

    assert fields["teacher_best_action_id"] == 1
    assert fields["teacher_q_best"] == 10.0
    assert fields["teacher_q_selected"] == 5.0
    assert fields["teacher_q_regret_selected"] == 5.0
    assert fields["teacher_q_raw_top1"] is not None
    assert fields["teacher_selected_action_match"] is False
    assert "teacher_best_action_def_json" in fields
    assert "teacher_topk_action_ids_json" in fields
    assert "teacher_topk_q_values_json" in fields


def test_build_leaf_audit_fields_action_definitions() -> None:
    """raw_top1 和 selected 动作定义应被正确写出。"""

    policy = _make_policy()
    state_vector = (0.0,)
    _, tree_info = policy.select_action_id(
        state_vector, (True, True, False), include_decision_trace=True
    )

    fields = _build_leaf_audit_fields(
        step_index=0,
        state_vector=state_vector,
        feature_names=policy.feature_names,
        tree_policy=policy,
        tree_info=tree_info,
        selected_action_id=1,
        valid_action_mask=(True, True, False),
        teacher_diag=None,
        leaf_audit_state_mode="split",
        leaf_audit_top_k_actions=3,
    )

    raw_top1_def = json.loads(str(fields["tree_raw_top1_action_def_json"]))
    assert raw_top1_def["action_id"] == 0

    sel_def = json.loads(str(fields["tree_selected_action_def_json"]))
    assert sel_def["action_id"] == 1


def test_build_leaf_audit_fields_all_json_serializable() -> None:
    """所有 _build_leaf_audit_fields 的输出应可被 JSON 序列化（不能含 numpy 类型）。"""

    policy = _make_policy()
    state_vector = (0.0,)
    _, tree_info = policy.select_action_id(
        state_vector, (True, True, False), include_decision_trace=True
    )

    teacher_diag = {
        "best_action_id": 0,
        "q_best": 10.0,
        "raw_q_values": [10.0, 5.0, 3.0],
    }

    for mode in ("none", "split", "all"):
        fields = _build_leaf_audit_fields(
            step_index=0,
            state_vector=state_vector,
            feature_names=policy.feature_names,
            tree_policy=policy,
            tree_info=tree_info,
            selected_action_id=0,
            valid_action_mask=(True, True, False),
            teacher_diag=teacher_diag,
            leaf_audit_state_mode=mode,
            leaf_audit_top_k_actions=3,
        )
        # 所有字段应可被 json.dumps 序列化
        result = json.dumps(fields, ensure_ascii=False)
        assert isinstance(result, str)
        # 反序列化后应一致
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


def test_summarize_tree_leaf_audit_outputs_csv(tmp_path: Path) -> None:
    """summarize_tree_leaf_audit.py 应从临时 JSONL 生成所需的 CSV 文件。"""

    from scripts.summarize_tree_leaf_audit import (
        _build_leaf_summary_all,
        _build_leaf_teacher_disagreement,
        _build_leaf_fallback_summary,
        _build_leaf_high_regret_cases,
    )

    # 构造模拟 JSONL 数据
    audit_rows: list[dict[str, object]] = []
    for seed in [1, 2]:
        for step in range(5):
            leaf_id = 0 if step < 3 else 1
            audit_rows.append({
                "taskset_seed": 1,
                "seed": seed,
                "method": "viper_tree_agent",
                "tree_id": "t0",
                "tree_leaf_id": leaf_id,
                "tree_path_depth": 2,
                "tree_leaf_n_node_samples": 100,
                "tree_leaf_impurity": 0.1,
                "tree_raw_top1_action_id": 0,
                "tree_selected_action_id": 0,
                "tree_fallback_used": False,
                "tree_raw_top1_invalid": False,
                "teacher_selected_action_match": leaf_id == 0,
                "teacher_raw_action_match": True,
                "teacher_q_regret_selected": 0.5 if leaf_id == 1 else 0.0,
                "teacher_q_best": 10.0,
                "teacher_q_selected": 9.5 if leaf_id == 1 else 10.0,
                "teacher_best_action_id": 0,
                "reward": 0.1,
                "accepted": True,
                "delta_deadline_misses": 0,
                "delta_lo_cancellations": 0,
                "delta_mode_changes": 0,
                "tree_path_predicates_json": json.dumps(
                    [{"node_id": 0, "feature_index": 0, "feature_name": "f0", "threshold": 0.5, "value": 0.3, "operator": "<=", "decision": "left"}],
                    ensure_ascii=False,
                ),
                "tree_path_feature_values_json": json.dumps({"f0": 0.3}, ensure_ascii=False),
            })

    output_dir = tmp_path / "summary"
    output_dir.mkdir()

    _build_leaf_summary_all(audit_rows, output_dir, min_hit_count=1)
    _build_leaf_teacher_disagreement(audit_rows, output_dir, min_hit_count=1)
    _build_leaf_fallback_summary(audit_rows, output_dir, min_hit_count=1)
    _build_leaf_high_regret_cases(audit_rows, output_dir, min_hit_count=1)

    assert (output_dir / "leaf_summary_all.csv").exists()
    assert (output_dir / "leaf_teacher_disagreement.csv").exists()
    assert (output_dir / "leaf_fallback_summary.csv").exists()
    assert (output_dir / "leaf_high_regret_cases.csv").exists()

    # 验证 leaf_summary_all.csv 的基本内容
    with (output_dir / "leaf_summary_all.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2  # 两个叶子
    hit_counts = sum(int(r["hit_count"]) for r in rows)
    assert hit_counts == len(audit_rows)
    # 验证 taskset_seed 字段存在
    assert all(r["taskset_seed"] == "1" for r in rows)


def test_min_hit_count_filters_leaves() -> None:
    """min_hit_count 过滤后，hit_count 低于阈值的叶子不应出现在输出中。"""

    from scripts.summarize_tree_leaf_audit import (
        _build_leaf_summary_all,
        _build_leaf_fallback_summary,
    )

    rows: list[dict[str, object]] = []
    for step in range(10):
        leaf_id = 0  # 10 行
        rows.append({
            "taskset_seed": 1,
            "seed": 1, "method": "m", "tree_id": "t0",
            "tree_leaf_id": leaf_id,
            "tree_path_depth": 1, "tree_leaf_n_node_samples": 10,
            "tree_leaf_impurity": 0.1, "tree_raw_top1_action_id": 0,
            "tree_selected_action_id": 0, "tree_fallback_used": False,
            "tree_raw_top1_invalid": False, "teacher_selected_action_match": True,
            "teacher_q_regret_selected": 0.1, "reward": 0.1, "accepted": True,
            "delta_deadline_misses": 0, "delta_lo_cancellations": 0,
            "delta_mode_changes": 0, "teacher_q_best": 1.0, "teacher_q_selected": 0.9,
            "teacher_best_action_id": 0, "tree_path_predicates_json": "[]",
        })
    # 只加 1 条 leaf_id=1
    rows.append({
        "taskset_seed": 1,
        "seed": 1, "method": "m", "tree_id": "t0",
        "tree_leaf_id": 1,
        "tree_path_depth": 1, "tree_leaf_n_node_samples": 1,
        "tree_leaf_impurity": 0.5, "tree_raw_top1_action_id": 0,
        "tree_selected_action_id": 0, "tree_fallback_used": False,
        "tree_raw_top1_invalid": False, "teacher_selected_action_match": True,
        "teacher_q_regret_selected": None, "reward": 0.1, "accepted": True,
        "delta_deadline_misses": 0, "delta_lo_cancellations": 0,
        "delta_mode_changes": 0, "teacher_q_best": None, "teacher_q_selected": None,
        "teacher_best_action_id": 0, "tree_path_predicates_json": "[]",
    })

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        out = Path(td)
        _build_leaf_summary_all(rows, out, min_hit_count=2)
        import csv
        with (out / "leaf_summary_all.csv").open("r", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))
        leaf_ids = {int(r["tree_leaf_id"]) for r in summary_rows}
        assert leaf_ids == {0}  # leaf 1 被过滤掉


def test_summarizer_hit_rate_denominator_per_method_tree_seed() -> None:
    """跨 method 汇总时，hit_rate denominator 应按 (taskset_seed, method, tree_id, seed) 分别计算。"""

    from scripts.summarize_tree_leaf_audit import _build_leaf_summary_all

    # 构造 method_a seed=1: 10 行（leaf 0: 5, leaf 1: 5）
    # 构造 method_b seed=1: 10 行（全部 leaf 0: 10）
    rows: list[dict[str, object]] = []
    for _ in range(5):
        rows.append({
            "taskset_seed": 185,
            "seed": 1, "method": "method_a", "tree_id": "t0",
            "tree_leaf_id": 0,
            "tree_path_depth": 1, "tree_leaf_n_node_samples": 10,
            "tree_leaf_impurity": 0.1, "tree_raw_top1_action_id": 0,
            "tree_selected_action_id": 0, "tree_fallback_used": False,
            "tree_raw_top1_invalid": False, "teacher_selected_action_match": True,
            "teacher_q_regret_selected": 0.1, "reward": 0.1, "accepted": True,
            "delta_deadline_misses": 0, "delta_lo_cancellations": 0,
            "delta_mode_changes": 0, "teacher_q_best": 1.0, "teacher_q_selected": 0.9,
            "teacher_best_action_id": 0, "tree_path_predicates_json": "[]",
        })
    for _ in range(5):
        rows.append({
            "taskset_seed": 185,
            "seed": 1, "method": "method_a", "tree_id": "t0",
            "tree_leaf_id": 1,
            "tree_path_depth": 1, "tree_leaf_n_node_samples": 10,
            "tree_leaf_impurity": 0.1, "tree_raw_top1_action_id": 1,
            "tree_selected_action_id": 1, "tree_fallback_used": False,
            "tree_raw_top1_invalid": False, "teacher_selected_action_match": True,
            "teacher_q_regret_selected": 0.1, "reward": 0.1, "accepted": True,
            "delta_deadline_misses": 0, "delta_lo_cancellations": 0,
            "delta_mode_changes": 0, "teacher_q_best": 1.0, "teacher_q_selected": 0.9,
            "teacher_best_action_id": 1, "tree_path_predicates_json": "[]",
        })
    for _ in range(10):
        rows.append({
            "taskset_seed": 185,
            "seed": 1, "method": "method_b", "tree_id": "t0",
            "tree_leaf_id": 0,
            "tree_path_depth": 1, "tree_leaf_n_node_samples": 10,
            "tree_leaf_impurity": 0.1, "tree_raw_top1_action_id": 0,
            "tree_selected_action_id": 0, "tree_fallback_used": False,
            "tree_raw_top1_invalid": False, "teacher_selected_action_match": True,
            "teacher_q_regret_selected": 0.1, "reward": 0.1, "accepted": True,
            "delta_deadline_misses": 0, "delta_lo_cancellations": 0,
            "delta_mode_changes": 0, "teacher_q_best": 1.0, "teacher_q_selected": 0.9,
            "teacher_best_action_id": 0, "tree_path_predicates_json": "[]",
        })

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        out = Path(td)
        _build_leaf_summary_all(rows, out, min_hit_count=1)
        import csv
        with (out / "leaf_summary_all.csv").open("r", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))

        # method_a 的两个 leaf 都应有 5 hit，denominator 应为 method_a 的 10
        # method_b 的 leaf 0 应有 10 hit，denominator 应为 method_b 的 10
        for r in summary_rows:
            method = r["method"]
            hit_count = int(r["hit_count"])
            hit_rate = float(r["hit_rate_mean_per_seed"])
            if method == "method_a":
                # hit_rate = hit_count / 10 (去掉了分隔符)
                assert abs(hit_rate - hit_count / 10) < 0.01, (
                    f"method_a: hit_rate={hit_rate}, expected {hit_count}/10"
                )
            elif method == "method_b":
                assert abs(hit_rate - 1.0) < 0.01, (
                    f"method_b: hit_rate={hit_rate}, expected 1.0"
                )


def test_summarizer_groups_by_taskset_seed() -> None:
    """不同 taskset_seed 即使 method/tree_id/seed 相同，也不能混合 leaf 统计。"""

    from scripts.summarize_tree_leaf_audit import _build_leaf_summary_all

    rows: list[dict[str, object]] = []

    # taskset 185: seed 1550，共 10 行，全部命中 leaf 1
    for _ in range(10):
        rows.append({
            "taskset_seed": 185,
            "seed": 1550,
            "method": "viper_tree_agent",
            "tree_id": "viper_iter_002",
            "tree_leaf_id": 1,
            "tree_path_depth": 1,
            "tree_leaf_n_node_samples": 10,
            "tree_leaf_impurity": 0.1,
            "tree_raw_top1_action_id": 8,
            "tree_selected_action_id": 8,
            "tree_fallback_used": False,
            "tree_raw_top1_invalid": False,
            "teacher_selected_action_match": True,
            "teacher_raw_action_match": True,
            "teacher_q_regret_selected": 0.0,
            "reward": 0.0,
            "accepted": True,
            "delta_deadline_misses": 0,
            "delta_lo_cancellations": 0,
            "delta_mode_changes": 0,
            "tree_path_predicates_json": "[]",
        })

    # taskset 358: seed 1550，共 10 行，全部命中 leaf 2
    for _ in range(10):
        rows.append({
            "taskset_seed": 358,
            "seed": 1550,
            "method": "viper_tree_agent",
            "tree_id": "viper_iter_002",
            "tree_leaf_id": 2,
            "tree_path_depth": 1,
            "tree_leaf_n_node_samples": 10,
            "tree_leaf_impurity": 0.1,
            "tree_raw_top1_action_id": 9,
            "tree_selected_action_id": 9,
            "tree_fallback_used": False,
            "tree_raw_top1_invalid": False,
            "teacher_selected_action_match": True,
            "teacher_raw_action_match": True,
            "teacher_q_regret_selected": 0.0,
            "reward": 0.0,
            "accepted": True,
            "delta_deadline_misses": 0,
            "delta_lo_cancellations": 0,
            "delta_mode_changes": 0,
            "tree_path_predicates_json": "[]",
        })

    import tempfile
    from pathlib import Path
    import csv

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        _build_leaf_summary_all(rows, out, min_hit_count=1)
        with (out / "leaf_summary_all.csv").open("r", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))

    assert len(summary_rows) == 2
    by_key = {(int(r["taskset_seed"]), int(r["tree_leaf_id"])): r for r in summary_rows}
    assert (185, 1) in by_key
    assert (358, 2) in by_key
    # 每个 taskset 的 hit_rate 应为 1.0（各自 denominator 为 10）
    assert abs(float(by_key[(185, 1)]["hit_rate_mean_per_seed"]) - 1.0) < 1e-9
    assert abs(float(by_key[(358, 2)]["hit_rate_mean_per_seed"]) - 1.0) < 1e-9
