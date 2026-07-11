"""VIPER 最小端到端 smoke 测试。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_viper_bc_e2e_smoke(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "KMP_DUPLICATE_LIB_OK": "TRUE"}
    teacher_dir = tmp_path / "teacher"
    dataset_dir = tmp_path / "dataset"
    trees_dir = tmp_path / "trees"
    eval_csv = tmp_path / "eval.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "small",
            "--episodes",
            "1",
            "--end-time",
            "80",
            "--agent-period",
            "20",
            "--observation-mode",
            "v11_full_10d",
            "--action-space",
            "single",
            "--validation-seeds",
            "",
            "--validate-every",
            "999",
            "--seed",
            "0",
            "--output-dir",
            str(teacher_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    teacher_model = teacher_dir / "model_final.pt"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_viper_teacher_data.py",
            "--model",
            str(teacher_model),
            "--teacher-id",
            "smoke_teacher",
            "--workload",
            "small",
            "--seeds",
            "0:1",
            "--end-time",
            "80",
            "--agent-period",
            "20",
            "--observation-mode",
            "v11_full_10d",
            "--action-space",
            "single",
            "--output-dir",
            str(dataset_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/train_viper_tree.py",
            "--method",
            "bc",
            "--teacher-model",
            str(teacher_model),
            "--teacher-id",
            "smoke_teacher",
            "--initial-dataset",
            str(dataset_dir),
            "--train-seeds",
            "0:1",
            "--validation-seeds",
            "2",
            "--iterations",
            "1",
            "--end-time",
            "80",
            "--validation-end-time",
            "80",
            "--agent-period",
            "20",
            "--workload",
            "small",
            "--observation-mode",
            "v11_full_10d",
            "--action-space",
            "single",
            "--max-depth-grid",
            "2",
            "--min-samples-leaf-grid",
            "1",
            "--output-dir",
            str(trees_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    run_config_path = trees_dir / "depth_2" / "leaf_1" / "run_config.json"
    best_dir = trees_dir / "depth_2" / "leaf_1" / "best"
    assert (best_dir / "model.joblib").exists()
    assert (best_dir / "metadata.json").exists()
    assert run_config_path.exists()
    with run_config_path.open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    assert run_config["method"] == "bc"
    assert run_config["iterations"] == 1
    assert run_config["teacher_id"] == "smoke_teacher"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(teacher_model),
            "--bc-tree-model",
            str(best_dir),
            "--tree-compare-teacher-model",
            str(teacher_model),
            "--baselines",
            "c_amc_sem_baseline,dqn_agent,bc_tree_agent",
            "--workload",
            "small",
            "--seeds",
            "0",
            "--end-time",
            "80",
            "--agent-period",
            "20",
            "--observation-mode",
            "v11_full_10d",
            "--action-space",
            "single",
            "--output",
            str(eval_csv),
            "--evaluation-workers",
            "1",
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    rows = _read_rows(eval_csv)
    methods = {row["method"] for row in rows}
    assert {"c_amc_sem_baseline", "dqn_agent", "bc_tree_agent"}.issubset(methods)
    tree_row = next(row for row in rows if row["method"] == "bc_tree_agent")
    assert "tree_selected_action_count" in tree_row
    assert "tree_raw_top1_invalid_rate" in tree_row
    assert "tree_fallback_rate" in tree_row
    assert "tree_q_regret_mean" in tree_row


def test_fixed_ranked_e2e_smoke(tmp_path: Path) -> None:
    """fixed-ranked 完整链：teacher-only dataset -> fixed-point CART -> ranked v2 artifact
    -> ranked validation rollout -> performance-compatible gate -> best/ -> HOUT。
    """
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "KMP_DUPLICATE_LIB_OK": "TRUE"}
    teacher_dir = tmp_path / "teacher"
    dataset_dir = tmp_path / "dataset"
    trees_dir = tmp_path / "trees"
    eval_csv = tmp_path / "eval_fixed_ranked.csv"

    # Step 1: 训练 teacher DQN
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload", "small",
            "--episodes", "1",
            "--end-time", "80",
            "--agent-period", "20",
            "--observation-mode", "v11_full_10d",
            "--action-space", "single",
            "--validation-seeds", "",
            "--validate-every", "999",
            "--seed", "0",
            "--output-dir", str(teacher_dir),
        ],
        check=True, cwd=PROJECT_ROOT, env=env,
    )
    teacher_model = teacher_dir / "model_final.pt"

    # Step 2: 采集 teacher-only dataset（ranked 模式）
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_viper_teacher_data.py",
            "--model", str(teacher_model),
            "--teacher-id", "fixed_ranked_e2e",
            "--workload", "small",
            "--seeds", "0:1",
            "--end-time", "80",
            "--agent-period", "20",
            "--observation-mode", "v11_full_10d",
            "--action-space", "single",
            "--fixed-ranked-deployment-v1",
            "--action-validation-mode", "formal_v1",
            "--strict-candidate-deploy-cap",
            "--carry-over-aware-safety",
            "--lo-budget-overrun-guard-units", "1",
            "--tree-state-encoding", "fixed_point_int",
            "--tree-fallback-mode", "ranked_valid_or_none",
            "--output-dir", str(dataset_dir),
        ],
        check=True, cwd=PROJECT_ROOT, env=env,
    )

    # Step 3: 训练 ranked v2 VIPER 树
    subprocess.run(
        [
            sys.executable,
            "scripts/train_viper_tree.py",
            "--method", "bc",
            "--teacher-model", str(teacher_model),
            "--teacher-id", "fixed_ranked_e2e",
            "--initial-dataset", str(dataset_dir),
            "--train-seeds", "0:1",
            "--validation-seeds", "2",
            "--iterations", "1",
            "--end-time", "80",
            "--validation-end-time", "80",
            "--agent-period", "20",
            "--workload", "small",
            "--observation-mode", "v11_full_10d",
            "--action-space", "single",
            "--max-depth-grid", "2",
            "--min-samples-leaf-grid", "1",
            "--fixed-ranked-deployment-v1",
            "--tree-state-encoding", "fixed_point_int",
            "--tree-fallback-mode", "ranked_valid_or_none",
            "--tree-selection-mode", "performance_compatible",
            "--action-validation-mode", "formal_v1",
            "--strict-candidate-deploy-cap",
            "--carry-over-aware-safety",
            "--lo-budget-overrun-guard-units", "1",
            "--require-integer-tree-artifact",
            "--output-dir", str(trees_dir),
        ],
        check=True, cwd=PROJECT_ROOT, env=env,
    )

    # 验证 best/ artifact 存在且为 ranked v2
    best_dir = trees_dir / "depth_2" / "leaf_1" / "best"
    assert best_dir.exists()
    assert (best_dir / "integer_tree.json").exists()
    assert (best_dir / "artifact_manifest.json").exists()
    with (best_dir / "metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["artifact_schema_version"] == "viper_integer_ranked_artifact_v2"
    assert meta["fallback_mode"] == "ranked_valid_or_none"
    assert meta["deployment_uses_sklearn"] is False

    # 使用已生成的 ranked integer artifact 构造固定的五次决策 mask，而不是依赖
    # 随机 teacher 恰好产生高 invalid rate。前两次屏蔽 top1、后三次允许 top1，
    # 因而 raw invalid/fallback 均为 2/5，且每一次 selected action 都合法。
    from amc_py.viper.artifacts import load_tree_policy_artifact

    ranked_policy = load_tree_policy_artifact(best_dir, require_integer_tree=True)
    controlled_state = tuple(0.0 for _ in range(int(meta["state_dim"])))
    ranking = ranked_policy.predict_action_ranking(controlled_state)
    selected_ranks: list[int] = []
    for block_top1 in (True, True, False, False, False):
        mask = tuple((action_id != ranking[0]) if block_top1 else True for action_id in range(len(ranking)))
        selected, info = ranked_policy.select_action_id(controlled_state, mask)
        assert selected is not None and mask[selected]
        assert info["tree_raw_top1_invalid"] is block_top1
        selected_ranks.append(int(info["tree_selected_rank"]))
    assert sum(rank > 0 for rank in selected_ranks) / len(selected_ranks) > 0.20
    assert max(selected_ranks) >= 1

    # 验证 candidates.csv 和 gate report 存在
    tree_run_dir = trees_dir / "depth_2" / "leaf_1"
    assert (tree_run_dir / "candidates.csv").exists()
    assert (tree_run_dir / "selection_gate_report.csv").exists()
    assert (tree_run_dir / "selection_gate_report.json").exists()

    # Step 4: HOUT 评估（使用 --require-integer-tree-artifact 和 --fixed-ranked-deployment-v1）
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model", str(teacher_model),
            "--bc-tree-model", str(best_dir),
            "--tree-compare-teacher-model", str(teacher_model),
            "--baselines", "bc_tree_agent",
            "--workload", "small",
            "--seeds", "0",
            "--end-time", "80",
            "--agent-period", "20",
            "--observation-mode", "v11_full_10d",
            "--action-space", "single",
            "--fixed-ranked-deployment-v1",
            "--require-integer-tree-artifact",
            "--tree-state-encoding", "fixed_point_int",
            "--tree-fallback-mode", "ranked_valid_or_none",
            "--action-validation-mode", "formal_v1",
            "--strict-candidate-deploy-cap",
            "--carry-over-aware-safety",
            "--lo-budget-overrun-guard-units", "1",
            "--output", str(eval_csv),
            "--evaluation-workers", "1",
        ],
        check=True, cwd=PROJECT_ROOT, env=env,
    )

    rows = _read_rows(eval_csv)
    methods = {row["method"] for row in rows}
    assert "bc_tree_agent" in methods
    tree_row = next(row for row in rows if row["method"] == "bc_tree_agent")
    assert "tree_selected_action_count" in tree_row
    assert "tree_raw_top1_invalid_rate" in tree_row
    assert "tree_ranked_fallback_rate" in tree_row
    assert "tree_no_valid_action_rate" in tree_row
    assert "tree_selected_rank_mean" in tree_row
    # HOUT 行必须同时保留 artifact 实际声明与本次 fixed-ranked profile 期望值。
    assert tree_row["artifact_tree_state_encoding"] == "fixed_point_int"
    assert tree_row["runtime_expected_tree_state_encoding"] == "fixed_point_int"
    assert tree_row["artifact_tree_fallback_mode"] == "ranked_valid_or_none"
    assert tree_row["runtime_expected_tree_fallback_mode"] == "ranked_valid_or_none"
    assert tree_row["artifact_deployment_semantics_version"] == "fixed_ranked_deployment_v1"
    assert tree_row["runtime_expected_deployment_semantics_version"] == "fixed_ranked_deployment_v1"
    assert tree_row["semantic_validation_passed"] == "True"
