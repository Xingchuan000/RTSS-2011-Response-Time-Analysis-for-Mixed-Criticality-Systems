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
