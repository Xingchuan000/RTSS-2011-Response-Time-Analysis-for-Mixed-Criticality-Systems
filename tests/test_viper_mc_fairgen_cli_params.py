"""VIPER mc_fairgen CLI 参数透传与一致性校验测试。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """统一执行 CLI，便于测试里复用相同的 cwd / env 配置。"""

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _train_smoke_teacher(output_dir: Path, *, env: dict[str, str]) -> Path:
    """训练一个最小 smoke teacher，供后续 VIPER CLI 测试复用。"""

    _run(
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
            str(output_dir),
        ],
        env=env,
    )
    return output_dir / "model_final.pt"


def _mc_fairgen_args() -> list[str]:
    """返回计划文档要求重点覆盖的非默认 mc_fairgen 参数。"""

    return [
        "--workload",
        "mc_fairgen",
        "--scenario",
        "stress",
        "--mc-fairgen-mode",
        "paper_learnable_headroom",
        "--mc-fairgen-num-tasks",
        "8",
        "--mc-fairgen-hi-ratio",
        "0.5",
        "--mc-fairgen-period-source",
        "controlled_medium",
        "--mc-fairgen-period-scale",
        "500",
        "--mc-fairgen-u-lo-lo-min",
        "0.25",
        "--mc-fairgen-u-lo-lo-max",
        "0.45",
        "--mc-fairgen-lo-budget-rho-min",
        "0.20",
        "--mc-fairgen-lo-budget-rho-max",
        "0.40",
        "--mc-fairgen-lo-overrun-prob",
        "0.12",
        "--mc-fairgen-lo-overrun-factor-min",
        "1.02",
        "--mc-fairgen-lo-overrun-factor-max",
        "1.25",
        "--fixed-taskset-seed",
        "2535",
        "--scenario-seed-offset",
        "100000",
    ]


def _assert_mc_fairgen_config(config: dict[str, object]) -> None:
    """断言落盘配置里包含计划要求的关键非默认参数。"""

    assert config["workload"] == "mc_fairgen"
    assert config["scenario"] == "stress"
    assert config["mc_fairgen_num_tasks"] == 8
    assert config["mc_fairgen_period_source"] == "controlled_medium"
    assert config["mc_fairgen_period_scale"] == 500
    assert config["mc_fairgen_u_lo_lo_min"] == 0.25
    assert config["mc_fairgen_u_lo_lo_max"] == 0.45
    assert config["mc_fairgen_lo_budget_rho_min"] == 0.20
    assert config["mc_fairgen_lo_budget_rho_max"] == 0.40
    assert config["mc_fairgen_lo_overrun_prob"] == 0.12
    assert config["mc_fairgen_lo_overrun_factor_min"] == 1.02
    assert config["mc_fairgen_lo_overrun_factor_max"] == 1.25
    assert config["fixed_taskset_seed"] == 2535
    assert config["scenario_seed_offset"] == 100000


def test_collect_viper_teacher_data_records_full_mc_fairgen_args(tmp_path: Path) -> None:
    """collector 应把完整 mc_fairgen CLI 参数写入 dataset manifest。"""

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "KMP_DUPLICATE_LIB_OK": "TRUE"}
    teacher_model = _train_smoke_teacher(tmp_path / "teacher", env=env)
    dataset_dir = tmp_path / "dataset"

    _run(
        [
            sys.executable,
            "scripts/collect_viper_teacher_data.py",
            "--model",
            str(teacher_model),
            "--teacher-id",
            "smoke_teacher",
            "--output-dir",
            str(dataset_dir),
            "--seeds",
            "0",
            "--end-time",
            "20000",
            "--agent-period",
            "10000",
            "--dqn-runtime-semantics",
            "C_AMC_SEM",
            "--c-amc-sem-xf",
            "0.5",
            "--reward-mode",
            "mendes",
            "--action-space",
            "single",
            "--observation-mode",
            "v11_full_10d",
            *_mc_fairgen_args(),
        ],
        env=env,
    )

    with (dataset_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _assert_mc_fairgen_config(manifest["workload_cli_config"])


def test_train_viper_tree_records_mc_fairgen_args_and_preflight_mismatch(tmp_path: Path) -> None:
    """tree 训练应记录 workload 参数，并对 dataset mismatch 执行 preflight 校验。"""

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "KMP_DUPLICATE_LIB_OK": "TRUE"}
    teacher_model = _train_smoke_teacher(tmp_path / "teacher", env=env)
    dataset_dir = tmp_path / "dataset"
    trees_dir = tmp_path / "trees"
    mismatch_dir = tmp_path / "trees_mismatch_allowed"

    _run(
        [
            sys.executable,
            "scripts/collect_viper_teacher_data.py",
            "--model",
            str(teacher_model),
            "--teacher-id",
            "smoke_teacher",
            "--output-dir",
            str(dataset_dir),
            "--seeds",
            "0",
            "--end-time",
            "20000",
            "--agent-period",
            "10000",
            "--dqn-runtime-semantics",
            "C_AMC_SEM",
            "--c-amc-sem-xf",
            "0.5",
            "--reward-mode",
            "mendes",
            "--action-space",
            "single",
            "--observation-mode",
            "v11_full_10d",
            *_mc_fairgen_args(),
        ],
        env=env,
    )

    _run(
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
            "--output-dir",
            str(trees_dir),
            "--train-seeds",
            "0",
            "--validation-seeds",
            "1",
            "--iterations",
            "1",
            "--end-time",
            "20000",
            "--validation-end-time",
            "20000",
            "--agent-period",
            "10000",
            "--dqn-runtime-semantics",
            "C_AMC_SEM",
            "--c-amc-sem-xf",
            "0.5",
            "--reward-mode",
            "mendes",
            "--action-space",
            "single",
            "--observation-mode",
            "v11_full_10d",
            "--max-depth-grid",
            "2",
            "--min-samples-leaf-grid",
            "1",
            *_mc_fairgen_args(),
        ],
        env=env,
    )

    run_config_path = trees_dir / "depth_2" / "leaf_1" / "run_config.json"
    with run_config_path.open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    _assert_mc_fairgen_config(run_config["workload_cli_config"])
    assert run_config["workload_mismatch_warning"] is None

    mismatch_cmd = [
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
        "0",
        "--validation-seeds",
        "1",
        "--iterations",
        "1",
        "--end-time",
        "20000",
        "--validation-end-time",
        "20000",
        "--agent-period",
        "10000",
        "--dqn-runtime-semantics",
        "C_AMC_SEM",
        "--c-amc-sem-xf",
        "0.5",
        "--reward-mode",
        "mendes",
        "--action-space",
        "single",
        "--observation-mode",
        "v11_full_10d",
        "--max-depth-grid",
        "2",
        "--min-samples-leaf-grid",
        "1",
        *_mc_fairgen_args(),
        "--mc-fairgen-lo-overrun-prob",
        "0.13",
    ]
    failed = subprocess.run(
        [*mismatch_cmd, "--output-dir", str(tmp_path / "trees_should_fail")],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert failed.returncode != 0
    assert "mc_fairgen workload 参数与 initial_dataset 不一致" in (failed.stderr + failed.stdout)

    _run(
        [
            *mismatch_cmd,
            "--output-dir",
            str(mismatch_dir),
            "--allow-workload-mismatch",
        ],
        env=env,
    )
    with (mismatch_dir / "depth_2" / "leaf_1" / "run_config.json").open("r", encoding="utf-8") as handle:
        mismatch_run_config = json.load(handle)
    assert mismatch_run_config["workload_mismatch_warning"] is not None
    assert "mc_fairgen workload 参数与 initial_dataset 不一致" in mismatch_run_config["workload_mismatch_warning"]
