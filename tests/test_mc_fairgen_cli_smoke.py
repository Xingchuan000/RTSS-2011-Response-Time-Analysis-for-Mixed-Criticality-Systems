"""MC-FairGen CLI smoke 回归测试。"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(cwd)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def test_train_dqn_amc_mc_fairgen_logs_effective_num_tasks(tmp_path: Path) -> None:
    """mc_fairgen 训练日志与 config 中的 num_tasks 应等于实际配置值。"""

    # 使用测试文件位置推导仓库根目录，避免硬编码绝对路径导致跨机器失败。
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "train_out"
    _run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "mc_fairgen",
            "--mc-fairgen-num-tasks",
            "8",
            "--episodes",
            "1",
            "--end-time",
            "20000",
            "--agent-period",
            "10000",
            "--seed",
            "0",
            "--fixed-taskset-seed",
            "0",
            "--validate-every",
            "0",
            "--output-dir",
            str(out_dir),
        ],
        repo,
    )

    with (out_dir / "train_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert int(float(row["num_tasks"])) == 8

    with (out_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["workload"] == "mc_fairgen"
    assert int(cfg["mc_fairgen_num_tasks"]) == 8


def test_mc_fairgen_manifest_roundtrip_scan(tmp_path: Path) -> None:
    """generator manifest 应可被 scan 直接 roundtrip 使用。"""

    # 使用测试文件位置推导仓库根目录，避免硬编码绝对路径导致跨机器失败。
    repo = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.csv"
    reject = tmp_path / "reject.csv"
    scan_out = tmp_path / "scan.csv"

    _run(
        [
            sys.executable,
            "scripts/generate_learnable_tasksets.py",
            "--workload",
            "mc_fairgen",
            "--mc-fairgen-num-tasks",
            "8",
            "--num-tasksets",
            "1",
            "--learnable-max-attempts",
            "8",
            "--learnable-fast-eval-seeds",
            "1",
            "--learnable-fast-end-time",
            "20000",
            "--learnable-fast-event-min",
            "0",
            "--learnable-fast-event-max",
            "1000",
            "--learnable-fast-min-lo-cancellations",
            "0",
            "--learnable-fast-min-lo-cancellation-ratio",
            "0",
            "--learnable-fast-min-valid-decrease",
            "0",
            "--enable-constraint-guided-pair-diagnostic",
            "--constraint-guided-pair-min-valid-count",
            "0",
            "--learnable-selection-target",
            "constraint_guided_pair",
            "--output-manifest",
            str(manifest),
            "--output-rejections",
            str(reject),
        ],
        repo,
    )

    _run(
        [
            sys.executable,
            "scripts/scan_taskset_headroom.py",
            "--workload",
            "mc_fairgen",
            "--taskset-manifest",
            str(manifest),
            "--budget-scales",
            "1.00",
            "--seeds",
            "200:202",
            "--end-time",
            "20000",
            "--agent-period",
            "10000",
            "--enable-constraint-guided-pair-diagnostic",
            "--constraint-guided-pair-min-valid-count",
            "0",
            "--workers",
            "1",
            "--output",
            str(scan_out),
        ],
        repo,
    )

    with scan_out.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    assert rows
    assert "candidate_seed" in fields
    assert "baseline_lo_cancellations_mean" in fields
    assert "valid_constraint_guided_pair_count_mean" in fields
