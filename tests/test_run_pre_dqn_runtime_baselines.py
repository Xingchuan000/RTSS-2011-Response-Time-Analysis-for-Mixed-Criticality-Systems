"""阶段 8：pre-DQN baseline 脚本测试。"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_script_runs_and_generates_csv(tmp_path: Path) -> None:
    """脚本应可运行并输出 CSV 文件。"""

    output = tmp_path / "pre_dqn.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_pre_dqn_runtime_baselines.py",
            "--end-time",
            "100",
            "--seed",
            "0",
            "--output",
            str(output),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    assert output.exists()


def test_csv_contains_required_fields_and_noop_matches_baseline(tmp_path: Path) -> None:
    """CSV 应包含必要字段，且 NoOp 与 baseline 指标一致。"""

    output = tmp_path / "pre_dqn.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_pre_dqn_runtime_baselines.py",
            "--end-time",
            "100",
            "--seed",
            "0",
            "--scenario",
            "all",
            "--output",
            str(output),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    with output.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "scenario",
        "method",
        "seed",
        "end_time",
        "agent_period",
        "mode_changes",
        "lo_cancellations",
        "deadline_misses",
        "hdm",
        "jne",
        "ldm",
        "nid",
        "tid",
        "total_time",
        "jne_plus_ldm",
        "accepted_actions",
        "rejected_actions",
        "noop_actions",
        "total_reward",
    }
    assert rows
    assert required.issubset(rows[0].keys())

    table = {row["method"]: row for row in rows}
    nominal_rows = {row["method"]: row for row in rows if row["scenario"] == "nominal"}
    assert nominal_rows["amc_plus_baseline"]["mode_changes"] == nominal_rows["noop_agent"]["mode_changes"]
    assert nominal_rows["amc_plus_baseline"]["lo_cancellations"] == nominal_rows["noop_agent"]["lo_cancellations"]
    assert nominal_rows["amc_plus_baseline"]["deadline_misses"] == nominal_rows["noop_agent"]["deadline_misses"]
    assert "heuristic_agent" in table

    for row in rows:
        assert int(row["mode_changes"]) >= 0
        assert int(row["lo_cancellations"]) >= 0
        assert int(row["deadline_misses"]) >= 0
        assert int(row["hdm"]) >= 0
        assert int(row["jne"]) >= 0
        assert int(row["ldm"]) >= 0
        assert int(row["nid"]) >= 0
        assert int(row["tid"]) >= 0
        assert int(row["total_time"]) >= 0
        assert int(row["jne_plus_ldm"]) == int(row["jne"]) + int(row["ldm"])
        assert int(row["accepted_actions"]) >= 0
        assert int(row["rejected_actions"]) >= 0
        assert int(row["noop_actions"]) >= 0


def test_ra_csv_contains_degradation_columns_and_preserves_old_fields(tmp_path: Path) -> None:
    """AMC_RA 运行时导出的 CSV 应同时包含旧字段与论文 degraded-service 字段。"""

    output = tmp_path / "pre_dqn_ra.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_pre_dqn_runtime_baselines.py",
            "--end-time",
            "100",
            "--seed",
            "0",
            "--scenario",
            "stress",
            "--runtime-semantics",
            "AMC_RA",
            "--output",
            str(output),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    with output.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    baseline_rows = [row for row in rows if row["method"] == "amc_ra_baseline"]
    assert baseline_rows
    row = baseline_rows[0]
    assert "mode_changes" in row
    assert "lo_cancellations" in row
    assert "deadline_misses" in row
    assert "hdm" in row
    assert "jne" in row
    assert "ldm" in row
    assert "nid" in row
    assert "tid" in row
    assert "total_time" in row
    assert "jne_plus_ldm" in row
    assert int(row["jne_plus_ldm"]) == int(row["jne"]) + int(row["ldm"])
