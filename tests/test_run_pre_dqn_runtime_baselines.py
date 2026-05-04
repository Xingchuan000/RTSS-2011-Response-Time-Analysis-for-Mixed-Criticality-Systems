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
        assert int(row["accepted_actions"]) >= 0
        assert int(row["rejected_actions"]) >= 0
        assert int(row["noop_actions"]) >= 0
