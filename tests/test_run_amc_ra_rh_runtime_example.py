"""AMC-RA / AMC-RH 示例脚本可运行性测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_amc_ra_rh_runtime_example_outputs_expected_keywords() -> None:
    """脚本应输出 RA/RH 对比与关键统计字段。"""

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "run_amc_ra_rh_runtime_example.py"
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    assert "AMC_RA" in output
    assert "AMC_RH" in output
    assert "dropped_lo_jobs" in output
    assert "jne" in output
    assert "tid" in output
