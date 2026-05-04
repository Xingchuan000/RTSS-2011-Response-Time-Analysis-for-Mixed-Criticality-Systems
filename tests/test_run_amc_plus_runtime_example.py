"""阶段 9 示例脚本可运行性测试。"""

from __future__ import annotations

import subprocess
import sys
import sys
from pathlib import Path


def test_run_amc_plus_runtime_example_script_outputs_expected_keywords() -> None:
    """脚本应成功运行，并输出 AMC/AMC_PLUS 与关键统计字段。"""

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "run_amc_plus_runtime_example.py"
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    assert "AMC_PLUS" in output
    assert "AMC" in output
    assert "mode_changes" in output
    assert "lo_cancellations" in output

