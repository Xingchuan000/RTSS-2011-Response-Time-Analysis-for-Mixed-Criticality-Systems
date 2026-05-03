"""阶段九事件驱动示例脚本可运行性测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_event_runtime_example_script_outputs_expected_keywords() -> None:
    """脚本应成功运行，并输出事件驱动语义要求的关键摘要。"""

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "run_event_runtime_example.py"
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    assert "case,semantics,mode_changes,lo_cancellations,recoveries,deadline_misses,final_mode" in output
    assert "case1_lo_overrun_event,AMC_PLUS,mode_changes=0,lo_cancellations=1" in output
    assert "case2_lo_overrun_event,AMC,mode_changes=1" in output
    assert "case3_dynamic_budget_updated_event,AMC_PLUS" in output
    assert "case3_dynamic_budget_updated_event,AMC_PLUS,mode_changes=0,lo_cancellations=0" in output
