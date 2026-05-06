"""workload 模块导入边界测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_automotive_workload_does_not_import_dqn() -> None:
    """单独导入 automotive workload 时，不应顺带拉起 DQN agent 或 torch。

    这里使用独立 Python 子进程验证导入边界，避免当前 pytest 进程中其他测试
    已经导入过 DQN 或 torch，导致 `sys.modules` 被污染。
    """

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import amc_py.workloads.automotive; "
                "assert 'amc_py.dqn.agent' not in sys.modules; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=project_root,
        check=False,
    )
    assert result.returncode == 0
