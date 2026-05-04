"""阶段 0 ablation 入口脚本测试。"""

from __future__ import annotations

from pathlib import Path

from scripts.run_ablation import _build_command


def test_build_command_supports_bool_flags_and_scalar_values() -> None:
    """命令构建应正确输出布尔开关与普通参数。"""

    cmd = _build_command(
        Path("scripts/train_dqn_amc.py"),
        {
            "workload": "rtss11",
            "include_explicit_noop": True,
            "require_schedulable": False,
            "episodes": 3,
        },
    )

    assert cmd[0]
    assert cmd[1].endswith("scripts/train_dqn_amc.py")
    assert "--workload" in cmd and "rtss11" in cmd
    assert "--include-explicit-noop" in cmd
    assert "--no-require-schedulable" in cmd
    assert "--episodes" in cmd and "3" in cmd
