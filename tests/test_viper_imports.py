"""VIPER 模块导入测试。"""

from __future__ import annotations


def test_viper_package_imports() -> None:
    import amc_py.viper as viper

    assert hasattr(viper, "TreeBudgetPolicy")
    assert hasattr(viper, "collect_teacher_labeled_rollouts")
