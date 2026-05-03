"""阶段 9：Pre-DQN 文档存在性与关键字测试。"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pre_dqn_doc_exists_and_contains_keywords() -> None:
    """文档应存在且包含核心关键词。"""

    # 基于测试文件位置推导项目根目录，避免绑定开发者本机绝对路径。
    doc = PROJECT_ROOT / "docs" / "pre_dqn_runtime_interface.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8").lower()
    for kw in ["observation", "action space", "safety checker", "reset", "step"]:
        assert kw in text
