"""verified summary 到 Markdown 的稳定投影。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def render_markdown(summary: Mapping[str, Any], output: Path) -> None:
    lines = ["# Formal proof report", "", "## Input identity", "",
             f"- fixture_id: `{summary.get('fixture_id', 'unknown')}`",
             f"- fixture_kind: `{summary.get('fixture_kind', 'unknown')}`",
             f"- profile: `{summary.get('profile', 'unknown')}`", "",
             "## Claim result", "",
             f"- result_status: `{summary.get('result_status', 'unknown')}`",
             f"- real_seed_evaluation: `{summary.get('real_seed_evaluation', 'DEFERRED')}`", ""]
    Path(output).write_text("\n".join(lines), encoding="utf-8")
