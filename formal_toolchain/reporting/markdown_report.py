"""verified summary 到 Markdown 的稳定投影。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def render_markdown(summary: Mapping[str, Any], output: Path) -> None:
    lines = ["# Formal proof report", "", "## Input identity", "",
             f"- target_id: `{summary.get('target_id', summary.get('fixture_id', 'unknown'))}`",
             f"- target_kind: `{summary.get('target_kind', summary.get('fixture_kind', 'unknown'))}`",
             f"- profile: `{summary.get('profile', 'unknown')}`", "",
             "## Claim result", "",
             f"- result_status: `{summary.get('result_status', 'unknown')}`",
             f"- real_seed_evaluation: `{summary.get('real_seed_evaluation', 'DEFERRED')}`", ""]
    statuses = summary.get("obligation_statuses", {})
    if isinstance(statuses, Mapping):
        lines.extend(["## Verified result", "", "| Obligation | Fresh verifier |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in sorted(statuses.items()))
        lines.append("")
    Path(output).write_text("\n".join(lines), encoding="utf-8")
