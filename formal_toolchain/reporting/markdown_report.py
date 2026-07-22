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
             f"- real_seed_evaluation: `{summary.get('real_seed_evaluation', 'DEFERRED')}`", "",
             "## Key verification milestones", "",
             f"- RTA replay: `{summary.get('rta_replay_verified', 'UNKNOWN')}`",
             f"- Certified envelope: `{summary.get('certified_envelope_verified', 'UNKNOWN')}`",
             f"- Bridge proof: `{summary.get('bridge_proof_verified', 'UNKNOWN')}`",
             f"- Outer bundle root: `{summary.get('outer_bundle_root', 'N/A')[:16]}...`", ""]
    statuses = summary.get("obligation_statuses", {})
    if isinstance(statuses, Mapping):
        lines.extend(["## Verified obligation statuses", "", "| Obligation | Status |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in sorted(statuses.items()))
        lines.append("")
    Path(output).write_text("\n".join(lines), encoding="utf-8")
