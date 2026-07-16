"""人类可读报告 CLI。"""

import argparse
import json
from pathlib import Path


def render_report(verified: Path, output: Path) -> None:
    summary = json.loads((verified / "proof_summary.json").read_text(encoding="utf-8"))
    root = summary.get("outer_bundle_root", "未生成")
    lines = ["# Formal proof report", "", "## Input identity", "",
             f"- fixture_id: `{summary.get('fixture_id', 'unknown')}`",
             f"- fixture_kind: `{summary.get('fixture_kind', 'unknown')}`",
             f"- profile: `{summary.get('profile')}`",
             f"- primary claim: `{summary.get('primary_claim')}`", "",
             "## Claim result", "",
             f"- result_status: `{summary.get('result_status')}`",
             f"- workflow_status: `{summary.get('workflow_status')}`",
             f"- certificate_context_hash: `{summary.get('certificate_context_hash', '未生成')}`",
             f"- outer_bundle_root: `{root}`", "",
             "## Obligation status", ""]
    for obligation_id, status in sorted(summary.get("obligation_statuses", {}).items()):
        lines.append(f"- `{obligation_id}`: `{status}`")
    lines.extend(["", "## Real-seed status", "",
                  f"- real_seed_evaluation: `{summary.get('real_seed_evaluation', 'DEFERRED')}`", ""])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="渲染 verified bundle 报告")
    parser.add_argument("--verified", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        render_report(args.verified, args.out)
    except Exception as exc:
        print(f"report render failed: {exc}")
        return 70
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
