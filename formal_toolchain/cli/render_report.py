"""人类可读报告 CLI。"""

import argparse
import json
from pathlib import Path


def render_report(verified: Path, output: Path) -> None:
    summary = json.loads((verified / "proof_summary.json").read_text(encoding="utf-8"))
    root = summary.get("outer_bundle_root", "未生成")
    # 报告只投影 verified 目录中的结果，不把报告自身写入 context 或 root。
    lines = ["# Formal proof report", "", "## Target identity", "",
             f"- target_id: `{summary.get('target_id', summary.get('fixture_id', 'unknown'))}`",
             f"- target_kind: `{summary.get('target_kind', summary.get('fixture_kind', 'unknown'))}`",
             f"- taskset_seed: `{summary.get('taskset_seed', 'unknown')}`",
             f"- tree_variant: `{summary.get('tree_variant', 'unknown')}`",
             f"- profile: `{summary.get('profile')}`",
             f"- proof_route: `{summary.get('proof_route', 'legacy_strict_full')}`",
             f"- analysis_taskset_kind: `{summary.get('analysis_taskset_kind', 'unknown')}`",
             f"- primary claim: `{summary.get('primary_claim')}`", "",
             "## Claim result", "",
             f"- result_status: `{summary.get('result_status')}`",
             f"- workflow_status: `{summary.get('workflow_status')}`",
             f"- certificate_context_hash: `{summary.get('certificate_context_hash', '未生成')}`",
             f"- outer_bundle_root: `{root}`", "",
             "## Obligation status", ""]
    for obligation_id, status in sorted(summary.get("obligation_statuses", {}).items()):
        lines.append(f"- `{obligation_id}`: `{status}`")
    lines.extend(["", "## Independent evidence", "",
                  f"- rta_replay_verified: `{summary.get('rta_replay_verified', False)}`",
                  f"- bridge_proof_verified: `{summary.get('bridge_proof_verified', False)}`",
                  f"- claim_aggregation_source: `{summary.get('claim_aggregation_source', 'unknown')}`",
                  f"- route_terminal_status: `{summary.get('route_terminal_status', 'UNKNOWN')}`",
                  "", "## Context and TCB", "",
                  "- outer root 包含状态决定证据、active obligation 集合和 claim request；不包含本报告。",
                  "- DEPLOYED_TREE_PROVED 仅相对于 TheoryManifest 中声明的 TCB 成立。",
                  "- 没有 proof assistant proof object 时，不宣称所有数学定理已被机器证明。",
                  "", "## Real-seed status", "",
                  f"- real_seed_evaluation: `{summary.get('real_seed_evaluation', 'UNRESOLVED')}`", ""])
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
