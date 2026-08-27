"""Human-readable V9.1 proof report."""

import argparse
import json
from pathlib import Path


def render_report(verified: Path, output: Path) -> None:
    summary = json.loads((Path(verified) / "proof_summary.json").read_text(encoding="utf-8"))
    lines = [
        "# V9.1 Formal Proof Report", "",
        "## Claim", "",
        f"- proof_route: `{summary.get('proof_route')}`",
        f"- scope: `{summary.get('scope')}`",
        f"- primary_claim: `{summary.get('primary_claim')}`",
        f"- result_status: `{summary.get('result_status')}`",
        f"- target_id: `{summary.get('target_id')}`",
        f"- taskset_seed: `{summary.get('taskset_seed')}`",
        f"- binding_root_hash: `{summary.get('binding_root_hash', 'not-produced')}`", "",
        "## Obligations", "",
    ]
    for obligation, status in sorted(summary.get("obligation_statuses", {}).items()):
        lines.append(f"- `{obligation}`: `{status}`")
    if summary.get("failure_code"):
        lines.extend(["", "## Blocking result", "", f"- failure_code: `{summary['failure_code']}`"])
        if summary.get("failure_message"):
            lines.append(f"- failure_message: {summary['failure_message']}")
    lines.extend([
        "", "## Semantics", "",
        "- This route contains no fixed-WCET all-task RTA terminal.",
        "- A SAT first-bad-window formula is not classified as concrete unsafety unless boot-safe-prefix reachability and an independent concrete replay are machine-verified.",
        "- Only `DEPLOYED_TREE_PROVED_P0` establishes the deployed P0 HI-safety claim.", "",
    ])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="render a V9.1 verified bundle report")
    parser.add_argument("--verified", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        render_report(args.verified, args.out)
    except (OSError, ValueError, KeyError) as exc:
        print(f"report render failed: {exc}")
        return 30
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
