"""Human-readable V10.1 proof report."""

import argparse
import json
from pathlib import Path


def render_report(verified: Path, output: Path) -> None:
    summary = json.loads((Path(verified) / "proof_summary.json").read_text(encoding="utf-8"))
    lines = [
        "# V10.1 Formal Proof Report", "",
        "## Claim", "",
        f"- proof_route: `{summary.get('proof_route')}`",
        f"- scope: `{summary.get('scope')}`",
        f"- primary_claim: `{summary.get('primary_claim')}`",
        f"- result_status: `{summary.get('result_status')}`",
        f"- target_id: `{summary.get('target_id')}`",
        f"- taskset_seed: `{summary.get('taskset_seed')}`",
        f"- binding_root_hash: `{summary.get('binding_root_hash', 'not-produced')}`",
        f"- Event Graph in PASS dependency: `{summary.get('event_graph_in_pass_dependency', False)}`", "",
        "## Target terminal routes", "",
    ]
    for cert in summary.get("target_certificates", []):
        lines.append(
            f"- `{cert.get('target')}`: `{cert.get('status')}`"
            + (f", R={cert.get('response_bound')}" if cert.get('response_bound') is not None else "")
            + (f", blocker=`{cert.get('failure_code')}`" if cert.get('failure_code') else "")
        )
    lines.extend(["", "## Obligations", ""])
    for obligation, status in sorted(summary.get("obligation_statuses", {}).items()):
        lines.append(f"- `{obligation}`: `{status}`")
    if summary.get("failure_code"):
        lines.extend(["", "## Blocking result", "", f"- failure_code: `{summary['failure_code']}`"])
        if summary.get("failure_message"):
            lines.append(f"- failure_message: {summary['failure_message']}")
    lines.extend([
        "", "## V10.1 semantics", "",
        "- Terminal order is BASE C-AMC-sem refinement/certificate first, then PCSSC for unresolved HI targets.",
        "- PCSSC uses one common switch profile per target window and controller-epoch macro transitions; it does not enumerate scheduler events.",
        "- A PCSSC PASS requires a concrete tested horizon R with `W#(R) <= R`, plus controller-prefix, carry-in, arrival-count, and fixed-priority accounting receipts.",
        "- Analytical failure is UNRESOLVED. Only an independently replayed concrete runtime counterexample may be labelled unsafe.",
        "- Only `DEPLOYED_TREE_PROVED_P0_V10_1` establishes the deployed P0 HI-safety claim.", "",
    ])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="render a V10.1 verified bundle report")
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
