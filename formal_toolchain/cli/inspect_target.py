"""Phase L preflight 入口。"""

import argparse
import json
import sys
from pathlib import Path

from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.core.formal_checks import calculate_raw_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查第一轮 proof request 与 synthetic target")
    parser.add_argument("artifact_dir", type=Path, nargs="?")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--out", "--output", dest="output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.request is not None:
            computed = calculate_raw_evidence(args.request, source_root=Path.cwd(), include_reference=False)
            result = {"workflow_status": "PREFLIGHT", "obligation_status": "PASS",
                      "certificate_context_hash": computed["context_hash"],
                      "fixture": "synthetic_p0" if computed["request"].get("fixture_id") == "synthetic_p0" else None,
                      "inventory": computed["inventory"], "preflight": computed["evidence"]["PREFLIGHT"]}
        elif args.artifact_dir is not None:
            result = inspect_tree_artifact(args.artifact_dir)
        else:
            parser.error("必须提供 artifact_dir 或 --request")
    except (OSError, ValueError, KeyError) as exc:
        print(f"ARTIFACT_INVALID: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "seed_import_diagnostic.json").write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
