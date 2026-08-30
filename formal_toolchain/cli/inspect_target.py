"""V10.1 request/tree preflight CLI."""

import argparse
import json
import sys
from pathlib import Path

from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.v10_1.bindings import build_bindings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="inspect a V10.1 proof request or integer-tree artifact")
    parser.add_argument("artifact_dir", type=Path, nargs="?")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", "--output", dest="output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.request is not None:
            bindings = build_bindings(args.request, source_root=args.source_root)
            result = {
                "workflow_status": "PREFLIGHT",
                "obligation_status": "PASS",
                "proof_route": bindings["proof_route"],
                "binding_root_hash": bindings["binding_root_hash"],
                "taskset_hash": bindings["seed_task_binding"]["taskset_hash"],
                "environment_domain_hash": bindings["environment_binding"]["demand_domain_hash"],
            }
        elif args.artifact_dir is not None:
            result = inspect_tree_artifact(args.artifact_dir)
        else:
            parser.error("provide artifact_dir or --request")
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
