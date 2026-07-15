"""artifact 检查入口的最小实现。"""

import argparse
import json
import sys
from pathlib import Path

from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查第一轮整数树 artifact")
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = inspect_tree_artifact(args.artifact_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ARTIFACT_INVALID: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
