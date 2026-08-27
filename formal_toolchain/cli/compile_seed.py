"""V9.1 candidate compiler CLI."""

import argparse
import json
from pathlib import Path

from formal_toolchain.v9_1.compiler import compile_request_v9_1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="compile a V9.1 candidate proof bundle")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = compile_request_v9_1(args.request, args.out, source_root=args.source_root)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAILED", "failure_code": "V9_1_COMPILER_INPUT_ERROR",
                          "failure_message": str(exc)}, ensure_ascii=False))
        return 30
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
