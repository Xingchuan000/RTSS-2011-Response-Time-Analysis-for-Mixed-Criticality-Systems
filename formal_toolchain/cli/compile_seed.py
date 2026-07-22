"""candidate compiler CLI。"""

import argparse
from pathlib import Path

from formal_toolchain.compiler.compile import compile_request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成不可授信的 candidate proof bundle")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = compile_request(args.request, args.out, source_root=args.source_root)
    except Exception as exc:
        print(f"candidate compile failed: {exc}")
        return 70
    print(__import__("json").dumps(summary, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
