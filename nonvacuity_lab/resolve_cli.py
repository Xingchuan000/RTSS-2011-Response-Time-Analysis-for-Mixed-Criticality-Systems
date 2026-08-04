from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import load_campaign
from .resolver.manifest_resolver import resolve_campaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--require-all-resolved",
        action="store_true",
        help="存在未解析目标时返回非零；仍写出 disabled manifest 和 receipt 供审计。",
    )
    args = parser.parse_args(argv)

    config = load_campaign(args.template)
    resolved, receipt = resolve_campaign(config, audit_root=args.audit_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_path = args.out.with_suffix(".resolver_receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Re-load through the public loader to prove the resolved output still
    # conforms to the campaign schema and path rules.
    load_campaign(args.out)
    if args.require_all_resolved and receipt["unresolved_targets"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
