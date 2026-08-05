from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import load_campaign
from .resolver.manifest_resolver import resolve_campaign
from .config_io import validate_config_kind, verify_config_hash


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

    raw_template = json.loads(args.template.read_text(encoding="utf-8"))
    if raw_template.get("schema_version") == "nonvacuity_campaign_v2":
        from .config_resolver import resolve_campaign as resolve_v2
        resolved = resolve_v2(args.template, args.audit_root, Path.cwd(), args.out)
        receipt = resolved.get("resolver_receipt", {})
    else:
        config = load_campaign(args.template)
        resolved, receipt = resolve_campaign(config, audit_root=args.audit_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if raw_template.get("schema_version") != "nonvacuity_campaign_v2":
        args.out.write_text(json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt_path = args.out.with_suffix(".resolver_receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if raw_template.get("schema_version") == "nonvacuity_campaign_v2":
        validate_config_kind(json.loads(args.out.read_text(encoding="utf-8")))
        verify_config_hash(json.loads(args.out.read_text(encoding="utf-8")))
        unresolved = [item for item in receipt.get("records", []) if item.get("status") != "RESOLVED"]
    else:
        load_campaign(args.out)
        unresolved = receipt["unresolved_targets"]
    if args.require_all_resolved and unresolved:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
