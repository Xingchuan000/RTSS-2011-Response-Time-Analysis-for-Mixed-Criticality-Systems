"""Generate the Phase-3 audit bundle used by campaign resolvers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .analysis.rta_slack import scan_rta_slack
from .audit.bundle_inventory import build_bundle_inventory
from .audit.leaf_coverage import audit_all_leaves
from .canonical import canonical_json_hash, file_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", required=True, type=Path)
    parser.add_argument("--proof-bundle-root", required=True, type=Path)
    parser.add_argument("--hout-root", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    seed_root = args.seed_root.resolve()
    proof_root = args.proof_bundle_root.resolve()
    hout_root = args.hout_root.resolve()
    source_root = args.source_root.resolve()
    for label, path in (
        ("seed-root", seed_root),
        ("proof-bundle-root", proof_root),
        ("hout-root", hout_root),
        ("source-root", source_root),
    ):
        if not path.is_dir():
            raise SystemExit(f"{label} 不存在或不是目录: {path}")

    args.out.mkdir(parents=True, exist_ok=False)
    leaf_rows = audit_all_leaves(
        seed_root=seed_root,
        hout_root=hout_root,
        source_root=source_root,
    )
    rta_rows = scan_rta_slack([proof_root])
    bundle_rows = build_bundle_inventory(proof_root)

    _write_json(args.out / "leaf_audit.json", {"rows": leaf_rows})
    _write_json(args.out / "rta_slack.json", {"rows": rta_rows})
    risks = [
        {
            "seed": row["seed"],
            "tree_variant": row["tree_variant"],
            "leaf_id": row["leaf_id"],
            **risk,
        }
        for row in leaf_rows
        for risk in row.get("action_risks", [])
    ]
    _write_json(args.out / "action_risk_catalog.json", {"rows": risks})
    reasons: dict[str, int] = {}
    for row in leaf_rows:
        for reason, count in row.get("reject_reason_histogram", {}).items():
            reasons[str(reason)] = reasons.get(str(reason), 0) + int(count)
    _write_json(args.out / "reject_reason_catalog.json", {"rows": reasons})
    hout_inventory = [
        {
            "path": path.relative_to(hout_root).as_posix(),
            "sha256": file_hash(path),
            "size": path.stat().st_size,
        }
        for path in sorted(hout_root.rglob("*.json"))
    ]
    _write_json(args.out / "hout_inventory.json", {"rows": hout_inventory})
    _write_json(args.out / "bundle_inventory.json", {"rows": bundle_rows})
    _write_csv(args.out / "leaf_audit.csv", leaf_rows)
    _write_csv(args.out / "rta_slack.csv", rta_rows)

    summary = {
        "schema_version": "nonvacuity_audit_v1",
        "seed_root": str(seed_root),
        "proof_bundle_root": str(proof_root),
        "hout_root": str(hout_root),
        "source_root": str(source_root),
        "tree_count": len({row.get("tree_hash") for row in leaf_rows}),
        "leaf_count": len(leaf_rows),
        "rta_record_count": len(rta_rows),
        "bundle_inventory_count": len(bundle_rows),
        "hout_file_count": len(hout_inventory),
        "content_hash": canonical_json_hash(
            {
                "leaf_rows": leaf_rows,
                "rta_rows": rta_rows,
                "bundle_rows": bundle_rows,
                "hout_inventory": hout_inventory,
            }
        ),
    }
    _write_json(args.out / "audit_summary.json", summary)
    return 0


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    keys = sorted({str(key) for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: _csv_value(row.get(key))
                    for key in keys
                }
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
