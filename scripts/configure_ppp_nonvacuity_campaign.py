"""Enable or disable selected rows in a resolved PPP non-vacuity campaign.

This helper is intentionally small and research-oriented.  It never resolves
missing inputs; it only changes the two explicit execution gates and reseals
the v2 configuration.  Doctor/preflight remain responsible for refusing an
incomplete enabled campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nonvacuity_lab.config_io import write_resolved_campaign


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mutation", action="append", default=[])
    parser.add_argument("--group", action="append", default=[], help="Enable canonical prefixes such as A, B, C, D, E, F, P")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--disable", action="store_true", help="Disable selected rows; with no selection disables the campaign and all rows")
    return parser.parse_args()


def _canonical_group(mutation_id: str) -> str:
    prefix = mutation_id.split("_", 1)[0]
    return prefix[:1].upper() if prefix else ""


def configure_campaign(
    config_path: Path,
    *,
    mutation_ids: tuple[str, ...] = (),
    groups: tuple[str, ...] = (),
    enable_all: bool = False,
    disable: bool = False,
) -> dict:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "nonvacuity_campaign_v2":
        raise ValueError("config must use nonvacuity_campaign_v2")
    if config.get("config_kind") != "RESOLVED":
        raise ValueError("only a RESOLVED campaign can be configured")

    rows = [item for item in config.get("mutations", []) if isinstance(item, dict)]
    known = {str(item.get("mutation_id")) for item in rows}
    requested = {str(item) for item in mutation_ids}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown mutation ids: {unknown}")
    group_set = {str(item).upper() for item in groups}
    invalid_groups = sorted(group_set - {"P", "A", "B", "C", "D", "E", "F"})
    if invalid_groups:
        raise ValueError(f"unknown groups: {invalid_groups}")

    selection_present = bool(requested or group_set or enable_all)
    if disable and not selection_present:
        selected = known
    else:
        selected = {
            mutation_id
            for mutation_id in known
            if enable_all
            or mutation_id in requested
            or _canonical_group(mutation_id) in group_set
        }
    if not selected and not (disable and not selection_present):
        raise ValueError("no mutations selected")

    for mutation in rows:
        mutation_id = str(mutation.get("mutation_id"))
        if mutation_id in selected:
            mutation["enabled"] = not disable
    config["enabled"] = any(bool(item.get("enabled")) for item in rows)
    write_resolved_campaign(config_path, config)
    sealed = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "status": "CAMPAIGN_CONFIGURED",
        "config": str(config_path),
        "campaign_enabled": bool(sealed.get("enabled")),
        "enabled_mutations": [
            str(item.get("mutation_id")) for item in sealed.get("mutations", []) if item.get("enabled")
        ],
        "config_sha256": sealed.get("config_sha256"),
    }


def main() -> int:
    args = _parse_args()
    result = configure_campaign(
        args.config,
        mutation_ids=tuple(args.mutation),
        groups=tuple(args.group),
        enable_all=bool(args.all),
        disable=bool(args.disable),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
