from __future__ import annotations

import json
from pathlib import Path

from .schema import DoctorCheck


def load_obligation_ids(source_root: Path) -> set[str]:
    """Load obligation IDs from both legacy and current registry schemas."""

    path = source_root / "formal_toolchain/specs/obligation_registry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("entries")
    id_field = "id"
    if not isinstance(rows, list):
        rows = data.get("obligations", [])
        id_field = "obligation_id"
    if not isinstance(rows, list):
        raise ValueError("obligation registry must contain entries/obligations array")
    result = {
        str(item[id_field])
        for item in rows
        if isinstance(item, dict) and item.get(id_field) not in (None, "")
    }
    routes_root = source_root / "formal_toolchain/specs/routes"
    if routes_root.is_dir():
        for route_path in sorted(routes_root.glob("*_registry.json")):
            route_data = json.loads(route_path.read_text(encoding="utf-8"))
            route_ids = route_data.get("obligation_ids", ())
            if isinstance(route_ids, list):
                result.update(str(item) for item in route_ids if item not in (None, ""))
    if not result:
        raise ValueError("obligation registry contains no obligation ids")
    return result


def check_expected_obligations(config: dict, obligation_ids: set[str]):
    unknown = []
    for mutation in config.get("mutations", []):
        expected = mutation.get("expected", {})
        for obligation in expected.get("allowed_first_failing_obligations", expected.get("first_failing_obligations", ())):
            if obligation not in obligation_ids:
                unknown.append({"mutation_id": mutation.get("mutation_id"), "obligation_id": obligation})
    return DoctorCheck.fail("obligation_registry", "unknown expected obligation ids", unknown=unknown) if unknown else DoctorCheck.pass_("obligation_registry", "all expected obligations exist", count=len(obligation_ids))
