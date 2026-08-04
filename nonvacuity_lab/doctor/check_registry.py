from __future__ import annotations

import json
from pathlib import Path

from .schema import DoctorCheck


def load_obligation_ids(source_root: Path) -> set[str]:
    path = source_root / "formal_toolchain/specs/obligation_registry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["obligation_id"]) for item in data.get("obligations", [])}


def check_expected_obligations(config: dict, obligation_ids: set[str]):
    unknown = []
    for mutation in config.get("mutations", []):
        expected = mutation.get("expected", {})
        for obligation in expected.get("allowed_first_failing_obligations", expected.get("first_failing_obligations", ())):
            if obligation not in obligation_ids:
                unknown.append({"mutation_id": mutation.get("mutation_id"), "obligation_id": obligation})
    return DoctorCheck.fail("obligation_registry", "unknown expected obligation ids", unknown=unknown) if unknown else DoctorCheck.pass_("obligation_registry", "all expected obligations exist", count=len(obligation_ids))
