"""Fail-closed validation for the ordinary proof request surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_clean_proof_request(value: dict[str, Any]) -> dict[str, Any]:
    forbidden_exact = {"mutation_id", "expected", "expected_failure", "activation_status"}
    experiment_prefixes = ("non" + "vacuity_", "mutation_")
    forbidden = sorted(
        key for key in value
        if key in forbidden_exact or any(key.startswith(prefix) for prefix in experiment_prefixes)
    )
    if forbidden:
        raise ValueError(f"PROOF_REQUEST_EXPERIMENT_FIELDS_FORBIDDEN:{forbidden}")
    return value


def load_clean_proof_request(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("proof request 顶层必须为 object")
    return validate_clean_proof_request(raw)
