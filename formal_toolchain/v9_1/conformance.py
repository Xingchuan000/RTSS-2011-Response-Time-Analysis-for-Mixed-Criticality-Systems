"""Concrete-to-kernel projection diagnostics for V9.1.

A finite concrete trace can find a conformance bug, but it cannot prove the
universal kernel-step/refinement/reflection obligations.  This module therefore
never emits an UNSAT proof object from a sampled trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .concrete_projection import project_prefix
from .timestamp_trace import TimestampSemanticRecord


@dataclass(frozen=True, slots=True)
class ConformanceResult:
    status: str
    code: str | None
    projected_length: int


def check_projection_conformance(records: Iterable[TimestampSemanticRecord]) -> ConformanceResult:
    """Diagnostic check over supplied records; never a universal proof."""

    rows = tuple(records)
    try:
        projected = project_prefix(rows)
    except ValueError as exc:
        return ConformanceResult("FAIL", str(exc), 0)
    for record in rows:
        if record.budget_after_controller is not None and record.controller_action is None:
            return ConformanceResult("FAIL", "RELEASE_SNAPSHOT_CONFORMANCE_FAILED", len(projected))
    return ConformanceResult("PASS", None, len(projected))


def build_conformance_proof_objects(
    records: Iterable[TimestampSemanticRecord], output_dir: Path
) -> dict[str, object]:
    """Refuse to turn finite diagnostic samples into theorem proof objects."""

    result = check_projection_conformance(records)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return {
        "status": "UNRESOLVED" if result.status == "PASS" else "FAIL",
        "code": (
            "V9_1_UNIVERSAL_CONFORMANCE_PROOF_UNBOUND"
            if result.status == "PASS"
            else result.code
        ),
        "projected_length": result.projected_length,
        "proof_object_hashes": {},
    }


__all__ = ["ConformanceResult", "build_conformance_proof_objects", "check_projection_conformance"]
