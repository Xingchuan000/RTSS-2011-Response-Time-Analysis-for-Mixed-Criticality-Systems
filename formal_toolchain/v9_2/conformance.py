"""Concrete-to-kernel projection diagnostics for V9.2.

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



__all__ = ["ConformanceResult", "check_projection_conformance"]
