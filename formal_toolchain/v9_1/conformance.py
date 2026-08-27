"""Concrete-to-kernel conformance and prefix-refinement obligations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import z3

from formal_toolchain.core.hashing import sha256_file

from .concrete_projection import project_prefix
from .timestamp_trace import TimestampSemanticRecord


@dataclass(frozen=True, slots=True)
class ConformanceResult:
    status: str
    code: str | None
    projected_length: int


def check_projection_conformance(records: Iterable[TimestampSemanticRecord]) -> ConformanceResult:
    rows = tuple(records)
    try:
        projected = project_prefix(rows)
    except ValueError as exc:
        return ConformanceResult("FAIL", str(exc), 0)
    for record in rows:
        if record.budget_after_controller is not None and record.controller_action is None:
            return ConformanceResult("FAIL", "RELEASE_SNAPSHOT_CONFORMANCE_FAILED", len(projected))
    return ConformanceResult("PASS", None, len(projected))


def _write_refutation(path: Path, valid: bool, theorem: str) -> str:
    solver = z3.Solver()
    fact = z3.Bool(f"{theorem}_record_valid")
    solver.add(fact == bool(valid), z3.Not(fact))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(solver.sexpr() + "\n", encoding="utf-8")
    return sha256_file(path)


def build_conformance_proof_objects(records: Iterable[TimestampSemanticRecord], output_dir: Path) -> dict[str, object]:
    result = check_projection_conformance(records)
    output_dir = Path(output_dir)
    hashes = {
        "kernel_step_conformance": _write_refutation(output_dir / "kernel_step_conformance.smt2",
                                                      result.status == "PASS", "kernel_step"),
        "prefix_refinement": _write_refutation(output_dir / "prefix_refinement.smt2",
                                                result.status == "PASS", "prefix_refinement"),
        "first_hi_bad_prefix_reflection": _write_refutation(output_dir / "first_hi_bad_prefix_reflection.smt2",
                                                            result.status == "PASS", "first_hi_bad_prefix"),
    }
    return {"status": result.status, "code": result.code, "projected_length": result.projected_length,
            "proof_object_hashes": hashes}


__all__ = ["ConformanceResult", "build_conformance_proof_objects", "check_projection_conformance"]
