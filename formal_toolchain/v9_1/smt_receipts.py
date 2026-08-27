"""Fresh Z3 replay for V9.1 assertion-only SMT-LIB proof obligations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file


def replay_unsat(path: Path, *, timeout_ms: int = 120_000) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {"status": "MISSING", "path": path.name}
    try:
        import z3
    except ImportError:
        return {"status": "UNKNOWN", "reason": "Z3_NOT_INSTALLED", "sha256": sha256_file(path)}
    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    try:
        assertions = z3.parse_smt2_file(str(path))
        solver.add(assertions)
        result = solver.check()
    except z3.Z3Exception as exc:
        return {"status": "INVALID", "reason": str(exc), "sha256": sha256_file(path)}
    if result == z3.unsat:
        status = "UNSAT"
    elif result == z3.sat:
        status = "SAT"
    else:
        status = "UNKNOWN"
    receipt: dict[str, Any] = {
        "status": status,
        "sha256": sha256_file(path),
        "solver": z3.get_version_string(),
        "timeout_ms": int(timeout_ms),
    }
    if status == "UNKNOWN":
        receipt["reason"] = solver.reason_unknown()
    return receipt
