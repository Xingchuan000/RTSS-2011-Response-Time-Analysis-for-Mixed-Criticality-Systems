"""Fresh in-memory Z3 solving for verifier-regenerated V9.1 formulas."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import z3


@dataclass(frozen=True, slots=True)
class FormulaReceipt:
    obligation_id: str
    result: str
    formula_hash: str
    solver_version: str
    timeout_ms: int
    reason: str | None = None
    model: Any = None

    def as_dict(self, *, include_model: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "obligation_id": self.obligation_id,
            "result": self.result,
            "formula_hash": self.formula_hash,
            "solver_version": self.solver_version,
            "timeout_ms": self.timeout_ms,
        }
        if self.reason is not None:
            row["reason"] = self.reason
        if include_model and self.model is not None:
            row["model"] = self.model
        return row


def canonical_formula_text(formula: z3.BoolRef) -> str:
    solver = z3.Solver()
    solver.add(formula)
    return solver.sexpr()


def solve_formula(
    obligation_id: str,
    formula: z3.BoolRef,
    *,
    timeout_ms: int = 120_000,
    capture_model: bool = False,
) -> FormulaReceipt:
    """Solve one regenerated formula in a fresh solver instance."""

    text = canonical_formula_text(formula)
    formula_hash = sha256(text.encode("utf-8")).hexdigest()
    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    solver.add(formula)
    result = solver.check()
    if result == z3.unsat:
        return FormulaReceipt(
            obligation_id, "UNSAT", formula_hash, z3.get_version_string(), int(timeout_ms)
        )
    if result == z3.sat:
        model = None
        if capture_model:
            model = {str(decl): str(solver.model()[decl]) for decl in solver.model().decls()}
        return FormulaReceipt(
            obligation_id, "SAT", formula_hash, z3.get_version_string(), int(timeout_ms), model=model
        )
    return FormulaReceipt(
        obligation_id,
        "UNKNOWN",
        formula_hash,
        z3.get_version_string(),
        int(timeout_ms),
        reason=solver.reason_unknown(),
    )


__all__ = ["FormulaReceipt", "canonical_formula_text", "solve_formula"]
