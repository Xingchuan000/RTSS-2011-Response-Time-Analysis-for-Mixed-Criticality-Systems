"""Fresh in-memory Z3 solving for verifier-regenerated V10.1 formulas."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any

import z3

from .solver_runtime import make_solver


@dataclass(frozen=True, slots=True)
class FormulaReceipt:
    obligation_id: str
    result: str
    formula_hash: str
    solver_version: str
    timeout_ms: int
    reason: str | None = None
    model: Any = None
    formula_chars: int | None = None
    canonicalization_seconds: float | None = None
    solver_check_seconds: float | None = None

    def as_dict(self, *, include_model: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "obligation_id": self.obligation_id,
            "result": self.result,
            "formula_hash": self.formula_hash,
            "solver_version": self.solver_version,
            "timeout_ms": self.timeout_ms,
            "timeout_policy": "UNLIMITED" if int(self.timeout_ms) == 0 else "FINITE",
        }
        if self.reason is not None:
            row["reason"] = self.reason
        if self.formula_chars is not None:
            row["formula_chars"] = int(self.formula_chars)
        if self.canonicalization_seconds is not None:
            row["canonicalization_seconds"] = round(float(self.canonicalization_seconds), 6)
        if self.solver_check_seconds is not None:
            row["solver_check_seconds"] = round(float(self.solver_check_seconds), 6)
        if include_model and self.model is not None:
            row["model"] = self.model
        return row


def canonical_formula_text(formula: z3.BoolRef) -> str:
    solver = make_solver()
    solver.add(formula)
    return solver.sexpr()


def _solve_with_solver(
    obligation_id: str,
    formula: z3.BoolRef,
    *,
    solver: z3.Solver,
    timeout_ms: int,
    capture_model: bool,
) -> FormulaReceipt:
    started = perf_counter()
    text = canonical_formula_text(formula)
    canonicalization_seconds = perf_counter() - started
    formula_hash = sha256(text.encode("utf-8")).hexdigest()
    if int(timeout_ms) < 0:
        raise ValueError("timeout_ms must be non-negative; 0 means unlimited")
    if int(timeout_ms) > 0:
        solver.set(timeout=int(timeout_ms))
    solver.add(formula)
    check_started = perf_counter()
    result = solver.check()
    solver_check_seconds = perf_counter() - check_started
    metrics = {
        "formula_chars": len(text),
        "canonicalization_seconds": canonicalization_seconds,
        "solver_check_seconds": solver_check_seconds,
    }
    if result == z3.unsat:
        return FormulaReceipt(
            obligation_id, "UNSAT", formula_hash, z3.get_version_string(), int(timeout_ms),
            **metrics,
        )
    if result == z3.sat:
        model = None
        if capture_model:
            model = {str(decl): str(solver.model()[decl]) for decl in solver.model().decls()}
        return FormulaReceipt(
            obligation_id, "SAT", formula_hash, z3.get_version_string(), int(timeout_ms),
            model=model, **metrics,
        )
    return FormulaReceipt(
        obligation_id,
        "UNKNOWN",
        formula_hash,
        z3.get_version_string(),
        int(timeout_ms),
        reason=solver.reason_unknown(),
        **metrics,
    )


def solve_formula(
    obligation_id: str,
    formula: z3.BoolRef,
    *,
    timeout_ms: int = 0,
    capture_model: bool = False,
) -> FormulaReceipt:
    """Solve one regenerated formula in a fresh general-purpose solver."""

    return _solve_with_solver(
        obligation_id, formula, solver=make_solver(), timeout_ms=timeout_ms,
        capture_model=capture_model,
    )


def solve_qf_fp_formula(
    obligation_id: str,
    formula: z3.BoolRef,
    *,
    timeout_ms: int = 0,
    capture_model: bool = False,
) -> FormulaReceipt:
    """Solve one pure quantifier-free floating-point formula with the QF_FP engine.

    The V10.1 FP64 history obligations are pure QF_FP.  Sending a large
    disjunction of independent FP recurrences through the generic SMT solver
    causes avoidable bit-blast coupling and was observed to time out on s313.
    Keeping each recurrence separate and selecting the exact QF_FP logic is a
    solver-structure change only; the IEEE-754 formula itself is unchanged.
    """

    return _solve_with_solver(
        obligation_id, formula, solver=z3.SolverFor("QF_FP"),
        timeout_ms=timeout_ms, capture_model=capture_model,
    )


__all__ = [
    "FormulaReceipt", "canonical_formula_text", "solve_formula",
    "solve_qf_fp_formula",
]
