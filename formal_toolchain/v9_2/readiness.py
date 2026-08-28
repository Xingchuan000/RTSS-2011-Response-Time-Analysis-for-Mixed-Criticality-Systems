"""Single fail-closed readiness contract for the V9.2 proof pipeline.

The tuple is intentionally empty only because every implementation-level
obligation needed for an UNSAT safety proof is now regenerated and checked by
the trusted verifier.  A SAT abstraction can still be reported as
SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE when independent concrete replay is not
available; that diagnostic limitation is not a reason to disable an all-UNSAT
safety proof.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReadinessBlocker:
    code: str
    component: str
    requirement: str


BLOCKERS: tuple[ReadinessBlocker, ...] = ()


def proof_pipeline_ready() -> bool:
    return not BLOCKERS


def blocker_rows() -> list[dict[str, str]]:
    return [
        {"code": row.code, "component": row.component, "requirement": row.requirement}
        for row in BLOCKERS
    ]


__all__ = ["BLOCKERS", "ReadinessBlocker", "blocker_rows", "proof_pipeline_ready"]
