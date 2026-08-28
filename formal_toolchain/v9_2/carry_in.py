"""Machine-checkable two-slot carry-in adequacy obligations for V9.2."""

from __future__ import annotations

from dataclasses import dataclass

import z3

from .symbolic_state import BoundModel


@dataclass(frozen=True, slots=True)
class CarryInObligation:
    obligation_id: str
    counterexample: z3.BoolRef
    explanation: str


def build_two_slot_carry_in_obligations(model: BoundModel, *, prefix: str = "carry") -> tuple[CarryInObligation, ...]:
    """Return closed counterexample formulas; every one must be UNSAT.

    LO jobs of one task have identical fixed priority.  Because policy history is
    conservatively abstracted at P5, their only safety-relevant effect is total
    remaining processor work.  Slot 0 therefore aggregates all older LO work,
    while slot 1 is the exact most recent periodic release.
    """

    obligations: list[CarryInObligation] = []
    for task in model.tasks:
        if task.deadline > task.period:
            obligations.append(CarryInObligation(
                f"D_LE_T_{task.name}",
                z3.BoolVal(True),
                "two-slot V9.2 proof domain requires constrained deadlines",
            ))
            continue
        if task.criticality == "LO":
            agg = z3.Int(f"{prefix}.{task.name}.aggregate")
            exact = z3.Int(f"{prefix}.{task.name}.exact")
            concrete_after = z3.If(
                agg > 0,
                agg - 1 + exact,
                z3.If(exact > 0, exact - 1, 0),
            )
            folded = agg + exact
            aggregate_after = z3.If(folded > 0, folded - 1, 0)
            obligations.append(CarryInObligation(
                f"LO_AGGREGATE_ONE_QUANTUM_WORK_PRESERVATION_{task.name}",
                z3.And(agg >= 0, exact >= 0, concrete_after != aggregate_after),
                "folding old same-priority LO jobs preserves remaining processor work",
            ))
        else:
            release = z3.Int(f"{prefix}.{task.name}.release")
            now = release + task.period
            deadline = release + task.deadline
            service = z3.Int(f"{prefix}.{task.name}.service")
            demand = z3.Int(f"{prefix}.{task.name}.demand")
            incomplete = service < demand
            no_prior_miss = z3.Not(z3.And(deadline <= now, incomplete))
            obligations.append(CarryInObligation(
                f"HI_SINGLE_ACTIVE_JOB_{task.name}",
                z3.And(
                    release >= 0,
                    service >= 0,
                    demand >= 1,
                    task.deadline <= task.period,
                    incomplete,
                    no_prior_miss,
                ),
                "an incomplete previous HI job at its next release contradicts NoPriorHIMiss when D<=T",
            ))
    return tuple(obligations)


__all__ = ["CarryInObligation", "build_two_slot_carry_in_obligations"]
