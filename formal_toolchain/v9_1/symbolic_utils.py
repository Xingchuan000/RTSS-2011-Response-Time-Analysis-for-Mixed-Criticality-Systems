"""Small helpers shared by the V9.1 symbolic encoders."""

from __future__ import annotations

import z3


def exactly_one(terms: list[z3.BoolRef] | tuple[z3.BoolRef, ...]) -> z3.BoolRef:
    if not terms:
        raise ValueError("exactly_one requires at least one term")
    return z3.PbEq([(term, 1) for term in terms], 1)


def ite_chain(choices: list[tuple[z3.BoolRef, z3.ExprRef]], default: z3.ExprRef) -> z3.ExprRef:
    result = default
    for condition, value in reversed(choices):
        result = z3.If(condition, value, result)
    return result


__all__ = ["exactly_one", "ite_chain"]
