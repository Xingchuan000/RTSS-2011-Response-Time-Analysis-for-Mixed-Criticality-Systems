"""Small solver-construction policy for the V9.3 research verifier."""

from __future__ import annotations

import z3


DEFAULT_Z3_THREADS = 3

def make_solver(*, ctx: z3.Context | None = None, threads: int = DEFAULT_Z3_THREADS) -> z3.Solver:
    solver = z3.Solver(ctx=ctx) if ctx is not None else z3.Solver()
    solver.set("threads", int(threads))
    return solver


__all__ = ["DEFAULT_Z3_THREADS", "make_solver"]
