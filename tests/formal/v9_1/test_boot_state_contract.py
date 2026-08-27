from __future__ import annotations

import z3

from formal_toolchain.v9_1.boot_state import encode_canonical_boot_state
from formal_toolchain.v9_1.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_1.symbolic_state import BoundModel, TaskBound, declare_state


def _model() -> BoundModel:
    return BoundModel(tasks=(TaskBound(
        name="hi", priority=0, period=10, deadline=10,
        criticality="HI", c_lo=2, c_hi=4, initial_budget=2,
    ),), agent_period=5, max_jobs_per_task=2)


def test_canonical_boot_is_inside_safe_prefix_invariant() -> None:
    model = _model()
    state = declare_state("boot", model)
    solver = z3.Solver()
    solver.add(*encode_canonical_boot_state(state, model))
    solver.add(z3.Not(SafePrefixInvariant(model).formula(state)))
    assert solver.check() == z3.unsat


def test_absent_unused_slot_ready_is_fixed_false_at_boot() -> None:
    model = _model()
    state = declare_state("boot.ready", model)
    solver = z3.Solver()
    solver.add(*encode_canonical_boot_state(state, model))
    solver.add(state.jobs[("hi", 1)].ready)
    assert solver.check() == z3.unsat
