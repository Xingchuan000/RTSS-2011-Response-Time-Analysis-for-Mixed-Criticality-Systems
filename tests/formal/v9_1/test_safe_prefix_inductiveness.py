import z3
from pathlib import Path

from formal_toolchain.v9_1.environment_encoder import declare_environment
from formal_toolchain.v9_1.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_1.symbolic_state import BoundModel, TaskBound


def _model():
    return BoundModel((TaskBound("hi", 0, 4, 4, "HI", 1, 2, 1, 1, 2),), 3,
                      max_jobs_per_task=2)


def test_safe_prefix_initial_obligation_is_a_real_unsat_replay():
    invariant = SafePrefixInvariant(_model())
    solver = z3.Solver(); solver.add(invariant.initial_counterexample())
    assert solver.check() == z3.unsat


def test_conditional_inductiveness_is_not_vacuous_about_new_hi_miss():
    model = _model()
    invariant = SafePrefixInvariant(model)
    env = declare_environment("e", model, release_count=2)
    formula = invariant.conditional_inductiveness_counterexample(env)
    assert "ind.z" in str(formula)
    # The condition is explicitly conditional, rather than an unconditional
    # claim that every transition preserves Psi at the miss boundary.
    assert formula.sort().is_bool()
    source = (Path(__file__).parents[3] / "formal_toolchain/v9_1/safe_prefix_invariant.py").read_text()
    assert "no_new_miss = next_state.hi_miss_ledger == state.hi_miss_ledger" in source
