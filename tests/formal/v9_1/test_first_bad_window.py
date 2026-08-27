import z3

from formal_toolchain.v9_1.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_1.symbolic_state import BoundModel, TaskBound
from formal_toolchain.v9_1.window_encoder import ENCODER_COMPLETE, build_first_bad_window


def test_first_bad_window_unrolls_to_target_p2_without_forbidding_simultaneous_misses():
    model = BoundModel((
        TaskBound("hi0", 0, 3, 3, "HI", 1, 2, 1, 1, 2),
        TaskBound("hi1", 1, 4, 3, "HI", 1, 2, 1, 1, 2),
    ), 2, max_jobs_per_task=3)
    encoding = build_first_bad_window(model, SafePrefixInvariant(model), "hi0")
    assert encoding.deadline == 3
    assert len(encoding.states) == 3 * 8 + 3
    assert "target_deadline_observe_encoded" in encoding.source_obligations
    assert "no_earlier_hi_miss_strictly_before_target_timestamp" in encoding.source_obligations
    assert ENCODER_COMPLETE is False


def test_window_formula_is_a_search_for_a_counterexample_not_a_pass_flag():
    model = BoundModel((TaskBound("hi", 0, 2, 2, "HI", 1, 2, 1, 1, 2),), 3,
                       max_jobs_per_task=2)
    encoding = build_first_bad_window(model, SafePrefixInvariant(model), "hi")
    solver = z3.Solver(); solver.add(encoding.formula)
    assert solver.check() in (z3.sat, z3.unsat, z3.unknown)
    assert "(= window.z.0.M 0)" in encoding.smt2()


def test_window_refuses_silent_release_slot_underapproximation():
    import pytest

    model = BoundModel((
        TaskBound("fast", 0, 1, 1, "HI", 1, 1, 1, 1, 1),
        TaskBound("target", 1, 5, 5, "HI", 1, 2, 1, 1, 2),
    ), 2, max_jobs_per_task=4)
    with pytest.raises(ValueError, match="WINDOW_RELEASE_SLOT_CAPACITY_INSUFFICIENT"):
        build_first_bad_window(model, SafePrefixInvariant(model), "target")
