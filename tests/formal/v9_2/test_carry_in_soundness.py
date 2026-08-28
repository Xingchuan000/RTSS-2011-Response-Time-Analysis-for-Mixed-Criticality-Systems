import z3

from formal_toolchain.v9_2.carry_in import (
    build_carry_in_summary, check_carry_in_summary_soundness, encode_carry_in_adequacy,
)
from formal_toolchain.v9_2.symbolic_state import BoundModel, TaskBound


def _model():
    return BoundModel((TaskBound("hi", 0, 5, 5, "HI", 1, 3, 1, 1, 3),), 2)


def test_carry_in_has_window_capacity_and_explicit_saturation():
    summary = build_carry_in_summary(_model(), window_length=5)
    result = check_carry_in_summary_soundness(summary, _model())
    assert result["status"] == "PASS"
    assert result["explicit_slots"] == {"hi": 5}
    assert result["obligation"] == "CARRY_IN_SUMMARY_ADEQUACY"


def test_saturated_tail_cannot_be_used_as_an_untracked_completion():
    model = _model(); summary = build_carry_in_summary(model, window_length=2)
    adequacy = encode_carry_in_adequacy(summary, model)
    solver = z3.Solver(); solver.add(*adequacy.constraints, summary.saturated_tail["hi"])
    first = summary.explicit_jobs["hi"][0]
    solver.add(first.completion_observable)
    assert solver.check() == z3.unsat
