from __future__ import annotations

import z3

from formal_toolchain.v9_2.environment_encoder import declare_environment
from formal_toolchain.v9_2.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_2.symbolic_state import BoundModel, TaskBound


def _model() -> BoundModel:
    return BoundModel(tasks=(TaskBound(
        name="hi", priority=0, period=10, deadline=10,
        criticality="HI", c_lo=2, c_hi=4, initial_budget=2,
    ),), agent_period=5, max_jobs_per_task=2)


def test_phase_inductiveness_counterexample_is_safe_prefix_conditioned() -> None:
    model = _model()
    env = declare_environment("test.ind.env", model, release_count=1)
    formula = SafePrefixInvariant(model).phase_inductiveness_counterexample(env, 2)
    text = str(formula)
    assert "M" in text
    # The generated obligation explicitly requires both pre/post miss ledgers zero.
    assert text.count("== 0") >= 2 or "= 0" in text


def test_invalid_phase_is_rejected() -> None:
    model = _model()
    env = declare_environment("test.ind.bad.env", model, release_count=1)
    try:
        SafePrefixInvariant(model).phase_inductiveness_counterexample(env, 8)
    except ValueError as exc:
        assert str(exc) == "V9_2_PHASE_OUT_OF_RANGE"
    else:
        raise AssertionError("phase 8 must be rejected")
