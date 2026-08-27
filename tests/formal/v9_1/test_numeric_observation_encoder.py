import z3

from amc_py.viper.fixed_point import FixedPointConfig
from formal_toolchain.v9_1.action_encoder import encode_first_valid_explicit_noop
from formal_toolchain.v9_1.mask_encoder import encode_action_mask
from formal_toolchain.v9_1.numeric_encoder import encode_quantized_feature
from formal_toolchain.v9_1.symbolic_state import BoundModel, TaskBound, declare_state


def _model():
    return BoundModel((TaskBound("hi", 0, 10, 10, "HI", 1, 3, 1, 1, 3),), 5,
                      action_dim=2, noop_id=1,
                      fixed_point_config=FixedPointConfig(scale=1000, output_max=1000),
                      max_jobs_per_task=1)


def test_binary64_quantizer_is_bounded_and_allows_boundary_envelope():
    q, relation = encode_quantized_feature(z3.FPVal(0.5, z3.Float64()), _model().fixed_point_config, name="q")
    solver = z3.Solver(); solver.add(*relation, q != 500)
    assert solver.check() == z3.sat
    solver = z3.Solver(); solver.add(*relation, q < 499, q > 501)
    assert solver.check() == z3.unsat


def test_action_mask_and_firstvalid_share_explicit_noop():
    model = _model()
    state = declare_state("z", model)
    actions = (
        {"action_id": 0, "target_task": "hi", "direction": "increase", "is_noop": False},
        {"action_id": 1, "is_noop": True},
    )
    masks, candidates, constraints = encode_action_mask(state.budgets, actions, model)
    selected, selector = encode_first_valid_explicit_noop((0, 1), masks, action_dim=2, noop_id=1)
    solver = z3.Solver(); solver.add(*constraints, *selector, state.budgets["hi"] == 3, selected != 1)
    assert solver.check() == z3.unsat


def test_unsupported_dynamic_action_does_not_become_noop():
    import pytest
    model = _model()
    state = declare_state("z", model)
    with pytest.raises(ValueError, match="ACTION_SEMANTICS_UNBOUND"):
        encode_action_mask(state.budgets, ({"is_residual_ranked": True, "is_noop": False},
                                            {"is_noop": True}), model)
