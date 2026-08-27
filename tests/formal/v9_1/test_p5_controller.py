import z3

from amc_py.viper.fixed_point import FixedPointConfig
from amc_py.viper.integer_tree import IntegerTreeLeaf, IntegerTreeModel
from formal_toolchain.v9_1.symbolic_state import BoundModel, TaskBound, declare_state
from formal_toolchain.v9_1.transition_encoder import encode_p5_controller


def _leaf_tree(state_dim: int, ranking: tuple[int, ...]) -> IntegerTreeModel:
    return IntegerTreeModel(
        schema_version="integer_tree_v1",
        root_node_id=0,
        state_dim=state_dim,
        action_dim=len(ranking),
        nodes=(),
        leaves=(IntegerTreeLeaf(0, ranking[0], ranking, tuple(1.0 for _ in ranking), 1, 1.0, 0.0),),
        feature_names=tuple(f"f{i}" for i in range(state_dim)),
        fixed_point_config_hash="test",
    )


def _model() -> BoundModel:
    task = TaskBound("hi", 0, 10, 10, "HI", 1, 4, 2, 1, 4)
    state_dim = 18
    actions = (
        {"action_id": 0, "action_space_type": "single", "is_noop": False,
         "increase_task": "hi", "decrease_tasks": [], "increase_ratio": 0.5, "decrease_ratio": 0.1},
        {"action_id": 1, "action_space_type": "single", "is_noop": True,
         "increase_task": None, "decrease_tasks": [], "increase_ratio": 0.5, "decrease_ratio": 0.1},
    )
    return BoundModel(
        (task,),
        5,
        action_dim=2,
        noop_id=1,
        feature_names=tuple(f"f{i}" for i in range(state_dim)),
        fixed_point_config=FixedPointConfig(scale=1000, output_max=1000),
        tree=_leaf_tree(state_dim, (0, 1)),
        action_definitions=actions,
        feature_config={"max_cost_weight": 0.7, "risk_max_scale": 3.0},
        max_jobs_per_task=1,
    )


def _bind_history(solver: z3.Solver, state) -> None:
    solver.add(state.chi.recent_cost["hi"] == 1,
               state.chi.ema_cost["hi"] == 1,
               state.chi.overrun_ema["hi"] == 0,
               state.chi.max_cost_k["hi"] == 1)
    for window in (state.chi.mode_change_window, state.chi.lo_cancel_window,
                   state.chi.hi_overrun_window, state.chi.lo_overrun_window,
                   state.chi.job_start_window):
        solver.add(*(value == 0 for value in window))


def test_p5_applies_ranked_first_valid_budget_update_when_enabled():
    model = _model()
    z, zp = declare_state("p5.z", model), declare_state("p5.zp", model)
    solver = z3.Solver()
    solver.add(encode_p5_controller(z, zp, model), z.p == 5, z.t == 10, z.budgets["hi"] == 2)
    _bind_history(solver, z)
    # increase 2 by binary64 factor 1.5 -> ceil(3.0) = 3
    solver.add(zp.budgets["hi"] != 3)
    assert solver.check() == z3.unsat


def test_p5_is_budget_identity_outside_agent_activation():
    model = _model()
    z, zp = declare_state("p5off.z", model), declare_state("p5off.zp", model)
    solver = z3.Solver()
    solver.add(encode_p5_controller(z, zp, model), z.p == 5, z.t == 11, z.budgets["hi"] == 2,
               zp.budgets["hi"] != 2)
    assert solver.check() == z3.unsat


def test_p5_falls_through_to_explicit_noop_when_ranked_action_masked():
    model = _model()
    z, zp = declare_state("p5noop.z", model), declare_state("p5noop.zp", model)
    solver = z3.Solver()
    solver.add(encode_p5_controller(z, zp, model), z.p == 5, z.t == 10, z.budgets["hi"] == 4)
    _bind_history(solver, z)
    # The increase candidate clips to the upper bound and is rejected as
    # no_effective_budget_change. Ranked FirstValid must therefore choose noop.
    solver.add(zp.budgets["hi"] != 4)
    assert solver.check() == z3.unsat
