from __future__ import annotations

import z3

from amc_py.viper.fixed_point import FixedPointConfig
from amc_py.viper.integer_tree import IntegerTreeLeaf, IntegerTreeModel
from formal_toolchain.v9_2.environment_encoder import declare_environment
from formal_toolchain.v9_2.p5_summary import build_p5_summary_soundness_obligations
from formal_toolchain.v9_2.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_2.symbolic_state import BoundModel, TaskBound


def _model() -> BoundModel:
    task = TaskBound("hi", 0, 10, 10, "HI", 1, 4, 2, 1, 4)
    state_dim = 18
    ranking = (0, 1)
    tree = IntegerTreeModel(
        schema_version="integer_tree_v1",
        root_node_id=0,
        state_dim=state_dim,
        action_dim=2,
        nodes=(),
        leaves=(IntegerTreeLeaf(
            0, ranking[0], ranking, (1.0, 1.0), 1, 1.0, 0.0
        ),),
        feature_names=tuple(f"f{i}" for i in range(state_dim)),
        fixed_point_config_hash="test",
    )
    actions = (
        {
            "action_id": 0, "action_space_type": "single", "is_noop": False,
            "increase_task": "hi", "decrease_tasks": [],
            "increase_ratio": 0.5, "decrease_ratio": 0.1,
        },
        {
            "action_id": 1, "action_space_type": "single", "is_noop": True,
            "increase_task": None, "decrease_tasks": [],
            "increase_ratio": 0.5, "decrease_ratio": 0.1,
        },
    )
    return BoundModel(
        (task,),
        5,
        action_dim=2,
        noop_id=1,
        feature_names=tuple(f"f{i}" for i in range(state_dim)),
        fixed_point_config=FixedPointConfig(scale=1000, output_max=1000),
        tree=tree,
        action_definitions=actions,
        feature_config={"max_cost_weight": 0.7, "risk_max_scale": 3.0},
        max_jobs_per_task=1,
    )


def test_p5_summary_compositional_soundness_obligations_are_unsat() -> None:
    model = _model()
    for obligation in build_p5_summary_soundness_obligations(model):
        solver = z3.Solver()
        solver.add(obligation.counterexample)
        assert solver.check() == z3.unsat, obligation.obligation_id


def test_p5_safe_prefix_induction_uses_sound_overapprox_after_contract() -> None:
    model = _model()
    invariant = SafePrefixInvariant(model)
    env = declare_environment("p5.summary.ind.env", model, release_count=1)
    formula = invariant.phase_inductiveness_counterexample(
        env, 5, prefix="p5.summary.ind", use_p5_summary=True
    )
    solver = z3.Solver()
    solver.add(formula)
    assert solver.check() == z3.unsat
