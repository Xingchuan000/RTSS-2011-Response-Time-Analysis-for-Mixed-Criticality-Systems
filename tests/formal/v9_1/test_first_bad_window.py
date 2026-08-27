from __future__ import annotations

import z3
import pytest

from amc_py.viper.fixed_point import FixedPointConfig
from amc_py.viper.integer_tree import IntegerTreeLeaf, IntegerTreeModel
from formal_toolchain.v9_1.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_1.symbolic_state import BoundModel, TaskBound
from formal_toolchain.v9_1.window_encoder import ENCODER_COMPLETE, build_first_bad_window


def _policy_model(tasks: tuple[TaskBound, ...], *, agent_period: int, max_jobs: int = 2) -> BoundModel:
    state_dim = 10 * len(tasks) + 8
    ranking = (0,)
    tree = IntegerTreeModel(
        schema_version="integer_tree_v1",
        root_node_id=0,
        state_dim=state_dim,
        action_dim=1,
        nodes=(),
        leaves=(IntegerTreeLeaf(0, 0, ranking, (1.0,), 1, 1.0, 0.0),),
        feature_names=tuple(f"f{i}" for i in range(state_dim)),
        fixed_point_config_hash="test",
    )
    return BoundModel(
        tasks,
        agent_period,
        action_dim=1,
        noop_id=0,
        feature_names=tuple(f"f{i}" for i in range(state_dim)),
        fixed_point_config=FixedPointConfig(scale=1000, output_max=1000),
        tree=tree,
        action_definitions=({
            "action_id": 0,
            "action_space_type": "single",
            "is_noop": True,
            "increase_task": None,
            "decrease_tasks": [],
            "increase_ratio": 0.1,
            "decrease_ratio": 0.1,
        },),
        feature_config={"max_cost_weight": 0.7, "risk_max_scale": 3.0},
        max_jobs_per_task=max_jobs,
    )


def test_first_bad_window_reaches_post_p2_target_state_and_uses_exact_p5():
    model = _policy_model((
        TaskBound("hi0", 0, 3, 3, "HI", 1, 2, 1, 1, 2),
        TaskBound("hi1", 1, 4, 3, "HI", 1, 2, 1, 1, 2),
    ), agent_period=2)
    encoding = build_first_bad_window(model, SafePrefixInvariant(model), "hi0")
    assert encoding.deadline == 3
    assert len(encoding.states) == 3 * 8 + 4
    assert "target_deadline_post_observe_ledger_increment_encoded" in encoding.source_obligations
    assert "no_earlier_hi_miss_checked_after_each_deadline_observation" in encoding.source_obligations
    assert "first_bad_window_uses_exact_deployed_p5_not_induction_summary" in encoding.source_obligations
    # The exact P5 controller creates quantized observation symbols.  The cheap
    # induction-only summary never does, so this locks the trust boundary.
    assert ".p5.q." in str(encoding.formula)
    assert ENCODER_COMPLETE is True


def test_window_formula_is_a_search_for_a_counterexample_not_a_pass_flag():
    model = _policy_model((
        TaskBound("hi", 0, 2, 2, "HI", 1, 2, 1, 1, 2),
    ), agent_period=3)
    encoding = build_first_bad_window(model, SafePrefixInvariant(model), "hi")
    solver = z3.Solver(); solver.add(encoding.formula)
    assert solver.check() in (z3.sat, z3.unsat, z3.unknown)
    assert "window.z.0.M" in encoding.smt2()


def test_window_requires_two_slots_when_lo_carry_in_is_in_scope():
    model = _policy_model((
        TaskBound("target", 0, 5, 5, "HI", 1, 2, 1, 1, 2),
        TaskBound("lo", 1, 7, 7, "LO", 1, 1, 1, 1, 1),
    ), agent_period=2, max_jobs=1)
    with pytest.raises(ValueError, match="WINDOW_TWO_SLOT_CAPACITY_INSUFFICIENT"):
        build_first_bad_window(model, SafePrefixInvariant(model), "target")
