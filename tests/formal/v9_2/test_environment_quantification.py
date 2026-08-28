import z3

from formal_toolchain.v9_2.environment_encoder import (
    classify_from_actual_demand,
    declare_environment,
    target_release_constraints,
)
from formal_toolchain.v9_2.symbolic_state import BoundModel, TaskBound, declare_state, well_formed


def _model():
    return BoundModel(
        tasks=(
            TaskBound("hi", 0, 10, 10, "HI", 2, 5, 2, 1, 5),
            TaskBound("lo", 1, 15, 15, "LO", 3, 3, 3, 1, 3),
        ),
        agent_period=7,
    )


def test_every_release_has_an_independent_bounded_actual_demand():
    env = declare_environment("e", _model(), release_count=3)
    assert len(env.actual_demands) == 6
    assert len({str(value) for value in env.actual_demands.values()}) == 6
    solver = z3.Solver()
    solver.add(*env.constraints)
    solver.add(env.actual_demands[("hi", 0)] == 2)
    solver.add(env.actual_demands[("hi", 1)] == 5)
    assert solver.check() == z3.sat
    assert str(classify_from_actual_demand(env.actual_demands[("hi", 0)], _model().tasks[0])) == "e.A.hi.0 > 2"


def test_target_release_keeps_exact_periodic_phase_and_controller_residue():
    model = _model()
    env = declare_environment("e", model, release_count=1)
    constraints = target_release_constraints(env, model.tasks[0])
    solver = z3.Solver()
    solver.add(*env.constraints, *constraints, env.phase.absolute_time_residue == 20)
    assert solver.check() == z3.sat
    solver.add(env.phase.absolute_time_residue == 21)
    assert solver.check() == z3.unsat


def test_symbolic_state_contains_all_jobs_and_well_formed_contract():
    model = _model()
    state = declare_state("z", model)
    assert set(state.jobs) == {("hi", 0), ("hi", 1), ("hi", 2), ("hi", 3),
                               ("lo", 0), ("lo", 1), ("lo", 2), ("lo", 3)}
    solver = z3.Solver()
    solver.add(well_formed(state, model))
    solver.add(state.t == 0, state.p == 0, state.hi_miss_ledger == 0)
    assert solver.check() == z3.sat
