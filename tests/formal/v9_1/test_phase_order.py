import z3

from formal_toolchain.v9_1.environment_encoder import declare_environment
from formal_toolchain.v9_1.symbolic_state import BoundModel, TaskBound, declare_state
from formal_toolchain.v9_1.transition_encoder import (
    encode_p2_deadline_observe, encode_p3_arrival_freeze, encode_p7_time_and_service, encode_step,
)


def _model():
    return BoundModel((TaskBound("hi", 0, 4, 4, "HI", 1, 2, 1, 1, 2),), 3, max_jobs_per_task=2)


def test_only_p7_advances_time_and_resets_phase():
    model = _model()
    env = declare_environment("e", model, release_count=2)
    z, zp = declare_state("z", model), declare_state("zp", model)
    solver = z3.Solver()
    solver.add(encode_p7_time_and_service(z, zp, model), z.t == 0, zp.t != 1)
    assert solver.check() == z3.unsat


def test_p2_observes_without_removing_incomplete_hi_job():
    model = _model()
    z, zp = declare_state("z", model), declare_state("zp", model)
    job = z.jobs[("hi", 0)]
    solver = z3.Solver()
    solver.add(encode_p2_deadline_observe(z, zp, model), z.t == 4,
               job.present, job.absolute_deadline == 4, job.effective_demand == 2,
               job.executed_service == 0, z.hi_miss_ledger == 0, zp.hi_miss_ledger != 1)
    assert solver.check() == z3.unsat


def test_phase_relation_rejects_swapping_p2_and_p3():
    model = _model()
    env = declare_environment("e", model, release_count=2)
    z, zp = declare_state("z", model), declare_state("zp", model)
    solver = z3.Solver()
    solver.add(encode_step(z, zp, model, env), z.p == 2, zp.p == 4)
    assert solver.check() == z3.unsat


def test_p3_freezes_release_snapshot_before_later_phases():
    model = _model()
    env = declare_environment("e", model, release_count=2)
    z, zp = declare_state("z", model), declare_state("zp", model)
    job = zp.jobs[("hi", 0)]
    solver = z3.Solver()
    solver.add(encode_p3_arrival_freeze(z, zp, model, env), z.p == 3, z.t == 0,
               z.eta["hi"] == 4, z.mode_hi == z3.BoolVal(False),
               z.budgets["hi"] == 1, env.actual_demands[("hi", 0)] == 2,
               z3.Not(z.jobs[("hi", 0)].present),
               job.present, job.budget_at_release != 1)
    assert solver.check() == z3.unsat


def test_p6_dispatch_uses_symbolic_tie_break_not_slot_number():
    from formal_toolchain.v9_1.transition_encoder import encode_p6_dispatch

    model = BoundModel((TaskBound("hi", 0, 4, 4, "HI", 1, 2, 1, 1, 2),), 3,
                       max_jobs_per_task=2)
    z, zp = declare_state("dispatch.z", model), declare_state("dispatch.zp", model)
    first = z.jobs[("hi", 0)]
    second = z.jobs[("hi", 1)]
    solver = z3.Solver()
    solver.add(encode_p6_dispatch(z, zp, model), z.p == 6,
               first.present, first.ready, z3.Not(first.removed),
               first.effective_demand == 2, first.executed_service == 0,
               second.present, second.ready, z3.Not(second.removed),
               second.effective_demand == 2, second.executed_service == 0,
               first.tie_break == 10, second.tie_break == 5,
               zp.frontier.selected_slot != 1)
    assert solver.check() == z3.unsat
