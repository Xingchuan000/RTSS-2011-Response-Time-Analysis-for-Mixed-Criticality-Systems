from __future__ import annotations

import z3

from formal_toolchain.v9_2.environment_encoder import declare_environment
from formal_toolchain.v9_2.invariant_templates import build_psi, job_field_consistency
from formal_toolchain.v9_2.symbolic_state import BoundModel, TaskBound, declare_state
from formal_toolchain.v9_2.transition_encoder import encode_p3_arrival_freeze


def _model() -> BoundModel:
    # Mirrors the important s313 relation: controller floor 0.9*C_LO while
    # C-AMC-sem degraded release snapshot is about 0.5*C_LO.
    lo = TaskBound(
        "lo", 0, 100, 100, "LO", 100, 100,
        initial_budget=100, budget_floor=90, action_hard_upper=200,
        degraded_cost=50, actual_demand_min=1, actual_demand_max=150,
    )
    return BoundModel((lo,), agent_period=25, max_jobs_per_task=2)


def test_p3_degraded_lo_snapshot_preserves_psi_even_below_action_floor() -> None:
    model = _model()
    env = declare_environment("deg.env", model, release_count=1)
    z = declare_state("deg.z", model)
    zp = declare_state("deg.zp", model)
    solver = z3.Solver()
    solver.add(
        build_psi(z, model),
        z.p == 3,
        z.t == 0,
        z.mode_hi,
        z.eta["lo"] == 100,
        z.budgets["lo"] == 100,
        z.hi_miss_ledger == 0,
        env.phase.origin_time == z.t,
        env.actual_demands[("lo", 0)] == 120,
        encode_p3_arrival_freeze(z, zp, model, env),
        zp.hi_miss_ledger == 0,
        z3.Not(build_psi(zp, model)),
    )
    assert solver.check() == z3.unsat


def test_exact_degraded_lo_snapshot_is_cdeg_not_controller_floor() -> None:
    model = _model()
    z = declare_state("snapshot.z", model)
    job = z.jobs[("lo", 1)]
    solver = z3.Solver()
    solver.add(
        job.present,
        job.release_index == 1,
        job.release_time == 100,
        job.absolute_deadline == 200,
        job.tie_break == 1,
        job.release_entry_mode_hi,
        job.actual_demand == 120,
        job.effective_demand == 50,
        job.budget_at_release == 50,
        job_field_consistency(z, model),
    )
    assert solver.check() == z3.sat


def test_exact_primary_lo_snapshot_still_obeys_controller_budget_bounds() -> None:
    model = _model()
    z = declare_state("primary.z", model)
    job = z.jobs[("lo", 1)]
    solver = z3.Solver()
    solver.add(
        job.present,
        job.release_index == 1,
        job.release_time == 100,
        job.absolute_deadline == 200,
        job.tie_break == 1,
        z3.Not(job.release_entry_mode_hi),
        job.actual_demand == 120,
        job.effective_demand == 81,
        job.budget_at_release == 80,
        job_field_consistency(z, model),
    )
    assert solver.check() == z3.unsat
