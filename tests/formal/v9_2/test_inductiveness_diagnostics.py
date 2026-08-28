from __future__ import annotations

from formal_toolchain.v9_2.environment_encoder import declare_environment
from formal_toolchain.v9_2.safe_prefix_invariant import SafePrefixInvariant
from formal_toolchain.v9_2.symbolic_state import BoundModel, TaskBound


def test_phase_diagnostics_cover_every_named_psi_clause() -> None:
    model = BoundModel(
        (TaskBound("hi", 0, 10, 10, "HI", 2, 4, 2, 1, 4),),
        agent_period=5,
        max_jobs_per_task=2,
    )
    invariant = SafePrefixInvariant(model)
    env = declare_environment("diag.env", model, release_count=1)
    rows = invariant.phase_inductiveness_clause_counterexamples(env, 3, prefix="diag")
    assert set(rows) == {
        "state_well_formedness",
        "budget_bounds",
        "exact_periodic_eta",
        "job_field_consistency",
        "no_prior_hi_miss_consistency",
        "settled_job_consistency",
        "frontier_consistency",
        "history_bounds",
        "carry_in_consistency",
    }
