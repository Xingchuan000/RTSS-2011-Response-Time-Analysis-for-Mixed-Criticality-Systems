"""Required coverage of the V9.1 first-HI-miss finite-window encoder."""

from .window_encoder import ENCODER_COMPLETE, ENCODER_VERSION

WINDOW_ENCODER_VERSION = ENCODER_VERSION
# The implementation owns this gate; a verifier cannot be opened by editing a
# second copy of the flag in the contract module.
WINDOW_ENCODER_IMPLEMENTED = ENCODER_COMPLETE

REQUIRED_SOUNDNESS_CLAUSES = (
    "all_admissible_release_choices_quantified",
    "all_admissible_classifications_quantified",
    "all_admissible_actual_demands_quantified",
    "release_eligibility_eta_encoded",
    "canonical_eight_phase_order_encoded",
    "same_timestamp_closure_encoded",
    "numeric_observation_tree_mask_firstvalid_noop_budget_update_exact",
    "p3_release_snapshot_before_p4_p5_encoded",
    "mode_switch_and_idle_recovery_encoded",
    "fixed_priority_dispatch_and_tie_break_encoded",
    "p7_exactly_one_integer_service_quantum",
    "hi_demand_not_truncated",
    "deadline_observe_only_and_incomplete_hi_not_removed",
    "carry_in_summary_adequate_or_none",
    "no_earlier_hi_miss_strictly_before_target_timestamp",
    "target_deadline_observe_encoded",
    "window_start_requires_psi_no_prior_miss_and_target_eligibility",
    "finite_environment_prefix_extendable_to_full_admissible_environment",
)
