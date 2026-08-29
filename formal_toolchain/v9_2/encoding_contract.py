"""Required coverage of the V9.2 exact Event FirstBadWindow realization."""

from .event_refinement import EVENT_TERMINAL_OBLIGATIONS
from .event_window_encoder import ENCODER_COMPLETE, ENCODER_VERSION

EVENT_WINDOW_ENCODER_VERSION = ENCODER_VERSION
EVENT_WINDOW_ENCODER_IMPLEMENTED = ENCODER_COMPLETE

REQUIRED_SOUNDNESS_CLAUSES = (
    "all_admissible_release_choices_quantified",
    "all_admissible_classifications_quantified",
    "all_admissible_actual_demands_quantified",
    "canonical_p0_p7_full_kernel_retained",
    "event_start_exact_full_p0_projection",
    "event_state_future_sufficient_without_new_aggregation",
    "next_event_exact_minimum",
    "no_skipped_discrete_event",
    "no_spurious_event_source",
    "silent_interval_service_and_eta_exact",
    "release_event_coverage",
    "hi_deadline_event_coverage",
    "selected_completion_event_coverage",
    "controller_event_coverage",
    "exact_p5_at_controller_event",
    "exact_controller_pool_formula_factoring",
    "indexed_demand_lookup_formula_factoring",
    "phase_ssa_frame_elimination_formula_equivalence",
    "incremental_terminal_depth_partition_exact",
    "full_to_event_segment_simulation",
    "event_to_full_segment_realizability",
    "first_hi_bad_event_prefix_bi_reflection",
    "finite_event_count_bound_structural",
    "event_layer_added_abstractions_empty",
    "small_horizon_full_event_differential_consistency",
)

__all__ = [
    "EVENT_TERMINAL_OBLIGATIONS",
    "EVENT_WINDOW_ENCODER_IMPLEMENTED",
    "EVENT_WINDOW_ENCODER_VERSION",
    "REQUIRED_SOUNDNESS_CLAUSES",
]
