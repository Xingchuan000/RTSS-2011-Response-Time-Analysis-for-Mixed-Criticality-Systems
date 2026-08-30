"""Required coverage of the V9.3 target-local Event-graph realization."""

from .event_refinement import EVENT_TERMINAL_OBLIGATIONS
from .event_window_encoder import ENCODER_COMPLETE, ENCODER_VERSION

EVENT_WINDOW_ENCODER_VERSION = ENCODER_VERSION
EVENT_WINDOW_ENCODER_IMPLEMENTED = ENCODER_COMPLETE

REQUIRED_SOUNDNESS_CLAUSES = (
    "all_admissible_release_choices_quantified",
    "all_admissible_classifications_quantified",
    "all_admissible_actual_demands_quantified",
    "canonical_p0_p7_full_kernel_retained",
    "event_start_target_local_safe_prefix_sound",
    "target_local_fixed_priority_interference_dominance",
    "target_priority_prefix_jobs_and_eta_retained",
    "full_policy_budget_and_history_state_retained",
    "lazy_release_demand_independence_exact",
    "event_node_closure_reuses_canonical_p0_p6",
    "controller_policy_case_partition_exact",
    "next_event_exact_minimum",
    "no_target_or_higher_priority_event_skipped",
    "silent_interval_service_and_eta_exact",
    "exact_p5_at_controller_event",
    "full_to_projected_event_prefix_simulation",
    "first_hi_bad_full_to_projected_reflection",
    "finite_event_count_bound_structural",
    "sat_projected_path_requires_full_replay",
    "single_event_graph_route_no_terminal_fallback",
)

__all__ = [
    "EVENT_TERMINAL_OBLIGATIONS",
    "EVENT_WINDOW_ENCODER_IMPLEMENTED",
    "EVENT_WINDOW_ENCODER_VERSION",
    "REQUIRED_SOUNDNESS_CLAUSES",
]
