from formal_toolchain.v9_2.encoding_contract import (
    EVENT_TERMINAL_OBLIGATIONS,
    EVENT_WINDOW_ENCODER_IMPLEMENTED,
    EVENT_WINDOW_ENCODER_VERSION,
    REQUIRED_SOUNDNESS_CLAUSES,
)
from formal_toolchain.v9_2.event_window_encoder import ENCODER_COMPLETE, ENCODER_VERSION


def test_event_encoder_is_the_only_terminal_finite_realization():
    assert EVENT_WINDOW_ENCODER_VERSION == ENCODER_VERSION
    assert EVENT_WINDOW_ENCODER_IMPLEMENTED is ENCODER_COMPLETE
    assert ENCODER_COMPLETE is True
    assert "target_local_fixed_priority_interference_dominance" in REQUIRED_SOUNDNESS_CLAUSES
    assert "lazy_release_demand_independence_exact" in REQUIRED_SOUNDNESS_CLAUSES
    assert "controller_policy_case_partition_exact" in REQUIRED_SOUNDNESS_CLAUSES
    assert "single_event_graph_route_no_terminal_fallback" in REQUIRED_SOUNDNESS_CLAUSES
    assert "FULL_TO_PROJECTED_EVENT_PREFIX_SIMULATION" in EVENT_TERMINAL_OBLIGATIONS
    assert "FIRST_HI_BAD_PROJECTED_EVENT_REFLECTION" in EVENT_TERMINAL_OBLIGATIONS
