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
    assert "event_layer_added_abstractions_empty" in REQUIRED_SOUNDNESS_CLAUSES
    assert "microstep_terminal_fallback_forbidden" in REQUIRED_SOUNDNESS_CLAUSES
    assert "EVENT_TO_FULL_SEGMENT_REALIZABILITY" in EVENT_TERMINAL_OBLIGATIONS
    assert "MICROSTEP_EVENT_DIFFERENTIAL_CONSISTENCY" in EVENT_TERMINAL_OBLIGATIONS
