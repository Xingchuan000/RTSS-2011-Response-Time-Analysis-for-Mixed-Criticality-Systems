from formal_toolchain.v9_1.encoding_contract import WINDOW_ENCODER_IMPLEMENTED, WINDOW_ENCODER_VERSION
from formal_toolchain.v9_1.window_encoder import ENCODER_COMPLETE, ENCODER_READINESS_GAPS, ENCODER_VERSION


def test_verifier_gate_is_derived_from_window_encoder_and_fail_closed():
    assert WINDOW_ENCODER_VERSION == ENCODER_VERSION
    assert WINDOW_ENCODER_IMPLEMENTED is ENCODER_COMPLETE
    assert ENCODER_COMPLETE is False
    assert "V9_1_CONTROLLER_PHASE_SYMBOLIC_POLICY_UNBOUND" not in ENCODER_READINESS_GAPS
    assert "V9_1_PROOF_REGENERATION_IN_VERIFIER_UNBOUND" in ENCODER_READINESS_GAPS
