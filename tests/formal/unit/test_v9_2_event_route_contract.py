from pathlib import Path

from formal_toolchain.v9_2.constants import CANONICAL_PHASES, PROOF_ROUTE, RESULT_PROVED
from formal_toolchain.v9_2.kernel import Phase, next_zero_time_phase

ROOT = Path(__file__).parents[3]


def test_v9_2_single_event_route_and_canonical_phase_order():
    assert PROOF_ROUTE == "POLICY_CONSTRAINED_EVENT_REFINED_SAFE_PREFIX_HI_SAFETY_V9_2"
    assert RESULT_PROVED == "DEPLOYED_TREE_PROVED_P0"
    assert CANONICAL_PHASES == tuple(phase.name for phase in Phase)
    for phase in list(Phase)[:-1]:
        assert int(next_zero_time_phase(phase)) == int(phase) + 1


def test_active_cli_has_no_legacy_route_selection():
    text = (ROOT / "formal_toolchain/cli/prove_seed.py").read_text(encoding="utf-8")
    for legacy in (
        "--proof-route", "v8_auto", "protected_prefix", "raw_protected_prefix",
        "strict_full", "refresh-phase-k-map",
    ):
        assert legacy not in text


def test_event_terminal_route_has_no_microstep_fallback():
    verifier = (ROOT / "formal_toolchain/v9_2/verifier.py").read_text(encoding="utf-8")
    event_window = (ROOT / "formal_toolchain/v9_2/event_window_encoder.py").read_text(encoding="utf-8")
    assert "build_incremental_event_first_bad_window" in verifier
    assert "solve_incremental_event_window" in verifier
    assert "build_first_bad_window" not in verifier
    assert '"candidate_assertions_trusted": False' in verifier
    assert '"microstep_terminal_fallback_used": False' in verifier
    assert "encode_p5_controller" in (ROOT / "formal_toolchain/v9_2/event_kernel.py").read_text(encoding="utf-8")
    assert "encode_p5_invariant_summary" not in event_window


def test_terminal_event_window_is_unlimited_but_global_timeout_remains_available():
    verifier = (ROOT / "formal_toolchain/v9_2/verifier.py").read_text(encoding="utf-8")
    bmc = (ROOT / "formal_toolchain/v9_2/incremental_event_bmc.py").read_text(encoding="utf-8")
    cli = (ROOT / "formal_toolchain/cli/prove_seed.py").read_text(encoding="utf-8")
    assert "TERMINAL_EVENT_WINDOW_TIMEOUT_MS = 0" in verifier
    assert "timeout_ms=TERMINAL_EVENT_WINDOW_TIMEOUT_MS" in verifier
    assert "probe_timeout_ms=int(timeout_ms)" in verifier
    assert "global_obligation_timeout_ms=int(timeout_ms)" in verifier
    # V5 reuses one solver within a depth.  Setting timeout=0 explicitly
    # clears a previous bounded probe timeout before exact-leaf resume.
    assert "solver.set(timeout=timeout_ms)" in bmc
    assert "probe timeout must be positive" in bmc
    assert "0 means unlimited" in bmc
    assert "same_solver_timeout_resume" in bmc
    assert "--solver-timeout-ms" in cli
