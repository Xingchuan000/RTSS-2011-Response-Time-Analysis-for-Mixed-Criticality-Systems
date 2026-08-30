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


def test_event_terminal_route_is_explicit_graph_only():
    verifier = (ROOT / "formal_toolchain/v9_2/verifier.py").read_text(encoding="utf-8")
    graph = (ROOT / "formal_toolchain/v9_2/event_graph_solver.py").read_text(encoding="utf-8")
    assert "build_event_graph_problem" in verifier
    assert "solve_event_graph" in verifier
    assert "solve_incremental_event_window" not in verifier
    assert "for depth in range(" not in graph
    assert "FRESH_SPECIALIZED_LEAF" not in graph
    assert "timeout" not in graph.lower()
    assert not (ROOT / "formal_toolchain/v9_2/incremental_event_bmc.py").exists()
