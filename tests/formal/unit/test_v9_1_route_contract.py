from pathlib import Path

from formal_toolchain.v9_1.constants import CANONICAL_PHASES, PROOF_ROUTE, RESULT_PROVED
from formal_toolchain.v9_1.kernel import (
    Phase, classify_hi_release, effective_demand_hi, effective_demand_lo_degraded,
    effective_demand_lo_primary, next_zero_time_phase,
)

ROOT = Path(__file__).parents[3]


def test_v9_1_normative_ids_and_phase_order():
    assert PROOF_ROUTE == "POLICY_CONSTRAINED_SAFE_PREFIX_HI_SAFETY_V9_1"
    assert RESULT_PROVED == "DEPLOYED_TREE_PROVED_P0"
    assert CANONICAL_PHASES == tuple(phase.name for phase in Phase)
    for phase in list(Phase)[:-1]:
        assert int(next_zero_time_phase(phase)) == int(phase) + 1


def test_v9_1_demand_semantics_are_not_hi_truncating():
    assert effective_demand_hi(7) == 7
    assert effective_demand_lo_primary(10, 3) == 4
    assert effective_demand_lo_degraded(10, 2) == 2
    assert classify_hi_release(2, 2, 5) == "NORMAL"
    assert classify_hi_release(3, 2, 5) == "ABNORMAL"


def test_active_cli_has_no_v8_route_or_phase_k_compatibility():
    text = (ROOT / "formal_toolchain/cli/prove_seed.py").read_text(encoding="utf-8")
    for legacy in ("--proof-route", "v8_auto", "protected_prefix", "raw_protected_prefix", "strict_full", "refresh-phase-k-map"):
        assert legacy not in text


def test_active_v9_verifier_has_no_rta_terminal():
    text = (ROOT / "formal_toolchain/v9_1/verifier.py").read_text(encoding="utf-8")
    for legacy in ("FULL_ALL_TASK_REFERENCE_RTA", "RAW_PREFIX_ALL_TASK_RTA", "PROTECTED_PREFIX_ALL_TASK_RTA"):
        assert legacy not in text
    assert "WINDOW_ENCODING_UNRESOLVED" in text
    assert "V9_1_TRUSTED_PROOF_REGENERATION_UNBOUND" not in text
    assert "build_first_bad_window" in text
    assert "solve_formula" in text
    assert '"candidate_assertions_trusted": False' in text
    assert "replay_unsat" not in text
