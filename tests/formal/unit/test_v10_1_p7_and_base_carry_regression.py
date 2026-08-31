from pathlib import Path
from types import SimpleNamespace

from formal_toolchain.v10_1.base_section4_1 import prove_original_c_amc_sem_section4_1
from formal_toolchain.v10_1.periodic_release import p7_eta_residue_counterexample

ROOT = Path(__file__).resolve().parents[3]


def test_p7_induction_is_clause_decomposed_and_uses_sparse_successor():
    safe = (ROOT / "formal_toolchain/v10_1/safe_prefix.py").read_text(encoding="utf-8")
    verifier = (ROOT / "formal_toolchain/v10_1/verifier.py").read_text(encoding="utf-8")
    assert "p7_clause_inductiveness_obligations" in safe
    assert "declare_sparse_successor" in safe
    assert 'mutable=frozenset({"t", "eta", "frontier", "jobs.executed_service"})' in safe
    assert "A & not(And(C_i))" in safe
    assert "CONJUNCTS_WITH_SPARSE_P7_SSA" in verifier
    assert 'SAFE_PREFIX_INDUCTIVE_P7::{clause_name}' in safe


def test_p7_exact_periodic_eta_uses_complete_finite_residue_proof_not_smt():
    safe = (ROOT / "formal_toolchain/v10_1/safe_prefix.py").read_text(encoding="utf-8")
    verifier = (ROOT / "formal_toolchain/v10_1/verifier.py").read_text(encoding="utf-8")
    assert 'clause_name == "exact_periodic_eta"' in safe
    assert "p7_eta_residue_counterexample" in safe
    assert "EXHAUSTIVE_PERIOD_RESIDUE_ENUMERATION" in safe
    assert "certify_p7_exact_periodic_eta(model)" in verifier
    for period in (1, 2, 3, 5, 17, 5500, 100000):
        assert p7_eta_residue_counterexample(period) is None


def test_section4_1_receipt_exports_successful_prefix_completion_envelopes():
    model = SimpleNamespace(
        tasks=(
            SimpleNamespace(
                name="hi0", priority=0, period=10, deadline=10,
                criticality="HI", c_lo=1, c_hi=2, degraded_cost=None,
                actual_demand_upper=2,
            ),
            SimpleNamespace(
                name="lo1", priority=1, period=20, deadline=20,
                criticality="LO", c_lo=2, c_hi=2, degraded_cost=1,
                actual_demand_upper=2,
            ),
            SimpleNamespace(
                name="lo_bad", priority=2, period=20, deadline=20,
                criticality="LO", c_lo=30, c_hi=30, degraded_cost=30,
                actual_demand_upper=30,
            ),
        )
    )
    result = prove_original_c_amc_sem_section4_1(model)
    assert result["status"] == "UNRESOLVED"
    assert result["completion_bound_by_task"]["hi0"] <= 10
    assert result["completion_bound_by_task"]["lo1"] <= 20
    assert "lo_bad" not in result["completion_bound_by_task"]
    for name, row in result["completion_bound_basis_by_task"].items():
        assert row["completion_bound"] == max(row["R_LO_eq4"], row["R_HI_eq15"])
        assert result["completion_bound_by_task"][name] == row["completion_bound"]


def test_pcssc_merges_base_completion_envelopes_before_carry_in_gate():
    pcssc = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    verifier = (ROOT / "formal_toolchain/v10_1/verifier.py").read_text(encoding="utf-8")
    assert "base_completion_by_task" in pcssc
    assert "BASE_C_AMC_SEM_SECTION4_1_SUCCESSFUL_PREFIX" in pcssc
    assert "base_section4_1_completion_envelopes_reused" in pcssc
    assert "effective_completion_envelopes" in pcssc
    assert "BASE_SECTION4_1_COMPLETION_ENVELOPE_REUSE" in pcssc
    assert 'base_sched.get("completion_bound_by_task", {})' in verifier


def test_retired_event_graph_language_is_not_in_v10_carry_in_module():
    text = (ROOT / "formal_toolchain/v10_1/kernel/carry_in.py").read_text(encoding="utf-8")
    assert "FirstBadEventWindow" not in text
    assert "Event formula allocation" not in text
