from pathlib import Path

from formal_toolchain.bridge.early_stop_gate import build_early_stop_configuration_gate
from formal_toolchain.theory.loader import MACHINE_PREMISES


def test_n6_registry_and_source_topology_exclude_reference_safety():
    assert "REFERENCE_HI_SUBSET_SAFETY" not in MACHINE_PREMISES["FINITE_HI_BAD_PREFIX_REFLECTION"]
    source = (Path(__file__).parents[3] / "formal_toolchain/verifier/recompute.py").read_text()
    n6_source = source[source.index("def _fresh_bad_prefix_proof"):source.index("def _semantic_certificate")]
    assert "REFERENCE_HI_SUBSET_SAFETY" not in n6_source


def test_early_stop_gate_modes_are_exact():
    disabled = build_early_stop_configuration_gate(runtime_config=type("C", (), {"stop_at_first_miss": False})(), context_hash="a" * 64)
    assert disabled["obligation_status"] == "PASS"
    missing = build_early_stop_configuration_gate(runtime_config=type("C", (), {"stop_at_first_miss": True})(), context_hash="a" * 64)
    assert missing["obligation_status"] == "UNRESOLVED"
