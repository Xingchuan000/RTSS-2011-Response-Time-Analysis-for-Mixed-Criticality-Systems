from pathlib import Path

from formal_toolchain.theory.loader import verify_theory_library


def test_phase_k_theory_chain_has_parameterized_n5_backend():
    result = verify_theory_library(Path(__file__).parents[3] / "formal_toolchain" / "theory")
    assert result["status"] == "PASS"
