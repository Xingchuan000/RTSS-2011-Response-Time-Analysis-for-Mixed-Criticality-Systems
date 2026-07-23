import json
from pathlib import Path

from formal_toolchain.theory.loader import TCB_BACKENDS, load_verified_theory_statement


def test_casewise_prefix_induction_is_fresh_machine_checked():
    root = Path(__file__).parents[3] / "formal_toolchain" / "theory"
    theorem = load_verified_theory_statement(root, "CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT")
    assert theorem["assurance_level"] == "MACHINE_CHECKED_PROJECT_LEMMA"
    backend = TCB_BACKENDS[theorem["proof_object"]["backend"]]
    receipt = backend.verify(root / theorem["proof_object"]["path"], theorem=theorem)
    assert receipt["status"] == "PASS"
