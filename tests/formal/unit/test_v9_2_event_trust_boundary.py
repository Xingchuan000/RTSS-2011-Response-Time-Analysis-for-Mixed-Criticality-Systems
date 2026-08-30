from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_candidate_compiler_transports_identity_not_assertions():
    source = (ROOT / "formal_toolchain/v9_2/compiler.py").read_text(encoding="utf-8")
    assert "copy_proof_inputs" not in source
    assert '"candidate_assertions_trusted": False' in source
    assert '"event_layer_added_abstractions": []' in source


def test_verifier_regenerates_event_graph_and_checks_refinement():
    source = (ROOT / "formal_toolchain/v9_2/verifier.py").read_text(encoding="utf-8")
    assert "build_bindings(request_path, source_root=source_root)" in source
    assert "prove_event_refinement" in source
    assert "build_event_graph_problem" in source
    assert "solve_event_graph" in source
    assert "solve_formula" in source
    assert "replay_unsat" not in source
    assert "build_incremental_event_first_bad_window" not in source
