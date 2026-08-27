from pathlib import Path


def test_v9_1_compiler_does_not_copy_untrusted_assertion_only_smt():
    source = (Path(__file__).parents[3] / "formal_toolchain/v9_1/compiler.py").read_text(encoding="utf-8")
    assert "copy_proof_inputs" not in source
    assert "NOT_ACCEPTED_FROM_UNTRUSTED_INPUT" in source


def test_v9_1_verifier_requires_fresh_regeneration_before_any_proved_path():
    source = (Path(__file__).parents[3] / "formal_toolchain/v9_1/verifier.py").read_text(encoding="utf-8")
    assert "replay_unsat" not in source
    assert "V9_1_TRUSTED_PROOF_REGENERATION_UNBOUND" in source
