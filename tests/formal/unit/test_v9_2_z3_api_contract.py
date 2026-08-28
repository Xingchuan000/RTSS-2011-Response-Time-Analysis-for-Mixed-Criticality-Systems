from pathlib import Path


def test_controller_uses_arithmetic_mod_operator_not_nonexistent_z3_mod_helper():
    source = Path("formal_toolchain/v9_2/controller_encoder.py").read_text(encoding="utf-8")
    assert "z3.Mod(" not in source
    assert "state.t % model.agent_period" in source


def test_workflow_preserves_verifier_process_failure_diagnostic():
    source = Path("formal_toolchain/workflow/prove_seed.py").read_text(encoding="utf-8")
    assert "V9_2_VERIFIER_PROCESS_FAILED" in source
    assert "verify_bundle.stderr.log" in source
    assert '"result_status": RESULT_UNRESOLVED' in source
