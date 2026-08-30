from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_fp64_history_closure_is_decomposed_into_independent_qf_fp_obligations():
    text = (ROOT / "formal_toolchain/v10_1/feature_transfer.py").read_text(encoding="utf-8")
    assert "_fp_history_update_obligations" in text
    assert "solve_qf_fp_formula" in text
    assert 'FULL_LEGAL_HISTORY_UPDATE_DOMAIN_CLOSURE::{task.name}::EMA_COST' in text
    assert 'FULL_LEGAL_HISTORY_UPDATE_DOMAIN_CLOSURE::OVERRUN_EMA' in text
    assert "_fp_history_update_counterexample" not in text
    assert "z3.Or(*bad)" not in text


def test_qf_fp_obligations_use_specialized_solver_logic_without_formula_relaxation():
    text = (ROOT / "formal_toolchain/v10_1/kernel/formula_solver.py").read_text(encoding="utf-8")
    assert 'z3.SolverFor("QF_FP")' in text
    assert "solve_qf_fp_formula" in text
    # The exact IEEE-754 formula remains in feature_transfer; only solver decomposition changes.
    feature = (ROOT / "formal_toolchain/v10_1/feature_transfer.py").read_text(encoding="utf-8")
    for token in ("z3.Float64()", "z3.RNE()", "z3.fpMul", "z3.fpAdd", "z3.fpIsNaN", "z3.fpIsInf"):
        assert token in feature


def test_feature_transfer_preserves_aggregate_history_closure_receipt():
    text = (ROOT / "formal_toolchain/v10_1/feature_transfer.py").read_text(encoding="utf-8")
    assert '"obligation_id": "FULL_LEGAL_HISTORY_UPDATE_DOMAIN_CLOSURE"' in text
    assert '"solver_logic": "QF_FP"' in text
    assert '"child_obligations"' in text
    assert '"child_formula_hashes"' in text


def test_prove_seed_writes_directly_to_requested_output_without_staging_publish_layer():
    text = (ROOT / "formal_toolchain/workflow/prove_seed.py").read_text(encoding="utf-8")
    assert '"workspace_mode": "DIRECT_OUTPUT"' in text
    assert "_publish_staging" not in text
    assert ".staging" not in text
    assert ".rename(" not in text
    assert "copytree" not in text
    assert "freeze_seed_workspace_v10_1(\n            seed_dir, tree_variant, out" in text
