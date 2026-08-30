from pathlib import Path

from formal_toolchain.v10_1.constants import PROOF_ROUTE, RESULT_PROVED
from formal_toolchain.adapters.source_manifest import FORMAL_TARGET_FILES, build_source_manifest

ROOT = Path(__file__).parents[3]


def test_v10_1_is_the_only_active_cli_terminal_route():
    assert PROOF_ROUTE == "C_AMC_SEM_BASE_OR_POLICY_CONSTRAINED_SINGLE_SWITCH_CERTIFICATE_V10_1"
    assert RESULT_PROVED == "DEPLOYED_TREE_PROVED_P0_V10_1"
    text = (ROOT / "formal_toolchain/cli/prove_seed.py").read_text(encoding="utf-8")
    assert "formal_toolchain.workflow.prove_seed" in text
    assert "V10.1" in text
    for legacy in ("--proof-route", "v8_auto", "protected_prefix", "raw_protected_prefix", "strict_full"):
        assert legacy not in text


def test_v10_1_terminal_has_no_event_graph_dependency():
    verifier = (ROOT / "formal_toolchain/v10_1/verifier.py").read_text(encoding="utf-8")
    pcssc = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "event_graph_solver" not in verifier
    assert "build_event_graph_problem" not in verifier
    assert "FIRST_BAD_EVENT_WINDOW" not in verifier
    assert "solve_event_graph" not in pcssc
    assert '"event_graph_in_pass_dependency": False' in verifier


def test_active_source_manifest_points_at_v10_1_terminal_only():
    required = {
        "formal_toolchain/v10_1/verifier.py",
        "formal_toolchain/v10_1/base_section4_1.py",
        "formal_toolchain/v10_1/controller_macro.py",
        "formal_toolchain/v10_1/pcssc.py",
    }
    assert required <= set(FORMAL_TARGET_FILES)
    assert not any("event_graph" in row for row in FORMAL_TARGET_FILES)
    assert not any("/v9_2/verifier.py" in row for row in FORMAL_TARGET_FILES)


def test_v10_1_has_no_legacy_version_or_candidate_compiler_layer():
    assert not (ROOT / "formal_toolchain/v9_1").exists()
    assert not (ROOT / "formal_toolchain/v9_2").exists()
    assert not (ROOT / "formal_toolchain/v10_1/compiler.py").exists()
    assert not (ROOT / "formal_toolchain/cli/compile_seed.py").exists()
    workflow = (ROOT / "formal_toolchain/workflow/prove_seed.py").read_text(encoding="utf-8")
    assert "formal_toolchain.cli.compile_seed" not in workflow
    assert '"--bundle"' not in workflow


def test_active_source_manifest_binds_only_active_theory_artifacts():
    text = (ROOT / "formal_toolchain/adapters/source_manifest.py").read_text(encoding="utf-8")
    assert "_active_theory_artifacts" in text
    assert "formal_toolchain/theory/theory_manifest.json" in text
    assert "_all_formal_toolchain_sources" not in text


def test_v10_1_kernel_does_not_expose_event_graph_environment_or_enabled_case_api():
    env = (ROOT / "formal_toolchain/v10_1/kernel/environment_encoder.py").read_text(encoding="utf-8")
    transition = (ROOT / "formal_toolchain/v10_1/kernel/transition_encoder.py").read_text(encoding="utf-8")
    assert "declare_event_graph_environment" not in env
    assert "lazy_release_demands" not in env
    assert "encode_p5_controller_enabled_case" not in transition


def test_built_semantic_manifest_contains_active_theory_and_no_v9_route():
    manifest = build_source_manifest(ROOT)
    paths = {row["path"] for row in manifest["files"]}
    assert "formal_toolchain/theory/theory_manifest.json" in paths
    assert "formal_toolchain/theory/hashes.json" in paths
    assert any(path.startswith("formal_toolchain/theory/proofs/") for path in paths)
    assert not any("/v9_" in path for path in paths)
    assert not any("event_graph_solver" in path for path in paths)
