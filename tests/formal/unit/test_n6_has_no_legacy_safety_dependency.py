from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_bad_prefix_implementation_does_not_reference_old_safety_corollary():
    paths = [
        ROOT / "formal_toolchain/bridge/bad_prefix.py",
        ROOT / "formal_toolchain/bridge/compile_bridge.py",
        ROOT / "formal_toolchain/verifier/bridge_proof_checker.py",
        ROOT / "formal_toolchain/verifier/recompute.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "PROTECTED_HI_SAFETY_COROLLARY" not in text
    assert '"protected_hi"' not in text
