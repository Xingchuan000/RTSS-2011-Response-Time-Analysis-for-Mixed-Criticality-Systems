"""R02：没有唯一 evidence 的 obligation 必须 fail-closed。"""

from formal_toolchain.verifier.checker_catalog import checker_for


def test_missing_evidence_is_unresolved() -> None:
    checker = checker_for("BOOT_INITIALIZATION")
    assert checker is not None
    result = checker(evidence={})
    assert result["status"] == "UNRESOLVED"
    assert result["code"] == "OBLIGATION_EVIDENCE_MISSING"
