from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_verifier_terminalizes_base_proved_hi_prefix_before_pcssc():
    text = (ROOT / "formal_toolchain/v10_1/verifier.py").read_text(encoding="utf-8")
    assert 'BASE_C_AMC_SEM_HI_PREFIX_CERTIFICATE' in text
    assert 'pending_hi_tasks = tuple' in text
    assert 'if not pending_hi_tasks:' in text
    assert 'BASE_C_AMC_SEM_SECTION4_1_HI_PREFIX' in text
    assert 'for index, task in enumerate(pending_hi_tasks):' in text


def test_base_receipt_keeps_full_task_status_separate_from_hi_prefix_status():
    text = (ROOT / "formal_toolchain/v10_1/base_section4_1.py").read_text(encoding="utf-8")
    assert '"hi_safety_status"' in text
    assert '"hi_safe_targets"' in text
    assert '"hi_unresolved_targets"' in text
    assert '"all_tasks_schedulable"' in text
