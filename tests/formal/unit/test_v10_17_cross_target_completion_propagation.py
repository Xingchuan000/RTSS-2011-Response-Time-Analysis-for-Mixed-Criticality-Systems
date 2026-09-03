from dataclasses import dataclass
from pathlib import Path

import pytest

from formal_toolchain.v10_1.completion_certificates import (
    BASE_COMPLETION_SOURCE,
    BASE_COMPLETION_THEOREM,
    PCSSC_COMPLETION_SOURCE,
    PCSSC_GUARDED_COMPLETION_THEOREM_V10_17,
    CompletionCertificateError,
    build_base_completion_certificates,
    completion_prefix_for_target,
    export_pcssc_completion_certificate,
    merge_certified_completion,
)

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _Task:
    name: str
    priority: int
    period: int
    deadline: int
    criticality: str = "HI"


@dataclass(frozen=True)
class _Model:
    tasks: tuple[_Task, ...]

    @property
    def task_by_name(self):
        return {task.name: task for task in self.tasks}


def _task(name: str, priority: int, *, deadline: int = 10, period: int = 20) -> _Task:
    return _Task(name=name, priority=priority, period=period, deadline=deadline)


def _model() -> _Model:
    return _Model(tasks=(_task("hi0", 0), _task("hi1", 1), _task("hi2", 2)))


def test_base_is_unconditional_and_pcssc_is_guarded_safe_prefix():
    model = _model()
    base = build_base_completion_certificates(model, {"hi0": 8})["hi0"]
    assert base.source == BASE_COMPLETION_SOURCE
    assert base.theorem_basis == BASE_COMPLETION_THEOREM

    pcssc = export_pcssc_completion_certificate(
        model, "hi0", status="PASS", response_bound=6,
        theorem_basis=PCSSC_GUARDED_COMPLETION_THEOREM_V10_17,
    )
    assert pcssc.source == PCSSC_COMPLETION_SOURCE
    assert pcssc.theorem_basis == PCSSC_GUARDED_COMPLETION_THEOREM_V10_17


def test_completion_prefix_is_strictly_forward_in_priority():
    model = _model()
    hi0 = export_pcssc_completion_certificate(
        model, "hi0", status="PASS", response_bound=6,
        theorem_basis=PCSSC_GUARDED_COMPLETION_THEOREM_V10_17,
    )
    prefix = completion_prefix_for_target(model, "hi1", {"hi0": hi0})
    assert tuple(prefix) == ("hi0",)

    hi1 = export_pcssc_completion_certificate(
        model, "hi1", status="PASS", response_bound=7,
        theorem_basis=PCSSC_GUARDED_COMPLETION_THEOREM_V10_17,
    )
    with pytest.raises(CompletionCertificateError, match="PRIORITY_ORDERED_CERTIFICATE_DAG_VIOLATION"):
        completion_prefix_for_target(model, "hi1", {"hi1": hi1})


def test_pcssc_completion_export_accepts_only_v10_17_guarded_theorem():
    model = _model()
    with pytest.raises(CompletionCertificateError, match="UNKNOWN_THEOREM"):
        export_pcssc_completion_certificate(
            model, "hi0", status="PASS", response_bound=6,
            theorem_basis="PCSSC_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_12",
        )
    with pytest.raises(CompletionCertificateError, match="REQUIRES_PASS"):
        export_pcssc_completion_certificate(
            model, "hi0", status="UNRESOLVED", response_bound=6,
            theorem_basis=PCSSC_GUARDED_COMPLETION_THEOREM_V10_17,
        )
    with pytest.raises(CompletionCertificateError, match="EXCEEDS_DEADLINE"):
        export_pcssc_completion_certificate(
            model, "hi0", status="PASS", response_bound=11,
            theorem_basis=PCSSC_GUARDED_COMPLETION_THEOREM_V10_17,
        )


def test_merge_preserves_unconditional_base_semantics_over_tighter_guarded_bound():
    model = _model()
    base = build_base_completion_certificates(model, {"hi0": 8})["hi0"]
    pcssc = export_pcssc_completion_certificate(
        model, "hi0", status="PASS", response_bound=6,
        theorem_basis=PCSSC_GUARDED_COMPLETION_THEOREM_V10_17,
    )
    merged = merge_certified_completion(base, pcssc)
    assert merged.response_bound == 8
    assert merged.source == BASE_COMPLETION_SOURCE


def test_production_does_not_consume_conditional_pcssc_completion_without_prefix_local_receipt():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "conditional_pcssc_completion_available_not_consumed" in text
    assert "conditional_completion_tightening_enabled\": False" in text
    assert "CROSS_TARGET_COMPLETION_BOUND_REUSE::" not in text
    assert "CROSS_TARGET_PCSSC_COMPLETION_PROPAGATION" not in text
    assert "GUARDED_SAFE_PREFIX_COMPLETION_EXPORT::" in text


def test_verifier_exports_base_unconditional_receipts_and_uses_one_controller_macro():
    text = (ROOT / "formal_toolchain/v10_1/verifier.py").read_text(encoding="utf-8")
    assert text.count("build_controller_macro_path(") == 1
    assert "BASE_UNCONDITIONAL_COMPLETION_EXPORT::" in text
    assert "base_refine_hash" in text
    assert "certified_completion_by_task=completion_prefix" in text
    assert "export_pcssc_completion_certificate(" in text
    assert "controller_macro_rebuilds_due_to_propagation\": 0" in text
    assert "PRIORITY_ORDERED_CERTIFICATE_DAG_ACYCLIC" in text


def test_active_source_manifest_binds_completion_certificate_logic():
    text = (ROOT / "formal_toolchain/adapters/source_manifest.py").read_text(encoding="utf-8")
    assert '"formal_toolchain/v10_1/completion_certificates.py"' in text
