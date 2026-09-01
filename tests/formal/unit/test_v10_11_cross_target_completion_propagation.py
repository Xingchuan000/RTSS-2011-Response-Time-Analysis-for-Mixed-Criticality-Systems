from dataclasses import dataclass
from pathlib import Path

import pytest

from formal_toolchain.v10_1.completion_certificates import (
    BASE_COMPLETION_SOURCE,
    PCSSC_COMPLETION_SOURCE,
    PCSSC_CASE_COMPLETION_THEOREM,
    PCSSC_POINTWISE_COMPLETION_THEOREM,
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


def test_base_and_pcssc_sources_keep_the_tighter_sound_bound():
    model = _model()
    base = build_base_completion_certificates(model, {"hi0": 8})["hi0"]
    assert base.source == BASE_COMPLETION_SOURCE

    tighter_pcssc = export_pcssc_completion_certificate(
        model, "hi0", status="PASS", response_bound=6,
        theorem_basis=PCSSC_POINTWISE_COMPLETION_THEOREM,
    )
    merged = merge_certified_completion(base, tighter_pcssc)
    assert merged.response_bound == 6
    assert merged.source == PCSSC_COMPLETION_SOURCE

    looser_pcssc = export_pcssc_completion_certificate(
        model, "hi0", status="PASS", response_bound=9,
        theorem_basis=PCSSC_CASE_COMPLETION_THEOREM,
    )
    merged = merge_certified_completion(base, looser_pcssc)
    assert merged.response_bound == 8
    assert merged.source == BASE_COMPLETION_SOURCE


def test_completion_prefix_is_strictly_forward_in_priority_and_rejects_self_use():
    model = _model()
    hi0 = export_pcssc_completion_certificate(
        model, "hi0", status="PASS", response_bound=6,
        theorem_basis=PCSSC_POINTWISE_COMPLETION_THEOREM,
    )
    prefix = completion_prefix_for_target(model, "hi1", {"hi0": hi0})
    assert tuple(prefix) == ("hi0",)

    hi1 = export_pcssc_completion_certificate(
        model, "hi1", status="PASS", response_bound=7,
        theorem_basis=PCSSC_CASE_COMPLETION_THEOREM,
    )
    with pytest.raises(CompletionCertificateError, match="PRIORITY_ORDERED_CERTIFICATE_DAG_VIOLATION"):
        completion_prefix_for_target(model, "hi1", {"hi1": hi1})

    hi2 = export_pcssc_completion_certificate(
        model, "hi2", status="PASS", response_bound=7,
        theorem_basis=PCSSC_POINTWISE_COMPLETION_THEOREM,
    )
    with pytest.raises(CompletionCertificateError, match="PRIORITY_ORDERED_CERTIFICATE_DAG_VIOLATION"):
        completion_prefix_for_target(model, "hi1", {"hi2": hi2})


def test_pcssc_completion_export_requires_closed_pass_and_r_le_d_le_t():
    model = _model()
    with pytest.raises(CompletionCertificateError, match="REQUIRES_PASS"):
        export_pcssc_completion_certificate(
            model, "hi0", status="UNRESOLVED", response_bound=6,
            theorem_basis=PCSSC_POINTWISE_COMPLETION_THEOREM,
        )
    with pytest.raises(CompletionCertificateError, match="MISSING_RESPONSE_BOUND"):
        export_pcssc_completion_certificate(
            model, "hi0", status="PASS", response_bound=None,
            theorem_basis=PCSSC_POINTWISE_COMPLETION_THEOREM,
        )
    with pytest.raises(CompletionCertificateError, match="EXCEEDS_DEADLINE"):
        export_pcssc_completion_certificate(
            model, "hi0", status="PASS", response_bound=11,
            theorem_basis=PCSSC_POINTWISE_COMPLETION_THEOREM,
        )

    bad_model = _Model(tasks=(_task("bad", 0, deadline=21, period=20),))
    with pytest.raises(CompletionCertificateError, match="REQUIRES_D_LE_T"):
        export_pcssc_completion_certificate(
            bad_model, "bad", status="PASS", response_bound=10,
            theorem_basis=PCSSC_CASE_COMPLETION_THEOREM,
        )


def test_verifier_uses_one_controller_macro_and_incremental_completion_map():
    text = (ROOT / "formal_toolchain/v10_1/verifier.py").read_text(encoding="utf-8")
    assert text.count("build_controller_macro_path(") == 1
    assert "certified_completion_by_task = build_base_completion_certificates" in text
    assert "completion_prefix_for_target(" in text
    assert "certified_completion_by_task=completion_prefix" in text
    assert "export_pcssc_completion_certificate(" in text
    assert "controller_macro_rebuilds_due_to_propagation\": 0" in text
    assert "PRIORITY_ORDERED_CERTIFICATE_DAG_ACYCLIC" in text


def test_pcssc_records_cross_target_reuse_and_safe_prefix_theorem_basis():
    text = (ROOT / "formal_toolchain/v10_1/pcssc.py").read_text(encoding="utf-8")
    assert "CERTIFIED_COMPLETION_PREFIX_SOUND::" in text
    assert "CROSS_TARGET_COMPLETION_BOUND_REUSE::" in text
    assert "CROSS_TARGET_PCSSC_COMPLETION_PROPAGATION" in text
    completion_text = (ROOT / "formal_toolchain/v10_1/completion_certificates.py").read_text(encoding="utf-8")
    assert "PCSSC_SAFE_PREFIX_COMPLETION_EXPORT_V10_11" in completion_text
    assert "PCSSC_CASE_SAFE_PREFIX_COMPLETION_EXPORT_V10_12" in completion_text
    assert "PCSSC_CASE_CONDITIONED_SAFE_PREFIX_COMPLETION_EXPORT_V10_13" in completion_text
    assert "PRIORITY_ORDERED_CERTIFICATE_DAG_VIOLATION" in text


def test_active_source_manifest_binds_completion_certificate_logic():
    text = (ROOT / "formal_toolchain/adapters/source_manifest.py").read_text(encoding="utf-8")
    assert '"formal_toolchain/v10_1/completion_certificates.py"' in text
