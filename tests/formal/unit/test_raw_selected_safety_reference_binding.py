from __future__ import annotations
from types import SimpleNamespace
from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.reference.finite_bad_prefix_contradiction import build_finite_bad_prefix_contradiction
from formal_toolchain.routes.raw_prefix_checkers import check_selected_safety

_CONTEXT = "1" * 64
_FINGERPRINT = "a" * 64
_REFERENCE_SYSTEM = "FIXED_EXECUTABLE_REFERENCE_P0_V3"

class _Taskset:
    def to_dict(self):
        return {"fingerprint": _FINGERPRINT}


def test_raw_selected_safety_carries_full_reference_system_identity() -> None:
    checked = check_selected_safety(
        fresh_state=SimpleNamespace(full_reference_taskset=_Taskset()),
        verified_predecessors={"REFERENCE_HI_SAFETY_FROM_RAW_PREFIX": {"obligation_status": "PASS"}},
    )
    assert checked["status"] == "PASS"
    assert checked["witness"]["reference_taskset_fingerprint"] == _FINGERPRINT
    assert checked["witness"]["reference_transition_system_id"] == _REFERENCE_SYSTEM


def test_raw_selected_safety_binding_is_sufficient_for_finite_contradiction_join() -> None:
    checked = check_selected_safety(
        fresh_state=SimpleNamespace(full_reference_taskset=_Taskset()),
        verified_predecessors={"REFERENCE_HI_SAFETY_FROM_RAW_PREFIX": {"obligation_status": "PASS"}},
    )
    safety = obligation_certificate(obligation_id="SELECTED_REFERENCE_HI_SAFETY", status="PASS", context_hash=_CONTEXT,
                                    witness=checked["witness"], inputs={}, checker_id="test.raw_selected_safety", checker_version="test-v1")
    reflection = obligation_certificate(obligation_id="HI_BAD_CLOSED_PREFIX_REFLECTION", status="PASS", context_hash=_CONTEXT,
        witness={"theorem_id": "FINITE_HI_BAD_PREFIX_REFLECTION",
                 "quantification": "FOR_ALL_FIRST_FINITE_CONCRETE_HI_BAD_CLOSED_PREFIXES",
                 "reference_taskset_fingerprint": _FINGERPRINT,
                 "reference_transition_system_id": _REFERENCE_SYSTEM,
                 "first_miss_set": None},
        inputs={}, checker_id="test.bad_prefix_reflection", checker_version="test-v1")
    result = build_finite_bad_prefix_contradiction(
        reference_hi_safety_certificate=safety, bad_prefix_reflection_certificate=reflection,
        theorem={"theorem_id": "FINITE_BAD_PREFIX_CONTRADICTION", "statement_hash": "b" * 64},
        composition_context_hash=_CONTEXT)
    assert result["obligation_status"] == "PASS"
    assert result["witness"]["reference_binding"]["same_reference_taskset"] is True
    assert result["witness"]["reference_binding"]["same_reference_system"] is True
