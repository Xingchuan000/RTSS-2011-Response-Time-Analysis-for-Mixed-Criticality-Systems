from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.reference.theorem_invocation import (
    build_reference_hi_subset_safety_certificate,
    build_reference_schedulability_certificate,
)


def _cert(obligation_id: str, fingerprint: str, witness_extra=None):
    witness = {
        "reference_taskset_fingerprint": fingerprint,
    }
    witness.update(witness_extra or {})
    return obligation_certificate(
        obligation_id=obligation_id,
        status="PASS",
        context_hash="a" * 64,
        inputs={
            "reference_taskset_fingerprint": fingerprint,
        },
        witness=witness,
        checker_id="test",
        checker_version="1",
    )


def _sched_theorem():
    return {
        "theorem_id": "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY",
        "statement_hash": "b" * 64,
        "assumption_hash": "c" * 64,
        "library_version_hash": "d" * 64,
    }


def _hi_theorem():
    return {
        "theorem_id": "REFERENCE_HI_SUBSET_SAFETY_FROM_TASKSET_SCHEDULABILITY",
        "statement_hash": "e" * 64,
    }


def test_schedulability_rejects_mismatched_reference_tasksets():
    result = build_reference_schedulability_certificate(
        all_task_rta_certificate=_cert(
            "ALL_TASK_REFERENCE_RTA_ARITHMETIC",
            "1" * 64,
            {
                "schema_version": "all_task_rta_v3",
                "all_tasks_covered": True,
                "all_deadlines_met": True,
            },
        ),
        model_conformance_certificate=_cert(
            "REFERENCE_MODEL_CONFORMANCE",
            "2" * 64,
        ),
        theorem=_sched_theorem(),
        reference_context_hash="a" * 64,
    )

    assert result["obligation_status"] == "UNRESOLVED"


def test_schedulability_rejects_wrong_theorem():
    theorem = _sched_theorem()
    theorem["theorem_id"] = "WRONG_THEOREM"
    result = build_reference_schedulability_certificate(
        all_task_rta_certificate=_cert(
            "ALL_TASK_REFERENCE_RTA_ARITHMETIC",
            "1" * 64,
            {"all_tasks_covered": True, "all_deadlines_met": True},
        ),
        model_conformance_certificate=_cert(
            "REFERENCE_MODEL_CONFORMANCE",
            "1" * 64,
        ),
        theorem=theorem,
        reference_context_hash="a" * 64,
    )
    assert result["obligation_status"] == "UNRESOLVED"


def test_hi_safety_propagates_reference_binding():
    schedulability = build_reference_schedulability_certificate(
        all_task_rta_certificate=_cert(
            "ALL_TASK_REFERENCE_RTA_ARITHMETIC",
            "1" * 64,
            {
                "schema_version": "all_task_rta_v3",
                "all_tasks_covered": True,
                "all_deadlines_met": True,
            },
        ),
        model_conformance_certificate=_cert(
            "REFERENCE_MODEL_CONFORMANCE",
            "1" * 64,
        ),
        theorem=_sched_theorem(),
        reference_context_hash="a" * 64,
    )

    result = build_reference_hi_subset_safety_certificate(
        schedulability_certificate=schedulability,
        theorem=_hi_theorem(),
        reference_context_hash="a" * 64,
        hi_task_names=["tau_hi"],
    )

    assert result["obligation_status"] == "PASS"
    assert result["witness"]["reference_taskset_fingerprint"] == "1" * 64
    assert result["witness"]["reference_transition_system_id"] == (
        "FIXED_EXECUTABLE_REFERENCE_P0_V3"
    )
