"""Layer-aware outer result classification."""

from __future__ import annotations

from typing import Any, Mapping

from ..schema import ActivationStatus, ExperimentStatus, ExpectedResult


def classify_experiment(
    *,
    expected: ExpectedResult,
    proof_result: Mapping[str, Any] | None,
    activation_result: Mapping[str, Any] | None,
    setup_valid: bool = True,
    baseline_passed: bool = True,
    integrity: bool = False,
) -> dict[str, Any]:
    if not setup_valid:
        return _result(ExperimentStatus.SETUP_INVALID, "setup validation failed")
    if not baseline_passed:
        return _result(ExperimentStatus.BASELINE_REGRESSION, "baseline did not prove")
    proof = dict(proof_result or {})
    if integrity:
        execution_statuses = {
            ExperimentStatus.TOOL_EXECUTION_FAILED.value,
            ExperimentStatus.VERIFIER_TIMEOUT.value,
            ExperimentStatus.VERIFIER_OUTPUT_MISSING.value,
        }
        if proof.get("result_status") in execution_statuses:
            return _result(
                ExperimentStatus(str(proof["result_status"])),
                str(proof.get("reason", proof["result_status"])),
            )
        if proof.get("result_status") == expected.integrity_result_status:
            return _result(ExperimentStatus.INTEGRITY_REJECTION_EXPECTED, "old bundle rejected")
        return _result(
            ExperimentStatus.INTEGRITY_REJECTION_MISSING,
            f"expected {expected.integrity_result_status}, got {proof.get('result_status')}",
        )
    if activation_result is not None:
        activation_status = str(activation_result.get("status"))
        if activation_status == ActivationStatus.NOT_ACTIVATED.value:
            return _result(ExperimentStatus.NOT_ACTIVATED, "mutation activation gate not met")
        if activation_status in {
            ActivationStatus.ACTIVATION_INCONCLUSIVE.value,
            ActivationStatus.ACTIVATION_SETUP_INVALID.value,
        }:
            return _result(ExperimentStatus.SETUP_INVALID, activation_status)
    actual = proof.get("result_status")
    expected_status = expected.result_status
    obligation = proof.get("violated_obligation_id")
    if actual == "DEPLOYED_TREE_PROVED":
        if expected_status in {None, "DEPLOYED_TREE_PROVED"}:
            return _result(ExperimentStatus.PASS_EXPECTED, "proof passed as allowed")
        return _result(ExperimentStatus.UNEXPECTED_PASS, f"expected {expected_status}")
    if expected_status == "DEPLOYED_TREE_PROVED":
        return _result(ExperimentStatus.UNEXPECTED_FAIL, f"unexpected {actual}")
    allowed_exact = set(expected.first_failing_obligations)
    allowed_upstream = (
        set(expected.allowed_upstream_obligations)
        if expected.allow_strict_upstream_failure
        else set()
    )
    if allowed_exact or allowed_upstream:
        if obligation not in allowed_exact | allowed_upstream:
            return _result(
                ExperimentStatus.WRONG_FAILURE_LAYER,
                f"obligation {obligation!r} is not allowed",
            )
    if expected_status is None:
        return _result(
            ExperimentStatus.FAIL_EXPECTED,
            "activated semantic rejection recorded without predeclared final status",
        )
    if actual == expected_status:
        return _result(ExperimentStatus.FAIL_EXPECTED, "expected rejection observed")
    return _result(
        ExperimentStatus.UNEXPECTED_FAIL,
        f"expected {expected_status}, got {actual}",
    )


def _result(status: ExperimentStatus, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "expectation_check_v1",
        "status": status.value,
        "reason": reason,
    }
