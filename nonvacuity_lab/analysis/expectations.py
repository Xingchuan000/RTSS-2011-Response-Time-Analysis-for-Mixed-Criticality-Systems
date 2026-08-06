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
        actual = proof.get("result_status")
        if actual in execution_statuses:
            return _result(
                ExperimentStatus(str(actual)),
                str(proof.get("reason", actual)),
            )
        allowed_integrity = set(expected.canonical_integrity_statuses)
        if actual not in allowed_integrity:
            return _result(
                ExperimentStatus.INTEGRITY_REJECTION_MISSING,
                f"expected one of {sorted(allowed_integrity)}, got {actual}",
            )
        obligation = proof.get("violated_obligation_id")
        failure_route = proof.get("failure_route")
        allowed_exact = set(expected.allowed_first_failing_obligations)
        allowed_upstream = (
            set(expected.allowed_upstream_obligations)
            if expected.allow_strict_upstream_failure
            else set()
        )
        if (allowed_exact or allowed_upstream) and obligation not in allowed_exact | allowed_upstream:
            return _result(
                ExperimentStatus.INTEGRITY_REJECTION_MISSING,
                f"integrity rejection obligation {obligation!r} is not allowed",
            )
        if expected.allowed_failure_routes and failure_route not in set(expected.allowed_failure_routes):
            return _result(
                ExperimentStatus.INTEGRITY_REJECTION_MISSING,
                f"integrity rejection route {failure_route!r} is not allowed",
            )
        return _result(
            ExperimentStatus.INTEGRITY_REJECTION_EXPECTED,
            f"tampered input rejected as {actual}",
        )

    if expected.require_activation:
        if activation_result is None:
            return _result(ExperimentStatus.SETUP_INVALID, "activation evidence missing")
        activation_status = str(activation_result.get("status"))
        if activation_status == ActivationStatus.NOT_ACTIVATED.value:
            return _result(ExperimentStatus.NOT_ACTIVATED, "mutation activation gate not met")
        if activation_status in {
            ActivationStatus.ACTIVATION_INCONCLUSIVE.value,
            ActivationStatus.ACTIVATION_SETUP_INVALID.value,
        }:
            return _result(ExperimentStatus.SETUP_INVALID, activation_status)

    actual = proof.get("result_status")
    obligation = proof.get("violated_obligation_id")
    failure_route = proof.get("failure_route")
    allowed_statuses = set(expected.canonical_statuses)

    if actual == "DEPLOYED_TREE_PROVED":
        if expected.require_failure:
            return _result(ExperimentStatus.UNEXPECTED_PASS, "a rejection was required")
        if not allowed_statuses or actual in allowed_statuses:
            return _result(ExperimentStatus.PASS_EXPECTED, "proof passed as allowed")
        return _result(
            ExperimentStatus.UNEXPECTED_PASS,
            f"allowed statuses are {sorted(allowed_statuses)}",
        )

    if expected.require_proved:
        return _result(ExperimentStatus.UNEXPECTED_FAIL, f"proof was required, got {actual}")
    if allowed_statuses and actual not in allowed_statuses:
        return _result(
            ExperimentStatus.UNEXPECTED_FAIL,
            f"allowed statuses are {sorted(allowed_statuses)}, got {actual}",
        )

    allowed_exact = set(expected.allowed_first_failing_obligations)
    if not allowed_exact:
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
    if expected.allowed_failure_routes and failure_route not in set(expected.allowed_failure_routes):
        return _result(
            ExperimentStatus.WRONG_FAILURE_LAYER,
            f"failure route {failure_route!r} is not allowed",
        )
    return _result(ExperimentStatus.FAIL_EXPECTED, "expected rejection observed")


def _result(status: ExperimentStatus, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "expectation_check_v1",
        "status": status.value,
        "reason": reason,
    }
