"""Strict outer-only classification of ordinary PPP observations."""

from __future__ import annotations

from dataclasses import dataclass

from ..schema import ExpectedResult, ExperimentStatus


@dataclass(frozen=True)
class ProofObservation:
    result_status: str
    first_failing_obligation: str | None = None
    failure_route: str | None = None
    verifier_exit_code: int = 0


def classify_experiment(
    observation: ProofObservation,
    expected: ExpectedResult,
    *,
    activated: bool,
) -> ExperimentStatus:
    expected.validate()
    if expected.require_activation and not activated:
        return ExperimentStatus.NOT_ACTIVATED
    if expected.require_proved and observation.result_status != "DEPLOYED_TREE_PROVED":
        return ExperimentStatus.WRONG_FAILURE_LAYER
    if expected.require_failure and observation.result_status == "DEPLOYED_TREE_PROVED":
        return ExperimentStatus.UNEXPECTED_PASS
    statuses = set(expected.canonical_statuses)
    status_ok = not statuses or observation.result_status in statuses
    obligation_ok = (
        not expected.allowed_first_failing_obligations
        or observation.first_failing_obligation in expected.allowed_first_failing_obligations
    )
    route_ok = (
        not expected.allowed_failure_routes
        or observation.failure_route in expected.allowed_failure_routes
    )
    if not status_ok:
        if observation.result_status == "DEPLOYED_TREE_PROVED":
            return ExperimentStatus.UNEXPECTED_PASS
        return ExperimentStatus.WRONG_FAILURE_LAYER
    if not obligation_ok or not route_ok:
        return ExperimentStatus.WRONG_FAILURE_LAYER
    if observation.result_status == "DEPLOYED_TREE_PROVED":
        return ExperimentStatus.ACCEPTED_SAFE_MUTATION
    return ExperimentStatus.EXPECTED_REJECTION
