from nonvacuity_lab.analysis.expectations import classify_experiment
from nonvacuity_lab.schema import ExpectedResult


def test_expectation_checker_covers_layer_and_activation_gate():
    expected = ExpectedResult(
        result_status="POLICY_CONTRACT_VIOLATION",
        first_failing_obligations=("DEPLOYED_POLICY_PRESERVATION",),
        allowed_upstream_obligations=("SOURCE_BINDING",),
        allow_strict_upstream_failure=True,
    )
    exact = classify_experiment(
        expected=expected,
        proof_result={
            "result_status": "POLICY_CONTRACT_VIOLATION",
            "violated_obligation_id": "DEPLOYED_POLICY_PRESERVATION",
        },
        activation_result={"status": "ACTIVATED"},
    )
    upstream = classify_experiment(
        expected=expected,
        proof_result={
            "result_status": "POLICY_CONTRACT_VIOLATION",
            "violated_obligation_id": "SOURCE_BINDING",
        },
        activation_result={"status": "ACTIVATED"},
    )
    wrong = classify_experiment(
        expected=expected,
        proof_result={
            "result_status": "POLICY_CONTRACT_VIOLATION",
            "violated_obligation_id": "UNRELATED",
        },
        activation_result={"status": "ACTIVATED"},
    )
    inactive = classify_experiment(
        expected=expected,
        proof_result={
            "result_status": "POLICY_CONTRACT_VIOLATION",
            "violated_obligation_id": "DEPLOYED_POLICY_PRESERVATION",
        },
        activation_result={"status": "NOT_ACTIVATED"},
    )
    assert exact["status"] == "FAIL_EXPECTED"
    assert upstream["status"] == "FAIL_EXPECTED"
    assert wrong["status"] == "WRONG_FAILURE_LAYER"
    assert inactive["status"] == "NOT_ACTIVATED"


def test_integrity_result_requires_bundle_invalid():
    expected = ExpectedResult()
    good = classify_experiment(
        expected=expected,
        proof_result={"result_status": "PROOF_BUNDLE_INVALID"},
        activation_result=None,
        integrity=True,
    )
    bad = classify_experiment(
        expected=expected,
        proof_result={"result_status": "DEPLOYED_TREE_PROVED"},
        activation_result=None,
        integrity=True,
    )
    assert good["status"] == "INTEGRITY_REJECTION_EXPECTED"
    assert bad["status"] == "INTEGRITY_REJECTION_MISSING"
