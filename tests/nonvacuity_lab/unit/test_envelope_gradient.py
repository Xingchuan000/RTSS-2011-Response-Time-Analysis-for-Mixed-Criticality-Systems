from nonvacuity_lab.runners.envelope_gradient import search_delta_star


def test_gradient_finds_minimum_integer_failure_and_monotone_slack():
    def evaluate(delta):
        return {
            "result_status": (
                "DEPLOYED_TREE_PROVED"
                if delta < 5
                else "REFERENCE_CERTIFICATE_FAILED"
            ),
            "slack": 9 - delta,
            "violated_obligation_id": "RTA_HI_BOUND" if delta >= 5 else None,
            "witness": {"delta": delta} if delta >= 5 else None,
        }

    result = search_delta_star(evaluate, initial_step=3, maximum_delta=12)
    assert result["status"] == "DELTA_STAR_FOUND"
    assert result["delta_star"] == 5
    assert result["first_failing_obligation"] == "RTA_HI_BOUND"
    assert result["slack_nonincreasing"] is True
