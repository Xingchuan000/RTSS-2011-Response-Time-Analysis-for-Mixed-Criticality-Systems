from nonvacuity_lab.validation.proof_request_clean import assert_experiment_blind_request


def test_ordinary_request_guard_rejects_experiment_fields():
    assert_experiment_blind_request({"seed_dir": "/tmp/seed", "proof_route": "protected_prefix"})


def test_ordinary_request_guard_is_recursive():
    try:
        assert_experiment_blind_request({"context": {"mutation_id": "A1"}})
    except ValueError:
        pass
    else:
        raise AssertionError("experiment field escaped isolation guard")
