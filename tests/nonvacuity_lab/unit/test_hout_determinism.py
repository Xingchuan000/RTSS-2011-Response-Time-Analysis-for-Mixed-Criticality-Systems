from nonvacuity_lab.runners.paired_hout import _runtime_determinism_checks


def test_paired_hout_requires_matching_runtime_fingerprints():
    checks = _runtime_determinism_checks(
        {
            "demand_trace_fingerprint": "demand-a",
            "taskset_hash": "taskset",
            "scenario_ids": [1, 2],
            "horizon": 100,
        },
        {
            "demand_trace_fingerprint": "demand-b",
            "taskset_hash": "taskset",
            "scenario_ids": [1, 2],
            "horizon": 100,
        },
    )
    assert checks["demand_trace_fingerprint"]["match"] is False
    assert checks["taskset_hash"]["match"] is True
    assert all(
        checks[key]["match"] for key in ("taskset_hash", "scenario_list", "horizon")
    )


def test_paired_hout_missing_runtime_fingerprint_is_not_a_match():
    checks = _runtime_determinism_checks({}, {})
    assert all(item["present"] is False for item in checks.values())
    assert all(item["match"] is False for item in checks.values())
