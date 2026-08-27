from formal_toolchain.cli.proof_readiness import readiness_report


def test_v9_1_has_no_remaining_implementation_blockers():
    report = readiness_report()
    assert report["implementation_blockers"] == []
    assert report["implementation_blocker_count"] == 0
    assert report["proof_pipeline_ready"] is report["formal_dependency_z3_available"]
