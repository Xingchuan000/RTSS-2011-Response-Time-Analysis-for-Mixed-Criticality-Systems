from formal_toolchain.v9_1.counterexample_replay import ReplayResult
from formal_toolchain.v9_1.readiness import proof_pipeline_ready


def test_sat_diagnostic_does_not_block_all_unsat_safety_route():
    assert proof_pipeline_ready() is True


def test_replay_result_is_machine_readable():
    row = ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", {"reason": "x"})
    assert row.as_dict() == {
        "status": "UNRESOLVED",
        "code": "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
        "details": {"reason": "x"},
    }
