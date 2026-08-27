from pathlib import Path

from formal_toolchain.v9_1.concrete_projection import project_timestamp_record
from formal_toolchain.v9_1.conformance import build_conformance_proof_objects, check_projection_conformance
from formal_toolchain.v9_1.timestamp_trace import TimestampSemanticRecord


def test_concrete_timestamp_projects_in_canonical_order_and_pre_dispatch_stutters(tmp_path: Path):
    record = TimestampSemanticRecord(time=10, controller_action=1, final_dispatch="hi", service_quantum=1,
                                     preliminary_dispatch={"service_quantum": 0, "time_delta": 0})
    projected = project_timestamp_record(record)
    assert projected.phases == ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
    evidence = build_conformance_proof_objects((record,), tmp_path)
    assert evidence["status"] == "UNRESOLVED"
    assert evidence["code"] == "V9_1_UNIVERSAL_CONFORMANCE_PROOF_UNBOUND"
    assert evidence["proof_object_hashes"] == {}
    assert not (tmp_path / "kernel_step_conformance.smt2").exists()


def test_controller_preliminary_service_is_not_ignored():
    record = TimestampSemanticRecord(time=10, preliminary_dispatch={"service_quantum": 1, "time_delta": 0})
    result = check_projection_conformance((record,))
    assert result.status == "FAIL"
    assert result.code == "EVENT_ORDER_CONFORMANCE_FAILED"
