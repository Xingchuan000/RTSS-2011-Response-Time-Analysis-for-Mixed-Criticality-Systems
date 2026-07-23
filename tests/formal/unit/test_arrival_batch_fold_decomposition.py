from pathlib import Path

from formal_toolchain.bridge.handler_decomposition import (
    build_arrival_batch_decomposition_certificate,
    build_release_event_key_uniqueness_certificate,
)


def test_real_arrival_handler_has_structural_decomposition():
    root = Path(__file__).parents[3]
    uniqueness = build_release_event_key_uniqueness_certificate(source_root=root)
    assert uniqueness["status"] == "PASS"
    result = build_arrival_batch_decomposition_certificate(source_root=root, transition_case_certificates=())
    assert result["status"] == "UNRESOLVED"
    assert result["final_reschedule_once"] is True
    assert result["finite_batch"] is True
