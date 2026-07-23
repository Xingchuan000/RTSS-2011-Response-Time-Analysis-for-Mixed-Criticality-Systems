import pytest

from formal_toolchain.bridge.state_relation import build_n6_relation_interface, validate_n6_relation_interface


def test_n6_interface_is_parameterized_and_slot_free():
    interface = build_n6_relation_interface()
    validate_n6_relation_interface(interface)
    assert interface["schema_version"] == "n6_closed_prefix_relation_interface_v2"
    assert "job_slots" not in interface


def test_n6_rejects_legacy_slots():
    with pytest.raises(ValueError):
        validate_n6_relation_interface({"schema_version": "n6_closed_prefix_relation_interface_v1", "job_slots": 4})
