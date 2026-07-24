from formal_toolchain.reference.protected_priority_prefix.runtime_schema import build_runtime_schema_certificate, verify_runtime_schema_certificate


def test_runtime_schema_is_source_bound_and_all_obligations_are_checked():
    certificate = build_runtime_schema_certificate()
    assert certificate["status"] == "PASS"
    assert len(certificate["checks"]) >= 10
    assert verify_runtime_schema_certificate(certificate)["status"] == "PASS"
